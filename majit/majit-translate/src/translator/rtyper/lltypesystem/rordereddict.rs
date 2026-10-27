//! RPython `rpython/rtyper/lltypesystem/rordereddict.py`.
//!
//! PyPy aliases all dict annotations to `SomeOrderedDict`, so this is
//! the default concrete repr selected by `SomeDict.rtyper_makerepr`.
//! This slice lands `OrderedDictRepr`'s low-level table shape and the public
//! constant/helper names around the low-level method section, plus the host
//! `_ptr` ports of `ll_dict_lookup` / `_ll_dict_setitem_lookup_done` /
//! `ll_dict_grow` / `ll_dict_reindex` / `ll_dict_rehash_after_translation`.
#![allow(non_camel_case_types, non_snake_case)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, LazyLock};

use crate::annotator::bookkeeper::is_objectmodel_r_dict_instance;
use crate::annotator::dictdef::DictDef;
use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue,
    HostObject, Link, ReprConst, SpaceOperation, Variable, c_last_exception,
};
use crate::flowspace::pygraph::PyGraph;
use crate::rlib::jit::{LookInsidePred, mint_look_inside_iff};
#[cfg(test)]
use crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::llinterp::eval_funcptr;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, Array, GCREF, LowLevelType, LowLevelValue, MallocFlavor, Ptr, PtrTarget, Struct,
    cast_opaque_ptr, malloc,
};
use crate::translator::rtyper::lltypesystem::rstr::sub_helper_funcptr_constant;
use crate::translator::rtyper::rdict::{AbstractDictIteratorRepr, AbstractDictRepr};
use crate::translator::rtyper::rmodel::{
    RTypeResult, Repr, ReprState, gc_flavor_const, lowlevel_type_const,
};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, GenopResult, HighLevelOp, LowLevelFunction, RPythonTyper, constant_with_lltype,
    exception_args, helper_pygraph_from_graph, variable_with_lltype, void_field_const,
};
use crate::translator::tool::taskengine::TaskError;

fn ptr_to_gc_array(of: LowLevelType) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::gc(of)),
    }))
}

/// RPython `DICTINDEX_*` pointer aliases (`rordereddict.py:489-492`).
///
/// PyPy uses width-specific `rffi.UCHAR` / `USHORT` / `UINT` element types.
/// Pyre's current `lltype`/`rffi` surface does not yet distinguish those C
/// integer widths, so all four aliases carry the available unsigned primitive
/// while preserving the exact public names and `Ptr(GcArray(...))` shape.
pub static DICTINDEX_LONG: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));
pub static DICTINDEX_INT: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));
pub static DICTINDEX_SHORT: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));
pub static DICTINDEX_BYTE: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));

/// RPython `IS_64BIT = sys.maxint != 2 ** 31 - 1` (`rordereddict.py`).
pub const IS_64BIT: bool = usize::BITS == 64;

/// RPython lookup table selector constants (`rordereddict.py:496-503`).
pub const FUNC_SHIFT: i64 = if IS_64BIT { 3 } else { 2 };
pub const FUNC_MASK: i64 = if IS_64BIT { 0x07 } else { 0x03 };
pub const FUNC_BYTE: i64 = 0;
pub const FUNC_SHORT: i64 = 1;
pub const FUNC_INT: i64 = 2;
pub const FUNC_LONG: i64 = if IS_64BIT { 3 } else { 2 };
pub const FUNC_MUST_REINDEX: i64 = if IS_64BIT { 4 } else { 3 };

/// RPython `TYPE_*` aliases (`rordereddict.py:504-507`). See
/// [`DICTINDEX_BYTE`] for the current width-collapsing note.
pub const TYPE_BYTE: LowLevelType = LowLevelType::Unsigned;
pub const TYPE_SHORT: LowLevelType = LowLevelType::Unsigned;
pub const TYPE_INT: LowLevelType = LowLevelType::Unsigned;
pub const TYPE_LONG: LowLevelType = LowLevelType::Unsigned;

pub const PERTURB_SHIFT: i64 = 5;
pub const FREE: i64 = 0;
pub const DELETED: i64 = 1;
pub const VALID_OFFSET: i64 = 2;
pub const MIN_INDEXES_MINUS_ENTRIES: i64 = VALID_OFFSET + 1;

pub const FLAG_LOOKUP: i64 = 0;
pub const FLAG_STORE: i64 = 1;

/// RPython `DICT_INITSIZE = 16` (`rordereddict.py`).
pub const DICT_INITSIZE: i64 = 16;

/// Shared lookup-chain helpers minted by [`OrderedDictRepr::lookup_chain_helpers`].
/// `hash_takes_dict` is true for `ll_keyhash_custom(d, key)` (`rordereddict.py`);
/// plain dicts call the key repr hash as `keyhash(key)`.
struct LookupChainHelpers {
    hash_fn_const: Constant,
    call_lookup_const: Constant,
    hash_takes_dict: bool,
}

fn keyhash_call_args(
    hash_fn: Constant,
    d: &Variable,
    key: &Variable,
    hash_takes_dict: bool,
) -> Vec<Hlvalue> {
    if hash_takes_dict {
        vec![
            Hlvalue::Constant(hash_fn),
            Hlvalue::Variable(d.clone()),
            Hlvalue::Variable(key.clone()),
        ]
    } else {
        vec![Hlvalue::Constant(hash_fn), Hlvalue::Variable(key.clone())]
    }
}

fn keyeq_call_args(
    eq_fn: Constant,
    d: &Variable,
    ckey: &Variable,
    key: &Variable,
    eq_takes_dict: bool,
) -> Vec<Hlvalue> {
    if eq_takes_dict {
        vec![
            Hlvalue::Constant(eq_fn),
            Hlvalue::Variable(d.clone()),
            Hlvalue::Variable(ckey.clone()),
            Hlvalue::Variable(key.clone()),
        ]
    } else {
        vec![
            Hlvalue::Constant(eq_fn),
            Hlvalue::Variable(ckey.clone()),
            Hlvalue::Variable(key.clone()),
        ]
    }
}

fn dictentry_has_f_hash(entries_ptr_lltype: &LowLevelType) -> bool {
    let LowLevelType::Ptr(ptr) = entries_ptr_lltype else {
        return false;
    };
    match &ptr.TO {
        PtrTarget::Array(arr) => match &arr.OF {
            LowLevelType::Struct(s) => s._names.iter().any(|n| n == "f_hash"),
            _ => false,
        },
        _ => false,
    }
}

fn dictentry_key_lltype(entries_ptr_lltype: &LowLevelType) -> Option<LowLevelType> {
    let LowLevelType::Ptr(ptr) = entries_ptr_lltype else {
        return None;
    };
    match &ptr.TO {
        PtrTarget::Array(arr) => match &arr.OF {
            LowLevelType::Struct(s) => s._flds.get("key").cloned(),
            _ => None,
        },
        _ => None,
    }
}

fn is_func_ptr_lltype(lltype: &LowLevelType) -> bool {
    matches!(
        lltype,
        LowLevelType::Ptr(ptr) if matches!(ptr.TO, PtrTarget::Func(_))
    )
}

fn hlvalue_lltype(v: &Hlvalue) -> Result<LowLevelType, TyperError> {
    match v {
        Hlvalue::Variable(var) => var.concretetype().ok_or_else(|| {
            TyperError::message("hlinvoke stored callable: argument missing concretetype")
        }),
        Hlvalue::Constant(c) => c.concretetype.clone().ok_or_else(|| {
            TyperError::message("hlinvoke stored callable: constant argument missing concretetype")
        }),
    }
}

fn emit_direct_call(block: &BlockRef, args: Vec<Hlvalue>, result: &Variable) {
    block.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        args,
        Hlvalue::Variable(result.clone()),
    ));
}

fn emit_getfield_stored(
    block: &BlockRef,
    d: &Variable,
    field: &str,
    fn_lltype: LowLevelType,
) -> Variable {
    let stored = variable_with_lltype(field, fn_lltype);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(d.clone()), void_field_const(field)],
        Hlvalue::Variable(stored.clone()),
    ));
    stored
}

fn emit_indirect_call(block: &BlockRef, fnptr: Variable, args: Vec<Hlvalue>, result: &Variable) {
    let mut call_args = vec![Hlvalue::Variable(fnptr)];
    call_args.extend(args);
    call_args.push(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    )));
    block.borrow_mut().operations.push(SpaceOperation::new(
        "indirect_call",
        call_args,
        Hlvalue::Variable(result.clone()),
    ));
}

fn hl_vars(vars: &[Variable]) -> Vec<Hlvalue> {
    vars.iter().map(|v| Hlvalue::Variable(v.clone())).collect()
}

fn clone_vars_like(vars: &[Variable]) -> Vec<Variable> {
    vars.iter()
        .map(|v| {
            let ty = v
                .concretetype()
                .expect("miss-path variable has a concretetype");
            variable_with_lltype(&v.name(), ty)
        })
        .collect()
}

fn mapped_miss_args(
    miss_vars: &[Variable],
    originals: &[Variable],
    copies: &[Variable],
    suffix: &[Hlvalue],
) -> Vec<Hlvalue> {
    let mut args = Vec::with_capacity(miss_vars.len() + suffix.len());
    for v in miss_vars {
        let idx = originals
            .iter()
            .position(|o| o == v)
            .expect("miss-path variable is carried through paranoia");
        args.push(Hlvalue::Variable(copies[idx].clone()));
    }
    args.extend(suffix.iter().cloned());
    args
}

/// `entries.entry_hash(d, i) == hash` (`ll_dict_lookup`). Cached `f_hash`
/// is `ll_hash_from_cache`; without it, recompute via `ll_keyhash_custom`
/// / `ll_hash_custom_fast`.
fn emit_entry_hash_compare(
    block: &BlockRef,
    entries: &Variable,
    slot: &Variable,
    lookup_hash: &Variable,
    d: &Variable,
    ckey: &Variable,
    has_f_hash: bool,
    keyhash_fn: Option<&Constant>,
    hash_takes_dict: bool,
) -> Result<Variable, TyperError> {
    let stored = variable_with_lltype("entry_hash", LowLevelType::Signed);
    if has_f_hash {
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorfield",
            vec![
                Hlvalue::Variable(entries.clone()),
                Hlvalue::Variable(slot.clone()),
                void_field_const("f_hash"),
            ],
            Hlvalue::Variable(stored.clone()),
        ));
    } else {
        let hf = keyhash_fn.cloned().ok_or_else(|| {
            TyperError::message(
                "ll_dict_lookup: entry_hash needs f_hash or keyhash (ll_hash_from_cache / ll_hash_custom_fast)",
            )
        })?;
        block.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            keyhash_call_args(hf, d, ckey, hash_takes_dict),
            Hlvalue::Variable(stored.clone()),
        ));
    }
    let hash_eq = variable_with_lltype("hash_eq", LowLevelType::Bool);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(stored),
            Hlvalue::Variable(lookup_hash.clone()),
        ],
        Hlvalue::Variable(hash_eq.clone()),
    ));
    Ok(hash_eq)
}

/// FunctionRepr (Void unique) bakes `get_unique_llfn` into a
/// `direct_call` constant. FunctionsPBCRepr / Ptr-typed callables store
/// the funcptr on `DICT.fnkeyhash` / `fnkeyeq` and
/// `ll_keyhash_custom` / `ll_keyeq_custom` `getfield` + `indirect_call`
/// it (`hlinvoke` of `r_rdict_hashfn` / `r_rdict_eqfn`).
fn baked_callable_from_repr(r: &dyn Repr) -> Result<Option<Constant>, TyperError> {
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rpbc::FunctionRepr;

    if r.repr_class_id() == ReprClassId::FunctionRepr {
        let fr = (r as &dyn std::any::Any)
            .downcast_ref::<FunctionRepr>()
            .ok_or_else(|| {
                TyperError::message("baked_callable_from_repr: FunctionRepr downcast failed")
            })?;
        match fr.get_unique_llfn()? {
            Hlvalue::Constant(c) => Ok(Some(c)),
            other => Err(TyperError::message(format!(
                "FunctionRepr.get_unique_llfn did not return a Constant, got {other:?}"
            ))),
        }
    } else {
        Ok(None)
    }
}

/// `ll_keyhash_custom(d, key)` (`rordereddict.py`):
/// `objectmodel.hlinvoke(DICT.r_rdict_hashfn, d.fnkeyhash, key)`.
fn build_ll_keyhash_custom_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    hash_fn_lltype: LowLevelType,
    baked: Option<Constant>,
    r_callable: Option<Arc<dyn Repr>>,
) -> Result<PyGraph, TyperError> {
    let d = variable_with_lltype("d", dict_ptr_lltype);
    let key = variable_with_lltype("key", key_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(key.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let hash = variable_with_lltype("hash", LowLevelType::Signed);
    emit_hlinvoke_of_stored_callable(
        &startblock,
        baked,
        &d,
        "fnkeyhash",
        hash_fn_lltype,
        vec![Hlvalue::Variable(key)],
        &hash,
        r_callable.as_deref(),
    )?;
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(hash)],
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
        vec!["d".to_string(), "key".to_string()],
        func,
    ))
}

/// `ll_keyeq_custom(d, key1, key2)` (`rordereddict.py`):
/// `objectmodel.hlinvoke(DICT.r_rdict_eqfn, d.fnkeyeq, key1, key2)`.
fn build_ll_keyeq_custom_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    eq_fn_lltype: LowLevelType,
    baked: Option<Constant>,
    r_callable: Option<Arc<dyn Repr>>,
) -> Result<PyGraph, TyperError> {
    let d = variable_with_lltype("d", dict_ptr_lltype);
    let key1 = variable_with_lltype("key1", key_lltype.clone());
    let key2 = variable_with_lltype("key2", key_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(key1.clone()),
        Hlvalue::Variable(key2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let found = variable_with_lltype("found", LowLevelType::Bool);
    emit_hlinvoke_of_stored_callable(
        &startblock,
        baked,
        &d,
        "fnkeyeq",
        eq_fn_lltype,
        vec![Hlvalue::Variable(key1), Hlvalue::Variable(key2)],
        &found,
        r_callable.as_deref(),
    )?;
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(found)],
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
        vec!["d".to_string(), "key1".to_string(), "key2".to_string()],
        func,
    ))
}

/// `rtype_hlinvoke` (`rbuiltin.py`) + `FunctionReprBase.call` /
/// `SmallFunctionSetPBCRepr.call` / `MethodsPBCRepr.redispatch_call`
/// (`rpbc.py`) for a stored `fnkeyhash` / `fnkeyeq`.
///
/// FunctionRepr Void unique: `direct_call(baked, *args)`.
/// FunctionsPBCRepr / Ptr(Func): `getfield(d, field)` then
/// `indirect_call(fn, *args, c_graphs)`.
/// SmallFunctionSetPBCRepr: `getfield` Char selector then
/// `direct_call(dispatcher, sel, *args)`.
/// MethodsPBCRepr / MethodOfFrozenPBCRepr with a unique FunctionRepr
/// impl: `getfield` receiver then `direct_call(llfn, receiver, *args)`.
fn emit_hlinvoke_of_stored_callable(
    block: &BlockRef,
    baked: Option<Constant>,
    d: &Variable,
    field: &str,
    fn_lltype: LowLevelType,
    args: Vec<Hlvalue>,
    result: &Variable,
    r_callable: Option<&dyn Repr>,
) -> Result<(), TyperError> {
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rpbc::SmallFunctionSetPBCRepr;

    if let Some(baked_fn) = baked {
        let mut call_args = vec![Hlvalue::Constant(baked_fn)];
        call_args.extend(args);
        emit_direct_call(block, call_args, result);
        return Ok(());
    }
    if let Some(r) = r_callable {
        match r.repr_class_id() {
            ReprClassId::SmallFunctionSetPBCRepr => {
                let r_set = (r as &dyn std::any::Any)
                    .downcast_ref::<SmallFunctionSetPBCRepr>()
                    .ok_or_else(|| {
                        TyperError::message("hlinvoke: SmallFunctionSetPBCRepr downcast failed")
                    })?;
                let sel = emit_getfield_stored(block, d, field, fn_lltype);
                let dispatcher = small_function_set_dispatcher_for_args(r_set, &args, result)?;
                let mut call_args = vec![Hlvalue::Constant(dispatcher), Hlvalue::Variable(sel)];
                call_args.extend(args);
                emit_direct_call(block, call_args, result);
                return Ok(());
            }
            ReprClassId::MethodsPBCRepr | ReprClassId::MethodOfFrozenPBCRepr => {
                return emit_bound_method_hlinvoke(block, d, field, fn_lltype, r, args, result);
            }
            _ => {}
        }
    }
    if is_func_ptr_lltype(&fn_lltype) {
        let fnptr = emit_getfield_stored(block, d, field, fn_lltype);
        emit_indirect_call(block, fnptr, args, result);
        return Ok(());
    }
    Err(TyperError::message(format!(
        "hlinvoke of stored {field}: lltype {fn_lltype:?} is not a function pointer \
         (FunctionsPBCRepr / Ptr(Func)); FunctionRepr unique uses direct_call; \
         SmallFunctionSetPBCRepr / MethodsPBCRepr need their callable Repr"
    )))
}

fn small_function_set_dispatcher_for_args(
    r_set: &crate::translator::rtyper::rpbc::SmallFunctionSetPBCRepr,
    extra_args: &[Hlvalue],
    result: &Variable,
) -> Result<Constant, TyperError> {
    let argtypes: Vec<LowLevelType> = extra_args
        .iter()
        .map(hlvalue_lltype)
        .collect::<Result<Vec<_>, _>>()?;
    let resulttype = result.concretetype().ok_or_else(|| {
        TyperError::message("hlinvoke SmallFunctionSetPBCRepr: result missing concretetype")
    })?;
    let family = r_set.base.callfamily.as_ref().ok_or_else(|| {
        TyperError::message("hlinvoke SmallFunctionSetPBCRepr: callfamily missing")
    })?;
    let want = extra_args.len();
    let (shape, index) = {
        let family_ref = family.borrow();
        let (shape, table) = family_ref
            .calltables
            .iter()
            .find(|(shape, table)| shape.shape_cnt == want && !table.is_empty())
            .or_else(|| {
                family_ref
                    .calltables
                    .iter()
                    .find(|(_, table)| !table.is_empty())
            })
            .ok_or_else(|| {
                TyperError::message("hlinvoke SmallFunctionSetPBCRepr: empty calltable")
            })?;
        let _ = table;
        (shape.clone(), 0usize)
    };
    r_set.dispatcher(&shape, index, &argtypes, &resulttype)
}

fn emit_bound_method_hlinvoke(
    block: &BlockRef,
    d: &Variable,
    field: &str,
    fn_lltype: LowLevelType,
    r_callable: &dyn Repr,
    args: Vec<Hlvalue>,
    result: &Variable,
) -> Result<(), TyperError> {
    let (r_func, nimplicit) = r_callable.get_r_implfunc()?;
    if nimplicit == 0 {
        return Err(TyperError::message(format!(
            "hlinvoke of stored {field}: bound-method get_r_implfunc nimplicitarg is 0"
        )));
    }
    let receiver = emit_getfield_stored(block, d, field, fn_lltype);
    match baked_callable_from_repr(r_func.as_ref())? {
        Some(llfn) => {
            let mut call_args = vec![Hlvalue::Constant(llfn), Hlvalue::Variable(receiver)];
            call_args.extend(args);
            emit_direct_call(block, call_args, result);
            Ok(())
        }
        None => Err(TyperError::message(format!(
            "hlinvoke of stored {field}: non-unique impl {} needs hop.dispatch \
             (MethodsPBCRepr.redispatch_call / getclsfield)",
            r_func.repr_string()
        ))),
    }
}

/// After `found = d.keyeq(...)`. Without paranoia, switch on `found`.
/// With paranoia, restart the lookup (`return ll_dict_lookup(...)`) if
/// the user eq mutated the table; chained exitswitches because there is
/// no `bool_or`.
///
/// Miss-path variables (`miss_vars` plus `miss_const_suffix`) are
/// copies on every paranoia block so the False link out of
/// `block_para_found` is well-formed (`ll_dict_lookup`).
#[expect(
    clippy::too_many_arguments,
    reason = "Mirrors the ll_dict_lookup paranoia predicate operands"
)]
fn close_keyeq_found_or_paranoia(
    keyeq_block: &BlockRef,
    found: &Variable,
    d: &Variable,
    entries: &Variable,
    indexes: &Variable,
    key: &Variable,
    hash: &Variable,
    sf: &Variable,
    slot: &Variable,
    ckey: &Variable,
    miss_vars: &[Variable],
    miss_const_suffix: Vec<Hlvalue>,
    miss_target: BlockRef,
    startblock: BlockRef,
    returnblock: BlockRef,
    dict_ptr_lltype: &LowLevelType,
    entries_ptr_lltype: &LowLevelType,
    indexes_ptr_lltype: &LowLevelType,
    key_lltype: &LowLevelType,
    eq_op: &str,
    paranoia: bool,
    t_const: Hlvalue,
) {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let mut miss_from_keyeq = hl_vars(miss_vars);
    miss_from_keyeq.extend(miss_const_suffix.iter().cloned());
    if !paranoia {
        keyeq_block.borrow_mut().exitswitch = Some(var(found));
        keyeq_block.closeblock(vec![
            Link::new(vec![var(slot)], Some(returnblock), Some(bool_true())).into_ref(),
            Link::new(miss_from_keyeq, Some(miss_target), Some(bool_false())).into_ref(),
        ]);
        return;
    }

    let carry_origs = vec![
        d.clone(),
        entries.clone(),
        indexes.clone(),
        key.clone(),
        hash.clone(),
        sf.clone(),
        slot.clone(),
        ckey.clone(),
        found.clone(),
    ];
    let extra_origs: Vec<Variable> = miss_vars
        .iter()
        .filter(|v| !carry_origs.contains(v))
        .cloned()
        .collect();
    let map_origs: Vec<Variable> = carry_origs
        .iter()
        .cloned()
        .chain(extra_origs.iter().cloned())
        .collect();
    let carry: Vec<Hlvalue> = hl_vars(&carry_origs)
        .into_iter()
        .chain(hl_vars(&extra_origs))
        .collect();

    let para_inputs = |d: &Variable,
                       entries: &Variable,
                       indexes: &Variable,
                       key: &Variable,
                       hash: &Variable,
                       sf: &Variable,
                       slot: &Variable,
                       ckey: &Variable,
                       found: &Variable,
                       extras: &[Variable]| {
        let mut args = vec![
            var(d),
            var(entries),
            var(indexes),
            var(key),
            var(hash),
            var(sf),
            var(slot),
            var(ckey),
            var(found),
        ];
        args.extend(hl_vars(extras));
        args
    };
    let map_copies = |nine: [&Variable; 9], extras: &[Variable]| -> Vec<Variable> {
        nine.into_iter()
            .cloned()
            .chain(extras.iter().cloned())
            .collect()
    };

    let pe_d = new_var("d", dict_ptr_lltype.clone());
    let pe_entries = new_var("entries", entries_ptr_lltype.clone());
    let pe_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let pe_key = new_var("key", key_lltype.clone());
    let pe_hash = new_var("hash", LowLevelType::Signed);
    let pe_sf = new_var("store_flag", LowLevelType::Signed);
    let pe_slot = new_var("slot", LowLevelType::Signed);
    let pe_ckey = new_var("checkingkey", key_lltype.clone());
    let pe_found = new_var("found", LowLevelType::Bool);
    let pe_extra = clone_vars_like(&extra_origs);
    let block_para_entries = Block::shared(para_inputs(
        &pe_d,
        &pe_entries,
        &pe_indexes,
        &pe_key,
        &pe_hash,
        &pe_sf,
        &pe_slot,
        &pe_ckey,
        &pe_found,
        &pe_extra,
    ));
    let pe_map = map_copies(
        [
            &pe_d,
            &pe_entries,
            &pe_indexes,
            &pe_key,
            &pe_hash,
            &pe_sf,
            &pe_slot,
            &pe_ckey,
            &pe_found,
        ],
        &pe_extra,
    );

    let pi_d = new_var("d", dict_ptr_lltype.clone());
    let pi_entries = new_var("entries", entries_ptr_lltype.clone());
    let pi_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let pi_key = new_var("key", key_lltype.clone());
    let pi_hash = new_var("hash", LowLevelType::Signed);
    let pi_sf = new_var("store_flag", LowLevelType::Signed);
    let pi_slot = new_var("slot", LowLevelType::Signed);
    let pi_ckey = new_var("checkingkey", key_lltype.clone());
    let pi_found = new_var("found", LowLevelType::Bool);
    let pi_extra = clone_vars_like(&extra_origs);
    let block_para_indexes = Block::shared(para_inputs(
        &pi_d,
        &pi_entries,
        &pi_indexes,
        &pi_key,
        &pi_hash,
        &pi_sf,
        &pi_slot,
        &pi_ckey,
        &pi_found,
        &pi_extra,
    ));
    let pi_map = map_copies(
        [
            &pi_d,
            &pi_entries,
            &pi_indexes,
            &pi_key,
            &pi_hash,
            &pi_sf,
            &pi_slot,
            &pi_ckey,
            &pi_found,
        ],
        &pi_extra,
    );

    let pv_d = new_var("d", dict_ptr_lltype.clone());
    let pv_entries = new_var("entries", entries_ptr_lltype.clone());
    let pv_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let pv_key = new_var("key", key_lltype.clone());
    let pv_hash = new_var("hash", LowLevelType::Signed);
    let pv_sf = new_var("store_flag", LowLevelType::Signed);
    let pv_slot = new_var("slot", LowLevelType::Signed);
    let pv_ckey = new_var("checkingkey", key_lltype.clone());
    let pv_found = new_var("found", LowLevelType::Bool);
    let pv_extra = clone_vars_like(&extra_origs);
    let block_para_valid = Block::shared(para_inputs(
        &pv_d,
        &pv_entries,
        &pv_indexes,
        &pv_key,
        &pv_hash,
        &pv_sf,
        &pv_slot,
        &pv_ckey,
        &pv_found,
        &pv_extra,
    ));
    let pv_map = map_copies(
        [
            &pv_d,
            &pv_entries,
            &pv_indexes,
            &pv_key,
            &pv_hash,
            &pv_sf,
            &pv_slot,
            &pv_ckey,
            &pv_found,
        ],
        &pv_extra,
    );

    let pk_d = new_var("d", dict_ptr_lltype.clone());
    let pk_entries = new_var("entries", entries_ptr_lltype.clone());
    let pk_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let pk_key = new_var("key", key_lltype.clone());
    let pk_hash = new_var("hash", LowLevelType::Signed);
    let pk_sf = new_var("store_flag", LowLevelType::Signed);
    let pk_slot = new_var("slot", LowLevelType::Signed);
    let pk_ckey = new_var("checkingkey", key_lltype.clone());
    let pk_found = new_var("found", LowLevelType::Bool);
    let pk_extra = clone_vars_like(&extra_origs);
    let block_para_key = Block::shared(para_inputs(
        &pk_d,
        &pk_entries,
        &pk_indexes,
        &pk_key,
        &pk_hash,
        &pk_sf,
        &pk_slot,
        &pk_ckey,
        &pk_found,
        &pk_extra,
    ));
    let pk_map = map_copies(
        [
            &pk_d,
            &pk_entries,
            &pk_indexes,
            &pk_key,
            &pk_hash,
            &pk_sf,
            &pk_slot,
            &pk_ckey,
            &pk_found,
        ],
        &pk_extra,
    );

    let pf_found = new_var("found", LowLevelType::Bool);
    let pf_slot = new_var("slot", LowLevelType::Signed);
    let pf_miss = clone_vars_like(miss_vars);
    let mut pf_inputs = vec![var(&pf_found), var(&pf_slot)];
    pf_inputs.extend(hl_vars(&pf_miss));
    let block_para_found = Block::shared(pf_inputs);

    keyeq_block.closeblock(vec![
        Link::new(carry, Some(block_para_entries.clone()), None).into_ref(),
    ]);

    let pe_entries2 = new_var("entries2", entries_ptr_lltype.clone());
    push(
        &block_para_entries,
        "getfield",
        vec![var(&pe_d), void_field_const("entries")],
        &pe_entries2,
    );
    let pe_ne = new_var("entries_ne", LowLevelType::Bool);
    push(
        &block_para_entries,
        "ptr_ne",
        vec![var(&pe_entries), var(&pe_entries2)],
        &pe_ne,
    );
    block_para_entries.borrow_mut().exitswitch = Some(var(&pe_ne));
    let pe_cont = hl_vars(&pe_map);
    block_para_entries.closeblock(vec![
        Link::new(
            vec![
                var(&pe_d),
                var(&pe_key),
                var(&pe_hash),
                var(&pe_sf),
                t_const.clone(),
            ],
            Some(startblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            pe_cont,
            Some(block_para_indexes.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let pi_gcref_now = new_var("indexes_gcref", GCREF.clone());
    push(
        &block_para_indexes,
        "getfield",
        vec![var(&pi_d), void_field_const("indexes")],
        &pi_gcref_now,
    );
    let pi_gcref_old = new_var("indexes_gcref_old", GCREF.clone());
    push(
        &block_para_indexes,
        "cast_opaque_ptr",
        vec![var(&pi_indexes)],
        &pi_gcref_old,
    );
    let pi_ne = new_var("indexes_ne", LowLevelType::Bool);
    push(
        &block_para_indexes,
        "ptr_ne",
        vec![var(&pi_gcref_old), var(&pi_gcref_now)],
        &pi_ne,
    );
    block_para_indexes.borrow_mut().exitswitch = Some(var(&pi_ne));
    let pi_cont = hl_vars(&pi_map);
    block_para_indexes.closeblock(vec![
        Link::new(
            vec![
                var(&pi_d),
                var(&pi_key),
                var(&pi_hash),
                var(&pi_sf),
                t_const.clone(),
            ],
            Some(startblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(pi_cont, Some(block_para_valid.clone()), Some(bool_false())).into_ref(),
    ]);

    let pv_valid = new_var("valid", LowLevelType::Bool);
    push(
        &block_para_valid,
        "getinteriorfield",
        vec![var(&pv_entries), var(&pv_slot), void_field_const("f_valid")],
        &pv_valid,
    );
    let pv_not_valid = new_var("not_valid", LowLevelType::Bool);
    push(
        &block_para_valid,
        "bool_not",
        vec![var(&pv_valid)],
        &pv_not_valid,
    );
    block_para_valid.borrow_mut().exitswitch = Some(var(&pv_not_valid));
    let pv_cont = hl_vars(&pv_map);
    block_para_valid.closeblock(vec![
        Link::new(
            vec![
                var(&pv_d),
                var(&pv_key),
                var(&pv_hash),
                var(&pv_sf),
                t_const.clone(),
            ],
            Some(startblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(pv_cont, Some(block_para_key.clone()), Some(bool_false())).into_ref(),
    ]);

    let pk_now = new_var("key_now", key_lltype.clone());
    push(
        &block_para_key,
        "getinteriorfield",
        vec![var(&pk_entries), var(&pk_slot), void_field_const("key")],
        &pk_now,
    );
    let pk_same = new_var("key_same", LowLevelType::Bool);
    push(
        &block_para_key,
        eq_op,
        vec![var(&pk_now), var(&pk_ckey)],
        &pk_same,
    );
    let pk_changed = new_var("key_changed", LowLevelType::Bool);
    push(
        &block_para_key,
        "bool_not",
        vec![var(&pk_same)],
        &pk_changed,
    );
    block_para_key.borrow_mut().exitswitch = Some(var(&pk_changed));
    let mut pk_to_found = vec![var(&pk_found), var(&pk_slot)];
    pk_to_found.extend(mapped_miss_args(miss_vars, &map_origs, &pk_map, &[]));
    block_para_key.closeblock(vec![
        Link::new(
            vec![
                var(&pk_d),
                var(&pk_key),
                var(&pk_hash),
                var(&pk_sf),
                t_const,
            ],
            Some(startblock),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            pk_to_found,
            Some(block_para_found.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    block_para_found.borrow_mut().exitswitch = Some(var(&pf_found));
    let mut pf_miss_args = hl_vars(&pf_miss);
    pf_miss_args.extend(miss_const_suffix);
    block_para_found.closeblock(vec![
        Link::new(vec![var(&pf_slot)], Some(returnblock), Some(bool_true())).into_ref(),
        Link::new(pf_miss_args, Some(miss_target), Some(bool_false())).into_ref(),
    ]);
}

/// `tool.uid.Hashable` cache key for `OrderedDictRepr.convert_const`
/// (`rordereddict.py`): `self.dict_cache[Constant(dictobj)]`.
///
/// `ConstValue::Dict` is `Arc<IndexMap<..>>`; the key clones that handle
/// and hashes `Arc::as_ptr` (`uid.py` `id(dictobj)`). A prebuilt
/// `objectmodel.r_dict` is a `HostObject` instance; `Hashable` falls
/// back to `id(dictobj)` and `HostObject` Hash/Eq is `Arc::ptr_eq`.
/// The strong reference keeps the allocation alive so a later dict
/// cannot reuse the address and alias the cache entry.
#[derive(Clone, Debug)]
enum DictCacheKey {
    Dict {
        items: Arc<indexmap::IndexMap<ConstValue, ConstValue>>,
    },
    RDict {
        obj: HostObject,
    },
}

impl PartialEq for DictCacheKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Dict { items: a }, Self::Dict { items: b }) => Arc::ptr_eq(a, b),
            (Self::RDict { obj: a }, Self::RDict { obj: b }) => a == b,
            _ => false,
        }
    }
}

impl Eq for DictCacheKey {}

impl Hash for DictCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Dict { items } => (Arc::as_ptr(items) as usize).hash(state),
            Self::RDict { obj } => obj.hash(state),
        }
    }
}

fn dict_cache_key(value: &ConstValue) -> Result<DictCacheKey, TyperError> {
    match value {
        ConstValue::Dict(items) => Ok(DictCacheKey::Dict {
            items: Arc::clone(items),
        }),
        ConstValue::HostObject(obj) if is_objectmodel_r_dict_instance(obj) => {
            Ok(DictCacheKey::RDict { obj: obj.clone() })
        }
        other => Err(TyperError::message(format!(
            "dict_cache_key: expected a dict, got {other:?}"
        ))),
    }
}

fn dict_const_len(value: &ConstValue) -> Result<usize, TyperError> {
    match value {
        ConstValue::Dict(items) => Ok(items.len()),
        ConstValue::HostObject(obj) if is_objectmodel_r_dict_instance(obj) => {
            match host_instance_attr(obj, "_dict")? {
                ConstValue::Dict(items) => Ok(items.len()),
                other => Err(TyperError::message(format!(
                    "r_dict._dict is not a dict, got {other:?}"
                ))),
            }
        }
        other => Err(TyperError::message(format!(
            "dict_const_len: expected a dict, got {other:?}"
        ))),
    }
}

fn host_instance_attr(obj: &HostObject, name: &str) -> Result<ConstValue, TyperError> {
    obj.instance_get(name).map_err(|err| {
        TyperError::message(format!(
            "objectmodel.r_dict missing attribute {name}: {err:?}"
        ))
    })
}

/// RPython `class OrderedDictRepr(AbstractDictRepr)`
/// (`lltypesystem/rordereddict.py`).
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct OrderedDictRepr {
    pub base: AbstractDictRepr,
    pub DICTKEY: LowLevelType,
    pub DICTVALUE: LowLevelType,
    /// RPython `Struct("odictentry", ...)`.
    pub DICTENTRY: Struct,
    pub DICTENTRYARRAY: Array,
    /// RPython `GcStruct("dicttable", ...)`.
    pub DICT: Struct,
    lowleveltype: LowLevelType,
    /// `OrderedDictRepr.convert_const` (`rordereddict.py`):
    /// `self.dict_cache[Constant(dictobj)]`. Keyed by [`DictCacheKey`]
    /// (`Hashable` / `Constant(dictobj)`), matching
    /// `InstanceRepr.iprebuiltinstances` (`HostObject` Arc).
    dict_cache: RefCell<HashMap<DictCacheKey, _ptr>>,
}

impl OrderedDictRepr {
    pub fn new(
        rtyper: Rc<RPythonTyper>,
        key_repr: Arc<dyn Repr>,
        value_repr: Arc<dyn Repr>,
        dictdef: DictDef,
        custom_eq_hash_repr: Option<(Arc<dyn Repr>, Arc<dyn Repr>)>,
        force_non_null: bool,
        simple_hash_eq: bool,
    ) -> Result<Self, TyperError> {
        let custom_eq_hash = custom_eq_hash_repr.is_some();
        let (external_key_repr, key_repr) =
            AbstractDictRepr::pickrepr(&rtyper, key_repr, custom_eq_hash)?;
        let (external_value_repr, value_repr) = if custom_eq_hash {
            (value_repr.clone(), value_repr)
        } else {
            crate::translator::rtyper::rclass::externalvsinternal(&rtyper, value_repr, true)?
        };
        let dictkey_lltype = key_repr.lowleveltype().clone();
        let dictvalue_lltype = value_repr.lowleveltype().clone();

        let mut entryfields = vec![
            ("key".into(), dictkey_lltype.clone()),
            ("f_valid".into(), LowLevelType::Bool),
            ("value".into(), dictvalue_lltype.clone()),
        ];
        if !simple_hash_eq {
            entryfields.push(("f_hash".into(), LowLevelType::Signed));
        }
        let dictentry = Struct::new("odictentry", entryfields);
        let dictentryarray = Array::gc(LowLevelType::Struct(Box::new(dictentry.clone())));
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(dictentryarray.clone()),
        }));
        let mut fields = vec![
            ("num_live_items".into(), LowLevelType::Signed),
            ("num_ever_used_items".into(), LowLevelType::Signed),
            ("resize_counter".into(), LowLevelType::Signed),
            ("indexes".into(), GCREF.clone()),
            ("lookup_function_no".into(), LowLevelType::Signed),
            ("entries".into(), entries_ptr),
        ];
        let mut adtmeths = Vec::new();
        if let Some((r_rdict_eqfn, r_rdict_hashfn)) = &custom_eq_hash_repr {
            // get_ll_dict (`rordereddict.py`): fnkeyeq / fnkeyhash hold the
            // runtime PBC funcptrs; DICT adtmeths carry r_rdict_eqfn /
            // r_rdict_hashfn (the Repr objects hlinvoke reads), keyhash /
            // keyeq (ll_keyhash_custom / ll_keyeq_custom), and paranoia.
            fields.push(("fnkeyeq".into(), r_rdict_eqfn.lowleveltype().clone()));
            fields.push(("fnkeyhash".into(), r_rdict_hashfn.lowleveltype().clone()));
            adtmeths.push(("keyhash".into(), ConstValue::byte_str("ll_keyhash_custom")));
            adtmeths.push(("keyeq".into(), ConstValue::byte_str("ll_keyeq_custom")));
            // get_ll_dict stores the live Repr objects as adtmeths so
            // hlinvoke / helper specialization sees the callback pair,
            // not just FunctionRepr's Void lowleveltype.
            adtmeths.push((
                "r_rdict_eqfn".into(),
                ConstValue::Repr(ReprConst::new(Arc::clone(r_rdict_eqfn))),
            ));
            adtmeths.push((
                "r_rdict_hashfn".into(),
                ConstValue::Repr(ReprConst::new(Arc::clone(r_rdict_hashfn))),
            ));
            adtmeths.push(("paranoia".into(), ConstValue::Bool(!simple_hash_eq)));
        } else {
            // get_ll_dict (`rordereddict.py`) plain-dict arm: staticAdtMethod
            // of the key repr's `get_ll_hash_function` / `get_ll_eq_function`
            // (the latter may be None), plus `paranoia = False`.
            let hash_fn = key_repr.get_ll_hash_function(&rtyper)?.ok_or_else(|| {
                TyperError::message(
                    "OrderedDictRepr: key repr has no ll hash function — dict key hash not wired",
                )
            })?;
            adtmeths.push((
                "keyhash".into(),
                sub_helper_funcptr_constant(&rtyper, &hash_fn)?.value,
            ));
            let eq_value = match key_repr.get_ll_eq_function(&rtyper)? {
                Some(eq_fn) => sub_helper_funcptr_constant(&rtyper, &eq_fn)?.value,
                None => ConstValue::None,
            };
            adtmeths.push(("keyeq".into(), eq_value));
            adtmeths.push(("paranoia".into(), ConstValue::Bool(false)));
        }
        adtmeths.push((
            "KEY".into(),
            ConstValue::LowLevelType(Box::new(dictkey_lltype.clone())),
        ));
        adtmeths.push((
            "VALUE".into(),
            ConstValue::LowLevelType(Box::new(dictvalue_lltype.clone())),
        ));
        let dict = Struct::gc_with_adtmeths_and_hints(
            "dicttable",
            fields,
            adtmeths,
            vec![("dict".into(), ConstValue::Bool(true))],
        );
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(dict.clone()),
        }));

        Ok(OrderedDictRepr {
            base: AbstractDictRepr {
                state: ReprState::new(),
                rtyper,
                external_key_repr,
                key_repr,
                external_value_repr,
                value_repr,
                dictdef,
                custom_eq_hash,
                force_non_null,
                simple_hash_eq,
                custom_eq_hash_repr,
            },
            DICTKEY: dictkey_lltype,
            DICTVALUE: dictvalue_lltype,
            DICTENTRY: dictentry,
            DICTENTRYARRAY: dictentryarray,
            DICT: dict,
            lowleveltype,
            dict_cache: RefCell::new(HashMap::new()),
        })
    }

    /// RPython `OrderedDictRepr.ll_newdict = staticmethod(ll_newdict)`
    /// (`rordereddict.py`), reached from `rdict.rtype_newdict`
    /// (`rdict.py:60-65`):
    ///
    /// ```python
    /// def rtype_newdict(hop):
    ///     hop.inputargs()    # no arguments expected
    ///     r_dict = hop.r_result
    ///     cDICT = hop.inputconst(lltype.Void, r_dict.DICT)
    ///     v_result = hop.gendirectcall(r_dict.ll_newdict, cDICT)
    ///     return v_result
    /// ```
    ///
    /// The `cDICT` argument carries no runtime value (`Void`), so instead of
    /// threading it through `gendirectcall` the dict specialization is baked
    /// into the cached helper via `lowleveltype`/`DICT`/`DICTENTRYARRAY` —
    /// same shape as [`crate::translator::rtyper::rlist::rtype_newlist`]
    /// baking its item type into the `ll_newlist` helper closure.
    pub fn ll_newdict(&self, hop: &HighLevelOp) -> RTypeResult {
        let ptr_lltype = self.lowleveltype.clone();
        let dict_struct = self.DICT.clone();
        let entryarray = self.DICTENTRYARRAY.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_newdict".to_string(),
            vec![],
            ptr_lltype.clone(),
            move |_rtyper, _args, _result| {
                build_ll_dict_newdict_helper_graph(
                    "ll_newdict",
                    ptr_lltype.clone(),
                    dict_struct.clone(),
                    entryarray.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, vec![])
    }

    /// Shared `convert_const` body for a host `dict` and a prebuilt
    /// `objectmodel.r_dict` (`rordereddict.py` OrderedDictRepr.convert_const).
    fn convert_const_prebuilt_dict(&self, dictobj: &ConstValue) -> Result<Constant, TyperError> {
        let cache_key = dict_cache_key(dictobj)?;
        if let Some(cached) = self.dict_cache.borrow().get(&cache_key) {
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(cached.clone())),
                self.lowleveltype.clone(),
            ));
        }
        self.setup()?;
        self.setup_final()?;
        let mut l_dict = ll_newdict_size(&self.DICT, dict_const_len(dictobj)?)?;
        ll_no_initial_index(&mut l_dict)?;
        self.dict_cache
            .borrow_mut()
            .insert(cache_key, l_dict.clone());
        if matches!(self.base.key_repr.lowleveltype(), LowLevelType::Address) {
            return Err(TyperError::message("No prebuilt dicts of address keys"));
        }
        match dictobj {
            ConstValue::HostObject(obj) if is_objectmodel_r_dict_instance(obj) => {
                self.convert_const_r_dict(obj, &mut l_dict)?;
            }
            ConstValue::Dict(items) => {
                for (dictkey, dictvalue) in items.iter() {
                    let llkey = convert_dict_item_const(self.base.key_repr.as_ref(), dictkey)?;
                    let llvalue =
                        convert_dict_item_const(self.base.value_repr.as_ref(), dictvalue)?;
                    _ll_dict_insert_no_index(&mut l_dict, llkey, llvalue)?;
                }
            }
            other => {
                return Err(TyperError::message(format!("expected a dict: {other:?}")));
            }
        }
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(l_dict)),
            self.lowleveltype.clone(),
        ))
    }

    /// `isinstance(dictobj, objectmodel.r_dict)` arm of
    /// `OrderedDictRepr.convert_const` (`rordereddict.py`).
    fn convert_const_r_dict(
        &self,
        dictobj: &HostObject,
        l_dict: &mut _ptr,
    ) -> Result<(), TyperError> {
        let (r_eqfn, r_hashfn) = self.base.custom_eq_hash_repr.as_ref().ok_or_else(|| {
            TyperError::message(
                "OrderedDictRepr.convert_const: r_dict requires r_rdict_eqfn / r_rdict_hashfn",
            )
        })?;
        if !matches!(r_eqfn.lowleveltype(), LowLevelType::Void) {
            let key_eq = host_instance_attr(dictobj, "key_eq")?;
            let l_fn = convert_dict_item_const(r_eqfn.as_ref(), &key_eq)?;
            l_dict
                .setattr("fnkeyeq", l_fn)
                .map_err(TyperError::message)?;
        }
        if !matches!(r_hashfn.lowleveltype(), LowLevelType::Void) {
            let key_hash = host_instance_attr(dictobj, "key_hash")?;
            let l_fn = convert_dict_item_const(r_hashfn.as_ref(), &key_hash)?;
            l_dict
                .setattr("fnkeyhash", l_fn)
                .map_err(TyperError::message)?;
        }
        let inner = host_instance_attr(dictobj, "_dict")?;
        let ConstValue::Dict(items) = inner else {
            return Err(TyperError::message(format!(
                "r_dict._dict is not a dict, got {inner:?}"
            )));
        };
        for (dictkeycontainer, dictvalue) in items.iter() {
            let dictkey = match dictkeycontainer {
                ConstValue::HostObject(container) => host_instance_attr(container, "key")?,
                other => {
                    return Err(TyperError::message(format!(
                        "r_dict._dict key is not _r_dictkey, got {other:?}"
                    )));
                }
            };
            let llkey = convert_dict_item_const(self.base.key_repr.as_ref(), &dictkey)?;
            let llvalue = convert_dict_item_const(self.base.value_repr.as_ref(), dictvalue)?;
            _ll_dict_insert_no_index(l_dict, llkey, llvalue)?;
        }
        Ok(())
    }

    fn entries_ptr_lltype(&self) -> LowLevelType {
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(self.DICTENTRYARRAY.clone()),
        }))
    }

    /// Mint the shared lookup chain (`ll_dict_lookup` ->
    /// `ll_malloc_indexes_and_choose_lookup` -> `ll_dict_create_initial_index`
    /// -> `ll_ensure_indexes` -> `ll_call_lookup_function`) plus the key
    /// hash helper, returning both as callable `direct_call` funcptr
    /// constants for [`Repr::rtype_getitem`] / [`pair_ordereddict_repr_rtype_contains`]
    /// to bake into their own getitem/contains helper graphs.
    ///
    /// Custom `r_dict` (`custom_eq_hash`) uses `ll_keyhash_custom` /
    /// `ll_keyeq_custom` (`rordereddict.py`) instead of the key repr's
    /// `get_ll_hash_function` / `get_ll_eq_function`. Those helpers
    /// hlinvoke the stored `fnkeyhash` / `fnkeyeq` fields; `d.paranoia`
    /// is `not simple_hash_eq` and enables the mutation-restart path
    /// in `ll_dict_lookup`.
    fn lookup_chain_helpers(&self, hop: &HighLevelOp) -> Result<LookupChainHelpers, TyperError> {
        self.lookup_chain_helpers_on(&hop.rtyper)
    }

    /// Mint the lookup chain for a dict specialization without a `HighLevelOp`.
    /// `ll_dict_getitem` / `ll_dict_setitem` (`rordereddict.py`) are ordinary
    /// helpers; a synthesized caller (`rffi.py _get_raw_address_buf_from_string`)
    /// needs the same funcptrs `rtype_getitem` / `rtype_setitem` bake.
    fn lookup_chain_helpers_on(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<LookupChainHelpers, TyperError> {
        let dict_ptr = self.lowleveltype.clone();
        let entries_ptr = self.entries_ptr_lltype();
        let key_lltype = self.DICTKEY.clone();

        let (hash_fn_const, eq_fn_const, hash_takes_dict, eq_takes_dict, paranoia) = if self
            .base
            .custom_eq_hash
        {
            let (r_eqfn, r_hashfn) = self.base.custom_eq_hash_repr.as_ref().ok_or_else(|| {
                TyperError::message(
                    "OrderedDictRepr: custom_eq_hash reprs missing for r_dict lookup",
                )
            })?;
            let hash_fn_lltype = r_hashfn.lowleveltype().clone();
            let eq_fn_lltype = r_eqfn.lowleveltype().clone();
            let baked_hash = baked_callable_from_repr(r_hashfn.as_ref())?;
            let baked_eq = baked_callable_from_repr(r_eqfn.as_ref())?;
            let r_hashfn = Arc::clone(r_hashfn);
            let r_eqfn = Arc::clone(r_eqfn);
            let hash_fn = {
                let dict_ptr = dict_ptr.clone();
                let key_lltype = key_lltype.clone();
                let hash_fn_lltype = hash_fn_lltype.clone();
                let baked_hash = baked_hash.clone();
                let r_hashfn = r_hashfn.clone();
                rtyper.lowlevel_helper_function_with_builder(
                    "ll_keyhash_custom",
                    vec![dict_ptr.clone(), key_lltype.clone()],
                    LowLevelType::Signed,
                    move |_rtyper, _args, _result| {
                        build_ll_keyhash_custom_helper_graph(
                            "ll_keyhash_custom",
                            dict_ptr.clone(),
                            key_lltype.clone(),
                            hash_fn_lltype.clone(),
                            baked_hash.clone(),
                            Some(r_hashfn.clone()),
                        )
                    },
                )?
            };
            let eq_fn = {
                let dict_ptr = dict_ptr.clone();
                let key_lltype = key_lltype.clone();
                let eq_fn_lltype = eq_fn_lltype.clone();
                let baked_eq = baked_eq.clone();
                let r_eqfn = r_eqfn.clone();
                rtyper.lowlevel_helper_function_with_builder(
                    "ll_keyeq_custom",
                    vec![dict_ptr.clone(), key_lltype.clone(), key_lltype.clone()],
                    LowLevelType::Bool,
                    move |_rtyper, _args, _result| {
                        build_ll_keyeq_custom_helper_graph(
                            "ll_keyeq_custom",
                            dict_ptr.clone(),
                            key_lltype.clone(),
                            eq_fn_lltype.clone(),
                            baked_eq.clone(),
                            Some(r_eqfn.clone()),
                        )
                    },
                )?
            };
            (
                sub_helper_funcptr_constant(rtyper, &hash_fn)?,
                Some(sub_helper_funcptr_constant(rtyper, &eq_fn)?),
                true,
                true,
                !self.base.simple_hash_eq,
            )
        } else {
            let hash_fn: LowLevelFunction = self
                    .base
                    .key_repr
                    .get_ll_hash_function(rtyper)?
                    .ok_or_else(|| {
                        TyperError::message(
                            "OrderedDictRepr: key repr has no ll hash function — dict key eq function not wired",
                        )
                    })?;
            let hash_fn_const = sub_helper_funcptr_constant(rtyper, &hash_fn)?;
            // `d.keyeq` for a plain (non-custom_eq_hash) dict is the key repr's
            // own `get_ll_eq_function` result — `None` for identity keys
            // (int/bool/char/unichar/instance-without-__eq__), `Some(ll_streq)`
            // for str keys (`rordereddict.py`).
            let eq_fn_const = self
                .base
                .key_repr
                .get_ll_eq_function(rtyper)?
                .map(|eq_fn| sub_helper_funcptr_constant(rtyper, &eq_fn))
                .transpose()?;
            (hash_fn_const, eq_fn_const, false, false, false)
        };

        // `@jit.oopspec('ordereddict.lookup(...)')` is stamped on orig; `mint_look_inside_iff` moves it onto the trampoline (`rlib/jit.py look_inside_iff`).
        let lookup_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let key_lltype = key_lltype.clone();
            let eq_fn_const = eq_fn_const.clone();
            let hash_fn_for_lookup = hash_fn_const.clone();
            mint_look_inside_iff(
                rtyper,
                "ll_dict_lookup",
                &["d", "key", "hash", "store_flag", "T"],
                vec![
                    dict_ptr.clone(),
                    key_lltype.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                    LowLevelType::Void,
                ],
                LowLevelType::Signed,
                move |name| {
                    build_ll_dict_lookup_helper_graph(
                        name,
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        LowLevelType::Unsigned,
                        key_lltype.clone(),
                        eq_fn_const.clone(),
                        eq_takes_dict,
                        paranoia,
                        Some(hash_fn_for_lookup.clone()),
                        hash_takes_dict,
                    )
                },
                &[LookInsidePred::IsVirtual(0), LookInsidePred::IsConstant(1)],
            )?
        };
        let lookup_fn_const = sub_helper_funcptr_constant(rtyper, &lookup_fn)?;

        let malloc_choose_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_malloc_indexes_and_choose_lookup",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_malloc_indexes_and_choose_lookup_helper_graph(
                        "ll_malloc_indexes_and_choose_lookup",
                        dict_ptr.clone(),
                    )
                },
            )?
        };
        let malloc_choose_const = sub_helper_funcptr_constant(rtyper, &malloc_choose_fn)?;

        // `ll_dict_rehash_after_translation` (`rordereddict.py`) needs
        // `ll_dict_reindex` → `ll_dict_store_clean` → `_ll_write_indexes`.
        // Minted here so `ll_dict_create_initial_index` can direct_call
        // the nonempty arm; later setitem minting hits the same
        // (name, args, result) cache.
        let write_indexes_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "_ll_write_indexes",
                vec![dict_ptr.clone(), LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_write_indexes_helper_graph(
                        "_ll_write_indexes",
                        dict_ptr.clone(),
                        LowLevelType::Unsigned,
                    )
                },
            )?
        };
        let write_indexes_const = sub_helper_funcptr_constant(rtyper, &write_indexes_fn)?;
        let store_clean_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_store_clean",
                vec![dict_ptr.clone(), LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_store_clean_helper_graph(
                        "ll_dict_store_clean",
                        dict_ptr.clone(),
                        write_indexes_const.clone(),
                    )
                },
            )?
        };
        let store_clean_const = sub_helper_funcptr_constant(rtyper, &store_clean_fn)?;
        let clear_indexes_const = mint_ll_clear_indexes(rtyper, &dict_ptr)?;
        let reindex_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let malloc_choose_const = malloc_choose_const.clone();
            let clear_indexes_const = clear_indexes_const.clone();
            let store_clean_const = store_clean_const.clone();
            let keyhash_fn = hash_fn_const.clone();
            let hash_takes_dict = hash_takes_dict;
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_reindex",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_reindex_helper_graph(
                        "ll_dict_reindex",
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        malloc_choose_const.clone(),
                        clear_indexes_const.clone(),
                        store_clean_const.clone(),
                        Some(keyhash_fn.clone()),
                        hash_takes_dict,
                    )
                },
            )?
        };
        let reindex_const = sub_helper_funcptr_constant(rtyper, &reindex_fn)?;
        let rehash_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let reindex_const = reindex_const.clone();
            let keyhash_fn = hash_fn_const.clone();
            let hash_takes_dict = hash_takes_dict;
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_rehash_after_translation",
                vec![dict_ptr.clone()],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_rehash_after_translation_helper_graph(
                        "ll_dict_rehash_after_translation",
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        reindex_const.clone(),
                        keyhash_fn.clone(),
                        hash_takes_dict,
                    )
                },
            )?
        };
        let rehash_const = sub_helper_funcptr_constant(rtyper, &rehash_fn)?;

        let create_initial_index_fn = mint_ll_dict_create_initial_index_look_inside_iff(
            rtyper,
            dict_ptr.clone(),
            malloc_choose_const.clone(),
            rehash_const.clone(),
        )?;
        let create_initial_index_const =
            sub_helper_funcptr_constant(rtyper, &create_initial_index_fn)?;

        let ensure_indexes_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_ensure_indexes",
                vec![dict_ptr.clone()],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_ensure_indexes_helper_graph(
                        "ll_ensure_indexes",
                        dict_ptr.clone(),
                        create_initial_index_const.clone(),
                    )
                },
            )?
        };
        let ensure_indexes_const = sub_helper_funcptr_constant(rtyper, &ensure_indexes_fn)?;

        let call_lookup_fn = {
            let dict_ptr = dict_ptr.clone();
            let key_lltype = key_lltype.clone();
            mint_look_inside_iff(
                rtyper,
                "ll_call_lookup_function",
                &["d", "key", "hash", "flag"],
                vec![
                    dict_ptr.clone(),
                    key_lltype.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Signed,
                move |name| {
                    build_ll_call_lookup_function_helper_graph(
                        name,
                        dict_ptr.clone(),
                        key_lltype.clone(),
                        ensure_indexes_const.clone(),
                        lookup_fn_const.clone(),
                    )
                },
                &[LookInsidePred::IsVirtual(0)],
            )?
        };
        let call_lookup_const = sub_helper_funcptr_constant(rtyper, &call_lookup_fn)?;

        Ok(LookupChainHelpers {
            hash_fn_const,
            call_lookup_const,
            hash_takes_dict,
        })
    }

    /// Mint the shared setitem insert/overwrite tail:
    /// `_ll_dict_setitem_lookup_done` -> `ll_dict_grow` / `ll_dict_resize` ->
    /// `ll_dict_reindex` -> `ll_call_insert_clean_function` ->
    /// `ll_dict_store_clean` -> `_ll_write_indexes`. `ll_dict_setitem` and
    /// `ll_dict_setdefault` both reuse this tail after their lookup.
    fn ll_dict_setitem_lookup_done_helper(
        &self,
        hop: &HighLevelOp,
    ) -> Result<LowLevelFunction, TyperError> {
        self.ll_dict_setitem_lookup_done_helper_on(&hop.rtyper)
    }

    fn ll_dict_setitem_lookup_done_helper_on(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<LowLevelFunction, TyperError> {
        let chain = self.lookup_chain_helpers_on(rtyper)?;
        let dict_ptr = self.lowleveltype.clone();
        let entries_ptr = self.entries_ptr_lltype();
        let key_lltype = self.DICTKEY.clone();
        let value_lltype = self.DICTVALUE.clone();
        let entries_array_ty = LowLevelType::Array(Box::new(self.DICTENTRYARRAY.clone()));

        // _ll_write_indexes(d, i, value) (rordereddict.py) — the real
        // helper exists for `build_ll_dict_lookup_helper_graph`'s
        // inlined store path; mint it here as a callable funcptr for the
        // non-inlined callers (`ll_dict_store_clean`).
        let write_indexes_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "_ll_write_indexes",
                vec![dict_ptr.clone(), LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_write_indexes_helper_graph(
                        "_ll_write_indexes",
                        dict_ptr.clone(),
                        LowLevelType::Unsigned,
                    )
                },
            )?
        };
        let write_indexes_const = sub_helper_funcptr_constant(rtyper, &write_indexes_fn)?;

        // ll_dict_store_clean(d, hash, index, T) (rordereddict.py).
        let store_clean_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_store_clean",
                vec![dict_ptr.clone(), LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_store_clean_helper_graph(
                        "ll_dict_store_clean",
                        dict_ptr.clone(),
                        write_indexes_const.clone(),
                    )
                },
            )?
        };
        let store_clean_const = sub_helper_funcptr_constant(rtyper, &store_clean_fn)?;

        // ll_call_insert_clean_function(d, hash, i) (rordereddict.py)
        // — FUNC_* dispatch collapses to a single `ll_dict_store_clean` call,
        // same shape as `ll_call_lookup_function`.
        let insert_clean_fn = {
            let dict_ptr = dict_ptr.clone();
            let store_clean_const = store_clean_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_call_insert_clean_function",
                vec![dict_ptr.clone(), LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_call_insert_clean_function_helper_graph(
                        "ll_call_insert_clean_function",
                        dict_ptr.clone(),
                        store_clean_const.clone(),
                    )
                },
            )?
        };
        let insert_clean_const = sub_helper_funcptr_constant(rtyper, &insert_clean_fn)?;

        // ll_malloc_indexes_and_choose_lookup — same (name, args, result) key
        // as `lookup_chain_helpers`'s own mint, so this hits the memoized
        // cache rather than rebuilding a second copy of the graph.
        let malloc_choose_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_malloc_indexes_and_choose_lookup",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_malloc_indexes_and_choose_lookup_helper_graph(
                        "ll_malloc_indexes_and_choose_lookup",
                        dict_ptr.clone(),
                    )
                },
            )?
        };
        let malloc_choose_const = sub_helper_funcptr_constant(rtyper, &malloc_choose_fn)?;

        // ll_dict_reindex(d, new_size) (rordereddict.py).
        let clear_indexes_const = mint_ll_clear_indexes(rtyper, &dict_ptr)?;
        let reindex_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let malloc_choose_const = malloc_choose_const.clone();
            let clear_indexes_const = clear_indexes_const.clone();
            let store_clean_const = store_clean_const.clone();
            let keyhash_fn = chain.hash_fn_const.clone();
            let hash_takes_dict = chain.hash_takes_dict;
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_reindex",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_reindex_helper_graph(
                        "ll_dict_reindex",
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        malloc_choose_const.clone(),
                        clear_indexes_const.clone(),
                        store_clean_const.clone(),
                        Some(keyhash_fn.clone()),
                        hash_takes_dict,
                    )
                },
            )?
        };
        let reindex_const = sub_helper_funcptr_constant(rtyper, &reindex_fn)?;

        // ll_dict_remove_deleted_items(d) (rordereddict.py).
        let remove_deleted_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let entries_array_ty = entries_array_ty.clone();
            let key_lltype = key_lltype.clone();
            let value_lltype = value_lltype.clone();
            let reindex_const = reindex_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_remove_deleted_items",
                vec![dict_ptr.clone()],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_remove_deleted_items_helper_graph(
                        "ll_dict_remove_deleted_items",
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        entries_array_ty.clone(),
                        key_lltype.clone(),
                        value_lltype.clone(),
                        reindex_const.clone(),
                    )
                },
            )?
        };
        let remove_deleted_const = sub_helper_funcptr_constant(rtyper, &remove_deleted_fn)?;

        // _ll_dict_resize_to(d, num_extra) (rordereddict.py).
        let resize_to_fn = {
            let dict_ptr = dict_ptr.clone();
            let reindex_const = reindex_const.clone();
            let remove_deleted_const = remove_deleted_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "_ll_dict_resize_to",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_resize_to_helper_graph(
                        "_ll_dict_resize_to",
                        dict_ptr.clone(),
                        remove_deleted_const.clone(),
                        reindex_const.clone(),
                    )
                },
            )?
        };
        let resize_to_const = sub_helper_funcptr_constant(rtyper, &resize_to_fn)?;

        // ll_dict_resize(d) (rordereddict.py).
        let resize_fn = {
            let dict_ptr = dict_ptr.clone();
            let resize_to_const = resize_to_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_resize",
                vec![dict_ptr.clone()],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_resize_helper_graph(
                        "ll_dict_resize",
                        dict_ptr.clone(),
                        resize_to_const.clone(),
                    )
                },
            )?
        };
        let resize_const = sub_helper_funcptr_constant(rtyper, &resize_fn)?;

        // Struct-array `rgc.ll_arraycopy` specialisation for `odictentry` —
        // rlist's generic scalar `ll_arraycopy` moves bare `Ptr(GcArray(ITEM))`
        // elements; a struct-element array needs a field-by-field copy
        // instead, so this is a fresh (documented) loop, not a reuse of
        // rlist's helper.
        let arraycopy_fn = {
            let entries_ptr = entries_ptr.clone();
            let key_lltype = key_lltype.clone();
            let value_lltype = value_lltype.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_entries_arraycopy",
                vec![
                    entries_ptr.clone(),
                    entries_ptr.clone(),
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_entries_arraycopy_helper_graph(
                        "ll_dict_entries_arraycopy",
                        entries_ptr.clone(),
                        key_lltype.clone(),
                        value_lltype.clone(),
                    )
                },
            )?
        };
        let arraycopy_const = sub_helper_funcptr_constant(rtyper, &arraycopy_fn)?;

        // _ll_dict_entries_size_too_big(d, new_allocated) -> Bool (rordereddict.py).
        let size_too_big_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "_ll_dict_entries_size_too_big",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Bool,
                move |_rtyper, _args, _result| {
                    build__ll_dict_entries_size_too_big_helper_graph(
                        "_ll_dict_entries_size_too_big",
                        dict_ptr.clone(),
                    )
                },
            )?
        };
        let size_too_big_const = sub_helper_funcptr_constant(rtyper, &size_too_big_fn)?;

        // ll_dict_grow(d) -> Bool (rordereddict.py).
        let grow_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let entries_array_ty = entries_array_ty.clone();
            let arraycopy_const = arraycopy_const.clone();
            let remove_deleted_const = remove_deleted_const.clone();
            let size_too_big_const = size_too_big_const.clone();
            mint_look_inside_iff(
                rtyper,
                "ll_dict_grow",
                &["d"],
                vec![dict_ptr.clone()],
                LowLevelType::Bool,
                move |name| {
                    build_ll_dict_grow_helper_graph(
                        name,
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        entries_array_ty.clone(),
                        remove_deleted_const.clone(),
                        size_too_big_const.clone(),
                        arraycopy_const.clone(),
                    )
                },
                &[LookInsidePred::IsVirtual(0)],
            )?
        };
        let grow_const = sub_helper_funcptr_constant(rtyper, &grow_fn)?;

        // `_ll_dict_rescue(d)` (`rordereddict.py`): reindex in place
        // after a malloc failure left `indexes` inconsistent.
        let rescue_fn = {
            let dict_ptr = dict_ptr.clone();
            let reindex_const = reindex_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "_ll_dict_rescue",
                vec![dict_ptr.clone()],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_rescue_helper_graph(
                        "_ll_dict_rescue",
                        dict_ptr.clone(),
                        reindex_const.clone(),
                    )
                },
            )?
        };
        let rescue_const = sub_helper_funcptr_constant(rtyper, &rescue_fn)?;

        // _ll_dict_setitem_lookup_done(d, key, value, hash, i)
        // (rordereddict.py).
        let lookup_done_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let key_lltype = key_lltype.clone();
            let value_lltype = value_lltype.clone();
            mint_look_inside_iff(
                rtyper,
                "_ll_dict_setitem_lookup_done",
                &["d", "key", "value", "hash", "i"],
                vec![
                    dict_ptr.clone(),
                    key_lltype.clone(),
                    value_lltype.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |name| {
                    build_ll_dict_setitem_lookup_done_helper_graph(
                        name,
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        key_lltype.clone(),
                        value_lltype.clone(),
                        grow_const.clone(),
                        resize_const.clone(),
                        insert_clean_const.clone(),
                        rescue_const.clone(),
                    )
                },
                &[LookInsidePred::IsVirtual(0), LookInsidePred::IsConstant(1)],
            )?
        };
        Ok(lookup_done_fn)
    }

    /// Mint `ll_dict_setitem` (fused with `ll_dict_setitem_with_hash`) around
    /// the shared `_ll_dict_setitem_lookup_done` tail.
    fn ll_dict_setitem_helper(&self, hop: &HighLevelOp) -> Result<LowLevelFunction, TyperError> {
        self.ll_dict_setitem_helper_on(&hop.rtyper)
    }

    /// `ll_dict_setitem` specialized for this DICT, callable without a hop
    /// (`rffi.py` `raw_copies.__setitem__` is this helper).
    pub(crate) fn ll_dict_setitem_helper_on(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<LowLevelFunction, TyperError> {
        let chain = self.lookup_chain_helpers_on(rtyper)?;
        let dict_ptr = self.lowleveltype.clone();
        let key_lltype = self.DICTKEY.clone();
        let value_lltype = self.DICTVALUE.clone();
        let lookup_done_fn = self.ll_dict_setitem_lookup_done_helper_on(rtyper)?;
        let lookup_done_const = sub_helper_funcptr_constant(rtyper, &lookup_done_fn)?;
        // ll_dict_setitem(d, key, value) (rordereddict.py, fused
        // with its `_with_hash` half and the FLAG_STORE lookup call).
        rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_setitem",
            vec![dict_ptr.clone(), key_lltype.clone(), value_lltype.clone()],
            LowLevelType::Void,
            move |_rtyper, _args, _result| {
                build_ll_dict_setitem_helper_graph(
                    "ll_dict_setitem",
                    dict_ptr.clone(),
                    key_lltype.clone(),
                    value_lltype.clone(),
                    chain.hash_fn_const.clone(),
                    chain.call_lookup_const.clone(),
                    lookup_done_const.clone(),
                    chain.hash_takes_dict,
                )
            },
        )
    }

    /// `ll_dict_getitem` specialized for this DICT, callable without a hop
    /// (`rffi.py` `raw_copies.__getitem__` is this helper).
    pub(crate) fn ll_dict_getitem_helper_on(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<LowLevelFunction, TyperError> {
        let chain = self.lookup_chain_helpers_on(rtyper)?;
        let dict_ptr = self.lowleveltype.clone();
        let entries_ptr = self.entries_ptr_lltype();
        let key_lltype = self.DICTKEY.clone();
        let value_lltype = self.DICTVALUE.clone();
        rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_getitem",
            vec![dict_ptr.clone(), key_lltype.clone()],
            value_lltype.clone(),
            move |_rtyper, _args, _result| {
                build_ll_dict_getitem_helper_graph(
                    "ll_dict_getitem",
                    dict_ptr.clone(),
                    entries_ptr.clone(),
                    key_lltype.clone(),
                    value_lltype.clone(),
                    chain.hash_fn_const.clone(),
                    chain.call_lookup_const.clone(),
                    chain.hash_takes_dict,
                )
            },
        )
    }

    /// Mint `ll_dict_insert`: `ll_dict_lookup(FLAG_STORE)` then
    /// `_ll_dict_setitem_lookup_done`, returning the displaced value.
    ///
    /// The same pieces `ll_dict_setitem_with_hash` (`rordereddict.py`)
    /// composes, plus the value-or-None read `ll_dict_get` produces.
    fn ll_dict_insert_helper(&self, hop: &HighLevelOp) -> Result<LowLevelFunction, TyperError> {
        let chain = self.lookup_chain_helpers(hop)?;
        let rtyper = &hop.rtyper;
        let dict_ptr = self.lowleveltype.clone();
        let entries_ptr = self.entries_ptr_lltype();
        let key_lltype = self.DICTKEY.clone();
        let value_lltype = self.DICTVALUE.clone();
        let lookup_done_fn = self.ll_dict_setitem_lookup_done_helper(hop)?;
        let lookup_done_const = sub_helper_funcptr_constant(rtyper, &lookup_done_fn)?;
        rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_insert",
            vec![dict_ptr.clone(), key_lltype.clone(), value_lltype.clone()],
            value_lltype.clone(),
            move |_rtyper, _args, _result| {
                build_ll_dict_insert_helper_graph(
                    "ll_dict_insert",
                    dict_ptr.clone(),
                    entries_ptr.clone(),
                    key_lltype.clone(),
                    value_lltype.clone(),
                    chain.hash_fn_const.clone(),
                    chain.call_lookup_const.clone(),
                    lookup_done_const.clone(),
                    chain.hash_takes_dict,
                )
            },
        )
    }

    /// Mint `ll_dict_delitem` (fused with `ll_dict_delitem_with_hash`) and its
    /// deletion tail: `ll_call_delete_by_entry_index` ->
    /// `ll_dict_delete_by_entry_index` -> `_ll_dict_del_entry`, with
    /// `ll_dict_resize` available for the 87.5%-dead shrink rule.
    fn ll_dict_delitem_helper(&self, hop: &HighLevelOp) -> Result<LowLevelFunction, TyperError> {
        let chain = self.lookup_chain_helpers(hop)?;
        let rtyper = &hop.rtyper;
        let dict_ptr = self.lowleveltype.clone();
        let entries_ptr = self.entries_ptr_lltype();
        let key_lltype = self.DICTKEY.clone();
        let value_lltype = self.DICTVALUE.clone();
        let entries_array_ty = LowLevelType::Array(Box::new(self.DICTENTRYARRAY.clone()));

        // _ll_write_indexes(d, i, value) (rordereddict.py).
        let write_indexes_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "_ll_write_indexes",
                vec![dict_ptr.clone(), LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_write_indexes_helper_graph(
                        "_ll_write_indexes",
                        dict_ptr.clone(),
                        LowLevelType::Unsigned,
                    )
                },
            )?
        };
        let write_indexes_const = sub_helper_funcptr_constant(rtyper, &write_indexes_fn)?;

        // ll_dict_delete_by_entry_index(d, hash, locate_index, replace_with, T)
        // (rordereddict.py).
        let delete_by_entry_fn = {
            let dict_ptr = dict_ptr.clone();
            let write_indexes_const = write_indexes_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_delete_by_entry_index",
                vec![
                    dict_ptr.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_delete_by_entry_index_helper_graph(
                        "ll_dict_delete_by_entry_index",
                        dict_ptr.clone(),
                        write_indexes_const.clone(),
                    )
                },
            )?
        };
        let delete_by_entry_const = sub_helper_funcptr_constant(rtyper, &delete_by_entry_fn)?;

        // ll_call_delete_by_entry_index(d, hash, i, replace_with)
        // (rordereddict.py) — FUNC_* dispatch collapses to a single
        // delete helper call for the same width-collapse reason as
        // ll_call_insert_clean_function.
        let call_delete_fn = {
            let dict_ptr = dict_ptr.clone();
            let delete_by_entry_const = delete_by_entry_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_call_delete_by_entry_index",
                vec![
                    dict_ptr.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_call_delete_by_entry_index_helper_graph(
                        "ll_call_delete_by_entry_index",
                        dict_ptr.clone(),
                        delete_by_entry_const.clone(),
                    )
                },
            )?
        };
        let call_delete_const = sub_helper_funcptr_constant(rtyper, &call_delete_fn)?;

        // ll_dict_store_clean(d, hash, index, T) (rordereddict.py),
        // used by ll_dict_reindex below.
        let store_clean_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_store_clean",
                vec![dict_ptr.clone(), LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_store_clean_helper_graph(
                        "ll_dict_store_clean",
                        dict_ptr.clone(),
                        write_indexes_const.clone(),
                    )
                },
            )?
        };
        let store_clean_const = sub_helper_funcptr_constant(rtyper, &store_clean_fn)?;

        let malloc_choose_fn = {
            let dict_ptr = dict_ptr.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_malloc_indexes_and_choose_lookup",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_malloc_indexes_and_choose_lookup_helper_graph(
                        "ll_malloc_indexes_and_choose_lookup",
                        dict_ptr.clone(),
                    )
                },
            )?
        };
        let malloc_choose_const = sub_helper_funcptr_constant(rtyper, &malloc_choose_fn)?;

        let clear_indexes_const = mint_ll_clear_indexes(rtyper, &dict_ptr)?;
        let reindex_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let malloc_choose_const = malloc_choose_const.clone();
            let clear_indexes_const = clear_indexes_const.clone();
            let store_clean_const = store_clean_const.clone();
            let keyhash_fn = chain.hash_fn_const.clone();
            let hash_takes_dict = chain.hash_takes_dict;
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_reindex",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_reindex_helper_graph(
                        "ll_dict_reindex",
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        malloc_choose_const.clone(),
                        clear_indexes_const.clone(),
                        store_clean_const.clone(),
                        Some(keyhash_fn.clone()),
                        hash_takes_dict,
                    )
                },
            )?
        };
        let reindex_const = sub_helper_funcptr_constant(rtyper, &reindex_fn)?;

        let remove_deleted_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let entries_array_ty = entries_array_ty.clone();
            let key_lltype = key_lltype.clone();
            let value_lltype = value_lltype.clone();
            let reindex_const = reindex_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_remove_deleted_items",
                vec![dict_ptr.clone()],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_remove_deleted_items_helper_graph(
                        "ll_dict_remove_deleted_items",
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        entries_array_ty.clone(),
                        key_lltype.clone(),
                        value_lltype.clone(),
                        reindex_const.clone(),
                    )
                },
            )?
        };
        let remove_deleted_const = sub_helper_funcptr_constant(rtyper, &remove_deleted_fn)?;

        let resize_to_fn = {
            let dict_ptr = dict_ptr.clone();
            let remove_deleted_const = remove_deleted_const.clone();
            let reindex_const = reindex_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "_ll_dict_resize_to",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_resize_to_helper_graph(
                        "_ll_dict_resize_to",
                        dict_ptr.clone(),
                        remove_deleted_const.clone(),
                        reindex_const.clone(),
                    )
                },
            )?
        };
        let resize_to_const = sub_helper_funcptr_constant(rtyper, &resize_to_fn)?;

        let resize_fn = {
            let dict_ptr = dict_ptr.clone();
            let resize_to_const = resize_to_const.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "ll_dict_resize",
                vec![dict_ptr.clone()],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_resize_helper_graph(
                        "ll_dict_resize",
                        dict_ptr.clone(),
                        resize_to_const.clone(),
                    )
                },
            )?
        };
        let resize_const = sub_helper_funcptr_constant(rtyper, &resize_fn)?;

        let del_entry_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let key_lltype = key_lltype.clone();
            let value_lltype = value_lltype.clone();
            rtyper.lowlevel_helper_function_with_builder(
                "_ll_dict_del_entry",
                vec![dict_ptr.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_dict_del_entry_helper_graph(
                        "_ll_dict_del_entry",
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        key_lltype.clone(),
                        value_lltype.clone(),
                    )
                },
            )?
        };
        let del_entry_const = sub_helper_funcptr_constant(rtyper, &del_entry_fn)?;

        let del_fn = {
            let dict_ptr = dict_ptr.clone();
            let entries_ptr = entries_ptr.clone();
            let call_delete_const = call_delete_const.clone();
            let del_entry_const = del_entry_const.clone();
            let resize_const = resize_const.clone();
            mint_look_inside_iff(
                rtyper,
                "_ll_dict_del",
                &["d", "hash", "index"],
                vec![dict_ptr.clone(), LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Void,
                move |name| {
                    build_ll_dict_del_helper_graph(
                        name,
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                        call_delete_const.clone(),
                        del_entry_const.clone(),
                        resize_const.clone(),
                    )
                },
                &[LookInsidePred::IsVirtual(0), LookInsidePred::IsConstant(2)],
            )?
        };
        let del_const = sub_helper_funcptr_constant(rtyper, &del_fn)?;

        rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_delitem",
            vec![dict_ptr.clone(), key_lltype.clone()],
            LowLevelType::Void,
            move |_rtyper, _args, _result| {
                build_ll_dict_delitem_helper_graph(
                    "ll_dict_delitem",
                    dict_ptr.clone(),
                    key_lltype.clone(),
                    chain.hash_fn_const.clone(),
                    chain.call_lookup_const.clone(),
                    del_const.clone(),
                    chain.hash_takes_dict,
                )
            },
        )
    }

    fn ll_dict_get_helper(&self, hop: &HighLevelOp) -> Result<LowLevelFunction, TyperError> {
        let chain = self.lookup_chain_helpers(hop)?;
        let rtyper = &hop.rtyper;
        let dict_ptr = self.lowleveltype.clone();
        let entries_ptr = self.entries_ptr_lltype();
        let key_lltype = self.DICTKEY.clone();
        let value_lltype = self.DICTVALUE.clone();
        rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_get",
            vec![dict_ptr.clone(), key_lltype.clone(), value_lltype.clone()],
            value_lltype.clone(),
            move |_rtyper, _args, _result| {
                build_ll_dict_get_helper_graph(
                    "ll_dict_get",
                    dict_ptr.clone(),
                    entries_ptr.clone(),
                    key_lltype.clone(),
                    value_lltype.clone(),
                    chain.hash_fn_const.clone(),
                    chain.call_lookup_const.clone(),
                    chain.hash_takes_dict,
                )
            },
        )
    }

    fn ll_dict_setdefault_helper(&self, hop: &HighLevelOp) -> Result<LowLevelFunction, TyperError> {
        let chain = self.lookup_chain_helpers(hop)?;
        let rtyper = &hop.rtyper;
        let dict_ptr = self.lowleveltype.clone();
        let entries_ptr = self.entries_ptr_lltype();
        let key_lltype = self.DICTKEY.clone();
        let value_lltype = self.DICTVALUE.clone();
        let lookup_done_fn = self.ll_dict_setitem_lookup_done_helper(hop)?;
        let lookup_done_const = sub_helper_funcptr_constant(rtyper, &lookup_done_fn)?;
        rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_setdefault",
            vec![dict_ptr.clone(), key_lltype.clone(), value_lltype.clone()],
            value_lltype.clone(),
            move |_rtyper, _args, _result| {
                build_ll_dict_setdefault_helper_graph(
                    "ll_dict_setdefault",
                    dict_ptr.clone(),
                    entries_ptr.clone(),
                    key_lltype.clone(),
                    value_lltype.clone(),
                    chain.hash_fn_const.clone(),
                    chain.call_lookup_const.clone(),
                    lookup_done_const.clone(),
                    chain.hash_takes_dict,
                )
            },
        )
    }
}

impl Repr for OrderedDictRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.base.state
    }

    fn class_name(&self) -> &'static str {
        "OrderedDictRepr"
    }

    fn repr_class_id(&self) -> crate::translator::rtyper::pairtype::ReprClassId {
        crate::translator::rtyper::pairtype::ReprClassId::OrderedDictRepr
    }

    fn compact_repr(&self) -> String {
        self.base.compact_repr()
    }

    /// RPython `OrderedDictRepr.convert_const(self, dictobj)`
    /// (`rordereddict.py`): `None` → `nullptr(self.DICT)`; a host `dict`
    /// or prebuilt `objectmodel.r_dict` is `ll_newdict_size` +
    /// `ll_no_initial_index` then `_ll_dict_insert_no_index` per item,
    /// cached by `Constant(dictobj)`. The r_dict arm sets `fnkeyeq` /
    /// `fnkeyhash` (when those reprs are not Void) from `key_eq` /
    /// `key_hash` and inserts via `dictkeycontainer.key`. Host `d.keyhash`
    /// / `d.keyeq` evaluate the stored funcptr by identity (`get_ll_hash_function`
    /// / `get_ll_eq_function`); custom `ll_keyhash_custom` / `ll_keyeq_custom`
    /// stay deferred because llinterp cannot run those graphs.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        match value {
            ConstValue::None => {
                let LowLevelType::Ptr(ptr_t) = &self.lowleveltype else {
                    return Err(TyperError::message(
                        "OrderedDictRepr.lowleveltype is not a Ptr",
                    ));
                };
                Ok(Constant::with_concretetype(
                    ConstValue::LLPtr(Box::new(ptr_t._defl())),
                    self.lowleveltype.clone(),
                ))
            }
            ConstValue::Dict(_) => self.convert_const_prebuilt_dict(value),
            ConstValue::HostObject(obj) if is_objectmodel_r_dict_instance(obj) => {
                self.convert_const_prebuilt_dict(value)
            }
            other => Err(TyperError::message(format!("expected a dict: {other:?}"))),
        }
    }

    /// RPython `OrderedDictRepr.rtype_len(self, hop)`
    /// (`rordereddict.py`): `hop.gendirectcall(ll_dict_len, v_dict)`.
    /// `ll_dict_len(d)` (`rordereddict.py`) returns the
    /// `num_live_items` header field.
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_dict = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lowleveltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_len".to_string(),
            vec![ptr_lltype],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_dict_len_helper_graph("ll_dict_len", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, v_dict)
    }

    /// RPython `OrderedDictRepr.rtype_bool(self, hop)`
    /// (`rordereddict.py`): `hop.gendirectcall(ll_dict_bool, v_dict)`.
    /// `ll_dict_bool(d)` (`rordereddict.py`) is `bool(d) and
    /// d.num_live_items != 0` — the explicit `bool(d)` guard lets a None-typed
    /// dict read False without dereferencing, so this overrides the
    /// `int_is_true(len)` default (`rmodel.py`) which would deref the
    /// possibly-null receiver.
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_dict = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lowleveltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_bool".to_string(),
            vec![ptr_lltype],
            LowLevelType::Bool,
            move |_rtyper, _args, _result| {
                build_ll_dict_bool_helper_graph("ll_dict_bool", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, v_dict)
    }

    /// RPython `OrderedDictRepr.make_iterator_repr(self, *variant)`
    /// (`rordereddict.py`):
    ///
    /// ```python
    /// def make_iterator_repr(self, *variant):
    ///     return DictIteratorRepr(self, *variant)
    /// ```
    ///
    /// Only the no-variant case (implicit `for k in d:`, `DictIteratorRepr`'s
    /// default `variant="keys"`) is wired. `reversed(d)`
    /// (`variant=("reversed",)`) needs the `ll_dictiter_reversed`/
    /// `_ll_dictnext_reversed` constructors (still `ordered_dict_runtime_deferred`
    /// stubs); accepting it here without swapping those in would silently walk
    /// the dict forward, so it is rejected fail-closed instead.
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
            return Err(ordered_dict_runtime_deferred(&format!(
                "make_iterator_repr variant {variant:?} (reversed dict iteration)"
            )));
        }
        Ok(Arc::new(DictIteratorRepr::new(self, "keys")))
    }

    /// RPython `pairtype(OrderedDictRepr, rmodel.Repr).rtype_getitem`
    /// (`rordereddict.py`):
    ///
    /// ```python
    /// def rtype_getitem((r_dict, r_key), hop):
    ///     v_dict, v_key = hop.inputargs(r_dict, r_dict.key_repr)
    ///     if not r_dict.custom_eq_hash:
    ///         hop.has_implicit_exception(KeyError)   # record that we know about it
    ///     hop.exception_is_here()
    ///     v_res = hop.gendirectcall(ll_dict_getitem, v_dict, v_key)
    ///     return r_dict.recast_value(hop.llops, v_res)
    /// ```
    ///
    /// The second pairtype receiver (`r_key`) is never read in the upstream
    /// body — only `r_dict.key_repr` is — so this port dispatches it as a
    /// direct `Repr::rtype_getitem` override (same shape as
    /// `ListRepr::rtype_getitem`) rather than a `pair_*` free function; the
    /// `(OrderedDictRepr, _, "getitem")` pairtype-dispatch arm forwards here
    /// for any key repr.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::Repr(self.base.key_repr.as_ref()),
        ])?;
        if !self.base.custom_eq_hash {
            hop.has_implicit_exception("KeyError");
        }
        hop.exception_is_here()?;

        let helper = self.ll_dict_getitem_helper_on(&hop.rtyper)?;
        let v_res = hop.gendirectcall(&helper, args)?.ok_or_else(|| {
            TyperError::message("rtype_getitem: ll_dict_getitem call produced no value")
        })?;
        let (value_repr, external_value_repr) = self.base.recast_value();
        let recast = hop.llops.borrow_mut().convertvar(
            v_res,
            value_repr.as_ref(),
            external_value_repr.as_ref(),
        )?;
        Ok(Some(recast))
    }

    /// RPython `pairtype(OrderedDictRepr, rmodel.Repr).rtype_setitem`
    /// (`rordereddict.py`):
    ///
    /// ```python
    /// def rtype_setitem((r_dict, r_key), hop):
    ///     v_dict, v_key, v_value = hop.inputargs(r_dict, r_dict.key_repr, r_dict.value_repr)
    ///     if r_dict.custom_eq_hash:
    ///         hop.exception_is_here()
    ///     else:
    ///         hop.exception_cannot_occur()
    ///     hop.gendirectcall(ll_dict_setitem, v_dict, v_key, v_value)
    /// ```
    ///
    /// Same wildcard-`_` dispatch rationale as [`Repr::rtype_getitem`] above
    /// (`r_key` is never read, only `r_dict.key_repr`); the
    /// `(OrderedDictRepr, _, "setitem")` pairtype arm forwards here for any
    /// key repr.
    fn rtype_setitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::Repr(self.base.key_repr.as_ref()),
            ConvertedTo::Repr(self.base.value_repr.as_ref()),
        ])?;
        if self.base.custom_eq_hash {
            hop.exception_is_here()?;
        } else {
            hop.exception_cannot_occur()?;
        }

        let helper = self.ll_dict_setitem_helper(hop)?;
        hop.gendirectcall(&helper, args)?;
        Ok(None)
    }

    /// RPython `pairtype(OrderedDictRepr, rmodel.Repr).rtype_delitem`
    /// (`rordereddict.py`):
    ///
    /// ```python
    /// def rtype_delitem((r_dict, r_key), hop):
    ///     v_dict, v_key = hop.inputargs(r_dict, r_dict.key_repr)
    ///     if not r_dict.custom_eq_hash:
    ///         hop.has_implicit_exception(KeyError)
    ///     hop.exception_is_here()
    ///     hop.gendirectcall(ll_dict_delitem, v_dict, v_key)
    /// ```
    fn rtype_delitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::Repr(self.base.key_repr.as_ref()),
        ])?;
        if !self.base.custom_eq_hash {
            hop.has_implicit_exception("KeyError");
        }
        hop.exception_is_here()?;

        let helper = self.ll_dict_delitem_helper(hop)?;
        hop.gendirectcall(&helper, args)?;
        Ok(None)
    }

    /// RPython `OrderedDictRepr.rtype_method_get` / `rtype_method_setdefault`
    /// (`rordereddict.py`), plus `insert` (`ll_dict_lookup(FLAG_STORE)` +
    /// `_ll_dict_setitem_lookup_done`).
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            // `is_null` is the lltype `_ptr` nullity probe on the dict
            // pointer — lower it as `ptr_iszero` (opimpl.py
            // `op_ptr_iszero`), the same lowering `InstanceRepr` uses for
            // the `ptr_method_is_null` bound method the annotator seats on
            // pointer-carrying receivers.
            "is_null" => {
                let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
                Ok(hop.genop("ptr_iszero", vlist, GenopResult::LLType(LowLevelType::Bool)))
            }
            "get" => {
                let args = if hop.nb_args() == 3 {
                    hop.inputargs(vec![
                        ConvertedTo::Repr(self),
                        ConvertedTo::Repr(self.base.key_repr.as_ref()),
                        ConvertedTo::Repr(self.base.value_repr.as_ref()),
                    ])?
                } else if hop.nb_args() == 2 {
                    let mut args = hop.inputargs(vec![
                        ConvertedTo::Repr(self),
                        ConvertedTo::Repr(self.base.key_repr.as_ref()),
                    ])?;
                    let default = HighLevelOp::inputconst(
                        ConvertedTo::Repr(self.base.value_repr.as_ref()),
                        &ConstValue::None,
                    )?;
                    args.push(Hlvalue::Constant(default));
                    args
                } else {
                    return Err(TyperError::message(format!(
                        "OrderedDictRepr.rtype_method_get expects 2 or 3 args, got {}",
                        hop.nb_args()
                    )));
                };
                hop.exception_cannot_occur()?;
                let helper = self.ll_dict_get_helper(hop)?;
                let v_res = hop.gendirectcall(&helper, args)?.ok_or_else(|| {
                    TyperError::message("rtype_method_get: ll_dict_get call produced no value")
                })?;
                let (value_repr, external_value_repr) = self.base.recast_value();
                let recast = hop.llops.borrow_mut().convertvar(
                    v_res,
                    value_repr.as_ref(),
                    external_value_repr.as_ref(),
                )?;
                Ok(Some(recast))
            }
            "setdefault" => {
                let args = hop.inputargs(vec![
                    ConvertedTo::Repr(self),
                    ConvertedTo::Repr(self.base.key_repr.as_ref()),
                    ConvertedTo::Repr(self.base.value_repr.as_ref()),
                ])?;
                hop.exception_cannot_occur()?;
                let helper = self.ll_dict_setdefault_helper(hop)?;
                let v_res = hop.gendirectcall(&helper, args)?.ok_or_else(|| {
                    TyperError::message(
                        "rtype_method_setdefault: ll_dict_setdefault call produced no value",
                    )
                })?;
                let (value_repr, external_value_repr) = self.base.recast_value();
                let recast = hop.llops.borrow_mut().convertvar(
                    v_res,
                    value_repr.as_ref(),
                    external_value_repr.as_ref(),
                )?;
                Ok(Some(recast))
            }
            // `RDict::insert(key, value)` — `ll_dict_lookup(FLAG_STORE)` +
            // `_ll_dict_setitem_lookup_done` (`ll_dict_setitem_with_hash`),
            // returning the displaced value. `custom_eq_hash` can raise
            // from the key comparison, same policy as `rtype_setitem`.
            "insert" => {
                let args = hop.inputargs(vec![
                    ConvertedTo::Repr(self),
                    ConvertedTo::Repr(self.base.key_repr.as_ref()),
                    ConvertedTo::Repr(self.base.value_repr.as_ref()),
                ])?;
                if self.base.custom_eq_hash {
                    hop.exception_is_here()?;
                } else {
                    hop.exception_cannot_occur()?;
                }
                let helper = self.ll_dict_insert_helper(hop)?;
                let v_res = hop.gendirectcall(&helper, args)?.ok_or_else(|| {
                    TyperError::message(
                        "rtype_method_insert: ll_dict_insert call produced no value",
                    )
                })?;
                let (value_repr, external_value_repr) = self.base.recast_value();
                let recast = hop.llops.borrow_mut().convertvar(
                    v_res,
                    value_repr.as_ref(),
                    external_value_repr.as_ref(),
                )?;
                Ok(Some(recast))
            }
            // RPython `OrderedDictRepr.rtype_method_iterkeys`/
            // `rtype_method_itervalues`/`rtype_method_iteritems`
            // (`rordereddict.py`):
            //
            // ```python
            // def rtype_method_iterkeys(self, hop):
            //     hop.exception_cannot_occur()
            //     return DictIteratorRepr(self, "keys").newiter(hop)
            // ```
            //
            // `iterkeys_with_hash`/`iteritems_with_hash` (`:354-364`) need
            // `ll_ensure_indexes`/`entry_hash`, unported — out of scope here.
            "iterkeys" => {
                hop.exception_cannot_occur()?;
                DictIteratorRepr::new(self, "keys").newiter(hop)
            }
            "itervalues" => {
                hop.exception_cannot_occur()?;
                DictIteratorRepr::new(self, "values").newiter(hop)
            }
            "iteritems" => {
                hop.exception_cannot_occur()?;
                DictIteratorRepr::new(self, "items").newiter(hop)
            }
            _ => Err(TyperError::message(format!(
                "missing OrderedDictRepr.rtype_method_{method_name}"
            ))),
        }
    }
}

/// Synthesise `ll_dict_len(d) -> Signed` (`rordereddict.py`):
///
/// ```python
/// def ll_dict_len(d):
///     return d.num_live_items
/// ```
///
/// Single-block graph: `getfield(d, "num_live_items") -> Signed`, the live
/// entry count tracked in the `dicttable` header.
pub(crate) fn build_ll_dict_len_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("d", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_len = variable_with_lltype("num_live_items", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(arg), void_field_const("num_live_items")],
        Hlvalue::Variable(v_len.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_len)],
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
        vec!["d".to_string()],
        func,
    ))
}

/// Synthesise `ll_dict_bool(d) -> Bool` (`rordereddict.py`):
///
/// ```python
/// def ll_dict_bool(d):
///     # check if a dict is True, allowing for None
///     return bool(d) and d.num_live_items != 0
/// ```
///
/// Two-block CFG plus the returnblock:
/// - **start**: `v_nz = ptr_nonzero(d)`; branch on it. True → `check_len`
///   (forwarding `d`), False → returnblock(`False`) without dereferencing.
/// - **check_len**: `getfield(d, "num_live_items")`, `int_ne(n, 0)` →
///   returnblock(result).
pub(crate) fn build_ll_dict_bool_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("d", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_zero = || constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);

    // check_len inputarg: same `d` ptr forwarded through the True branch.
    let d_for_len = variable_with_lltype("d", ptr_lltype);
    let block_check_len = Block::shared(vec![Hlvalue::Variable(d_for_len.clone())]);

    // start: ptr_nonzero(d); branch on the result.
    let v_nz = variable_with_lltype("v_nz", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(arg.clone())],
        Hlvalue::Variable(v_nz.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_nz));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(arg)],
        Some(block_check_len.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let start_false_link = Link::new(
        vec![bool_false()],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    startblock.closeblock(vec![start_true_link, start_false_link]);

    // check_len: getfield(num_live_items); int_ne(n, 0).
    let v_count = variable_with_lltype("num_live_items", LowLevelType::Signed);
    block_check_len
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(d_for_len),
                void_field_const("num_live_items"),
            ],
            Hlvalue::Variable(v_count.clone()),
        ));
    let v_result = variable_with_lltype("result", LowLevelType::Bool);
    block_check_len
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ne",
            vec![Hlvalue::Variable(v_count), signed_zero()],
            Hlvalue::Variable(v_result.clone()),
        ));
    block_check_len.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_result)],
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
        vec!["d".to_string()],
        func,
    ))
}

/// Synthesise `ll_newdict(DICT) -> Ptr(DICT)`
/// (`rordereddict.py` + `:509-518`):
///
/// ```python
/// def ll_newdict(DICT):
///     d = DICT.allocate()                # _ll_malloc_dict: lltype.malloc(DICT)
///     d.entries = _ll_empty_array(DICT)  # DICT.entries.TO.allocate(0)
///     d.num_live_items = 0
///     d.num_ever_used_items = 0
///     ll_no_initial_index(d)
///     return d
///
/// def ll_no_initial_index(d):
///     d.lookup_function_no = FUNC_MUST_REINDEX
///     d.indexes = lltype.nullptr(llmemory.GCREF.TO)
/// ```
///
/// Single-block graph, no runtime arguments — the `DICT` class argument is
/// `Void` upstream and is instead baked into the cached helper shape by
/// [`OrderedDictRepr::ll_newdict`] (`ptr_lltype`/`dict_struct`/`entryarray`
/// close over the specific dict specialization, matching how
/// [`build_ll_newlist_helper_graph`](super::super::rlist) bakes its item type
/// rather than threading a `Void` const through `gendirectcall`).
///
/// `_ll_empty_array` is a `@specialize.memo()` prebuilt zero-length array
/// upstream (`rordereddict.py`), shared across every empty dict of
/// a given specialization. This port allocates a fresh `malloc_varsize(...,
/// 0)` entries array per call instead of sharing one prebuilt instance — the
/// memo-sharing is deferred (no local prebuilt-const cache mechanism exists
/// yet for this shape); behavior is unaffected since a 0-length array is
/// immutable in practice, only the allocation is not shared.
pub(crate) fn build_ll_dict_newdict_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    dict_struct: Struct,
    entryarray: Array,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let startblock = Block::shared(vec![]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let signed_zero = || constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);
    let void_result = || variable_with_lltype("v", LowLevelType::Void);

    // d = DICT.allocate() -- `_ll_malloc_dict(DICT)` = `lltype.malloc(DICT)`.
    let d = variable_with_lltype("d", ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![
            lowlevel_type_const(LowLevelType::Struct(Box::new(dict_struct))),
            gc_flavor_const()?,
        ],
        Hlvalue::Variable(d.clone()),
    ));

    // d.entries = _ll_empty_array(DICT) -- `DICT.entries.TO.allocate(0)`
    // = `_ll_malloc_entries(ENTRIES, 0)` = `lltype.malloc(ENTRIES, 0, zero=True)`;
    // `malloc_varsize` zero-fills, matching `zero=True`.
    let entries_array_type = LowLevelType::Array(Box::new(entryarray.clone()));
    let entries_ptr_lltype = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(entryarray),
    }));
    let entries = variable_with_lltype("entries", entries_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(entries_array_type),
            gc_flavor_const()?,
            signed_zero(),
        ],
        Hlvalue::Variable(entries.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(d.clone()),
            void_field_const("entries"),
            Hlvalue::Variable(entries),
        ],
        Hlvalue::Variable(void_result()),
    ));

    // d.num_live_items = 0
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(d.clone()),
            void_field_const("num_live_items"),
            signed_zero(),
        ],
        Hlvalue::Variable(void_result()),
    ));
    // d.num_ever_used_items = 0
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(d.clone()),
            void_field_const("num_ever_used_items"),
            signed_zero(),
        ],
        Hlvalue::Variable(void_result()),
    ));

    // ll_no_initial_index(d): d.lookup_function_no = FUNC_MUST_REINDEX;
    // d.indexes = lltype.nullptr(llmemory.GCREF.TO).
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(d.clone()),
            void_field_const("lookup_function_no"),
            constant_with_lltype(ConstValue::Int(FUNC_MUST_REINDEX), LowLevelType::Signed),
        ],
        Hlvalue::Variable(void_result()),
    ));
    let null_indexes =
        Hlvalue::Constant(Constant::with_concretetype(ConstValue::None, GCREF.clone()));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(d.clone()),
            void_field_const("indexes"),
            null_indexes,
        ],
        Hlvalue::Variable(void_result()),
    ));

    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(d)],
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
    Ok(helper_pygraph_from_graph(graph, vec![], func))
}

/// RPython `ll_newdict_size(DICT, orig_length_estimate)` (`rordereddict.py`).
/// Host-side (`convert_const`): malloc the DICT struct and the entries
/// array, then `ll_malloc_indexes_and_choose_lookup(d, n)`. Prebuilt
/// containers are immortal (`llstr` / `empty` BYTEARRAY).
/// `OrderedDictRepr.convert_const` then calls `ll_no_initial_index`.
pub fn ll_newdict_size(DICT: &Struct, orig_length_estimate: usize) -> Result<_ptr, TyperError> {
    let length_estimate = (orig_length_estimate / 2) * 3;
    let mut n = DICT_INITSIZE as usize;
    while n < length_estimate {
        n *= 2;
    }
    let mut d = malloc(
        LowLevelType::Struct(Box::new(DICT.clone())),
        None,
        MallocFlavor::Gc,
        true,
    )
    .map_err(TyperError::message)?;
    let entries_ty = DICT
        .getattr_field_type("entries")
        .ok_or_else(|| TyperError::message("ll_newdict_size: DICT has no entries field"))?;
    let LowLevelType::Ptr(entries_ptr_t) = entries_ty else {
        return Err(TyperError::message(
            "ll_newdict_size: DICT.entries is not a Ptr",
        ));
    };
    let PtrTarget::Array(array_t) = entries_ptr_t.TO else {
        return Err(TyperError::message(
            "ll_newdict_size: DICT.entries.TO is not an Array",
        ));
    };
    let entries = malloc(
        LowLevelType::Array(Box::new(array_t)),
        Some(orig_length_estimate),
        MallocFlavor::Gc,
        true,
    )
    .map_err(TyperError::message)?;
    d.setattr("entries", LowLevelValue::Ptr(Box::new(entries)))
        .map_err(TyperError::message)?;
    ll_malloc_indexes_and_choose_lookup(&mut d, n)?;
    set_dict_signed_field(&mut d, "num_live_items", 0)?;
    set_dict_signed_field(&mut d, "num_ever_used_items", 0)?;
    set_dict_signed_field(&mut d, "resize_counter", (n as i64) * 2)?;
    Ok(d)
}

/// RPython `ll_malloc_indexes_and_choose_lookup(d, n)` (`rordereddict.py`):
/// allocate a zeroed index array of length `n` and stamp
/// `lookup_function_no` from the size cascade (`FUNC_BYTE` / `FUNC_SHORT`
/// / `FUNC_INT` / `FUNC_LONG`).
pub fn ll_malloc_indexes_and_choose_lookup(d: &mut _ptr, n: usize) -> Result<(), TyperError> {
    let func_no = if n <= 256 {
        FUNC_BYTE
    } else if n <= 65536 {
        FUNC_SHORT
    } else if IS_64BIT && n <= (1usize << 32) {
        FUNC_INT
    } else {
        FUNC_LONG
    };
    let LowLevelType::Ptr(index_ptr_t) = &*DICTINDEX_BYTE else {
        return Err(TyperError::message("DICTINDEX_BYTE is not a Ptr"));
    };
    let indexes = malloc(
        LowLevelType::from(index_ptr_t.TO.clone()),
        Some(n),
        MallocFlavor::Gc,
        true,
    )
    .map_err(TyperError::message)?;
    let LowLevelType::Ptr(gcref_ptr) = &*GCREF else {
        return Err(TyperError::message("GCREF is not a Ptr"));
    };
    let gcref = cast_opaque_ptr(gcref_ptr, &indexes).map_err(TyperError::message)?;
    d.setattr("indexes", LowLevelValue::Ptr(Box::new(gcref)))
        .map_err(TyperError::message)?;
    set_dict_signed_field(d, "lookup_function_no", func_no)?;
    Ok(())
}

fn ordered_dict_runtime_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.rordereddict.{name} — ordered hash table runtime deferred"
    ))
}

pub fn get_ll_dict() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("get_ll_dict"))
}

/// RPython `ll_no_initial_index(d)` (`rordereddict.py`):
/// `ll_assert(d.num_live_items == d.num_ever_used_items, ...)`,
/// `lookup_function_no = FUNC_MUST_REINDEX`, `indexes = nullptr(GCREF.TO)`.
pub fn ll_no_initial_index(d: &mut _ptr) -> Result<(), TyperError> {
    let live = dict_signed_field(d, "num_live_items")?;
    let ever_used = dict_signed_field(d, "num_ever_used_items")?;
    if live != ever_used {
        return Err(TyperError::message(
            "ll_no_initial_index(): dict already in use",
        ));
    }
    d.setattr(
        "lookup_function_no",
        LowLevelValue::Signed(FUNC_MUST_REINDEX),
    )
    .map_err(TyperError::message)?;
    let LowLevelType::Ptr(gcref_ptr) = &*GCREF else {
        return Err(TyperError::message("GCREF is not a Ptr"));
    };
    d.setattr("indexes", LowLevelValue::Ptr(Box::new(gcref_ptr._defl())))
        .map_err(TyperError::message)?;
    Ok(())
}

/// RPython `ll_clear_indexes(d, n)` (`rordereddict.py`):
/// `lookup_function_no &= FUNC_MASK`, then zero the index array.
pub fn ll_clear_indexes(d: &mut _ptr, _n: usize) -> Result<(), TyperError> {
    let fun = dict_signed_field(d, "lookup_function_no")? & FUNC_MASK;
    set_dict_signed_field(d, "lookup_function_no", fun)?;
    let mut indexes = dict_indexes_array(d)?;
    // `rgc.ll_arrayclear`: the whole array, `n` is unused upstream too.
    for i in 0..indexes.len().map_err(TyperError::message)? {
        indexes
            .setitem(i, LowLevelValue::Unsigned(0))
            .map_err(TyperError::message)?;
    }
    Ok(())
}

/// Synthesise `_ll_write_indexes(d, i, value, T)` (`rordereddict.py`):
///
/// ```python
/// def _ll_write_indexes(d, i, value, T):
///     INDEXES = _ll_ptr_to_array_of(T)
///     indexes = lltype.cast_opaque_ptr(INDEXES, d.indexes)
///     cast_value = rffi.cast(T, value)
///     ll_assert(intmask(cast_value) == value, "...")   # debug-only, omitted
///     indexes[i] = cast_value
/// ```
///
/// Single-block graph storing `value` into the sparse index array at slot `i`.
/// GCREF -> INDEXES is `cast_opaque_ptr` (`_ll_write_indexes`; `lltype.castable`
/// rejects `cast_pointer` across an opaque type); `rffi.cast(T, value)` narrows
/// the Signed slot value to the unsigned index element type via
/// `cast_int_to_uint`. All `DICTINDEX_*` widths collapse to
/// `Ptr(GcArray(Unsigned))` here, so this one impl serves every FUNC_* width.
/// The `ll_assert` is a debug-only range check with no runtime effect after
/// translation and is not modelled.
pub fn build_ll_write_indexes_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    index_elem_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let indexes_ptr_lltype = _ll_ptr_to_array_of(index_elem_lltype.clone());

    let d = variable_with_lltype("d", dict_ptr_lltype);
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let value = variable_with_lltype("value", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(value.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // indexes = cast_opaque_ptr(INDEXES, d.indexes)
    let v_gcref = variable_with_lltype("indexes_gcref", GCREF.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(d), void_field_const("indexes")],
        Hlvalue::Variable(v_gcref.clone()),
    ));
    let v_indexes = variable_with_lltype("indexes", indexes_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_opaque_ptr",
        vec![Hlvalue::Variable(v_gcref)],
        Hlvalue::Variable(v_indexes.clone()),
    ));
    // cast_value = rffi.cast(T, value): narrow Signed slot value to the
    // unsigned index element width.
    let v_cast = variable_with_lltype("cast_value", index_elem_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(value)],
        Hlvalue::Variable(v_cast.clone()),
    ));
    // indexes[i] = cast_value
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(v_indexes),
            Hlvalue::Variable(i),
            Hlvalue::Variable(v_cast),
        ],
        Hlvalue::Variable(v_void),
    ));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string(), "i".to_string(), "value".to_string()],
        func,
    ))
}

/// Select the key-equality op for the simple-hash-eq `checkingkey == key`
/// comparison (`direct_compare`), keyed on the entry key lltype. Custom
/// `keyeq`/`paranoia` keys are out of scope (PBC `hlinvoke`).
fn direct_compare_op(key_lltype: &LowLevelType) -> Result<&'static str, TyperError> {
    Ok(match key_lltype {
        LowLevelType::Signed | LowLevelType::Bool => "int_eq",
        LowLevelType::Unsigned => "uint_eq",
        LowLevelType::SignedLongLong => "llong_eq",
        LowLevelType::SignedLongLongLong => "lllong_eq",
        LowLevelType::UnsignedLongLong => "ullong_eq",
        LowLevelType::UnsignedLongLongLong => "ulllong_eq",
        LowLevelType::Float => "float_eq",
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        LowLevelType::Ptr(_) => "ptr_eq",
        other => {
            return Err(TyperError::message(format!(
                "ll_dict_lookup direct_compare unsupported key lltype {other:?}"
            )));
        }
    })
}

fn lltype_must_clear_gc_ptr(lltype: &LowLevelType) -> bool {
    matches!(lltype, LowLevelType::Ptr(ptr) if ptr._needsgc())
}

/// Synthesise `ll_dict_lookup(d, key, hash, store_flag, T) -> Signed`
/// (`rordereddict.py`), the open-addressing perturb-probe that
/// every dict access routes through.
///
/// ```python
/// def ll_dict_lookup(d, key, hash, store_flag, T):
///     INDEXES = _ll_ptr_to_array_of(T)
///     entries = d.entries
///     indexes = lltype.cast_opaque_ptr(INDEXES, d.indexes)
///     mask = len(indexes) - 1
///     i = r_uint(hash & mask)
///     index = rffi.cast(lltype.Signed, indexes[intmask(i)])
///     if index >= VALID_OFFSET:
///         checkingkey = entries[index - VALID_OFFSET].key
///         if direct_compare and checkingkey == key:
///             return index - VALID_OFFSET
///         if d.keyeq is not None and entries.entry_hash(d, index - VALID_OFFSET) == hash:
///             found = d.keyeq(checkingkey, key)
///             if d.paranoia: ...                 # restart on mutation, r_dict-only
///             if found:
///                 return index - VALID_OFFSET
///         deletedslot = -1
///     elif index == DELETED:
///         deletedslot = intmask(i)
///     else:                                       # pristine -- lookup failed
///         if store_flag == FLAG_STORE:
///             _ll_write_indexes(d, i, d.num_ever_used_items + VALID_OFFSET, T)
///         return -1
///     perturb = r_uint(hash)
///     while 1:
///         i = (i << 2) + i + perturb + 1
///         i = i & mask
///         index = rffi.cast(lltype.Signed, indexes[intmask(i)])
///         if index == FREE:
///             if store_flag == FLAG_STORE:
///                 if deletedslot == -1:
///                     deletedslot = intmask(i)
///                 _ll_write_indexes(d, deletedslot,
///                                   d.num_ever_used_items + VALID_OFFSET, T)
///             return -1
///         elif index >= VALID_OFFSET:
///             checkingkey = entries[index - VALID_OFFSET].key
///             if direct_compare and checkingkey == key:
///                 return index - VALID_OFFSET
///             if d.keyeq is not None and entries.entry_hash(d, index - VALID_OFFSET) == hash:
///                 found = d.keyeq(checkingkey, key)
///                 if found:
///                     return index - VALID_OFFSET
///         elif deletedslot == -1:
///             deletedslot = intmask(i)
///         perturb >>= PERTURB_SHIFT
/// ```
///
/// **Scope (faithful subset):** `direct_compare` is always `True` here (no
/// key repr this file wires sets `no_direct_compare`, e.g. `ll_streq` does
/// not — see `rint.py:627`/`_rweakkeydict.py:107` for the two upstream cases
/// that do, both out of scope). `d.keyeq` is `eq_fn_const` — `None` for
/// identity keys (int/bool/char/unichar/instance-without-`__eq__`, matching
/// `get_ll_eq_function() -> None`, `rordereddict.py`), or the key
/// repr's `get_ll_eq_function()` result for keys with real structural
/// equality (`Some(ll_streq)` for str) or `ll_keyeq_custom` for
/// `custom_eq_hash` r_dicts. When `eq_fn_const` is `Some`, a
/// direct-compare miss falls through to `direct_call(eq_fn_const, ...)`
/// at both comparison sites. Custom ADT methods take `(d, checkingkey,
/// key)` (`eq_takes_dict`); plain keyeq takes `(checkingkey, key)`.
/// `d.paranoia` (`not simple_hash_eq` on a custom r_dict) restarts the
/// lookup after a user eq call if the table mutated
/// (`entries != d.entries or cast_opaque_ptr(GCREF, indexes) != d.indexes
/// or not entries.valid(slot) or entries[slot].key != checkingkey`).
/// `hash` is loop-carried so an in-loop restart still has it.
///
/// `entries.entry_hash(d, index) == hash` gates `d.keyeq(...)` at both
/// comparison sites (`ll_dict_lookup`). Custom equality may raise or
/// mutate the table, so a hash mismatch must not call it.
///
/// All `DICTINDEX_*` widths collapse to `Ptr(GcArray(Unsigned))`, so this one
/// graph serves every `FUNC_*` width (the `T`/`ll_call_lookup_function`
/// 4-way dispatch is inert).
///
/// **Unsigned arithmetic is load-bearing, not cosmetic:** `i` and `perturb`
/// are `r_uint`. `perturb >>= PERTURB_SHIFT` is a *logical* shift
/// (`uint_rshift`); a Signed `int_rshift` would sign-extend for the common
/// negative `hash`, walking a different probe sequence. So `i`/`perturb` are
/// modelled `Unsigned` end to end (`cast_int_to_uint` at the boundaries,
/// `cast_uint_to_int` for `intmask(i)` indexing). The `& mask` keeps the
/// signed/unsigned index value bit-identical, but the shift is not.
///
/// **Store path inlined:** `store_flag == FLAG_STORE` writes the index slot
/// via `cast_int_to_uint` + `setarrayitem` on the already-extracted `indexes`
/// — byte-equivalent to [`build_ll_write_indexes_helper_graph`]'s body (the
/// `indexes` pointer is invariant across a lookup, which never resizes). The
/// standalone `_ll_write_indexes` helper remains for the non-inlined callers
/// (`ll_dict_store_clean`, reindex) ported in later slices.
///
/// 13-block CFG (plus the returnblock) when `eq_fn_const` is `None`, 17-block
/// when `Some` (hash-check + `direct_call(eq_fn_const, ...)` per
/// comparison site). `paranoia` adds the mutation-restart chain after
/// each keyeq call. First-try-before-loop mirrors the "do the first try
/// before any looping" optimisation; the loop body re-derives `i`, reads the
/// slot, and 3-way branches FREE / VALID / DELETED before the `perturb`
/// shift back-edge.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors ll_dict_lookup; grouping would obscure line-by-line parity"
)]
pub fn build_ll_dict_lookup_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    index_elem_lltype: LowLevelType,
    key_lltype: LowLevelType,
    eq_fn_const: Option<Constant>,
    eq_takes_dict: bool,
    paranoia: bool,
    keyhash_fn: Option<Constant>,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let indexes_ptr_lltype = _ll_ptr_to_array_of(index_elem_lltype.clone());
    let eq_op = direct_compare_op(&key_lltype)?;
    let has_f_hash = dictentry_has_f_hash(&entries_ptr_lltype);

    // Value/const constructors.
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let unsigned =
        |n: i64| constant_with_lltype(ConstValue::UInt(n as u64), LowLevelType::Unsigned);
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let key_field = || void_field_const("key");
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let sig = || LowLevelType::Signed;
    let uns = || LowLevelType::Unsigned;
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);

    // startblock inputargs: (d, key, hash, store_flag, T). `T` is the Void
    // lltype (`ll_dict_lookup`); the body uses `index_elem_lltype` for INDEXES.
    let d = new_var("d", dict_ptr_lltype.clone());
    let key = new_var("key", key_lltype.clone());
    let hash = new_var("hash", sig());
    let store_flag = new_var("store_flag", sig());
    let t = new_var("T", LowLevelType::Void);
    let startblock = Block::shared(vec![
        var(&d),
        var(&key),
        var(&hash),
        var(&store_flag),
        var(&t),
    ]);
    let return_var = new_var("result", sig());
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    // Pre-create every downstream block with fresh inputarg copies so the
    // back-edge target exists when each block is closed.
    // block_first_valid / block_first_notvalid carry the post-first-probe state:
    //   (d, entries, indexes, mask_u, key, store_flag, hash, i, index).
    let make_inv = |suffix: &str| {
        (
            new_var("d", dict_ptr_lltype.clone()),
            new_var("entries", entries_ptr_lltype.clone()),
            new_var("indexes", indexes_ptr_lltype.clone()),
            new_var("mask_u", uns()),
            new_var("key", key_lltype.clone()),
            new_var(&format!("store_flag{suffix}"), sig()),
        )
    };

    // block_first_valid.
    let (fv_d, fv_entries, fv_indexes, fv_mask, fv_key, fv_sf) = make_inv("");
    let fv_hash = new_var("hash", sig());
    let fv_i = new_var("i", uns());
    let fv_index = new_var("index", sig());
    let block_first_valid = Block::shared(vec![
        var(&fv_d),
        var(&fv_entries),
        var(&fv_indexes),
        var(&fv_mask),
        var(&fv_key),
        var(&fv_sf),
        var(&fv_hash),
        var(&fv_i),
        var(&fv_index),
    ]);

    // block_first_notvalid.
    let (nv_d, nv_entries, nv_indexes, nv_mask, nv_key, nv_sf) = make_inv("");
    let nv_hash = new_var("hash", sig());
    let nv_i = new_var("i", uns());
    let nv_index = new_var("index", sig());
    let block_first_notvalid = Block::shared(vec![
        var(&nv_d),
        var(&nv_entries),
        var(&nv_indexes),
        var(&nv_mask),
        var(&nv_key),
        var(&nv_sf),
        var(&nv_hash),
        var(&nv_i),
        var(&nv_index),
    ]);

    // block_first_pristine_store: (d, indexes, store_flag, i).
    let ps_d = new_var("d", dict_ptr_lltype.clone());
    let ps_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let ps_sf = new_var("store_flag", sig());
    let ps_i = new_var("i", uns());
    let block_first_pristine_store =
        Block::shared(vec![var(&ps_d), var(&ps_indexes), var(&ps_sf), var(&ps_i)]);

    // block_store_at: (d, indexes, slot) — inlined _ll_write_indexes + return -1.
    let st_d = new_var("d", dict_ptr_lltype.clone());
    let st_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let st_slot = new_var("slot", sig());
    let block_store_at = Block::shared(vec![var(&st_d), var(&st_indexes), var(&st_slot)]);

    // block_loop_init: (d, entries, indexes, mask_u, key, store_flag, hash, i, deletedslot).
    let (li_d, li_entries, li_indexes, li_mask, li_key, li_sf) = make_inv("");
    let li_hash = new_var("hash", sig());
    let li_i = new_var("i", uns());
    let li_ds = new_var("deletedslot", sig());
    let block_loop_init = Block::shared(vec![
        var(&li_d),
        var(&li_entries),
        var(&li_indexes),
        var(&li_mask),
        var(&li_key),
        var(&li_sf),
        var(&li_hash),
        var(&li_i),
        var(&li_ds),
    ]);

    // block_loop_body: (d, entries, indexes, mask_u, key, store_flag, hash, perturb, i, deletedslot).
    // `hash` is loop-carried so a paranoia restart after user eq still has it.
    let (lb_d, lb_entries, lb_indexes, lb_mask, lb_key, lb_sf) = make_inv("");
    let lb_hash = new_var("hash", sig());
    let lb_perturb = new_var("perturb", uns());
    let lb_i = new_var("i", uns());
    let lb_ds = new_var("deletedslot", sig());
    let block_loop_body = Block::shared(vec![
        var(&lb_d),
        var(&lb_entries),
        var(&lb_indexes),
        var(&lb_mask),
        var(&lb_key),
        var(&lb_sf),
        var(&lb_hash),
        var(&lb_perturb),
        var(&lb_i),
        var(&lb_ds),
    ]);

    // block_loop_notfree: loop_body + index.
    let (nf_d, nf_entries, nf_indexes, nf_mask, nf_key, nf_sf) = make_inv("");
    let nf_hash = new_var("hash", sig());
    let nf_perturb = new_var("perturb", uns());
    let nf_i = new_var("i", uns());
    let nf_ds = new_var("deletedslot", sig());
    let nf_index = new_var("index", sig());
    let block_loop_notfree = Block::shared(vec![
        var(&nf_d),
        var(&nf_entries),
        var(&nf_indexes),
        var(&nf_mask),
        var(&nf_key),
        var(&nf_sf),
        var(&nf_hash),
        var(&nf_perturb),
        var(&nf_i),
        var(&nf_ds),
        var(&nf_index),
    ]);

    // block_loop_valid: loop_body + index.
    let (lv_d, lv_entries, lv_indexes, lv_mask, lv_key, lv_sf) = make_inv("");
    let lv_hash = new_var("hash", sig());
    let lv_perturb = new_var("perturb", uns());
    let lv_i = new_var("i", uns());
    let lv_ds = new_var("deletedslot", sig());
    let lv_index = new_var("index", sig());
    let block_loop_valid = Block::shared(vec![
        var(&lv_d),
        var(&lv_entries),
        var(&lv_indexes),
        var(&lv_mask),
        var(&lv_key),
        var(&lv_sf),
        var(&lv_hash),
        var(&lv_perturb),
        var(&lv_i),
        var(&lv_ds),
        var(&lv_index),
    ]);

    // block_loop_deleted: loop_body shape.
    let (ld_d, ld_entries, ld_indexes, ld_mask, ld_key, ld_sf) = make_inv("");
    let ld_hash = new_var("hash", sig());
    let ld_perturb = new_var("perturb", uns());
    let ld_i = new_var("i", uns());
    let ld_ds = new_var("deletedslot", sig());
    let block_loop_deleted = Block::shared(vec![
        var(&ld_d),
        var(&ld_entries),
        var(&ld_indexes),
        var(&ld_mask),
        var(&ld_key),
        var(&ld_sf),
        var(&ld_hash),
        var(&ld_perturb),
        var(&ld_i),
        var(&ld_ds),
    ]);

    // block_perturb_shift: loop_body shape.
    let (sh_d, sh_entries, sh_indexes, sh_mask, sh_key, sh_sf) = make_inv("");
    let sh_hash = new_var("hash", sig());
    let sh_perturb = new_var("perturb", uns());
    let sh_i = new_var("i", uns());
    let sh_ds = new_var("deletedslot", sig());
    let block_perturb_shift = Block::shared(vec![
        var(&sh_d),
        var(&sh_entries),
        var(&sh_indexes),
        var(&sh_mask),
        var(&sh_key),
        var(&sh_sf),
        var(&sh_hash),
        var(&sh_perturb),
        var(&sh_i),
        var(&sh_ds),
    ]);

    // block_loop_free: (d, indexes, store_flag, i, deletedslot).
    let lf_d = new_var("d", dict_ptr_lltype.clone());
    let lf_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let lf_sf = new_var("store_flag", sig());
    let lf_i = new_var("i", uns());
    let lf_ds = new_var("deletedslot", sig());
    let block_loop_free = Block::shared(vec![
        var(&lf_d),
        var(&lf_indexes),
        var(&lf_sf),
        var(&lf_i),
        var(&lf_ds),
    ]);

    // block_free_choose_slot: (d, indexes, i, deletedslot).
    let fc_d = new_var("d", dict_ptr_lltype.clone());
    let fc_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let fc_i = new_var("i", uns());
    let fc_ds = new_var("deletedslot", sig());
    let block_free_choose_slot =
        Block::shared(vec![var(&fc_d), var(&fc_indexes), var(&fc_i), var(&fc_ds)]);

    // startblock
    let entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("entries")],
        &entries,
    );
    let gcref = new_var("indexes_gcref", GCREF.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("indexes")],
        &gcref,
    );
    let indexes = new_var("indexes", indexes_ptr_lltype.clone());
    // `ll_dict_lookup`: `indexes = lltype.cast_opaque_ptr(INDEXES, d.indexes)`.
    push(&startblock, "cast_opaque_ptr", vec![var(&gcref)], &indexes);
    let len = new_var("len", sig());
    push(&startblock, "getarraysize", vec![var(&indexes)], &len);
    let mask = new_var("mask", sig());
    push(&startblock, "int_sub", vec![var(&len), signed(1)], &mask);
    let mask_u = new_var("mask_u", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&mask)], &mask_u);
    let hashmask = new_var("hashmask", sig());
    push(
        &startblock,
        "int_and",
        vec![var(&hash), var(&mask)],
        &hashmask,
    );
    let i0 = new_var("i", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&hashmask)], &i0);
    let i0_s = new_var("i_s", sig());
    push(&startblock, "cast_uint_to_int", vec![var(&i0)], &i0_s);
    let elem0 = new_var("elem", uns());
    push(
        &startblock,
        "getarrayitem",
        vec![var(&indexes), var(&i0_s)],
        &elem0,
    );
    let index0 = new_var("index", sig());
    push(&startblock, "cast_uint_to_int", vec![var(&elem0)], &index0);
    let ge0 = new_var("ge", LowLevelType::Bool);
    push(
        &startblock,
        "int_ge",
        vec![var(&index0), signed(VALID_OFFSET)],
        &ge0,
    );
    startblock.borrow_mut().exitswitch = Some(var(&ge0));
    let first_args = vec![
        var(&d),
        var(&entries),
        var(&indexes),
        var(&mask_u),
        var(&key),
        var(&store_flag),
        var(&hash),
        var(&i0),
        var(&index0),
    ];
    startblock.closeblock(vec![
        Link::new(
            first_args.clone(),
            Some(block_first_valid.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            first_args,
            Some(block_first_notvalid.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_first_valid: checkingkey == key on the first probe.
    let fv_slot = new_var("slot", sig());
    push(
        &block_first_valid,
        "int_sub",
        vec![var(&fv_index), signed(VALID_OFFSET)],
        &fv_slot,
    );
    let fv_ckey = new_var("checkingkey", key_lltype.clone());
    push(
        &block_first_valid,
        "getinteriorfield",
        vec![var(&fv_entries), var(&fv_slot), key_field()],
        &fv_ckey,
    );
    let fv_eq = new_var("keyeq", LowLevelType::Bool);
    push(
        &block_first_valid,
        eq_op,
        vec![var(&fv_ckey), var(&fv_key)],
        &fv_eq,
    );
    block_first_valid.borrow_mut().exitswitch = Some(var(&fv_eq));
    let fv_loop_init_args = vec![
        var(&fv_d),
        var(&fv_entries),
        var(&fv_indexes),
        var(&fv_mask),
        var(&fv_key),
        var(&fv_sf),
        var(&fv_hash),
        var(&fv_i),
        signed(-1),
    ];
    if let Some(eq_fn) = eq_fn_const.clone() {
        // d.keyeq is not None and entries.entry_hash(...) == hash
        // (`ll_dict_lookup`): hash mismatch skips the user eq call.
        let hashcheck_inputs = |d: &Variable,
                                entries: &Variable,
                                indexes: &Variable,
                                mask: &Variable,
                                key: &Variable,
                                sf: &Variable,
                                hash: &Variable,
                                i: &Variable,
                                slot: &Variable,
                                ckey: &Variable| {
            vec![
                var(d),
                var(entries),
                var(indexes),
                var(mask),
                var(key),
                var(sf),
                var(hash),
                var(i),
                var(slot),
                var(ckey),
            ]
        };
        let fhc_d = new_var("d", dict_ptr_lltype.clone());
        let fhc_entries = new_var("entries", entries_ptr_lltype.clone());
        let fhc_indexes = new_var("indexes", indexes_ptr_lltype.clone());
        let fhc_mask = new_var("mask_u", uns());
        let fhc_key = new_var("key", key_lltype.clone());
        let fhc_sf = new_var("store_flag", sig());
        let fhc_hash = new_var("hash", sig());
        let fhc_i = new_var("i", uns());
        let fhc_slot = new_var("slot", sig());
        let fhc_ckey = new_var("checkingkey", key_lltype.clone());
        let block_first_hashcheck = Block::shared(hashcheck_inputs(
            &fhc_d,
            &fhc_entries,
            &fhc_indexes,
            &fhc_mask,
            &fhc_key,
            &fhc_sf,
            &fhc_hash,
            &fhc_i,
            &fhc_slot,
            &fhc_ckey,
        ));
        let fkc_d = new_var("d", dict_ptr_lltype.clone());
        let fkc_entries = new_var("entries", entries_ptr_lltype.clone());
        let fkc_indexes = new_var("indexes", indexes_ptr_lltype.clone());
        let fkc_mask = new_var("mask_u", uns());
        let fkc_key = new_var("key", key_lltype.clone());
        let fkc_sf = new_var("store_flag", sig());
        let fkc_hash = new_var("hash", sig());
        let fkc_i = new_var("i", uns());
        let fkc_slot = new_var("slot", sig());
        let fkc_ckey = new_var("checkingkey", key_lltype.clone());
        let block_first_keyeq_call = Block::shared(hashcheck_inputs(
            &fkc_d,
            &fkc_entries,
            &fkc_indexes,
            &fkc_mask,
            &fkc_key,
            &fkc_sf,
            &fkc_hash,
            &fkc_i,
            &fkc_slot,
            &fkc_ckey,
        ));
        block_first_valid.closeblock(vec![
            Link::new(
                vec![var(&fv_slot)],
                Some(graph.returnblock.clone()),
                Some(bool_true()),
            )
            .into_ref(),
            Link::new(
                vec![
                    var(&fv_d),
                    var(&fv_entries),
                    var(&fv_indexes),
                    var(&fv_mask),
                    var(&fv_key),
                    var(&fv_sf),
                    var(&fv_hash),
                    var(&fv_i),
                    var(&fv_slot),
                    var(&fv_ckey),
                ],
                Some(block_first_hashcheck.clone()),
                Some(bool_false()),
            )
            .into_ref(),
        ]);
        let fhc_ok = emit_entry_hash_compare(
            &block_first_hashcheck,
            &fhc_entries,
            &fhc_slot,
            &fhc_hash,
            &fhc_d,
            &fhc_ckey,
            has_f_hash,
            keyhash_fn.as_ref(),
            hash_takes_dict,
        )?;
        block_first_hashcheck.borrow_mut().exitswitch = Some(var(&fhc_ok));
        block_first_hashcheck.closeblock(vec![
            Link::new(
                vec![
                    var(&fhc_d),
                    var(&fhc_entries),
                    var(&fhc_indexes),
                    var(&fhc_mask),
                    var(&fhc_key),
                    var(&fhc_sf),
                    var(&fhc_hash),
                    var(&fhc_i),
                    var(&fhc_slot),
                    var(&fhc_ckey),
                ],
                Some(block_first_keyeq_call.clone()),
                Some(bool_true()),
            )
            .into_ref(),
            Link::new(
                vec![
                    var(&fhc_d),
                    var(&fhc_entries),
                    var(&fhc_indexes),
                    var(&fhc_mask),
                    var(&fhc_key),
                    var(&fhc_sf),
                    var(&fhc_hash),
                    var(&fhc_i),
                    signed(-1),
                ],
                Some(block_loop_init.clone()),
                Some(bool_false()),
            )
            .into_ref(),
        ]);

        let fkc_found = new_var("found", LowLevelType::Bool);
        push(
            &block_first_keyeq_call,
            "direct_call",
            keyeq_call_args(eq_fn, &fkc_d, &fkc_ckey, &fkc_key, eq_takes_dict),
            &fkc_found,
        );
        let first_miss_vars = vec![
            fkc_d.clone(),
            fkc_entries.clone(),
            fkc_indexes.clone(),
            fkc_mask.clone(),
            fkc_key.clone(),
            fkc_sf.clone(),
            fkc_hash.clone(),
            fkc_i.clone(),
        ];
        close_keyeq_found_or_paranoia(
            &block_first_keyeq_call,
            &fkc_found,
            &fkc_d,
            &fkc_entries,
            &fkc_indexes,
            &fkc_key,
            &fkc_hash,
            &fkc_sf,
            &fkc_slot,
            &fkc_ckey,
            &first_miss_vars,
            vec![signed(-1)],
            block_loop_init.clone(),
            startblock.clone(),
            graph.returnblock.clone(),
            &dict_ptr_lltype,
            &entries_ptr_lltype,
            &indexes_ptr_lltype,
            &key_lltype,
            eq_op,
            paranoia,
            lowlevel_type_const(index_elem_lltype.clone()),
        );
    } else {
        block_first_valid.closeblock(vec![
            Link::new(
                vec![var(&fv_slot)],
                Some(graph.returnblock.clone()),
                Some(bool_true()),
            )
            .into_ref(),
            Link::new(
                fv_loop_init_args,
                Some(block_loop_init.clone()),
                Some(bool_false()),
            )
            .into_ref(),
        ]);
    }

    // block_first_notvalid: DELETED vs pristine FREE.
    let nv_i_s = new_var("i_s", sig());
    push(
        &block_first_notvalid,
        "cast_uint_to_int",
        vec![var(&nv_i)],
        &nv_i_s,
    );
    let nv_is_deleted = new_var("is_deleted", LowLevelType::Bool);
    push(
        &block_first_notvalid,
        "int_eq",
        vec![var(&nv_index), signed(DELETED)],
        &nv_is_deleted,
    );
    block_first_notvalid.borrow_mut().exitswitch = Some(var(&nv_is_deleted));
    block_first_notvalid.closeblock(vec![
        Link::new(
            vec![
                var(&nv_d),
                var(&nv_entries),
                var(&nv_indexes),
                var(&nv_mask),
                var(&nv_key),
                var(&nv_sf),
                var(&nv_hash),
                var(&nv_i),
                var(&nv_i_s),
            ],
            Some(block_loop_init.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![var(&nv_d), var(&nv_indexes), var(&nv_sf), var(&nv_i)],
            Some(block_first_pristine_store.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_first_pristine_store: store at i iff FLAG_STORE, else -1.
    let ps_i_s = new_var("i_s", sig());
    push(
        &block_first_pristine_store,
        "cast_uint_to_int",
        vec![var(&ps_i)],
        &ps_i_s,
    );
    let ps_is_store = new_var("is_store", LowLevelType::Bool);
    push(
        &block_first_pristine_store,
        "int_eq",
        vec![var(&ps_sf), signed(FLAG_STORE)],
        &ps_is_store,
    );
    block_first_pristine_store.borrow_mut().exitswitch = Some(var(&ps_is_store));
    block_first_pristine_store.closeblock(vec![
        Link::new(
            vec![var(&ps_d), var(&ps_indexes), var(&ps_i_s)],
            Some(block_store_at.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed(-1)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_store_at: indexes[slot] = num_ever_used + VALID_OFFSET; return -1.
    let st_neu = new_var("num_ever_used_items", sig());
    push(
        &block_store_at,
        "getfield",
        vec![var(&st_d), void_field_const("num_ever_used_items")],
        &st_neu,
    );
    let st_value = new_var("value", sig());
    push(
        &block_store_at,
        "int_add",
        vec![var(&st_neu), signed(VALID_OFFSET)],
        &st_value,
    );
    let st_cast = new_var("cast_value", index_elem_lltype.clone());
    push(
        &block_store_at,
        "cast_int_to_uint",
        vec![var(&st_value)],
        &st_cast,
    );
    let st_void = new_var("v", LowLevelType::Void);
    push(
        &block_store_at,
        "setarrayitem",
        vec![var(&st_indexes), var(&st_slot), var(&st_cast)],
        &st_void,
    );
    block_store_at.closeblock(vec![
        Link::new(vec![signed(-1)], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // block_loop_init: perturb = r_uint(hash); enter loop.
    let li_perturb = new_var("perturb", uns());
    push(
        &block_loop_init,
        "cast_int_to_uint",
        vec![var(&li_hash)],
        &li_perturb,
    );
    block_loop_init.closeblock(vec![
        Link::new(
            vec![
                var(&li_d),
                var(&li_entries),
                var(&li_indexes),
                var(&li_mask),
                var(&li_key),
                var(&li_sf),
                var(&li_hash),
                var(&li_perturb),
                var(&li_i),
                var(&li_ds),
            ],
            Some(block_loop_body.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_loop_body: i = ((i<<2)+i+perturb+1)&mask; read slot; branch FREE.
    let lb_ish = new_var("ish", uns());
    push(
        &block_loop_body,
        "uint_lshift",
        vec![var(&lb_i), signed(2)],
        &lb_ish,
    );
    let lb_ipi = new_var("ipi", uns());
    push(
        &block_loop_body,
        "uint_add",
        vec![var(&lb_ish), var(&lb_i)],
        &lb_ipi,
    );
    let lb_ipp = new_var("ipp", uns());
    push(
        &block_loop_body,
        "uint_add",
        vec![var(&lb_ipi), var(&lb_perturb)],
        &lb_ipp,
    );
    let lb_iinc = new_var("iinc", uns());
    push(
        &block_loop_body,
        "uint_add",
        vec![var(&lb_ipp), unsigned(1)],
        &lb_iinc,
    );
    let lb_inew = new_var("i", uns());
    push(
        &block_loop_body,
        "uint_and",
        vec![var(&lb_iinc), var(&lb_mask)],
        &lb_inew,
    );
    let lb_inew_s = new_var("i_s", sig());
    push(
        &block_loop_body,
        "cast_uint_to_int",
        vec![var(&lb_inew)],
        &lb_inew_s,
    );
    let lb_elem = new_var("elem", uns());
    push(
        &block_loop_body,
        "getarrayitem",
        vec![var(&lb_indexes), var(&lb_inew_s)],
        &lb_elem,
    );
    let lb_index = new_var("index", sig());
    push(
        &block_loop_body,
        "cast_uint_to_int",
        vec![var(&lb_elem)],
        &lb_index,
    );
    let lb_is_free = new_var("is_free", LowLevelType::Bool);
    push(
        &block_loop_body,
        "int_eq",
        vec![var(&lb_index), signed(FREE)],
        &lb_is_free,
    );
    block_loop_body.borrow_mut().exitswitch = Some(var(&lb_is_free));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                var(&lb_d),
                var(&lb_indexes),
                var(&lb_sf),
                var(&lb_inew),
                var(&lb_ds),
            ],
            Some(block_loop_free.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&lb_d),
                var(&lb_entries),
                var(&lb_indexes),
                var(&lb_mask),
                var(&lb_key),
                var(&lb_sf),
                var(&lb_hash),
                var(&lb_perturb),
                var(&lb_inew),
                var(&lb_ds),
                var(&lb_index),
            ],
            Some(block_loop_notfree.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_notfree: index >= VALID_OFFSET vs DELETED.
    let nf_ge = new_var("ge", LowLevelType::Bool);
    push(
        &block_loop_notfree,
        "int_ge",
        vec![var(&nf_index), signed(VALID_OFFSET)],
        &nf_ge,
    );
    block_loop_notfree.borrow_mut().exitswitch = Some(var(&nf_ge));
    let nf_carry = vec![
        var(&nf_d),
        var(&nf_entries),
        var(&nf_indexes),
        var(&nf_mask),
        var(&nf_key),
        var(&nf_sf),
        var(&nf_hash),
        var(&nf_perturb),
        var(&nf_i),
        var(&nf_ds),
    ];
    let mut nf_valid_args = nf_carry.clone();
    nf_valid_args.push(var(&nf_index));
    block_loop_notfree.closeblock(vec![
        Link::new(
            nf_valid_args,
            Some(block_loop_valid.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            nf_carry,
            Some(block_loop_deleted.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_loop_valid: checkingkey == key on a probed slot.
    let lv_slot = new_var("slot", sig());
    push(
        &block_loop_valid,
        "int_sub",
        vec![var(&lv_index), signed(VALID_OFFSET)],
        &lv_slot,
    );
    let lv_ckey = new_var("checkingkey", key_lltype.clone());
    push(
        &block_loop_valid,
        "getinteriorfield",
        vec![var(&lv_entries), var(&lv_slot), key_field()],
        &lv_ckey,
    );
    let lv_eq = new_var("keyeq", LowLevelType::Bool);
    push(
        &block_loop_valid,
        eq_op,
        vec![var(&lv_ckey), var(&lv_key)],
        &lv_eq,
    );
    block_loop_valid.borrow_mut().exitswitch = Some(var(&lv_eq));
    let lv_perturb_shift_args = vec![
        var(&lv_d),
        var(&lv_entries),
        var(&lv_indexes),
        var(&lv_mask),
        var(&lv_key),
        var(&lv_sf),
        var(&lv_hash),
        var(&lv_perturb),
        var(&lv_i),
        var(&lv_ds),
    ];
    if let Some(eq_fn) = eq_fn_const.clone() {
        // d.keyeq is not None and entries.entry_hash(...) == hash
        // (`ll_dict_lookup`), in-loop mirror of the first-try hash gate.
        let loop_cmp_inputs = |d: &Variable,
                               entries: &Variable,
                               indexes: &Variable,
                               mask: &Variable,
                               key: &Variable,
                               sf: &Variable,
                               hash: &Variable,
                               perturb: &Variable,
                               i: &Variable,
                               ds: &Variable,
                               slot: &Variable,
                               ckey: &Variable| {
            vec![
                var(d),
                var(entries),
                var(indexes),
                var(mask),
                var(key),
                var(sf),
                var(hash),
                var(perturb),
                var(i),
                var(ds),
                var(slot),
                var(ckey),
            ]
        };
        let lhc_d = new_var("d", dict_ptr_lltype.clone());
        let lhc_entries = new_var("entries", entries_ptr_lltype.clone());
        let lhc_indexes = new_var("indexes", indexes_ptr_lltype.clone());
        let lhc_mask = new_var("mask_u", uns());
        let lhc_key = new_var("key", key_lltype.clone());
        let lhc_sf = new_var("store_flag", sig());
        let lhc_hash = new_var("hash", sig());
        let lhc_perturb = new_var("perturb", uns());
        let lhc_i = new_var("i", uns());
        let lhc_ds = new_var("deletedslot", sig());
        let lhc_slot = new_var("slot", sig());
        let lhc_ckey = new_var("checkingkey", key_lltype.clone());
        let block_loop_hashcheck = Block::shared(loop_cmp_inputs(
            &lhc_d,
            &lhc_entries,
            &lhc_indexes,
            &lhc_mask,
            &lhc_key,
            &lhc_sf,
            &lhc_hash,
            &lhc_perturb,
            &lhc_i,
            &lhc_ds,
            &lhc_slot,
            &lhc_ckey,
        ));
        let lkc_d = new_var("d", dict_ptr_lltype.clone());
        let lkc_entries = new_var("entries", entries_ptr_lltype.clone());
        let lkc_indexes = new_var("indexes", indexes_ptr_lltype.clone());
        let lkc_mask = new_var("mask_u", uns());
        let lkc_key = new_var("key", key_lltype.clone());
        let lkc_sf = new_var("store_flag", sig());
        let lkc_hash = new_var("hash", sig());
        let lkc_perturb = new_var("perturb", uns());
        let lkc_i = new_var("i", uns());
        let lkc_ds = new_var("deletedslot", sig());
        let lkc_slot = new_var("slot", sig());
        let lkc_ckey = new_var("checkingkey", key_lltype.clone());
        let block_loop_keyeq_call = Block::shared(loop_cmp_inputs(
            &lkc_d,
            &lkc_entries,
            &lkc_indexes,
            &lkc_mask,
            &lkc_key,
            &lkc_sf,
            &lkc_hash,
            &lkc_perturb,
            &lkc_i,
            &lkc_ds,
            &lkc_slot,
            &lkc_ckey,
        ));
        block_loop_valid.closeblock(vec![
            Link::new(
                vec![var(&lv_slot)],
                Some(graph.returnblock.clone()),
                Some(bool_true()),
            )
            .into_ref(),
            Link::new(
                vec![
                    var(&lv_d),
                    var(&lv_entries),
                    var(&lv_indexes),
                    var(&lv_mask),
                    var(&lv_key),
                    var(&lv_sf),
                    var(&lv_hash),
                    var(&lv_perturb),
                    var(&lv_i),
                    var(&lv_ds),
                    var(&lv_slot),
                    var(&lv_ckey),
                ],
                Some(block_loop_hashcheck.clone()),
                Some(bool_false()),
            )
            .into_ref(),
        ]);
        let lhc_ok = emit_entry_hash_compare(
            &block_loop_hashcheck,
            &lhc_entries,
            &lhc_slot,
            &lhc_hash,
            &lhc_d,
            &lhc_ckey,
            has_f_hash,
            keyhash_fn.as_ref(),
            hash_takes_dict,
        )?;
        block_loop_hashcheck.borrow_mut().exitswitch = Some(var(&lhc_ok));
        block_loop_hashcheck.closeblock(vec![
            Link::new(
                vec![
                    var(&lhc_d),
                    var(&lhc_entries),
                    var(&lhc_indexes),
                    var(&lhc_mask),
                    var(&lhc_key),
                    var(&lhc_sf),
                    var(&lhc_hash),
                    var(&lhc_perturb),
                    var(&lhc_i),
                    var(&lhc_ds),
                    var(&lhc_slot),
                    var(&lhc_ckey),
                ],
                Some(block_loop_keyeq_call.clone()),
                Some(bool_true()),
            )
            .into_ref(),
            Link::new(
                vec![
                    var(&lhc_d),
                    var(&lhc_entries),
                    var(&lhc_indexes),
                    var(&lhc_mask),
                    var(&lhc_key),
                    var(&lhc_sf),
                    var(&lhc_hash),
                    var(&lhc_perturb),
                    var(&lhc_i),
                    var(&lhc_ds),
                ],
                Some(block_perturb_shift.clone()),
                Some(bool_false()),
            )
            .into_ref(),
        ]);

        let lkc_found = new_var("found", LowLevelType::Bool);
        push(
            &block_loop_keyeq_call,
            "direct_call",
            keyeq_call_args(eq_fn, &lkc_d, &lkc_ckey, &lkc_key, eq_takes_dict),
            &lkc_found,
        );
        let loop_miss_vars = vec![
            lkc_d.clone(),
            lkc_entries.clone(),
            lkc_indexes.clone(),
            lkc_mask.clone(),
            lkc_key.clone(),
            lkc_sf.clone(),
            lkc_hash.clone(),
            lkc_perturb.clone(),
            lkc_i.clone(),
            lkc_ds.clone(),
        ];
        close_keyeq_found_or_paranoia(
            &block_loop_keyeq_call,
            &lkc_found,
            &lkc_d,
            &lkc_entries,
            &lkc_indexes,
            &lkc_key,
            &lkc_hash,
            &lkc_sf,
            &lkc_slot,
            &lkc_ckey,
            &loop_miss_vars,
            vec![],
            block_perturb_shift.clone(),
            startblock.clone(),
            graph.returnblock.clone(),
            &dict_ptr_lltype,
            &entries_ptr_lltype,
            &indexes_ptr_lltype,
            &key_lltype,
            eq_op,
            paranoia,
            lowlevel_type_const(index_elem_lltype.clone()),
        );
    } else {
        block_loop_valid.closeblock(vec![
            Link::new(
                vec![var(&lv_slot)],
                Some(graph.returnblock.clone()),
                Some(bool_true()),
            )
            .into_ref(),
            Link::new(
                lv_perturb_shift_args,
                Some(block_perturb_shift.clone()),
                Some(bool_false()),
            )
            .into_ref(),
        ]);
    }

    // block_loop_deleted: record first deleted slot (deletedslot == -1).
    let ld_i_s = new_var("i_s", sig());
    push(
        &block_loop_deleted,
        "cast_uint_to_int",
        vec![var(&ld_i)],
        &ld_i_s,
    );
    let ld_ds_m1 = new_var("ds_is_m1", LowLevelType::Bool);
    push(
        &block_loop_deleted,
        "int_eq",
        vec![var(&ld_ds), signed(-1)],
        &ld_ds_m1,
    );
    block_loop_deleted.borrow_mut().exitswitch = Some(var(&ld_ds_m1));
    let ld_head = vec![
        var(&ld_d),
        var(&ld_entries),
        var(&ld_indexes),
        var(&ld_mask),
        var(&ld_key),
        var(&ld_sf),
        var(&ld_hash),
        var(&ld_perturb),
        var(&ld_i),
    ];
    let mut ld_set_args = ld_head.clone();
    ld_set_args.push(var(&ld_i_s));
    let mut ld_keep_args = ld_head;
    ld_keep_args.push(var(&ld_ds));
    block_loop_deleted.closeblock(vec![
        Link::new(
            ld_set_args,
            Some(block_perturb_shift.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            ld_keep_args,
            Some(block_perturb_shift.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_perturb_shift: perturb >>= PERTURB_SHIFT; back-edge to loop body.
    let sh_perturb_new = new_var("perturb", uns());
    push(
        &block_perturb_shift,
        "uint_rshift",
        vec![var(&sh_perturb), signed(PERTURB_SHIFT)],
        &sh_perturb_new,
    );
    block_perturb_shift.closeblock(vec![
        Link::new(
            vec![
                var(&sh_d),
                var(&sh_entries),
                var(&sh_indexes),
                var(&sh_mask),
                var(&sh_key),
                var(&sh_sf),
                var(&sh_hash),
                var(&sh_perturb_new),
                var(&sh_i),
                var(&sh_ds),
            ],
            Some(block_loop_body.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_loop_free: store at deletedslot iff FLAG_STORE, else -1.
    let lf_is_store = new_var("is_store", LowLevelType::Bool);
    push(
        &block_loop_free,
        "int_eq",
        vec![var(&lf_sf), signed(FLAG_STORE)],
        &lf_is_store,
    );
    block_loop_free.borrow_mut().exitswitch = Some(var(&lf_is_store));
    block_loop_free.closeblock(vec![
        Link::new(
            vec![var(&lf_d), var(&lf_indexes), var(&lf_i), var(&lf_ds)],
            Some(block_free_choose_slot.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed(-1)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // block_free_choose_slot: deletedslot==-1 ? i : deletedslot, then store.
    let fc_i_s = new_var("i_s", sig());
    push(
        &block_free_choose_slot,
        "cast_uint_to_int",
        vec![var(&fc_i)],
        &fc_i_s,
    );
    let fc_ds_m1 = new_var("ds_is_m1", LowLevelType::Bool);
    push(
        &block_free_choose_slot,
        "int_eq",
        vec![var(&fc_ds), signed(-1)],
        &fc_ds_m1,
    );
    block_free_choose_slot.borrow_mut().exitswitch = Some(var(&fc_ds_m1));
    block_free_choose_slot.closeblock(vec![
        Link::new(
            vec![var(&fc_d), var(&fc_indexes), var(&fc_i_s)],
            Some(block_store_at.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![var(&fc_d), var(&fc_indexes), var(&fc_ds)],
            Some(block_store_at.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `rordereddict.py ll_dict_lookup`: `@jit.oopspec('ordereddict.lookup(d, key, hash, store_flag, T)')`.
    func.oopspec = Some("ordereddict.lookup(d, key, hash, store_flag, T)".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "d".to_string(),
            "key".to_string(),
            "hash".to_string(),
            "store_flag".to_string(),
            "T".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_malloc_indexes_and_choose_lookup(d, n)` (`rordereddict.py`):
///
/// ```python
/// def ll_malloc_indexes_and_choose_lookup(d, n):
///     if n <= 256:
///         d.indexes = ...; d.lookup_function_no = FUNC_BYTE
///     elif n <= 65536:
///         d.indexes = ...; d.lookup_function_no = FUNC_SHORT
///     elif IS_64BIT and n <= 2 ** 32:
///         d.indexes = ...; d.lookup_function_no = FUNC_INT
///     else:
///         d.indexes = ...; d.lookup_function_no = FUNC_LONG
/// ```
///
/// All `DICTINDEX_*` widths collapse to the identical `Ptr(GcArray(Unsigned))`
/// shape locally (see [`DICTINDEX_BYTE`]), so every branch allocates the same
/// array; only the `FUNC_*` selector constant stamped into
/// `lookup_function_no` differs. `IS_64BIT` is a Rust compile-time constant
/// here exactly as upstream's module-level `IS_64BIT and` guard is decided
/// once, not per call — the `n <= 2 ** 32` check is only emitted when built
/// for a 64-bit target (matching `FUNC_INT`'s own conditional definition,
/// `:59-60`).
pub(crate) fn build_ll_malloc_indexes_and_choose_lookup_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let indexes_array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Unsigned)));
    let indexes_ptr_lltype = _ll_ptr_to_array_of(LowLevelType::Unsigned);

    let d0 = variable_with_lltype("d", dict_ptr_lltype.clone());
    let n0 = variable_with_lltype("n", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d0.clone()),
        Hlvalue::Variable(n0.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let returnblock = graph.returnblock.clone();

    // Shared leaf body: malloc the (uniformly `Unsigned`-element) index
    // array, then stamp `d.indexes` / `d.lookup_function_no`. Only
    // `func_const` differs branch to branch.
    let alloc_leaf = |func_const: i64| -> Result<BlockRef, TyperError> {
        let d = variable_with_lltype("d", dict_ptr_lltype.clone());
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let block = Block::shared(vec![
            Hlvalue::Variable(d.clone()),
            Hlvalue::Variable(n.clone()),
        ]);
        let arr = variable_with_lltype("arr", indexes_ptr_lltype.clone());
        block.borrow_mut().operations.push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(indexes_array_ty.clone()),
                gc_flavor_const()?,
                Hlvalue::Variable(n),
            ],
            Hlvalue::Variable(arr.clone()),
        ));
        let gcref = variable_with_lltype("indexes_gcref", GCREF.clone());
        block.borrow_mut().operations.push(SpaceOperation::new(
            "cast_opaque_ptr",
            vec![Hlvalue::Variable(arr)],
            Hlvalue::Variable(gcref.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(d.clone()),
                void_field_const("indexes"),
                Hlvalue::Variable(gcref),
            ],
            Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(d),
                void_field_const("lookup_function_no"),
                constant_with_lltype(ConstValue::Int(func_const), LowLevelType::Signed),
            ],
            Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
        ));
        let none_const = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ));
        block.closeblock(vec![
            Link::new(vec![none_const], Some(returnblock.clone()), None).into_ref(),
        ]);
        Ok(block)
    };

    let mut thresholds: Vec<(i64, i64)> = vec![(256, FUNC_BYTE), (65536, FUNC_SHORT)];
    if IS_64BIT {
        thresholds.push((1i64 << 32, FUNC_INT));
    }
    let long_leaf = alloc_leaf(FUNC_LONG)?;

    // Cascade the threshold checks from the last (LONG fallback, no check
    // needed) to the first (BYTE), so each check block's false-edge target
    // already exists when the block closes.
    let mut false_target = long_leaf;
    for (threshold, func_const) in thresholds.into_iter().rev() {
        let leaf = alloc_leaf(func_const)?;
        let d = variable_with_lltype("d", dict_ptr_lltype.clone());
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let block = Block::shared(vec![
            Hlvalue::Variable(d.clone()),
            Hlvalue::Variable(n.clone()),
        ]);
        let le = variable_with_lltype("le", LowLevelType::Bool);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "int_le",
            vec![
                Hlvalue::Variable(n.clone()),
                constant_with_lltype(ConstValue::Int(threshold), LowLevelType::Signed),
            ],
            Hlvalue::Variable(le.clone()),
        ));
        block.borrow_mut().exitswitch = Some(Hlvalue::Variable(le));
        block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(d.clone()), Hlvalue::Variable(n.clone())],
                Some(leaf),
                Some(constant_with_lltype(
                    ConstValue::Bool(true),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(d), Hlvalue::Variable(n)],
                Some(false_target),
                Some(constant_with_lltype(
                    ConstValue::Bool(false),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
        ]);
        false_target = block;
    }

    // startblock forwards straight into the first (BYTE) threshold check.
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(d0), Hlvalue::Variable(n0)],
            Some(false_target),
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
        vec!["d".to_string(), "n".to_string()],
        func,
    ))
}

/// Synthesise `ll_dict_create_initial_index(d)` (`rordereddict.py`):
///
/// ```python
/// def ll_dict_create_initial_index(d):
///     if d.num_live_items == 0:
///         ll_malloc_indexes_and_choose_lookup(d, DICT_INITSIZE)
///         d.resize_counter = DICT_INITSIZE * 2
///     else:
///         ll_dict_rehash_after_translation(d)
/// ```
pub(crate) fn build_ll_dict_create_initial_index_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    malloc_choose_fn: Constant,
    rehash_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);

    let d = new_var("d", dict_ptr_lltype.clone());
    let startblock = Block::shared(vec![var(&d)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let num_live = new_var("num_live", LowLevelType::Signed);
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("num_live_items")],
        &num_live,
    );
    let is_empty = new_var("is_empty", LowLevelType::Bool);
    push(
        &startblock,
        "int_eq",
        vec![var(&num_live), signed(0)],
        &is_empty,
    );
    startblock.borrow_mut().exitswitch = Some(var(&is_empty));

    let empty_d = new_var("d", dict_ptr_lltype.clone());
    let block_empty = Block::shared(vec![var(&empty_d)]);
    let rehash_d = new_var("d", dict_ptr_lltype.clone());
    let block_rehash = Block::shared(vec![var(&rehash_d)]);

    startblock.closeblock(vec![
        Link::new(
            vec![var(&d)],
            Some(block_empty.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d)],
            Some(block_rehash.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    push(
        &block_empty,
        "direct_call",
        vec![
            Hlvalue::Constant(malloc_choose_fn),
            var(&empty_d),
            signed(DICT_INITSIZE),
        ],
        &new_var("v", LowLevelType::Void),
    );
    push(
        &block_empty,
        "setfield",
        vec![
            var(&empty_d),
            void_field_const("resize_counter"),
            signed(DICT_INITSIZE * 2),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_empty.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    push(
        &block_rehash,
        "direct_call",
        vec![Hlvalue::Constant(rehash_fn), var(&rehash_d)],
        &new_var("v", LowLevelType::Void),
    );
    block_rehash.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string()],
        func,
    ))
}

/// Mint `ll_dict_create_initial_index` through `look_inside_iff`
/// (`rlib/jit.py`): original body keeps `func.__name__`, trampoline is
/// `<name>_trampoline`, wrapper is `<name>_look_inside_iff`. Returns the wrapper.
fn mint_ll_dict_create_initial_index_look_inside_iff(
    rtyper: &RPythonTyper,
    dict_ptr: LowLevelType,
    malloc_choose_fn: Constant,
    rehash_fn: Constant,
) -> Result<LowLevelFunction, TyperError> {
    mint_look_inside_iff(
        rtyper,
        "ll_dict_create_initial_index",
        &["d"],
        vec![dict_ptr.clone()],
        LowLevelType::Void,
        {
            let dict_ptr = dict_ptr;
            move |name| {
                build_ll_dict_create_initial_index_helper_graph(
                    name,
                    dict_ptr.clone(),
                    malloc_choose_fn.clone(),
                    rehash_fn.clone(),
                )
            }
        },
        &[LookInsidePred::IsVirtual(0)],
    )
}

/// Synthesise `ll_dict_rehash_after_translation(d)` (`rordereddict.py`):
///
/// ```python
/// def ll_dict_rehash_after_translation(d):
///     for i in range(d.num_ever_used_items):
///         d_entry = d.entries[i]
///         h = d.keyhash(d_entry.key)
///         if hasattr(ENTRY, 'f_hash'):
///             d_entry.f_hash = h
///     new_size = DICT_INITSIZE
///     while new_size * 2 - d.num_live_items * 3 <= 0:
///         new_size *= 2
///     ll_dict_reindex(d, new_size)
/// ```
///
/// `d.num_ever_used_items` is the range bound (read once). `d.entries` is
/// re-read each loop iteration before `d_entry.key`, after the previous
/// iteration's `d.keyhash`. `d.num_live_items` is read in the `new_size`
/// while-test after the hash loop (`ll_dict_rehash_after_translation`).
/// The `assert`s (`num_live_items == num_ever_used_items`, `not d.indexes`,
/// `entries.valid(i)`) are debug-only, same omission as
/// [`build_ll_ensure_indexes_helper_graph`].
pub(crate) fn build_ll_dict_rehash_after_translation_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    reindex_fn: Constant,
    keyhash_fn: Constant,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let startblock = Block::shared(vec![var(&d)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let ibound = new_var("ibound", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("num_ever_used_items")],
        &ibound,
    );

    let cd_d = new_var("d", dict_ptr_lltype.clone());
    let cd_ibound = new_var("ibound", sig());
    let cd_i = new_var("i", sig());
    let block_cond = Block::shared(vec![var(&cd_d), var(&cd_ibound), var(&cd_i)]);

    startblock.closeblock(vec![
        Link::new(
            vec![var(&d), var(&ibound), signed(0)],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let bd_d = new_var("d", dict_ptr_lltype.clone());
    let bd_ibound = new_var("ibound", sig());
    let bd_i = new_var("i", sig());
    let block_body = Block::shared(vec![var(&bd_d), var(&bd_ibound), var(&bd_i)]);

    let sz_d = new_var("d", dict_ptr_lltype.clone());
    let sz_size = new_var("new_size", sig());
    let block_size_cond = Block::shared(vec![var(&sz_d), var(&sz_size)]);

    let cond = new_var("cond", LowLevelType::Bool);
    push(
        &block_cond,
        "int_lt",
        vec![var(&cd_i), var(&cd_ibound)],
        &cond,
    );
    block_cond.borrow_mut().exitswitch = Some(var(&cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![var(&cd_d), var(&cd_ibound), var(&cd_i)],
            Some(block_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&cd_d), signed(DICT_INITSIZE)],
            Some(block_size_cond.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let key_ll = dictentry_key_lltype(&entries_ptr_lltype).ok_or_else(|| {
        TyperError::message("ll_dict_rehash_after_translation: ENTRY.key missing")
    })?;
    let entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &block_body,
        "getfield",
        vec![var(&bd_d), void_field_const("entries")],
        &entries,
    );
    let entry_key = new_var("entry_key", key_ll);
    push(
        &block_body,
        "getinteriorfield",
        vec![var(&entries), var(&bd_i), void_field_const("key")],
        &entry_key,
    );
    let h = new_var("h", sig());
    push(
        &block_body,
        "direct_call",
        keyhash_call_args(keyhash_fn, &bd_d, &entry_key, hash_takes_dict),
        &h,
    );
    if dictentry_has_f_hash(&entries_ptr_lltype) {
        push(
            &block_body,
            "setinteriorfield",
            vec![
                var(&entries),
                var(&bd_i),
                void_field_const("f_hash"),
                var(&h),
            ],
            &new_var("v", LowLevelType::Void),
        );
    }
    let i_next = new_var("i", sig());
    push(&block_body, "int_add", vec![var(&bd_i), signed(1)], &i_next);
    block_body.closeblock(vec![
        Link::new(
            vec![var(&bd_d), var(&bd_ibound), var(&i_next)],
            Some(block_cond),
            None,
        )
        .into_ref(),
    ]);

    let db_d = new_var("d", dict_ptr_lltype.clone());
    let db_size = new_var("new_size", sig());
    let block_size_body = Block::shared(vec![var(&db_d), var(&db_size)]);

    let cr_d = new_var("d", dict_ptr_lltype.clone());
    let cr_size = new_var("new_size", sig());
    let block_call_reindex = Block::shared(vec![var(&cr_d), var(&cr_size)]);

    let live = new_var("live", sig());
    push(
        &block_size_cond,
        "getfield",
        vec![var(&sz_d), void_field_const("num_live_items")],
        &live,
    );
    let t1 = new_var("t1", sig());
    push(
        &block_size_cond,
        "int_mul",
        vec![var(&sz_size), signed(2)],
        &t1,
    );
    let t2 = new_var("t2", sig());
    push(
        &block_size_cond,
        "int_mul",
        vec![var(&live), signed(3)],
        &t2,
    );
    let t3 = new_var("t3", sig());
    push(&block_size_cond, "int_sub", vec![var(&t1), var(&t2)], &t3);
    let grow = new_var("grow", LowLevelType::Bool);
    push(&block_size_cond, "int_le", vec![var(&t3), signed(0)], &grow);
    block_size_cond.borrow_mut().exitswitch = Some(var(&grow));
    block_size_cond.closeblock(vec![
        Link::new(
            vec![var(&sz_d), var(&sz_size)],
            Some(block_size_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&sz_d), var(&sz_size)],
            Some(block_call_reindex.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let db_size2 = new_var("new_size", sig());
    push(
        &block_size_body,
        "int_mul",
        vec![var(&db_size), signed(2)],
        &db_size2,
    );
    block_size_body.closeblock(vec![
        Link::new(
            vec![var(&db_d), var(&db_size2)],
            Some(block_size_cond),
            None,
        )
        .into_ref(),
    ]);

    push(
        &block_call_reindex,
        "direct_call",
        vec![Hlvalue::Constant(reindex_fn), var(&cr_d), var(&cr_size)],
        &new_var("v", LowLevelType::Void),
    );
    block_call_reindex.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `@jit.dont_look_inside` (`ll_dict_rehash_after_translation`).
    func._jit_look_inside_ = Some(false);
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string()],
        func,
    ))
}

/// Synthesise `ll_ensure_indexes(d)` (`rordereddict.py`):
///
/// ```python
/// def ll_ensure_indexes(d):
///     num = d.lookup_function_no
///     if num == FUNC_MUST_REINDEX:
///         ll_dict_create_initial_index(d)
///     else:
///         ll_assert((num & FUNC_MASK) != FUNC_MUST_REINDEX, "...")
/// ```
///
/// The `else` arm's `ll_assert` is a debug-only sanity check with no runtime
/// effect after translation (same omission as
/// [`build_ll_write_indexes_helper_graph`]'s `ll_assert`) — that arm just
/// returns.
pub(crate) fn build_ll_ensure_indexes_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    create_initial_index_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let d = variable_with_lltype("d", dict_ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(d.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let d_for_reindex = variable_with_lltype("d", dict_ptr_lltype);
    let block_reindex = Block::shared(vec![Hlvalue::Variable(d_for_reindex.clone())]);

    let num = variable_with_lltype("num", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(d.clone()),
            void_field_const("lookup_function_no"),
        ],
        Hlvalue::Variable(num.clone()),
    ));
    let is_must_reindex = variable_with_lltype("is_must_reindex", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(num),
            constant_with_lltype(ConstValue::Int(FUNC_MUST_REINDEX), LowLevelType::Signed),
        ],
        Hlvalue::Variable(is_must_reindex.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_must_reindex));
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(d)],
            Some(block_reindex.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![none_const()],
            Some(graph.returnblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    block_reindex
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(create_initial_index_fn),
                Hlvalue::Variable(d_for_reindex),
            ],
            Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
        ));
    block_reindex.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string()],
        func,
    ))
}

/// Synthesise `ll_call_lookup_function(d, key, hash, flag) -> Signed`
/// (`rordereddict.py`):
///
/// ```python
/// def ll_call_lookup_function(d, key, hash, flag):
///     while True:
///         fun = d.lookup_function_no & FUNC_MASK
///         if likely(fun == FUNC_BYTE): return ll_dict_lookup(d, key, hash, flag, TYPE_BYTE)
///         elif fun == FUNC_SHORT: return ll_dict_lookup(d, key, hash, flag, TYPE_SHORT)
///         elif IS_64BIT and fun == FUNC_INT: return ll_dict_lookup(d, key, hash, flag, TYPE_INT)
///         elif fun == FUNC_LONG: return ll_dict_lookup(d, key, hash, flag, TYPE_LONG)
///         else:
///             ll_dict_create_initial_index(d)
///             # then, retry
/// ```
///
/// Every `TYPE_*`/`DICTINDEX_*` width collapses to the same
/// `Ptr(GcArray(Unsigned))` shape locally (see [`DICTINDEX_BYTE`]), so the
/// 4-way `FUNC_*` dispatch always calls the *same*
/// [`build_ll_dict_lookup_helper_graph`] specialization regardless of which
/// selector `ll_ensure_indexes` chose — the `while True` retry loop
/// collapses to "ensure the index exists, then look up once":
/// [`build_ll_ensure_indexes_helper_graph`] already implements the
/// `fun == FUNC_MUST_REINDEX` retry-trigger check, so calling it
/// unconditionally before a single `ll_dict_lookup` call is behaviorally
/// identical to the upstream loop for this port's collapsed width space.
pub(crate) fn build_ll_call_lookup_function_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    ensure_indexes_fn: Constant,
    lookup_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let d = variable_with_lltype("d", dict_ptr_lltype);
    let key = variable_with_lltype("key", key_lltype);
    let hash = variable_with_lltype("hash", LowLevelType::Signed);
    let store_flag = variable_with_lltype("store_flag", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(key.clone()),
        Hlvalue::Variable(hash.clone()),
        Hlvalue::Variable(store_flag.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(ensure_indexes_fn),
            Hlvalue::Variable(d.clone()),
        ],
        Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
    ));
    let index = variable_with_lltype("index", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(lookup_fn),
            Hlvalue::Variable(d),
            Hlvalue::Variable(key),
            Hlvalue::Variable(hash),
            Hlvalue::Variable(store_flag),
            lowlevel_type_const(TYPE_LONG.clone()),
        ],
        Hlvalue::Variable(index.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(index)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `rordereddict.py ll_call_lookup_function`: `@jit.oopspec('ordereddict.lookup(d, key, hash, flag)')`.
    func.oopspec = Some("ordereddict.lookup(d, key, hash, flag)".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "d".to_string(),
            "key".to_string(),
            "hash".to_string(),
            "store_flag".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_dict_getitem(d, key) -> DICTVALUE` (`rordereddict.py`,
/// fused with its `_with_hash` half):
///
/// ```python
/// def ll_dict_getitem(d, key):
///     return ll_dict_getitem_with_hash(d, key, d.keyhash(key))
///
/// def ll_dict_getitem_with_hash(d, key, hash):
///     index = d.lookup_function(d, key, hash, FLAG_LOOKUP)
///     if index >= 0:
///         return d.entries[index].value
///     else:
///         raise KeyError
/// ```
///
/// `d.keyhash`/`d.lookup_function` are RPython adtmeths — resolved once per
/// dict specialization at translation time, not a runtime vtable
/// indirection. This port bakes the specialization's hash helper
/// (`hash_fn`) and `ll_call_lookup_function` (`call_lookup_fn`) into the
/// closure exactly as [`OrderedDictRepr::ll_newdict`] bakes `DICT`, rather
/// than threading them as `Void` consts. Fusing the two upstream functions
/// into one graph (instead of a `direct_call` chain) keeps the `raise
/// KeyError` exceptblock link in the same graph that emits it — the same
/// byte-equivalent-fusion precedent as `build_ll_dict_lookup_helper_graph`'s
/// "Store path inlined" note.
#[expect(
    clippy::too_many_arguments,
    reason = "Parameter order mirrors ll_dict_getitem; hash_takes_dict selects ll_keyhash_custom(d, key)"
)]
pub(crate) fn build_ll_dict_getitem_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    value_lltype: LowLevelType,
    hash_fn: Constant,
    call_lookup_fn: Constant,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let exc_args = exception_args("KeyError")?;

    let d = variable_with_lltype("d", dict_ptr_lltype.clone());
    let key = variable_with_lltype("key", key_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(key.clone()),
    ]);
    let return_var = variable_with_lltype("result", value_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let d_valid = variable_with_lltype("d", dict_ptr_lltype);
    let index_valid = variable_with_lltype("index", LowLevelType::Signed);
    let block_valid = Block::shared(vec![
        Hlvalue::Variable(d_valid.clone()),
        Hlvalue::Variable(index_valid.clone()),
    ]);

    let hash = variable_with_lltype("hash", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        keyhash_call_args(hash_fn, &d, &key, hash_takes_dict),
        Hlvalue::Variable(hash.clone()),
    ));
    let index = variable_with_lltype("index", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(call_lookup_fn),
            Hlvalue::Variable(d.clone()),
            Hlvalue::Variable(key),
            Hlvalue::Variable(hash),
            constant_with_lltype(ConstValue::Int(FLAG_LOOKUP), LowLevelType::Signed),
        ],
        Hlvalue::Variable(index.clone()),
    ));
    let found = variable_with_lltype("found", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![
            Hlvalue::Variable(index.clone()),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(found.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(found));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(d), Hlvalue::Variable(index)],
            Some(block_valid.clone()),
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

    let entries = variable_with_lltype("entries", entries_ptr_lltype);
    block_valid
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(d_valid), void_field_const("entries")],
            Hlvalue::Variable(entries.clone()),
        ));
    let value = variable_with_lltype("value", value_lltype);
    block_valid
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getinteriorfield",
            vec![
                Hlvalue::Variable(entries),
                Hlvalue::Variable(index_valid),
                void_field_const("value"),
            ],
            Hlvalue::Variable(value.clone()),
        ));
    block_valid.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(value)],
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
        vec!["d".to_string(), "key".to_string()],
        func,
    ))
}

/// Synthesise `ll_dict_contains(d, key) -> Bool` (`rordereddict.py`,
/// fused with its `_with_hash` half — same rationale as
/// [`build_ll_dict_getitem_helper_graph`], minus the raise since `contains`
/// never fails for the direct-compare key space this port covers):
///
/// ```python
/// def ll_dict_contains(d, key):
///     return ll_dict_contains_with_hash(d, key, d.keyhash(key))
///
/// def ll_dict_contains_with_hash(d, key, hash):
///     i = d.lookup_function(d, key, hash, FLAG_LOOKUP)
///     return i >= 0
/// ```
pub(crate) fn build_ll_dict_contains_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    hash_fn: Constant,
    call_lookup_fn: Constant,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let d = variable_with_lltype("d", dict_ptr_lltype);
    let key = variable_with_lltype("key", key_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(key.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let hash = variable_with_lltype("hash", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        keyhash_call_args(hash_fn, &d, &key, hash_takes_dict),
        Hlvalue::Variable(hash.clone()),
    ));
    let index = variable_with_lltype("index", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(call_lookup_fn),
            Hlvalue::Variable(d),
            Hlvalue::Variable(key),
            Hlvalue::Variable(hash),
            constant_with_lltype(ConstValue::Int(FLAG_LOOKUP), LowLevelType::Signed),
        ],
        Hlvalue::Variable(index.clone()),
    ));
    let result = variable_with_lltype("result", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![
            Hlvalue::Variable(index),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
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
        vec!["d".to_string(), "key".to_string()],
        func,
    ))
}

/// RPython `pairtype(OrderedDictRepr, rmodel.Repr).rtype_contains`
/// (`rordereddict.py`):
///
/// ```python
/// def rtype_contains((r_dict, r_key), hop):
///     v_dict, v_key = hop.inputargs(r_dict, r_dict.key_repr)
///     hop.exception_is_here()
///     return hop.gendirectcall(ll_dict_contains, v_dict, v_key)
/// ```
pub fn pair_ordereddict_repr_rtype_contains(
    r1: &dyn Repr,
    _r2: &dyn Repr,
    hop: &HighLevelOp,
) -> RTypeResult {
    let any_r: &dyn std::any::Any = r1;
    let r_dict = any_r.downcast_ref::<OrderedDictRepr>().ok_or_else(|| {
        TyperError::message("pair_ordereddict_repr_rtype_contains: r1 is not an OrderedDictRepr")
    })?;

    let args = hop.inputargs(vec![
        ConvertedTo::Repr(r_dict),
        ConvertedTo::Repr(r_dict.base.key_repr.as_ref()),
    ])?;
    hop.exception_is_here()?;

    let chain = r_dict.lookup_chain_helpers(hop)?;
    let dict_ptr = r_dict.lowleveltype().clone();
    let key_lltype = r_dict.DICTKEY.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        "ll_dict_contains",
        vec![dict_ptr.clone(), key_lltype.clone()],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| {
            build_ll_dict_contains_helper_graph(
                "ll_dict_contains",
                dict_ptr.clone(),
                key_lltype.clone(),
                chain.hash_fn_const.clone(),
                chain.call_lookup_const.clone(),
                chain.hash_takes_dict,
            )
        },
    )?;
    hop.gendirectcall(&helper, args)
}

/// Synthesise `ll_dict_store_clean(d, hash, index, T)` (`rordereddict.py`):
///
/// ```python
/// def ll_dict_store_clean(d, hash, index, T):
///     INDEXES = _ll_ptr_to_array_of(T)
///     indexes = lltype.cast_opaque_ptr(INDEXES, d.indexes)
///     mask = len(indexes) - 1
///     i = r_uint(hash & mask)
///     perturb = r_uint(hash)
///     while rffi.cast(lltype.Signed, indexes[i]) != FREE:
///         i = (i << 2) + i + perturb + 1
///         i = i & mask
///         perturb >>= PERTURB_SHIFT
///     _ll_write_indexes(d, i, index + VALID_OFFSET, T)
/// ```
///
/// A simplified `ll_dict_lookup`: no key comparison, no `deletedslot`
/// tracking — the caller (reindex, insert-clean) guarantees the hash is not
/// already present, so the probe only needs to find the first `FREE` slot.
/// Same unsigned-arithmetic idiom as [`build_ll_dict_lookup_helper_graph`]
/// (`i`/`perturb` are `r_uint`; the perturb shift is the *logical*
/// `uint_rshift`). All `DICTINDEX_*` widths collapse to
/// `Ptr(GcArray(Unsigned))` locally, so `T` is inert — this one graph serves
/// every `FUNC_*` width (same collapse as `ll_call_lookup_function`).
/// Calls the real [`build_ll_write_indexes_helper_graph`] helper for the
/// final store (not inlined) — that helper's own doc note already flagged
/// it as reserved for "the non-inlined callers (`ll_dict_store_clean`,
/// reindex) ported in later slices".
pub(crate) fn build_ll_dict_store_clean_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    write_indexes_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let indexes_ptr_lltype = _ll_ptr_to_array_of(LowLevelType::Unsigned);
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let sig = || LowLevelType::Signed;
    let uns = || LowLevelType::Unsigned;
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);

    let d = new_var("d", dict_ptr_lltype.clone());
    let hash = new_var("hash", sig());
    let index = new_var("index", sig());
    let startblock = Block::shared(vec![var(&d), var(&hash), var(&index)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let gcref = new_var("indexes_gcref", GCREF.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("indexes")],
        &gcref,
    );
    let indexes = new_var("indexes", indexes_ptr_lltype.clone());
    // `ll_dict_store_clean`: `indexes = lltype.cast_opaque_ptr(INDEXES, d.indexes)`.
    push(&startblock, "cast_opaque_ptr", vec![var(&gcref)], &indexes);
    let len = new_var("len", sig());
    push(&startblock, "getarraysize", vec![var(&indexes)], &len);
    let mask = new_var("mask", sig());
    push(&startblock, "int_sub", vec![var(&len), signed(1)], &mask);
    let mask_u = new_var("mask_u", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&mask)], &mask_u);
    let hashmask = new_var("hashmask", sig());
    push(
        &startblock,
        "int_and",
        vec![var(&hash), var(&mask)],
        &hashmask,
    );
    let i0 = new_var("i", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&hashmask)], &i0);
    let perturb0 = new_var("perturb", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&hash)], &perturb0);

    // block_cond(d, indexes, mask_u, index, i, perturb).
    let cd_d = new_var("d", dict_ptr_lltype.clone());
    let cd_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let cd_mask_u = new_var("mask_u", uns());
    let cd_index = new_var("index", sig());
    let cd_i = new_var("i", uns());
    let cd_perturb = new_var("perturb", uns());
    let block_cond = Block::shared(vec![
        var(&cd_d),
        var(&cd_indexes),
        var(&cd_mask_u),
        var(&cd_index),
        var(&cd_i),
        var(&cd_perturb),
    ]);

    startblock.closeblock(vec![
        Link::new(
            vec![
                var(&d),
                var(&indexes),
                var(&mask_u),
                var(&index),
                var(&i0),
                var(&perturb0),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_write(d, i_s, index): value = index + VALID_OFFSET; direct_call.
    let wr_d = new_var("d", dict_ptr_lltype.clone());
    let wr_i_s = new_var("i_s", sig());
    let wr_index = new_var("index", sig());
    let block_write = Block::shared(vec![var(&wr_d), var(&wr_i_s), var(&wr_index)]);

    // block_advance(d, indexes, mask_u, index, i, perturb).
    let ad_d = new_var("d", dict_ptr_lltype.clone());
    let ad_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let ad_mask_u = new_var("mask_u", uns());
    let ad_index = new_var("index", sig());
    let ad_i = new_var("i", uns());
    let ad_perturb = new_var("perturb", uns());
    let block_advance = Block::shared(vec![
        var(&ad_d),
        var(&ad_indexes),
        var(&ad_mask_u),
        var(&ad_index),
        var(&ad_i),
        var(&ad_perturb),
    ]);

    // block_cond body: read indexes[i]; branch on == FREE.
    let cd_i_s = new_var("i_s", sig());
    push(&block_cond, "cast_uint_to_int", vec![var(&cd_i)], &cd_i_s);
    let cd_elem = new_var("elem", uns());
    push(
        &block_cond,
        "getarrayitem",
        vec![var(&cd_indexes), var(&cd_i_s)],
        &cd_elem,
    );
    let cd_val = new_var("val", sig());
    push(
        &block_cond,
        "cast_uint_to_int",
        vec![var(&cd_elem)],
        &cd_val,
    );
    let cd_is_free = new_var("is_free", LowLevelType::Bool);
    push(
        &block_cond,
        "int_eq",
        vec![var(&cd_val), signed(FREE)],
        &cd_is_free,
    );
    block_cond.borrow_mut().exitswitch = Some(var(&cd_is_free));
    block_cond.closeblock(vec![
        Link::new(
            vec![var(&cd_d), var(&cd_i_s), var(&cd_index)],
            Some(block_write.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&cd_d),
                var(&cd_indexes),
                var(&cd_mask_u),
                var(&cd_index),
                var(&cd_i),
                var(&cd_perturb),
            ],
            Some(block_advance.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_advance body: i = ((i<<2)+i+perturb+1)&mask_u; perturb >>= PERTURB_SHIFT.
    let ad_ish = new_var("ish", uns());
    push(
        &block_advance,
        "uint_lshift",
        vec![var(&ad_i), signed(2)],
        &ad_ish,
    );
    let ad_ipi = new_var("ipi", uns());
    push(
        &block_advance,
        "uint_add",
        vec![var(&ad_ish), var(&ad_i)],
        &ad_ipi,
    );
    let ad_ipp = new_var("ipp", uns());
    push(
        &block_advance,
        "uint_add",
        vec![var(&ad_ipi), var(&ad_perturb)],
        &ad_ipp,
    );
    let ad_iinc = new_var("iinc", uns());
    push(
        &block_advance,
        "uint_add",
        vec![
            var(&ad_ipp),
            constant_with_lltype(ConstValue::UInt(1), uns()),
        ],
        &ad_iinc,
    );
    let ad_inew = new_var("i", uns());
    push(
        &block_advance,
        "uint_and",
        vec![var(&ad_iinc), var(&ad_mask_u)],
        &ad_inew,
    );
    let ad_pnew = new_var("perturb", uns());
    push(
        &block_advance,
        "uint_rshift",
        vec![var(&ad_perturb), signed(PERTURB_SHIFT)],
        &ad_pnew,
    );
    block_advance.closeblock(vec![
        Link::new(
            vec![
                var(&ad_d),
                var(&ad_indexes),
                var(&ad_mask_u),
                var(&ad_index),
                var(&ad_inew),
                var(&ad_pnew),
            ],
            Some(block_cond),
            None,
        )
        .into_ref(),
    ]);

    // block_write body: _ll_write_indexes(d, i_s, index + VALID_OFFSET).
    let wr_value = new_var("value", sig());
    push(
        &block_write,
        "int_add",
        vec![var(&wr_index), signed(VALID_OFFSET)],
        &wr_value,
    );
    push(
        &block_write,
        "direct_call",
        vec![
            Hlvalue::Constant(write_indexes_fn),
            var(&wr_d),
            var(&wr_i_s),
            var(&wr_value),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_write.closeblock(vec![
        Link::new(
            vec![Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            ))],
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
        vec!["d".to_string(), "hash".to_string(), "index".to_string()],
        func,
    ))
}

/// Synthesise `ll_call_insert_clean_function(d, hash, i)`
/// (`rordereddict.py`):
///
/// ```python
/// def ll_call_insert_clean_function(d, hash, i):
///     fun = d.lookup_function_no & FUNC_MASK
///     if fun == FUNC_BYTE: ll_dict_store_clean(d, hash, i, TYPE_BYTE)
///     elif fun == FUNC_SHORT: ll_dict_store_clean(d, hash, i, TYPE_SHORT)
///     elif IS_64BIT and fun == FUNC_INT: ll_dict_store_clean(d, hash, i, TYPE_INT)
///     elif fun == FUNC_LONG: ll_dict_store_clean(d, hash, i, TYPE_LONG)
///     else: ll_assert(False, "...")
/// ```
///
/// Same width-collapse rationale as `ll_call_lookup_function`'s FUNC_*
/// dispatch: every `TYPE_*`/`DICTINDEX_*` width aliases the same
/// `Ptr(GcArray(Unsigned))` shape locally, so the 4-way dispatch collapses
/// to a single unconditional `ll_dict_store_clean` call. The `ll_assert`
/// dead-fallthrough is debug-only and not modelled.
pub(crate) fn build_ll_call_insert_clean_function_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    store_clean_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let d = variable_with_lltype("d", dict_ptr_lltype);
    let hash = variable_with_lltype("hash", LowLevelType::Signed);
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(hash.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(store_clean_fn),
            Hlvalue::Variable(d),
            Hlvalue::Variable(hash),
            Hlvalue::Variable(i),
        ],
        Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
    ));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `@jit.dont_look_inside` (`ll_call_insert_clean_function`).
    func._jit_look_inside_ = Some(false);
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string(), "hash".to_string(), "i".to_string()],
        func,
    ))
}

/// Synthesise `ll_call_delete_by_entry_index(d, hash, i, replace_with)`
/// (`rordereddict.py`):
///
/// ```python
/// def ll_call_delete_by_entry_index(d, hash, i, replace_with):
///     fun = d.lookup_function_no & FUNC_MASK
///     if fun == FUNC_BYTE: ll_dict_delete_by_entry_index(..., TYPE_BYTE)
///     elif fun == FUNC_SHORT: ll_dict_delete_by_entry_index(..., TYPE_SHORT)
///     elif IS_64BIT and fun == FUNC_INT: ll_dict_delete_by_entry_index(..., TYPE_INT)
///     elif fun == FUNC_LONG: ll_dict_delete_by_entry_index(..., TYPE_LONG)
/// ```
///
/// rordereddict.py — the `FUNC_*` dispatch collapses to one
/// `ll_dict_delete_by_entry_index` call because all `DICTINDEX_*` aliases are
/// currently `Ptr(GcArray(Unsigned))`; same #148-width deviation as
/// [`build_ll_call_insert_clean_function_helper_graph`].
pub(crate) fn build_ll_call_delete_by_entry_index_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    delete_by_entry_index_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let d = variable_with_lltype("d", dict_ptr_lltype);
    let hash = variable_with_lltype("hash", LowLevelType::Signed);
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let replace_with = variable_with_lltype("replace_with", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(hash.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(replace_with.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(delete_by_entry_index_fn),
            Hlvalue::Variable(d),
            Hlvalue::Variable(hash),
            Hlvalue::Variable(i),
            Hlvalue::Variable(replace_with),
        ],
        Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
    ));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "d".to_string(),
            "hash".to_string(),
            "i".to_string(),
            "replace_with".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_dict_delete_by_entry_index(d, hash, locate_index,
/// replace_with, T)` (`rordereddict.py`):
///
/// ```python
/// indexes = lltype.cast_opaque_ptr(INDEXES, d.indexes)
/// locate_value = locate_index + VALID_OFFSET
/// while rffi.cast(lltype.Signed, indexes[i]) != locate_value:
///     i = (i << 2) + i + perturb + 1
///     i = i & mask
///     perturb >>= PERTURB_SHIFT
/// _ll_write_indexes(d, i, replace_with, T)
/// ```
pub(crate) fn build_ll_dict_delete_by_entry_index_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    write_indexes_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let indexes_ptr_lltype = _ll_ptr_to_array_of(LowLevelType::Unsigned);
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let unsigned =
        |n: i64| constant_with_lltype(ConstValue::UInt(n as u64), LowLevelType::Unsigned);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let sig = || LowLevelType::Signed;
    let uns = || LowLevelType::Unsigned;
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);

    let d = new_var("d", dict_ptr_lltype.clone());
    let hash = new_var("hash", sig());
    let locate_index = new_var("locate_index", sig());
    let replace_with = new_var("replace_with", sig());
    let startblock = Block::shared(vec![
        var(&d),
        var(&hash),
        var(&locate_index),
        var(&replace_with),
    ]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let gcref = new_var("indexes_gcref", GCREF.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("indexes")],
        &gcref,
    );
    let indexes = new_var("indexes", indexes_ptr_lltype.clone());
    // `ll_dict_delete_by_entry_index`: `indexes = lltype.cast_opaque_ptr(INDEXES, d.indexes)`.
    push(&startblock, "cast_opaque_ptr", vec![var(&gcref)], &indexes);
    let len = new_var("len", sig());
    push(&startblock, "getarraysize", vec![var(&indexes)], &len);
    let mask = new_var("mask", sig());
    push(&startblock, "int_sub", vec![var(&len), signed(1)], &mask);
    let mask_u = new_var("mask_u", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&mask)], &mask_u);
    let hashmask = new_var("hashmask", sig());
    push(
        &startblock,
        "int_and",
        vec![var(&hash), var(&mask)],
        &hashmask,
    );
    let i0 = new_var("i", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&hashmask)], &i0);
    let perturb0 = new_var("perturb", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&hash)], &perturb0);
    let locate_value = new_var("locate_value", sig());
    push(
        &startblock,
        "int_add",
        vec![var(&locate_index), signed(VALID_OFFSET)],
        &locate_value,
    );

    let cd_d = new_var("d", dict_ptr_lltype.clone());
    let cd_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let cd_mask = new_var("mask_u", uns());
    let cd_replace = new_var("replace_with", sig());
    let cd_locate = new_var("locate_value", sig());
    let cd_i = new_var("i", uns());
    let cd_perturb = new_var("perturb", uns());
    let block_cond = Block::shared(vec![
        var(&cd_d),
        var(&cd_indexes),
        var(&cd_mask),
        var(&cd_replace),
        var(&cd_locate),
        var(&cd_i),
        var(&cd_perturb),
    ]);
    startblock.closeblock(vec![
        Link::new(
            vec![
                var(&d),
                var(&indexes),
                var(&mask_u),
                var(&replace_with),
                var(&locate_value),
                var(&i0),
                var(&perturb0),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let wr_d = new_var("d", dict_ptr_lltype.clone());
    let wr_i_s = new_var("i_s", sig());
    let wr_replace = new_var("replace_with", sig());
    let block_write = Block::shared(vec![var(&wr_d), var(&wr_i_s), var(&wr_replace)]);

    let ad_d = new_var("d", dict_ptr_lltype.clone());
    let ad_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let ad_mask = new_var("mask_u", uns());
    let ad_replace = new_var("replace_with", sig());
    let ad_locate = new_var("locate_value", sig());
    let ad_i = new_var("i", uns());
    let ad_perturb = new_var("perturb", uns());
    let block_advance = Block::shared(vec![
        var(&ad_d),
        var(&ad_indexes),
        var(&ad_mask),
        var(&ad_replace),
        var(&ad_locate),
        var(&ad_i),
        var(&ad_perturb),
    ]);

    let cd_i_s = new_var("i_s", sig());
    push(&block_cond, "cast_uint_to_int", vec![var(&cd_i)], &cd_i_s);
    let cd_elem = new_var("elem", uns());
    push(
        &block_cond,
        "getarrayitem",
        vec![var(&cd_indexes), var(&cd_i_s)],
        &cd_elem,
    );
    let cd_value = new_var("value", sig());
    push(
        &block_cond,
        "cast_uint_to_int",
        vec![var(&cd_elem)],
        &cd_value,
    );
    let found = new_var("found", LowLevelType::Bool);
    push(
        &block_cond,
        "int_eq",
        vec![var(&cd_value), var(&cd_locate)],
        &found,
    );
    block_cond.borrow_mut().exitswitch = Some(var(&found));
    block_cond.closeblock(vec![
        Link::new(
            vec![var(&cd_d), var(&cd_i_s), var(&cd_replace)],
            Some(block_write.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&cd_d),
                var(&cd_indexes),
                var(&cd_mask),
                var(&cd_replace),
                var(&cd_locate),
                var(&cd_i),
                var(&cd_perturb),
            ],
            Some(block_advance.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let ish = new_var("ish", uns());
    push(
        &block_advance,
        "uint_lshift",
        vec![var(&ad_i), signed(2)],
        &ish,
    );
    let ipi = new_var("ipi", uns());
    push(
        &block_advance,
        "uint_add",
        vec![var(&ish), var(&ad_i)],
        &ipi,
    );
    let ipp = new_var("ipp", uns());
    push(
        &block_advance,
        "uint_add",
        vec![var(&ipi), var(&ad_perturb)],
        &ipp,
    );
    let iinc = new_var("iinc", uns());
    push(
        &block_advance,
        "uint_add",
        vec![var(&ipp), unsigned(1)],
        &iinc,
    );
    let inew = new_var("i", uns());
    push(
        &block_advance,
        "uint_and",
        vec![var(&iinc), var(&ad_mask)],
        &inew,
    );
    let perturb_new = new_var("perturb", uns());
    push(
        &block_advance,
        "uint_rshift",
        vec![var(&ad_perturb), signed(PERTURB_SHIFT)],
        &perturb_new,
    );
    block_advance.closeblock(vec![
        Link::new(
            vec![
                var(&ad_d),
                var(&ad_indexes),
                var(&ad_mask),
                var(&ad_replace),
                var(&ad_locate),
                var(&inew),
                var(&perturb_new),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    push(
        &block_write,
        "direct_call",
        vec![
            Hlvalue::Constant(write_indexes_fn),
            var(&wr_d),
            var(&wr_i_s),
            var(&wr_replace),
        ],
        &new_var("v", LowLevelType::Void),
    );
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    block_write.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `@jit.unroll_safe`; `_ll_dict_del`'s `look_inside_iff` decides when the JIT gets here.
    func._jit_unroll_safe_ = true;
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "d".to_string(),
            "hash".to_string(),
            "locate_index".to_string(),
            "replace_with".to_string(),
        ],
        func,
    ))
}

/// Synthesise a struct-array specialisation of `rgc.ll_arraycopy` (`rgc.py`)
/// for the `odictentry` entries array — the same primitive
/// `build_ll_arraycopy_helper_graph` in `rlist.rs` specialises for bare
/// scalar-item `Ptr(GcArray(ITEM))` lists:
///
/// ```python
/// def ll_arraycopy(source, dest, 0, 0, length):
///     i = 0
///     while i < length:
///         dest[i].key = source[i].key
///         dest[i].f_valid = source[i].f_valid
///         dest[i].value = source[i].value
///         if hasattr(ENTRY, 'f_hash'):
///             dest[i].f_hash = source[i].f_hash
///         i += 1
/// ```
///
/// rlist's generic `ll_arraycopy` moves one scalar element per iteration via
/// `getarrayitem`/`setarrayitem`; an array of `odictentry` GcStructs has no
/// single-op "copy this struct element" primitive here, so each field is
/// copied individually via `getinteriorfield`/`setinteriorfield` — a fresh,
/// documented loop rather than a reuse of rlist's helper. Used only by
/// [`build_ll_dict_grow_helper_graph`]'s always-`source_start == dest_start
/// == 0` grow copy, matching rlist's own 3-arg specialisation shape (not the
/// 5-arg general form `ll_extend` needs).
pub(crate) fn build_ll_dict_entries_arraycopy_helper_graph(
    name: &str,
    entries_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    value_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let src = new_var("source", entries_ptr_lltype.clone());
    let dst = new_var("dest", entries_ptr_lltype.clone());
    let length = new_var("length", sig());
    let startblock = Block::shared(vec![var(&src), var(&dst), var(&length)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let src_c = new_var("source", entries_ptr_lltype.clone());
    let dst_c = new_var("dest", entries_ptr_lltype.clone());
    let len_c = new_var("length", sig());
    let i_c = new_var("i", sig());
    let block_cond = Block::shared(vec![var(&src_c), var(&dst_c), var(&len_c), var(&i_c)]);

    let src_b = new_var("source", entries_ptr_lltype.clone());
    let dst_b = new_var("dest", entries_ptr_lltype.clone());
    let len_b = new_var("length", sig());
    let i_b = new_var("i", sig());
    let block_body = Block::shared(vec![var(&src_b), var(&dst_b), var(&len_b), var(&i_b)]);

    startblock.closeblock(vec![
        Link::new(
            vec![var(&src), var(&dst), var(&length), signed(0)],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let cond = new_var("cond", LowLevelType::Bool);
    push(&block_cond, "int_lt", vec![var(&i_c), var(&len_c)], &cond);
    block_cond.borrow_mut().exitswitch = Some(var(&cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![var(&src_c), var(&dst_c), var(&len_c), var(&i_c)],
            Some(block_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![none_void()],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let kv = new_var("kv", key_lltype);
    push(
        &block_body,
        "getinteriorfield",
        vec![var(&src_b), var(&i_b), void_field_const("key")],
        &kv,
    );
    push(
        &block_body,
        "setinteriorfield",
        vec![var(&dst_b), var(&i_b), void_field_const("key"), var(&kv)],
        &new_var("v", LowLevelType::Void),
    );
    let fv = new_var("fv", LowLevelType::Bool);
    push(
        &block_body,
        "getinteriorfield",
        vec![var(&src_b), var(&i_b), void_field_const("f_valid")],
        &fv,
    );
    push(
        &block_body,
        "setinteriorfield",
        vec![
            var(&dst_b),
            var(&i_b),
            void_field_const("f_valid"),
            var(&fv),
        ],
        &new_var("v", LowLevelType::Void),
    );
    let vv = new_var("vv", value_lltype);
    push(
        &block_body,
        "getinteriorfield",
        vec![var(&src_b), var(&i_b), void_field_const("value")],
        &vv,
    );
    push(
        &block_body,
        "setinteriorfield",
        vec![var(&dst_b), var(&i_b), void_field_const("value"), var(&vv)],
        &new_var("v", LowLevelType::Void),
    );
    if dictentry_has_f_hash(&entries_ptr_lltype) {
        let hv = new_var("hv", sig());
        push(
            &block_body,
            "getinteriorfield",
            vec![var(&src_b), var(&i_b), void_field_const("f_hash")],
            &hv,
        );
        push(
            &block_body,
            "setinteriorfield",
            vec![var(&dst_b), var(&i_b), void_field_const("f_hash"), var(&hv)],
            &new_var("v", LowLevelType::Void),
        );
    }
    let i_next = new_var("i", sig());
    push(&block_body, "int_add", vec![var(&i_b), signed(1)], &i_next);
    block_body.closeblock(vec![
        Link::new(
            vec![var(&src_b), var(&dst_b), var(&len_b), var(&i_next)],
            Some(block_cond),
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
        vec![
            "source".to_string(),
            "dest".to_string(),
            "length".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_dict_remove_deleted_items(d)` (`rordereddict.py`):
///
/// ```python
/// if d.num_live_items < len(d.entries) // 4:
///     newitems = malloc(..., _overallocate_entries_len(d.num_live_items))
/// else:
///     newitems = d.entries
/// isrc = idst = 0
/// while isrc < d.num_ever_used_items:
///     if d.entries.valid(isrc):
///         newitems[idst] = d.entries[isrc]
///         idst += 1
///     isrc += 1
/// d.num_ever_used_items = idst
/// if (must_clear_key or must_clear_value) and d.entries == newitems:
///     clear trailing pointer fields
/// else:
///     d.entries = newitems
/// ll_dict_reindex(d, _ll_len_of_d_indexes(d))
/// ```
///
/// `ll_dict_remove_deleted_items` barriers the reused entries array once,
/// before the copy loop, instead of relying on card-by-card write barriers.
pub(crate) fn build_ll_dict_remove_deleted_items_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    entries_array_ty: LowLevelType,
    key_lltype: LowLevelType,
    value_lltype: LowLevelType,
    reindex_fn: Constant,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let must_clear_key = lltype_must_clear_gc_ptr(&key_lltype);
    let must_clear_value = lltype_must_clear_gc_ptr(&value_lltype);
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let null_of = |lltype: &LowLevelType| {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            lltype.clone(),
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let startblock = Block::shared(vec![var(&d)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("entries")],
        &entries,
    );
    let len_entries = new_var("len_entries", sig());
    push(
        &startblock,
        "getarraysize",
        vec![var(&entries)],
        &len_entries,
    );
    let quarter = new_var("quarter", sig());
    push(
        &startblock,
        "int_rshift",
        vec![var(&len_entries), signed(2)],
        &quarter,
    );
    let num_live = new_var("num_live", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("num_live_items")],
        &num_live,
    );
    let shrink = new_var("shrink", LowLevelType::Bool);
    push(
        &startblock,
        "int_lt",
        vec![var(&num_live), var(&quarter)],
        &shrink,
    );
    startblock.borrow_mut().exitswitch = Some(var(&shrink));

    let al_d = new_var("d", dict_ptr_lltype.clone());
    let al_old = new_var("old_entries", entries_ptr_lltype.clone());
    let al_live = new_var("num_live", sig());
    let block_alloc = Block::shared(vec![var(&al_d), var(&al_old), var(&al_live)]);

    let ip_d = new_var("d", dict_ptr_lltype.clone());
    let ip_old = new_var("old_entries", entries_ptr_lltype.clone());
    let block_inplace = Block::shared(vec![var(&ip_d), var(&ip_old)]);

    startblock.closeblock(vec![
        Link::new(
            vec![var(&d), var(&entries), var(&num_live)],
            Some(block_alloc.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d), var(&entries)],
            Some(block_inplace.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let li_d = new_var("d", dict_ptr_lltype.clone());
    let li_old = new_var("old_entries", entries_ptr_lltype.clone());
    let li_new = new_var("newitems", entries_ptr_lltype.clone());
    let li_replace_entries = new_var("replace_entries", LowLevelType::Bool);
    let block_loop_init = Block::shared(vec![
        var(&li_d),
        var(&li_old),
        var(&li_new),
        var(&li_replace_entries),
    ]);

    let live_shr = new_var("live_shr", sig());
    push(
        &block_alloc,
        "int_rshift",
        vec![var(&al_live), signed(3)],
        &live_shr,
    );
    let newsize0 = new_var("newsize0", sig());
    push(
        &block_alloc,
        "int_add",
        vec![var(&al_live), var(&live_shr)],
        &newsize0,
    );
    let new_allocated = new_var("new_allocated", sig());
    push(
        &block_alloc,
        "int_add",
        vec![var(&newsize0), signed(8)],
        &new_allocated,
    );
    let newitems = new_var("newitems", entries_ptr_lltype.clone());
    push(
        &block_alloc,
        "malloc_varsize",
        vec![
            lowlevel_type_const(entries_array_ty),
            gc_flavor_const()?,
            var(&new_allocated),
        ],
        &newitems,
    );
    block_alloc.closeblock(vec![
        Link::new(
            vec![var(&al_d), var(&al_old), var(&newitems), bool_const(true)],
            Some(block_loop_init.clone()),
            None,
        )
        .into_ref(),
    ]);

    // rordereddict.py::ll_dict_remove_deleted_items, the non-shrinking arm:
    // llop.gc_writebarrier(lltype.Void, newitems), where newitems is d.entries.
    let barrier_result = new_var("v", LowLevelType::Void);
    push(
        &block_inplace,
        "gc_writebarrier",
        vec![var(&ip_old)],
        &barrier_result,
    );
    block_inplace.closeblock(vec![
        Link::new(
            vec![var(&ip_d), var(&ip_old), var(&ip_old), bool_const(false)],
            Some(block_loop_init.clone()),
            None,
        )
        .into_ref(),
    ]);

    let isrclimit = new_var("isrclimit", sig());
    push(
        &block_loop_init,
        "getfield",
        vec![var(&li_d), void_field_const("num_ever_used_items")],
        &isrclimit,
    );

    let cd_d = new_var("d", dict_ptr_lltype.clone());
    let cd_old = new_var("old_entries", entries_ptr_lltype.clone());
    let cd_new = new_var("newitems", entries_ptr_lltype.clone());
    let cd_limit = new_var("isrclimit", sig());
    let cd_isrc = new_var("isrc", sig());
    let cd_idst = new_var("idst", sig());
    let cd_replace_entries = new_var("replace_entries", LowLevelType::Bool);
    let block_cond = Block::shared(vec![
        var(&cd_d),
        var(&cd_old),
        var(&cd_new),
        var(&cd_limit),
        var(&cd_isrc),
        var(&cd_idst),
        var(&cd_replace_entries),
    ]);

    block_loop_init.closeblock(vec![
        Link::new(
            vec![
                var(&li_d),
                var(&li_old),
                var(&li_new),
                var(&isrclimit),
                signed(0),
                signed(0),
                var(&li_replace_entries),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let bd_d = new_var("d", dict_ptr_lltype.clone());
    let bd_old = new_var("old_entries", entries_ptr_lltype.clone());
    let bd_new = new_var("newitems", entries_ptr_lltype.clone());
    let bd_limit = new_var("isrclimit", sig());
    let bd_isrc = new_var("isrc", sig());
    let bd_idst = new_var("idst", sig());
    let bd_replace_entries = new_var("replace_entries", LowLevelType::Bool);
    let block_body = Block::shared(vec![
        var(&bd_d),
        var(&bd_old),
        var(&bd_new),
        var(&bd_limit),
        var(&bd_isrc),
        var(&bd_idst),
        var(&bd_replace_entries),
    ]);

    let done_d = new_var("d", dict_ptr_lltype.clone());
    let done_old = new_var("old_entries", entries_ptr_lltype.clone());
    let done_new = new_var("newitems", entries_ptr_lltype.clone());
    let done_limit = new_var("isrclimit", sig());
    let done_idst = new_var("idst", sig());
    let done_replace_entries = new_var("replace_entries", LowLevelType::Bool);
    let block_after_loop = Block::shared(vec![
        var(&done_d),
        var(&done_old),
        var(&done_new),
        var(&done_limit),
        var(&done_idst),
        var(&done_replace_entries),
    ]);

    let cond = new_var("cond", LowLevelType::Bool);
    push(
        &block_cond,
        "int_lt",
        vec![var(&cd_isrc), var(&cd_limit)],
        &cond,
    );
    block_cond.borrow_mut().exitswitch = Some(var(&cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![
                var(&cd_d),
                var(&cd_old),
                var(&cd_new),
                var(&cd_limit),
                var(&cd_isrc),
                var(&cd_idst),
                var(&cd_replace_entries),
            ],
            Some(block_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&cd_d),
                var(&cd_old),
                var(&cd_new),
                var(&cd_limit),
                var(&cd_idst),
                var(&cd_replace_entries),
            ],
            Some(block_after_loop.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let cp_d = new_var("d", dict_ptr_lltype.clone());
    let cp_old = new_var("old_entries", entries_ptr_lltype.clone());
    let cp_new = new_var("newitems", entries_ptr_lltype.clone());
    let cp_limit = new_var("isrclimit", sig());
    let cp_isrc = new_var("isrc", sig());
    let cp_idst = new_var("idst", sig());
    let cp_replace_entries = new_var("replace_entries", LowLevelType::Bool);
    let block_copy = Block::shared(vec![
        var(&cp_d),
        var(&cp_old),
        var(&cp_new),
        var(&cp_limit),
        var(&cp_isrc),
        var(&cp_idst),
        var(&cp_replace_entries),
    ]);

    let nx_d = new_var("d", dict_ptr_lltype.clone());
    let nx_old = new_var("old_entries", entries_ptr_lltype.clone());
    let nx_new = new_var("newitems", entries_ptr_lltype.clone());
    let nx_limit = new_var("isrclimit", sig());
    let nx_isrc = new_var("isrc", sig());
    let nx_idst = new_var("idst", sig());
    let nx_replace_entries = new_var("replace_entries", LowLevelType::Bool);
    let block_next = Block::shared(vec![
        var(&nx_d),
        var(&nx_old),
        var(&nx_new),
        var(&nx_limit),
        var(&nx_isrc),
        var(&nx_idst),
        var(&nx_replace_entries),
    ]);

    let valid = new_var("valid", LowLevelType::Bool);
    push(
        &block_body,
        "getinteriorfield",
        vec![var(&bd_old), var(&bd_isrc), void_field_const("f_valid")],
        &valid,
    );
    block_body.borrow_mut().exitswitch = Some(var(&valid));
    block_body.closeblock(vec![
        Link::new(
            vec![
                var(&bd_d),
                var(&bd_old),
                var(&bd_new),
                var(&bd_limit),
                var(&bd_isrc),
                var(&bd_idst),
                var(&bd_replace_entries),
            ],
            Some(block_copy.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&bd_d),
                var(&bd_old),
                var(&bd_new),
                var(&bd_limit),
                var(&bd_isrc),
                var(&bd_idst),
                var(&bd_replace_entries),
            ],
            Some(block_next.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let kv = new_var("kv", key_lltype.clone());
    push(
        &block_copy,
        "getinteriorfield",
        vec![var(&cp_old), var(&cp_isrc), void_field_const("key")],
        &kv,
    );
    push(
        &block_copy,
        "setinteriorfield",
        vec![
            var(&cp_new),
            var(&cp_idst),
            void_field_const("key"),
            var(&kv),
        ],
        &new_var("v", LowLevelType::Void),
    );
    let vv = new_var("vv", value_lltype.clone());
    push(
        &block_copy,
        "getinteriorfield",
        vec![var(&cp_old), var(&cp_isrc), void_field_const("value")],
        &vv,
    );
    push(
        &block_copy,
        "setinteriorfield",
        vec![
            var(&cp_new),
            var(&cp_idst),
            void_field_const("value"),
            var(&vv),
        ],
        &new_var("v", LowLevelType::Void),
    );
    if dictentry_has_f_hash(&entries_ptr_lltype) {
        let hv = new_var("hv", sig());
        push(
            &block_copy,
            "getinteriorfield",
            vec![var(&cp_old), var(&cp_isrc), void_field_const("f_hash")],
            &hv,
        );
        push(
            &block_copy,
            "setinteriorfield",
            vec![
                var(&cp_new),
                var(&cp_idst),
                void_field_const("f_hash"),
                var(&hv),
            ],
            &new_var("v", LowLevelType::Void),
        );
    }
    push(
        &block_copy,
        "setinteriorfield",
        vec![
            var(&cp_new),
            var(&cp_idst),
            void_field_const("f_valid"),
            bool_const(true),
        ],
        &new_var("v", LowLevelType::Void),
    );
    let idst_next = new_var("idst", sig());
    push(
        &block_copy,
        "int_add",
        vec![var(&cp_idst), signed(1)],
        &idst_next,
    );
    block_copy.closeblock(vec![
        Link::new(
            vec![
                var(&cp_d),
                var(&cp_old),
                var(&cp_new),
                var(&cp_limit),
                var(&cp_isrc),
                var(&idst_next),
                var(&cp_replace_entries),
            ],
            Some(block_next.clone()),
            None,
        )
        .into_ref(),
    ]);

    let isrc_next = new_var("isrc", sig());
    push(
        &block_next,
        "int_add",
        vec![var(&nx_isrc), signed(1)],
        &isrc_next,
    );
    block_next.closeblock(vec![
        Link::new(
            vec![
                var(&nx_d),
                var(&nx_old),
                var(&nx_new),
                var(&nx_limit),
                var(&isrc_next),
                var(&nx_idst),
                var(&nx_replace_entries),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    push(
        &block_after_loop,
        "setfield",
        vec![
            var(&done_d),
            void_field_const("num_ever_used_items"),
            var(&done_idst),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_after_loop.borrow_mut().exitswitch = Some(var(&done_replace_entries));

    let se_d = new_var("d", dict_ptr_lltype.clone());
    let se_new = new_var("newitems", entries_ptr_lltype.clone());
    let block_set_entries = Block::shared(vec![var(&se_d), var(&se_new)]);

    let cl_d = new_var("d", dict_ptr_lltype.clone());
    let cl_new = new_var("newitems", entries_ptr_lltype.clone());
    let cl_limit = new_var("isrclimit", sig());
    let cl_idst = new_var("idst", sig());
    let block_clear_or_reindex = Block::shared(vec![
        var(&cl_d),
        var(&cl_new),
        var(&cl_limit),
        var(&cl_idst),
    ]);

    block_after_loop.closeblock(vec![
        Link::new(
            vec![var(&done_d), var(&done_new)],
            Some(block_set_entries.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&done_d),
                var(&done_new),
                var(&done_limit),
                var(&done_idst),
            ],
            Some(block_clear_or_reindex.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let ri_d = new_var("d", dict_ptr_lltype.clone());
    let block_reindex = Block::shared(vec![var(&ri_d)]);

    push(
        &block_set_entries,
        "setfield",
        vec![var(&se_d), void_field_const("entries"), var(&se_new)],
        &new_var("v", LowLevelType::Void),
    );
    block_set_entries.closeblock(vec![
        Link::new(vec![var(&se_d)], Some(block_reindex.clone()), None).into_ref(),
    ]);

    if must_clear_key || must_clear_value {
        let cc_d = new_var("d", dict_ptr_lltype.clone());
        let cc_entries = new_var("newitems", entries_ptr_lltype.clone());
        let cc_limit = new_var("isrclimit", sig());
        let cc_i = new_var("idst", sig());
        let block_clear_cond = Block::shared(vec![
            var(&cc_d),
            var(&cc_entries),
            var(&cc_limit),
            var(&cc_i),
        ]);

        block_clear_or_reindex.closeblock(vec![
            Link::new(
                vec![var(&cl_d), var(&cl_new), var(&cl_limit), var(&cl_idst)],
                Some(block_clear_cond.clone()),
                None,
            )
            .into_ref(),
        ]);

        let cb_d = new_var("d", dict_ptr_lltype.clone());
        let cb_entries = new_var("newitems", entries_ptr_lltype.clone());
        let cb_limit = new_var("isrclimit", sig());
        let cb_i = new_var("idst", sig());
        let block_clear_body = Block::shared(vec![
            var(&cb_d),
            var(&cb_entries),
            var(&cb_limit),
            var(&cb_i),
        ]);

        let clear_cond = new_var("clear_cond", LowLevelType::Bool);
        push(
            &block_clear_cond,
            "int_lt",
            vec![var(&cc_i), var(&cc_limit)],
            &clear_cond,
        );
        block_clear_cond.borrow_mut().exitswitch = Some(var(&clear_cond));
        block_clear_cond.closeblock(vec![
            Link::new(
                vec![var(&cc_d), var(&cc_entries), var(&cc_limit), var(&cc_i)],
                Some(block_clear_body.clone()),
                Some(bool_const(true)),
            )
            .into_ref(),
            Link::new(
                vec![var(&cc_d)],
                Some(block_reindex.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        ]);

        if must_clear_key {
            push(
                &block_clear_body,
                "setinteriorfield",
                vec![
                    var(&cb_entries),
                    var(&cb_i),
                    void_field_const("key"),
                    null_of(&key_lltype),
                ],
                &new_var("v", LowLevelType::Void),
            );
        }
        if must_clear_value {
            push(
                &block_clear_body,
                "setinteriorfield",
                vec![
                    var(&cb_entries),
                    var(&cb_i),
                    void_field_const("value"),
                    null_of(&value_lltype),
                ],
                &new_var("v", LowLevelType::Void),
            );
        }
        let clear_next = new_var("idst", sig());
        push(
            &block_clear_body,
            "int_add",
            vec![var(&cb_i), signed(1)],
            &clear_next,
        );
        block_clear_body.closeblock(vec![
            Link::new(
                vec![
                    var(&cb_d),
                    var(&cb_entries),
                    var(&cb_limit),
                    var(&clear_next),
                ],
                Some(block_clear_cond),
                None,
            )
            .into_ref(),
        ]);
    } else {
        block_clear_or_reindex.closeblock(vec![
            Link::new(vec![var(&cl_d)], Some(block_reindex.clone()), None).into_ref(),
        ]);
    }

    let indexes_gcref = new_var("indexes_gcref", GCREF.clone());
    push(
        &block_reindex,
        "getfield",
        vec![var(&ri_d), void_field_const("indexes")],
        &indexes_gcref,
    );
    let len_indexes = new_var("len_indexes", sig());
    push(
        &block_reindex,
        "length_of_simple_gcarray_from_opaque",
        vec![var(&indexes_gcref)],
        &len_indexes,
    );
    push(
        &block_reindex,
        "direct_call",
        vec![Hlvalue::Constant(reindex_fn), var(&ri_d), var(&len_indexes)],
        &new_var("v", LowLevelType::Void),
    );
    block_reindex.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `@jit.dont_look_inside` (`ll_dict_remove_deleted_items`).
    func._jit_look_inside_ = Some(false);
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string()],
        func,
    ))
}

/// Synthesise `_ll_dict_entries_size_too_big(d, new_allocated) -> Bool`
/// (`rordereddict.py`):
///
/// ```python
/// def _ll_dict_entries_size_too_big(d, new_allocated):
///     fun = d.lookup_function_no & FUNC_MASK
///     if fun == FUNC_BYTE:
///         assert d.num_live_items < ((1 << 8) - MIN_INDEXES_MINUS_ENTRIES)
///         return new_allocated > ((1 << 8) - MIN_INDEXES_MINUS_ENTRIES)
///     elif fun == FUNC_SHORT:
///         assert d.num_live_items < ((1 << 16) - MIN_INDEXES_MINUS_ENTRIES)
///         return new_allocated > ((1 << 16) - MIN_INDEXES_MINUS_ENTRIES)
///     elif IS_64BIT and fun == FUNC_INT:
///         assert d.num_live_items < ((1 << 32) - MIN_INDEXES_MINUS_ENTRIES)
///         return new_allocated > ((1 << 32) - MIN_INDEXES_MINUS_ENTRIES)
///     return False
/// ```
///
/// The `IS_64BIT and` guard is decided once — the `FUNC_INT` arm is only
/// emitted on a 64-bit target. Each true arm encodes `assert` as a
/// conditional `AssertionError` raise to exceptblock
/// (`build_ll_clear_indexes_helper_graph`), then returns
/// `int_gt(new_allocated, LIMIT)`.
pub(crate) fn build__ll_dict_entries_size_too_big_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let new_allocated = new_var("new_allocated", sig());
    let startblock = Block::shared(vec![var(&d), var(&new_allocated)]);
    let return_var = new_var("result", LowLevelType::Bool);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));
    let returnblock = graph.returnblock.clone();
    let exceptblock = graph.exceptblock.clone();
    // `assert d.num_live_items < LIMIT` (`_ll_dict_entries_size_too_big`):
    // raise AssertionError to exceptblock (`exception_args` + exceptblock,
    // same form as `build_ll_clear_indexes_helper_graph`'s `assert False`).
    let exc_args = exception_args("AssertionError")?;

    let fun0 = new_var("fun0", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("lookup_function_no")],
        &fun0,
    );
    let fun = new_var("fun", sig());
    push(
        &startblock,
        "int_and",
        vec![var(&fun0), signed(FUNC_MASK)],
        &fun,
    );

    let arm_leaf = |limit: i64| -> BlockRef {
        let d = new_var("d", dict_ptr_lltype.clone());
        let new_allocated = new_var("new_allocated", sig());
        let block = Block::shared(vec![var(&d), var(&new_allocated)]);
        let live = new_var("num_live", sig());
        push(
            &block,
            "getfield",
            vec![var(&d), void_field_const("num_live_items")],
            &live,
        );
        let ok = new_var("ok", LowLevelType::Bool);
        push(&block, "int_lt", vec![var(&live), signed(limit)], &ok);
        block.borrow_mut().exitswitch = Some(var(&ok));

        let ret_na = new_var("new_allocated", sig());
        let block_ret = Block::shared(vec![var(&ret_na)]);
        let gt = new_var("gt", LowLevelType::Bool);
        push(&block_ret, "int_gt", vec![var(&ret_na), signed(limit)], &gt);
        block_ret.closeblock(vec![
            Link::new(vec![var(&gt)], Some(returnblock.clone()), None).into_ref(),
        ]);

        block.closeblock(vec![
            Link::new(
                vec![var(&new_allocated)],
                Some(block_ret),
                Some(bool_const(true)),
            )
            .into_ref(),
            Link::new(
                exc_args.clone(),
                Some(exceptblock.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        ]);
        block
    };

    let mut widths: Vec<(i64, i64)> = vec![
        (FUNC_BYTE, (1i64 << 8) - MIN_INDEXES_MINUS_ENTRIES),
        (FUNC_SHORT, (1i64 << 16) - MIN_INDEXES_MINUS_ENTRIES),
    ];
    if IS_64BIT {
        widths.push((FUNC_INT, (1i64 << 32) - MIN_INDEXES_MINUS_ENTRIES));
    }

    // Cascade from the last FUNC_* check back to BYTE so each false-edge
    // target already exists. The last false edge is `return False`.
    let mut false_target: Option<BlockRef> = None;
    for (func_const, limit) in widths.into_iter().rev() {
        let leaf = arm_leaf(limit);
        let cd = new_var("d", dict_ptr_lltype.clone());
        let cna = new_var("new_allocated", sig());
        let cfun = new_var("fun", sig());
        let block = Block::shared(vec![var(&cd), var(&cna), var(&cfun)]);
        let eq = new_var("eq", LowLevelType::Bool);
        push(&block, "int_eq", vec![var(&cfun), signed(func_const)], &eq);
        block.borrow_mut().exitswitch = Some(var(&eq));
        let false_link = match &false_target {
            Some(next) => Link::new(
                vec![var(&cd), var(&cna), var(&cfun)],
                Some(next.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
            None => Link::new(
                vec![bool_const(false)],
                Some(returnblock.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        };
        block.closeblock(vec![
            Link::new(
                vec![var(&cd), var(&cna)],
                Some(leaf),
                Some(bool_const(true)),
            )
            .into_ref(),
            false_link,
        ]);
        false_target = Some(block);
    }

    startblock.closeblock(vec![
        Link::new(
            vec![var(&d), var(&new_allocated), var(&fun)],
            Some(false_target.expect("FUNC_* cascade is non-empty")),
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
        vec!["d".into(), "new_allocated".into()],
        func,
    ))
}

/// Synthesise `ll_dict_grow(d) -> Bool` (`rordereddict.py`):
///
/// ```python
/// def ll_dict_grow(d):
///     if d.num_live_items < d.num_ever_used_items // 2:
///         ll_dict_remove_deleted_items(d)
///         return True
///     new_allocated = _overallocate_entries_len(len(d.entries))
///     if _ll_dict_entries_size_too_big(d, new_allocated):
///         ll_dict_remove_deleted_items(d)
///         assert d.num_live_items == d.num_ever_used_items
///         return True
///     newitems = lltype.malloc(lltype.typeOf(d).TO.entries.TO, new_allocated)
///     rgc.ll_arraycopy(d.entries, newitems, 0, 0, len(d.entries))
///     d.entries = newitems
///     return False
/// ```
pub(crate) fn build_ll_dict_grow_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    entries_array_ty: LowLevelType,
    remove_deleted_items_fn: Constant,
    size_too_big_fn: Constant,
    arraycopy_fn: Constant,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let startblock = Block::shared(vec![var(&d)]);
    let return_var = new_var("result", LowLevelType::Bool);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let num_live = new_var("num_live", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("num_live_items")],
        &num_live,
    );
    let num_ever = new_var("num_ever", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("num_ever_used_items")],
        &num_ever,
    );
    let half_ever = new_var("half_ever", sig());
    push(
        &startblock,
        "int_rshift",
        vec![var(&num_ever), signed(1)],
        &half_ever,
    );
    let do_compact = new_var("do_compact", LowLevelType::Bool);
    push(
        &startblock,
        "int_lt",
        vec![var(&num_live), var(&half_ever)],
        &do_compact,
    );
    startblock.borrow_mut().exitswitch = Some(var(&do_compact));

    let cp_d = new_var("d", dict_ptr_lltype.clone());
    let block_compact = Block::shared(vec![var(&cp_d)]);
    let gr_d = new_var("d", dict_ptr_lltype.clone());
    let block_grow = Block::shared(vec![var(&gr_d)]);

    startblock.closeblock(vec![
        Link::new(
            vec![var(&d)],
            Some(block_compact.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d)],
            Some(block_grow.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    push(
        &block_compact,
        "direct_call",
        vec![
            Hlvalue::Constant(remove_deleted_items_fn.clone()),
            var(&cp_d),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_compact.closeblock(vec![
        Link::new(
            vec![bool_const(true)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &block_grow,
        "getfield",
        vec![var(&gr_d), void_field_const("entries")],
        &entries,
    );
    let len_entries = new_var("len_entries", sig());
    push(
        &block_grow,
        "getarraysize",
        vec![var(&entries)],
        &len_entries,
    );
    // _overallocate_entries_len(baselen) = baselen + (baselen >> 3) + 8.
    let shr = new_var("shr", sig());
    push(
        &block_grow,
        "int_rshift",
        vec![var(&len_entries), signed(3)],
        &shr,
    );
    let newsize = new_var("newsize", sig());
    push(
        &block_grow,
        "int_add",
        vec![var(&len_entries), var(&shr)],
        &newsize,
    );
    let new_allocated = new_var("new_allocated", sig());
    push(
        &block_grow,
        "int_add",
        vec![var(&newsize), signed(8)],
        &new_allocated,
    );
    let too_big = new_var("too_big", LowLevelType::Bool);
    push(
        &block_grow,
        "direct_call",
        vec![
            Hlvalue::Constant(size_too_big_fn),
            var(&gr_d),
            var(&new_allocated),
        ],
        &too_big,
    );
    block_grow.borrow_mut().exitswitch = Some(var(&too_big));

    let tb_d = new_var("d", dict_ptr_lltype.clone());
    let block_too_big = Block::shared(vec![var(&tb_d)]);
    let ml_d = new_var("d", dict_ptr_lltype);
    let ml_entries = new_var("entries", entries_ptr_lltype.clone());
    let ml_len = new_var("len_entries", sig());
    let ml_new_allocated = new_var("new_allocated", sig());
    let block_malloc = Block::shared(vec![
        var(&ml_d),
        var(&ml_entries),
        var(&ml_len),
        var(&ml_new_allocated),
    ]);

    block_grow.closeblock(vec![
        Link::new(
            vec![var(&gr_d)],
            Some(block_too_big.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&gr_d),
                var(&entries),
                var(&len_entries),
                var(&new_allocated),
            ],
            Some(block_malloc.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // `_ll_dict_entries_size_too_big` True: compact, then
    // `assert d.num_live_items == d.num_ever_used_items`, return True.
    let exc_args = exception_args("AssertionError")?;
    push(
        &block_too_big,
        "direct_call",
        vec![Hlvalue::Constant(remove_deleted_items_fn), var(&tb_d)],
        &new_var("v", LowLevelType::Void),
    );
    let tb_live = new_var("num_live", sig());
    push(
        &block_too_big,
        "getfield",
        vec![var(&tb_d), void_field_const("num_live_items")],
        &tb_live,
    );
    let tb_ever = new_var("num_ever", sig());
    push(
        &block_too_big,
        "getfield",
        vec![var(&tb_d), void_field_const("num_ever_used_items")],
        &tb_ever,
    );
    let tb_eq = new_var("eq", LowLevelType::Bool);
    push(
        &block_too_big,
        "int_eq",
        vec![var(&tb_live), var(&tb_ever)],
        &tb_eq,
    );
    block_too_big.borrow_mut().exitswitch = Some(var(&tb_eq));
    block_too_big.closeblock(vec![
        Link::new(
            vec![bool_const(true)],
            Some(graph.returnblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let newitems = new_var("newitems", entries_ptr_lltype);
    push(
        &block_malloc,
        "malloc_varsize",
        vec![
            lowlevel_type_const(entries_array_ty),
            gc_flavor_const()?,
            var(&ml_new_allocated),
        ],
        &newitems,
    );
    push(
        &block_malloc,
        "direct_call",
        vec![
            Hlvalue::Constant(arraycopy_fn),
            var(&ml_entries),
            var(&newitems),
            var(&ml_len),
        ],
        &new_var("v", LowLevelType::Void),
    );
    push(
        &block_malloc,
        "setfield",
        vec![var(&ml_d), void_field_const("entries"), var(&newitems)],
        &new_var("v", LowLevelType::Void),
    );
    block_malloc.closeblock(vec![
        Link::new(
            vec![bool_const(false)],
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
        vec!["d".to_string()],
        func,
    ))
}

/// Mint `ll_arrayclear` then `ll_clear_indexes` and return the latter's
/// funcptr constant. Shared by reindex mint sites that previously inlined
/// the same pair.
fn mint_ll_clear_indexes(
    rtyper: &RPythonTyper,
    dict_ptr: &LowLevelType,
) -> Result<Constant, TyperError> {
    let arrayclear_fn = rtyper.lowlevel_helper_function_with_builder(
        "ll_arrayclear".to_string(),
        vec![_ll_ptr_to_array_of(LowLevelType::Unsigned)],
        LowLevelType::Void,
        move |_r, _a, _res| {
            crate::translator::rtyper::lltypesystem::rbuilder::build_ll_arrayclear_helper_graph(
                "ll_arrayclear",
                LowLevelType::Unsigned,
            )
        },
    )?;
    let arrayclear_const = sub_helper_funcptr_constant(rtyper, &arrayclear_fn)?;
    let dict_ptr = dict_ptr.clone();
    let clear_indexes_fn = rtyper.lowlevel_helper_function_with_builder(
        "ll_clear_indexes",
        vec![dict_ptr.clone(), LowLevelType::Signed],
        LowLevelType::Void,
        move |_rtyper, _args, _result| {
            build_ll_clear_indexes_helper_graph(
                "ll_clear_indexes",
                dict_ptr.clone(),
                arrayclear_const.clone(),
            )
        },
    )?;
    sub_helper_funcptr_constant(rtyper, &clear_indexes_fn)
}

/// Synthesise `ll_clear_indexes(d, n)` (`rordereddict.py`):
///
/// ```python
/// def ll_clear_indexes(d, n):
///     fun = d.lookup_function_no & FUNC_MASK
///     d.lookup_function_no = fun
///     if fun == FUNC_BYTE:
///         rgc.ll_arrayclear(lltype.cast_opaque_ptr(DICTINDEX_BYTE, d.indexes))
///     elif fun == FUNC_SHORT:
///         rgc.ll_arrayclear(lltype.cast_opaque_ptr(DICTINDEX_SHORT, d.indexes))
///     elif IS_64BIT and fun == FUNC_INT:
///         rgc.ll_arrayclear(lltype.cast_opaque_ptr(DICTINDEX_INT, d.indexes))
///     elif fun == FUNC_LONG:
///         rgc.ll_arrayclear(lltype.cast_opaque_ptr(DICTINDEX_LONG, d.indexes))
///     else:
///         assert False
/// ```
///
/// `IS_64BIT` is a Rust compile-time constant here exactly as upstream's
/// module-level `IS_64BIT and` guard is decided once — the `FUNC_INT` arm
/// is only emitted on a 64-bit target. Every `DICTINDEX_*` alias is still
/// `Ptr(GcArray(Unsigned))` (see [`DICTINDEX_BYTE`]); the cascade still
/// names each static so the result type of `cast_opaque_ptr` tracks the
/// width once those aliases split. `n` is unused, as upstream. The if/elif
/// chain is the same block/link construction as
/// [`build_ll_malloc_indexes_and_choose_lookup_helper_graph`].
pub(crate) fn build_ll_clear_indexes_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    arrayclear_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let n = new_var("n", sig());
    let startblock = Block::shared(vec![var(&d), var(&n)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));
    let returnblock = graph.returnblock.clone();
    // `assert False` (`ll_clear_indexes`): raise AssertionError to
    // exceptblock (`exception_args` + exceptblock, same form as
    // `build_ll_dict_getitem_helper_graph`'s `raise KeyError`).
    let exc_args = exception_args("AssertionError")?;

    let fun0 = new_var("fun0", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("lookup_function_no")],
        &fun0,
    );
    let fun = new_var("fun", sig());
    push(
        &startblock,
        "int_and",
        vec![var(&fun0), signed(FUNC_MASK)],
        &fun,
    );
    push(
        &startblock,
        "setfield",
        vec![var(&d), void_field_const("lookup_function_no"), var(&fun)],
        &new_var("v", LowLevelType::Void),
    );

    // Shared leaf: `rgc.ll_arrayclear(lltype.cast_opaque_ptr(DICTINDEX_X, d.indexes))`.
    // The `direct_call ll_arrayclear` is the same clear lowering the previous
    // single-width builder emitted; only the `cast_opaque_ptr` result type
    // differs branch to branch.
    let clear_leaf = |index_ty: LowLevelType| -> BlockRef {
        let d = new_var("d", dict_ptr_lltype.clone());
        let block = Block::shared(vec![var(&d)]);
        let indexes_gcref = new_var("indexes_gcref", GCREF.clone());
        push(
            &block,
            "getfield",
            vec![var(&d), void_field_const("indexes")],
            &indexes_gcref,
        );
        let indexes = new_var("indexes", index_ty);
        push(
            &block,
            "cast_opaque_ptr",
            vec![var(&indexes_gcref)],
            &indexes,
        );
        push(
            &block,
            "direct_call",
            vec![Hlvalue::Constant(arrayclear_fn.clone()), var(&indexes)],
            &new_var("v", LowLevelType::Void),
        );
        block.closeblock(vec![
            Link::new(vec![none_void()], Some(returnblock.clone()), None).into_ref(),
        ]);
        block
    };

    let mut widths: Vec<(i64, LowLevelType)> = vec![
        (FUNC_BYTE, DICTINDEX_BYTE.clone()),
        (FUNC_SHORT, DICTINDEX_SHORT.clone()),
    ];
    if IS_64BIT {
        widths.push((FUNC_INT, DICTINDEX_INT.clone()));
    }
    widths.push((FUNC_LONG, DICTINDEX_LONG.clone()));

    // Cascade from LONG back to BYTE so each check's false-edge target
    // already exists when the block closes. The LONG false edge is
    // `assert False`.
    let mut false_target: Option<BlockRef> = None;
    for (func_const, index_ty) in widths.into_iter().rev() {
        let leaf = clear_leaf(index_ty);
        let cd = new_var("d", dict_ptr_lltype.clone());
        let cfun = new_var("fun", sig());
        let block = Block::shared(vec![var(&cd), var(&cfun)]);
        let eq = new_var("eq", LowLevelType::Bool);
        push(&block, "int_eq", vec![var(&cfun), signed(func_const)], &eq);
        block.borrow_mut().exitswitch = Some(var(&eq));
        let false_link = match &false_target {
            Some(next) => Link::new(
                vec![var(&cd), var(&cfun)],
                Some(next.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
            None => Link::new(
                exc_args.clone(),
                Some(graph.exceptblock.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        };
        block.closeblock(vec![
            Link::new(vec![var(&cd)], Some(leaf), Some(bool_const(true))).into_ref(),
            false_link,
        ]);
        false_target = Some(block);
    }

    startblock.closeblock(vec![
        Link::new(
            vec![var(&d), var(&fun)],
            Some(false_target.expect("FUNC_* cascade is non-empty")),
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
        vec!["d".into(), "n".into()],
        func,
    ))
}

/// Synthesise `ll_dict_reindex(d, new_size)` (`rordereddict.py`):
///
/// ```python
/// def ll_dict_reindex(d, new_size):
///     if bool(d.indexes) and _ll_len_of_d_indexes(d) == new_size:
///         ll_clear_indexes(d, new_size)
///     else:
///         ll_malloc_indexes_and_choose_lookup(d, new_size)
///     d.resize_counter = new_size * 2 - d.num_live_items * 3
///     entries = d.entries
///     i = 0
///     ibound = d.num_ever_used_items
///     fun = d.lookup_function_no
///     while i < ibound:
///         if entries.valid(i):
///             ll_dict_store_clean(d, entries.entry_hash(d, i), i, TYPE_*)
///         i += 1
/// ```
///
/// `_ll_len_of_d_indexes(d)` (`rordereddict.py`):
///
/// ```python
/// def _ll_len_of_d_indexes(d):
///     # Must only be called by @jit.dont_look_inside functions.
///     return lltype.length_of_simple_gcarray_from_opaque(d.indexes)
/// ```
///
/// The reuse branch clears the existing array through `ll_clear_indexes`, so
/// `_ll_dict_rescue`'s call with `_ll_len_of_d_indexes(d)` allocates nothing.
/// The `fun`/`FUNC_*` 4-way dispatch in the loop collapses to a single
/// `ll_dict_store_clean` call (same width-collapse rationale as
/// `ll_call_lookup_function`, Slice 2). `entries.entry_hash(d, i)` is
/// `ll_hash_from_cache` when `hasattr(ENTRY, 'f_hash')`, else
/// `ll_hash_custom_fast` (`get_ll_dict` for `simple_hash_eq`): `hlinvoke` of
/// `d.fnkeyhash` on `entries[i].key`. `entries.valid(i)` reads `f_valid`
/// (`ll_valid_from_flag`).
#[expect(
    clippy::too_many_arguments,
    reason = "Mirrors ll_dict_reindex operands plus the entry_hash keyhash path"
)]
pub(crate) fn build_ll_dict_reindex_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    malloc_choose_fn: Constant,
    clear_indexes_fn: Constant,
    store_clean_fn: Constant,
    keyhash_fn: Option<Constant>,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let new_size = new_var("new_size", sig());
    let startblock = Block::shared(vec![var(&d), var(&new_size)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    // len_block(d, new_size, indexes_gcref).
    let ln_d = new_var("d", dict_ptr_lltype.clone());
    let ln_size = new_var("new_size", sig());
    let ln_gcref = new_var("indexes_gcref", GCREF.clone());
    let len_block = Block::shared(vec![var(&ln_d), var(&ln_size), var(&ln_gcref)]);

    // clear_block(d, new_size).
    let cl_d = new_var("d", dict_ptr_lltype.clone());
    let cl_size = new_var("new_size", sig());
    let clear_block = Block::shared(vec![var(&cl_d), var(&cl_size)]);

    // malloc_block(d, new_size).
    let ml_d = new_var("d", dict_ptr_lltype.clone());
    let ml_size = new_var("new_size", sig());
    let malloc_block = Block::shared(vec![var(&ml_d), var(&ml_size)]);

    // body_block(d, new_size): resize_counter store + loop setup.
    let bd0_d = new_var("d", dict_ptr_lltype.clone());
    let bd0_size = new_var("new_size", sig());
    let body_block = Block::shared(vec![var(&bd0_d), var(&bd0_size)]);

    let indexes_gcref = new_var("indexes_gcref", GCREF.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("indexes")],
        &indexes_gcref,
    );
    let has_indexes = new_var("has_indexes", LowLevelType::Bool);
    push(
        &startblock,
        "ptr_nonzero",
        vec![var(&indexes_gcref)],
        &has_indexes,
    );
    startblock.borrow_mut().exitswitch = Some(var(&has_indexes));
    startblock.closeblock(vec![
        Link::new(
            vec![var(&d), var(&new_size), var(&indexes_gcref)],
            Some(len_block.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d), var(&new_size)],
            Some(malloc_block.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let n = new_var("n", sig());
    push(
        &len_block,
        "length_of_simple_gcarray_from_opaque",
        vec![var(&ln_gcref)],
        &n,
    );
    let same = new_var("same", LowLevelType::Bool);
    push(&len_block, "int_eq", vec![var(&n), var(&ln_size)], &same);
    len_block.borrow_mut().exitswitch = Some(var(&same));
    len_block.closeblock(vec![
        Link::new(
            vec![var(&ln_d), var(&ln_size)],
            Some(clear_block.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&ln_d), var(&ln_size)],
            Some(malloc_block.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    push(
        &clear_block,
        "direct_call",
        vec![
            Hlvalue::Constant(clear_indexes_fn),
            var(&cl_d),
            var(&cl_size),
        ],
        &new_var("v", LowLevelType::Void),
    );
    clear_block.closeblock(vec![
        Link::new(
            vec![var(&cl_d), var(&cl_size)],
            Some(body_block.clone()),
            None,
        )
        .into_ref(),
    ]);

    push(
        &malloc_block,
        "direct_call",
        vec![
            Hlvalue::Constant(malloc_choose_fn),
            var(&ml_d),
            var(&ml_size),
        ],
        &new_var("v", LowLevelType::Void),
    );
    malloc_block.closeblock(vec![
        Link::new(
            vec![var(&ml_d), var(&ml_size)],
            Some(body_block.clone()),
            None,
        )
        .into_ref(),
    ]);

    let t1 = new_var("t1", sig());
    push(&body_block, "int_mul", vec![var(&bd0_size), signed(2)], &t1);
    let num_live = new_var("num_live", sig());
    push(
        &body_block,
        "getfield",
        vec![var(&bd0_d), void_field_const("num_live_items")],
        &num_live,
    );
    let t2 = new_var("t2", sig());
    push(&body_block, "int_mul", vec![var(&num_live), signed(3)], &t2);
    let rc = new_var("rc", sig());
    push(&body_block, "int_sub", vec![var(&t1), var(&t2)], &rc);
    push(
        &body_block,
        "setfield",
        vec![var(&bd0_d), void_field_const("resize_counter"), var(&rc)],
        &new_var("v", LowLevelType::Void),
    );
    let entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &body_block,
        "getfield",
        vec![var(&bd0_d), void_field_const("entries")],
        &entries,
    );
    let ibound = new_var("ibound", sig());
    push(
        &body_block,
        "getfield",
        vec![var(&bd0_d), void_field_const("num_ever_used_items")],
        &ibound,
    );

    // block_cond(d, entries, ibound, i).
    let cd_d = new_var("d", dict_ptr_lltype.clone());
    let cd_entries = new_var("entries", entries_ptr_lltype.clone());
    let cd_ibound = new_var("ibound", sig());
    let cd_i = new_var("i", sig());
    let block_cond = Block::shared(vec![
        var(&cd_d),
        var(&cd_entries),
        var(&cd_ibound),
        var(&cd_i),
    ]);

    body_block.closeblock(vec![
        Link::new(
            vec![var(&bd0_d), var(&entries), var(&ibound), signed(0)],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_body(d, entries, ibound, i): check f_valid.
    let bd_d = new_var("d", dict_ptr_lltype.clone());
    let bd_entries = new_var("entries", entries_ptr_lltype.clone());
    let bd_ibound = new_var("ibound", sig());
    let bd_i = new_var("i", sig());
    let block_body = Block::shared(vec![
        var(&bd_d),
        var(&bd_entries),
        var(&bd_ibound),
        var(&bd_i),
    ]);

    let cond = new_var("cond", LowLevelType::Bool);
    push(
        &block_cond,
        "int_lt",
        vec![var(&cd_i), var(&cd_ibound)],
        &cond,
    );
    block_cond.borrow_mut().exitswitch = Some(var(&cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![var(&cd_d), var(&cd_entries), var(&cd_ibound), var(&cd_i)],
            Some(block_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![none_void()],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_store(d, entries, ibound, i): entry_hash + store_clean.
    let st_d = new_var("d", dict_ptr_lltype.clone());
    let st_entries = new_var("entries", entries_ptr_lltype.clone());
    let st_ibound = new_var("ibound", sig());
    let st_i = new_var("i", sig());
    let block_store = Block::shared(vec![
        var(&st_d),
        var(&st_entries),
        var(&st_ibound),
        var(&st_i),
    ]);

    // block_next(d, entries, ibound, i).
    let nx_d = new_var("d", dict_ptr_lltype.clone());
    let nx_entries = new_var("entries", entries_ptr_lltype.clone());
    let nx_ibound = new_var("ibound", sig());
    let nx_i = new_var("i", sig());
    let block_next = Block::shared(vec![
        var(&nx_d),
        var(&nx_entries),
        var(&nx_ibound),
        var(&nx_i),
    ]);

    let valid = new_var("valid", LowLevelType::Bool);
    push(
        &block_body,
        "getinteriorfield",
        vec![var(&bd_entries), var(&bd_i), void_field_const("f_valid")],
        &valid,
    );
    block_body.borrow_mut().exitswitch = Some(var(&valid));
    block_body.closeblock(vec![
        Link::new(
            vec![var(&bd_d), var(&bd_entries), var(&bd_ibound), var(&bd_i)],
            Some(block_store.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&bd_d), var(&bd_entries), var(&bd_ibound), var(&bd_i)],
            Some(block_next.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let h = new_var("h", sig());
    if dictentry_has_f_hash(&entries_ptr_lltype) {
        push(
            &block_store,
            "getinteriorfield",
            vec![var(&st_entries), var(&st_i), void_field_const("f_hash")],
            &h,
        );
    } else {
        // ll_hash_custom_fast: key = entries[i].key; hlinvoke hash(key)
        let key_ll = dictentry_key_lltype(&entries_ptr_lltype).ok_or_else(|| {
            TyperError::message("ll_dict_reindex: ll_hash_custom_fast needs ENTRY.key")
        })?;
        let entry_key = new_var("entry_key", key_ll);
        push(
            &block_store,
            "getinteriorfield",
            vec![var(&st_entries), var(&st_i), void_field_const("key")],
            &entry_key,
        );
        let hf = keyhash_fn.clone().ok_or_else(|| {
            TyperError::message(
                "ll_dict_reindex: entry_hash needs f_hash or keyhash (ll_hash_custom_fast)",
            )
        })?;
        push(
            &block_store,
            "direct_call",
            keyhash_call_args(hf, &st_d, &entry_key, hash_takes_dict),
            &h,
        );
    }
    push(
        &block_store,
        "direct_call",
        vec![
            Hlvalue::Constant(store_clean_fn),
            var(&st_d),
            var(&h),
            var(&st_i),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_store.closeblock(vec![
        Link::new(
            vec![var(&st_d), var(&st_entries), var(&st_ibound), var(&st_i)],
            Some(block_next.clone()),
            None,
        )
        .into_ref(),
    ]);

    let i_next = new_var("i", sig());
    push(&block_next, "int_add", vec![var(&nx_i), signed(1)], &i_next);
    block_next.closeblock(vec![
        Link::new(
            vec![var(&nx_d), var(&nx_entries), var(&nx_ibound), var(&i_next)],
            Some(block_cond),
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
        vec!["d".to_string(), "new_size".to_string()],
        func,
    ))
}

/// Synthesise `_ll_dict_resize_to(d, num_extra)` (`rordereddict.py`):
///
/// ```python
/// def _ll_dict_resize_to(d, num_extra):
///     new_estimate = (d.num_live_items + num_extra) * 2
///     new_size = DICT_INITSIZE
///     while new_size <= new_estimate:
///         new_size *= 2
///     if new_size < _ll_len_of_d_indexes(d):
///         ll_dict_remove_deleted_items(d)
///     else:
///         ll_dict_reindex(d, new_size)
/// ```
pub(crate) fn build_ll_dict_resize_to_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    remove_deleted_items_fn: Constant,
    reindex_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let num_extra = new_var("num_extra", sig());
    let startblock = Block::shared(vec![var(&d), var(&num_extra)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let num_live = new_var("num_live", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("num_live_items")],
        &num_live,
    );
    let sum1 = new_var("sum1", sig());
    push(
        &startblock,
        "int_add",
        vec![var(&num_live), var(&num_extra)],
        &sum1,
    );
    let new_estimate = new_var("new_estimate", sig());
    push(
        &startblock,
        "int_mul",
        vec![var(&sum1), signed(2)],
        &new_estimate,
    );

    // block_double_cond(d, new_estimate, new_size).
    let dc_d = new_var("d", dict_ptr_lltype.clone());
    let dc_estimate = new_var("new_estimate", sig());
    let dc_size = new_var("new_size", sig());
    let block_double_cond = Block::shared(vec![var(&dc_d), var(&dc_estimate), var(&dc_size)]);

    startblock.closeblock(vec![
        Link::new(
            vec![var(&d), var(&new_estimate), signed(DICT_INITSIZE)],
            Some(block_double_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_double_body(d, new_estimate, new_size): new_size *= 2.
    let db_d = new_var("d", dict_ptr_lltype.clone());
    let db_estimate = new_var("new_estimate", sig());
    let db_size = new_var("new_size", sig());
    let block_double_body = Block::shared(vec![var(&db_d), var(&db_estimate), var(&db_size)]);

    // block_check_shrink(d, new_size): compare with len(d.indexes).
    let cs_d = new_var("d", dict_ptr_lltype.clone());
    let cs_size = new_var("new_size", sig());
    let block_check_shrink = Block::shared(vec![var(&cs_d), var(&cs_size)]);

    // block_call_remove(d): ll_dict_remove_deleted_items(d).
    let rm_d = new_var("d", dict_ptr_lltype.clone());
    let block_call_remove = Block::shared(vec![var(&rm_d)]);

    // block_call_reindex(d, new_size).
    let cr_d = new_var("d", dict_ptr_lltype.clone());
    let cr_size = new_var("new_size", sig());
    let block_call_reindex = Block::shared(vec![var(&cr_d), var(&cr_size)]);

    let cond = new_var("cond", LowLevelType::Bool);
    push(
        &block_double_cond,
        "int_le",
        vec![var(&dc_size), var(&dc_estimate)],
        &cond,
    );
    block_double_cond.borrow_mut().exitswitch = Some(var(&cond));
    block_double_cond.closeblock(vec![
        Link::new(
            vec![var(&dc_d), var(&dc_estimate), var(&dc_size)],
            Some(block_double_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&dc_d), var(&dc_size)],
            Some(block_check_shrink.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let db_size2 = new_var("new_size", sig());
    push(
        &block_double_body,
        "int_mul",
        vec![var(&db_size), signed(2)],
        &db_size2,
    );
    block_double_body.closeblock(vec![
        Link::new(
            vec![var(&db_d), var(&db_estimate), var(&db_size2)],
            Some(block_double_cond),
            None,
        )
        .into_ref(),
    ]);

    let indexes_gcref = new_var("indexes_gcref", GCREF.clone());
    push(
        &block_check_shrink,
        "getfield",
        vec![var(&cs_d), void_field_const("indexes")],
        &indexes_gcref,
    );
    let len_indexes = new_var("len_indexes", sig());
    push(
        &block_check_shrink,
        "length_of_simple_gcarray_from_opaque",
        vec![var(&indexes_gcref)],
        &len_indexes,
    );
    let shrink = new_var("shrink", LowLevelType::Bool);
    push(
        &block_check_shrink,
        "int_lt",
        vec![var(&cs_size), var(&len_indexes)],
        &shrink,
    );
    block_check_shrink.borrow_mut().exitswitch = Some(var(&shrink));
    block_check_shrink.closeblock(vec![
        Link::new(
            vec![var(&cs_d)],
            Some(block_call_remove.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&cs_d), var(&cs_size)],
            Some(block_call_reindex.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    push(
        &block_call_remove,
        "direct_call",
        vec![Hlvalue::Constant(remove_deleted_items_fn), var(&rm_d)],
        &new_var("v", LowLevelType::Void),
    );
    block_call_remove.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    push(
        &block_call_reindex,
        "direct_call",
        vec![Hlvalue::Constant(reindex_fn), var(&cr_d), var(&cr_size)],
        &new_var("v", LowLevelType::Void),
    );
    block_call_reindex.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string(), "num_extra".to_string()],
        func,
    ))
}

/// Synthesise `ll_dict_resize(d)` (`rordereddict.py`):
///
/// ```python
/// def ll_dict_resize(d):
///     num_extra = min(d.num_live_items + 1, 30000)
///     _ll_dict_resize_to(d, num_extra)
/// ```
pub(crate) fn build_ll_dict_resize_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    resize_to_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let startblock = Block::shared(vec![var(&d)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let num_live = new_var("num_live", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("num_live_items")],
        &num_live,
    );
    let plus1 = new_var("plus1", sig());
    push(
        &startblock,
        "int_add",
        vec![var(&num_live), signed(1)],
        &plus1,
    );
    let lt = new_var("lt", LowLevelType::Bool);
    push(&startblock, "int_lt", vec![var(&plus1), signed(30000)], &lt);
    startblock.borrow_mut().exitswitch = Some(var(&lt));

    // block_call(d, num_extra) — shared tail for both arms of min(...).
    let cl_d = new_var("d", dict_ptr_lltype.clone());
    let cl_num_extra = new_var("num_extra", sig());
    let block_call = Block::shared(vec![var(&cl_d), var(&cl_num_extra)]);

    startblock.closeblock(vec![
        Link::new(
            vec![var(&d), var(&plus1)],
            Some(block_call.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d), signed(30000)],
            Some(block_call.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    push(
        &block_call,
        "direct_call",
        vec![
            Hlvalue::Constant(resize_to_fn),
            var(&cl_d),
            var(&cl_num_extra),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_call.closeblock(vec![
        Link::new(
            vec![Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            ))],
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
        vec!["d".to_string()],
        func,
    ))
}

/// Synthesise `_ll_dict_del_entry(d, index)` (`rordereddict.py`):
///
/// ```python
/// d.entries.mark_deleted(index)
/// d.num_live_items -= 1
/// if ENTRIES.must_clear_key: entry.key = nullptr(...)
/// if ENTRIES.must_clear_value: entry.value = nullptr(...)
/// ```
pub(crate) fn build_ll_dict_del_entry_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    value_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype);
    let index = new_var("index", sig());
    let startblock = Block::shared(vec![var(&d), var(&index)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("entries")],
        &entries,
    );
    push(
        &startblock,
        "setinteriorfield",
        vec![
            var(&entries),
            var(&index),
            void_field_const("f_valid"),
            bool_const(false),
        ],
        &new_var("v", LowLevelType::Void),
    );
    let num_live = new_var("num_live", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("num_live_items")],
        &num_live,
    );
    let live_after = new_var("live_after", sig());
    push(
        &startblock,
        "int_sub",
        vec![var(&num_live), signed(1)],
        &live_after,
    );
    push(
        &startblock,
        "setfield",
        vec![
            var(&d),
            void_field_const("num_live_items"),
            var(&live_after),
        ],
        &new_var("v", LowLevelType::Void),
    );

    // rordereddict.py:879-882 — conditional GC pointer clears map to local
    // pointer low-level types; non-pointer key/value fields need no clear.
    if lltype_must_clear_gc_ptr(&key_lltype) {
        let null_key = Hlvalue::Constant(Constant::with_concretetype(ConstValue::None, key_lltype));
        push(
            &startblock,
            "setinteriorfield",
            vec![
                var(&entries),
                var(&index),
                void_field_const("key"),
                null_key,
            ],
            &new_var("v", LowLevelType::Void),
        );
    }
    if lltype_must_clear_gc_ptr(&value_lltype) {
        let null_value =
            Hlvalue::Constant(Constant::with_concretetype(ConstValue::None, value_lltype));
        push(
            &startblock,
            "setinteriorfield",
            vec![
                var(&entries),
                var(&index),
                void_field_const("value"),
                null_value,
            ],
            &new_var("v", LowLevelType::Void),
        );
    }
    startblock.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string(), "index".to_string()],
        func,
    ))
}

/// Synthesise `_ll_dict_del(d, hash, index)` (`rordereddict.py`):
///
/// ```python
/// ll_call_delete_by_entry_index(d, hash, index, DELETED)
/// _ll_dict_del_entry(d, index)
/// if d.num_live_items == 0: reset num_ever + lookup_function_no high bits
/// elif index == d.num_ever_used_items - 1: reclaim trailing deleted entries
/// if d.num_live_items + DICT_INITSIZE <= len(d.entries) / 8: ll_dict_resize(d)
/// ```
pub(crate) fn build_ll_dict_del_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    call_delete_fn: Constant,
    del_entry_fn: Constant,
    resize_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let hash = new_var("hash", sig());
    let index = new_var("index", sig());
    let startblock = Block::shared(vec![var(&d), var(&hash), var(&index)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let empty_d = new_var("d", dict_ptr_lltype.clone());
    let block_empty = Block::shared(vec![var(&empty_d)]);
    let lc_d = new_var("d", dict_ptr_lltype.clone());
    let lc_index = new_var("index", sig());
    let block_last_check = Block::shared(vec![var(&lc_d), var(&lc_index)]);
    let sh_d = new_var("d", dict_ptr_lltype.clone());
    let block_shrink_check = Block::shared(vec![var(&sh_d)]);

    push(
        &startblock,
        "direct_call",
        vec![
            Hlvalue::Constant(call_delete_fn),
            var(&d),
            var(&hash),
            var(&index),
            signed(DELETED),
        ],
        &new_var("v", LowLevelType::Void),
    );
    push(
        &startblock,
        "direct_call",
        vec![Hlvalue::Constant(del_entry_fn), var(&d), var(&index)],
        &new_var("v", LowLevelType::Void),
    );
    let num_live = new_var("num_live", sig());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("num_live_items")],
        &num_live,
    );
    let is_empty = new_var("is_empty", LowLevelType::Bool);
    push(
        &startblock,
        "int_eq",
        vec![var(&num_live), signed(0)],
        &is_empty,
    );
    startblock.borrow_mut().exitswitch = Some(var(&is_empty));
    startblock.closeblock(vec![
        Link::new(
            vec![var(&d)],
            Some(block_empty.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d), var(&index)],
            Some(block_last_check.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    push(
        &block_empty,
        "setfield",
        vec![
            var(&empty_d),
            void_field_const("num_ever_used_items"),
            signed(0),
        ],
        &new_var("v", LowLevelType::Void),
    );
    let lookup_no = new_var("lookup_no", sig());
    push(
        &block_empty,
        "getfield",
        vec![var(&empty_d), void_field_const("lookup_function_no")],
        &lookup_no,
    );
    let lookup_masked = new_var("lookup_masked", sig());
    push(
        &block_empty,
        "int_and",
        vec![var(&lookup_no), signed(FUNC_MASK)],
        &lookup_masked,
    );
    push(
        &block_empty,
        "setfield",
        vec![
            var(&empty_d),
            void_field_const("lookup_function_no"),
            var(&lookup_masked),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_empty.closeblock(vec![
        Link::new(vec![var(&empty_d)], Some(block_shrink_check.clone()), None).into_ref(),
    ]);

    let num_ever = new_var("num_ever", sig());
    push(
        &block_last_check,
        "getfield",
        vec![var(&lc_d), void_field_const("num_ever_used_items")],
        &num_ever,
    );
    let last = new_var("last", sig());
    push(
        &block_last_check,
        "int_sub",
        vec![var(&num_ever), signed(1)],
        &last,
    );
    let is_last = new_var("is_last", LowLevelType::Bool);
    push(
        &block_last_check,
        "int_eq",
        vec![var(&lc_index), var(&last)],
        &is_last,
    );
    block_last_check.borrow_mut().exitswitch = Some(var(&is_last));

    let sc_d = new_var("d", dict_ptr_lltype.clone());
    let sc_i = new_var("i", sig());
    let block_scan = Block::shared(vec![var(&sc_d), var(&sc_i)]);
    block_last_check.closeblock(vec![
        Link::new(
            vec![var(&lc_d), var(&lc_index)],
            Some(block_scan.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&lc_d)],
            Some(block_shrink_check.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // rordereddict.py:899 — the `assert i >= 0` in the trailing-delete scan
    // is debug-only; the non-empty branch guarantees one valid entry remains.
    let i_prev = new_var("i", sig());
    push(&block_scan, "int_sub", vec![var(&sc_i), signed(1)], &i_prev);
    let entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &block_scan,
        "getfield",
        vec![var(&sc_d), void_field_const("entries")],
        &entries,
    );
    let valid = new_var("valid", LowLevelType::Bool);
    push(
        &block_scan,
        "getinteriorfield",
        vec![var(&entries), var(&i_prev), void_field_const("f_valid")],
        &valid,
    );
    block_scan.borrow_mut().exitswitch = Some(var(&valid));

    let se_d = new_var("d", dict_ptr_lltype.clone());
    let se_i = new_var("i", sig());
    let block_set_ever = Block::shared(vec![var(&se_d), var(&se_i)]);
    block_scan.closeblock(vec![
        Link::new(
            vec![var(&sc_d), var(&i_prev)],
            Some(block_set_ever.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&sc_d), var(&i_prev)],
            Some(block_scan.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let ever_after = new_var("ever_after", sig());
    push(
        &block_set_ever,
        "int_add",
        vec![var(&se_i), signed(1)],
        &ever_after,
    );
    push(
        &block_set_ever,
        "setfield",
        vec![
            var(&se_d),
            void_field_const("num_ever_used_items"),
            var(&ever_after),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_set_ever.closeblock(vec![
        Link::new(vec![var(&se_d)], Some(block_shrink_check.clone()), None).into_ref(),
    ]);

    let live = new_var("live", sig());
    push(
        &block_shrink_check,
        "getfield",
        vec![var(&sh_d), void_field_const("num_live_items")],
        &live,
    );
    let live_plus_init = new_var("live_plus_init", sig());
    push(
        &block_shrink_check,
        "int_add",
        vec![var(&live), signed(DICT_INITSIZE)],
        &live_plus_init,
    );
    let sh_entries = new_var("entries", entries_ptr_lltype);
    push(
        &block_shrink_check,
        "getfield",
        vec![var(&sh_d), void_field_const("entries")],
        &sh_entries,
    );
    let len_entries = new_var("len_entries", sig());
    push(
        &block_shrink_check,
        "getarraysize",
        vec![var(&sh_entries)],
        &len_entries,
    );
    let one_eighth = new_var("one_eighth", sig());
    push(
        &block_shrink_check,
        "int_rshift",
        vec![var(&len_entries), signed(3)],
        &one_eighth,
    );
    let should_resize = new_var("should_resize", LowLevelType::Bool);
    push(
        &block_shrink_check,
        "int_le",
        vec![var(&live_plus_init), var(&one_eighth)],
        &should_resize,
    );
    block_shrink_check.borrow_mut().exitswitch = Some(var(&should_resize));

    let rs_d = new_var("d", dict_ptr_lltype);
    let block_call_resize = Block::shared(vec![var(&rs_d)]);
    block_shrink_check.closeblock(vec![
        Link::new(
            vec![var(&sh_d)],
            Some(block_call_resize.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![none_void()],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    push(
        &block_call_resize,
        "direct_call",
        vec![Hlvalue::Constant(resize_fn), var(&rs_d)],
        &new_var("v", LowLevelType::Void),
    );
    block_call_resize.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string(), "hash".to_string(), "index".to_string()],
        func,
    ))
}

/// Synthesise `ll_dict_delitem(d, key)` (`rordereddict.py`, fused
/// with `ll_dict_delitem_with_hash`):
///
/// ```python
/// index = d.lookup_function(d, key, d.keyhash(key), FLAG_LOOKUP)
/// if index < 0: raise KeyError
/// _ll_dict_del(d, hash, index)
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "Parameter order mirrors ll_dict_delitem; hash_takes_dict selects ll_keyhash_custom(d, key)"
)]
pub(crate) fn build_ll_dict_delitem_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    hash_fn: Constant,
    call_lookup_fn: Constant,
    del_fn: Constant,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let exc_args = exception_args("KeyError")?;
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype.clone());
    let key = new_var("key", key_lltype);
    let startblock = Block::shared(vec![var(&d), var(&key)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let hash = new_var("hash", sig());
    push(
        &startblock,
        "direct_call",
        keyhash_call_args(hash_fn, &d, &key, hash_takes_dict),
        &hash,
    );
    let index = new_var("index", sig());
    push(
        &startblock,
        "direct_call",
        vec![
            Hlvalue::Constant(call_lookup_fn),
            var(&d),
            var(&key),
            var(&hash),
            signed(FLAG_LOOKUP),
        ],
        &index,
    );
    let missing = new_var("missing", LowLevelType::Bool);
    push(
        &startblock,
        "int_lt",
        vec![var(&index), signed(0)],
        &missing,
    );
    startblock.borrow_mut().exitswitch = Some(var(&missing));

    let fd = new_var("d", dict_ptr_lltype);
    let fh = new_var("hash", sig());
    let fi = new_var("index", sig());
    let block_found = Block::shared(vec![var(&fd), var(&fh), var(&fi)]);
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d), var(&hash), var(&index)],
            Some(block_found.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    push(
        &block_found,
        "direct_call",
        vec![Hlvalue::Constant(del_fn), var(&fd), var(&fh), var(&fi)],
        &new_var("v", LowLevelType::Void),
    );
    block_found.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string(), "key".to_string()],
        func,
    ))
}

/// Synthesise `ll_dict_get(dict, key, default)` (`rordereddict.py`).
#[expect(
    clippy::too_many_arguments,
    reason = "Parameter order mirrors ll_dict_get; hash_takes_dict selects ll_keyhash_custom(d, key)"
)]
pub(crate) fn build_ll_dict_get_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    value_lltype: LowLevelType,
    hash_fn: Constant,
    call_lookup_fn: Constant,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("dict", dict_ptr_lltype.clone());
    let key = new_var("key", key_lltype);
    let default = new_var("default", value_lltype.clone());
    let startblock = Block::shared(vec![var(&d), var(&key), var(&default)]);
    let return_var = new_var("result", value_lltype.clone());
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let hash = new_var("hash", sig());
    push(
        &startblock,
        "direct_call",
        keyhash_call_args(hash_fn, &d, &key, hash_takes_dict),
        &hash,
    );
    let index = new_var("index", sig());
    push(
        &startblock,
        "direct_call",
        vec![
            Hlvalue::Constant(call_lookup_fn),
            var(&d),
            var(&key),
            var(&hash),
            signed(FLAG_LOOKUP),
        ],
        &index,
    );
    let missing = new_var("missing", LowLevelType::Bool);
    push(
        &startblock,
        "int_lt",
        vec![var(&index), signed(0)],
        &missing,
    );
    startblock.borrow_mut().exitswitch = Some(var(&missing));

    let fd = new_var("dict", dict_ptr_lltype);
    let fi = new_var("index", sig());
    let block_found = Block::shared(vec![var(&fd), var(&fi)]);
    startblock.closeblock(vec![
        Link::new(
            vec![var(&default)],
            Some(graph.returnblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d), var(&index)],
            Some(block_found.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let entries = new_var("entries", entries_ptr_lltype);
    push(
        &block_found,
        "getfield",
        vec![var(&fd), void_field_const("entries")],
        &entries,
    );
    let value = new_var("value", value_lltype);
    push(
        &block_found,
        "getinteriorfield",
        vec![var(&entries), var(&fi), void_field_const("value")],
        &value,
    );
    block_found.closeblock(vec![
        Link::new(vec![var(&value)], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["dict".to_string(), "key".to_string(), "default".to_string()],
        func,
    ))
}

/// Synthesise `ll_dict_setdefault(dict, key, default)`
/// (`rordereddict.py`).
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
pub(crate) fn build_ll_dict_setdefault_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    value_lltype: LowLevelType,
    hash_fn: Constant,
    call_lookup_fn: Constant,
    lookup_done_fn: Constant,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("dict", dict_ptr_lltype.clone());
    let key = new_var("key", key_lltype.clone());
    let default = new_var("default", value_lltype.clone());
    let startblock = Block::shared(vec![var(&d), var(&key), var(&default)]);
    let return_var = new_var("result", value_lltype.clone());
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let hash = new_var("hash", sig());
    push(
        &startblock,
        "direct_call",
        keyhash_call_args(hash_fn, &d, &key, hash_takes_dict),
        &hash,
    );
    let index = new_var("index", sig());
    push(
        &startblock,
        "direct_call",
        vec![
            Hlvalue::Constant(call_lookup_fn),
            var(&d),
            var(&key),
            var(&hash),
            signed(FLAG_STORE),
        ],
        &index,
    );
    let missing = new_var("missing", LowLevelType::Bool);
    push(
        &startblock,
        "int_lt",
        vec![var(&index), signed(0)],
        &missing,
    );
    startblock.borrow_mut().exitswitch = Some(var(&missing));

    let md = new_var("dict", dict_ptr_lltype.clone());
    let mk = new_var("key", key_lltype);
    let mv = new_var("default", value_lltype.clone());
    let mh = new_var("hash", sig());
    let block_missing = Block::shared(vec![var(&md), var(&mk), var(&mv), var(&mh)]);
    let fd = new_var("dict", dict_ptr_lltype);
    let fi = new_var("index", sig());
    let block_found = Block::shared(vec![var(&fd), var(&fi)]);
    startblock.closeblock(vec![
        Link::new(
            vec![var(&d), var(&key), var(&default), var(&hash)],
            Some(block_missing.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d), var(&index)],
            Some(block_found.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    push(
        &block_missing,
        "direct_call",
        vec![
            Hlvalue::Constant(lookup_done_fn),
            var(&md),
            var(&mk),
            var(&mv),
            var(&mh),
            signed(-1),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_missing.closeblock(vec![
        Link::new(vec![var(&mv)], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let entries = new_var("entries", entries_ptr_lltype);
    push(
        &block_found,
        "getfield",
        vec![var(&fd), void_field_const("entries")],
        &entries,
    );
    let value = new_var("value", value_lltype);
    push(
        &block_found,
        "getinteriorfield",
        vec![var(&entries), var(&fi), void_field_const("value")],
        &value,
    );
    block_found.closeblock(vec![
        Link::new(vec![var(&value)], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["dict".to_string(), "key".to_string(), "default".to_string()],
        func,
    ))
}

/// Synthesise `_ll_dict_rescue(d)` (`rordereddict.py`):
///
/// ```python
/// def _ll_dict_rescue(d):
///     ll_dict_reindex(d, _ll_len_of_d_indexes(d))
/// ```
///
/// `_ll_len_of_d_indexes` is `getfield(indexes)` then
/// `length_of_simple_gcarray_from_opaque` (`_ll_len_of_d_indexes`).
pub(crate) fn build_ll_dict_rescue_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    reindex_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("d", dict_ptr_lltype);
    let startblock = Block::shared(vec![var(&d)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let indexes_gcref = new_var("indexes_gcref", GCREF.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("indexes")],
        &indexes_gcref,
    );
    let len_indexes = new_var("len_indexes", sig());
    push(
        &startblock,
        "length_of_simple_gcarray_from_opaque",
        vec![var(&indexes_gcref)],
        &len_indexes,
    );
    push(
        &startblock,
        "direct_call",
        vec![Hlvalue::Constant(reindex_fn), var(&d), var(&len_indexes)],
        &new_var("v", LowLevelType::Void),
    );
    startblock.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `@jit.dont_look_inside` + `_dont_inline_ = True` (`_ll_dict_rescue`).
    func._jit_look_inside_ = Some(false);
    func._dont_inline_ = true;
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string()],
        func,
    ))
}

fn close_canraise_with_rescue(
    block: &BlockRef,
    exception_class: &Hlvalue,
    last_etype: Hlvalue,
    last_evalue: Hlvalue,
    normal_target: BlockRef,
    normal_args: Vec<Hlvalue>,
    rescue_target: BlockRef,
    rescue_args: Vec<Hlvalue>,
) {
    block.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
    let normal = Link::new(normal_args, Some(normal_target), None).into_ref();
    let mut exc = Link::new(
        rescue_args,
        Some(rescue_target),
        Some(exception_class.clone()),
    );
    exc.extravars(Some(last_etype), Some(last_evalue));
    block.closeblock(vec![normal, exc.into_ref()]);
}

/// Synthesise `_ll_dict_setitem_lookup_done(d, key, value, hash, i)`
/// (`rordereddict.py`):
///
/// ```python
/// def _ll_dict_setitem_lookup_done(d, key, value, hash, i):
///     ENTRY = lltype.typeOf(d.entries).TO.OF
///     if i >= 0:
///         entry = d.entries[i]
///         entry.value = value
///     else:
///         reindexed = False
///         if len(d.entries) == d.num_ever_used_items:
///             try:
///                 reindexed = ll_dict_grow(d)
///             except:
///                 _ll_dict_rescue(d)
///                 raise
///         rc = d.resize_counter - 3
///         if rc <= 0:
///             try:
///                 ll_dict_resize(d)
///                 reindexed = True
///             except:
///                 _ll_dict_rescue(d)
///                 raise
///             rc = d.resize_counter - 3
///         if reindexed:
///             ll_call_insert_clean_function(d, hash, d.num_ever_used_items)
///         d.resize_counter = rc
///         entry = d.entries[d.num_ever_used_items]
///         entry.key = key
///         entry.value = value
///         if hasattr(ENTRY, 'f_hash'):
///             entry.f_hash = hash
///         if hasattr(ENTRY, 'f_valid'):
///             entry.f_valid = True
///         d.num_ever_used_items += 1
///         d.num_live_items += 1
/// ```
///
/// `hasattr(ENTRY, 'f_hash')` is omitted when `simple_hash_eq`
/// (`get_ll_dict` / `ll_hash_custom_fast`).
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
pub(crate) fn build_ll_dict_setitem_lookup_done_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    value_lltype: LowLevelType,
    grow_fn: Constant,
    resize_fn: Constant,
    insert_clean_fn: Constant,
    rescue_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let none_void = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;
    let exception = exception_args("Exception")?;
    let exception_class = exception[0].clone();

    let d = new_var("d", dict_ptr_lltype.clone());
    let key = new_var("key", key_lltype.clone());
    let value = new_var("value", value_lltype.clone());
    let hash = new_var("hash", sig());
    let i = new_var("i", sig());
    let startblock = Block::shared(vec![var(&d), var(&key), var(&value), var(&hash), var(&i)]);
    let return_var = new_var("result", LowLevelType::Void);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    // overwrite path (i >= 0): entries[i].value = value.
    let ow_d = new_var("d", dict_ptr_lltype.clone());
    let ow_i = new_var("i", sig());
    let ow_value = new_var("value", value_lltype.clone());
    let block_overwrite = Block::shared(vec![var(&ow_d), var(&ow_i), var(&ow_value)]);

    // insert path (i < 0).
    let ins_d = new_var("d", dict_ptr_lltype.clone());
    let ins_key = new_var("key", key_lltype.clone());
    let ins_value = new_var("value", value_lltype.clone());
    let ins_hash = new_var("hash", sig());
    let block_insert_entry = Block::shared(vec![
        var(&ins_d),
        var(&ins_key),
        var(&ins_value),
        var(&ins_hash),
    ]);

    let ge = new_var("ge", LowLevelType::Bool);
    push(&startblock, "int_ge", vec![var(&i), signed(0)], &ge);
    startblock.borrow_mut().exitswitch = Some(var(&ge));
    startblock.closeblock(vec![
        Link::new(
            vec![var(&d), var(&i), var(&value)],
            Some(block_overwrite.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d), var(&key), var(&value), var(&hash)],
            Some(block_insert_entry.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_overwrite body.
    let ow_entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &block_overwrite,
        "getfield",
        vec![var(&ow_d), void_field_const("entries")],
        &ow_entries,
    );
    push(
        &block_overwrite,
        "setinteriorfield",
        vec![
            var(&ow_entries),
            var(&ow_i),
            void_field_const("value"),
            var(&ow_value),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_overwrite.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // block_insert_entry body: len(entries) == num_ever_used_items?
    let orig_entries = new_var("orig_entries", entries_ptr_lltype.clone());
    push(
        &block_insert_entry,
        "getfield",
        vec![var(&ins_d), void_field_const("entries")],
        &orig_entries,
    );
    let len_entries = new_var("len_entries", sig());
    push(
        &block_insert_entry,
        "getarraysize",
        vec![var(&orig_entries)],
        &len_entries,
    );
    let num_ever0 = new_var("num_ever", sig());
    push(
        &block_insert_entry,
        "getfield",
        vec![var(&ins_d), void_field_const("num_ever_used_items")],
        &num_ever0,
    );
    let need_grow = new_var("need_grow", LowLevelType::Bool);
    push(
        &block_insert_entry,
        "int_eq",
        vec![var(&len_entries), var(&num_ever0)],
        &need_grow,
    );
    block_insert_entry.borrow_mut().exitswitch = Some(var(&need_grow));

    // block_do_grow(d, key, value, hash).
    let dg_d = new_var("d", dict_ptr_lltype.clone());
    let dg_key = new_var("key", key_lltype.clone());
    let dg_value = new_var("value", value_lltype.clone());
    let dg_hash = new_var("hash", sig());
    let block_do_grow = Block::shared(vec![
        var(&dg_d),
        var(&dg_key),
        var(&dg_value),
        var(&dg_hash),
    ]);

    // block_after_grow(d, key, value, hash, reindexed).
    let ag_d = new_var("d", dict_ptr_lltype.clone());
    let ag_key = new_var("key", key_lltype.clone());
    let ag_value = new_var("value", value_lltype.clone());
    let ag_hash = new_var("hash", sig());
    let ag_reindexed = new_var("reindexed", LowLevelType::Bool);
    let block_after_grow = Block::shared(vec![
        var(&ag_d),
        var(&ag_key),
        var(&ag_value),
        var(&ag_hash),
        var(&ag_reindexed),
    ]);

    block_insert_entry.closeblock(vec![
        Link::new(
            vec![var(&ins_d), var(&ins_key), var(&ins_value), var(&ins_hash)],
            Some(block_do_grow.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&ins_d),
                var(&ins_key),
                var(&ins_value),
                var(&ins_hash),
                bool_const(false),
            ],
            Some(block_after_grow.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_do_grow body: try: reindexed = ll_dict_grow(d) except: rescue; raise
    let reindexed_grow = new_var("reindexed_grow", LowLevelType::Bool);
    push(
        &block_do_grow,
        "direct_call",
        vec![Hlvalue::Constant(grow_fn), var(&dg_d)],
        &reindexed_grow,
    );
    let rg_d = new_var("d", dict_ptr_lltype.clone());
    let rg_etype = Variable::named("etype");
    let rg_evalue = Variable::named("evalue");
    let block_rescue_grow = Block::shared(vec![
        var(&rg_d),
        Hlvalue::Variable(rg_etype.clone()),
        Hlvalue::Variable(rg_evalue.clone()),
    ]);
    let grow_etype = Variable::named("etype");
    let grow_evalue = Variable::named("evalue");
    close_canraise_with_rescue(
        &block_do_grow,
        &exception_class,
        Hlvalue::Variable(grow_etype.clone()),
        Hlvalue::Variable(grow_evalue.clone()),
        block_after_grow.clone(),
        vec![
            var(&dg_d),
            var(&dg_key),
            var(&dg_value),
            var(&dg_hash),
            var(&reindexed_grow),
        ],
        block_rescue_grow.clone(),
        vec![
            var(&dg_d),
            Hlvalue::Variable(grow_etype),
            Hlvalue::Variable(grow_evalue),
        ],
    );
    push(
        &block_rescue_grow,
        "direct_call",
        vec![Hlvalue::Constant(rescue_fn.clone()), var(&rg_d)],
        &new_var("v", LowLevelType::Void),
    );
    block_rescue_grow.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(rg_etype), Hlvalue::Variable(rg_evalue)],
            Some(graph.exceptblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_after_grow body: rc = resize_counter - 3; branch rc <= 0.
    let rc0 = new_var("rc0", sig());
    push(
        &block_after_grow,
        "getfield",
        vec![var(&ag_d), void_field_const("resize_counter")],
        &rc0,
    );
    let rc = new_var("rc", sig());
    push(
        &block_after_grow,
        "int_sub",
        vec![var(&rc0), signed(3)],
        &rc,
    );
    let need_resize = new_var("need_resize", LowLevelType::Bool);
    push(
        &block_after_grow,
        "int_le",
        vec![var(&rc), signed(0)],
        &need_resize,
    );
    block_after_grow.borrow_mut().exitswitch = Some(var(&need_resize));

    // block_do_resize(d, key, value, hash).
    let dr_d = new_var("d", dict_ptr_lltype.clone());
    let dr_key = new_var("key", key_lltype.clone());
    let dr_value = new_var("value", value_lltype.clone());
    let dr_hash = new_var("hash", sig());
    let block_do_resize = Block::shared(vec![
        var(&dr_d),
        var(&dr_key),
        var(&dr_value),
        var(&dr_hash),
    ]);

    // block_after_resize(d, key, value, hash, reindexed, rc).
    let ar_d = new_var("d", dict_ptr_lltype.clone());
    let ar_key = new_var("key", key_lltype.clone());
    let ar_value = new_var("value", value_lltype.clone());
    let ar_hash = new_var("hash", sig());
    let ar_reindexed = new_var("reindexed", LowLevelType::Bool);
    let ar_rc = new_var("rc", sig());
    let block_after_resize = Block::shared(vec![
        var(&ar_d),
        var(&ar_key),
        var(&ar_value),
        var(&ar_hash),
        var(&ar_reindexed),
        var(&ar_rc),
    ]);

    block_after_grow.closeblock(vec![
        Link::new(
            vec![var(&ag_d), var(&ag_key), var(&ag_value), var(&ag_hash)],
            Some(block_do_resize.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&ag_d),
                var(&ag_key),
                var(&ag_value),
                var(&ag_hash),
                var(&ag_reindexed),
                var(&rc),
            ],
            Some(block_after_resize.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_do_resize body: try: ll_dict_resize(d); reindexed=True except: rescue; raise
    // then rc = resize_counter - 3 on the success path (`_ll_dict_setitem_lookup_done`).
    push(
        &block_do_resize,
        "direct_call",
        vec![Hlvalue::Constant(resize_fn), var(&dr_d)],
        &new_var("v", LowLevelType::Void),
    );
    let rr_d = new_var("d", dict_ptr_lltype.clone());
    let rr_etype = Variable::named("etype");
    let rr_evalue = Variable::named("evalue");
    let block_rescue_resize = Block::shared(vec![
        var(&rr_d),
        Hlvalue::Variable(rr_etype.clone()),
        Hlvalue::Variable(rr_evalue.clone()),
    ]);
    let rc_d = new_var("d", dict_ptr_lltype.clone());
    let rc_key = new_var("key", key_lltype.clone());
    let rc_value = new_var("value", value_lltype.clone());
    let rc_hash = new_var("hash", sig());
    let block_resize_rc = Block::shared(vec![
        var(&rc_d),
        var(&rc_key),
        var(&rc_value),
        var(&rc_hash),
    ]);
    let resize_etype = Variable::named("etype");
    let resize_evalue = Variable::named("evalue");
    close_canraise_with_rescue(
        &block_do_resize,
        &exception_class,
        Hlvalue::Variable(resize_etype.clone()),
        Hlvalue::Variable(resize_evalue.clone()),
        block_resize_rc.clone(),
        vec![var(&dr_d), var(&dr_key), var(&dr_value), var(&dr_hash)],
        block_rescue_resize.clone(),
        vec![
            var(&dr_d),
            Hlvalue::Variable(resize_etype),
            Hlvalue::Variable(resize_evalue),
        ],
    );
    push(
        &block_rescue_resize,
        "direct_call",
        vec![Hlvalue::Constant(rescue_fn), var(&rr_d)],
        &new_var("v", LowLevelType::Void),
    );
    block_rescue_resize.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(rr_etype), Hlvalue::Variable(rr_evalue)],
            Some(graph.exceptblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let rc2_0 = new_var("rc2_0", sig());
    push(
        &block_resize_rc,
        "getfield",
        vec![var(&rc_d), void_field_const("resize_counter")],
        &rc2_0,
    );
    let rc2 = new_var("rc2", sig());
    push(
        &block_resize_rc,
        "int_sub",
        vec![var(&rc2_0), signed(3)],
        &rc2,
    );
    block_resize_rc.closeblock(vec![
        Link::new(
            vec![
                var(&rc_d),
                var(&rc_key),
                var(&rc_value),
                var(&rc_hash),
                bool_const(true),
                var(&rc2),
            ],
            Some(block_after_resize.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_after_resize body: branch on reindexed.
    block_after_resize.borrow_mut().exitswitch = Some(var(&ar_reindexed));

    // block_do_insert_clean(d, key, value, hash, rc).
    let ic_d = new_var("d", dict_ptr_lltype.clone());
    let ic_key = new_var("key", key_lltype.clone());
    let ic_value = new_var("value", value_lltype.clone());
    let ic_hash = new_var("hash", sig());
    let ic_rc = new_var("rc", sig());
    let block_do_insert_clean = Block::shared(vec![
        var(&ic_d),
        var(&ic_key),
        var(&ic_value),
        var(&ic_hash),
        var(&ic_rc),
    ]);

    // block_write_entry(d, key, value, hash, rc).
    let we_d = new_var("d", dict_ptr_lltype.clone());
    let we_key = new_var("key", key_lltype.clone());
    let we_value = new_var("value", value_lltype.clone());
    let we_hash = new_var("hash", sig());
    let we_rc = new_var("rc", sig());
    let block_write_entry = Block::shared(vec![
        var(&we_d),
        var(&we_key),
        var(&we_value),
        var(&we_hash),
        var(&we_rc),
    ]);

    block_after_resize.closeblock(vec![
        Link::new(
            vec![
                var(&ar_d),
                var(&ar_key),
                var(&ar_value),
                var(&ar_hash),
                var(&ar_rc),
            ],
            Some(block_do_insert_clean.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&ar_d),
                var(&ar_key),
                var(&ar_value),
                var(&ar_hash),
                var(&ar_rc),
            ],
            Some(block_write_entry.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_do_insert_clean body.
    let num_ever_ic = new_var("num_ever_ic", sig());
    push(
        &block_do_insert_clean,
        "getfield",
        vec![var(&ic_d), void_field_const("num_ever_used_items")],
        &num_ever_ic,
    );
    push(
        &block_do_insert_clean,
        "direct_call",
        vec![
            Hlvalue::Constant(insert_clean_fn),
            var(&ic_d),
            var(&ic_hash),
            var(&num_ever_ic),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_do_insert_clean.closeblock(vec![
        Link::new(
            vec![
                var(&ic_d),
                var(&ic_key),
                var(&ic_value),
                var(&ic_hash),
                var(&ic_rc),
            ],
            Some(block_write_entry.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_write_entry body: write the fresh entry, bump counters.
    push(
        &block_write_entry,
        "setfield",
        vec![var(&we_d), void_field_const("resize_counter"), var(&we_rc)],
        &new_var("v", LowLevelType::Void),
    );
    let entries2 = new_var("entries2", entries_ptr_lltype.clone());
    push(
        &block_write_entry,
        "getfield",
        vec![var(&we_d), void_field_const("entries")],
        &entries2,
    );
    let num_ever2 = new_var("num_ever2", sig());
    push(
        &block_write_entry,
        "getfield",
        vec![var(&we_d), void_field_const("num_ever_used_items")],
        &num_ever2,
    );
    push(
        &block_write_entry,
        "setinteriorfield",
        vec![
            var(&entries2),
            var(&num_ever2),
            void_field_const("key"),
            var(&we_key),
        ],
        &new_var("v", LowLevelType::Void),
    );
    push(
        &block_write_entry,
        "setinteriorfield",
        vec![
            var(&entries2),
            var(&num_ever2),
            void_field_const("value"),
            var(&we_value),
        ],
        &new_var("v", LowLevelType::Void),
    );
    if dictentry_has_f_hash(&entries_ptr_lltype) {
        push(
            &block_write_entry,
            "setinteriorfield",
            vec![
                var(&entries2),
                var(&num_ever2),
                void_field_const("f_hash"),
                var(&we_hash),
            ],
            &new_var("v", LowLevelType::Void),
        );
    }
    push(
        &block_write_entry,
        "setinteriorfield",
        vec![
            var(&entries2),
            var(&num_ever2),
            void_field_const("f_valid"),
            bool_const(true),
        ],
        &new_var("v", LowLevelType::Void),
    );
    let new_num_ever = new_var("new_num_ever", sig());
    push(
        &block_write_entry,
        "int_add",
        vec![var(&num_ever2), signed(1)],
        &new_num_ever,
    );
    push(
        &block_write_entry,
        "setfield",
        vec![
            var(&we_d),
            void_field_const("num_ever_used_items"),
            var(&new_num_ever),
        ],
        &new_var("v", LowLevelType::Void),
    );
    let num_live = new_var("num_live", sig());
    push(
        &block_write_entry,
        "getfield",
        vec![var(&we_d), void_field_const("num_live_items")],
        &num_live,
    );
    let new_num_live = new_var("new_num_live", sig());
    push(
        &block_write_entry,
        "int_add",
        vec![var(&num_live), signed(1)],
        &new_num_live,
    );
    push(
        &block_write_entry,
        "setfield",
        vec![
            var(&we_d),
            void_field_const("num_live_items"),
            var(&new_num_live),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_write_entry.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "d".to_string(),
            "key".to_string(),
            "value".to_string(),
            "hash".to_string(),
            "i".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_dict_setitem(d, key, value)` (`rordereddict.py`,
/// fused with its `_with_hash` half and the `FLAG_STORE` lookup call — same
/// fusion rationale as [`build_ll_dict_getitem_helper_graph`]):
///
/// ```python
/// def ll_dict_setitem(d, key, value):
///     ll_dict_setitem_with_hash(d, key, d.keyhash(key), value)
///
/// def ll_dict_setitem_with_hash(d, key, hash, value):
///     index = d.lookup_function(d, key, hash, FLAG_STORE)
///     _ll_dict_setitem_lookup_done(d, key, value, hash, index)
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "Parameter order mirrors ll_dict_setitem; hash_takes_dict selects ll_keyhash_custom(d, key)"
)]
pub(crate) fn build_ll_dict_setitem_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    value_lltype: LowLevelType,
    hash_fn: Constant,
    call_lookup_fn: Constant,
    lookup_done_fn: Constant,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let d = variable_with_lltype("d", dict_ptr_lltype);
    let key = variable_with_lltype("key", key_lltype);
    let value = variable_with_lltype("value", value_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(key.clone()),
        Hlvalue::Variable(value.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let hash = variable_with_lltype("hash", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        keyhash_call_args(hash_fn, &d, &key, hash_takes_dict),
        Hlvalue::Variable(hash.clone()),
    ));
    let index = variable_with_lltype("index", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(call_lookup_fn),
            Hlvalue::Variable(d.clone()),
            Hlvalue::Variable(key.clone()),
            Hlvalue::Variable(hash.clone()),
            constant_with_lltype(ConstValue::Int(FLAG_STORE), LowLevelType::Signed),
        ],
        Hlvalue::Variable(index.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(lookup_done_fn),
            Hlvalue::Variable(d),
            Hlvalue::Variable(key),
            Hlvalue::Variable(value),
            Hlvalue::Variable(hash),
            Hlvalue::Variable(index),
        ],
        Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
    ));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["d".to_string(), "key".to_string(), "value".to_string()],
        func,
    ))
}

/// Synthesise `ll_dict_insert(d, key, value)` — one FLAG_STORE probe
/// returning the displaced value.
///
/// Composition of `ll_dict_lookup` (`FLAG_STORE`) +
/// `_ll_dict_setitem_lookup_done`, the same pieces
/// `ll_dict_setitem_with_hash` (`rordereddict.py`) composes. The
/// value-or-None read is the result `ll_dict_get` produces: when
/// `index >= 0`, `d.entries[index].value`; otherwise `None`. Then
/// `_ll_dict_setitem_lookup_done(d, key, value, hash, index)` writes
/// the new value (overwrite or insert).
#[expect(
    clippy::too_many_arguments,
    reason = "Parameter order mirrors ll_dict_insert; hash_takes_dict selects ll_keyhash_custom(d, key)"
)]
pub(crate) fn build_ll_dict_insert_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    key_lltype: LowLevelType,
    value_lltype: LowLevelType,
    hash_fn: Constant,
    call_lookup_fn: Constant,
    lookup_done_fn: Constant,
    hash_takes_dict: bool,
) -> Result<PyGraph, TyperError> {
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);
    let sig = || LowLevelType::Signed;

    let d = new_var("dict", dict_ptr_lltype.clone());
    let key = new_var("key", key_lltype.clone());
    let value = new_var("value", value_lltype.clone());
    let startblock = Block::shared(vec![var(&d), var(&key), var(&value)]);
    let return_var = new_var("result", value_lltype.clone());
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    let hash = new_var("hash", sig());
    push(
        &startblock,
        "direct_call",
        keyhash_call_args(hash_fn, &d, &key, hash_takes_dict),
        &hash,
    );
    let index = new_var("index", sig());
    push(
        &startblock,
        "direct_call",
        vec![
            Hlvalue::Constant(call_lookup_fn),
            var(&d),
            var(&key),
            var(&hash),
            signed(FLAG_STORE),
        ],
        &index,
    );
    let missing = new_var("missing", LowLevelType::Bool);
    push(
        &startblock,
        "int_lt",
        vec![var(&index), signed(0)],
        &missing,
    );
    startblock.borrow_mut().exitswitch = Some(var(&missing));

    let fd = new_var("dict", dict_ptr_lltype.clone());
    let fk = new_var("key", key_lltype.clone());
    let fv = new_var("value", value_lltype.clone());
    let fh = new_var("hash", sig());
    let fi = new_var("index", sig());
    let block_found = Block::shared(vec![var(&fd), var(&fk), var(&fv), var(&fh), var(&fi)]);

    let sd = new_var("dict", dict_ptr_lltype);
    let sk = new_var("key", key_lltype);
    let sv = new_var("value", value_lltype.clone());
    let sh = new_var("hash", sig());
    let si = new_var("index", sig());
    let sdisplaced = new_var("displaced", value_lltype.clone());
    let block_store = Block::shared(vec![
        var(&sd),
        var(&sk),
        var(&sv),
        var(&sh),
        var(&si),
        var(&sdisplaced),
    ]);

    // Miss returns NULL the way `ll_dict_get` does (`rordereddict.py`).
    // `ConstValue::None` is only a valid constant for a pointer lltype.
    if !matches!(value_lltype, LowLevelType::Ptr(_)) {
        return Err(TyperError::message(format!(
            "ll_dict_insert: value type {value_lltype:?} cannot represent None on miss"
        )));
    }
    let none_value = constant_with_lltype(ConstValue::None, value_lltype.clone());
    startblock.closeblock(vec![
        Link::new(
            vec![
                var(&d),
                var(&key),
                var(&value),
                var(&hash),
                var(&index),
                none_value,
            ],
            Some(block_store.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![var(&d), var(&key), var(&value), var(&hash), var(&index)],
            Some(block_found.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let entries = new_var("entries", entries_ptr_lltype);
    push(
        &block_found,
        "getfield",
        vec![var(&fd), void_field_const("entries")],
        &entries,
    );
    let displaced = new_var("displaced", value_lltype);
    push(
        &block_found,
        "getinteriorfield",
        vec![var(&entries), var(&fi), void_field_const("value")],
        &displaced,
    );
    block_found.closeblock(vec![
        Link::new(
            vec![
                var(&fd),
                var(&fk),
                var(&fv),
                var(&fh),
                var(&fi),
                var(&displaced),
            ],
            Some(block_store.clone()),
            None,
        )
        .into_ref(),
    ]);

    push(
        &block_store,
        "direct_call",
        vec![
            Hlvalue::Constant(lookup_done_fn),
            var(&sd),
            var(&sk),
            var(&sv),
            var(&sh),
            var(&si),
        ],
        &new_var("v", LowLevelType::Void),
    );
    block_store.closeblock(vec![
        Link::new(
            vec![var(&sdisplaced)],
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
        vec!["dict".to_string(), "key".to_string(), "value".to_string()],
        func,
    ))
}

/// Synthesise `_ll_dictnext(iter) -> Signed` (`rordereddict.py`),
/// the forward dict-iterator step shared by keys/values/items iteration.
///
/// ```python
/// def _ll_dictnext(iter):
///     dict = iter.dict
///     if dict:
///         entries = dict.entries
///         index = iter.index
///         entries_len = dict.num_ever_used_items
///         while index < entries_len:
///             nextindex = index + 1
///             if entries.valid(index):
///                 iter.index = nextindex
///                 return index
///             else:
///                 if index == (dict.lookup_function_no >> FUNC_SHIFT):
///                     dict.lookup_function_no += (1 << FUNC_SHIFT)
///             index = nextindex
///         iter.dict = nullptr(...)   # clear the reference, prevent restarts
///     raise StopIteration
/// ```
///
/// `entries.valid(index)` is the simple-hash-eq `f_valid` flag
/// (`getinteriorfield(entries, index, "f_valid")`); the dummy-obj valid path
/// is out of scope. The `else` arm is the `popitem(last=False)` fast-forward
/// hack that bumps the high bits of `lookup_function_no` so a repeated
/// iteration over a shrinking prefix starts later; the `ll_assert` guarding
/// it is debug-only and omitted. Self-contained: no lookup, hash, or malloc —
/// a direct entries-array walk.
///
/// 8-block CFG plus the returnblock and exceptblock. Both the null-`dict`
/// guard and the loop-exhausted tail raise `StopIteration` via `exceptblock`.
///
/// PORT STATUS — `newiter`/`rtype_next`/`make_iterator_repr` are now wired
/// (`OrderedDictRepr::make_iterator_repr`, `DictIteratorRepr::newiter`/
/// `rtype_next`, `rtype_method_iterkeys`/`itervalues`/`iteritems` — #145).
/// `rtype_next` `gendirectcall`s the `_ll_dictnext_look_inside_iff` wrapper
/// (`rlib/jit.py look_inside_iff`), reads `iter.dict.entries`, and
/// recasts through `variant_keys`/`variant_values`/`variant_items`
/// (`rdict.py AbstractDictIteratorRepr.rtype_next`);
/// `build_ll_dictiter_helper_graph` mints the matching `newiter`
/// constructor. Still deferred, fail-closed via
/// `ordered_dict_runtime_deferred`: the `reversed(d)` variant
/// (`ll_dictiter_reversed`/`_ll_dictnext_reversed`) and the `*_with_hash`
/// variants (need `ll_ensure_indexes`/`entry_hash`). Tracked by the DictRepr
/// port epic (#140).
pub fn build_ll_dictnext_helper_graph(
    name: &str,
    iter_ptr_lltype: LowLevelType,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let null_dict = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            dict_ptr_lltype.clone(),
        ))
    };
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let sig = || LowLevelType::Signed;
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);

    let exc_args = exception_args("StopIteration")?;

    // startblock inputargs: (iter).
    let iter = new_var("iter", iter_ptr_lltype.clone());
    let startblock = Block::shared(vec![var(&iter)]);
    let return_var = new_var("result", sig());
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    // Pre-create the downstream blocks with fresh inputarg copies.
    // block_setup: (iter, dict).
    let su_iter = new_var("iter", iter_ptr_lltype.clone());
    let su_dict = new_var("dict", dict_ptr_lltype.clone());
    let block_setup = Block::shared(vec![var(&su_iter), var(&su_dict)]);

    // block_loop_cond: (iter, dict, entries, index, entries_len).
    let lc_iter = new_var("iter", iter_ptr_lltype.clone());
    let lc_dict = new_var("dict", dict_ptr_lltype.clone());
    let lc_entries = new_var("entries", entries_ptr_lltype.clone());
    let lc_index = new_var("index", sig());
    let lc_len = new_var("entries_len", sig());
    let block_loop_cond = Block::shared(vec![
        var(&lc_iter),
        var(&lc_dict),
        var(&lc_entries),
        var(&lc_index),
        var(&lc_len),
    ]);

    // block_loop_body: same shape as loop_cond.
    let lb_iter = new_var("iter", iter_ptr_lltype.clone());
    let lb_dict = new_var("dict", dict_ptr_lltype.clone());
    let lb_entries = new_var("entries", entries_ptr_lltype.clone());
    let lb_index = new_var("index", sig());
    let lb_len = new_var("entries_len", sig());
    let block_loop_body = Block::shared(vec![
        var(&lb_iter),
        var(&lb_dict),
        var(&lb_entries),
        var(&lb_index),
        var(&lb_len),
    ]);

    // block_return_valid: (iter, nextindex, index).
    let rv_iter = new_var("iter", iter_ptr_lltype.clone());
    let rv_nextindex = new_var("nextindex", sig());
    let rv_index = new_var("index", sig());
    let block_return_valid = Block::shared(vec![var(&rv_iter), var(&rv_nextindex), var(&rv_index)]);

    // block_invalid: (iter, dict, entries, nextindex, entries_len, index).
    let iv_iter = new_var("iter", iter_ptr_lltype.clone());
    let iv_dict = new_var("dict", dict_ptr_lltype.clone());
    let iv_entries = new_var("entries", entries_ptr_lltype.clone());
    let iv_nextindex = new_var("nextindex", sig());
    let iv_len = new_var("entries_len", sig());
    let iv_index = new_var("index", sig());
    let block_invalid = Block::shared(vec![
        var(&iv_iter),
        var(&iv_dict),
        var(&iv_entries),
        var(&iv_nextindex),
        var(&iv_len),
        var(&iv_index),
    ]);

    // block_bump: (iter, dict, entries, nextindex, entries_len, lfn).
    let bp_iter = new_var("iter", iter_ptr_lltype.clone());
    let bp_dict = new_var("dict", dict_ptr_lltype.clone());
    let bp_entries = new_var("entries", entries_ptr_lltype.clone());
    let bp_nextindex = new_var("nextindex", sig());
    let bp_len = new_var("entries_len", sig());
    let bp_lfn = new_var("lfn", sig());
    let block_bump = Block::shared(vec![
        var(&bp_iter),
        var(&bp_dict),
        var(&bp_entries),
        var(&bp_nextindex),
        var(&bp_len),
        var(&bp_lfn),
    ]);

    // block_clear: (iter).
    let cl_iter = new_var("iter", iter_ptr_lltype.clone());
    let block_clear = Block::shared(vec![var(&cl_iter)]);

    // startblock: dict = iter.dict; if dict raise-guard.
    let dict0 = new_var("dict", dict_ptr_lltype.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&iter), void_field_const("dict")],
        &dict0,
    );
    let nz = new_var("nz", LowLevelType::Bool);
    push(&startblock, "ptr_nonzero", vec![var(&dict0)], &nz);
    startblock.borrow_mut().exitswitch = Some(var(&nz));
    startblock.closeblock(vec![
        Link::new(
            vec![var(&iter), var(&dict0)],
            Some(block_setup.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // block_setup: read entries, index, entries_len; enter loop.
    let su_entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &block_setup,
        "getfield",
        vec![var(&su_dict), void_field_const("entries")],
        &su_entries,
    );
    let su_index = new_var("index", sig());
    push(
        &block_setup,
        "getfield",
        vec![var(&su_iter), void_field_const("index")],
        &su_index,
    );
    let su_len = new_var("entries_len", sig());
    push(
        &block_setup,
        "getfield",
        vec![var(&su_dict), void_field_const("num_ever_used_items")],
        &su_len,
    );
    block_setup.closeblock(vec![
        Link::new(
            vec![
                var(&su_iter),
                var(&su_dict),
                var(&su_entries),
                var(&su_index),
                var(&su_len),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_loop_cond: while index < entries_len.
    let lt = new_var("lt", LowLevelType::Bool);
    push(
        &block_loop_cond,
        "int_lt",
        vec![var(&lc_index), var(&lc_len)],
        &lt,
    );
    block_loop_cond.borrow_mut().exitswitch = Some(var(&lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                var(&lc_iter),
                var(&lc_dict),
                var(&lc_entries),
                var(&lc_index),
                var(&lc_len),
            ],
            Some(block_loop_body.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![var(&lc_iter)],
            Some(block_clear.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // block_loop_body: nextindex = index + 1; branch on entries.valid.
    let lb_next = new_var("nextindex", sig());
    push(
        &block_loop_body,
        "int_add",
        vec![var(&lb_index), signed(1)],
        &lb_next,
    );
    let lb_valid = new_var("valid", LowLevelType::Bool);
    push(
        &block_loop_body,
        "getinteriorfield",
        vec![
            var(&lb_entries),
            var(&lb_index),
            void_field_const("f_valid"),
        ],
        &lb_valid,
    );
    block_loop_body.borrow_mut().exitswitch = Some(var(&lb_valid));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![var(&lb_iter), var(&lb_next), var(&lb_index)],
            Some(block_return_valid.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&lb_iter),
                var(&lb_dict),
                var(&lb_entries),
                var(&lb_next),
                var(&lb_len),
                var(&lb_index),
            ],
            Some(block_invalid.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // block_return_valid: iter.index = nextindex; return index.
    let rv_void = new_var("v", LowLevelType::Void);
    push(
        &block_return_valid,
        "setfield",
        vec![var(&rv_iter), void_field_const("index"), var(&rv_nextindex)],
        &rv_void,
    );
    block_return_valid.closeblock(vec![
        Link::new(vec![var(&rv_index)], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // block_invalid: popitem(last=False) fast-forward hack.
    let iv_lfn = new_var("lfn", sig());
    push(
        &block_invalid,
        "getfield",
        vec![var(&iv_dict), void_field_const("lookup_function_no")],
        &iv_lfn,
    );
    let iv_shifted = new_var("shifted", sig());
    push(
        &block_invalid,
        "int_rshift",
        vec![var(&iv_lfn), signed(FUNC_SHIFT)],
        &iv_shifted,
    );
    let iv_eq = new_var("is_head", LowLevelType::Bool);
    push(
        &block_invalid,
        "int_eq",
        vec![var(&iv_index), var(&iv_shifted)],
        &iv_eq,
    );
    block_invalid.borrow_mut().exitswitch = Some(var(&iv_eq));
    block_invalid.closeblock(vec![
        Link::new(
            vec![
                var(&iv_iter),
                var(&iv_dict),
                var(&iv_entries),
                var(&iv_nextindex),
                var(&iv_len),
                var(&iv_lfn),
            ],
            Some(block_bump.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&iv_iter),
                var(&iv_dict),
                var(&iv_entries),
                var(&iv_nextindex),
                var(&iv_len),
            ],
            Some(block_loop_cond.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // block_bump: lookup_function_no += (1 << FUNC_SHIFT); loop back.
    let bp_new_lfn = new_var("lfn", sig());
    push(
        &block_bump,
        "int_add",
        vec![var(&bp_lfn), signed(1 << FUNC_SHIFT)],
        &bp_new_lfn,
    );
    let bp_void = new_var("v", LowLevelType::Void);
    push(
        &block_bump,
        "setfield",
        vec![
            var(&bp_dict),
            void_field_const("lookup_function_no"),
            var(&bp_new_lfn),
        ],
        &bp_void,
    );
    block_bump.closeblock(vec![
        Link::new(
            vec![
                var(&bp_iter),
                var(&bp_dict),
                var(&bp_entries),
                var(&bp_nextindex),
                var(&bp_len),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_clear: iter.dict = nullptr; raise StopIteration.
    let cl_void = new_var("v", LowLevelType::Void);
    push(
        &block_clear,
        "setfield",
        vec![var(&cl_iter), void_field_const("dict"), null_dict()],
        &cl_void,
    );
    block_clear.closeblock(vec![
        Link::new(exc_args, Some(graph.exceptblock.clone()), None).into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `rordereddict.py _ll_dictnext`: `@jit.oopspec("odictiter.next(iter)")`.
    func.oopspec = Some("odictiter.next(iter)".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["iter".to_string()],
        func,
    ))
}

/// Synthesise `ll_dictiter(ITERPTR, d)` (`rordereddict.py`):
/// allocate a dict iterator, store the source dict, seed the index from
/// `d.lookup_function_no >> FUNC_SHIFT`, and return the iterator.
pub fn build_ll_dictiter_helper_graph(
    name: &str,
    iter_ptr_lltype: LowLevelType,
    dict_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let LowLevelType::Ptr(iter_ptr) = &iter_ptr_lltype else {
        return Err(TyperError::message(
            "build_ll_dictiter_helper_graph: iter lltype must be Ptr",
        ));
    };
    let iter_struct = match &iter_ptr.TO {
        PtrTarget::Struct(st) => st.clone(),
        other => {
            return Err(TyperError::message(format!(
                "build_ll_dictiter_helper_graph: iterator Ptr target must be Struct, got {other:?}"
            )));
        }
    };

    let d_arg = variable_with_lltype("d", dict_ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(d_arg.clone())]);
    let return_var = variable_with_lltype("result", iter_ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let iter = variable_with_lltype("iter", iter_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![
            lowlevel_type_const(LowLevelType::Struct(Box::new(iter_struct))),
            gc_flavor_const()?,
        ],
        Hlvalue::Variable(iter.clone()),
    ));
    let set_dict_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(iter.clone()),
            void_field_const("dict"),
            Hlvalue::Variable(d_arg.clone()),
        ],
        Hlvalue::Variable(set_dict_void),
    ));
    let lookup_function_no = variable_with_lltype("lookup_function_no", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(d_arg),
            void_field_const("lookup_function_no"),
        ],
        Hlvalue::Variable(lookup_function_no.clone()),
    ));
    let index = variable_with_lltype("index", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_rshift",
        vec![
            Hlvalue::Variable(lookup_function_no),
            constant_with_lltype(ConstValue::Int(FUNC_SHIFT), LowLevelType::Signed),
        ],
        Hlvalue::Variable(index.clone()),
    ));
    let set_index_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(iter.clone()),
            void_field_const("index"),
            Hlvalue::Variable(index),
        ],
        Hlvalue::Variable(set_index_void),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(iter)],
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
        vec!["d".to_string()],
        func,
    ))
}

/// RPython `ll_call_insert_clean_function(d, hash, i)` (`rordereddict.py`).
pub fn ll_call_insert_clean_function(d: &_ptr, hash: i64, i: i64) -> Result<(), TyperError> {
    if i < 0 {
        return Err(TyperError::message(
            "ll_call_insert_clean_function(): i < 0",
        ));
    }
    let fun = dict_signed_field(d, "lookup_function_no")? & FUNC_MASK;
    if fun == FUNC_BYTE || fun == FUNC_SHORT || (IS_64BIT && fun == FUNC_INT) || fun == FUNC_LONG {
        ll_dict_store_clean(d, hash, i)
    } else {
        Err(TyperError::message(
            "ll_call_insert_clean_function(): invalid lookup_fun",
        ))
    }
}

/// RPython `ll_call_delete_by_entry_index(d, hash, i, replace_with)`.
pub fn ll_call_delete_by_entry_index(
    d: &_ptr,
    hash: i64,
    i: i64,
    replace_with: i64,
) -> Result<(), TyperError> {
    let fun = dict_signed_field(d, "lookup_function_no")? & FUNC_MASK;
    if fun == FUNC_BYTE || fun == FUNC_SHORT || (IS_64BIT && fun == FUNC_INT) || fun == FUNC_LONG {
        ll_dict_delete_by_entry_index(d, hash, i, replace_with)
    } else {
        Err(TyperError::message(
            "ll_call_delete_by_entry_index(): invalid lookup_fun",
        ))
    }
}

pub fn ll_valid_from_flag() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_valid_from_flag"))
}

pub fn ll_valid_from_key() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_valid_from_key"))
}

pub fn ll_valid_from_value() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_valid_from_value"))
}

pub fn ll_mark_deleted_in_flag() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_mark_deleted_in_flag"))
}

pub fn ll_mark_deleted_in_key() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_mark_deleted_in_key"))
}

pub fn ll_mark_deleted_in_value() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_mark_deleted_in_value"))
}

pub fn ll_hash_from_cache() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_hash_from_cache"))
}

pub fn ll_hash_recomputed() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_hash_recomputed"))
}

pub fn ll_hash_custom_fast() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_hash_custom_fast"))
}

pub fn ll_keyhash_custom() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_keyhash_custom"))
}

pub fn ll_keyeq_custom() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_keyeq_custom"))
}

pub fn ll_dict_len() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_len"))
}

pub fn ll_dict_bool() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_bool"))
}

fn dict_adtmeth(d: &_ptr, name: &str) -> Option<ConstValue> {
    match &d._TYPE.TO {
        PtrTarget::Struct(s) => s._adtmeths.get(name).cloned(),
        _ => None,
    }
}

fn adtmeth_is_byte_str(value: &ConstValue, expected: &[u8]) -> bool {
    matches!(value, ConstValue::ByteStr(s) if s == expected)
}

/// `objectmodel.hlinvoke(r_callable, callable, *args)` on the host: a Void
/// `FunctionRepr` names its unique function; otherwise the stored PBC field
/// is the funcptr.
fn host_hlinvoke(
    d: &_ptr,
    repr_adtmeth: &str,
    field: &str,
    args: &[LowLevelValue],
) -> Result<LowLevelValue, TyperError> {
    let adt = dict_adtmeth(d, repr_adtmeth).ok_or_else(|| {
        TyperError::message(format!("host_hlinvoke: missing adtmeth {repr_adtmeth}"))
    })?;
    let ConstValue::Repr(repr_const) = adt else {
        return Err(TyperError::message(format!(
            "host_hlinvoke: {repr_adtmeth} is not a ReprConst, got {adt:?}"
        )));
    };
    let repr = repr_const.as_repr();
    let p = match baked_callable_from_repr(repr.as_ref())? {
        Some(c) => match c.value {
            ConstValue::LLPtr(p) => *p,
            other => {
                return Err(TyperError::message(format!(
                    "host_hlinvoke: unique llfn is not an LLPtr, got {other:?}"
                )));
            }
        },
        None => match d.getattr(field).map_err(TyperError::message)? {
            LowLevelValue::Ptr(p) => *p,
            other => {
                return Err(TyperError::message(format!(
                    "host_hlinvoke: stored {field} is not a Ptr, got {other:?}"
                )));
            }
        },
    };
    eval_funcptr(&p, args)
        .map_err(TaskError::from)
        .map_err(|e| TyperError::message(e.message))
}

/// RPython `d.keyhash(key)` (`rordereddict.py` `convert_const`): call the
/// stored DICT adtmeth funcptr itself. A graph runs through
/// `LLInterpreter.eval_graph`; a graphless funcptr goes through its
/// `_callable` (`llinterp.py` `op_direct_call` / `perform_call`).
fn ll_keyhash(d: &_ptr, key: &LowLevelValue) -> Result<i64, TyperError> {
    let hashed = match dict_adtmeth(d, "keyhash") {
        Some(adt) if adtmeth_is_byte_str(&adt, b"ll_keyhash_custom") => {
            host_hlinvoke(d, "r_rdict_hashfn", "fnkeyhash", std::slice::from_ref(key))?
        }
        Some(adt) => {
            let fp = adtmeth_funcptr(&adt).ok_or_else(|| {
                TyperError::message(format!(
                    "d.keyhash: stored adtmeth is not a funcptr: {adt:?}"
                ))
            })?;
            eval_funcptr(&fp, std::slice::from_ref(key))
                .map_err(TaskError::from)
                .map_err(|e| TyperError::message(e.message))?
        }
        None => {
            return Err(TyperError::message(
                "d.keyhash: DICT has no keyhash adtmeth",
            ));
        }
    };
    match hashed {
        LowLevelValue::Signed(n) => Ok(n),
        other => Err(TyperError::message(format!(
            "d.keyhash: hash function returned {other:?}"
        ))),
    }
}

fn adtmeth_funcptr(value: &ConstValue) -> Option<_ptr> {
    match value {
        ConstValue::LLPtr(p) => Some((**p).clone()),
        _ => None,
    }
}

/// Primitive `==` used when `d.keyeq is None` (`rordereddict.py`
/// `ll_dict_lookup`). Float keys use IEEE equality so `0.0 == -0.0`.
fn ll_direct_compare(a: &LowLevelValue, b: &LowLevelValue) -> bool {
    match (a, b) {
        (LowLevelValue::Float(x), LowLevelValue::Float(y)) => {
            f64::from_bits(*x) == f64::from_bits(*y)
        }
        _ => a == b,
    }
}

/// RPython `d.keyeq(checkingkey, key)` (`rordereddict.py`). `None` for
/// identity keys; otherwise the stored `get_ll_eq_function` funcptr.
fn ll_keyeq(d: &_ptr, a: &LowLevelValue, b: &LowLevelValue) -> Result<Option<bool>, TyperError> {
    let compared = match dict_adtmeth(d, "keyeq") {
        Some(ConstValue::None) | None => return Ok(None),
        Some(adt) if adtmeth_is_byte_str(&adt, b"ll_keyeq_custom") => {
            host_hlinvoke(d, "r_rdict_eqfn", "fnkeyeq", &[a.clone(), b.clone()])?
        }
        Some(adt) => {
            let fp = adtmeth_funcptr(&adt).ok_or_else(|| {
                TyperError::message(format!("d.keyeq: stored adtmeth is not a funcptr: {adt:?}"))
            })?;
            eval_funcptr(&fp, &[a.clone(), b.clone()])
                .map_err(TaskError::from)
                .map_err(|e| TyperError::message(e.message))?
        }
    };
    match compared {
        LowLevelValue::Bool(eq) => Ok(Some(eq)),
        other => Err(TyperError::message(format!(
            "d.keyeq: eq function returned {other:?}"
        ))),
    }
}

fn dict_paranoia(d: &_ptr) -> bool {
    matches!(dict_adtmeth(d, "paranoia"), Some(ConstValue::Bool(true)))
}

fn entries_no_direct_compare(_entries: &_ptr) -> bool {
    // `get_ll_dict` sets `ENTRIES.no_direct_compare` only when
    // `ll_eq_function.no_direct_compare` (`ll_eq_shortint`). This host
    // layout has no ARRAY adtmeths, and no key repr we wire sets that flag.
    false
}

fn dict_indexes_array(d: &_ptr) -> Result<_ptr, TyperError> {
    let indexes_val = d.getattr("indexes").map_err(TyperError::message)?;
    let LowLevelValue::Ptr(gcref) = indexes_val else {
        return Err(TyperError::message("indexes is not a pointer"));
    };
    if !gcref.nonzero() {
        return Err(TyperError::message("indexes is null"));
    }
    let LowLevelType::Ptr(index_ptr_t) = &*DICTINDEX_BYTE else {
        return Err(TyperError::message("DICTINDEX_BYTE is not a Ptr"));
    };
    cast_opaque_ptr(index_ptr_t, &gcref).map_err(TyperError::message)
}

fn indexes_signed_at(indexes: &_ptr, i: usize) -> Result<i64, TyperError> {
    match indexes.getitem(i).map_err(TyperError::message)? {
        LowLevelValue::Unsigned(n) => Ok(n as i64),
        LowLevelValue::Signed(n) => Ok(n),
        other => Err(TyperError::message(format!(
            "index slot is not an integer: {other:?}"
        ))),
    }
}

fn _ll_write_indexes(d: &_ptr, i: usize, value: i64) -> Result<(), TyperError> {
    let mut indexes = dict_indexes_array(d)?;
    indexes
        .setitem(i, LowLevelValue::Unsigned(value as u64))
        .map_err(TyperError::message)
}

fn dict_entry_getattr(entry: &LowLevelValue, name: &str) -> Result<LowLevelValue, TyperError> {
    match entry {
        LowLevelValue::InteriorPtr(ip) => ip.getattr(name).map_err(TyperError::message),
        LowLevelValue::Ptr(p) => p.getattr(name).map_err(TyperError::message),
        other => Err(TyperError::message(format!(
            "dict entry is not a struct pointer: {other:?}"
        ))),
    }
}

fn write_dict_entry_value(entry: LowLevelValue, value: LowLevelValue) -> Result<(), TyperError> {
    match entry {
        LowLevelValue::InteriorPtr(mut ip) => {
            ip.setattr("value", value).map_err(TyperError::message)
        }
        LowLevelValue::Ptr(mut p) => p.setattr("value", value).map_err(TyperError::message),
        other => Err(TyperError::message(format!(
            "dict entry is not a struct pointer: {other:?}"
        ))),
    }
}

fn ll_dict_create_initial_index(d: &mut _ptr) -> Result<(), TyperError> {
    let live = dict_signed_field(d, "num_live_items")?;
    if live == 0 {
        ll_malloc_indexes_and_choose_lookup(d, DICT_INITSIZE as usize)?;
        set_dict_signed_field(d, "resize_counter", DICT_INITSIZE * 2)?;
        Ok(())
    } else {
        ll_dict_rehash_after_translation(d)
    }
}

/// RPython `ll_call_lookup_function(d, key, hash, flag)` (`rordereddict.py`).
/// `d.lookup_function` is this ADT method.
fn ll_call_lookup_function(
    d: &mut _ptr,
    key: &LowLevelValue,
    hash: i64,
    flag: i64,
) -> Result<i64, TyperError> {
    loop {
        let fun = dict_signed_field(d, "lookup_function_no")? & FUNC_MASK;
        if fun == FUNC_BYTE
            || fun == FUNC_SHORT
            || (IS_64BIT && fun == FUNC_INT)
            || fun == FUNC_LONG
        {
            return ll_dict_lookup_t(d, key, hash, flag);
        }
        ll_dict_create_initial_index(d)?;
    }
}

/// RPython `ll_dict_lookup(d, key, hash, store_flag, T)` (`rordereddict.py`).
/// All `TYPE_*` widths collapse to `Ptr(GcArray(Unsigned))`.
fn ll_dict_lookup_t(
    d: &mut _ptr,
    key: &LowLevelValue,
    hash: i64,
    store_flag: i64,
) -> Result<i64, TyperError> {
    let entries = dict_entries(d)?;
    let indexes = dict_indexes_array(d)?;
    let mask = (indexes.len().map_err(TyperError::message)? - 1) as u64;
    let mut i = (hash as u64) & mask;
    let direct_compare = !entries_no_direct_compare(&entries);
    let index = indexes_signed_at(&indexes, i as usize)?;
    let mut deletedslot: i64 = -1;
    if index >= VALID_OFFSET {
        let entry_i = index - VALID_OFFSET;
        let entry = entries
            .getitem(entry_i as usize)
            .map_err(TyperError::message)?;
        let checkingkey = dict_entry_getattr(&entry, "key")?;
        if direct_compare && ll_direct_compare(&checkingkey, key) {
            return Ok(entry_i);
        }
        match lookup_keyeq_match(d, &entries, &indexes, &checkingkey, key, hash, entry_i)? {
            LookupEq::Found => return Ok(entry_i),
            LookupEq::Restart => return ll_dict_lookup_t(d, key, hash, store_flag),
            LookupEq::Continue => {}
        }
    } else if index == DELETED {
        deletedslot = i as i64;
    } else {
        if store_flag == FLAG_STORE {
            let ever_used = dict_signed_field(d, "num_ever_used_items")?;
            _ll_write_indexes(d, i as usize, ever_used + VALID_OFFSET)?;
        }
        return Ok(-1);
    }
    let mut perturb = hash as u64;
    loop {
        i = i
            .wrapping_shl(2)
            .wrapping_add(i)
            .wrapping_add(perturb)
            .wrapping_add(1)
            & mask;
        let index = indexes_signed_at(&indexes, i as usize)?;
        if index == FREE {
            if store_flag == FLAG_STORE {
                if deletedslot == -1 {
                    deletedslot = i as i64;
                }
                let ever_used = dict_signed_field(d, "num_ever_used_items")?;
                _ll_write_indexes(d, deletedslot as usize, ever_used + VALID_OFFSET)?;
            }
            return Ok(-1);
        } else if index >= VALID_OFFSET {
            let entry_i = index - VALID_OFFSET;
            let entry = entries
                .getitem(entry_i as usize)
                .map_err(TyperError::message)?;
            let checkingkey = dict_entry_getattr(&entry, "key")?;
            if direct_compare && ll_direct_compare(&checkingkey, key) {
                return Ok(entry_i);
            }
            match lookup_keyeq_match(d, &entries, &indexes, &checkingkey, key, hash, entry_i)? {
                LookupEq::Found => return Ok(entry_i),
                LookupEq::Restart => return ll_dict_lookup_t(d, key, hash, store_flag),
                LookupEq::Continue => {}
            }
        } else if deletedslot == -1 {
            deletedslot = i as i64;
        }
        perturb >>= PERTURB_SHIFT as u32;
    }
}

enum LookupEq {
    Continue,
    Found,
    Restart,
}

fn lookup_keyeq_match(
    d: &mut _ptr,
    entries: &_ptr,
    indexes: &_ptr,
    checkingkey: &LowLevelValue,
    key: &LowLevelValue,
    hash: i64,
    entry_i: i64,
) -> Result<LookupEq, TyperError> {
    // `if d.keyeq is not None and entries.entry_hash(...) == hash`
    if matches!(dict_adtmeth(d, "keyeq"), Some(ConstValue::None) | None) {
        return Ok(LookupEq::Continue);
    }
    if entries_entry_hash(d, entries, entry_i as usize)? != hash {
        return Ok(LookupEq::Continue);
    }
    let Some(found) = ll_keyeq(d, checkingkey, key)? else {
        return Ok(LookupEq::Continue);
    };
    if dict_paranoia(d) {
        let cur_entries = dict_entries(d)?;
        let cur_indexes = dict_indexes_array(d)?;
        if cur_entries._hashable_identity() != entries._hashable_identity() {
            return Ok(LookupEq::Restart);
        }
        if cur_indexes._hashable_identity() != indexes._hashable_identity() {
            return Ok(LookupEq::Restart);
        }
        if !entries_valid(entries, entry_i as usize)? {
            return Ok(LookupEq::Restart);
        }
        let cur_key = dict_entry_getattr(
            &entries
                .getitem(entry_i as usize)
                .map_err(TyperError::message)?,
            "key",
        )?;
        if cur_key != *checkingkey {
            return Ok(LookupEq::Restart);
        }
    }
    Ok(if found {
        LookupEq::Found
    } else {
        LookupEq::Continue
    })
}

/// RPython `ll_dict_getitem(d, key)` (`rordereddict.py`):
/// `return ll_dict_getitem_with_hash(d, key, d.keyhash(key))`.
pub fn ll_dict_getitem(d: &mut _ptr, key: &LowLevelValue) -> Result<LowLevelValue, TyperError> {
    let hash = ll_keyhash(d, key)?;
    ll_dict_getitem_with_hash(d, key, hash)
}

/// RPython `ll_dict_getitem_with_hash(d, key, hash)` (`rordereddict.py`):
/// `index = d.lookup_function(d, key, hash, FLAG_LOOKUP)`;
/// `if index >= 0: return d.entries[index].value else raise KeyError`.
pub fn ll_dict_getitem_with_hash(
    d: &mut _ptr,
    key: &LowLevelValue,
    hash: i64,
) -> Result<LowLevelValue, TyperError> {
    let index = ll_call_lookup_function(d, key, hash, FLAG_LOOKUP)?;
    if index >= 0 {
        let entries_val = d.getattr("entries").map_err(TyperError::message)?;
        let LowLevelValue::Ptr(entries) = entries_val else {
            return Err(TyperError::message(
                "ll_dict_getitem: entries is not a pointer",
            ));
        };
        let entry = entries
            .getitem(index as usize)
            .map_err(TyperError::message)?;
        dict_entry_getattr(&entry, "value")
    } else {
        Err(TyperError::message("KeyError"))
    }
}

/// RPython `ll_dict_setitem(d, key, value)` (`rordereddict.py`):
/// `ll_dict_setitem_with_hash(d, key, d.keyhash(key), value)`.
pub fn ll_dict_setitem(
    d: &mut _ptr,
    key: LowLevelValue,
    value: LowLevelValue,
) -> Result<(), TyperError> {
    let hash = ll_keyhash(d, &key)?;
    ll_dict_setitem_with_hash(d, key, hash, value)
}

/// RPython `ll_dict_setitem_with_hash(d, key, hash, value)` (`rordereddict.py`):
/// `index = d.lookup_function(d, key, hash, FLAG_STORE)`;
/// `_ll_dict_setitem_lookup_done(d, key, value, hash, index)`.
pub fn ll_dict_setitem_with_hash(
    d: &mut _ptr,
    key: LowLevelValue,
    hash: i64,
    value: LowLevelValue,
) -> Result<(), TyperError> {
    let index = ll_call_lookup_function(d, &key, hash, FLAG_STORE)?;
    _ll_dict_setitem_lookup_done(d, key, value, hash, index)
}

pub fn ll_dict_lookup(
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    index_elem_lltype: LowLevelType,
    key_lltype: LowLevelType,
    eq_fn_const: Option<Constant>,
) -> Result<PyGraph, TyperError> {
    build_ll_dict_lookup_helper_graph(
        "ll_dict_lookup",
        dict_ptr_lltype,
        entries_ptr_lltype,
        index_elem_lltype,
        key_lltype,
        eq_fn_const,
        false,
        false,
        None,
        false,
    )
}

/// RPython `_ll_dict_setitem_lookup_done(d, key, value, hash, i)`
/// (`rordereddict.py`).
pub fn _ll_dict_setitem_lookup_done(
    d: &mut _ptr,
    key: LowLevelValue,
    value: LowLevelValue,
    hash: i64,
    i: i64,
) -> Result<(), TyperError> {
    if i >= 0 {
        let entries = dict_entries(d)?;
        let entry = entries.getitem(i as usize).map_err(TyperError::message)?;
        return write_dict_entry_value(entry, value);
    }
    let mut reindexed = false;
    let ever_used = dict_signed_field(d, "num_ever_used_items")?;
    let entries = dict_entries(d)?;
    if entries.len().map_err(TyperError::message)? == ever_used as usize {
        match ll_dict_grow(d) {
            Ok(grew_reindexed) => reindexed = grew_reindexed,
            Err(err) => {
                _ll_dict_rescue(d)?;
                return Err(err);
            }
        }
    }
    let mut rc = dict_signed_field(d, "resize_counter")? - 3;
    if rc <= 0 {
        match ll_dict_resize(d) {
            Ok(()) => reindexed = true,
            Err(err) => {
                _ll_dict_rescue(d)?;
                return Err(err);
            }
        }
        rc = dict_signed_field(d, "resize_counter")? - 3;
        if rc <= 0 {
            return Err(TyperError::message("ll_dict_resize failed?"));
        }
    }
    if reindexed {
        let ever_used = dict_signed_field(d, "num_ever_used_items")?;
        ll_call_insert_clean_function(d, hash, ever_used)?;
    }
    set_dict_signed_field(d, "resize_counter", rc)?;
    let ever_used = dict_signed_field(d, "num_ever_used_items")?;
    let entries = dict_entries(d)?;
    let entry = entries
        .getitem(ever_used as usize)
        .map_err(TyperError::message)?;
    let set_valid = dict_entry_has_field(d, "f_valid");
    let f_hash = dict_entry_has_field(d, "f_hash").then_some(hash);
    write_dict_entry_fields(entry, key, value, set_valid, f_hash)?;
    set_dict_signed_field(d, "num_ever_used_items", ever_used + 1)?;
    let live = dict_signed_field(d, "num_live_items")?;
    set_dict_signed_field(d, "num_live_items", live + 1)?;
    Ok(())
}

/// RPython `_ll_dict_rescue(d)` (`rordereddict.py`): reindex in place
/// after a malloc failure left `indexes` inconsistent.
pub fn _ll_dict_rescue(d: &mut _ptr) -> Result<(), TyperError> {
    let n = _ll_len_of_d_indexes(d)?;
    ll_dict_reindex(d, n as usize)
}

/// RPython `_ll_dict_insert_no_index(d, key, value)` (`rordereddict.py`):
/// never translated. Writes `entries[num_ever_used_items]`, sets
/// `f_valid`, leaves `f_hash` uninitialized, then bumps live/ever-used
/// and `resize_counter -= 3`.
pub fn _ll_dict_insert_no_index(
    d: &mut _ptr,
    key: LowLevelValue,
    value: LowLevelValue,
) -> Result<(), TyperError> {
    let entries_val = d.getattr("entries").map_err(TyperError::message)?;
    let LowLevelValue::Ptr(entries) = entries_val else {
        return Err(TyperError::message(
            "_ll_dict_insert_no_index: entries is not a pointer",
        ));
    };
    let ever_used = dict_signed_field(d, "num_ever_used_items")?;
    let entry = entries
        .getitem(ever_used as usize)
        .map_err(TyperError::message)?;
    let set_valid = dict_entry_has_field(d, "f_valid");
    write_dict_entry_fields(entry, key, value, set_valid, None)?;
    set_dict_signed_field(d, "num_ever_used_items", ever_used + 1)?;
    let live = dict_signed_field(d, "num_live_items")?;
    set_dict_signed_field(d, "num_live_items", live + 1)?;
    let rc = dict_signed_field(d, "resize_counter")?;
    set_dict_signed_field(d, "resize_counter", rc - 3)?;
    Ok(())
}

fn convert_dict_item_const(
    repr: &dyn Repr,
    value: &ConstValue,
) -> Result<LowLevelValue, TyperError> {
    // `r_key.convert_const(dictkey)` / `r_value.convert_const(dictvalue)`
    // (`OrderedDictRepr.convert_const`). The original host value is passed
    // through; `StringRepr.convert_const` accepts `None` as a null STR.
    let c = repr.convert_const(value)?;
    crate::translator::rtyper::rclass::constant_to_lowlevel_value(&c)
}

fn dict_signed_field(d: &_ptr, name: &str) -> Result<i64, TyperError> {
    match d.getattr(name).map_err(TyperError::message)? {
        LowLevelValue::Signed(n) => Ok(n),
        other => Err(TyperError::message(format!(
            "{name} is not Signed: {other:?}"
        ))),
    }
}

fn set_dict_signed_field(d: &mut _ptr, name: &str, n: i64) -> Result<(), TyperError> {
    d.setattr(name, LowLevelValue::Signed(n))
        .map_err(TyperError::message)
}

fn dict_entry_has_field(d: &_ptr, field: &str) -> bool {
    let Ok(entries_val) = d.getattr("entries") else {
        return false;
    };
    let LowLevelValue::Ptr(entries) = entries_val else {
        return false;
    };
    match &entries._TYPE.TO {
        PtrTarget::Array(array_t) => matches!(
            &array_t.OF,
            LowLevelType::Struct(s) if s.getattr_field_type(field).is_some()
        ),
        _ => false,
    }
}

fn write_dict_entry_fields(
    entry: LowLevelValue,
    key: LowLevelValue,
    value: LowLevelValue,
    set_valid: bool,
    f_hash: Option<i64>,
) -> Result<(), TyperError> {
    match entry {
        LowLevelValue::InteriorPtr(mut ip) => {
            ip.setattr("key", key).map_err(TyperError::message)?;
            ip.setattr("value", value).map_err(TyperError::message)?;
            if set_valid {
                ip.setattr("f_valid", LowLevelValue::Bool(true))
                    .map_err(TyperError::message)?;
            }
            if let Some(hash) = f_hash {
                ip.setattr("f_hash", LowLevelValue::Signed(hash))
                    .map_err(TyperError::message)?;
            }
            Ok(())
        }
        LowLevelValue::Ptr(mut p) => {
            p.setattr("key", key).map_err(TyperError::message)?;
            p.setattr("value", value).map_err(TyperError::message)?;
            if set_valid {
                p.setattr("f_valid", LowLevelValue::Bool(true))
                    .map_err(TyperError::message)?;
            }
            if let Some(hash) = f_hash {
                p.setattr("f_hash", LowLevelValue::Signed(hash))
                    .map_err(TyperError::message)?;
            }
            Ok(())
        }
        other => Err(TyperError::message(format!(
            "dict entry is not a struct pointer: {other:?}"
        ))),
    }
}

fn dict_entries(d: &_ptr) -> Result<_ptr, TyperError> {
    let entries_val = d.getattr("entries").map_err(TyperError::message)?;
    let LowLevelValue::Ptr(entries) = entries_val else {
        return Err(TyperError::message("entries is not a pointer"));
    };
    Ok(*entries)
}

fn dict_indexes_is_null(d: &_ptr) -> Result<bool, TyperError> {
    match d.getattr("indexes").map_err(TyperError::message)? {
        LowLevelValue::Ptr(p) => Ok(!p.nonzero()),
        other => Err(TyperError::message(format!(
            "indexes is not a pointer: {other:?}"
        ))),
    }
}

fn entries_valid(entries: &_ptr, i: usize) -> Result<bool, TyperError> {
    let entry = entries.getitem(i).map_err(TyperError::message)?;
    match dict_entry_getattr(&entry, "f_valid") {
        Ok(LowLevelValue::Bool(b)) => Ok(b),
        Ok(other) => Err(TyperError::message(format!(
            "f_valid is not Bool: {other:?}"
        ))),
        Err(_) => Ok(true),
    }
}

fn entries_entry_hash(d: &_ptr, entries: &_ptr, i: usize) -> Result<i64, TyperError> {
    let entry = entries.getitem(i).map_err(TyperError::message)?;
    if let Ok(LowLevelValue::Signed(h)) = dict_entry_getattr(&entry, "f_hash") {
        return Ok(h);
    }
    let key = dict_entry_getattr(&entry, "key")?;
    ll_keyhash(d, &key)
}

fn entries_mark_deleted(entries: &_ptr, i: usize) -> Result<(), TyperError> {
    let entry = entries.getitem(i).map_err(TyperError::message)?;
    match entry {
        LowLevelValue::InteriorPtr(mut ip) => ip
            .setattr("f_valid", LowLevelValue::Bool(false))
            .map_err(TyperError::message),
        LowLevelValue::Ptr(mut p) => p
            .setattr("f_valid", LowLevelValue::Bool(false))
            .map_err(TyperError::message),
        other => Err(TyperError::message(format!(
            "dict entry is not a struct pointer: {other:?}"
        ))),
    }
}

fn entry_field_type(d: &_ptr, field: &str) -> Option<LowLevelType> {
    let Ok(entries) = dict_entries(d) else {
        return None;
    };
    match &entries._TYPE.TO {
        PtrTarget::Array(array_t) => match &array_t.OF {
            LowLevelType::Struct(s) => s.getattr_field_type(field),
            _ => None,
        },
        _ => None,
    }
}

/// `get_ll_dict` (`rordereddict.py`): `must_clear_key` is
/// `(isinstance(DICTKEY, lltype.Ptr) and DICTKEY._needsgc())` (same for
/// `must_clear_value`). `_ll_dict_del_entry` / `ll_dict_remove_deleted_items`
/// clear the field only when this is true.
fn entry_field_must_clear(d: &_ptr, field: &str) -> Option<Ptr> {
    match entry_field_type(d, field) {
        Some(LowLevelType::Ptr(ptr_t)) if ptr_t._needsgc() => Some(*ptr_t),
        _ => None,
    }
}

fn write_entry_named_field(
    entry: LowLevelValue,
    name: &str,
    value: LowLevelValue,
) -> Result<(), TyperError> {
    match entry {
        LowLevelValue::InteriorPtr(mut ip) => ip.setattr(name, value).map_err(TyperError::message),
        LowLevelValue::Ptr(mut p) => p.setattr(name, value).map_err(TyperError::message),
        other => Err(TyperError::message(format!(
            "dict entry is not a struct pointer: {other:?}"
        ))),
    }
}

fn copy_dict_entry(
    src: LowLevelValue,
    dst: LowLevelValue,
    copy_hash: bool,
) -> Result<(), TyperError> {
    let key = dict_entry_getattr(&src, "key")?;
    let value = dict_entry_getattr(&src, "value")?;
    let set_valid = matches!(
        dict_entry_getattr(&src, "f_valid"),
        Ok(LowLevelValue::Bool(true))
    );
    let f_hash = if copy_hash {
        match dict_entry_getattr(&src, "f_hash") {
            Ok(LowLevelValue::Signed(h)) => Some(h),
            _ => None,
        }
    } else {
        None
    };
    write_dict_entry_fields(dst, key, value, set_valid, f_hash)
}

fn malloc_entries_array(d: &_ptr, n: usize) -> Result<_ptr, TyperError> {
    let entries = dict_entries(d)?;
    let PtrTarget::Array(array_t) = &entries._TYPE.TO else {
        return Err(TyperError::message("DICT.entries.TO is not an Array"));
    };
    malloc(
        LowLevelType::Array(Box::new(array_t.clone())),
        Some(n),
        MallocFlavor::Gc,
        true,
    )
    .map_err(TyperError::message)
}

fn set_dict_entries(d: &mut _ptr, entries: _ptr) -> Result<(), TyperError> {
    d.setattr("entries", LowLevelValue::Ptr(Box::new(entries)))
        .map_err(TyperError::message)
}

/// RPython `ll_len_of_d_indexes` / `_ll_len_of_d_indexes` (`rordereddict.py`).
pub fn ll_len_of_d_indexes(d: &_ptr) -> Result<i64, TyperError> {
    _ll_len_of_d_indexes(d)
}

pub fn _ll_len_of_d_indexes(d: &_ptr) -> Result<i64, TyperError> {
    let indexes = dict_indexes_array(d)?;
    Ok(indexes.len().map_err(TyperError::message)? as i64)
}

/// RPython `_overallocate_entries_len` (`rordereddict.py`).
pub fn _overallocate_entries_len(baselen: usize) -> usize {
    let newsize = baselen + (baselen >> 3);
    newsize + 8
}

/// RPython `_ll_dict_entries_size_too_big(d, new_allocated)`.
pub fn _ll_dict_entries_size_too_big(d: &_ptr, new_allocated: usize) -> Result<bool, TyperError> {
    let fun = dict_signed_field(d, "lookup_function_no")? & FUNC_MASK;
    let limit = if fun == FUNC_BYTE {
        (1usize << 8) - MIN_INDEXES_MINUS_ENTRIES as usize
    } else if fun == FUNC_SHORT {
        (1usize << 16) - MIN_INDEXES_MINUS_ENTRIES as usize
    } else if IS_64BIT && fun == FUNC_INT {
        (1usize << 32) - MIN_INDEXES_MINUS_ENTRIES as usize
    } else {
        return Ok(false);
    };
    Ok(new_allocated > limit)
}

fn ll_dict_store_clean(d: &_ptr, hash: i64, index: i64) -> Result<(), TyperError> {
    let indexes = dict_indexes_array(d)?;
    let mask = (indexes.len().map_err(TyperError::message)? - 1) as u64;
    let mut i = (hash as u64) & mask;
    let mut perturb = hash as u64;
    while indexes_signed_at(&indexes, i as usize)? != FREE {
        i = i
            .wrapping_shl(2)
            .wrapping_add(i)
            .wrapping_add(perturb)
            .wrapping_add(1)
            & mask;
        perturb >>= PERTURB_SHIFT as u32;
    }
    _ll_write_indexes(d, i as usize, index + VALID_OFFSET)
}

/// RPython `ll_dict_grow(d)` (`rordereddict.py`).
pub fn ll_dict_grow(d: &mut _ptr) -> Result<bool, TyperError> {
    let live = dict_signed_field(d, "num_live_items")?;
    let ever_used = dict_signed_field(d, "num_ever_used_items")?;
    if live < ever_used / 2 {
        ll_dict_remove_deleted_items(d)?;
        return Ok(true);
    }
    let entries = dict_entries(d)?;
    let new_allocated = _overallocate_entries_len(entries.len().map_err(TyperError::message)?);
    if _ll_dict_entries_size_too_big(d, new_allocated)? {
        ll_dict_remove_deleted_items(d)?;
        let live = dict_signed_field(d, "num_live_items")?;
        let ever_used = dict_signed_field(d, "num_ever_used_items")?;
        if live != ever_used {
            return Err(TyperError::message(
                "ll_dict_grow: num_live_items != num_ever_used_items after compact",
            ));
        }
        return Ok(true);
    }
    let newitems = malloc_entries_array(d, new_allocated)?;
    let old_len = entries.len().map_err(TyperError::message)?;
    let copy_hash = dict_entry_has_field(d, "f_hash");
    for i in 0..old_len {
        let src = entries.getitem(i).map_err(TyperError::message)?;
        let dst = newitems.getitem(i).map_err(TyperError::message)?;
        copy_dict_entry(src, dst, copy_hash)?;
    }
    set_dict_entries(d, newitems)?;
    Ok(false)
}

/// RPython `ll_dict_remove_deleted_items(d)` (`rordereddict.py`).
pub fn ll_dict_remove_deleted_items(d: &mut _ptr) -> Result<(), TyperError> {
    let live = dict_signed_field(d, "num_live_items")?;
    let entries = dict_entries(d)?;
    let old_len = entries.len().map_err(TyperError::message)?;
    let newitems = if live < (old_len as i64) / 4 {
        malloc_entries_array(d, _overallocate_entries_len(live as usize))?
    } else {
        entries.clone()
    };
    let copy_hash = dict_entry_has_field(d, "f_hash");
    let isrclimit = dict_signed_field(d, "num_ever_used_items")? as usize;
    let mut idst = 0usize;
    for isrc in 0..isrclimit {
        if entries_valid(&entries, isrc)? {
            let src = entries.getitem(isrc).map_err(TyperError::message)?;
            let dst = newitems.getitem(idst).map_err(TyperError::message)?;
            copy_dict_entry(src, dst, copy_hash)?;
            idst += 1;
        }
    }
    if dict_signed_field(d, "num_live_items")? != idst as i64 {
        return Err(TyperError::message(
            "ll_dict_remove_deleted_items: live != idst",
        ));
    }
    set_dict_signed_field(d, "num_ever_used_items", idst as i64)?;
    if entries._hashable_identity() == newitems._hashable_identity() {
        for i in idst..isrclimit {
            let entry = newitems.getitem(i).map_err(TyperError::message)?;
            if let Some(ptr_t) = entry_field_must_clear(d, "key") {
                write_entry_named_field(
                    entry.clone(),
                    "key",
                    LowLevelValue::Ptr(Box::new(ptr_t._defl())),
                )?;
            }
            if let Some(ptr_t) = entry_field_must_clear(d, "value") {
                write_entry_named_field(
                    entry,
                    "value",
                    LowLevelValue::Ptr(Box::new(ptr_t._defl())),
                )?;
            }
        }
    } else {
        set_dict_entries(d, newitems)?;
    }
    let n = _ll_len_of_d_indexes(d)?;
    ll_dict_reindex(d, n as usize)
}

/// RPython `ll_dict_resize(d)` (`rordereddict.py`).
pub fn ll_dict_resize(d: &mut _ptr) -> Result<(), TyperError> {
    let live = dict_signed_field(d, "num_live_items")?;
    let num_extra = (live + 1).min(30000);
    _ll_dict_resize_to(d, num_extra)
}

/// RPython `_ll_dict_resize_to(d, num_extra)` (`rordereddict.py`).
pub fn _ll_dict_resize_to(d: &mut _ptr, num_extra: i64) -> Result<(), TyperError> {
    let live = dict_signed_field(d, "num_live_items")?;
    let new_estimate = (live + num_extra) * 2;
    let mut new_size = DICT_INITSIZE;
    while new_size <= new_estimate {
        new_size *= 2;
    }
    if new_size < _ll_len_of_d_indexes(d)? {
        ll_dict_remove_deleted_items(d)
    } else {
        ll_dict_reindex(d, new_size as usize)
    }
}

/// RPython `ll_dict_reindex(d, new_size)` (`rordereddict.py`).
pub fn ll_dict_reindex(d: &mut _ptr, new_size: usize) -> Result<(), TyperError> {
    if !dict_indexes_is_null(d)? && _ll_len_of_d_indexes(d)? == new_size as i64 {
        ll_clear_indexes(d, new_size)?;
    } else {
        ll_malloc_indexes_and_choose_lookup(d, new_size)?;
    }
    let live = dict_signed_field(d, "num_live_items")?;
    let rc = (new_size as i64) * 2 - live * 3;
    if rc <= 0 {
        return Err(TyperError::message("reindex: resize_counter <= 0"));
    }
    set_dict_signed_field(d, "resize_counter", rc)?;
    let lookup_no = dict_signed_field(d, "lookup_function_no")?;
    if lookup_no >> FUNC_SHIFT != 0 {
        return Err(TyperError::message("reindex: lookup_fun >> SHIFT"));
    }
    let entries = dict_entries(d)?;
    let ibound = dict_signed_field(d, "num_ever_used_items")?;
    for i in 0..ibound {
        if entries_valid(&entries, i as usize)? {
            let h = entries_entry_hash(d, &entries, i as usize)?;
            ll_dict_store_clean(d, h, i)?;
        }
    }
    Ok(())
}

/// RPython `ll_dict_rehash_after_translation(d)` (`rordereddict.py`).
pub fn ll_dict_rehash_after_translation(d: &mut _ptr) -> Result<(), TyperError> {
    let live = dict_signed_field(d, "num_live_items")?;
    let ever_used = dict_signed_field(d, "num_ever_used_items")?;
    if live != ever_used {
        return Err(TyperError::message(
            "ll_dict_rehash_after_translation: num_live_items != num_ever_used_items",
        ));
    }
    if !dict_indexes_is_null(d)? {
        return Err(TyperError::message(
            "ll_dict_rehash_after_translation: indexes already present",
        ));
    }
    let entries = dict_entries(d)?;
    let has_f_hash = dict_entry_has_field(d, "f_hash");
    for i in 0..ever_used {
        if !entries_valid(&entries, i as usize)? {
            return Err(TyperError::message(
                "ll_dict_rehash_after_translation: deleted entry in prebuilt dict",
            ));
        }
        let entry = entries.getitem(i as usize).map_err(TyperError::message)?;
        let key = dict_entry_getattr(&entry, "key")?;
        let h = ll_keyhash(d, &key)?;
        if has_f_hash {
            write_entry_named_field(entry, "f_hash", LowLevelValue::Signed(h))?;
        }
    }
    let mut new_size = DICT_INITSIZE;
    while new_size * 2 - live * 3 <= 0 {
        new_size *= 2;
    }
    ll_dict_reindex(d, new_size as usize)
}

fn ll_dict_delete_by_entry_index(
    d: &_ptr,
    hash: i64,
    locate_index: i64,
    replace_with: i64,
) -> Result<(), TyperError> {
    let indexes = dict_indexes_array(d)?;
    let mask = (indexes.len().map_err(TyperError::message)? - 1) as u64;
    let mut i = (hash as u64) & mask;
    let mut perturb = hash as u64;
    let mut index = indexes_signed_at(&indexes, i as usize)?;
    while index != locate_index + VALID_OFFSET {
        debug_assert!(
            index != FREE,
            "ll_dict_delete_by_entry_index: hit a FREE slot"
        );
        i = i
            .wrapping_shl(2)
            .wrapping_add(i)
            .wrapping_add(perturb)
            .wrapping_add(1)
            & mask;
        perturb >>= PERTURB_SHIFT as u32;
        index = indexes_signed_at(&indexes, i as usize)?;
    }
    _ll_write_indexes(d, i as usize, replace_with)
}

fn _ll_dict_del_entry(d: &mut _ptr, index: i64) -> Result<(), TyperError> {
    let entries = dict_entries(d)?;
    entries_mark_deleted(&entries, index as usize)?;
    let live = dict_signed_field(d, "num_live_items")?;
    set_dict_signed_field(d, "num_live_items", live - 1)?;
    let entry = entries
        .getitem(index as usize)
        .map_err(TyperError::message)?;
    if let Some(ptr_t) = entry_field_must_clear(d, "key") {
        write_entry_named_field(
            entry.clone(),
            "key",
            LowLevelValue::Ptr(Box::new(ptr_t._defl())),
        )?;
    }
    if let Some(ptr_t) = entry_field_must_clear(d, "value") {
        write_entry_named_field(entry, "value", LowLevelValue::Ptr(Box::new(ptr_t._defl())))?;
    }
    Ok(())
}

fn _ll_dict_del(d: &mut _ptr, hash: i64, index: i64) -> Result<(), TyperError> {
    ll_call_delete_by_entry_index(d, hash, index, DELETED)?;
    _ll_dict_del_entry(d, index)?;
    let live = dict_signed_field(d, "num_live_items")?;
    if live == 0 {
        set_dict_signed_field(d, "num_ever_used_items", 0)?;
        let fun = dict_signed_field(d, "lookup_function_no")? & FUNC_MASK;
        set_dict_signed_field(d, "lookup_function_no", fun)?;
    } else if index == dict_signed_field(d, "num_ever_used_items")? - 1 {
        let entries = dict_entries(d)?;
        let mut i = index;
        loop {
            i -= 1;
            if i < 0 {
                return Err(TyperError::message("_ll_dict_del: no valid entry"));
            }
            if entries_valid(&entries, i as usize)? {
                break;
            }
        }
        set_dict_signed_field(d, "num_ever_used_items", i + 1)?;
    }
    let entries = dict_entries(d)?;
    let entries_len = entries.len().map_err(TyperError::message)? as i64;
    if live + DICT_INITSIZE <= entries_len / 8 {
        ll_dict_resize(d)?;
    }
    Ok(())
}

/// RPython `ll_dict_delitem(d, key)` (`rordereddict.py`).
pub fn ll_dict_delitem(d: &mut _ptr, key: &LowLevelValue) -> Result<(), TyperError> {
    let hash = ll_keyhash(d, key)?;
    ll_dict_delitem_with_hash(d, key, hash)
}

/// RPython `ll_dict_delitem_with_hash(d, key, hash)` (`rordereddict.py`).
pub fn ll_dict_delitem_with_hash(
    d: &mut _ptr,
    key: &LowLevelValue,
    hash: i64,
) -> Result<(), TyperError> {
    let index = ll_call_lookup_function(d, key, hash, FLAG_LOOKUP)?;
    if index < 0 {
        return Err(TyperError::message("KeyError"));
    }
    _ll_dict_del(d, hash, index)
}

pub fn ll_dict_delitem_if_value_is() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_delitem_if_value_is"))
}

/// RPython `_ll_ptr_to_array_of(T)` (`rordereddict.py`).
pub fn _ll_ptr_to_array_of(T: LowLevelType) -> LowLevelType {
    ptr_to_gc_array(T)
}

pub fn _ll_empty_array() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_empty_array"))
}

pub fn _ll_malloc_dict() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_malloc_dict"))
}

pub fn _ll_malloc_entries() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_malloc_entries"))
}

pub fn _ll_free_entries() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_free_entries"))
}

pub fn ll_dictiter(
    iter_ptr_lltype: LowLevelType,
    dict_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_dictiter_helper_graph("ll_dictiter", iter_ptr_lltype, dict_ptr_lltype)
}

pub fn ll_dictiter_reversed() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dictiter_reversed"))
}

pub fn _ll_dictnext_reversed() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_dictnext_reversed"))
}

pub fn ll_dict_copy() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_copy"))
}

pub fn ll_dict_clear() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_clear"))
}

pub fn ll_dict_update() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_update"))
}

pub fn ll_prepare_dict_update() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_prepare_dict_update"))
}

pub fn recast() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("recast"))
}

pub fn _make_ll_keys_values_items() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_make_ll_keys_values_items"))
}

pub fn ll_dict_keys() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_keys"))
}

pub fn ll_dict_values() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_values"))
}

pub fn ll_dict_items() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_items"))
}

pub fn _ll_getnextitem() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_getnextitem"))
}

pub fn ll_dict_popitem() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_popitem"))
}

pub fn ll_dict_pop() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_pop"))
}

pub fn ll_dict_pop_default() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_pop_default"))
}

pub fn ll_dict_move_to_end() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_move_to_end"))
}

pub fn ll_dict_move_to_last() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_move_to_last"))
}

pub fn ll_dict_move_to_first() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_move_to_first"))
}

pub fn _ll_dict_move_to_first_shift_items() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "_ll_dict_move_to_first_shift_items",
    ))
}

/// RPython `get_ll_dictiter(DICTPTR)` (`rordereddict.py`).
pub fn get_ll_dictiter(DICTPTR: LowLevelType) -> LowLevelType {
    let dictiter = Struct::gc(
        "dictiter",
        vec![
            ("dict".into(), DICTPTR),
            ("index".into(), LowLevelType::Signed),
        ],
    );
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(dictiter),
    }))
}

/// RPython `class DictIteratorRepr(AbstractDictIteratorRepr)`
/// (`lltypesystem/rordereddict.py`).
///
/// Upstream stores the whole `r_dict` (`DictIteratorRepr.__init__`)
/// and reaches `r_dict.recast_key`/`recast_value` through it.
/// Rust has no single owned/`Arc`-shared `OrderedDictRepr` to stash at
/// `make_iterator_repr` time (mirroring how `ListIteratorRepr`/
/// `RangeIteratorRepr` decompose their container repr into the specific
/// fields they need instead of holding it whole), so the pieces
/// `newiter`/`rtype_next` actually read — the dict/entries lltypes and the
/// key/value recast repr pairs — are copied out at construction time
/// instead.
#[derive(Debug)]
pub struct DictIteratorRepr {
    pub base: AbstractDictIteratorRepr,
    pub r_dict_lowleveltype: LowLevelType,
    pub entries_lowleveltype: LowLevelType,
    pub key_repr: Arc<dyn Repr>,
    pub external_key_repr: Arc<dyn Repr>,
    pub value_repr: Arc<dyn Repr>,
    pub external_value_repr: Arc<dyn Repr>,
}

impl DictIteratorRepr {
    /// RPython `DictIteratorRepr.__init__(self, r_dict, variant="keys")`
    /// (`rordereddict.py`). Only the forward `ll_dictiter`/
    /// `_ll_dictnext` constructors are wired — the `variant == "reversed"`
    /// swap to `ll_dictiter_reversed`/`_ll_dictnext_reversed` is rejected by
    /// [`Repr::make_iterator_repr`] before this ever runs with that variant.
    pub fn new(r_dict: &OrderedDictRepr, variant: impl Into<String>) -> Self {
        let variant = variant.into();
        let r_dict_lowleveltype = r_dict.lowleveltype().clone();
        let lowleveltype = get_ll_dictiter(r_dict_lowleveltype.clone());
        let (key_repr, external_key_repr) = r_dict.base.recast_key();
        let (value_repr, external_value_repr) = r_dict.base.recast_value();
        DictIteratorRepr {
            base: AbstractDictIteratorRepr::new(lowleveltype, vec![variant]),
            r_dict_lowleveltype,
            entries_lowleveltype: r_dict.entries_ptr_lltype(),
            key_repr: key_repr.clone(),
            external_key_repr: external_key_repr.clone(),
            value_repr: value_repr.clone(),
            external_value_repr: external_value_repr.clone(),
        }
    }

    /// RPython `AbstractDictIteratorRepr.variant_keys` (`rdict.py`);
    /// also `variant_reversed = variant_keys` (`:120`) — the read step is
    /// identical for the forward and reversed variants, only the iterator
    /// walk direction differs (in the unported `ll_dictiter_reversed`/
    /// `_ll_dictnext_reversed` constructors).
    fn variant_keys(&self, hop: &HighLevelOp, v_entries: Hlvalue, v_index: Hlvalue) -> RTypeResult {
        let key_lltype = self.key_repr.lowleveltype().clone();
        let v_key = hop
            .genop(
                "getinteriorfield",
                vec![v_entries, v_index, void_field_const("key")],
                GenopResult::LLType(key_lltype),
            )
            .ok_or_else(|| {
                TyperError::message(
                    "DictIteratorRepr.variant_keys: getinteriorfield 'key' returned no value",
                )
            })?;
        let recast = hop.llops.borrow_mut().convertvar(
            v_key,
            self.key_repr.as_ref(),
            self.external_key_repr.as_ref(),
        )?;
        Ok(Some(recast))
    }

    /// RPython `AbstractDictIteratorRepr.variant_values` (`rdict.py`).
    fn variant_values(
        &self,
        hop: &HighLevelOp,
        v_entries: Hlvalue,
        v_index: Hlvalue,
    ) -> RTypeResult {
        let value_lltype = self.value_repr.lowleveltype().clone();
        let v_value = hop
            .genop(
                "getinteriorfield",
                vec![v_entries, v_index, void_field_const("value")],
                GenopResult::LLType(value_lltype),
            )
            .ok_or_else(|| {
                TyperError::message(
                    "DictIteratorRepr.variant_values: getinteriorfield 'value' returned no value",
                )
            })?;
        let recast = hop.llops.borrow_mut().convertvar(
            v_value,
            self.value_repr.as_ref(),
            self.external_value_repr.as_ref(),
        )?;
        Ok(Some(recast))
    }

    /// RPython `AbstractDictIteratorRepr.variant_items` (`rdict.py`),
    /// via `get_tuple_result` (`:95-111`). The local `items` result type is
    /// already a [`crate::translator::rtyper::rtuple::TupleRepr`] of
    /// `(external_key_repr, external_value_repr)` (`hop.r_result`), so the
    /// tuple malloc is [`crate::translator::rtyper::rtuple::TupleRepr::newtuple`]
    /// rather than a hand-rolled malloc — the local equivalent of upstream's
    /// inline tuple allocation, same idiom as
    /// [`crate::translator::rtyper::rtuple::TupleRepr::newtuple_cached`]
    /// reading `hop.r_result`.
    fn variant_items(
        &self,
        hop: &HighLevelOp,
        v_entries: Hlvalue,
        v_index: Hlvalue,
    ) -> RTypeResult {
        let v_key = self
            .variant_keys(hop, v_entries.clone(), v_index.clone())?
            .ok_or_else(|| {
                TyperError::message(
                    "DictIteratorRepr.variant_items: variant_keys produced no value",
                )
            })?;
        let v_value = self
            .variant_values(hop, v_entries, v_index)?
            .ok_or_else(|| {
                TyperError::message(
                    "DictIteratorRepr.variant_items: variant_values produced no value",
                )
            })?;
        let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
            TyperError::message("DictIteratorRepr.variant_items: hop.r_result missing")
        })?;
        let any_r: &dyn std::any::Any = r_result.as_ref();
        let r_tuple = any_r
            .downcast_ref::<crate::translator::rtyper::rtuple::TupleRepr>()
            .ok_or_else(|| {
                TyperError::message(
                    "DictIteratorRepr.variant_items: hop.r_result is not a TupleRepr",
                )
            })?;
        let v_tuple = crate::translator::rtyper::rtuple::TupleRepr::newtuple(
            &mut hop.llops.borrow_mut(),
            r_tuple,
            vec![v_key, v_value],
        )?;
        Ok(Some(v_tuple))
    }
}

impl Repr for DictIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        self.base.lowleveltype()
    }

    fn state(&self) -> &ReprState {
        self.base.state()
    }

    fn class_name(&self) -> &'static str {
        "DictIteratorRepr"
    }

    fn repr_class_id(&self) -> crate::translator::rtyper::pairtype::ReprClassId {
        crate::translator::rtyper::pairtype::ReprClassId::DictIteratorRepr
    }

    /// RPython `AbstractDictIteratorRepr.newiter(self, hop)` (`rdict.py`):
    ///
    /// ```python
    /// def newiter(self, hop):
    ///     v_dict, = hop.inputargs(self.r_dict)
    ///     citerptr = hop.inputconst(lltype.Void, self.lowleveltype)
    ///     return hop.gendirectcall(self.ll_dictiter, citerptr, v_dict)
    /// ```
    ///
    /// Same shape as [`super::super::rlist::ListIteratorRepr::newiter`]: the
    /// iterator lltype is baked into the `ll_dictiter` helper builder
    /// instead of threaded as a `citerptr` Void const, and the
    /// `hop.inputargs(self.r_dict)` conversion target is the operand's own
    /// dict repr (`hop.args_r[0]`) — there is no `OrderedDictRepr ->
    /// OrderedDictRepr` conversion, so `convertvar` needs the operand's own
    /// repr to short-circuit as identity.
    fn newiter(&self, hop: &HighLevelOp) -> RTypeResult {
        let r_dict = {
            let args_r = hop.args_r.borrow();
            args_r
                .first()
                .and_then(|o| o.clone())
                .ok_or_else(|| TyperError::message("DictIteratorRepr.newiter: arg0 repr missing"))?
        };
        let vargs = hop.inputargs(vec![ConvertedTo::Repr(r_dict.as_ref())])?;
        let dict_lltype = self.r_dict_lowleveltype.clone();
        let iter_lltype = self.lowleveltype().clone();
        let dict_for_builder = dict_lltype.clone();
        let iter_for_builder = iter_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_dictiter".to_string(),
            vec![dict_lltype],
            iter_lltype,
            move |_rtyper, _args, _result| {
                build_ll_dictiter_helper_graph(
                    "ll_dictiter",
                    iter_for_builder.clone(),
                    dict_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, vargs)
    }

    /// RPython `AbstractDictIteratorRepr.rtype_next(self, hop)`
    /// (`rdict.py` `AbstractDictIteratorRepr.rtype_next`):
    ///
    /// ```python
    /// def rtype_next(self, hop):
    ///     v_iter, = hop.inputargs(self)
    ///     hop.has_implicit_exception(StopIteration)
    ///     hop.has_implicit_exception(RuntimeError)
    ///     hop.exception_is_here()
    ///     v_index = hop.gendirectcall(self._ll_dictnext, v_iter)
    ///     DICT = self.lowleveltype.TO.dict
    ///     v_dict = hop.genop('getfield', [v_iter, c_dict], resulttype=DICT)
    ///     ENTRIES = DICT.TO.entries
    ///     v_entries = hop.genop('getfield', [v_dict, c_entries], resulttype=ENTRIES)
    ///     method = getattr(self, 'variant_' + self.variant)
    ///     return method(hop, ENTRIES, v_entries, v_dict, v_index)
    /// ```
    ///
    /// `keys_with_hash`/`items_with_hash` (needing `ll_ensure_indexes` +
    /// `entry_hash`) are unported and fail closed via the wildcard arm below.
    fn rtype_next(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_iter = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.has_implicit_exception("StopIteration");
        hop.has_implicit_exception("RuntimeError");
        hop.exception_is_here()?;

        let iter_lltype = self.lowleveltype().clone();
        let dict_lltype = self.r_dict_lowleveltype.clone();
        let entries_lltype = self.entries_lowleveltype.clone();
        let iter_for_builder = iter_lltype.clone();
        let dict_for_builder = dict_lltype.clone();
        let entries_for_builder = entries_lltype.clone();
        // `_ll_dictnext`: `@jit.look_inside_iff(lambda iter: jit.isvirtual(iter)
        // and (iter.dict is None or jit.isvirtual(iter.dict)))`
        // `@jit.oopspec("odictiter.next(iter)")` (`rordereddict.py _ll_dictnext`).
        let helper = mint_look_inside_iff(
            &hop.rtyper,
            "_ll_dictnext",
            &["iter"],
            vec![iter_lltype],
            LowLevelType::Signed,
            move |name| {
                build_ll_dictnext_helper_graph(
                    name,
                    iter_for_builder.clone(),
                    dict_for_builder.clone(),
                    entries_for_builder.clone(),
                )
            },
            &[
                LookInsidePred::IsVirtual(0),
                LookInsidePred::FieldNullOrIsVirtual {
                    arg: 0,
                    field: "dict",
                },
            ],
        )?;
        let v_index = hop.gendirectcall(&helper, v_iter.clone())?.ok_or_else(|| {
            TyperError::message("DictIteratorRepr.rtype_next: _ll_dictnext returned Void")
        })?;

        // DICT = self.lowleveltype.TO.dict ; v_dict = getfield(v_iter, 'dict')
        let v_dict = hop
            .genop(
                "getfield",
                vec![v_iter[0].clone(), void_field_const("dict")],
                GenopResult::LLType(dict_lltype),
            )
            .ok_or_else(|| {
                TyperError::message(
                    "DictIteratorRepr.rtype_next: getfield 'dict' returned no value",
                )
            })?;
        // ENTRIES = DICT.TO.entries ; v_entries = getfield(v_dict, 'entries')
        let v_entries = hop
            .genop(
                "getfield",
                vec![v_dict, void_field_const("entries")],
                GenopResult::LLType(entries_lltype),
            )
            .ok_or_else(|| {
                TyperError::message(
                    "DictIteratorRepr.rtype_next: getfield 'entries' returned no value",
                )
            })?;

        match self.base.variant.first().map(|s| s.as_str()) {
            Some("keys") | Some("reversed") => self.variant_keys(hop, v_entries, v_index),
            Some("values") => self.variant_values(hop, v_entries, v_index),
            Some("items") => self.variant_items(hop, v_entries, v_index),
            other => Err(ordered_dict_runtime_deferred(&format!(
                "DictIteratorRepr.rtype_next variant {other:?} (with_hash variants need \
                 ll_ensure_indexes/entry_hash)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::dictdef::DictDef;
    use crate::annotator::model::{SomeFloat, SomeInteger, SomeString, SomeTuple, SomeValue};
    use crate::translator::rtyper::llinterp::{hash_float_value, hash_unichar_ords, python_frexp};
    use crate::translator::rtyper::lltypesystem::lltype::{
        _ptr_obj, FuncType, functionptr_for_graph_with_type,
    };
    use crate::translator::rtyper::lltypesystem::rstr::{UNICODE, ll_strhash_value, llstr};
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rfloat::float_repr;
    use crate::translator::rtyper::rint::signed_repr;
    use crate::translator::rtyper::rstr::{string_repr, unicode_repr};
    use crate::translator::rtyper::rtuple::TupleRepr;
    use crate::translator::rtyper::rtyper::LowLevelOpList;
    use std::cell::RefCell;

    #[test]
    fn ordereddictrepr_builds_sparse_index_dicttable_shape() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );

        let repr = OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr");

        assert_eq!(repr.repr_class_id(), ReprClassId::OrderedDictRepr);
        assert_eq!(repr.DICTENTRY._name, "odictentry");
        assert_eq!(
            repr.DICTENTRY._names,
            vec!["key", "f_valid", "value", "f_hash"]
        );
        assert_eq!(
            repr.DICT._names,
            vec![
                "num_live_items",
                "num_ever_used_items",
                "resize_counter",
                "indexes",
                "lookup_function_no",
                "entries"
            ]
        );
        assert_eq!(
            ReprClassId::OrderedDictRepr.mro(),
            &[
                ReprClassId::OrderedDictRepr,
                ReprClassId::AbstractDictRepr,
                ReprClassId::Repr
            ]
        );
    }

    fn ptr_gcarray_of(value: &LowLevelType) -> &Array {
        let LowLevelType::Ptr(ptr) = value else {
            panic!("expected Ptr(GcArray), got {value:?}");
        };
        let PtrTarget::Array(array) = &ptr.TO else {
            panic!("expected Ptr(GcArray), got {value:?}");
        };
        array
    }

    #[test]
    fn lookup_constants_match_64bit_upstream_layout() {
        assert_eq!(FUNC_SHIFT, 3);
        assert_eq!(FUNC_MASK, 0x07);
        assert_eq!(FUNC_BYTE, 0);
        assert_eq!(FUNC_SHORT, 1);
        assert_eq!(FUNC_INT, 2);
        assert_eq!(FUNC_LONG, 3);
        assert_eq!(FUNC_MUST_REINDEX, 4);
        assert_eq!(PERTURB_SHIFT, 5);
        assert_eq!(FREE, 0);
        assert_eq!(DELETED, 1);
        assert_eq!(VALID_OFFSET, 2);
        assert_eq!(MIN_INDEXES_MINUS_ENTRIES, 3);
        assert_eq!(FLAG_LOOKUP, 0);
        assert_eq!(FLAG_STORE, 1);
        assert_eq!(DICT_INITSIZE, 16);
    }

    #[test]
    fn dictindex_aliases_keep_ptr_gcarray_shape() {
        for alias in [
            &*DICTINDEX_BYTE,
            &*DICTINDEX_SHORT,
            &*DICTINDEX_INT,
            &*DICTINDEX_LONG,
        ] {
            assert_eq!(ptr_gcarray_of(alias).OF, LowLevelType::Unsigned);
        }
        assert_eq!(TYPE_BYTE, LowLevelType::Unsigned);
        assert_eq!(TYPE_SHORT, LowLevelType::Unsigned);
        assert_eq!(TYPE_INT, LowLevelType::Unsigned);
        assert_eq!(TYPE_LONG, LowLevelType::Unsigned);
    }

    #[test]
    fn runtime_helper_surface_is_explicitly_deferred() {
        let err = ll_dict_len().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_len"));

        let err = ll_dict_delitem_if_value_is().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_delitem_if_value_is"));

        let err = ll_dict_keys().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_keys"));

        let err = ll_dict_move_to_first().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_move_to_first"));
    }

    #[test]
    fn overallocate_entries_len_matches_upstream_growth_pattern() {
        let got: Vec<_> = [0, 8, 17, 27, 38, 50, 64, 80]
            .into_iter()
            .map(_overallocate_entries_len)
            .collect();
        assert_eq!(got, vec![8, 17, 27, 38, 50, 64, 80, 98]);
    }

    #[test]
    fn get_ll_dictiter_builds_dict_and_index_fields() {
        let dictptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(Struct::gc("dicttable", vec![])),
        }));
        let iterptr = get_ll_dictiter(dictptr.clone());
        let LowLevelType::Ptr(ptr) = iterptr else {
            panic!("expected Ptr(GcStruct), got {iterptr:?}");
        };
        let PtrTarget::Struct(iter) = &ptr.TO else {
            panic!("expected Ptr(GcStruct), got {ptr:?}");
        };
        assert_eq!(iter._name, "dictiter");
        assert_eq!(iter._names, vec!["dict", "index"]);
        assert_eq!(iter._flds.get("dict"), Some(&dictptr));
        assert_eq!(iter._flds.get("index"), Some(&LowLevelType::Signed));
    }

    #[test]
    fn dictiteratorrepr_extends_abstract_iterator_repr() {
        let r_dict = sample_ordered_dict_repr();
        let dictptr = r_dict.lowleveltype().clone();
        let repr = DictIteratorRepr::new(&r_dict, "keys");

        assert_eq!(repr.class_name(), "DictIteratorRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::DictIteratorRepr);
        assert_eq!(repr.base.variant, vec!["keys".to_string()]);
        assert_eq!(repr.r_dict_lowleveltype, dictptr);
        assert_eq!(
            ReprClassId::DictIteratorRepr.mro(),
            &[
                ReprClassId::DictIteratorRepr,
                ReprClassId::AbstractDictIteratorRepr,
                ReprClassId::Repr
            ]
        );
    }

    /// `ll_dictiter` mints a fresh `dictiter{dict, index}`: `iter.dict = d`
    /// and `iter.index = d.lookup_function_no >> FUNC_SHIFT`
    /// (`rordereddict.py`) — no constant-zero index (unlike
    /// `ll_listiter`), since `_ll_dictnext`'s popitem(last=False)
    /// fast-forward hack stashes a resume offset in `lookup_function_no`.
    #[test]
    fn build_ll_dictiter_helper_mints_dict_and_shifted_index() {
        let dict_lltype = sample_dict_ptr_lltype();
        let iter_lltype = get_ll_dictiter(dict_lltype.clone());
        let helper =
            build_ll_dictiter_helper_graph("ll_dictiter", iter_lltype.clone(), dict_lltype)
                .expect("build_ll_dictiter_helper_graph");
        assert_eq!(helper.func.name, "ll_dictiter");
        let inner = helper.graph.borrow();

        let start_ops: Vec<_> = inner
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(
            start_ops,
            vec!["malloc", "setfield", "getfield", "int_rshift", "setfield"],
        );
        // single-block CFG: startblock links straight to the returnblock.
        assert_eq!(inner.startblock.borrow().exits.len(), 1);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(iter_lltype),
            "ll_dictiter returns the dictiter Ptr"
        );
    }

    /// `OrderedDictRepr::make_iterator_repr` (`rordereddict.py`)
    /// mints a `DictIteratorRepr` for the implicit `for k in d:` iteration
    /// (no variant); the `reversed(d)` variant is rejected fail-closed since
    /// `ll_dictiter_reversed`/`_ll_dictnext_reversed` are unported.
    #[test]
    fn make_iterator_repr_yields_dictiteratorrepr_and_rejects_reversed_variant() {
        let r_dict = sample_ordered_dict_repr();

        let it = r_dict
            .make_iterator_repr(&[], false)
            .expect("make_iterator_repr");
        assert_eq!(it.class_name(), "DictIteratorRepr");
        assert_eq!(it.repr_class_id(), ReprClassId::DictIteratorRepr);

        let err = r_dict
            .make_iterator_repr(&["reversed".to_string()], false)
            .expect_err("reversed variant must be rejected");
        assert!(err.is_missing_rtype_operation());
    }

    /// `DictIteratorRepr::new` selects the `keys`/`values`/`items` variant
    /// (`rordereddict.py` — `rtype_method_iterkeys`/`itervalues`/
    /// `iteritems` each construct `DictIteratorRepr(self, "<variant>")`).
    #[test]
    fn dictiteratorrepr_new_selects_keys_values_items_variant() {
        let r_dict = sample_ordered_dict_repr();
        for variant in ["keys", "values", "items"] {
            let repr = DictIteratorRepr::new(&r_dict, variant);
            assert_eq!(repr.base.variant, vec![variant.to_string()]);
        }
    }

    fn hop_llops(rtyper: &Rc<RPythonTyper>) -> Rc<RefCell<LowLevelOpList>> {
        Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)))
    }

    /// `iter(d)` rtypes through `DictIteratorRepr::newiter` to a
    /// `direct_call(ll_dictiter, v_dict)` (`rdict.py`).
    #[test]
    fn newiter_emits_direct_call_to_ll_dictiter() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let r_dict = OrderedDictRepr::new(
            rtyper.clone(),
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef.clone(),
            None,
            false,
            false,
        )
        .expect("ordered dict repr");
        let iter_repr = DictIteratorRepr::new(&r_dict, "keys");
        let r_dict_arc: Arc<dyn Repr> = Arc::new(r_dict);

        let llops = hop_llops(&rtyper);
        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(r_dict_arc.lowleveltype().clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(iter_repr.lowleveltype().clone()));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "iter".to_string(),
                vec![Hlvalue::Variable(v_dict)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Dict(crate::annotator::model::SomeDict::new(
                dictdef,
            )));
        hop.args_r.borrow_mut().push(Some(r_dict_arc));

        let result = iter_repr
            .newiter(&hop)
            .unwrap_or_else(|err| panic!("dict iter: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_dictiter"),
            "expected 'll_dictiter' in {dbg}"
        );
    }

    /// `next(iter)` rtypes through `DictIteratorRepr::rtype_next`
    /// (`rdict.py`): `direct_call(_ll_dictnext_look_inside_iff, v_iter)`
    /// (`rlib/jit.py look_inside_iff` wrapper), then
    /// `getfield(v_iter, 'dict')` / `getfield(v_dict, 'entries')`, then the
    /// `variant_keys` read `getinteriorfield(entries, index, 'key')`.
    #[test]
    fn rtype_next_keys_variant_walks_dict_entries_then_key() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let r_dict = OrderedDictRepr::new(
            rtyper.clone(),
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef.clone(),
            None,
            false,
            false,
        )
        .expect("ordered dict repr");
        let iter_repr: Arc<dyn Repr> = Arc::new(DictIteratorRepr::new(&r_dict, "keys"));

        let llops = hop_llops(&rtyper);
        let v_iter = Variable::new();
        v_iter.set_concretetype(Some(iter_repr.lowleveltype().clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "next".to_string(),
                vec![Hlvalue::Variable(v_iter)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().push(SomeValue::Iterator(
            crate::annotator::model::SomeIterator::new(
                SomeValue::Dict(crate::annotator::model::SomeDict::new(dictdef)),
                vec![],
            ),
        ));
        hop.args_r.borrow_mut().push(Some(iter_repr.clone()));

        let result = iter_repr
            .rtype_next(&hop)
            .unwrap_or_else(|err| panic!("dict next: {err:?}"));
        assert!(result.is_some());
        let ops = llops.borrow();
        let opnames: Vec<&str> = ops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(
            opnames,
            vec!["direct_call", "getfield", "getfield", "getinteriorfield"],
        );
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_next must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("_ll_dictnext_look_inside_iff"),
            "expected '_ll_dictnext_look_inside_iff' in {dbg}"
        );

        let find = |name: &str| {
            ann.translator
                .graphs
                .borrow()
                .iter()
                .find(|g| g.borrow().name == name)
                .unwrap_or_else(|| panic!("missing graph {name}"))
                .clone()
        };
        let orig_func = find("_ll_dictnext")
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("orig GraphFunc");
        assert_eq!(
            orig_func.oopspec, None,
            "look_inside_iff deletes orig oopspec (`rlib/jit.py look_inside_iff`)"
        );
        let tramp_func = find("_ll_dictnext_trampoline")
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("trampoline GraphFunc");
        assert_eq!(
            tramp_func.oopspec.as_deref(),
            Some("odictiter.next(iter)"),
            "look_inside_iff moves oopspec onto the trampoline"
        );
    }

    fn sample_dict_ptr_lltype() -> LowLevelType {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let repr = OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr");
        repr.lowleveltype().clone()
    }

    fn sample_ordered_dict_repr() -> OrderedDictRepr {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr")
    }

    /// `ll_dict_len` is a one-block graph reading the `num_live_items`
    /// header field and returning it as `Signed`.
    #[test]
    fn build_ll_dict_len_reads_num_live_items_field() {
        let helper = build_ll_dict_len_helper_graph("ll_dict_len", sample_dict_ptr_lltype())
            .expect("build_ll_dict_len_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_len");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getfield"]);
        let field = &startblock.operations[0].args[1];
        assert!(
            matches!(field, Hlvalue::Constant(c) if c.value == ConstValue::byte_str("num_live_items")),
            "len helper must read num_live_items, got {field:?}"
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "ll_dict_len returns Signed"
        );
    }

    /// `ll_dict_bool` branches on `ptr_nonzero(d)`: the False arm returns the
    /// `False` constant without dereferencing the receiver; the True arm
    /// forwards `d` into a block that reads `num_live_items` and compares it
    /// `!= 0`.
    #[test]
    fn build_ll_dict_bool_guards_null_then_checks_num_live_items() {
        let helper = build_ll_dict_bool_helper_graph("ll_dict_bool", sample_dict_ptr_lltype())
            .expect("build_ll_dict_bool_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_bool");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();

        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);

        // False arm: returns the False constant straight to the returnblock.
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit link present");
        let false_first_arg = false_link
            .borrow()
            .args
            .first()
            .and_then(|opt| opt.as_ref())
            .cloned()
            .expect("False link first arg present");
        assert!(matches!(
            false_first_arg,
            Hlvalue::Constant(c) if c.value == ConstValue::Bool(false)
        ));

        // True arm: forwards `d` into check_len = getfield(num_live_items) +
        // int_ne(n, 0).
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let check_len = true_link
            .borrow()
            .target
            .clone()
            .expect("True link target block");
        let check_ops: Vec<String> = check_len
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(check_ops, vec!["getfield", "int_ne"]);
        let getfield_field = check_len.borrow().operations[0].args[1].clone();
        assert!(
            matches!(getfield_field, Hlvalue::Constant(c) if c.value == ConstValue::byte_str("num_live_items")),
            "bool helper must read num_live_items"
        );

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_dict_bool returns Bool"
        );
    }

    /// `ll_newdict` is a single-block graph: `malloc(DICT)` for the header,
    /// `malloc_varsize(DICTENTRYARRAY, 0)` for the empty entries array, then
    /// five `setfield`s (`entries`, `num_live_items`, `num_ever_used_items`,
    /// `lookup_function_no`, `indexes`) before returning the fresh `d`.
    #[test]
    fn build_ll_dict_newdict_allocates_and_initializes_dict() {
        let repr = sample_ordered_dict_repr();
        let helper = build_ll_dict_newdict_helper_graph(
            "ll_newdict",
            repr.lowleveltype().clone(),
            repr.DICT.clone(),
            repr.DICTENTRYARRAY.clone(),
        )
        .expect("build_ll_dict_newdict_helper_graph");
        assert_eq!(helper.func.name, "ll_newdict");
        let inner = helper.graph.borrow();

        // No runtime arguments — the DICT specialization is baked into the
        // cached helper shape, not threaded through as a Void const.
        assert!(inner.startblock.borrow().inputargs.is_empty());

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "malloc",         // d = DICT.allocate()
                "malloc_varsize", // _ll_empty_array(DICT): entries = allocate(0)
                "setfield",       // d.entries = entries
                "setfield",       // d.num_live_items = 0
                "setfield",       // d.num_ever_used_items = 0
                "setfield",       // d.lookup_function_no = FUNC_MUST_REINDEX
                "setfield",       // d.indexes = nullptr(GCREF.TO)
            ]
        );

        // malloc_varsize's length argument is the constant 0.
        let entries_len = &startblock.operations[1].args[2];
        assert!(
            matches!(entries_len, Hlvalue::Constant(c) if c.value == ConstValue::Int(0)),
            "entries array must be allocated with length 0, got {entries_len:?}"
        );

        // Field-name constants, in setfield emission order.
        let field_names: Vec<ConstValue> = startblock.operations[2..]
            .iter()
            .map(|op| match &op.args[1] {
                Hlvalue::Constant(c) => c.value.clone(),
                other => panic!("expected Constant field name, got {other:?}"),
            })
            .collect();
        assert_eq!(
            field_names,
            vec![
                ConstValue::byte_str("entries"),
                ConstValue::byte_str("num_live_items"),
                ConstValue::byte_str("num_ever_used_items"),
                ConstValue::byte_str("lookup_function_no"),
                ConstValue::byte_str("indexes"),
            ]
        );

        // lookup_function_no is set to the FUNC_MUST_REINDEX selector.
        let lookup_fn_value = &startblock.operations[5].args[2];
        assert!(
            matches!(lookup_fn_value, Hlvalue::Constant(c) if c.value == ConstValue::Int(FUNC_MUST_REINDEX)),
            "lookup_function_no must be seeded FUNC_MUST_REINDEX, got {lookup_fn_value:?}"
        );

        // indexes is set to a null GCREF constant.
        let indexes_value = &startblock.operations[6].args[2];
        assert!(
            matches!(indexes_value, Hlvalue::Constant(c) if c.value == ConstValue::None),
            "indexes must be seeded null, got {indexes_value:?}"
        );

        assert!(startblock.exitswitch.is_none());
        assert_eq!(startblock.exits.len(), 1);
        drop(startblock);

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(repr.lowleveltype().clone()),
            "ll_newdict returns Ptr(DICT)"
        );
    }

    /// `_ll_write_indexes` is a single-block store: getfield(d,"indexes") ->
    /// cast_opaque_ptr (GCREF->INDEXES) -> cast_int_to_uint(value) ->
    /// setarrayitem(indexes, i, cast_value), Void return.
    #[test]
    fn build_ll_write_indexes_casts_gcref_then_stores_slot() {
        let helper = build_ll_write_indexes_helper_graph(
            "_ll_write_indexes",
            sample_dict_ptr_lltype(),
            LowLevelType::Unsigned,
        )
        .expect("build_ll_write_indexes_helper_graph");
        assert_eq!(helper.func.name, "_ll_write_indexes");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 3); // d, i, value
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec![
                "getfield",
                "cast_opaque_ptr",
                "cast_int_to_uint",
                "setarrayitem"
            ]
        );
        let field = &startblock.operations[0].args[1];
        assert!(
            matches!(field, Hlvalue::Constant(c) if c.value == ConstValue::byte_str("indexes")),
            "first op must read the indexes field, got {field:?}"
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    /// Int-keyed sample `(dict_ptr, entries_ptr, key_lltype)` for the lookup
    /// builder, derived from a freshly built `OrderedDictRepr`.
    fn sample_dict_lookup_lltypes() -> (LowLevelType, LowLevelType, LowLevelType) {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let repr = OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr");
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        (
            repr.lowleveltype().clone(),
            entries_ptr,
            repr.DICTKEY.clone(),
        )
    }

    /// Str-keyed sample `(dict_ptr, entries_ptr, key_lltype)` for the lookup
    /// builder, derived from a freshly built `OrderedDictRepr`.
    fn sample_dict_lookup_str_key_lltypes() -> (LowLevelType, LowLevelType, LowLevelType) {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::String(SomeString::new(false, false)),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let repr = OrderedDictRepr::new(
            rtyper,
            string_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr");
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        (
            repr.lowleveltype().clone(),
            entries_ptr,
            repr.DICTKEY.clone(),
        )
    }

    fn startblock_arg_names(graph: &FunctionGraph) -> Vec<String> {
        graph
            .startblock
            .borrow()
            .inputargs
            .iter()
            .map(|hlv| match hlv {
                Hlvalue::Variable(v) => {
                    let name = if v.renamed() {
                        v.name_prefix()
                    } else {
                        "arg".to_string()
                    };
                    name.strip_suffix('_').unwrap_or(&name).to_string()
                }
                Hlvalue::Constant(_) => panic!("startblock inputarg is a Constant"),
            })
            .collect()
    }

    /// Walk every block reachable from `start`, returning the visited block
    /// count and the flattened op-name list.
    fn walk_blocks(start: &BlockRef) -> (usize, Vec<String>) {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![start.clone()];
        let mut ops = Vec::new();
        let mut count = 0usize;
        while let Some(b) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&b) as usize) {
                continue;
            }
            count += 1;
            let bb = b.borrow();
            for op in &bb.operations {
                ops.push(op.opname.clone());
            }
            for link in &bb.exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        (count, ops)
    }

    fn bool_exit(block: &BlockRef, flag: bool) -> BlockRef {
        block
            .borrow()
            .exits
            .iter()
            .find(|l| {
                matches!(
                    l.borrow().exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(flag)
                )
            })
            .unwrap_or_else(|| panic!("missing {flag} exit"))
            .borrow()
            .target
            .clone()
            .expect("exit target")
    }

    fn reads_f_hash(block: &BlockRef) -> bool {
        block.borrow().operations.iter().any(|op| {
            op.opname == "getinteriorfield"
                && op.args.iter().any(|a| {
                    matches!(a, Hlvalue::Constant(c) if c.value == ConstValue::byte_str("f_hash"))
                })
        })
    }

    fn graph_mentions_field(start: &BlockRef, field: &str) -> bool {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![start.clone()];
        while let Some(b) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&b) as usize) {
                continue;
            }
            let bb = b.borrow();
            for op in &bb.operations {
                if op.args.iter().any(
                    |a| matches!(a, Hlvalue::Constant(c) if c.value == ConstValue::byte_str(field)),
                ) {
                    return true;
                }
            }
            for link in &bb.exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        false
    }

    fn assert_link_variables_defined(start: &BlockRef) {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![start.clone()];
        while let Some(b) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&b) as usize) {
                continue;
            }
            let bb = b.borrow();
            let mut defined: Vec<&Variable> = Vec::new();
            for a in &bb.inputargs {
                if let Hlvalue::Variable(v) = a {
                    defined.push(v);
                }
            }
            for op in &bb.operations {
                if let Hlvalue::Variable(v) = &op.result {
                    defined.push(v);
                }
            }
            for (link_i, link) in bb.exits.iter().enumerate() {
                let l = link.borrow();
                for (arg_i, arg) in l.args.iter().enumerate() {
                    let Some(hl) = arg.as_ref() else {
                        continue;
                    };
                    if let Hlvalue::Variable(v) = hl {
                        assert!(
                            defined.iter().any(|d| *d == v),
                            "link[{link_i}].args[{arg_i}] variable {} is not defined in the source block",
                            v.name()
                        );
                    }
                }
                if let Some(t) = l.target.clone() {
                    stack.push(t);
                }
            }
        }
    }

    /// `ll_dict_lookup` is the open-addressing perturb-probe. Validate the
    /// first-probe header, the full 13-block + returnblock CFG shape, the
    /// unsigned probe arithmetic (logical `uint_rshift` for the perturb
    /// shift), the interior key read, the inlined store, and the Signed slot
    /// return.
    #[test]
    fn build_ll_dict_lookup_assembles_perturb_probe_cfg() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_lookup_helper_graph(
            "ll_dict_lookup",
            dict_ptr,
            entries_ptr,
            LowLevelType::Unsigned,
            key_lltype,
            None,
            false,
            false,
            None,
            false,
        )
        .expect("build_ll_dict_lookup_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_lookup");
        let inner = helper.graph.borrow();

        // First-probe header: read entries + indexes, cast the GCREF index
        // array, derive mask, hash & mask, read+cast the slot, branch on
        // index >= VALID_OFFSET.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "getfield",         // d.entries
                "getfield",         // d.indexes (GCREF)
                "cast_opaque_ptr",  // cast_opaque_ptr(INDEXES, ...)
                "getarraysize",     // len(indexes)
                "int_sub",          // mask = len - 1
                "cast_int_to_uint", // mask_u
                "int_and",          // hash & mask
                "cast_int_to_uint", // i = r_uint(...)
                "cast_uint_to_int", // intmask(i)
                "getarrayitem",     // indexes[intmask(i)]
                "cast_uint_to_int", // rffi.cast(Signed, ...)
                "int_ge",           // index >= VALID_OFFSET
            ]
        );
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);
        assert_eq!(
            startblock_arg_names(&inner),
            ["d", "key", "hash", "store_flag", "T"]
        );

        // 13 work blocks + the returnblock are all reachable.
        let (block_count, ops) = walk_blocks(&inner.startblock);
        assert_eq!(block_count, 14, "13 work blocks + returnblock");

        // Distinctive ops of the probe must all appear.
        for needed in [
            "uint_lshift",      // i << 2
            "uint_add",         // + i / + perturb / + 1
            "uint_and",         // & mask
            "uint_rshift",      // perturb >>= PERTURB_SHIFT (logical!)
            "getinteriorfield", // entries[slot].key
            "int_eq",           // FREE / DELETED / FLAG_STORE / key (Signed)
            "setarrayitem",     // inlined _ll_write_indexes store
        ] {
            assert!(
                ops.iter().any(|o| o == needed),
                "lookup CFG must emit {needed}, got {ops:?}"
            );
        }
        // eq_fn_const is None (identity/int keys) — no keyeq call branch, no
        // direct_call anywhere in the CFG.
        assert!(
            !ops.iter().any(|o| o == "direct_call"),
            "eq_fn_const=None must stay direct-compare-only, got {ops:?}"
        );

        // Returns the Signed entry slot (index - VALID_OFFSET) or -1.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "ll_dict_lookup returns Signed"
        );
    }

    #[test]
    fn ll_dict_lookup_surface_builds_lookup_graph() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = ll_dict_lookup(
            dict_ptr,
            entries_ptr,
            LowLevelType::Unsigned,
            key_lltype,
            None,
        )
        .expect("ll_dict_lookup");
        assert_eq!(helper.func.name, "ll_dict_lookup");
    }

    #[test]
    fn build_ll_dictiter_allocates_and_initializes_iterator() {
        let (dict_ptr, _entries_ptr, _key) = sample_dict_lookup_lltypes();
        let iter_ptr = get_ll_dictiter(dict_ptr.clone());
        let helper = ll_dictiter(iter_ptr, dict_ptr).expect("ll_dictiter");
        assert_eq!(helper.func.name, "ll_dictiter");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec!["malloc", "setfield", "getfield", "int_rshift", "setfield"]
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert!(matches!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Ptr(_))
        ));
    }

    /// When `eq_fn_const` is `Some` (str keys), a direct-compare miss at
    /// both comparison sites hashes first (`entries.entry_hash == hash`)
    /// then `direct_call(eq_fn_const, checkingkey, key)` —
    /// `d.keyeq(checkingkey, key)` (`ll_dict_lookup`).
    /// Validate the block count grows by the hash-check + keyeq-call blocks
    /// and that `direct_call` reaches the minted `ll_streq` funcptr.
    #[test]
    fn build_ll_dict_lookup_wires_keyeq_call_for_str_keys() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_str_key_lltypes();
        let eq_fn = dummy_funcptr_const();
        let helper = build_ll_dict_lookup_helper_graph(
            "ll_dict_lookup",
            dict_ptr,
            entries_ptr,
            LowLevelType::Unsigned,
            key_lltype,
            Some(eq_fn),
            false,
            false,
            None,
            false,
        )
        .expect("build_ll_dict_lookup_helper_graph");
        let inner = helper.graph.borrow();

        let (block_count, ops) = walk_blocks(&inner.startblock);
        assert_eq!(
            block_count, 18,
            "13 work blocks + 2 hash-check + 2 keyeq-call blocks + returnblock"
        );
        let direct_call_count = ops.iter().filter(|o| *o == "direct_call").count();
        assert_eq!(
            direct_call_count, 2,
            "one direct_call per comparison site (first-try + in-loop), got {ops:?}"
        );
    }

    /// `ll_dict_lookup` calls `d.keyeq` only when `entries.entry_hash == hash`.
    #[test]
    fn build_ll_dict_lookup_gates_keyeq_on_entry_hash() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_str_key_lltypes();
        let helper = build_ll_dict_lookup_helper_graph(
            "ll_dict_lookup",
            dict_ptr,
            entries_ptr,
            LowLevelType::Unsigned,
            key_lltype,
            Some(dummy_funcptr_const()),
            false,
            false,
            None,
            false,
        )
        .expect("build_ll_dict_lookup_helper_graph");
        let inner = helper.graph.borrow();
        let first_valid = bool_exit(&inner.startblock, true);
        let hashcheck = bool_exit(&first_valid, false);
        assert!(
            reads_f_hash(&hashcheck),
            "direct-compare miss must read entries.entry_hash before keyeq"
        );
        let hashcheck_ops: Vec<String> = hashcheck
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert!(
            hashcheck_ops.iter().any(|o| o == "int_eq"),
            "entry_hash must be compared to the lookup hash, got {hashcheck_ops:?}"
        );
        assert!(
            !hashcheck_ops.iter().any(|o| o == "direct_call"),
            "hash mismatch must not call keyeq, got {hashcheck_ops:?}"
        );
        let keyeq = bool_exit(&hashcheck, true);
        assert!(
            keyeq
                .borrow()
                .operations
                .iter()
                .any(|op| op.opname == "direct_call"),
            "matching hash falls through to d.keyeq"
        );
        let miss = bool_exit(&hashcheck, false);
        assert!(
            !miss
                .borrow()
                .operations
                .iter()
                .any(|op| op.opname == "direct_call"),
            "hash miss continues the probe without keyeq"
        );

        let mut f_hash_reads = 0usize;
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![inner.startblock.clone()];
        while let Some(b) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&b) as usize) {
                continue;
            }
            if reads_f_hash(&b) {
                f_hash_reads += 1;
            }
            for link in &b.borrow().exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        assert_eq!(
            f_hash_reads, 2,
            "first-try and in-loop sites both gate keyeq on entry_hash"
        );
    }

    /// `_ll_dictnext` walks the entries array; on a valid entry it advances
    /// `iter.index` and returns the slot, on exhaustion it nulls `iter.dict`
    /// and raises StopIteration. Validate the null-guard header, the 8-block +
    /// return + except CFG, the `f_valid` interior read, the field writes, the
    /// popitem fast-forward hack ops, and the Signed return.
    #[test]
    fn build_ll_dictnext_walks_entries_then_raises_stopiteration() {
        let (dict_ptr, entries_ptr, _key) = sample_dict_lookup_lltypes();
        let iter_ptr = get_ll_dictiter(dict_ptr.clone());
        let helper =
            build_ll_dictnext_helper_graph("_ll_dictnext", iter_ptr, dict_ptr, entries_ptr)
                .expect("build_ll_dictnext_helper_graph");
        assert_eq!(helper.func.name, "_ll_dictnext");
        let inner = helper.graph.borrow();

        // Null-guard header: dict = iter.dict; ptr_nonzero(dict); branch.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getfield", "ptr_nonzero"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        // 8 work blocks + returnblock + exceptblock all reachable.
        let (block_count, ops) = walk_blocks(&inner.startblock);
        assert_eq!(block_count, 10, "8 work blocks + returnblock + exceptblock");

        for needed in [
            "getinteriorfield", // entries.valid(index) = f_valid
            "setfield",         // iter.index = nextindex / iter.dict = null / lfn bump
            "int_lt",           // index < entries_len
            "int_add",          // nextindex / lfn += (1 << FUNC_SHIFT)
            "int_rshift",       // lookup_function_no >> FUNC_SHIFT
            "int_eq",           // index == (lookup_function_no >> FUNC_SHIFT)
            "ptr_nonzero",      // if dict
        ] {
            assert!(
                ops.iter().any(|o| o == needed),
                "dictnext CFG must emit {needed}, got {ops:?}"
            );
        }

        // Returns the Signed entry index.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "_ll_dictnext returns Signed"
        );
        assert_eq!(
            helper.func.oopspec.as_deref(),
            Some("odictiter.next(iter)"),
            "builder stamps oopspec; mint_look_inside_iff moves it onto the trampoline"
        );
    }

    /// `rlib/jit.py look_inside_iff` on `_ll_dictnext`: orig has no oopspec,
    /// `_ll_dictnext_trampoline` carries `odictiter.next(iter)`, wrapper is
    /// `_ll_dictnext_look_inside_iff`.
    #[test]
    fn ll_dictnext_look_inside_iff_moves_oopspec_onto_trampoline() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let (dict_ptr, entries_ptr, _key) = sample_dict_lookup_lltypes();
        let iter_ptr = get_ll_dictiter(dict_ptr.clone());
        let wrapper = mint_look_inside_iff(
            &rtyper,
            "_ll_dictnext",
            &["iter"],
            vec![iter_ptr.clone()],
            LowLevelType::Signed,
            {
                let iter_ptr = iter_ptr;
                let dict_ptr = dict_ptr;
                let entries_ptr = entries_ptr;
                move |name| {
                    build_ll_dictnext_helper_graph(
                        name,
                        iter_ptr.clone(),
                        dict_ptr.clone(),
                        entries_ptr.clone(),
                    )
                }
            },
            &[
                LookInsidePred::IsVirtual(0),
                LookInsidePred::FieldNullOrIsVirtual {
                    arg: 0,
                    field: "dict",
                },
            ],
        )
        .expect("mint_look_inside_iff");
        assert_eq!(wrapper.name, "_ll_dictnext_look_inside_iff");

        let find = |name: &str| {
            ann.translator
                .graphs
                .borrow()
                .iter()
                .find(|g| g.borrow().name == name)
                .unwrap_or_else(|| panic!("missing graph {name}"))
                .clone()
        };
        let orig_func = find("_ll_dictnext")
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("orig GraphFunc");
        assert_eq!(
            orig_func.oopspec, None,
            "look_inside_iff deletes orig oopspec (`rlib/jit.py look_inside_iff`)"
        );
        let tramp_func = find("_ll_dictnext_trampoline")
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("trampoline GraphFunc");
        assert_eq!(
            tramp_func._jit_look_inside_,
            Some(false),
            "trampoline is dont_look_inside"
        );
        assert_eq!(
            tramp_func.oopspec.as_deref(),
            Some("odictiter.next(iter)"),
            "look_inside_iff moves oopspec onto the trampoline"
        );
    }

    fn dummy_funcptr_const() -> Constant {
        Constant::with_concretetype(ConstValue::None, LowLevelType::Void)
    }

    fn tagged_funcptr_const(tag: &str) -> Constant {
        Constant::with_concretetype(ConstValue::byte_str(tag), LowLevelType::Void)
    }

    fn direct_call_target_values(start: &BlockRef) -> Vec<ConstValue> {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![start.clone()];
        let mut targets = Vec::new();
        while let Some(b) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&b) as usize) {
                continue;
            }
            let bb = b.borrow();
            for op in &bb.operations {
                if op.opname == "direct_call"
                    && let Some(Hlvalue::Constant(c)) = op.args.first()
                {
                    targets.push(c.value.clone());
                }
            }
            for link in &bb.exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        targets
    }

    /// `ll_malloc_indexes_and_choose_lookup` cascades `n <= 256` / `<= 65536`
    /// / (64-bit) `<= 2**32` / else, each leaf allocating the (uniformly
    /// `Unsigned`-element) index array and stamping `indexes` +
    /// `lookup_function_no`.
    #[test]
    fn build_ll_malloc_indexes_and_choose_lookup_assembles_threshold_cascade() {
        let helper = build_ll_malloc_indexes_and_choose_lookup_helper_graph(
            "ll_malloc_indexes_and_choose_lookup",
            sample_dict_ptr_lltype(),
        )
        .expect("build_ll_malloc_indexes_and_choose_lookup_helper_graph");
        assert_eq!(helper.func.name, "ll_malloc_indexes_and_choose_lookup");
        let inner = helper.graph.borrow();

        // startblock just forwards (d, n) into the first threshold check.
        let startblock = inner.startblock.borrow();
        assert!(startblock.operations.is_empty());
        assert_eq!(startblock.exits.len(), 1);
        drop(startblock);

        let (_block_count, ops) = walk_blocks(&inner.startblock);
        assert_eq!(
            ops.iter().filter(|o| *o == "int_le").count(),
            3,
            "3 threshold checks on a 64-bit target (BYTE/SHORT/INT), got {ops:?}"
        );
        assert_eq!(
            ops.iter().filter(|o| *o == "malloc_varsize").count(),
            4,
            "4 leaves (BYTE/SHORT/INT/LONG), got {ops:?}"
        );
        assert_eq!(
            ops.iter().filter(|o| *o == "cast_opaque_ptr").count(),
            4,
            "each leaf casts the freshly malloc'd array to GCREF, got {ops:?}"
        );
        assert_eq!(
            ops.iter().filter(|o| *o == "setfield").count(),
            8,
            "each leaf sets indexes + lookup_function_no, got {ops:?}"
        );

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Void),
            "ll_malloc_indexes_and_choose_lookup returns Void"
        );
    }

    /// `ll_dict_create_initial_index` (`rordereddict.py`): empty dicts
    /// call `ll_malloc_indexes_and_choose_lookup(d, DICT_INITSIZE)` then
    /// set `resize_counter = DICT_INITSIZE * 2`; nonempty dicts call
    /// `ll_dict_rehash_after_translation(d)`.
    #[test]
    fn build_ll_dict_create_initial_index_calls_malloc_choose_then_sets_resize_counter() {
        let helper = build_ll_dict_create_initial_index_helper_graph(
            "ll_dict_create_initial_index",
            sample_dict_ptr_lltype(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_dict_create_initial_index_helper_graph");
        let inner = helper.graph.borrow();
        {
            let startblock = inner.startblock.borrow();
            let start_ops: Vec<&str> = startblock
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(start_ops, vec!["getfield", "int_eq"]);
            assert!(startblock.exitswitch.is_some());
        }

        let block_empty = bool_exit(&inner.startblock, true);
        let empty = block_empty.borrow();
        let empty_ops: Vec<&str> = empty
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(empty_ops, vec!["direct_call", "setfield"]);
        let call_args = &empty.operations[0].args;
        assert!(
            matches!(&call_args[2], Hlvalue::Constant(c) if c.value == ConstValue::Int(DICT_INITSIZE)),
            "malloc/choose call must pass DICT_INITSIZE, got {call_args:?}"
        );
        let setfield_args = &empty.operations[1].args;
        assert!(
            matches!(&setfield_args[1], Hlvalue::Constant(c) if c.value == ConstValue::byte_str("resize_counter"))
        );
        assert!(
            matches!(&setfield_args[2], Hlvalue::Constant(c) if c.value == ConstValue::Int(DICT_INITSIZE * 2))
        );
        drop(empty);

        let block_rehash = bool_exit(&inner.startblock, false);
        assert_eq!(
            block_rehash.borrow().operations[0].opname,
            "direct_call",
            "nonempty arm calls ll_dict_rehash_after_translation"
        );
    }

    /// `ll_ensure_indexes` reads `lookup_function_no`, branches on `==
    /// FUNC_MUST_REINDEX`; the True arm calls `ll_dict_create_initial_index`,
    /// the False arm (`ll_assert`, debug-only) just returns.
    #[test]
    fn build_ll_ensure_indexes_branches_on_must_reindex() {
        let helper = build_ll_ensure_indexes_helper_graph(
            "ll_ensure_indexes",
            sample_dict_ptr_lltype(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_ensure_indexes_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getfield", "int_eq"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);

        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let block_reindex = true_link
            .borrow()
            .target
            .clone()
            .expect("True link target block");
        assert_eq!(
            block_reindex.borrow().operations[0].opname,
            "direct_call",
            "must_reindex arm calls ll_dict_create_initial_index"
        );
    }

    /// `ll_call_lookup_function` collapses the upstream `FUNC_*` 4-way
    /// dispatch loop to "ensure the index exists, then look up once" — see
    /// the builder's doc comment for why that is behaviorally identical
    /// given this port's collapsed `DICTINDEX_*` widths.
    #[test]
    fn build_ll_call_lookup_function_calls_ensure_indexes_then_lookup() {
        let (dict_ptr, _entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_call_lookup_function_helper_graph(
            "ll_call_lookup_function",
            dict_ptr,
            key_lltype,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_call_lookup_function_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["direct_call", "direct_call"]);
        let lookup = &startblock.operations[1];
        assert_eq!(
            lookup.args.len(),
            6,
            "ll_dict_lookup(d, key, hash, flag, T)"
        );
        match &lookup.args[5] {
            Hlvalue::Constant(c) => {
                assert_eq!(c.concretetype, Some(LowLevelType::Void));
                assert_eq!(
                    c.value,
                    ConstValue::LowLevelType(Box::new(TYPE_LONG.clone()))
                );
            }
            other => panic!("T is a Void lltype constant, got {other:?}"),
        }

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "ll_call_lookup_function returns Signed"
        );
    }

    /// `ll_dict_getitem` hashes the key, looks it up, and either reads
    /// `entries[index].value` or raises `KeyError`.
    #[test]
    fn build_ll_dict_getitem_reads_value_or_raises_keyerror() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_getitem_helper_graph(
            "ll_dict_getitem",
            dict_ptr,
            entries_ptr,
            key_lltype,
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            false,
        )
        .expect("build_ll_dict_getitem_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["direct_call", "direct_call", "int_ge"]);
        assert_eq!(startblock.exits.len(), 2);

        // The False (index < 0) arm raises KeyError via the exceptblock.
        let raise_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit link present");
        assert!(Rc::ptr_eq(
            &raise_link
                .borrow()
                .target
                .clone()
                .expect("exceptblock target"),
            &inner.exceptblock,
        ));

        // The True (index >= 0) arm reads entries[index].value.
        let valid_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let block_valid = valid_link
            .borrow()
            .target
            .clone()
            .expect("True link target block");
        let valid_ops: Vec<String> = block_valid
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(valid_ops, vec!["getfield", "getinteriorfield"]);

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "ll_dict_getitem returns the value lltype"
        );
    }

    /// `ll_dict_contains` hashes the key, looks it up, and returns `index >=
    /// 0` — no raise, unlike `ll_dict_getitem`.
    #[test]
    fn build_ll_dict_contains_returns_index_ge_zero() {
        let (dict_ptr, _entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_contains_helper_graph(
            "ll_dict_contains",
            dict_ptr,
            key_lltype,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            false,
        )
        .expect("build_ll_dict_contains_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["direct_call", "direct_call", "int_ge"]);
        assert!(startblock.exitswitch.is_none(), "contains never branches");

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_dict_contains returns Bool"
        );
    }

    // NOTE: takes an already-constructed `rtyper` rather than building its
    // own `RPythonAnnotator` — `RPythonTyper::annotator` is only a `Weak`
    // reference (rtyper.rs, cycle avoidance), so a helper that builds and
    // drops its own annotator leaves every `lowlevel_helper_function_with_builder`
    // call downstream hitting "annotator weak reference dropped". Callers
    // must keep their own `ann`/`rtyper` alive for the test's duration.
    fn make_int_int_dict_repr(rtyper: Rc<RPythonTyper>) -> Arc<OrderedDictRepr> {
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        Arc::new(
            OrderedDictRepr::new(
                rtyper,
                signed_repr() as Arc<dyn Repr>,
                signed_repr() as Arc<dyn Repr>,
                dictdef,
                None,
                false,
                false,
            )
            .expect("ordered dict repr"),
        )
    }

    /// End-to-end wiring: `OrderedDictRepr::rtype_getitem` on an int-keyed
    /// dict (direct-compare, no eq-gate) emits the hash + lookup
    /// `direct_call` chain and threads `hop.exception_is_here()`.
    #[test]
    fn ordereddictrepr_rtype_getitem_int_key_emits_direct_call_chain() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_int_int_dict_repr(rtyper);
        let dict_lltype = repr.lowleveltype().clone();
        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            repr.base.rtyper.clone(),
            None,
        )));

        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));

        // A caught KeyError exitcase on this getitem.
        let exitblock = Block::shared(vec![]);
        let cls_keyerror = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("KeyError")
            .expect("KeyError class");
        let link_keyerror = Link::new(
            vec![],
            Some(exitblock),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                cls_keyerror,
            )))),
        )
        .into_ref();

        let hop = HighLevelOp::new(
            repr.base.rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_dict), Hlvalue::Variable(v_key)],
                Hlvalue::Variable(v_result),
            ),
            vec![link_keyerror],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Impossible, SomeValue::Impossible]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("OrderedDictRepr::rtype_getitem: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_getitem must call hop.exception_is_here()"
        );
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_getitem"));
    }

    /// End-to-end wiring: `pair_ordereddict_repr_rtype_contains` on an
    /// int-keyed dict emits the same hash + lookup chain, wrapped in
    /// `ll_dict_contains` (no raise).
    #[test]
    fn pair_ordereddict_repr_rtype_contains_int_key_emits_direct_call() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_int_int_dict_repr(rtyper);
        let dict_lltype = repr.lowleveltype().clone();
        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            repr.base.rtyper.clone(),
            None,
        )));

        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Bool));

        let hop = HighLevelOp::new(
            repr.base.rtyper.clone(),
            SpaceOperation::new(
                "contains".to_string(),
                vec![Hlvalue::Variable(v_dict), Hlvalue::Variable(v_key)],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Impossible, SomeValue::Impossible]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result =
            pair_ordereddict_repr_rtype_contains(repr.as_ref(), signed_repr().as_ref(), &hop)
                .unwrap_or_else(|err| panic!("pair_ordereddict_repr_rtype_contains: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_contains"));
    }

    /// The eq-gate lets plain (non-`custom_eq_hash`) str keys through: the
    /// direct-compare `ptr_eq` landmine is now covered by the
    /// `d.keyeq`-fallback wiring in `build_ll_dict_lookup_helper_graph`
    /// (`ll_streq`, via `key_repr.get_ll_eq_function()`), so
    /// `OrderedDictRepr::rtype_getitem` succeeds like the int-key case.
    #[test]
    fn ordereddictrepr_rtype_getitem_str_key_emits_direct_call_chain() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::String(SomeString::new(false, false)),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let repr = Arc::new(
            OrderedDictRepr::new(
                rtyper.clone(),
                string_repr() as Arc<dyn Repr>,
                signed_repr() as Arc<dyn Repr>,
                dictdef,
                None,
                false,
                false,
            )
            .expect("ordered dict repr"),
        );
        let dict_lltype = repr.lowleveltype().clone();
        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));

        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(string_repr().lowleveltype().clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));

        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_dict), Hlvalue::Variable(v_key)],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Impossible, SomeValue::Impossible]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(string_repr() as Arc<dyn Repr>),
        ]);

        let result = repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("OrderedDictRepr::rtype_getitem: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_getitem"));
    }

    /// `ll_dict_store_clean` probes for the first `FREE` slot (no key
    /// comparison, no `deletedslot` tracking) via the same unsigned
    /// perturb-probe arithmetic as `ll_dict_lookup`, then calls the
    /// write-indexes helper.
    #[test]
    fn build_ll_dict_store_clean_probes_for_free_slot() {
        let (dict_ptr, _entries_ptr, _key) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_store_clean_helper_graph(
            "ll_dict_store_clean",
            dict_ptr,
            dummy_funcptr_const(),
        )
        .expect("build_ll_dict_store_clean_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_store_clean");
        let inner = helper.graph.borrow();
        assert_eq!(inner.startblock.borrow().inputargs.len(), 3); // d, hash, index

        let (block_count, ops) = walk_blocks(&inner.startblock);
        assert_eq!(
            block_count, 5,
            "startblock + cond + write + advance + returnblock"
        );
        for needed in [
            "getarraysize",
            "cast_int_to_uint",
            "cast_uint_to_int",
            "getarrayitem",
            "int_eq",
            "uint_lshift",
            "uint_add",
            "uint_and",
            "uint_rshift",
            "direct_call",
        ] {
            assert!(
                ops.iter().any(|o| o == needed),
                "store_clean CFG must emit {needed}, got {ops:?}"
            );
        }
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    /// `ll_call_insert_clean_function`'s FUNC_* dispatch collapses to a
    /// single unconditional `ll_dict_store_clean` call.
    #[test]
    fn build_ll_call_insert_clean_function_calls_store_clean_directly() {
        let dict_ptr = sample_dict_ptr_lltype();
        let helper = build_ll_call_insert_clean_function_helper_graph(
            "ll_call_insert_clean_function",
            dict_ptr,
            dummy_funcptr_const(),
        )
        .expect("build_ll_call_insert_clean_function_helper_graph");
        assert_eq!(helper.func.name, "ll_call_insert_clean_function");
        assert_eq!(helper.func._jit_look_inside_, Some(false));
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 3); // d, hash, i
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["direct_call"]);
    }

    /// `ll_call_delete_by_entry_index`'s FUNC_* dispatch collapses to a
    /// single unconditional `ll_dict_delete_by_entry_index` call, matching
    /// the current uniform `DICTINDEX_* = Unsigned` layout.
    #[test]
    fn build_ll_call_delete_by_entry_index_calls_delete_directly() {
        let helper = build_ll_call_delete_by_entry_index_helper_graph(
            "ll_call_delete_by_entry_index",
            sample_dict_ptr_lltype(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_call_delete_by_entry_index_helper_graph");
        assert_eq!(helper.func.name, "ll_call_delete_by_entry_index");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 4); // d, hash, i, replace_with
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["direct_call"]);
    }

    /// `ll_dict_delete_by_entry_index` probes by hash until the stored
    /// `locate_index + VALID_OFFSET` is found, then writes `DELETED` through
    /// `_ll_write_indexes`.
    #[test]
    fn build_ll_dict_delete_by_entry_index_probes_for_locate_value() {
        let helper = build_ll_dict_delete_by_entry_index_helper_graph(
            "ll_dict_delete_by_entry_index",
            sample_dict_ptr_lltype(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_dict_delete_by_entry_index_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_delete_by_entry_index");
        assert!(helper.func._jit_unroll_safe_);
        let inner = helper.graph.borrow();
        assert_eq!(inner.startblock.borrow().inputargs.len(), 4);
        let (_block_count, ops) = walk_blocks(&inner.startblock);
        for needed in [
            "getarrayitem",
            "int_eq",
            "uint_lshift",
            "uint_add",
            "uint_and",
            "uint_rshift",
            "direct_call",
        ] {
            assert!(
                ops.iter().any(|o| o == needed),
                "delete_by_entry_index CFG must emit {needed}, got {ops:?}"
            );
        }
    }

    /// `_ll_dict_entries_size_too_big` masks `lookup_function_no` then
    /// compares `FUNC_BYTE` / `FUNC_SHORT` / (64-bit) `FUNC_INT`; each true
    /// arm asserts `num_live_items < LIMIT` and returns
    /// `new_allocated > LIMIT`. Fall-through returns False.
    #[test]
    fn build__ll_dict_entries_size_too_big_masks_and_compares_func_widths() {
        let helper = build__ll_dict_entries_size_too_big_helper_graph(
            "_ll_dict_entries_size_too_big",
            sample_dict_ptr_lltype(),
        )
        .expect("build__ll_dict_entries_size_too_big_helper_graph");
        assert_eq!(helper.func.name, "_ll_dict_entries_size_too_big");
        crate::flowspace::model::checkgraph(&helper.graph.borrow());
        let inner = helper.graph.borrow();
        let start = inner.startblock.borrow();
        assert_eq!(start.inputargs.len(), 2); // d, new_allocated
        let ops: Vec<&str> = start
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getfield", "int_and"]);
        assert_eq!(
            getfield_field(&start.operations[0]),
            Some(ConstValue::byte_str("lookup_function_no"))
        );
        assert_eq!(start.exits.len(), 1);
        drop(start);

        let mut expected: Vec<(i64, i64)> = vec![
            (FUNC_BYTE, (1i64 << 8) - MIN_INDEXES_MINUS_ENTRIES),
            (FUNC_SHORT, (1i64 << 16) - MIN_INDEXES_MINUS_ENTRIES),
        ];
        if IS_64BIT {
            expected.push((FUNC_INT, (1i64 << 32) - MIN_INDEXES_MINUS_ENTRIES));
        }

        let mut check = {
            let start = inner.startblock.borrow();
            start.exits[0]
                .borrow()
                .target
                .clone()
                .expect("first FUNC_* check")
        };
        for (i, (func, limit)) in expected.iter().enumerate() {
            {
                let cb = check.borrow();
                assert_eq!(cb.operations.len(), 1, "check block is a single int_eq");
                assert_eq!(cb.operations[0].opname, "int_eq");
                match cb.operations[0].args.get(1) {
                    Some(Hlvalue::Constant(c)) => {
                        assert_eq!(c.value, ConstValue::Int(*func), "int_eq against FUNC_*")
                    }
                    other => panic!("int_eq FUNC const, got {other:?}"),
                }
            }

            let leaf = bool_exit(&check, true);
            {
                let leafb = leaf.borrow();
                let leaf_ops: Vec<&str> = leafb
                    .operations
                    .iter()
                    .map(|op| op.opname.as_str())
                    .collect();
                assert_eq!(leaf_ops, vec!["getfield", "int_lt"]);
                assert_eq!(
                    getfield_field(&leafb.operations[0]),
                    Some(ConstValue::byte_str("num_live_items"))
                );
                match leafb.operations[1].args.get(1) {
                    Some(Hlvalue::Constant(c)) => {
                        assert_eq!(c.value, ConstValue::Int(*limit), "assert limit")
                    }
                    other => panic!("int_lt LIMIT const, got {other:?}"),
                }
            }
            assert!(
                Rc::ptr_eq(&bool_exit(&leaf, false), &inner.exceptblock),
                "assert false raises AssertionError via exceptblock"
            );
            let ret_arm = bool_exit(&leaf, true);
            {
                let retb = ret_arm.borrow();
                assert_eq!(retb.operations.len(), 1);
                assert_eq!(retb.operations[0].opname, "int_gt");
                match retb.operations[0].args.get(1) {
                    Some(Hlvalue::Constant(c)) => {
                        assert_eq!(c.value, ConstValue::Int(*limit), "int_gt LIMIT")
                    }
                    other => panic!("int_gt LIMIT const, got {other:?}"),
                }
                assert!(
                    Rc::ptr_eq(
                        retb.exits[0]
                            .borrow()
                            .target
                            .as_ref()
                            .expect("return target"),
                        &inner.returnblock
                    ),
                    "true arm returns Bool"
                );
            }

            let next = bool_exit(&check, false);
            if i + 1 == expected.len() {
                assert!(
                    Rc::ptr_eq(&next, &inner.returnblock),
                    "final false link returns False"
                );
            } else {
                check = next;
            }
        }

        let (_block_count, all_ops) = walk_blocks(&inner.startblock);
        assert_eq!(
            all_ops.iter().filter(|o| *o == "int_eq").count(),
            expected.len(),
            "FUNC_* comparisons, got {all_ops:?}"
        );
        assert_eq!(
            all_ops.iter().filter(|o| *o == "int_gt").count(),
            expected.len(),
            "one int_gt per FUNC_* arm, got {all_ops:?}"
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Bool));
    }

    /// `ll_dict_grow` first restores the upstream deleted-entry compaction
    /// branch (`num_live_items < num_ever_used_items // 2`), otherwise it
    /// calls `_ll_dict_entries_size_too_big` and either compacts or
    /// mallocs + copies + rebinds `d.entries` and returns `False`.
    #[test]
    fn build_ll_dict_grow_compacts_deleted_entries_before_growing() {
        let repr = sample_ordered_dict_repr();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let entries_array_ty = LowLevelType::Array(Box::new(repr.DICTENTRYARRAY.clone()));
        let helper = build_ll_dict_grow_helper_graph(
            "ll_dict_grow",
            repr.lowleveltype().clone(),
            entries_ptr,
            entries_array_ty,
            tagged_funcptr_const("ll_dict_remove_deleted_items"),
            tagged_funcptr_const("_ll_dict_entries_size_too_big"),
            tagged_funcptr_const("ll_dict_entries_arraycopy"),
        )
        .expect("build_ll_dict_grow_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_grow");
        crate::flowspace::model::checkgraph(&helper.graph.borrow());
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert!(
            startblock.exitswitch.is_some(),
            "ll_dict_grow branches on deleted-entry compaction"
        );
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getfield", "getfield", "int_rshift", "int_lt"]);
        drop(startblock);
        let (_block_count, ops) = walk_blocks(&inner.startblock);
        for needed in ["malloc_varsize", "direct_call", "setfield", "int_eq"] {
            assert!(
                ops.iter().any(|o| o == needed),
                "grow CFG must emit {needed}, got {ops:?}"
            );
        }
        assert_eq!(
            ops.iter().filter(|o| *o == "direct_call").count(),
            4,
            "compact remove + size_too_big + too-big remove + entries_arraycopy"
        );
        let targets = direct_call_target_values(&inner.startblock);
        assert!(
            targets
                .iter()
                .any(|v| *v == ConstValue::byte_str("_ll_dict_entries_size_too_big")),
            "grow must direct_call _ll_dict_entries_size_too_big, got {targets:?}"
        );
        assert_eq!(
            targets
                .iter()
                .filter(|v| **v == ConstValue::byte_str("ll_dict_remove_deleted_items"))
                .count(),
            2,
            "compact branch + size-too-big arm, got {targets:?}"
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Bool));
    }

    /// `ll_dict_remove_deleted_items` either allocates a smaller entries
    /// array or compacts in place, copies valid entries forward, clears
    /// trailing GC refs when needed, and reindexes against the current index
    /// length.
    #[test]
    fn build_ll_dict_remove_deleted_items_compacts_and_reindexes() {
        let repr = sample_ordered_dict_repr();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let entries_array_ty = LowLevelType::Array(Box::new(repr.DICTENTRYARRAY.clone()));
        let helper = build_ll_dict_remove_deleted_items_helper_graph(
            "ll_dict_remove_deleted_items",
            repr.lowleveltype().clone(),
            entries_ptr,
            entries_array_ty,
            repr.DICTKEY.clone(),
            repr.DICTVALUE.clone(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_dict_remove_deleted_items_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_remove_deleted_items");
        assert_eq!(helper.func._jit_look_inside_, Some(false));
        let inner = helper.graph.borrow();
        assert_eq!(inner.startblock.borrow().inputargs.len(), 1);
        let (_block_count, ops) = walk_blocks(&inner.startblock);
        for needed in [
            "malloc_varsize",
            "getinteriorfield",
            "setinteriorfield",
            "setfield",
            "getarraysize",
            "length_of_simple_gcarray_from_opaque",
            "direct_call",
        ] {
            assert!(
                ops.iter().any(|o| o == needed),
                "remove_deleted_items CFG must emit {needed}, got {ops:?}"
            );
        }
        assert_eq!(
            ops.iter().filter(|o| *o == "direct_call").count(),
            1,
            "tail reindex call"
        );
        let start = inner.startblock.borrow();
        for link in &start.exits {
            let link = link.borrow();
            let target = link.target.as_ref().unwrap().borrow();
            let barriers: Vec<_> = target
                .operations
                .iter()
                .filter(|op| op.opname == "gc_writebarrier")
                .collect();
            if target
                .operations
                .iter()
                .any(|op| op.opname == "malloc_varsize")
            {
                assert!(
                    barriers.is_empty(),
                    "fresh entries need no pre-copy barrier"
                );
            } else {
                assert_eq!(
                    barriers.len(),
                    1,
                    "in-place compaction barriers the old array once"
                );
                assert_eq!(barriers[0].args, vec![target.inputargs[1].clone()]);
                assert_eq!(target.operations.len(), 1, "barrier precedes the copy loop");
            }
        }
    }

    /// `_ll_dict_resize_to` restores the shrink-via-compaction branch:
    /// after the sizing loop, `new_size < len(d.indexes)` calls
    /// `ll_dict_remove_deleted_items`; otherwise it reindexes to `new_size`.
    #[test]
    fn build_ll_dict_resize_to_removes_deleted_items_on_shrink() {
        let helper = build_ll_dict_resize_to_helper_graph(
            "_ll_dict_resize_to",
            sample_dict_ptr_lltype(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_dict_resize_to_helper_graph");
        assert_eq!(helper.func.name, "_ll_dict_resize_to");
        let inner = helper.graph.borrow();
        assert_eq!(inner.startblock.borrow().inputargs.len(), 2); // d, num_extra
        let (_block_count, ops) = walk_blocks(&inner.startblock);
        for needed in [
            "length_of_simple_gcarray_from_opaque",
            "int_lt",
            "direct_call",
        ] {
            assert!(
                ops.iter().any(|o| o == needed),
                "resize_to CFG must emit {needed}, got {ops:?}"
            );
        }
        assert_eq!(
            ops.iter().filter(|o| *o == "direct_call").count(),
            2,
            "remove_deleted_items + reindex"
        );
    }

    /// `ll_dict_reindex` reuses the index array via `ll_clear_indexes` when
    /// `bool(d.indexes)` and the length matches `new_size`, otherwise mallocs,
    /// then loops over `[0, num_ever_used_items)` calling `ll_dict_store_clean`
    /// for every `f_valid` entry.
    #[test]
    fn build_ll_dict_reindex_loops_and_stores_valid_entries() {
        let repr = sample_ordered_dict_repr();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let malloc_choose_fn = tagged_funcptr_const("ll_malloc_indexes_and_choose_lookup");
        let clear_indexes_fn = tagged_funcptr_const("ll_clear_indexes");
        let helper = build_ll_dict_reindex_helper_graph(
            "ll_dict_reindex",
            repr.lowleveltype().clone(),
            entries_ptr,
            malloc_choose_fn,
            clear_indexes_fn,
            dummy_funcptr_const(),
            None,
            false,
        )
        .expect("build_ll_dict_reindex_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_reindex");
        let inner = helper.graph.borrow();
        assert_eq!(inner.startblock.borrow().inputargs.len(), 2); // d, new_size

        let (block_count, ops) = walk_blocks(&inner.startblock);
        assert_eq!(
            block_count, 10,
            "start + len + clear + malloc + body + cond + loop body + store + next + returnblock"
        );
        for needed in ["getinteriorfield", "int_lt", "int_add", "direct_call"] {
            assert!(
                ops.iter().any(|o| o == needed),
                "reindex CFG must emit {needed}, got {ops:?}"
            );
        }
        assert_eq!(
            ops.iter().filter(|o| *o == "direct_call").count(),
            3,
            "clear_indexes + malloc_choose + store_clean"
        );
        let targets = direct_call_target_values(&inner.startblock);
        assert!(
            targets
                .iter()
                .any(|v| *v == ConstValue::byte_str("ll_clear_indexes")),
            "reindex must direct_call ll_clear_indexes, got {targets:?}"
        );
        assert!(
            targets
                .iter()
                .any(|v| *v == ConstValue::byte_str("ll_malloc_indexes_and_choose_lookup")),
            "reindex must direct_call ll_malloc_indexes_and_choose_lookup, got {targets:?}"
        );
        let len_block = bool_exit(&inner.startblock, true);
        assert!(
            ops.iter()
                .any(|o| *o == "length_of_simple_gcarray_from_opaque"),
            "reindex CFG must emit length_of_simple_gcarray_from_opaque, got {ops:?}"
        );
        assert!(
            !ops.iter().any(|o| *o == "cast_pointer"),
            "reindex must not cast GCREF indexes to an array pointer, got {ops:?}"
        );
        assert!(
            len_block
                .borrow()
                .operations
                .iter()
                .any(|op| op.opname == "int_eq"),
            "len_block compares _ll_len_of_d_indexes with new_size"
        );
        assert!(
            len_block.borrow().exitswitch.is_some(),
            "len_block bool-exitswitch on the length comparison"
        );
    }

    /// `ll_clear_indexes` masks `lookup_function_no` then dispatches
    /// `rgc.ll_arrayclear` on each `DICTINDEX_*` width (`rordereddict.py`
    /// `ll_clear_indexes`): four `int_eq` comparisons on 64-bit (BYTE/
    /// SHORT/INT/LONG), three on 32-bit (no INT), and `assert False` via
    /// exceptblock.
    #[test]
    fn ll_clear_indexes_graph_masks_lookup_function_and_clears() {
        let arrayclear_fn = tagged_funcptr_const("ll_arrayclear");
        let helper = build_ll_clear_indexes_helper_graph(
            "ll_clear_indexes",
            sample_dict_ptr_lltype(),
            arrayclear_fn,
        )
        .expect("build_ll_clear_indexes_helper_graph");
        assert_eq!(helper.func.name, "ll_clear_indexes");
        crate::flowspace::model::checkgraph(&helper.graph.borrow());
        let inner = helper.graph.borrow();
        let start = inner.startblock.borrow();
        assert_eq!(start.inputargs.len(), 2); // d, n
        let ops: Vec<&str> = start
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getfield", "int_and", "setfield"]);
        assert_eq!(
            getfield_field(&start.operations[0]),
            Some(ConstValue::byte_str("lookup_function_no"))
        );
        match start.operations[2].args.get(1) {
            Some(Hlvalue::Constant(c)) => {
                assert_eq!(c.value, ConstValue::byte_str("lookup_function_no"))
            }
            other => panic!("setfield lookup_function_no, got {other:?}"),
        }
        assert_eq!(start.exits.len(), 1);
        drop(start);

        let mut expected_funcs = vec![FUNC_BYTE, FUNC_SHORT];
        let mut expected_types: Vec<&LowLevelType> = vec![&*DICTINDEX_BYTE, &*DICTINDEX_SHORT];
        if IS_64BIT {
            expected_funcs.push(FUNC_INT);
            expected_types.push(&*DICTINDEX_INT);
        }
        expected_funcs.push(FUNC_LONG);
        expected_types.push(&*DICTINDEX_LONG);

        let mut check = {
            let start = inner.startblock.borrow();
            start.exits[0]
                .borrow()
                .target
                .clone()
                .expect("first FUNC_* check")
        };
        for (i, (func, index_ty)) in expected_funcs.iter().zip(expected_types.iter()).enumerate() {
            {
                let cb = check.borrow();
                assert_eq!(cb.operations.len(), 1, "check block is a single int_eq");
                assert_eq!(cb.operations[0].opname, "int_eq");
                match cb.operations[0].args.get(1) {
                    Some(Hlvalue::Constant(c)) => {
                        assert_eq!(c.value, ConstValue::Int(*func), "int_eq against FUNC_*")
                    }
                    other => panic!("int_eq FUNC const, got {other:?}"),
                }
            }

            let leaf = bool_exit(&check, true);
            {
                let leafb = leaf.borrow();
                let leaf_ops: Vec<&str> = leafb
                    .operations
                    .iter()
                    .map(|op| op.opname.as_str())
                    .collect();
                assert_eq!(leaf_ops, vec!["getfield", "cast_opaque_ptr", "direct_call"]);
                assert_eq!(
                    getfield_field(&leafb.operations[0]),
                    Some(ConstValue::byte_str("indexes"))
                );
                match &leafb.operations[1].result {
                    Hlvalue::Variable(v) => {
                        assert_eq!(
                            v.concretetype.borrow().as_ref(),
                            Some(*index_ty),
                            "cast_opaque_ptr target is DICTINDEX_*"
                        );
                    }
                    other => panic!("cast_opaque_ptr result, got {other:?}"),
                }
                match leafb.operations[2].args.first() {
                    Some(Hlvalue::Constant(c)) => {
                        assert_eq!(c.value, ConstValue::byte_str("ll_arrayclear"))
                    }
                    other => panic!("arrayclear callee, got {other:?}"),
                }
            }

            let next = bool_exit(&check, false);
            if i + 1 == expected_funcs.len() {
                assert!(
                    Rc::ptr_eq(&next, &inner.exceptblock),
                    "final false link raises AssertionError via exceptblock"
                );
            } else {
                check = next;
            }
        }

        let (_block_count, all_ops) = walk_blocks(&inner.startblock);
        assert_eq!(
            all_ops.iter().filter(|o| *o == "int_eq").count(),
            expected_funcs.len(),
            "four (three on 32-bit) int_eq FUNC_* comparisons, got {all_ops:?}"
        );
    }

    /// `_ll_dict_setitem_lookup_done` branches on `i >= 0`: the overwrite
    /// path writes only `value`; the insert path threads grow/resize/
    /// insert-clean before writing all four entry fields and bumping both
    /// counters.
    #[test]
    fn build_ll_dict_setitem_lookup_done_branches_overwrite_vs_insert() {
        let repr = sample_ordered_dict_repr();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let helper = build_ll_dict_setitem_lookup_done_helper_graph(
            "_ll_dict_setitem_lookup_done",
            repr.lowleveltype().clone(),
            entries_ptr,
            repr.DICTKEY.clone(),
            repr.DICTVALUE.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_dict_setitem_lookup_done_helper_graph");
        assert_eq!(helper.func.name, "_ll_dict_setitem_lookup_done");
        let inner = helper.graph.borrow();
        assert_eq!(inner.startblock.borrow().inputargs.len(), 5); // d, key, value, hash, i
        assert!(inner.startblock.borrow().exitswitch.is_some());

        let (block_count, ops) = walk_blocks(&inner.startblock);
        assert_eq!(
            block_count, 14,
            "12 work blocks + returnblock + exceptblock"
        );
        // Overwrite path writes a single `value` field; the insert path
        // writes key/value/f_hash/f_valid (4) — 5 setinteriorfields total.
        assert_eq!(ops.iter().filter(|o| *o == "setinteriorfield").count(), 5);
        assert_eq!(
            ops.iter().filter(|o| *o == "direct_call").count(),
            5,
            "grow + resize + insert_clean + 2 rescue"
        );
    }

    /// `ll_dict_create_initial_index` / `ll_dict_rehash_after_translation` /
    /// `_ll_dict_rescue` / `_ll_dict_setitem_lookup_done` graphs pass
    /// `checkgraph` and `direct_call` the rehash / resize / rescue helpers
    /// (`rordereddict.py`).
    #[test]
    fn create_initial_index_and_lookup_done_graphs_check_and_call_helpers() {
        let repr = sample_ordered_dict_repr();
        let dict_ptr = repr.lowleveltype().clone();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let rehash_fn = tagged_funcptr_const("ll_dict_rehash_after_translation");
        let reindex_fn = tagged_funcptr_const("ll_dict_reindex");
        let resize_fn = tagged_funcptr_const("ll_dict_resize");
        let rescue_fn = tagged_funcptr_const("_ll_dict_rescue");

        let create = build_ll_dict_create_initial_index_helper_graph(
            "ll_dict_create_initial_index",
            dict_ptr.clone(),
            dummy_funcptr_const(),
            rehash_fn.clone(),
        )
        .expect("create_initial_index");
        crate::flowspace::model::checkgraph(&create.graph.borrow());
        let create_targets = direct_call_target_values(&create.graph.borrow().startblock);
        assert!(
            create_targets
                .iter()
                .any(|v| *v == ConstValue::byte_str("ll_dict_rehash_after_translation")),
            "create_initial_index must direct_call ll_dict_rehash_after_translation, got {create_targets:?}"
        );

        let rehash = build_ll_dict_rehash_after_translation_helper_graph(
            "ll_dict_rehash_after_translation",
            dict_ptr.clone(),
            entries_ptr.clone(),
            reindex_fn.clone(),
            dummy_funcptr_const(),
            false,
        )
        .expect("rehash_after_translation");
        crate::flowspace::model::checkgraph(&rehash.graph.borrow());
        let rehash_targets = direct_call_target_values(&rehash.graph.borrow().startblock);
        assert!(
            rehash_targets
                .iter()
                .any(|v| *v == ConstValue::byte_str("ll_dict_reindex")),
            "rehash must direct_call ll_dict_reindex, got {rehash_targets:?}"
        );

        let rescue =
            build_ll_dict_rescue_helper_graph("_ll_dict_rescue", dict_ptr.clone(), reindex_fn)
                .expect("rescue");
        crate::flowspace::model::checkgraph(&rescue.graph.borrow());
        let rescue_targets = direct_call_target_values(&rescue.graph.borrow().startblock);
        assert!(
            rescue_targets
                .iter()
                .any(|v| *v == ConstValue::byte_str("ll_dict_reindex")),
            "rescue must direct_call ll_dict_reindex, got {rescue_targets:?}"
        );

        let lookup_done = build_ll_dict_setitem_lookup_done_helper_graph(
            "_ll_dict_setitem_lookup_done",
            dict_ptr,
            entries_ptr,
            repr.DICTKEY.clone(),
            repr.DICTVALUE.clone(),
            dummy_funcptr_const(),
            resize_fn,
            dummy_funcptr_const(),
            rescue_fn,
        )
        .expect("setitem_lookup_done");
        crate::flowspace::model::checkgraph(&lookup_done.graph.borrow());
        let lookup_targets = direct_call_target_values(&lookup_done.graph.borrow().startblock);
        assert!(
            lookup_targets
                .iter()
                .any(|v| *v == ConstValue::byte_str("ll_dict_resize")),
            "lookup_done must direct_call ll_dict_resize, got {lookup_targets:?}"
        );
        assert!(
            lookup_targets
                .iter()
                .any(|v| *v == ConstValue::byte_str("_ll_dict_rescue")),
            "lookup_done must direct_call _ll_dict_rescue, got {lookup_targets:?}"
        );
    }

    /// `_ll_dict_del_entry` marks the entry invalid, decrements
    /// `num_live_items`, and conditionally clears GC key/value fields.
    #[test]
    fn build_ll_dict_del_entry_marks_invalid_and_decrements_live_count() {
        let repr = sample_ordered_dict_repr();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let helper = build_ll_dict_del_entry_helper_graph(
            "_ll_dict_del_entry",
            repr.lowleveltype().clone(),
            entries_ptr,
            repr.DICTKEY.clone(),
            repr.DICTVALUE.clone(),
        )
        .expect("build_ll_dict_del_entry_helper_graph");
        assert_eq!(helper.func.name, "_ll_dict_del_entry");
        let inner = helper.graph.borrow();
        let ops: Vec<String> = inner
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(
            ops,
            vec![
                "getfield",
                "setinteriorfield",
                "getfield",
                "int_sub",
                "setfield",
                "setinteriorfield"
            ],
            "int key needs no clear; string value is a GC pointer and is cleared"
        );
    }

    /// `_ll_dict_del` calls the index deletion helper, marks the entry
    /// deleted, reclaims trailing invalid entries, and applies the 87.5%
    /// dead-items shrink rule.
    #[test]
    fn build_ll_dict_del_restores_trailing_reclaim_and_dead_item_shrink() {
        let repr = sample_ordered_dict_repr();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let helper = build_ll_dict_del_helper_graph(
            "_ll_dict_del",
            repr.lowleveltype().clone(),
            entries_ptr,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_dict_del_helper_graph");
        assert_eq!(helper.func.name, "_ll_dict_del");
        let inner = helper.graph.borrow();
        assert_eq!(inner.startblock.borrow().inputargs.len(), 3);
        let (_block_count, ops) = walk_blocks(&inner.startblock);
        for needed in [
            "direct_call",
            "getinteriorfield",
            "setfield",
            "int_le",
            "int_rshift",
        ] {
            assert!(
                ops.iter().any(|o| o == needed),
                "_ll_dict_del CFG must emit {needed}, got {ops:?}"
            );
        }
        assert_eq!(
            ops.iter().filter(|o| *o == "direct_call").count(),
            3,
            "delete_by_entry_index + del_entry + resize"
        );
    }

    /// `ll_dict_delitem` hashes, looks up with `FLAG_LOOKUP`, raises
    /// KeyError on a miss, and calls `_ll_dict_del` on a hit.
    #[test]
    fn build_ll_dict_delitem_deletes_or_raises_keyerror() {
        let (dict_ptr, _entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_delitem_helper_graph(
            "ll_dict_delitem",
            dict_ptr,
            key_lltype,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            false,
        )
        .expect("build_ll_dict_delitem_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_delitem");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["direct_call", "direct_call", "int_lt"]);
        assert_eq!(startblock.exits.len(), 2);
        let raise_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        assert!(Rc::ptr_eq(
            &raise_link
                .borrow()
                .target
                .clone()
                .expect("exceptblock target"),
            &inner.exceptblock,
        ));
    }

    /// `ll_dict_get` returns the supplied default on a missing key, otherwise
    /// reads `entries[index].value`.
    #[test]
    fn build_ll_dict_get_returns_default_or_value() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_get_helper_graph(
            "ll_dict_get",
            dict_ptr,
            entries_ptr,
            key_lltype,
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            false,
        )
        .expect("build_ll_dict_get_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_get");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["direct_call", "direct_call", "int_lt"]);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);
        let (_block_count, ops) = walk_blocks(&inner.startblock);
        assert!(
            ops.iter().any(|o| o == "getinteriorfield"),
            "found arm reads entries[index].value"
        );
    }

    /// `ll_dict_setdefault` inserts the default through
    /// `_ll_dict_setitem_lookup_done(..., -1)` on a missing key, otherwise
    /// returns the existing value.
    #[test]
    fn build_ll_dict_setdefault_inserts_default_or_returns_existing_value() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_setdefault_helper_graph(
            "ll_dict_setdefault",
            dict_ptr,
            entries_ptr,
            key_lltype,
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            false,
        )
        .expect("build_ll_dict_setdefault_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_setdefault");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["direct_call", "direct_call", "int_lt"]);
        drop(startblock);
        let (_block_count, ops) = walk_blocks(&inner.startblock);
        assert!(
            ops.iter().any(|o| o == "direct_call"),
            "missing arm calls _ll_dict_setitem_lookup_done"
        );
        assert!(
            ops.iter().any(|o| o == "getinteriorfield"),
            "found arm reads entries[index].value"
        );
    }

    /// `ll_dict_insert` is one `FLAG_STORE` lookup plus one
    /// `_ll_dict_setitem_lookup_done` (`ll_dict_setitem_with_hash`),
    /// reading the displaced value the way `ll_dict_get` does.
    #[test]
    fn build_ll_dict_insert_has_one_lookup_and_one_lookup_done() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_insert_helper_graph(
            "ll_dict_insert",
            dict_ptr,
            entries_ptr,
            key_lltype,
            string_repr().lowleveltype().clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            false,
        )
        .expect("build_ll_dict_insert_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_insert");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["direct_call", "direct_call", "int_lt"]);
        let lookup = startblock
            .operations
            .iter()
            .find(|op| {
                op.opname == "direct_call"
                    && op.args.len() == 5
                    && matches!(
                        &op.args[4],
                        Hlvalue::Constant(c) if c.value == ConstValue::Int(FLAG_STORE)
                    )
            })
            .expect("startblock FLAG_STORE lookup");
        assert_eq!(lookup.args.len(), 5);
        drop(startblock);

        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![inner.startblock.clone()];
        let mut lookup_calls = 0usize;
        let mut lookup_done_calls = 0usize;
        while let Some(b) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&b) as usize) {
                continue;
            }
            let bb = b.borrow();
            for op in &bb.operations {
                if op.opname != "direct_call" {
                    continue;
                }
                if op.args.len() == 5
                    && matches!(
                        &op.args[4],
                        Hlvalue::Constant(c) if c.value == ConstValue::Int(FLAG_STORE)
                    )
                {
                    lookup_calls += 1;
                }
                if op.args.len() == 6 {
                    lookup_done_calls += 1;
                }
            }
            for link in &bb.exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        assert_eq!(lookup_calls, 1, "exactly one FLAG_STORE lookup");
        assert_eq!(
            lookup_done_calls, 1,
            "exactly one _ll_dict_setitem_lookup_done"
        );
        let (_block_count, ops) = walk_blocks(&inner.startblock);
        assert!(
            ops.iter().any(|o| o == "getinteriorfield"),
            "found arm reads entries[index].value"
        );
    }

    /// Custom r_dict hash is `ll_keyhash_custom(d, key)`, so the hash
    /// `direct_call` receives the dict (`rordereddict.py`).
    #[test]
    fn build_ll_dict_insert_custom_keyhash_takes_dict() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_insert_helper_graph(
            "ll_dict_insert",
            dict_ptr,
            entries_ptr,
            key_lltype,
            string_repr().lowleveltype().clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            true,
        )
        .expect("build_ll_dict_insert_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let hash_call = &startblock.operations[0];
        assert_eq!(hash_call.opname, "direct_call");
        assert_eq!(
            hash_call.args.len(),
            3,
            "custom keyhash is direct_call(hash_fn, d, key)"
        );
    }

    /// `ll_dict_insert` miss path is `None` as `value_lltype`. Signed
    /// cannot hold that constant (`rordereddict.py` `ll_dict_get`).
    #[test]
    fn build_ll_dict_insert_rejects_non_pointer_value_type() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let err = build_ll_dict_insert_helper_graph(
            "ll_dict_insert",
            dict_ptr,
            entries_ptr,
            key_lltype,
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            false,
        )
        .expect_err("Signed value type cannot represent miss None");
        let text = format!("{err:?}");
        assert!(
            text.contains("cannot represent None"),
            "expected miss-None diagnostic, got {text}"
        );
    }

    /// End-to-end wiring: `OrderedDictRepr::rtype_setitem` on an int-keyed
    /// dict (direct-compare, no eq-gate) emits the hash + lookup +
    /// `_ll_dict_setitem_lookup_done` chain fused into a single
    /// `ll_dict_setitem` `direct_call`, and threads
    /// `hop.exception_cannot_occur()` (the non-`custom_eq_hash` branch,
    /// `rordereddict.py`).
    #[test]
    fn ordereddictrepr_rtype_setitem_int_key_emits_direct_call_chain() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_int_int_dict_repr(rtyper);
        let dict_lltype = repr.lowleveltype().clone();
        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            repr.base.rtyper.clone(),
            None,
        )));

        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_value = Variable::new();
        v_value.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));

        let hop = HighLevelOp::new(
            repr.base.rtyper.clone(),
            SpaceOperation::new(
                "setitem".to_string(),
                vec![
                    Hlvalue::Variable(v_dict),
                    Hlvalue::Variable(v_key),
                    Hlvalue::Variable(v_value),
                ],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Impossible,
            SomeValue::Impossible,
            SomeValue::Impossible,
        ]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = repr
            .rtype_setitem(&hop)
            .unwrap_or_else(|err| panic!("OrderedDictRepr::rtype_setitem: {err:?}"));
        assert!(result.is_none(), "rtype_setitem returns None (Void op)");

        let ops = llops.borrow();
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_setitem must call hop.exception_cannot_occur() (non-custom_eq_hash)"
        );
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_setitem"));
    }

    /// End-to-end pairtype wiring: `(OrderedDictRepr, _).rtype_delitem`
    /// dispatches to `OrderedDictRepr::rtype_delitem`, emits
    /// `ll_dict_delitem`, and threads `hop.exception_is_here()`.
    #[test]
    fn pairtype_ordereddict_rtype_delitem_int_key_emits_direct_call_chain() {
        use crate::translator::rtyper::pairtype::pair_rtype_delitem;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_int_int_dict_repr(rtyper);
        let dict_lltype = repr.lowleveltype().clone();
        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            repr.base.rtyper.clone(),
            None,
        )));

        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));

        let exitblock = Block::shared(vec![]);
        let cls_keyerror = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("KeyError")
            .expect("KeyError class");
        let link_keyerror = Link::new(
            vec![],
            Some(exitblock),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                cls_keyerror,
            )))),
        )
        .into_ref();

        let hop = HighLevelOp::new(
            repr.base.rtyper.clone(),
            SpaceOperation::new(
                "delitem".to_string(),
                vec![Hlvalue::Variable(v_dict), Hlvalue::Variable(v_key)],
                Hlvalue::Variable(v_result),
            ),
            vec![link_keyerror],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Impossible, SomeValue::Impossible]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = pair_rtype_delitem(repr.as_ref(), signed_repr().as_ref(), &hop)
            .unwrap_or_else(|err| panic!("pair_rtype_delitem: {err:?}"));
        assert!(result.is_none(), "rtype_delitem returns None (Void op)");

        let ops = llops.borrow();
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_delitem must call hop.exception_is_here()"
        );
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_delitem"));
    }

    /// Method routing: `d.get(k, default)` goes through
    /// `OrderedDictRepr::rtype_method("get")` and emits `ll_dict_get`.
    #[test]
    fn ordereddictrepr_rtype_method_get_int_key_emits_ll_dict_get() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_int_int_dict_repr(rtyper);
        let dict_lltype = repr.lowleveltype().clone();
        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            repr.base.rtyper.clone(),
            None,
        )));

        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_default = Variable::new();
        v_default.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));

        let hop = HighLevelOp::new(
            repr.base.rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![
                    Hlvalue::Variable(v_dict),
                    Hlvalue::Variable(v_key),
                    Hlvalue::Variable(v_default),
                ],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Impossible,
            SomeValue::Impossible,
            SomeValue::Impossible,
        ]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = repr
            .rtype_method("get", &hop)
            .unwrap_or_else(|err| panic!("OrderedDictRepr::rtype_method_get: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_method_get must call hop.exception_cannot_occur()"
        );
        assert_eq!(ops.ops.len(), 1);
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_get"));
    }

    /// Method routing: `d.setdefault(k, default)` emits
    /// `ll_dict_setdefault` and returns the value repr.
    #[test]
    fn ordereddictrepr_rtype_method_setdefault_int_key_emits_ll_dict_setdefault() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_int_int_dict_repr(rtyper);
        let dict_lltype = repr.lowleveltype().clone();
        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            repr.base.rtyper.clone(),
            None,
        )));

        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_default = Variable::new();
        v_default.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));

        let hop = HighLevelOp::new(
            repr.base.rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![
                    Hlvalue::Variable(v_dict),
                    Hlvalue::Variable(v_key),
                    Hlvalue::Variable(v_default),
                ],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Impossible,
            SomeValue::Impossible,
            SomeValue::Impossible,
        ]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = repr
            .rtype_method("setdefault", &hop)
            .unwrap_or_else(|err| panic!("OrderedDictRepr::rtype_method_setdefault: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_method_setdefault must call hop.exception_cannot_occur()"
        );
        assert_eq!(ops.ops.len(), 1);
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_setdefault"));
    }

    /// The eq-gate lets plain str keys through on delitem too, same
    /// `d.keyeq`-fallback wiring as getitem/contains/setitem.
    #[test]
    fn ordereddictrepr_rtype_delitem_str_key_emits_direct_call_chain() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::String(SomeString::new(false, false)),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let repr = Arc::new(
            OrderedDictRepr::new(
                rtyper.clone(),
                string_repr() as Arc<dyn Repr>,
                signed_repr() as Arc<dyn Repr>,
                dictdef,
                None,
                false,
                false,
            )
            .expect("ordered dict repr"),
        );
        let dict_lltype = repr.lowleveltype().clone();
        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));

        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(string_repr().lowleveltype().clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));

        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                "delitem".to_string(),
                vec![Hlvalue::Variable(v_dict), Hlvalue::Variable(v_key)],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Impossible, SomeValue::Impossible]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(string_repr() as Arc<dyn Repr>),
        ]);

        let result = repr
            .rtype_delitem(&hop)
            .unwrap_or_else(|err| panic!("OrderedDictRepr::rtype_delitem: {err:?}"));
        assert!(result.is_none(), "rtype_delitem returns None (Void op)");

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_delitem"));
    }

    /// The eq-gate lets plain str keys through on setitem too — same
    /// `lookup_chain_helpers` wiring as getitem/contains/delitem.
    #[test]
    fn ordereddictrepr_rtype_setitem_str_key_emits_direct_call_chain() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::String(SomeString::new(false, false)),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let repr = Arc::new(
            OrderedDictRepr::new(
                rtyper.clone(),
                string_repr() as Arc<dyn Repr>,
                signed_repr() as Arc<dyn Repr>,
                dictdef,
                None,
                false,
                false,
            )
            .expect("ordered dict repr"),
        );
        let dict_lltype = repr.lowleveltype().clone();
        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));

        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(string_repr().lowleveltype().clone()));
        let v_value = Variable::new();
        v_value.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));

        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                "setitem".to_string(),
                vec![
                    Hlvalue::Variable(v_dict),
                    Hlvalue::Variable(v_key),
                    Hlvalue::Variable(v_value),
                ],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Impossible,
            SomeValue::Impossible,
            SomeValue::Impossible,
        ]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(string_repr() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = repr
            .rtype_setitem(&hop)
            .unwrap_or_else(|err| panic!("OrderedDictRepr::rtype_setitem: {err:?}"));
        assert!(result.is_none(), "rtype_setitem returns None (Void op)");

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_setitem"));
    }

    fn funcptr_repr(args: Vec<LowLevelType>, result: LowLevelType) -> Arc<dyn Repr> {
        use crate::translator::rtyper::rmodel::PtrRepr;
        Arc::new(PtrRepr::new(Ptr {
            TO: PtrTarget::Func(FuncType { args, result }),
        })) as Arc<dyn Repr>
    }

    fn custom_eq_hash_reprs() -> (Arc<dyn Repr>, Arc<dyn Repr>) {
        (
            funcptr_repr(
                vec![LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Bool,
            ),
            funcptr_repr(vec![LowLevelType::Signed], LowLevelType::Signed),
        )
    }

    fn funcptr_const(name: &str, args: Vec<LowLevelType>, result: LowLevelType) -> ConstValue {
        ConstValue::LLPtr(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::functionptr(
                FuncType { args, result },
                name,
                None,
                None::<String>,
            ),
        ))
    }

    fn signed_callback_funcptr(
        name: &str,
        n_args: usize,
        result: LowLevelType,
        opname: Option<&str>,
    ) -> ConstValue {
        let vars: Vec<Variable> = (0..n_args)
            .map(|i| {
                let v = Variable::named(format!("a{i}"));
                v.set_concretetype(Some(LowLevelType::Signed));
                v
            })
            .collect();
        let start = Block::shared(vars.iter().cloned().map(Hlvalue::Variable).collect());
        let ret_hl = if let Some(opname) = opname {
            let z = Variable::named("z");
            z.set_concretetype(Some(result.clone()));
            start.borrow_mut().operations.push(SpaceOperation::new(
                opname,
                vars.iter().cloned().map(Hlvalue::Variable).collect(),
                Hlvalue::Variable(z.clone()),
            ));
            Hlvalue::Variable(z)
        } else {
            Hlvalue::Variable(vars[0].clone())
        };
        let retvar = Variable::named("ret");
        retvar.set_concretetype(Some(result.clone()));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            name.to_string(),
            start.clone(),
            Hlvalue::Variable(retvar),
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![ret_hl], Some(returnblock), None).into_ref(),
        ]);
        ConstValue::LLPtr(Box::new(functionptr_for_graph_with_type(
            &graph,
            FuncType {
                args: vec![LowLevelType::Signed; n_args],
                result,
            },
        )))
    }

    fn make_prebuilt_r_dict(
        key_eq: ConstValue,
        key_hash: ConstValue,
        pairs: &[(i64, i64)],
    ) -> HostObject {
        let cls = crate::flowspace::model::HOST_ENV
            .lookup_builtin("rpython.rlib.objectmodel.r_dict")
            .expect("canonical r_dict class");
        let key_cls = HostObject::new_class("rpython.rlib.objectmodel._r_dictkey", vec![]);
        let r_dict = HostObject::new_instance(cls, vec![]);
        r_dict.instance_set("key_eq", key_eq).expect("key_eq");
        r_dict.instance_set("key_hash", key_hash).expect("key_hash");
        let mut inner = indexmap::IndexMap::new();
        for &(k, v) in pairs {
            let dk = HostObject::new_instance(key_cls.clone(), vec![]);
            dk.instance_set("dic", ConstValue::HostObject(r_dict.clone()))
                .expect("dic");
            dk.instance_set("key", ConstValue::Int(k)).expect("key");
            dk.instance_set("hash", ConstValue::Int(k)).expect("hash");
            inner.insert(ConstValue::HostObject(dk), ConstValue::Int(v));
        }
        r_dict
            .instance_set("_dict", ConstValue::Dict(Arc::new(inner)))
            .expect("_dict");
        r_dict
    }

    fn dict_cache_key_dict_items(
        key: &DictCacheKey,
    ) -> &Arc<indexmap::IndexMap<ConstValue, ConstValue>> {
        match key {
            DictCacheKey::Dict { items } => items,
            DictCacheKey::RDict { .. } => panic!("expected Dict cache key"),
        }
    }

    fn make_custom_eq_hash_dict_repr(rtyper: Rc<RPythonTyper>) -> Arc<OrderedDictRepr> {
        let (eq, hash) = custom_eq_hash_reprs();
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Integer(SomeInteger::default()),
            true,
            false,
            false,
        );
        Arc::new(
            OrderedDictRepr::new(
                rtyper,
                signed_repr() as Arc<dyn Repr>,
                signed_repr() as Arc<dyn Repr>,
                dictdef,
                Some((eq, hash)),
                false,
                false,
            )
            .expect("custom eq/hash ordered dict repr"),
        )
    }

    /// `get_ll_dict` (`rordereddict.py`) stores `fnkeyeq` / `fnkeyhash`
    /// and the `keyhash`/`keyeq`/`paranoia` adtmeths on a custom r_dict.
    #[test]
    fn ordereddictrepr_custom_eq_hash_installs_fnkey_fields_and_adtmeths() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_custom_eq_hash_dict_repr(rtyper);
        assert!(repr.base.custom_eq_hash);
        assert!(
            repr.DICT._names.iter().any(|n| n == "fnkeyeq"),
            "DICT must hold fnkeyeq, got {:?}",
            repr.DICT._names
        );
        assert!(
            repr.DICT._names.iter().any(|n| n == "fnkeyhash"),
            "DICT must hold fnkeyhash, got {:?}",
            repr.DICT._names
        );
        assert!(
            repr.DICT._adtmeths.get("keyhash").is_some(),
            "DICT adtmeths must name ll_keyhash_custom"
        );
        assert!(
            repr.DICT._adtmeths.get("keyeq").is_some(),
            "DICT adtmeths must name ll_keyeq_custom"
        );
        assert_eq!(
            repr.DICT._adtmeths.get("paranoia"),
            Some(&ConstValue::Bool(true)),
            "paranoia is not simple_hash_eq"
        );
        assert!(
            matches!(
                repr.DICT._adtmeths.get("r_rdict_eqfn"),
                Some(ConstValue::Repr(_))
            ),
            "r_rdict_eqfn adtmeth is the live Repr, got {:?}",
            repr.DICT._adtmeths.get("r_rdict_eqfn")
        );
        assert!(
            matches!(
                repr.DICT._adtmeths.get("r_rdict_hashfn"),
                Some(ConstValue::Repr(_))
            ),
            "r_rdict_hashfn adtmeth is the live Repr, got {:?}",
            repr.DICT._adtmeths.get("r_rdict_hashfn")
        );
    }

    /// `ll_keyhash_custom` (`rordereddict.py`): FunctionsPBCRepr path
    /// `getfield(d, fnkeyhash)` then `indirect_call`.
    #[test]
    fn build_ll_keyhash_custom_getfields_then_indirect_calls() {
        let (dict_ptr, _entries, key_lltype) = sample_dict_lookup_lltypes();
        let fn_lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![key_lltype.clone()],
                result: LowLevelType::Signed,
            }),
        }));
        let helper = build_ll_keyhash_custom_helper_graph(
            "ll_keyhash_custom",
            dict_ptr,
            key_lltype,
            fn_lltype,
            None,
            None,
        )
        .expect("build_ll_keyhash_custom_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getfield", "indirect_call"]);
        let field = &startblock.operations[0].args[1];
        assert!(
            matches!(field, Hlvalue::Constant(c) if c.value == ConstValue::byte_str("fnkeyhash")),
            "must read d.fnkeyhash, got {field:?}"
        );
    }

    /// `ll_keyeq_custom` (`rordereddict.py`): `getfield(d, fnkeyeq)` then
    /// `indirect_call(fn, key1, key2, c_graphs)`.
    #[test]
    fn build_ll_keyeq_custom_getfields_then_indirect_calls() {
        let (dict_ptr, _entries, key_lltype) = sample_dict_lookup_lltypes();
        let fn_lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![key_lltype.clone(), key_lltype.clone()],
                result: LowLevelType::Bool,
            }),
        }));
        let helper = build_ll_keyeq_custom_helper_graph(
            "ll_keyeq_custom",
            dict_ptr,
            key_lltype,
            fn_lltype,
            None,
            None,
        )
        .expect("build_ll_keyeq_custom_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getfield", "indirect_call"]);
        let field = &startblock.operations[0].args[1];
        assert!(
            matches!(field, Hlvalue::Constant(c) if c.value == ConstValue::byte_str("fnkeyeq")),
            "must read d.fnkeyeq, got {field:?}"
        );
        let call_args = &startblock.operations[1].args;
        assert_eq!(
            call_args.len(),
            4,
            "indirect_call(fn, key1, key2, c_graphs), got {call_args:?}"
        );
    }

    /// Custom `d.keyeq` is `ll_keyeq_custom(d, checkingkey, key)` so the
    /// lookup `direct_call` passes `d`. `d.paranoia` restarts after a
    /// user eq that mutated the table (`ll_dict_lookup`).
    #[test]
    fn build_ll_dict_lookup_custom_eq_takes_dict_and_paranoia_restarts() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_lookup_helper_graph(
            "ll_dict_lookup",
            dict_ptr,
            entries_ptr,
            LowLevelType::Unsigned,
            key_lltype,
            Some(dummy_funcptr_const()),
            true,
            true,
            None,
            false,
        )
        .expect("build_ll_dict_lookup_helper_graph");
        let inner = helper.graph.borrow();
        assert_link_variables_defined(&inner.startblock);
        let start_ptr = Rc::as_ptr(&inner.startblock) as usize;
        let (block_count, ops) = walk_blocks(&inner.startblock);
        assert!(
            block_count > 16,
            "paranoia restart chain adds check blocks, got {block_count}"
        );
        assert!(
            ops.iter().any(|o| o == "ptr_ne"),
            "paranoia checks entries/indexes pointer identity, got {ops:?}"
        );
        assert!(
            ops.iter().any(|o| o == "bool_not"),
            "paranoia checks not valid / key changed, got {ops:?}"
        );

        let mut restart_links = 0usize;
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![inner.startblock.clone()];
        let mut keyeq_call_arg_lens = Vec::new();
        while let Some(b) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&b) as usize) {
                continue;
            }
            let bb = b.borrow();
            for op in &bb.operations {
                if op.opname == "direct_call" {
                    keyeq_call_arg_lens.push(op.args.len());
                }
            }
            for link in &bb.exits {
                let l = link.borrow();
                if let Some(t) = l.target.clone() {
                    if Rc::as_ptr(&t) as usize == start_ptr && Rc::as_ptr(&b) as usize != start_ptr
                    {
                        restart_links += 1;
                        assert_eq!(
                            l.args.len(),
                            5,
                            "restart is ll_dict_lookup(d, key, hash, store_flag, T)"
                        );
                    }
                    stack.push(t);
                }
            }
        }
        assert!(
            restart_links >= 8,
            "four mutation checks at each of two keyeq sites, got {restart_links}"
        );
        assert_eq!(
            keyeq_call_arg_lens,
            vec![4, 4],
            "custom keyeq is direct_call(eq_fn, d, ckey, key) at both sites"
        );
    }

    /// Custom r_dict lookup helpers rtype setitem / getitem / contains
    /// through the normal OrderedDictRepr hop pipeline. Annotating a
    /// user graph of `r_dict(eq, hash)` still needs FunctionRepr
    /// `get_unique_llfn` (hlinvoke of the unique eq/hash); this test
    /// uses Ptr-typed callables so the helpers `getfield` + `indirect_call`
    /// the stored `fnkeyeq` / `fnkeyhash`.
    #[test]
    fn ordereddictrepr_custom_eq_hash_rtypes_setitem_getitem_contains() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_custom_eq_hash_dict_repr(rtyper.clone());
        let dict_lltype = repr.lowleveltype().clone();

        let llops_set = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype.clone()));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_value = Variable::new();
        v_value.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "setitem".to_string(),
                vec![
                    Hlvalue::Variable(v_dict),
                    Hlvalue::Variable(v_key),
                    Hlvalue::Variable(v_value),
                ],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops_set.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Impossible,
            SomeValue::Impossible,
            SomeValue::Impossible,
        ]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);
        let result = repr
            .rtype_setitem(&hop)
            .unwrap_or_else(|err| panic!("custom rtype_setitem: {err:?}"));
        assert!(result.is_none());
        assert!(
            llops_set.borrow()._called_exception_is_here_or_cannot_occur,
            "custom_eq_hash setitem uses hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &llops_set.borrow().ops[0].args[0] else {
            panic!("setitem funcptr");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_setitem"));

        let llops_get = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype.clone()));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_dict), Hlvalue::Variable(v_key)],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops_get.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Impossible, SomeValue::Impossible]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);
        let result = repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("custom rtype_getitem: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let Hlvalue::Constant(c) = &llops_get.borrow().ops[0].args[0] else {
            panic!("getitem funcptr");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_getitem"));

        let llops_in = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(rtyper, None)));
        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Bool));
        let hop = HighLevelOp::new(
            repr.base.rtyper.clone(),
            SpaceOperation::new(
                "contains".to_string(),
                vec![Hlvalue::Variable(v_dict), Hlvalue::Variable(v_key)],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops_in.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Impossible, SomeValue::Impossible]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);
        let result =
            pair_ordereddict_repr_rtype_contains(repr.as_ref(), signed_repr().as_ref(), &hop)
                .unwrap_or_else(|err| panic!("custom rtype_contains: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let Hlvalue::Constant(c) = &llops_in.borrow().ops[0].args[0] else {
            panic!("contains funcptr");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_contains"));
    }

    /// Custom r_dict `insert` uses the same `lookup_chain_helpers` hash
    /// wiring as `setitem` (`ll_keyhash_custom(d, key)`) and
    /// `hop.exception_is_here()` because the key comparison can raise.
    #[test]
    fn ordereddictrepr_custom_eq_hash_rtypes_insert() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = {
            let (eq, hash) = custom_eq_hash_reprs();
            let dictdef = DictDef::new(
                None,
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::String(SomeString::new(false, false)),
                true,
                false,
                false,
            );
            Arc::new(
                OrderedDictRepr::new(
                    rtyper.clone(),
                    signed_repr() as Arc<dyn Repr>,
                    string_repr() as Arc<dyn Repr>,
                    dictdef,
                    Some((eq, hash)),
                    false,
                    false,
                )
                .expect("custom eq/hash ordered dict repr with pointer values"),
            )
        };
        let dict_lltype = repr.lowleveltype().clone();
        let value_lltype = string_repr().lowleveltype().clone();

        let llops = Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_dict = Variable::new();
        v_dict.set_concretetype(Some(dict_lltype));
        let v_key = Variable::new();
        v_key.set_concretetype(Some(LowLevelType::Signed));
        let v_value = Variable::new();
        v_value.set_concretetype(Some(value_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(string_repr().lowleveltype().clone()));
        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![
                    Hlvalue::Variable(v_dict),
                    Hlvalue::Variable(v_key),
                    Hlvalue::Variable(v_value),
                ],
                Hlvalue::Variable(v_result),
            ),
            vec![],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Impossible,
            SomeValue::Impossible,
            SomeValue::Impossible,
        ]);
        hop.args_r.borrow_mut().extend([
            Some(repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(string_repr() as Arc<dyn Repr>),
        ]);
        let result = repr
            .rtype_method("insert", &hop)
            .unwrap_or_else(|err| panic!("custom rtype_method insert: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert!(
            llops.borrow()._called_exception_is_here_or_cannot_occur,
            "custom_eq_hash insert uses hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &llops.borrow().ops[0].args[0] else {
            panic!("insert funcptr");
        };
        assert!(format!("{:?}", c.value).contains("ll_dict_insert"));
    }

    fn function_repr_callback(
        ann: &RPythonAnnotator,
        rtyper: &Rc<RPythonTyper>,
        name: &str,
    ) -> Arc<dyn Repr> {
        use crate::annotator::description::{DescEntry, FunctionDesc};
        use crate::annotator::model::SomePBC;
        use crate::flowspace::argument::Signature;
        use crate::translator::rtyper::rpbc::FunctionRepr;
        let entry = DescEntry::function(Rc::new(RefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            name,
            Signature::new(vec![], None, None),
            None,
            None,
        ))));
        let s_pbc = SomePBC::new(vec![entry], false);
        FunctionRepr::new(rtyper, s_pbc).expect("FunctionRepr") as Arc<dyn Repr>
    }

    fn make_custom_dict_with_eq_hash(
        rtyper: Rc<RPythonTyper>,
        eq: Arc<dyn Repr>,
        hash: Arc<dyn Repr>,
    ) -> Arc<OrderedDictRepr> {
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Integer(SomeInteger::default()),
            true,
            false,
            false,
        );
        Arc::new(
            OrderedDictRepr::new(
                rtyper,
                signed_repr() as Arc<dyn Repr>,
                signed_repr() as Arc<dyn Repr>,
                dictdef,
                Some((eq, hash)),
                false,
                false,
            )
            .expect("custom eq/hash ordered dict repr"),
        )
    }

    /// Two FunctionRepr (Void) callback pairs must not share a DICT type:
    /// `get_ll_dict` stores `r_rdict_eqfn` / `r_rdict_hashfn` as the live
    /// Repr objects, which is what specializes `ll_keyhash_custom` /
    /// `ll_keyeq_custom` / `ll_dict_lookup`.
    #[test]
    fn ordereddictrepr_functionrepr_callbacks_keep_distinct_dict_types() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let d1 = make_custom_dict_with_eq_hash(
            rtyper.clone(),
            function_repr_callback(&ann, &rtyper, "eq_a"),
            function_repr_callback(&ann, &rtyper, "hash_a"),
        );
        let d2 = make_custom_dict_with_eq_hash(
            rtyper.clone(),
            function_repr_callback(&ann, &rtyper, "eq_b"),
            function_repr_callback(&ann, &rtyper, "hash_b"),
        );
        assert_eq!(d1.DICTKEY, d2.DICTKEY);
        assert_eq!(
            d1.DICT._flds.get("fnkeyeq"),
            d2.DICT._flds.get("fnkeyeq"),
            "FunctionRepr lowleveltype is Void on both dicts"
        );
        assert_ne!(
            d1.lowleveltype(),
            d2.lowleveltype(),
            "DICT type includes r_rdict_eqfn/r_rdict_hashfn Repr identity"
        );
        assert_ne!(
            d1.DICT._adtmeths.get("r_rdict_eqfn"),
            d2.DICT._adtmeths.get("r_rdict_eqfn")
        );
    }

    /// Same callback-pair Repr objects share a DICT type, so helper cache
    /// keys `(name, args, result)` stay stable for one r_dict specialization.
    #[test]
    fn ordereddictrepr_same_callback_reprs_share_dict_type() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let eq = function_repr_callback(&ann, &rtyper, "eq_shared");
        let hash = function_repr_callback(&ann, &rtyper, "hash_shared");
        let d1 = make_custom_dict_with_eq_hash(rtyper.clone(), eq.clone(), hash.clone());
        let d2 = make_custom_dict_with_eq_hash(rtyper, eq, hash);
        assert_eq!(d1.lowleveltype(), d2.lowleveltype());
    }

    /// Char / instance stored callbacks are not FunctionsPBCRepr funcptrs.
    /// `hlinvoke` must not emit `indirect_call` on a non-function value
    /// (`rtype_hlinvoke` / `SmallFunctionSetPBCRepr.call`).
    #[test]
    fn build_ll_keyhash_custom_rejects_non_funcptr_without_callable_repr() {
        let (dict_ptr, _entries, key_lltype) = sample_dict_lookup_lltypes();
        let err = build_ll_keyhash_custom_helper_graph(
            "ll_keyhash_custom",
            dict_ptr,
            key_lltype,
            LowLevelType::Char,
            None,
            None,
        )
        .expect_err("Char selector without SmallFunctionSetPBCRepr must not emit indirect_call");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("function pointer") || msg.contains("SmallFunctionSet"),
            "got {msg}"
        );
    }

    /// `ll_dict_lookup` `checkingkey == key` uses `float_eq` for Float keys
    /// (`direct_compare` / `ll_dict_lookup`).
    #[test]
    fn build_ll_dict_lookup_float_keys_use_float_eq() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Float(crate::annotator::model::SomeFloat::new()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let repr = OrderedDictRepr::new(
            rtyper,
            crate::translator::rtyper::rfloat::float_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("float-keyed ordered dict");
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let helper = build_ll_dict_lookup_helper_graph(
            "ll_dict_lookup",
            repr.lowleveltype().clone(),
            entries_ptr,
            LowLevelType::Unsigned,
            repr.DICTKEY.clone(),
            None,
            false,
            false,
            None,
            false,
        )
        .expect("float-keyed ll_dict_lookup");
        let inner = helper.graph.borrow();
        let (_n, ops) = walk_blocks(&inner.startblock);
        assert!(
            ops.iter().any(|o| o == "float_eq"),
            "ll_dict_lookup checkingkey == key uses float_eq, got {ops:?}"
        );
    }

    fn simple_hash_eq_custom_dict_repr() -> OrderedDictRepr {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let (eq, hash) = custom_eq_hash_reprs();
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Integer(SomeInteger::default()),
            true,
            false,
            true,
        );
        OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            Some((eq, hash)),
            false,
            true,
        )
        .expect("simple_hash_eq custom dict")
    }

    /// `get_ll_dict` omits `f_hash` when `simple_hash_eq`; insert /
    /// reindex / arraycopy use `ll_hash_custom_fast` instead of writing
    /// the missing field.
    #[test]
    fn simple_hash_eq_omits_f_hash_and_insert_reindex_use_ll_hash_custom_fast() {
        let repr = simple_hash_eq_custom_dict_repr();
        assert!(
            !repr.DICTENTRY._names.iter().any(|n| n == "f_hash"),
            "get_ll_dict omits f_hash when simple_hash_eq, got {:?}",
            repr.DICTENTRY._names
        );
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        assert!(!dictentry_has_f_hash(&entries_ptr));

        let lookup_done = build_ll_dict_setitem_lookup_done_helper_graph(
            "_ll_dict_setitem_lookup_done",
            repr.lowleveltype().clone(),
            entries_ptr.clone(),
            repr.DICTKEY.clone(),
            repr.DICTVALUE.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("setitem_lookup_done without f_hash");
        let inner = lookup_done.graph.borrow();
        let (_n, ops) = walk_blocks(&inner.startblock);
        assert_eq!(
            ops.iter().filter(|o| *o == "setinteriorfield").count(),
            4,
            "overwrite value + insert key/value/f_valid, no f_hash"
        );
        assert!(!graph_mentions_field(&inner.startblock, "f_hash"));

        let reindex = build_ll_dict_reindex_helper_graph(
            "ll_dict_reindex",
            repr.lowleveltype().clone(),
            entries_ptr.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            Some(dummy_funcptr_const()),
            true,
        )
        .expect("reindex ll_hash_custom_fast");
        let inner = reindex.graph.borrow();
        assert!(!graph_mentions_field(&inner.startblock, "f_hash"));
        let (_n, ops) = walk_blocks(&inner.startblock);
        assert_eq!(
            ops.iter().filter(|o| *o == "direct_call").count(),
            4,
            "clear_indexes + malloc_choose + ll_hash_custom_fast + store_clean"
        );

        let arraycopy = build_ll_dict_entries_arraycopy_helper_graph(
            "ll_dict_entries_arraycopy",
            entries_ptr,
            repr.DICTKEY.clone(),
            repr.DICTVALUE.clone(),
        )
        .expect("arraycopy without f_hash");
        let inner = arraycopy.graph.borrow();
        assert!(!graph_mentions_field(&inner.startblock, "f_hash"));
    }

    fn as_dict_ptr(c: &Constant) -> &_ptr {
        match &c.value {
            ConstValue::LLPtr(ptr) => ptr,
            other => panic!("convert_const must yield LLPtr, got {other:?}"),
        }
    }

    fn entry_signed_keys(ptr: &_ptr) -> Vec<i64> {
        let entries = ptr.getattr("entries").expect("entries");
        let LowLevelValue::Ptr(entries) = entries else {
            panic!("entries must be a pointer, got {entries:?}");
        };
        let n = dict_signed_field(ptr, "num_ever_used_items").expect("ever") as usize;
        let mut keys = Vec::with_capacity(n);
        for i in 0..n {
            let entry = entries.getitem(i).expect("entry");
            let key = match entry {
                LowLevelValue::InteriorPtr(ip) => ip.getattr("key").expect("key"),
                LowLevelValue::Ptr(p) => p.getattr("key").expect("key"),
                other => panic!("entry must be a struct pointer, got {other:?}"),
            };
            let LowLevelValue::Signed(k) = key else {
                panic!("key must be Signed, got {key:?}");
            };
            keys.push(k);
        }
        keys
    }

    fn signed_ccharp_ordered_dict_repr() -> OrderedDictRepr {
        use crate::translator::rtyper::lltypesystem::lltype::SomePtr;
        use crate::translator::rtyper::lltypesystem::rffi::CCHARP;
        use crate::translator::rtyper::rmodel::PtrRepr;
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let ccharp_ptr = match &*CCHARP {
            LowLevelType::Ptr(ptr) => (**ptr).clone(),
            other => panic!("CCHARP is not a Ptr: {other:?}"),
        };
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Ptr(SomePtr::new(ccharp_ptr.clone())),
            false,
            false,
            false,
        );
        OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            Arc::new(PtrRepr::new(ccharp_ptr)) as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("signed CCHARP ordered dict repr")
    }

    /// `OrderedDictRepr.convert_const(Dict({}))` (`rordereddict.py`):
    /// immortal `ll_newdict_size(DICT, 0)` + `ll_no_initial_index`.
    #[test]
    fn convert_const_empty_dict_is_immortal_dicttable_ptr() {
        let repr = sample_ordered_dict_repr();
        let c = repr
            .convert_const(&ConstValue::Dict(Default::default()))
            .expect("empty dict convert_const");
        let ptr = as_dict_ptr(&c);
        assert!(ptr._solid, "prebuilt dict is immortal");
        match &ptr._TYPE.TO {
            PtrTarget::Struct(s) => assert_eq!(s._name, "dicttable"),
            other => panic!("DICT pointer must target dicttable, got {other:?}"),
        }
        assert_eq!(
            dict_signed_field(ptr, "num_live_items").expect("num_live_items"),
            0
        );
        assert_eq!(
            dict_signed_field(ptr, "num_ever_used_items").expect("num_ever_used_items"),
            0
        );
        assert_eq!(
            dict_signed_field(ptr, "lookup_function_no").expect("lookup_function_no"),
            FUNC_MUST_REINDEX
        );
        let entries = ptr.getattr("entries").expect("entries");
        let LowLevelValue::Ptr(entries) = entries else {
            panic!("entries must be a pointer, got {entries:?}");
        };
        assert_eq!(entries.len().expect("entries length"), 0);
        let indexes = ptr.getattr("indexes").expect("indexes");
        let LowLevelValue::Ptr(indexes) = indexes else {
            panic!("indexes must be a pointer, got {indexes:?}");
        };
        assert!(
            matches!(indexes._obj0_value(), Ok(None)),
            "ll_no_initial_index leaves indexes null"
        );
        let none = repr.convert_const(&ConstValue::None).expect("None");
        let none_ptr = as_dict_ptr(&none);
        assert!(
            matches!(none_ptr._obj0_value(), Ok(None)),
            "None converts to nullptr(DICT)"
        );
    }

    /// Small `{int: CCHARP-null}` prebuilt: one `_ll_dict_insert_no_index`.
    /// `PtrRepr.convert_const` accepts a typed null `LLPtr`, not host `None`.
    #[test]
    fn convert_const_int_to_ccharp_null_inserts_one_entry() {
        use crate::translator::rtyper::lltypesystem::rffi::CCHARP;
        let repr = signed_ccharp_ordered_dict_repr();
        let mut items = indexmap::IndexMap::new();
        let LowLevelType::Ptr(ptr_t) = &*CCHARP else {
            panic!("CCHARP is not a Ptr");
        };
        items.insert(
            ConstValue::Int(7),
            ConstValue::LLPtr(Box::new(ptr_t._defl())),
        );
        let c = repr
            .convert_const(&ConstValue::Dict(Arc::new(items)))
            .expect("non-empty dict convert_const");
        let ptr = as_dict_ptr(&c);
        assert!(ptr._solid, "prebuilt dict is immortal");
        assert_eq!(
            dict_signed_field(ptr, "num_live_items").expect("num_live_items"),
            1
        );
        let entries = ptr.getattr("entries").expect("entries");
        let LowLevelValue::Ptr(entries) = entries else {
            panic!("entries must be a pointer, got {entries:?}");
        };
        assert_eq!(entries.len().expect("entries length"), 1);
        let entry = entries.getitem(0).expect("entries[0]");
        let (key, value, valid) = match entry {
            LowLevelValue::InteriorPtr(ip) => (
                ip.getattr("key").expect("key"),
                ip.getattr("value").expect("value"),
                ip.getattr("f_valid").expect("f_valid"),
            ),
            LowLevelValue::Ptr(p) => (
                p.getattr("key").expect("key"),
                p.getattr("value").expect("value"),
                p.getattr("f_valid").expect("f_valid"),
            ),
            other => panic!("entry must be a struct pointer, got {other:?}"),
        };
        assert!(
            matches!(key, LowLevelValue::Signed(7)),
            "key must be 7, got {key:?}"
        );
        let LowLevelValue::Ptr(value_ptr) = value else {
            panic!("CCHARP value must be a pointer, got {value:?}");
        };
        assert!(
            matches!(value_ptr._obj0_value(), Ok(None)),
            "CCHARP-null value"
        );
        assert!(
            matches!(valid, LowLevelValue::Bool(true)),
            "f_valid must be True, got {valid:?}"
        );
    }

    /// `dict_cache` keys by `id(dictobj)`: the same host dict object
    /// (or a clone of that `ConstValue`) reuses the prebuilt pointer;
    /// a distinct empty dict does not.
    #[test]
    fn convert_const_caches_by_host_dict_identity() {
        let repr = sample_ordered_dict_repr();
        let host = ConstValue::Dict(Default::default());
        let c1 = repr.convert_const(&host).expect("first");
        let c2 = repr.convert_const(&host).expect("second");
        let p1 = as_dict_ptr(&c1);
        let p2 = as_dict_ptr(&c2);
        assert_eq!(
            p1._hashable_identity(),
            p2._hashable_identity(),
            "same host dict identity shares the prebuilt pointer"
        );
        let cloned = host.clone();
        let c_clone = repr.convert_const(&cloned).expect("clone");
        assert_eq!(
            p1._hashable_identity(),
            as_dict_ptr(&c_clone)._hashable_identity(),
            "a clone of the same ConstValue::Dict shares the prebuilt pointer"
        );
        let other = ConstValue::Dict(Default::default());
        let c3 = repr.convert_const(&other).expect("other");
        let p3 = as_dict_ptr(&c3);
        assert_ne!(
            p1._hashable_identity(),
            p3._hashable_identity(),
            "a distinct host dict mints a fresh prebuilt pointer"
        );
    }

    /// Cache keys hold a clone of the host `ConstValue::Dict` Arc and
    /// hash `Arc::as_ptr` (`uid.py` `id(dictobj)`). The strong
    /// reference keeps the allocation alive so a later dict cannot
    /// reuse the address and alias the cache entry.
    #[test]
    fn convert_const_cache_retains_host_constant() {
        let repr = sample_ordered_dict_repr();
        let host = ConstValue::Dict(Default::default());
        let host_items = match &host {
            ConstValue::Dict(items) => Arc::clone(items),
            other => panic!("expected Dict, got {other:?}"),
        };
        let c1 = repr.convert_const(&host).expect("convert");
        {
            let cache = repr.dict_cache.borrow();
            assert_eq!(cache.len(), 1);
            let key = cache.keys().next().expect("cache key");
            let key_items = dict_cache_key_dict_items(key);
            assert!(
                Arc::ptr_eq(key_items, &host_items),
                "cache key holds the host dict Arc"
            );
            assert!(
                Arc::strong_count(key_items) >= 2,
                "key and host both hold the allocation"
            );
        }
        let p1 = as_dict_ptr(&c1)._hashable_identity();
        drop(host);
        {
            let cache = repr.dict_cache.borrow();
            let key = cache.keys().next().expect("cache key");
            let key_items = dict_cache_key_dict_items(key);
            assert!(
                Arc::strong_count(key_items) >= 1,
                "cache key keeps the host dict allocation alive"
            );
            assert!(Arc::ptr_eq(key_items, &host_items));
        }
        let other = ConstValue::Dict(Default::default());
        let c2 = repr.convert_const(&other).expect("other");
        assert_eq!(
            repr.dict_cache.borrow().len(),
            2,
            "a distinct host dict after the first is dropped is a new cache key"
        );
        assert_ne!(
            p1,
            as_dict_ptr(&c2)._hashable_identity(),
            "dropped dict cannot produce a false cache hit"
        );
    }

    #[test]
    fn dict_cache_key_eq_is_uid_identity_not_constant_id() {
        let host = ConstValue::Dict(Default::default());
        let k1 = dict_cache_key(&host).expect("k1");
        let k2 = dict_cache_key(&host).expect("k2");
        assert_eq!(k1, k2, "Hashable eq is id(dictobj), not Constant.id");
        assert!(
            Arc::ptr_eq(
                dict_cache_key_dict_items(&k1),
                dict_cache_key_dict_items(&k2)
            ),
            "keys cloned from the same ConstValue share the dict Arc"
        );
        let cloned = host.clone();
        let k_clone = dict_cache_key(&cloned).expect("clone");
        assert_eq!(
            k1, k_clone,
            "a cloned ConstValue::Dict is the same identity"
        );
        let other = ConstValue::Dict(Default::default());
        let k3 = dict_cache_key(&other).expect("k3");
        assert_eq!(
            host, other,
            "equal contents still compare equal as ConstValue"
        );
        assert_ne!(k1, k3, "distinct dict identities do not collide");
    }

    /// `OrderedDictRepr.convert_const` r_dict arm (`rordereddict.py`):
    /// `fnkeyeq` / `fnkeyhash` from `key_eq` / `key_hash`, inserts via
    /// `_r_dictkey.key`, `ll_no_initial_index`, cached by host identity.
    #[test]
    fn convert_const_prebuilt_r_dict_sets_fnkey_and_inserts_keys() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_custom_eq_hash_dict_repr(rtyper);
        let key_eq = funcptr_const(
            "key_eq",
            vec![LowLevelType::Signed, LowLevelType::Signed],
            LowLevelType::Bool,
        );
        let key_hash = funcptr_const("key_hash", vec![LowLevelType::Signed], LowLevelType::Signed);
        let r_dict = make_prebuilt_r_dict(key_eq.clone(), key_hash.clone(), &[(1, 10), (2, 20)]);
        let host = ConstValue::HostObject(r_dict);
        let c1 = repr.convert_const(&host).expect("r_dict convert_const");
        let p1 = as_dict_ptr(&c1);
        assert_eq!(dict_signed_field(p1, "num_live_items").expect("live"), 2);
        assert_eq!(
            dict_signed_field(p1, "num_ever_used_items").expect("ever"),
            2
        );
        assert_eq!(
            dict_signed_field(p1, "lookup_function_no").expect("lookup"),
            FUNC_MUST_REINDEX
        );
        let indexes = p1.getattr("indexes").expect("indexes");
        let LowLevelValue::Ptr(indexes) = indexes else {
            panic!("indexes must be a pointer, got {indexes:?}");
        };
        assert!(
            matches!(indexes._obj0_value(), Ok(None)),
            "ll_no_initial_index leaves indexes as nullptr"
        );

        let fnkeyeq = p1.getattr("fnkeyeq").expect("fnkeyeq");
        let fnkeyhash = p1.getattr("fnkeyhash").expect("fnkeyhash");
        let LowLevelValue::Ptr(eq_ptr) = fnkeyeq else {
            panic!("fnkeyeq must be a pointer, got {fnkeyeq:?}");
        };
        let LowLevelValue::Ptr(hash_ptr) = fnkeyhash else {
            panic!("fnkeyhash must be a pointer, got {fnkeyhash:?}");
        };
        let ConstValue::LLPtr(orig_eq) = &key_eq else {
            panic!("key_eq LLPtr");
        };
        let ConstValue::LLPtr(orig_hash) = &key_hash else {
            panic!("key_hash LLPtr");
        };
        assert_eq!(
            eq_ptr._hashable_identity(),
            orig_eq._hashable_identity(),
            "fnkeyeq is the converted key_eq funcptr"
        );
        assert_eq!(
            hash_ptr._hashable_identity(),
            orig_hash._hashable_identity(),
            "fnkeyhash is the converted key_hash funcptr"
        );

        let entries = p1.getattr("entries").expect("entries");
        let LowLevelValue::Ptr(entries) = entries else {
            panic!("entries must be a pointer, got {entries:?}");
        };
        assert_eq!(entries.len().expect("entries length"), 2);
        let mut kv = Vec::new();
        for i in 0..2 {
            let entry = entries.getitem(i).expect("entry");
            let (key, value, valid) = match entry {
                LowLevelValue::InteriorPtr(ip) => (
                    ip.getattr("key").expect("key"),
                    ip.getattr("value").expect("value"),
                    ip.getattr("f_valid").expect("f_valid"),
                ),
                LowLevelValue::Ptr(p) => (
                    p.getattr("key").expect("key"),
                    p.getattr("value").expect("value"),
                    p.getattr("f_valid").expect("f_valid"),
                ),
                other => panic!("entry must be a struct pointer, got {other:?}"),
            };
            assert!(
                matches!(valid, LowLevelValue::Bool(true)),
                "f_valid must be True, got {valid:?}"
            );
            let LowLevelValue::Signed(k) = key else {
                panic!("key must be Signed, got {key:?}");
            };
            let LowLevelValue::Signed(v) = value else {
                panic!("value must be Signed, got {value:?}");
            };
            kv.push((k, v));
        }
        assert_eq!(kv, vec![(1, 10), (2, 20)]);

        let c2 = repr.convert_const(&host).expect("cached");
        assert_eq!(
            p1._hashable_identity(),
            as_dict_ptr(&c2)._hashable_identity(),
            "same r_dict reuses the cached pointer"
        );
        let other = make_prebuilt_r_dict(key_eq, key_hash, &[(1, 10), (2, 20)]);
        let c3 = repr
            .convert_const(&ConstValue::HostObject(other))
            .expect("other r_dict");
        assert_ne!(
            p1._hashable_identity(),
            as_dict_ptr(&c3)._hashable_identity(),
            "equal contents on a distinct r_dict mint a fresh pointer"
        );
    }

    fn signed_signed_ordered_dict_repr() -> OrderedDictRepr {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("signed signed ordered dict repr")
    }

    /// `ll_newdict_size` ends with `ll_malloc_indexes_and_choose_lookup(d, n)`.
    #[test]
    fn ll_newdict_size_allocates_indexes_and_chooses_lookup() {
        let repr = signed_signed_ordered_dict_repr();
        let d = ll_newdict_size(&repr.DICT, 4).expect("ll_newdict_size");
        assert_eq!(
            dict_signed_field(&d, "lookup_function_no").expect("lookup_function_no"),
            FUNC_BYTE
        );
        assert_eq!(
            dict_signed_field(&d, "resize_counter").expect("resize_counter"),
            DICT_INITSIZE * 2
        );
        let indexes = d.getattr("indexes").expect("indexes");
        let LowLevelValue::Ptr(indexes) = indexes else {
            panic!("indexes must be a pointer, got {indexes:?}");
        };
        assert!(indexes.nonzero(), "ll_newdict_size allocates indexes");
        let array = dict_indexes_array(&d).expect("cast indexes");
        assert_eq!(
            array.len().expect("index length"),
            DICT_INITSIZE as usize,
            "n starts at DICT_INITSIZE for orig_length_estimate=4"
        );
        let n = 300usize;
        let mut d_short = ll_newdict_size(&repr.DICT, n).expect("short indexes");
        // length_estimate = (300/2)*3 = 450, n doubles 16..512.
        assert_eq!(
            dict_signed_field(&d_short, "lookup_function_no").expect("lookup_function_no"),
            FUNC_SHORT
        );
        ll_malloc_indexes_and_choose_lookup(&mut d_short, 65537).expect("long indexes");
        assert_eq!(
            dict_signed_field(&d_short, "lookup_function_no").expect("lookup_function_no"),
            FUNC_INT
        );
    }

    /// `ll_dict_setitem` then `ll_dict_getitem` round-trip; missing key
    /// raises KeyError.
    #[test]
    fn ll_dict_getitem_setitem_hit_miss_and_round_trip() {
        let repr = signed_signed_ordered_dict_repr();
        let mut d = ll_newdict_size(&repr.DICT, 4).expect("ll_newdict_size");
        ll_dict_setitem(&mut d, LowLevelValue::Signed(7), LowLevelValue::Signed(9))
            .expect("setitem 7 -> 9");
        let got = ll_dict_getitem(&mut d, &LowLevelValue::Signed(7)).expect("getitem 7");
        assert!(
            matches!(got, LowLevelValue::Signed(9)),
            "getitem hit, got {got:?}"
        );
        let miss = ll_dict_getitem(&mut d, &LowLevelValue::Signed(8)).expect_err("missing key");
        assert!(
            miss.to_string().contains("KeyError"),
            "getitem miss raises KeyError, got {miss}"
        );
        ll_dict_setitem(&mut d, LowLevelValue::Signed(7), LowLevelValue::Signed(11))
            .expect("overwrite 7 -> 11");
        let got = ll_dict_getitem(&mut d, &LowLevelValue::Signed(7)).expect("getitem overwrite");
        assert!(
            matches!(got, LowLevelValue::Signed(11)),
            "setitem overwrite round trip, got {got:?}"
        );
        ll_dict_setitem(&mut d, LowLevelValue::Signed(3), LowLevelValue::Signed(4))
            .expect("setitem 3 -> 4");
        let got = ll_dict_getitem(&mut d, &LowLevelValue::Signed(3)).expect("getitem 3");
        assert!(
            matches!(got, LowLevelValue::Signed(4)),
            "second insert, got {got:?}"
        );
    }

    /// Nonempty prebuilt dict: `ll_dict_create_initial_index` rehashes
    /// and every key is found.
    #[test]
    fn convert_const_nonempty_prebuilt_getitem_after_rehash() {
        let repr = signed_signed_ordered_dict_repr();
        let mut items = indexmap::IndexMap::new();
        items.insert(ConstValue::Int(1), ConstValue::Int(10));
        items.insert(ConstValue::Int(2), ConstValue::Int(20));
        items.insert(ConstValue::Int(3), ConstValue::Int(30));
        let c = repr
            .convert_const(&ConstValue::Dict(Arc::new(items)))
            .expect("prebuilt");
        let mut d = as_dict_ptr(&c).clone();
        assert_eq!(
            dict_signed_field(&d, "lookup_function_no").expect("lookup"),
            FUNC_MUST_REINDEX
        );
        for (k, v) in [(1, 10), (2, 20), (3, 30)] {
            let got = ll_dict_getitem(&mut d, &LowLevelValue::Signed(k)).expect("getitem");
            assert!(
                matches!(got, LowLevelValue::Signed(n) if n == v),
                "prebuilt key {k} -> {v}, got {got:?}"
            );
        }
        assert_ne!(
            dict_signed_field(&d, "lookup_function_no").expect("lookup after rehash"),
            FUNC_MUST_REINDEX
        );
    }

    /// `OrderedDictRepr.convert_const` (`rordereddict.py`) inserts from
    /// `dictobj.items()`; `ConstValue::Dict` keeps that insertion order.
    #[test]
    fn convert_const_prebuilt_dict_preserves_insertion_order() {
        let repr = signed_signed_ordered_dict_repr();
        let mut items = indexmap::IndexMap::new();
        items.insert(ConstValue::Int(3), ConstValue::Int(30));
        items.insert(ConstValue::Int(1), ConstValue::Int(10));
        items.insert(ConstValue::Int(2), ConstValue::Int(20));
        let c = repr
            .convert_const(&ConstValue::Dict(Arc::new(items)))
            .expect("prebuilt dict");
        let ptr = as_dict_ptr(&c);
        assert_eq!(
            dict_signed_field(ptr, "num_ever_used_items").expect("ever"),
            3
        );
        assert_eq!(entry_signed_keys(ptr), vec![3, 1, 2]);
    }

    /// r_dict `_dict` arm of `OrderedDictRepr.convert_const`
    /// (`rordereddict.py`): entries follow `_dict` insertion order.
    #[test]
    fn convert_const_prebuilt_r_dict_preserves_insertion_order() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_custom_eq_hash_dict_repr(rtyper);
        let key_eq = funcptr_const(
            "key_eq",
            vec![LowLevelType::Signed, LowLevelType::Signed],
            LowLevelType::Bool,
        );
        let key_hash = funcptr_const("key_hash", vec![LowLevelType::Signed], LowLevelType::Signed);
        let r_dict = make_prebuilt_r_dict(key_eq, key_hash, &[(3, 30), (1, 10), (2, 20)]);
        let c = repr
            .convert_const(&ConstValue::HostObject(r_dict))
            .expect("prebuilt r_dict");
        let ptr = as_dict_ptr(&c);
        assert_eq!(
            dict_signed_field(ptr, "num_ever_used_items").expect("ever"),
            3
        );
        assert_eq!(entry_signed_keys(ptr), vec![3, 1, 2]);
    }

    /// Host `ll_keyhash_custom` / `ll_keyeq_custom` run the stored
    /// `fnkeyhash` / `fnkeyeq` graphs (`objectmodel.hlinvoke`).
    #[test]
    fn host_custom_r_dict_lookup_runs_callbacks() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = make_custom_eq_hash_dict_repr(rtyper);
        let mut d = ll_newdict_size(&repr.DICT, 4).expect("ll_newdict_size");
        let ConstValue::LLPtr(hash_p) =
            signed_callback_funcptr("key_hash", 1, LowLevelType::Signed, None)
        else {
            panic!("hash funcptr");
        };
        let ConstValue::LLPtr(eq_p) =
            signed_callback_funcptr("key_eq", 2, LowLevelType::Bool, Some("int_eq"))
        else {
            panic!("eq funcptr");
        };
        d.setattr("fnkeyhash", LowLevelValue::Ptr(hash_p))
            .expect("fnkeyhash");
        d.setattr("fnkeyeq", LowLevelValue::Ptr(eq_p))
            .expect("fnkeyeq");
        ll_dict_setitem(&mut d, LowLevelValue::Signed(7), LowLevelValue::Signed(70))
            .expect("setitem");
        let got = ll_dict_getitem(&mut d, &LowLevelValue::Signed(7)).expect("getitem");
        assert!(
            matches!(got, LowLevelValue::Signed(70)),
            "custom r_dict lookup returned {got:?}"
        );
    }

    /// Inserting `DICT_INITSIZE * 2` keys grows/resizes; every key is found.
    #[test]
    fn ll_dict_setitem_grows_and_every_key_is_found() {
        let repr = signed_signed_ordered_dict_repr();
        let mut d = ll_newdict_size(&repr.DICT, 4).expect("ll_newdict_size");
        let n = (DICT_INITSIZE * 2) as i64;
        for i in 0..n {
            ll_dict_setitem(
                &mut d,
                LowLevelValue::Signed(i),
                LowLevelValue::Signed(i + 100),
            )
            .expect("setitem grow");
        }
        for i in 0..n {
            let got =
                ll_dict_getitem(&mut d, &LowLevelValue::Signed(i)).expect("getitem after grow");
            assert!(
                matches!(got, LowLevelValue::Signed(v) if v == i + 100),
                "key {i} after grow, got {got:?}"
            );
        }
        let live = dict_signed_field(&d, "num_live_items").expect("live");
        assert_eq!(live, n);
        let entries = dict_entries(&d).expect("entries");
        assert!(
            entries.len().expect("entries len") >= n as usize,
            "grow must have expanded entries"
        );
    }

    /// `ll_no_initial_index` asserts `num_live_items == num_ever_used_items`.
    #[test]
    fn ll_no_initial_index_after_delete_fails_assert() {
        let repr = signed_signed_ordered_dict_repr();
        let mut d = ll_newdict_size(&repr.DICT, 4).expect("ll_newdict_size");
        ll_dict_setitem(&mut d, LowLevelValue::Signed(1), LowLevelValue::Signed(2))
            .expect("setitem");
        ll_dict_setitem(&mut d, LowLevelValue::Signed(3), LowLevelValue::Signed(4))
            .expect("setitem 2");
        ll_dict_delitem(&mut d, &LowLevelValue::Signed(1)).expect("delitem");
        let live = dict_signed_field(&d, "num_live_items").expect("live");
        let ever = dict_signed_field(&d, "num_ever_used_items").expect("ever");
        assert!(live < ever, "delete leaves a deleted slot");
        let err = ll_no_initial_index(&mut d).expect_err("assert");
        assert!(
            err.to_string().contains("dict already in use"),
            "ll_no_initial_index assert, got {err}"
        );
    }

    fn str_signed_ordered_dict_repr() -> OrderedDictRepr {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::String(SomeString::new(false, false)),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        OrderedDictRepr::new(
            rtyper,
            string_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("str signed ordered dict repr")
    }

    /// STR-keyed dict: equal-but-distinct STR finds the entry via `ll_streq`.
    #[test]
    fn ll_dict_getitem_str_key_equal_but_distinct() {
        let repr = str_signed_ordered_dict_repr();
        let mut d = ll_newdict_size(&repr.DICT, 4).expect("ll_newdict_size");
        let k1 = llstr(b"hello").expect("k1");
        let k2 = llstr(b"hello").expect("k2");
        assert_ne!(
            k1._hashable_identity(),
            k2._hashable_identity(),
            "llstr produces distinct containers"
        );
        ll_dict_setitem(
            &mut d,
            LowLevelValue::Ptr(Box::new(k1)),
            LowLevelValue::Signed(42),
        )
        .expect("setitem STR");
        let got = ll_dict_getitem(&mut d, &LowLevelValue::Ptr(Box::new(k2))).expect("getitem STR");
        assert!(
            matches!(got, LowLevelValue::Signed(42)),
            "equal-but-distinct STR finds the entry, got {got:?}"
        );
    }

    fn str_str_ordered_dict_repr() -> OrderedDictRepr {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::String(SomeString::new(false, false)),
            SomeValue::String(SomeString::new(true, false)),
            false,
            false,
            false,
        );
        OrderedDictRepr::new(
            rtyper,
            string_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("str str ordered dict repr")
    }

    /// `{ "a": None }` with a nullable StringRepr value: `convert_const`
    /// passes host `None` through; the entry value is a null STR pointer.
    #[test]
    fn convert_const_str_none_value_is_null_str_pointer() {
        let repr = str_str_ordered_dict_repr();
        let mut items = indexmap::IndexMap::new();
        items.insert(ConstValue::byte_str("a"), ConstValue::None);
        let c = repr
            .convert_const(&ConstValue::Dict(Arc::new(items)))
            .expect("{a: None}");
        let ptr = as_dict_ptr(&c);
        let entries = ptr.getattr("entries").expect("entries");
        let LowLevelValue::Ptr(entries) = entries else {
            panic!("entries must be a pointer, got {entries:?}");
        };
        let entry = entries.getitem(0).expect("entries[0]");
        let value = match entry {
            LowLevelValue::InteriorPtr(ip) => ip.getattr("value").expect("value"),
            LowLevelValue::Ptr(p) => p.getattr("value").expect("value"),
            other => panic!("entry must be a struct pointer, got {other:?}"),
        };
        let LowLevelValue::Ptr(value_ptr) = value else {
            panic!("STR value must be a pointer, got {value:?}");
        };
        assert!(
            matches!(value_ptr._obj0_value(), Ok(None)),
            "None value is a null STR pointer"
        );
    }

    fn float_signed_ordered_dict_repr() -> OrderedDictRepr {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Float(SomeFloat::new()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        OrderedDictRepr::new(
            rtyper,
            float_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("float signed ordered dict repr")
    }

    /// Float keys use `_hash_float` (`objectmodel.py`). 1.5 hashes to
    /// 1610645504; `0.0` and `-0.0` share hash 0 and find each other.
    #[test]
    fn convert_const_float_keys_hash_and_lookup() {
        let repr = float_signed_ordered_dict_repr();
        assert!(
            matches!(
                repr.DICT._adtmeths.get("keyhash"),
                Some(ConstValue::LLPtr(_))
            ),
            "keyhash adtmeth is the stored _hash_float funcptr"
        );
        let mut items = indexmap::IndexMap::new();
        items.insert(ConstValue::float(1.5), ConstValue::Int(10));
        items.insert(ConstValue::float(0.0), ConstValue::Int(20));
        let c = repr
            .convert_const(&ConstValue::Dict(Arc::new(items)))
            .expect("float dict");
        let mut d = as_dict_ptr(&c).clone();
        let h_15 = ll_keyhash(&d, &LowLevelValue::Float(1.5f64.to_bits())).expect("hash 1.5");
        assert_eq!(h_15, 1610645504, "_hash_float(1.5)");
        let h_pos0 = ll_keyhash(&d, &LowLevelValue::Float(0.0f64.to_bits())).expect("hash 0.0");
        let h_neg0 = ll_keyhash(&d, &LowLevelValue::Float((-0.0f64).to_bits())).expect("hash -0.0");
        assert_eq!(h_pos0, 0);
        assert_eq!(h_neg0, 0);
        let got = ll_dict_getitem(&mut d, &LowLevelValue::Float(1.5f64.to_bits())).expect("1.5");
        assert!(
            matches!(got, LowLevelValue::Signed(10)),
            "lookup 1.5, {got:?}"
        );
        let got = ll_dict_getitem(&mut d, &LowLevelValue::Float((-0.0f64).to_bits()))
            .expect("-0.0 finds 0.0");
        assert!(
            matches!(got, LowLevelValue::Signed(20)),
            "0.0 == -0.0 lookup, {got:?}"
        );
    }

    fn tuple_int_str_signed_ordered_dict_repr() -> (OrderedDictRepr, Arc<TupleRepr>) {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let tuple_repr = Arc::new(
            TupleRepr::new(
                &rtyper,
                vec![
                    signed_repr() as Arc<dyn Repr>,
                    string_repr() as Arc<dyn Repr>,
                ],
            )
            .expect("TupleRepr"),
        );
        let dictdef = DictDef::new(
            None,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::String(SomeString::new(false, false)),
            ])),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let dict_repr = OrderedDictRepr::new(
            rtyper,
            tuple_repr.clone() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("tuple keyed dict");
        (dict_repr, tuple_repr)
    }

    /// Tuple keys `(1, "a")` hash structurally (`gen_hash_function`); two
    /// equal tuples built separately find each other.
    #[test]
    fn convert_const_tuple_keys_hash_structurally() {
        let (repr, tuple_repr) = tuple_int_str_signed_ordered_dict_repr();
        let mut items = indexmap::IndexMap::new();
        items.insert(
            ConstValue::Tuple(vec![ConstValue::Int(1), ConstValue::byte_str("a")]),
            ConstValue::Int(99),
        );
        let c = repr
            .convert_const(&ConstValue::Dict(Arc::new(items)))
            .expect("tuple dict");
        let mut d = as_dict_ptr(&c).clone();
        let mut t2 = tuple_repr.instantiate().expect("fresh tuple");
        t2.setattr("item0", LowLevelValue::Signed(1))
            .expect("item0");
        let s = llstr(b"a").expect("fresh STR");
        t2.setattr("item1", LowLevelValue::Ptr(Box::new(s)))
            .expect("item1");
        let got = ll_dict_getitem(&mut d, &LowLevelValue::Ptr(Box::new(t2)))
            .expect("structural tuple lookup");
        assert!(
            matches!(got, LowLevelValue::Signed(99)),
            "separately built (1, 'a') finds the entry, got {got:?}"
        );
    }

    fn malloc_unicode(text: &str) -> _ptr {
        let unicode_body = match UNICODE.clone() {
            LowLevelType::ForwardReference(fwd) => fwd
                .resolved()
                .expect("UNICODE ForwardReference is resolved"),
            other => other,
        };
        let mut ptr = malloc(
            unicode_body,
            Some(text.chars().count()),
            MallocFlavor::Gc,
            true,
        )
        .expect("malloc UNICODE");
        ptr.setattr("hash", LowLevelValue::Signed(0))
            .expect("hash = 0");
        let Some(_ptr_obj::Struct(s)) = ptr._obj0_value().expect("live UNICODE") else {
            panic!("UNICODE pointer does not target a Struct");
        };
        let Some(LowLevelValue::Array(chars)) = s._getattr("chars") else {
            panic!("UNICODE has no chars array");
        };
        for (i, ch) in text.chars().enumerate() {
            chars.setitem(i, LowLevelValue::UniChar(ch));
        }
        ptr
    }

    fn unicode_signed_ordered_dict_repr() -> OrderedDictRepr {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::UnicodeString(crate::annotator::model::SomeUnicodeString::new(
                false, false,
            )),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        OrderedDictRepr::new(
            rtyper,
            unicode_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("unicode signed ordered dict repr")
    }

    /// UNICODE keys hash UniChar code points (`_hash_string`), not UTF-8
    /// bytes. U+0100 differs from the UTF-8-byte hash of C4 80.
    #[test]
    fn unicode_key_hash_uses_codepoints_not_utf8() {
        let repr = unicode_signed_ordered_dict_repr();
        let mut d = ll_newdict_size(&repr.DICT, 4).expect("ll_newdict_size");
        let key = malloc_unicode("\u{0100}");
        let codepoint_hash = hash_unichar_ords(&['\u{0100}']);
        let utf8_hash = ll_strhash_value("\u{0100}".as_bytes());
        assert_ne!(
            codepoint_hash, utf8_hash,
            "U+0100 code-point hash differs from UTF-8-byte hash ({codepoint_hash} vs {utf8_hash})"
        );
        let got = ll_keyhash(&d, &LowLevelValue::Ptr(Box::new(key.clone()))).expect("unicode hash");
        assert_eq!(
            got, codepoint_hash,
            "UNICODE hash is the code-point formula"
        );
        ll_dict_setitem(
            &mut d,
            LowLevelValue::Ptr(Box::new(key)),
            LowLevelValue::Signed(7),
        )
        .expect("setitem UNICODE");
        let key2 = malloc_unicode("\u{0100}");
        let looked =
            ll_dict_getitem(&mut d, &LowLevelValue::Ptr(Box::new(key2))).expect("getitem UNICODE");
        assert!(
            matches!(looked, LowLevelValue::Signed(7)),
            "equal UNICODE code points find the entry, got {looked:?}"
        );
    }

    /// `math.frexp` subnormal path starts at `-1021` so the smallest
    /// positive subnormal `2**-1074` is `(0.5, -1073)`.
    ///
    /// `objectmodel._hash_float` arithmetic:
    /// `v, expo = (0.5, -1073)`; `v *= 2**31` → `1073741824`;
    /// `hipart + int(v) + (expo << 15)`
    /// = `1073741824 + 0 + (-1073 * 32768)` = `1038581760`.
    #[test]
    fn python_frexp_smallest_subnormal_matches_hash_float() {
        let smallest = f64::from_bits(1);
        let (mant, exp) = python_frexp(smallest);
        assert_eq!(mant, 0.5);
        assert_eq!(exp, -1073);
        assert_eq!(hash_float_value(smallest), 1038581760);
        let repr = float_signed_ordered_dict_repr();
        let mut items = indexmap::IndexMap::new();
        items.insert(ConstValue::float(smallest), ConstValue::Int(1));
        let c = repr
            .convert_const(&ConstValue::Dict(Arc::new(items)))
            .expect("subnormal dict");
        let d = as_dict_ptr(&c).clone();
        let got = ll_keyhash(&d, &LowLevelValue::Float(smallest.to_bits()))
            .expect("hash smallest subnormal");
        assert_eq!(got, 1038581760, "_hash_float(2**-1074)");
    }

    fn empty_tuple_signed_ordered_dict_repr() -> OrderedDictRepr {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let empty_tuple = Arc::new(TupleRepr::new(&rtyper, vec![]).expect("empty TupleRepr"));
        let dictdef = DictDef::new(
            None,
            SomeValue::Tuple(SomeTuple::new(vec![])),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        OrderedDictRepr::new(
            rtyper,
            empty_tuple as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("empty-tuple keyed dict")
    }

    fn nested_empty_tuple_int_signed_ordered_dict_repr() -> (OrderedDictRepr, Arc<TupleRepr>) {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let empty_tuple = Arc::new(TupleRepr::new(&rtyper, vec![]).expect("empty TupleRepr"));
        let nested = Arc::new(
            TupleRepr::new(
                &rtyper,
                vec![empty_tuple as Arc<dyn Repr>, signed_repr() as Arc<dyn Repr>],
            )
            .expect("nested TupleRepr"),
        );
        let dictdef = DictDef::new(
            None,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Tuple(SomeTuple::new(vec![])),
                SomeValue::Integer(SomeInteger::default()),
            ])),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let dict_repr = OrderedDictRepr::new(
            rtyper,
            nested.clone() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("nested-tuple keyed dict");
        (dict_repr, nested)
    }

    /// Empty-tuple keys use `gen_hash_function([])`, which returns the
    /// seed `0x345678` with no item loop. Nested `((), 1)` mixes
    /// `hash(()) = 0x345678` then `hash(1) = 1`:
    /// `x = 0x345678`; `x = intmask((1000003 * x) ^ 0x345678)`;
    /// `x = intmask((1000003 * x) ^ 1)`.
    #[test]
    fn convert_const_empty_and_nested_empty_tuple_keys() {
        let repr = empty_tuple_signed_ordered_dict_repr();
        let mut items = indexmap::IndexMap::new();
        items.insert(ConstValue::Tuple(vec![]), ConstValue::Int(11));
        let c = repr
            .convert_const(&ConstValue::Dict(Arc::new(items)))
            .expect("empty-tuple dict");
        let mut d = as_dict_ptr(&c).clone();
        let h = ll_keyhash(&d, &LowLevelValue::Void).expect("hash ()");
        assert_eq!(h, 0x345678, "gen_hash_function([]) returns the seed");
        let got = ll_dict_getitem(&mut d, &LowLevelValue::Void).expect("lookup ()");
        assert!(
            matches!(got, LowLevelValue::Signed(11)),
            "empty-tuple lookup, got {got:?}"
        );

        let (repr, tuple_repr) = nested_empty_tuple_int_signed_ordered_dict_repr();
        let mut items = indexmap::IndexMap::new();
        items.insert(
            ConstValue::Tuple(vec![ConstValue::Tuple(vec![]), ConstValue::Int(1)]),
            ConstValue::Int(22),
        );
        let c = repr
            .convert_const(&ConstValue::Dict(Arc::new(items)))
            .expect("nested tuple dict");
        let mut d = as_dict_ptr(&c).clone();
        let mut x: i64 = 0x345678;
        x = 1000003i64.wrapping_mul(x) ^ 0x345678;
        x = 1000003i64.wrapping_mul(x) ^ 1;
        let key_const = tuple_repr
            .convert_const(&ConstValue::Tuple(vec![
                ConstValue::Tuple(vec![]),
                ConstValue::Int(1),
            ]))
            .expect("nested key");
        let key = crate::translator::rtyper::rclass::constant_to_lowlevel_value(&key_const)
            .expect("nested key llvalue");
        let h = ll_keyhash(&d, &key).expect("hash ((), 1)");
        assert_eq!(h, x, "nested gen_hash_function mix");
        let got = ll_dict_getitem(&mut d, &key).expect("lookup ((), 1)");
        assert!(
            matches!(got, LowLevelValue::Signed(22)),
            "nested empty-tuple lookup, got {got:?}"
        );
    }

    /// `@jit.dont_look_inside` + `_dont_inline_ = True` on `_ll_dict_rescue`.
    #[test]
    fn ll_dict_rescue_is_dont_look_inside_and_dont_inline() {
        let reindex_fn = tagged_funcptr_const("ll_dict_reindex");
        let helper = build_ll_dict_rescue_helper_graph(
            "_ll_dict_rescue",
            sample_dict_ptr_lltype(),
            reindex_fn,
        )
        .expect("rescue helper");
        assert_eq!(
            helper.func._jit_look_inside_,
            Some(false),
            "_ll_dict_rescue is dont_look_inside"
        );
        assert!(
            helper.func._dont_inline_,
            "_ll_dict_rescue is _dont_inline_"
        );
        assert!(
            helper
                .graph
                .borrow()
                .func
                .as_ref()
                .is_some_and(|func| func._jit_look_inside_ == Some(false) && func._dont_inline_),
            "_ll_dict_rescue hints land on FunctionGraph.func too"
        );
    }

    /// `@jit.dont_look_inside` on `ll_dict_rehash_after_translation`.
    #[test]
    fn ll_dict_rehash_after_translation_is_dont_look_inside() {
        let repr = sample_ordered_dict_repr();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let helper = build_ll_dict_rehash_after_translation_helper_graph(
            "ll_dict_rehash_after_translation",
            repr.lowleveltype().clone(),
            entries_ptr,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            false,
        )
        .expect("rehash helper");
        assert_eq!(
            helper.func._jit_look_inside_,
            Some(false),
            "ll_dict_rehash_after_translation is dont_look_inside"
        );
    }

    fn getfield_field(op: &SpaceOperation) -> Option<ConstValue> {
        if op.opname != "getfield" {
            return None;
        }
        match op.args.get(1) {
            Some(Hlvalue::Constant(c)) => Some(c.value.clone()),
            _ => None,
        }
    }

    /// `ll_dict_rehash_after_translation`: `d.entries` is re-read in the
    /// loop body (not the entry block); `d.num_live_items` is read only
    /// after the hash loop, in the `new_size` while-test.
    #[test]
    fn ll_dict_rehash_after_translation_reads_entries_in_loop_live_after() {
        let repr = sample_ordered_dict_repr();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let helper = build_ll_dict_rehash_after_translation_helper_graph(
            "ll_dict_rehash_after_translation",
            repr.lowleveltype().clone(),
            entries_ptr,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            false,
        )
        .expect("rehash helper");
        crate::flowspace::model::checkgraph(&helper.graph.borrow());
        let start = helper.graph.borrow().startblock.clone();
        let start_fields: Vec<ConstValue> = start
            .borrow()
            .operations
            .iter()
            .filter_map(getfield_field)
            .collect();
        assert_eq!(
            start_fields,
            vec![ConstValue::byte_str("num_ever_used_items")],
            "range bound is read once in the entry block"
        );

        let cond = start
            .borrow()
            .exits
            .first()
            .and_then(|l| l.borrow().target.clone())
            .expect("loop cond");
        let body = bool_exit(&cond, true);
        let body_ops = body.borrow().operations.clone();
        assert_eq!(
            getfield_field(&body_ops[0]),
            Some(ConstValue::byte_str("entries")),
            "d.entries is re-read at the start of the loop body"
        );
        assert_eq!(body_ops[1].opname, "getinteriorfield");
        assert_eq!(body_ops[2].opname, "direct_call");
        assert!(
            body_ops
                .iter()
                .all(|op| getfield_field(op) != Some(ConstValue::byte_str("num_live_items"))),
            "num_live_items is not read in the hash loop"
        );

        let size_cond = bool_exit(&cond, false);
        let size_fields: Vec<ConstValue> = size_cond
            .borrow()
            .operations
            .iter()
            .filter_map(getfield_field)
            .collect();
        assert_eq!(
            size_fields,
            vec![ConstValue::byte_str("num_live_items")],
            "d.num_live_items is read in the new_size while-test after the loop"
        );
        assert!(
            start
                .borrow()
                .operations
                .iter()
                .all(|op| getfield_field(op) != Some(ConstValue::byte_str("entries"))),
            "d.entries is not hoisted into the entry block"
        );
    }

    /// `look_inside_iff(lambda d: jit.isvirtual(d))` split on
    /// `ll_dict_create_initial_index`: orig is unroll_safe, trampoline is
    /// dont_look_inside, wrapper branches on we_are_jitted then isvirtual.
    #[test]
    fn ll_dict_create_initial_index_look_inside_iff_split() {
        let insert_clean = build_ll_call_insert_clean_function_helper_graph(
            "ll_call_insert_clean_function",
            sample_dict_ptr_lltype(),
            dummy_funcptr_const(),
        )
        .expect("insert_clean");
        assert_eq!(insert_clean.func._jit_look_inside_, Some(false));

        let repr = sample_ordered_dict_repr();
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        let entries_array_ty = LowLevelType::Array(Box::new(repr.DICTENTRYARRAY.clone()));
        let remove_deleted = build_ll_dict_remove_deleted_items_helper_graph(
            "ll_dict_remove_deleted_items",
            repr.lowleveltype().clone(),
            entries_ptr,
            entries_array_ty,
            repr.DICTKEY.clone(),
            repr.DICTVALUE.clone(),
            dummy_funcptr_const(),
        )
        .expect("remove_deleted");
        assert_eq!(remove_deleted.func._jit_look_inside_, Some(false));

        let delete_by_entry = build_ll_dict_delete_by_entry_index_helper_graph(
            "ll_dict_delete_by_entry_index",
            sample_dict_ptr_lltype(),
            dummy_funcptr_const(),
        )
        .expect("delete_by_entry");
        assert!(delete_by_entry.func._jit_unroll_safe_);

        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dict_ptr = sample_dict_ptr_lltype();
        let wrapper = mint_look_inside_iff(
            &rtyper,
            "ll_dict_create_initial_index",
            &["d"],
            vec![dict_ptr.clone()],
            LowLevelType::Void,
            {
                let dict_ptr = dict_ptr;
                move |name| {
                    build_ll_dict_create_initial_index_helper_graph(
                        name,
                        dict_ptr.clone(),
                        dummy_funcptr_const(),
                        dummy_funcptr_const(),
                    )
                }
            },
            &[LookInsidePred::IsVirtual(0)],
        )
        .expect("wrapper");
        assert_eq!(wrapper.name, "ll_dict_create_initial_index_look_inside_iff");
        assert_eq!(
            wrapper
                .graph
                .as_ref()
                .expect("wrapper PyGraph")
                .func
                .annspecialcase,
            None,
            "look_inside_iff wrapper f does not get call_location"
        );

        let find = |name: &str| {
            ann.translator
                .graphs
                .borrow()
                .iter()
                .find(|g| g.borrow().name == name)
                .unwrap_or_else(|| panic!("missing graph {name}"))
                .clone()
        };
        let orig = find("ll_dict_create_initial_index");
        let orig_func = orig
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("orig GraphFunc");
        assert!(
            orig_func._jit_unroll_safe_,
            "look_inside_iff marks the original unroll_safe"
        );
        assert_eq!(
            orig_func.annspecialcase.as_deref(),
            Some("specialize:call_location")
        );

        let trampoline = find("ll_dict_create_initial_index_trampoline");
        let tramp_func = trampoline
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("trampoline GraphFunc");
        assert_eq!(
            tramp_func._jit_look_inside_,
            Some(false),
            "trampoline is dont_look_inside"
        );

        let wrapper_graph = find("ll_dict_create_initial_index_look_inside_iff");
        let inner = wrapper_graph.borrow();
        let startblock = inner.startblock.borrow();
        assert!(startblock.operations.is_empty());
        let Some(Hlvalue::Constant(c)) = &startblock.exitswitch else {
            panic!("exitswitch must be the we_are_jitted SpecTag constant");
        };
        assert_eq!(c.value, ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID));
        assert_eq!(startblock.exits.len(), 2);
        let true_first = &startblock.exits[0].borrow().exitcase;
        assert!(
            matches!(true_first, Some(Hlvalue::Constant(c)) if c.value == ConstValue::Bool(true)),
            "we_are_jitted true link first"
        );
    }

    /// `ll_dict_lookup` trampoline (`rlib/jit.py look_inside_iff`) keeps the
    /// Void `T` inputarg so `parse_oopspec` can resolve the oopspec name `T`.
    #[test]
    fn ll_dict_lookup_trampoline_carries_void_t_and_oopspec() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let repr = OrderedDictRepr::new(
            rtyper.clone(),
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr");
        repr.lookup_chain_helpers_on(&rtyper).expect("lookup chain");
        let trampoline = ann
            .translator
            .graphs
            .borrow()
            .iter()
            .find(|g| g.borrow().name == "ll_dict_lookup_trampoline")
            .unwrap_or_else(|| panic!("missing ll_dict_lookup_trampoline"))
            .clone();
        let inner = trampoline.borrow();
        assert_eq!(
            startblock_arg_names(&inner),
            ["d", "key", "hash", "store_flag", "T"]
        );
        assert_eq!(
            inner.func.as_ref().and_then(|f| f.oopspec.as_deref()),
            Some("ordereddict.lookup(d, key, hash, store_flag, T)"),
            "look_inside_iff moves oopspec onto the trampoline"
        );
    }
}
