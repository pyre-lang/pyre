//! RPython `rpython/rlib/rstring.py` — substring-search helpers as
//! translator helper-graph builders.
//!
//! `LLHelpers.ll_search` (`rtyper/lltypesystem/rstr.py`) calls
//! `rstring._search` after `hlstr` / `hlunicode`. `@specialize.argtype`
//! specialises each helper on the string kind: `STR` → `Char` /
//! `cast_char_to_int`, `UNICODE` → `UniChar` / `cast_unichar_to_int`.
//! One `GraphFunc` is cached per `(name, args, result)` so the two
//! kinds do not collide.
//!
//! `@jit.elidable` stamps `_elidable_function_` on `_search_elidable`
//! only. `@specialize` is not a JIT hint.
//!
//! `_search_normal` takes `value_can_be_none` for the caller's
//! `SomeString.can_be_None`. `LLHelpers.ll_search` asserts first, so
//! its `_search` / `_search_elidable` / `_search_normal` callees pass
//! `false` and `transform_dead_code` (`transform.py`) cuts the raise
//! link.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::codewriter::call::array_type_id_of_lltype;
use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::model::{
    BlockId, CallTarget, ConcreteType, ExitCase, ExitSwitch, FieldDescriptor,
    FunctionGraph as ModelGraph, Link as ModelLink, LinkArg, OpKind, ValueType,
};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    Array, LowLevelType, Ptr, PtrTarget, Struct,
};
use crate::translator::rtyper::lltypesystem::rstr::{
    chars_array_ptr_lltype_from_strptr, close_assert_ptr_not_none, sub_helper_funcptr_constant,
};
use crate::translator::rtyper::rint::signed_repr;
use crate::translator::rtyper::rlist::{
    ListLayout, ListRepr, build_ll_append_helper_graph, build_ll_arraycopy_helper_graph,
    build_ll_getitem_fast_helper_graph, build_ll_list_resize_ge_helper_graph,
    build_ll_newlist_helper_graph, build_ll_setitem_fast_helper_graph, items_array_ptr_lltype,
};
use crate::translator::rtyper::rmodel::{Repr, gc_flavor_const, lowlevel_type_const};
use crate::translator::rtyper::rtyper::{
    RPythonTyper, constant_with_lltype, exception_args, helper_pygraph_from_graph,
    variable_with_lltype, void_field_const,
};

/// `rstring.py` `SEARCH_COUNT`.
pub const SEARCH_COUNT: i64 = 0;
/// `rstring.py` `SEARCH_FIND`.
pub const SEARCH_FIND: i64 = 1;
/// `rstring.py` `SEARCH_RFIND`.
pub const SEARCH_RFIND: i64 = 2;

/// `rstring.py` `TWOWAY_MAX_SHIFT`.
pub const TWOWAY_MAX_SHIFT: i64 = 255;
/// `rstring.py` `TWOWAY_TABLE_SIZE`.
pub const TWOWAY_TABLE_SIZE: i64 = 64;
/// `rstring.py` `TWOWAY_TABLE_MASK`.
pub const TWOWAY_TABLE_MASK: i64 = TWOWAY_TABLE_SIZE - 1;

/// `rpython/rlib/rarithmetic.py` `LONG_BIT`, bound as `BLOOM_WIDTH`.
fn bloom_width() -> i64 {
    (crate::layout::target_word_size() * 8) as i64
}

/// Char-width specialisation of one `@specialize.argtype` helper.
#[derive(Clone)]
pub(crate) struct StringKind {
    pub ptr_lltype: LowLevelType,
    pub chars_array_ptr_lltype: LowLevelType,
    pub elem_lltype: LowLevelType,
    pub ord_op: &'static str,
    pub eq_op: &'static str,
    pub ne_op: &'static str,
    pub cast_int_to_elem: &'static str,
}

impl StringKind {
    pub(crate) fn from_ptr(ptr_lltype: LowLevelType) -> Result<Self, TyperError> {
        let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
        let elem_lltype = match &chars_array_ptr_lltype {
            LowLevelType::Ptr(chars_ptr) => match &chars_ptr.TO {
                PtrTarget::Array(arr) => arr.OF.clone(),
                other => {
                    return Err(TyperError::message(format!(
                        "rstring helper Ptr.TO must be Array, got {other:?}"
                    )));
                }
            },
            other => {
                return Err(TyperError::message(format!(
                    "rstring helper expects Ptr(Array(Char|UniChar)), got {other:?}"
                )));
            }
        };
        let (ord_op, eq_op, ne_op, cast_int_to_elem) = match elem_lltype {
            LowLevelType::Char => ("cast_char_to_int", "char_eq", "char_ne", "cast_int_to_char"),
            LowLevelType::UniChar => (
                "cast_unichar_to_int",
                "unichar_eq",
                "unichar_ne",
                "cast_int_to_unichar",
            ),
            ref other => {
                return Err(TyperError::message(format!(
                    "rstring helper unsupported chars element type {other:?}"
                )));
            }
        };
        Ok(Self {
            ptr_lltype,
            chars_array_ptr_lltype,
            elem_lltype,
            ord_op,
            eq_op,
            ne_op,
            cast_int_to_elem,
        })
    }

    pub(crate) fn is_unicode(&self) -> bool {
        self.elem_lltype == LowLevelType::UniChar
    }
}

/// `ListRepr` of Signed — `_twoway_preprocess` `table`
/// (`rstring.py`). The comprehension
/// `[not_found_shift for _i in range(TWOWAY_TABLE_SIZE)]` annotates as a
/// resized list, so the tuple carries `Ptr(GcStruct("list", length, items))`
/// rather than a bare `GcArray`.
fn table_list_ptr_lltype() -> LowLevelType {
    let item_repr: Arc<dyn Repr> = signed_repr();
    ListRepr::from_primitive_item_repr(item_repr)
        .lowleveltype()
        .clone()
}

fn tuple2_struct() -> Struct {
    Struct::gc(
        "tuple2",
        vec![
            ("item0".to_string(), LowLevelType::Signed),
            ("item1".to_string(), LowLevelType::Signed),
        ],
    )
}

fn tuple2_ptr_lltype() -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(tuple2_struct()),
    }))
}

fn tuple5_struct() -> Struct {
    Struct::gc(
        "tuple5",
        vec![
            ("item0".to_string(), LowLevelType::Signed),
            ("item1".to_string(), LowLevelType::Signed),
            ("item2".to_string(), LowLevelType::Signed),
            ("item3".to_string(), LowLevelType::Bool),
            ("item4".to_string(), table_list_ptr_lltype()),
        ],
    )
}

fn tuple5_ptr_lltype() -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(tuple5_struct()),
    }))
}

fn signed_const(n: i64) -> Hlvalue {
    constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed)
}

fn bool_const(b: bool) -> Hlvalue {
    constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool)
}

fn chars_field() -> Hlvalue {
    constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void)
}

fn v(name: &str, ty: LowLevelType) -> Variable {
    variable_with_lltype(name, ty)
}

fn hl(var: &Variable) -> Hlvalue {
    Hlvalue::Variable(var.clone())
}

fn emit(block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: Variable) {
    block.borrow_mut().operations.push(SpaceOperation::new(
        opname,
        args,
        Hlvalue::Variable(result),
    ));
}

fn emit_void(block: &BlockRef, opname: &str, args: Vec<Hlvalue>) {
    emit(block, opname, args, v("v", LowLevelType::Void));
}

fn emit_bin(
    block: &BlockRef,
    opname: &str,
    lhs: Hlvalue,
    rhs: Hlvalue,
    result_name: &str,
    result_ty: LowLevelType,
) -> Variable {
    let result = v(result_name, result_ty.clone());
    emit(block, opname, vec![lhs, rhs], result.clone());
    result
}

fn emit_un(
    block: &BlockRef,
    opname: &str,
    operand: Hlvalue,
    result_name: &str,
    result_ty: LowLevelType,
) -> Variable {
    let result = v(result_name, result_ty.clone());
    emit(block, opname, vec![operand], result.clone());
    result
}

fn branch(
    block: &BlockRef,
    cond: Variable,
    true_args: Vec<Hlvalue>,
    true_tgt: BlockRef,
    false_args: Vec<Hlvalue>,
    false_tgt: BlockRef,
) {
    block.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    block.closeblock(vec![
        Link::new(true_args, Some(true_tgt), Some(bool_const(true))).into_ref(),
        Link::new(false_args, Some(false_tgt), Some(bool_const(false))).into_ref(),
    ]);
}

fn goto(block: &BlockRef, args: Vec<Hlvalue>, tgt: BlockRef) {
    block.closeblock(vec![Link::new(args, Some(tgt), None).into_ref()]);
}

fn ret(block: &BlockRef, value: Hlvalue, returnblock: BlockRef) {
    goto(block, vec![value], returnblock);
}

fn blk(inputs: Vec<Variable>) -> BlockRef {
    Block::shared(inputs.into_iter().map(|var| hl(&var)).collect())
}

fn block_vars(block: &BlockRef) -> Vec<Variable> {
    block
        .borrow()
        .inputargs
        .iter()
        .filter_map(|h| match h {
            Hlvalue::Variable(var) => Some(var.clone()),
            _ => None,
        })
        .collect()
}

fn emit_getitem(kind: &StringKind, block: &BlockRef, s: &Variable, idx: Hlvalue) -> Variable {
    let chars = v("chars", kind.chars_array_ptr_lltype.clone());
    emit(
        block,
        "getsubstruct",
        vec![hl(s), chars_field()],
        chars.clone(),
    );
    let ch = v("ch", kind.elem_lltype.clone());
    emit(block, "getarrayitem", vec![hl(&chars), idx], ch.clone());
    ch
}

fn emit_len(kind: &StringKind, block: &BlockRef, s: &Variable) -> Variable {
    let chars = v("chars", kind.chars_array_ptr_lltype.clone());
    emit(
        block,
        "getsubstruct",
        vec![hl(s), chars_field()],
        chars.clone(),
    );
    let n = v("n", LowLevelType::Signed);
    emit(block, "getarraysize", vec![hl(&chars)], n.clone());
    n
}

fn emit_ord(kind: &StringKind, block: &BlockRef, ch: Variable) -> Variable {
    emit_un(block, kind.ord_op, hl(&ch), "ord", LowLevelType::Signed)
}

fn emit_nul(kind: &StringKind, block: &BlockRef) -> Variable {
    emit_un(
        block,
        kind.cast_int_to_elem,
        signed_const(0),
        "nul",
        kind.elem_lltype.clone(),
    )
}

fn emit_table_get(block: &BlockRef, getitem: &Hlvalue, table: &Variable, idx: Hlvalue) -> Variable {
    emit_direct_call(
        block,
        getitem.clone(),
        vec![hl(table), idx],
        "titem",
        LowLevelType::Signed,
    )
}

fn emit_malloc_struct(
    block: &BlockRef,
    body: Struct,
    result_lltype: LowLevelType,
    name: &str,
) -> Result<Variable, TyperError> {
    let result = v(name, result_lltype);
    emit(
        block,
        "malloc",
        vec![
            lowlevel_type_const(LowLevelType::Struct(Box::new(body))),
            gc_flavor_const()?,
        ],
        result.clone(),
    );
    Ok(result)
}

fn emit_setfield(block: &BlockRef, obj: &Variable, field: &str, value: Hlvalue) {
    emit_void(
        block,
        "setfield",
        vec![hl(obj), void_field_const(field), value],
    );
}

fn emit_getfield(
    block: &BlockRef,
    obj: &Variable,
    field: &str,
    result_name: &str,
    result_ty: LowLevelType,
) -> Variable {
    let result = v(result_name, result_ty.clone());
    emit(
        block,
        "getfield",
        vec![hl(obj), void_field_const(field)],
        result.clone(),
    );
    result
}

fn emit_tuple2(block: &BlockRef, item0: Hlvalue, item1: Hlvalue) -> Result<Variable, TyperError> {
    let tup = emit_malloc_struct(block, tuple2_struct(), tuple2_ptr_lltype(), "tup")?;
    emit_setfield(block, &tup, "item0", item0);
    emit_setfield(block, &tup, "item1", item1);
    Ok(tup)
}

fn emit_tuple5(
    block: &BlockRef,
    cut: Hlvalue,
    period: Hlvalue,
    gap: Hlvalue,
    is_periodic: Hlvalue,
    table: Hlvalue,
) -> Result<Variable, TyperError> {
    let tup = emit_malloc_struct(block, tuple5_struct(), tuple5_ptr_lltype(), "tup")?;
    emit_setfield(block, &tup, "item0", cut);
    emit_setfield(block, &tup, "item1", period);
    emit_setfield(block, &tup, "item2", gap);
    emit_setfield(block, &tup, "item3", is_periodic);
    emit_setfield(block, &tup, "item4", table);
    Ok(tup)
}

fn finish_graph(
    mut graph: FunctionGraph,
    name: &str,
    argnames: Vec<String>,
    elidable: bool,
) -> PyGraph {
    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    func._elidable_function_ = elidable;
    graph.func = Some(func.clone());
    helper_pygraph_from_graph(graph, argnames, func)
}

fn cached_helper_funcptr(
    rtyper: &RPythonTyper,
    name: &str,
    args: Vec<LowLevelType>,
    result: LowLevelType,
    builder: impl FnOnce(&RPythonTyper) -> Result<PyGraph, TyperError>,
) -> Result<Hlvalue, TyperError> {
    let helper = rtyper.lowlevel_helper_function_with_builder(
        name.to_string(),
        args,
        result,
        move |rtyper, _args, _result| builder(rtyper),
    )?;
    Ok(Hlvalue::Constant(sub_helper_funcptr_constant(
        rtyper, &helper,
    )?))
}

/// `ListRepr` helpers for the two-way bad-character table
/// (`rlist.py` `ll_newlist` / `ll_append` / `ll_setitem_nonneg` with a
/// known-nonneg index → `ll_setitem_fast` / `ll_getitem_nonneg` →
/// `ll_getitem_fast`). Sub-helpers are minted in the same order as
/// `ListRepr.rtype_method_append`.
struct TwowayTableOps {
    newlist: Hlvalue,
    append: Hlvalue,
    setitem: Hlvalue,
    getitem: Hlvalue,
}

fn twoway_table_ops(rtyper: &RPythonTyper) -> Result<TwowayTableOps, TyperError> {
    let list_lltype = table_list_ptr_lltype();
    let item_lltype = LowLevelType::Signed;
    let items_ptr = items_array_ptr_lltype(&item_lltype);

    let newlist = {
        let ptr = list_lltype.clone();
        let item = item_lltype.clone();
        cached_helper_funcptr(
            rtyper,
            "ll_newlist",
            vec![LowLevelType::Signed],
            list_lltype.clone(),
            move |_rtyper| {
                build_ll_newlist_helper_graph(
                    "ll_newlist",
                    ListLayout::Resized,
                    ptr.clone(),
                    item.clone(),
                )
            },
        )?
    };

    let arraycopy = {
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            "ll_arraycopy".to_string(),
            vec![items_ptr.clone(), items_ptr.clone(), LowLevelType::Signed],
            LowLevelType::Void,
            move |_rtyper, _args, _result| {
                build_ll_arraycopy_helper_graph("ll_arraycopy", item.clone())
            },
        )?
    };
    let setitem_fast = {
        let ptr = list_lltype.clone();
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            "ll_setitem_fast".to_string(),
            vec![
                list_lltype.clone(),
                LowLevelType::Signed,
                item_lltype.clone(),
            ],
            LowLevelType::Void,
            move |_rtyper, _args, _result| {
                build_ll_setitem_fast_helper_graph("ll_setitem_fast", ptr.clone(), item.clone())
            },
        )?
    };
    let resize_ge = {
        let ptr = list_lltype.clone();
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            "_ll_list_resize_ge".to_string(),
            vec![list_lltype.clone(), LowLevelType::Signed],
            LowLevelType::Void,
            move |rtyper, _args, _result| {
                build_ll_list_resize_ge_helper_graph(
                    rtyper,
                    "_ll_list_resize_ge",
                    ptr.clone(),
                    item.clone(),
                    &arraycopy,
                )
            },
        )?
    };
    let setitem = Hlvalue::Constant(sub_helper_funcptr_constant(rtyper, &setitem_fast)?);
    let append = {
        let ptr = list_lltype.clone();
        let item = item_lltype.clone();
        cached_helper_funcptr(
            rtyper,
            "ll_append",
            vec![list_lltype.clone(), item_lltype.clone()],
            LowLevelType::Void,
            move |rtyper| {
                build_ll_append_helper_graph(
                    rtyper,
                    "ll_append",
                    ptr.clone(),
                    item.clone(),
                    &resize_ge,
                    &setitem_fast,
                )
            },
        )?
    };
    let getitem = {
        let ptr = list_lltype.clone();
        let item = item_lltype.clone();
        cached_helper_funcptr(
            rtyper,
            "ll_getitem_fast",
            vec![list_lltype, LowLevelType::Signed],
            item_lltype,
            move |_rtyper| {
                build_ll_getitem_fast_helper_graph("ll_getitem_fast", ptr.clone(), item.clone())
            },
        )?
    };
    Ok(TwowayTableOps {
        newlist,
        append,
        setitem,
        getitem,
    })
}

fn emit_direct_call(
    block: &BlockRef,
    callee: Hlvalue,
    args: Vec<Hlvalue>,
    result_name: &str,
    result_ty: LowLevelType,
) -> Variable {
    let result = v(result_name, result_ty.clone());
    let mut call_args = vec![callee];
    call_args.extend(args);
    emit(block, "direct_call", call_args, result.clone());
    result
}

fn search_args(kind: &StringKind) -> Vec<LowLevelType> {
    vec![
        kind.ptr_lltype.clone(),
        kind.ptr_lltype.clone(),
        LowLevelType::Signed,
        LowLevelType::Signed,
        LowLevelType::Signed,
    ]
}

fn default_find_args(kind: &StringKind) -> Vec<LowLevelType> {
    vec![
        kind.ptr_lltype.clone(),
        LowLevelType::Signed,
        LowLevelType::Signed,
        kind.ptr_lltype.clone(),
        LowLevelType::Signed,
        LowLevelType::Signed,
    ]
}

fn two_way_args(kind: &StringKind) -> Vec<LowLevelType> {
    vec![
        kind.ptr_lltype.clone(),
        LowLevelType::Signed,
        LowLevelType::Signed,
        kind.ptr_lltype.clone(),
        LowLevelType::Signed,
        LowLevelType::Signed,
        LowLevelType::Signed,
        LowLevelType::Signed,
        LowLevelType::Bool,
        table_list_ptr_lltype(),
    ]
}

fn bloom_args(kind: &StringKind) -> Vec<LowLevelType> {
    vec![LowLevelType::Signed, kind.elem_lltype.clone()]
}

fn funcptr_bloom_add(rtyper: &RPythonTyper, kind: &StringKind) -> Result<Hlvalue, TyperError> {
    let ptr = kind.ptr_lltype.clone();
    cached_helper_funcptr(
        rtyper,
        "bloom_add",
        bloom_args(kind),
        LowLevelType::Signed,
        move |rtyper| build_bloom_add_helper_graph(rtyper, "bloom_add", ptr),
    )
}

fn funcptr_bloom(rtyper: &RPythonTyper, kind: &StringKind) -> Result<Hlvalue, TyperError> {
    let ptr = kind.ptr_lltype.clone();
    cached_helper_funcptr(
        rtyper,
        "bloom",
        bloom_args(kind),
        LowLevelType::Signed,
        move |rtyper| build_bloom_helper_graph(rtyper, "bloom", ptr),
    )
}

fn funcptr_helper(
    rtyper: &RPythonTyper,
    name: &'static str,
    args: Vec<LowLevelType>,
    result: LowLevelType,
    ptr: LowLevelType,
    builder: fn(&RPythonTyper, &str, LowLevelType) -> Result<PyGraph, TyperError>,
) -> Result<Hlvalue, TyperError> {
    cached_helper_funcptr(rtyper, name, args, result, move |rtyper| {
        builder(rtyper, name, ptr)
    })
}

fn funcptr_twoway_preprocess(
    rtyper: &RPythonTyper,
    kind: &StringKind,
) -> Result<Hlvalue, TyperError> {
    funcptr_helper(
        rtyper,
        "_twoway_preprocess",
        vec![kind.ptr_lltype.clone(), LowLevelType::Signed],
        tuple5_ptr_lltype(),
        kind.ptr_lltype.clone(),
        build_twoway_preprocess_helper_graph,
    )
}

fn funcptr_two_way(rtyper: &RPythonTyper, kind: &StringKind) -> Result<Hlvalue, TyperError> {
    funcptr_helper(
        rtyper,
        "_two_way",
        two_way_args(kind),
        LowLevelType::Signed,
        kind.ptr_lltype.clone(),
        build_two_way_helper_graph,
    )
}

fn funcptr_two_way_count(rtyper: &RPythonTyper, kind: &StringKind) -> Result<Hlvalue, TyperError> {
    funcptr_helper(
        rtyper,
        "_two_way_count",
        two_way_args(kind),
        LowLevelType::Signed,
        kind.ptr_lltype.clone(),
        build_two_way_count_helper_graph,
    )
}

fn funcptr_default_find(rtyper: &RPythonTyper, kind: &StringKind) -> Result<Hlvalue, TyperError> {
    funcptr_helper(
        rtyper,
        "_default_find",
        default_find_args(kind),
        LowLevelType::Signed,
        kind.ptr_lltype.clone(),
        build_default_find_helper_graph,
    )
}

fn funcptr_adaptive_find(rtyper: &RPythonTyper, kind: &StringKind) -> Result<Hlvalue, TyperError> {
    funcptr_helper(
        rtyper,
        "_adaptive_find",
        default_find_args(kind),
        LowLevelType::Signed,
        kind.ptr_lltype.clone(),
        build_adaptive_find_helper_graph,
    )
}

fn funcptr_search_normal(
    rtyper: &RPythonTyper,
    kind: &StringKind,
    value_can_be_none: bool,
) -> Result<Hlvalue, TyperError> {
    let ptr = kind.ptr_lltype.clone();
    cached_helper_funcptr(
        rtyper,
        "_search_normal",
        search_args(kind),
        LowLevelType::Signed,
        move |rtyper| {
            build_search_normal_helper_graph(rtyper, "_search_normal", ptr, value_can_be_none)
        },
    )
}

fn funcptr_search_elidable(
    rtyper: &RPythonTyper,
    kind: &StringKind,
    value_can_be_none: bool,
) -> Result<Hlvalue, TyperError> {
    let ptr = kind.ptr_lltype.clone();
    cached_helper_funcptr(
        rtyper,
        "_search_elidable",
        search_args(kind),
        LowLevelType::Signed,
        move |rtyper| {
            build_search_elidable_helper_graph(rtyper, "_search_elidable", ptr, value_can_be_none)
        },
    )
}

fn unpack_tuple5(
    block: &BlockRef,
    tup: &Variable,
) -> (Variable, Variable, Variable, Variable, Variable) {
    (
        emit_getfield(block, tup, "item0", "cut", LowLevelType::Signed),
        emit_getfield(block, tup, "item1", "period", LowLevelType::Signed),
        emit_getfield(block, tup, "item2", "gap", LowLevelType::Signed),
        emit_getfield(block, tup, "item3", "is_periodic", LowLevelType::Bool),
        emit_getfield(block, tup, "item4", "table", table_list_ptr_lltype()),
    )
}

/// `rstring.py` `bloom_add`.
pub(crate) fn build_bloom_add_helper_graph(
    _rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype)?;
    let mask = v("mask", LowLevelType::Signed);
    let c = v("c", kind.elem_lltype.clone());
    let startblock = blk(vec![mask.clone(), c.clone()]);
    let return_var = v("result", LowLevelType::Signed);
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));
    let o = emit_ord(&kind, &startblock, c);
    let masked = emit_bin(
        &startblock,
        "int_and",
        hl(&o),
        signed_const(bloom_width() - 1),
        "m",
        LowLevelType::Signed,
    );
    let bit = emit_bin(
        &startblock,
        "int_lshift",
        signed_const(1),
        hl(&masked),
        "bit",
        LowLevelType::Signed,
    );
    let result = emit_bin(
        &startblock,
        "int_or",
        hl(&mask),
        hl(&bit),
        "res",
        LowLevelType::Signed,
    );
    ret(&startblock, hl(&result), graph.returnblock.clone());
    Ok(finish_graph(
        graph,
        name,
        vec!["mask".into(), "c".into()],
        false,
    ))
}

/// `rstring.py` `bloom`.
pub(crate) fn build_bloom_helper_graph(
    _rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype)?;
    let mask = v("mask", LowLevelType::Signed);
    let c = v("c", kind.elem_lltype.clone());
    let startblock = blk(vec![mask.clone(), c.clone()]);
    let return_var = v("result", LowLevelType::Signed);
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));
    let o = emit_ord(&kind, &startblock, c);
    let masked = emit_bin(
        &startblock,
        "int_and",
        hl(&o),
        signed_const(bloom_width() - 1),
        "m",
        LowLevelType::Signed,
    );
    let bit = emit_bin(
        &startblock,
        "int_lshift",
        signed_const(1),
        hl(&masked),
        "bit",
        LowLevelType::Signed,
    );
    let result = emit_bin(
        &startblock,
        "int_and",
        hl(&mask),
        hl(&bit),
        "res",
        LowLevelType::Signed,
    );
    ret(&startblock, hl(&result), graph.returnblock.clone());
    Ok(finish_graph(
        graph,
        name,
        vec!["mask".into(), "c".into()],
        false,
    ))
}

/// `rstring.py` `_lex_search`. Returns `(max_suffix, period)` as `tuple2`.
pub(crate) fn build_lex_search_helper_graph(
    _rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype)?;
    let needle = v("needle", kind.ptr_lltype.clone());
    let len_needle = v("len_needle", LowLevelType::Signed);
    let invert_alphabet = v("invert_alphabet", LowLevelType::Bool);
    let startblock = blk(vec![
        needle.clone(),
        len_needle.clone(),
        invert_alphabet.clone(),
    ]);
    let return_var = v("result", tuple2_ptr_lltype());
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));

    let n_c = v("needle", kind.ptr_lltype.clone());
    let ln_c = v("len_needle", LowLevelType::Signed);
    let inv_c = v("invert_alphabet", LowLevelType::Bool);
    let max_c = v("max_suffix", LowLevelType::Signed);
    let cand_c = v("candidate", LowLevelType::Signed);
    let k_c = v("k", LowLevelType::Signed);
    let per_c = v("period", LowLevelType::Signed);
    let cond = blk(vec![
        n_c.clone(),
        ln_c.clone(),
        inv_c.clone(),
        max_c.clone(),
        cand_c.clone(),
        k_c.clone(),
        per_c.clone(),
    ]);

    let n_b = v("needle", kind.ptr_lltype.clone());
    let ln_b = v("len_needle", LowLevelType::Signed);
    let inv_b = v("invert_alphabet", LowLevelType::Bool);
    let max_b = v("max_suffix", LowLevelType::Signed);
    let cand_b = v("candidate", LowLevelType::Signed);
    let k_b = v("k", LowLevelType::Signed);
    let per_b = v("period", LowLevelType::Signed);
    let body = blk(vec![
        n_b.clone(),
        ln_b.clone(),
        inv_b.clone(),
        max_b.clone(),
        cand_b.clone(),
        k_b.clone(),
        per_b.clone(),
    ]);

    let n_d = v("needle", kind.ptr_lltype.clone());
    let ln_d = v("len_needle", LowLevelType::Signed);
    let inv_d = v("invert_alphabet", LowLevelType::Bool);
    let max_d = v("max_suffix", LowLevelType::Signed);
    let cand_d = v("candidate", LowLevelType::Signed);
    let k_d = v("k", LowLevelType::Signed);
    let per_d = v("period", LowLevelType::Signed);
    let a_d = v("a", LowLevelType::Signed);
    let b_d = v("b", LowLevelType::Signed);
    let lt_d = v("lt", LowLevelType::Bool);
    let dispatch = blk(vec![
        n_d.clone(),
        ln_d.clone(),
        inv_d.clone(),
        max_d.clone(),
        cand_d.clone(),
        k_d.clone(),
        per_d.clone(),
        a_d.clone(),
        b_d.clone(),
        lt_d.clone(),
    ]);

    let n_eq = v("needle", kind.ptr_lltype.clone());
    let ln_eq = v("len_needle", LowLevelType::Signed);
    let inv_eq = v("invert_alphabet", LowLevelType::Bool);
    let max_eq = v("max_suffix", LowLevelType::Signed);
    let cand_eq = v("candidate", LowLevelType::Signed);
    let k_eq = v("k", LowLevelType::Signed);
    let per_eq = v("period", LowLevelType::Signed);
    let a_eq = v("a", LowLevelType::Signed);
    let b_eq = v("b", LowLevelType::Signed);
    let eq_block = blk(vec![
        n_eq.clone(),
        ln_eq.clone(),
        inv_eq.clone(),
        max_eq.clone(),
        cand_eq.clone(),
        k_eq.clone(),
        per_eq.clone(),
        a_eq.clone(),
        b_eq.clone(),
    ]);

    let n_done = v("needle", kind.ptr_lltype.clone());
    let max_done = v("max_suffix", LowLevelType::Signed);
    let per_done = v("period", LowLevelType::Signed);
    let done = blk(vec![n_done.clone(), max_done.clone(), per_done.clone()]);

    let carry = |n: &Variable,
                 ln: &Variable,
                 inv: &Variable,
                 maxs: Hlvalue,
                 cand: Hlvalue,
                 k: Hlvalue,
                 per: Hlvalue| { vec![hl(n), hl(ln), hl(inv), maxs, cand, k, per] };

    goto(
        &startblock,
        carry(
            &needle,
            &len_needle,
            &invert_alphabet,
            signed_const(0),
            signed_const(1),
            signed_const(0),
            signed_const(1),
        ),
        cond.clone(),
    );

    let sum = emit_bin(
        &cond,
        "int_add",
        hl(&cand_c),
        hl(&k_c),
        "sum",
        LowLevelType::Signed,
    );
    let more = emit_bin(
        &cond,
        "int_lt",
        hl(&sum),
        hl(&ln_c),
        "more",
        LowLevelType::Bool,
    );
    branch(
        &cond,
        more,
        carry(
            &n_c,
            &ln_c,
            &inv_c,
            hl(&max_c),
            hl(&cand_c),
            hl(&k_c),
            hl(&per_c),
        ),
        body.clone(),
        vec![hl(&n_c), hl(&max_c), hl(&per_c)],
        done.clone(),
    );

    let idx_a = emit_bin(
        &body,
        "int_add",
        hl(&cand_b),
        hl(&k_b),
        "idx_a",
        LowLevelType::Signed,
    );
    let idx_b = emit_bin(
        &body,
        "int_add",
        hl(&max_b),
        hl(&k_b),
        "idx_b",
        LowLevelType::Signed,
    );
    let ch_a = emit_getitem(&kind, &body, &n_b, hl(&idx_a));
    let ch_b = emit_getitem(&kind, &body, &n_b, hl(&idx_b));
    let a = emit_ord(&kind, &body, ch_a);
    let b = emit_ord(&kind, &body, ch_b);
    let inv_block = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("invert_alphabet", LowLevelType::Bool),
        v("max_suffix", LowLevelType::Signed),
        v("candidate", LowLevelType::Signed),
        v("k", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("a", LowLevelType::Signed),
        v("b", LowLevelType::Signed),
    ]);
    let fwd_block = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("invert_alphabet", LowLevelType::Bool),
        v("max_suffix", LowLevelType::Signed),
        v("candidate", LowLevelType::Signed),
        v("k", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("a", LowLevelType::Signed),
        v("b", LowLevelType::Signed),
    ]);
    let inv_ins: Vec<Variable> = inv_block
        .borrow()
        .inputargs
        .iter()
        .filter_map(|h| match h {
            Hlvalue::Variable(var) => Some(var.clone()),
            _ => None,
        })
        .collect();
    let fwd_ins: Vec<Variable> = fwd_block
        .borrow()
        .inputargs
        .iter()
        .filter_map(|h| match h {
            Hlvalue::Variable(var) => Some(var.clone()),
            _ => None,
        })
        .collect();
    let pack9 = |n: &Variable,
                 ln: &Variable,
                 inv: &Variable,
                 maxs: &Variable,
                 cand: &Variable,
                 k: &Variable,
                 per: &Variable,
                 a: &Variable,
                 b: &Variable| {
        vec![
            hl(n),
            hl(ln),
            hl(inv),
            hl(maxs),
            hl(cand),
            hl(k),
            hl(per),
            hl(a),
            hl(b),
        ]
    };
    branch(
        &body,
        inv_b.clone(),
        pack9(&n_b, &ln_b, &inv_b, &max_b, &cand_b, &k_b, &per_b, &a, &b),
        inv_block.clone(),
        pack9(&n_b, &ln_b, &inv_b, &max_b, &cand_b, &k_b, &per_b, &a, &b),
        fwd_block.clone(),
    );

    let lt_inv = emit_bin(
        &inv_block,
        "int_lt",
        hl(&inv_ins[8]),
        hl(&inv_ins[7]),
        "lt",
        LowLevelType::Bool,
    );
    goto(
        &inv_block,
        vec![
            hl(&inv_ins[0]),
            hl(&inv_ins[1]),
            hl(&inv_ins[2]),
            hl(&inv_ins[3]),
            hl(&inv_ins[4]),
            hl(&inv_ins[5]),
            hl(&inv_ins[6]),
            hl(&inv_ins[7]),
            hl(&inv_ins[8]),
            hl(&lt_inv),
        ],
        dispatch.clone(),
    );
    let lt_fwd = emit_bin(
        &fwd_block,
        "int_lt",
        hl(&fwd_ins[7]),
        hl(&fwd_ins[8]),
        "lt",
        LowLevelType::Bool,
    );
    goto(
        &fwd_block,
        vec![
            hl(&fwd_ins[0]),
            hl(&fwd_ins[1]),
            hl(&fwd_ins[2]),
            hl(&fwd_ins[3]),
            hl(&fwd_ins[4]),
            hl(&fwd_ins[5]),
            hl(&fwd_ins[6]),
            hl(&fwd_ins[7]),
            hl(&fwd_ins[8]),
            hl(&lt_fwd),
        ],
        dispatch.clone(),
    );

    let fell = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("invert_alphabet", LowLevelType::Bool),
        v("max_suffix", LowLevelType::Signed),
        v("candidate", LowLevelType::Signed),
        v("k", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
    ]);
    let fell_ins: Vec<Variable> = fell
        .borrow()
        .inputargs
        .iter()
        .filter_map(|h| match h {
            Hlvalue::Variable(var) => Some(var.clone()),
            _ => None,
        })
        .collect();
    branch(
        &dispatch,
        lt_d,
        carry(
            &n_d,
            &ln_d,
            &inv_d,
            hl(&max_d),
            hl(&cand_d),
            hl(&k_d),
            hl(&per_d),
        ),
        fell.clone(),
        pack9(
            &n_d, &ln_d, &inv_d, &max_d, &cand_d, &k_d, &per_d, &a_d, &b_d,
        ),
        eq_block.clone(),
    );

    let k1 = emit_bin(
        &fell,
        "int_add",
        hl(&fell_ins[5]),
        signed_const(1),
        "k1",
        LowLevelType::Signed,
    );
    let cand2 = emit_bin(
        &fell,
        "int_add",
        hl(&fell_ins[4]),
        hl(&k1),
        "cand",
        LowLevelType::Signed,
    );
    let period2 = emit_bin(
        &fell,
        "int_sub",
        hl(&cand2),
        hl(&fell_ins[3]),
        "period",
        LowLevelType::Signed,
    );
    goto(
        &fell,
        carry(
            &fell_ins[0],
            &fell_ins[1],
            &fell_ins[2],
            hl(&fell_ins[3]),
            hl(&cand2),
            signed_const(0),
            hl(&period2),
        ),
        cond.clone(),
    );

    let eq = emit_bin(
        &eq_block,
        "int_eq",
        hl(&a_eq),
        hl(&b_eq),
        "eq",
        LowLevelType::Bool,
    );
    let eq_body = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("invert_alphabet", LowLevelType::Bool),
        v("max_suffix", LowLevelType::Signed),
        v("candidate", LowLevelType::Signed),
        v("k", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
    ]);
    let better = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("invert_alphabet", LowLevelType::Bool),
        v("max_suffix", LowLevelType::Signed),
        v("candidate", LowLevelType::Signed),
        v("k", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
    ]);
    let eq_ins: Vec<Variable> = eq_body
        .borrow()
        .inputargs
        .iter()
        .filter_map(|h| match h {
            Hlvalue::Variable(var) => Some(var.clone()),
            _ => None,
        })
        .collect();
    let better_ins: Vec<Variable> = better
        .borrow()
        .inputargs
        .iter()
        .filter_map(|h| match h {
            Hlvalue::Variable(var) => Some(var.clone()),
            _ => None,
        })
        .collect();
    branch(
        &eq_block,
        eq,
        carry(
            &n_eq,
            &ln_eq,
            &inv_eq,
            hl(&max_eq),
            hl(&cand_eq),
            hl(&k_eq),
            hl(&per_eq),
        ),
        eq_body.clone(),
        carry(
            &n_eq,
            &ln_eq,
            &inv_eq,
            hl(&max_eq),
            hl(&cand_eq),
            hl(&k_eq),
            hl(&per_eq),
        ),
        better.clone(),
    );

    let k_plus = emit_bin(
        &eq_body,
        "int_add",
        hl(&eq_ins[5]),
        signed_const(1),
        "k1",
        LowLevelType::Signed,
    );
    let k_ne_period = emit_bin(
        &eq_body,
        "int_ne",
        hl(&k_plus),
        hl(&eq_ins[6]),
        "ne",
        LowLevelType::Bool,
    );
    let inc_k = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("invert_alphabet", LowLevelType::Bool),
        v("max_suffix", LowLevelType::Signed),
        v("candidate", LowLevelType::Signed),
        v("k", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
    ]);
    let add_period = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("invert_alphabet", LowLevelType::Bool),
        v("max_suffix", LowLevelType::Signed),
        v("candidate", LowLevelType::Signed),
        v("k", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
    ]);
    let inc_ins: Vec<Variable> = inc_k
        .borrow()
        .inputargs
        .iter()
        .filter_map(|h| match h {
            Hlvalue::Variable(var) => Some(var.clone()),
            _ => None,
        })
        .collect();
    let addp_ins: Vec<Variable> = add_period
        .borrow()
        .inputargs
        .iter()
        .filter_map(|h| match h {
            Hlvalue::Variable(var) => Some(var.clone()),
            _ => None,
        })
        .collect();
    branch(
        &eq_body,
        k_ne_period,
        carry(
            &eq_ins[0],
            &eq_ins[1],
            &eq_ins[2],
            hl(&eq_ins[3]),
            hl(&eq_ins[4]),
            hl(&k_plus),
            hl(&eq_ins[6]),
        ),
        inc_k.clone(),
        carry(
            &eq_ins[0],
            &eq_ins[1],
            &eq_ins[2],
            hl(&eq_ins[3]),
            hl(&eq_ins[4]),
            hl(&eq_ins[5]),
            hl(&eq_ins[6]),
        ),
        add_period.clone(),
    );
    goto(
        &inc_k,
        carry(
            &inc_ins[0],
            &inc_ins[1],
            &inc_ins[2],
            hl(&inc_ins[3]),
            hl(&inc_ins[4]),
            hl(&inc_ins[5]),
            hl(&inc_ins[6]),
        ),
        cond.clone(),
    );
    let cand_p = emit_bin(
        &add_period,
        "int_add",
        hl(&addp_ins[4]),
        hl(&addp_ins[6]),
        "cand",
        LowLevelType::Signed,
    );
    goto(
        &add_period,
        carry(
            &addp_ins[0],
            &addp_ins[1],
            &addp_ins[2],
            hl(&addp_ins[3]),
            hl(&cand_p),
            signed_const(0),
            hl(&addp_ins[6]),
        ),
        cond.clone(),
    );

    let cand_inc = emit_bin(
        &better,
        "int_add",
        hl(&better_ins[4]),
        signed_const(1),
        "cand",
        LowLevelType::Signed,
    );
    goto(
        &better,
        carry(
            &better_ins[0],
            &better_ins[1],
            &better_ins[2],
            hl(&better_ins[4]),
            hl(&cand_inc),
            signed_const(0),
            signed_const(1),
        ),
        cond.clone(),
    );

    let tup = emit_tuple2(&done, hl(&max_done), hl(&per_done))?;
    ret(&done, hl(&tup), graph.returnblock.clone());
    Ok(finish_graph(
        graph,
        name,
        vec![
            "needle".into(),
            "len_needle".into(),
            "invert_alphabet".into(),
        ],
        false,
    ))
}

/// `rstring.py` `_factorize`. Returns `(cut, period)` as `tuple2`.
pub(crate) fn build_factorize_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype.clone())?;
    let needle = v("needle", kind.ptr_lltype.clone());
    let len_needle = v("len_needle", LowLevelType::Signed);
    let startblock = blk(vec![needle.clone(), len_needle.clone()]);
    let return_var = v("result", tuple2_ptr_lltype());
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));
    let lex = {
        let ptr = ptr_lltype.clone();
        cached_helper_funcptr(
            rtyper,
            "_lex_search",
            vec![
                kind.ptr_lltype.clone(),
                LowLevelType::Signed,
                LowLevelType::Bool,
            ],
            tuple2_ptr_lltype(),
            move |rtyper| build_lex_search_helper_graph(rtyper, "_lex_search", ptr),
        )?
    };
    let t1 = emit_direct_call(
        &startblock,
        lex.clone(),
        vec![hl(&needle), hl(&len_needle), bool_const(false)],
        "t1",
        tuple2_ptr_lltype(),
    );
    let cut1 = emit_getfield(&startblock, &t1, "item0", "cut1", LowLevelType::Signed);
    let period1 = emit_getfield(&startblock, &t1, "item1", "period1", LowLevelType::Signed);
    let t2 = emit_direct_call(
        &startblock,
        lex,
        vec![hl(&needle), hl(&len_needle), bool_const(true)],
        "t2",
        tuple2_ptr_lltype(),
    );
    let cut2 = emit_getfield(&startblock, &t2, "item0", "cut2", LowLevelType::Signed);
    let period2 = emit_getfield(&startblock, &t2, "item1", "period2", LowLevelType::Signed);
    let gt = emit_bin(
        &startblock,
        "int_gt",
        hl(&cut1),
        hl(&cut2),
        "gt",
        LowLevelType::Bool,
    );
    let take1 = blk(vec![
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
    ]);
    let take2 = blk(vec![
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
    ]);
    let take1_cut = match &take1.borrow().inputargs[0] {
        Hlvalue::Variable(var) => var.clone(),
        _ => unreachable!(),
    };
    let take1_per = match &take1.borrow().inputargs[1] {
        Hlvalue::Variable(var) => var.clone(),
        _ => unreachable!(),
    };
    let take2_cut = match &take2.borrow().inputargs[0] {
        Hlvalue::Variable(var) => var.clone(),
        _ => unreachable!(),
    };
    let take2_per = match &take2.borrow().inputargs[1] {
        Hlvalue::Variable(var) => var.clone(),
        _ => unreachable!(),
    };
    branch(
        &startblock,
        gt,
        vec![hl(&cut1), hl(&period1)],
        take1.clone(),
        vec![hl(&cut2), hl(&period2)],
        take2.clone(),
    );
    let tup1 = emit_tuple2(&take1, hl(&take1_cut), hl(&take1_per))?;
    ret(&take1, hl(&tup1), graph.returnblock.clone());
    let tup2 = emit_tuple2(&take2, hl(&take2_cut), hl(&take2_per))?;
    ret(&take2, hl(&tup2), graph.returnblock.clone());
    Ok(finish_graph(
        graph,
        name,
        vec!["needle".into(), "len_needle".into()],
        false,
    ))
}

/// `rstring.py` `_twoway_preprocess`. Returns
/// `(cut, period, gap, is_periodic, table)` as `tuple5`.
///
/// The table is a resized `ListRepr` of Signed. The comprehension
/// `[not_found_shift for _i in range(TWOWAY_TABLE_SIZE)]` lowers to
/// `ll_newlist(0)` plus an append loop (`flowspace` `LIST_APPEND`),
/// then the `table[ord(needle[i]) & MASK] = ...` loop is
/// `ll_setitem_fast` (`rlist.py` `ll_setitem_nonneg` with a known-nonneg
/// index).
pub(crate) fn build_twoway_preprocess_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype.clone())?;
    let needle = v("needle", kind.ptr_lltype.clone());
    let len_needle = v("len_needle", LowLevelType::Signed);
    let startblock = blk(vec![needle.clone(), len_needle.clone()]);
    let return_var = v("result", tuple5_ptr_lltype());
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));

    let table_ops = twoway_table_ops(rtyper)?;
    let factorize = {
        let ptr = ptr_lltype.clone();
        cached_helper_funcptr(
            rtyper,
            "_factorize",
            vec![kind.ptr_lltype.clone(), LowLevelType::Signed],
            tuple2_ptr_lltype(),
            move |rtyper| build_factorize_helper_graph(rtyper, "_factorize", ptr),
        )?
    };
    let fact = emit_direct_call(
        &startblock,
        factorize,
        vec![hl(&needle), hl(&len_needle)],
        "fact",
        tuple2_ptr_lltype(),
    );
    let cut = emit_getfield(&startblock, &fact, "item0", "cut", LowLevelType::Signed);
    let period = emit_getfield(&startblock, &fact, "item1", "period", LowLevelType::Signed);

    let n_p = v("needle", kind.ptr_lltype.clone());
    let ln_p = v("len_needle", LowLevelType::Signed);
    let cut_p = v("cut", LowLevelType::Signed);
    let per_p = v("period", LowLevelType::Signed);
    let i_p = v("i", LowLevelType::Signed);
    let period_check = blk(vec![
        n_p.clone(),
        ln_p.clone(),
        cut_p.clone(),
        per_p.clone(),
        i_p.clone(),
    ]);
    goto(
        &startblock,
        vec![
            hl(&needle),
            hl(&len_needle),
            hl(&cut),
            hl(&period),
            signed_const(0),
        ],
        period_check.clone(),
    );

    let still = emit_bin(
        &period_check,
        "int_lt",
        hl(&i_p),
        hl(&cut_p),
        "still",
        LowLevelType::Bool,
    );
    let pbody = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    let after_period = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
    ]);
    let pbody_ins = block_vars(&pbody);
    let after_p_ins = block_vars(&after_period);
    branch(
        &period_check,
        still,
        vec![hl(&n_p), hl(&ln_p), hl(&cut_p), hl(&per_p), hl(&i_p)],
        pbody.clone(),
        vec![
            hl(&n_p),
            hl(&ln_p),
            hl(&cut_p),
            hl(&per_p),
            bool_const(true),
        ],
        after_period.clone(),
    );

    let idx_r = emit_bin(
        &pbody,
        "int_add",
        hl(&pbody_ins[3]),
        hl(&pbody_ins[4]),
        "idx",
        LowLevelType::Signed,
    );
    let ch_l = emit_getitem(&kind, &pbody, &pbody_ins[0], hl(&pbody_ins[4]));
    let ch_r = emit_getitem(&kind, &pbody, &pbody_ins[0], hl(&idx_r));
    let ne = emit_bin(
        &pbody,
        kind.ne_op,
        hl(&ch_l),
        hl(&ch_r),
        "ne",
        LowLevelType::Bool,
    );
    let not_periodic = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
    ]);
    let np_ins = block_vars(&not_periodic);
    let i_next = emit_bin(
        &pbody,
        "int_add",
        hl(&pbody_ins[4]),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    branch(
        &pbody,
        ne,
        vec![
            hl(&pbody_ins[0]),
            hl(&pbody_ins[1]),
            hl(&pbody_ins[2]),
            hl(&pbody_ins[3]),
        ],
        not_periodic.clone(),
        vec![
            hl(&pbody_ins[0]),
            hl(&pbody_ins[1]),
            hl(&pbody_ins[2]),
            hl(&pbody_ins[3]),
            hl(&i_next),
        ],
        period_check.clone(),
    );
    goto(
        &not_periodic,
        vec![
            hl(&np_ins[0]),
            hl(&np_ins[1]),
            hl(&np_ins[2]),
            hl(&np_ins[3]),
            bool_const(false),
        ],
        after_period.clone(),
    );

    let gap_periodic = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
    ]);
    let gap_non = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
    ]);
    branch(
        &after_period,
        after_p_ins[4].clone(),
        vec![
            hl(&after_p_ins[0]),
            hl(&after_p_ins[1]),
            hl(&after_p_ins[2]),
            hl(&after_p_ins[3]),
            hl(&after_p_ins[4]),
            signed_const(0),
        ],
        gap_periodic.clone(),
        vec![
            hl(&after_p_ins[0]),
            hl(&after_p_ins[1]),
            hl(&after_p_ins[2]),
            hl(&after_p_ins[3]),
            hl(&after_p_ins[4]),
        ],
        gap_non.clone(),
    );

    let gp_ins = block_vars(&gap_periodic);
    let gn_ins = block_vars(&gap_non);

    let rest = emit_bin(
        &gap_non,
        "int_sub",
        hl(&gn_ins[1]),
        hl(&gn_ins[2]),
        "rest",
        LowLevelType::Signed,
    );
    let cut_gt_rest = emit_bin(
        &gap_non,
        "int_gt",
        hl(&gn_ins[2]),
        hl(&rest),
        "gt",
        LowLevelType::Bool,
    );
    let per_from_cut = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("half", LowLevelType::Signed),
    ]);
    let per_from_rest = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("half", LowLevelType::Signed),
    ]);
    branch(
        &gap_non,
        cut_gt_rest,
        vec![
            hl(&gn_ins[0]),
            hl(&gn_ins[1]),
            hl(&gn_ins[2]),
            hl(&gn_ins[4]),
            hl(&gn_ins[2]),
        ],
        per_from_cut.clone(),
        vec![
            hl(&gn_ins[0]),
            hl(&gn_ins[1]),
            hl(&gn_ins[2]),
            hl(&gn_ins[4]),
            hl(&rest),
        ],
        per_from_rest.clone(),
    );
    let pfc = block_vars(&per_from_cut);
    let pfr = block_vars(&per_from_rest);
    let period_cut = emit_bin(
        &per_from_cut,
        "int_add",
        hl(&pfc[4]),
        signed_const(1),
        "period",
        LowLevelType::Signed,
    );
    let period_rest = emit_bin(
        &per_from_rest,
        "int_add",
        hl(&pfr[4]),
        signed_const(1),
        "period",
        LowLevelType::Signed,
    );
    let gap_scan = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
        v("last", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    let enter_gap_scan = |block: &BlockRef, ins: &[Variable], period: &Variable| {
        let last_idx = emit_bin(
            block,
            "int_sub",
            hl(&ins[1]),
            signed_const(1),
            "last_idx",
            LowLevelType::Signed,
        );
        let last_ch = emit_getitem(&kind, block, &ins[0], hl(&last_idx));
        let last_ord = emit_ord(&kind, block, last_ch);
        let last = emit_bin(
            block,
            "int_and",
            hl(&last_ord),
            signed_const(TWOWAY_TABLE_MASK),
            "last",
            LowLevelType::Signed,
        );
        let i0 = emit_bin(
            block,
            "int_sub",
            hl(&ins[1]),
            signed_const(2),
            "i",
            LowLevelType::Signed,
        );
        goto(
            block,
            vec![
                hl(&ins[0]),
                hl(&ins[1]),
                hl(&ins[2]),
                hl(period),
                hl(&ins[3]),
                hl(&ins[1]),
                hl(&last),
                hl(&i0),
            ],
            gap_scan.clone(),
        );
        Ok::<(), TyperError>(())
    };
    enter_gap_scan(&per_from_cut, &pfc, &period_cut)?;
    enter_gap_scan(&per_from_rest, &pfr, &period_rest)?;

    let gs = block_vars(&gap_scan);
    let i_ge0 = emit_bin(
        &gap_scan,
        "int_ge",
        hl(&gs[7]),
        signed_const(0),
        "ge",
        LowLevelType::Bool,
    );
    let gs_body = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
        v("last", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    let after_gap = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
    ]);
    let gsb = block_vars(&gs_body);
    branch(
        &gap_scan,
        i_ge0,
        vec![
            hl(&gs[0]),
            hl(&gs[1]),
            hl(&gs[2]),
            hl(&gs[3]),
            hl(&gs[4]),
            hl(&gs[5]),
            hl(&gs[6]),
            hl(&gs[7]),
        ],
        gs_body.clone(),
        vec![
            hl(&gs[0]),
            hl(&gs[1]),
            hl(&gs[2]),
            hl(&gs[3]),
            hl(&gs[4]),
            hl(&gs[5]),
        ],
        after_gap.clone(),
    );
    let ch_i = emit_getitem(&kind, &gs_body, &gsb[0], hl(&gsb[7]));
    let o_i = emit_ord(&kind, &gs_body, ch_i);
    let masked_i = emit_bin(
        &gs_body,
        "int_and",
        hl(&o_i),
        signed_const(TWOWAY_TABLE_MASK),
        "m",
        LowLevelType::Signed,
    );
    let eq_last = emit_bin(
        &gs_body,
        "int_eq",
        hl(&masked_i),
        hl(&gsb[6]),
        "eq",
        LowLevelType::Bool,
    );
    let found_gap = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("i", LowLevelType::Signed),
    ]);
    let fg = block_vars(&found_gap);
    let i_dec = emit_bin(
        &gs_body,
        "int_sub",
        hl(&gsb[7]),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    branch(
        &gs_body,
        eq_last,
        vec![
            hl(&gsb[0]),
            hl(&gsb[1]),
            hl(&gsb[2]),
            hl(&gsb[3]),
            hl(&gsb[4]),
            hl(&gsb[7]),
        ],
        found_gap.clone(),
        vec![
            hl(&gsb[0]),
            hl(&gsb[1]),
            hl(&gsb[2]),
            hl(&gsb[3]),
            hl(&gsb[4]),
            hl(&gsb[5]),
            hl(&gsb[6]),
            hl(&i_dec),
        ],
        gap_scan.clone(),
    );
    let gap_found = emit_bin(
        &found_gap,
        "int_sub",
        hl(&fg[1]),
        signed_const(1),
        "t",
        LowLevelType::Signed,
    );
    let gap_found = emit_bin(
        &found_gap,
        "int_sub",
        hl(&gap_found),
        hl(&fg[5]),
        "gap",
        LowLevelType::Signed,
    );
    goto(
        &found_gap,
        vec![
            hl(&fg[0]),
            hl(&fg[1]),
            hl(&fg[2]),
            hl(&fg[3]),
            hl(&fg[4]),
            hl(&gap_found),
        ],
        after_gap.clone(),
    );

    let ag = block_vars(&after_gap);
    let gp_to_after = || {
        goto(
            &gap_periodic,
            vec![
                hl(&gp_ins[0]),
                hl(&gp_ins[1]),
                hl(&gp_ins[2]),
                hl(&gp_ins[3]),
                hl(&gp_ins[4]),
                hl(&gp_ins[5]),
            ],
            after_gap.clone(),
        );
    };
    gp_to_after();

    let _build_table = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
    ]);
    // after_gap already is build_table-shaped; reuse it as the table-build entry.
    let bt = ag;
    let lt_max = emit_bin(
        &after_gap,
        "int_lt",
        hl(&bt[1]),
        signed_const(TWOWAY_MAX_SHIFT),
        "lt",
        LowLevelType::Bool,
    );
    let nfs_len = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
        v("not_found_shift", LowLevelType::Signed),
    ]);
    let nfs_max = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
        v("not_found_shift", LowLevelType::Signed),
    ]);
    branch(
        &after_gap,
        lt_max,
        vec![
            hl(&bt[0]),
            hl(&bt[1]),
            hl(&bt[2]),
            hl(&bt[3]),
            hl(&bt[4]),
            hl(&bt[5]),
            hl(&bt[1]),
        ],
        nfs_len.clone(),
        vec![
            hl(&bt[0]),
            hl(&bt[1]),
            hl(&bt[2]),
            hl(&bt[3]),
            hl(&bt[4]),
            hl(&bt[5]),
            signed_const(TWOWAY_MAX_SHIFT),
        ],
        nfs_max.clone(),
    );

    let fill_entry = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
        v("not_found_shift", LowLevelType::Signed),
        v("table", table_list_ptr_lltype()),
        v("_i", LowLevelType::Signed),
    ]);
    let enter_fill = |block: &BlockRef| -> Result<(), TyperError> {
        let ins = block_vars(block);
        let table = emit_direct_call(
            block,
            table_ops.newlist.clone(),
            vec![signed_const(0)],
            "table",
            table_list_ptr_lltype(),
        );
        goto(
            block,
            vec![
                hl(&ins[0]),
                hl(&ins[1]),
                hl(&ins[2]),
                hl(&ins[3]),
                hl(&ins[4]),
                hl(&ins[5]),
                hl(&ins[6]),
                hl(&table),
                signed_const(0),
            ],
            fill_entry.clone(),
        );
        Ok(())
    };
    enter_fill(&nfs_len)?;
    enter_fill(&nfs_max)?;

    let fe = block_vars(&fill_entry);
    let i_lt = emit_bin(
        &fill_entry,
        "int_lt",
        hl(&fe[8]),
        signed_const(TWOWAY_TABLE_SIZE),
        "lt",
        LowLevelType::Bool,
    );
    let fill_body = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
        v("not_found_shift", LowLevelType::Signed),
        v("table", table_list_ptr_lltype()),
        v("_i", LowLevelType::Signed),
    ]);
    let after_fill = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
        v("not_found_shift", LowLevelType::Signed),
        v("table", table_list_ptr_lltype()),
    ]);
    let fb = block_vars(&fill_body);
    branch(
        &fill_entry,
        i_lt,
        vec![
            hl(&fe[0]),
            hl(&fe[1]),
            hl(&fe[2]),
            hl(&fe[3]),
            hl(&fe[4]),
            hl(&fe[5]),
            hl(&fe[6]),
            hl(&fe[7]),
            hl(&fe[8]),
        ],
        fill_body.clone(),
        vec![
            hl(&fe[0]),
            hl(&fe[1]),
            hl(&fe[2]),
            hl(&fe[3]),
            hl(&fe[4]),
            hl(&fe[5]),
            hl(&fe[6]),
            hl(&fe[7]),
        ],
        after_fill.clone(),
    );
    emit_direct_call(
        &fill_body,
        table_ops.append.clone(),
        vec![hl(&fb[7]), hl(&fb[6])],
        "v",
        LowLevelType::Void,
    );
    let i_inc = emit_bin(
        &fill_body,
        "int_add",
        hl(&fb[8]),
        signed_const(1),
        "_i",
        LowLevelType::Signed,
    );
    goto(
        &fill_body,
        vec![
            hl(&fb[0]),
            hl(&fb[1]),
            hl(&fb[2]),
            hl(&fb[3]),
            hl(&fb[4]),
            hl(&fb[5]),
            hl(&fb[6]),
            hl(&fb[7]),
            hl(&i_inc),
        ],
        fill_entry.clone(),
    );

    let af = block_vars(&after_fill);
    let i_start = emit_bin(
        &after_fill,
        "int_sub",
        hl(&af[1]),
        hl(&af[6]),
        "i",
        LowLevelType::Signed,
    );
    let shift_loop = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
        v("table", table_list_ptr_lltype()),
        v("i", LowLevelType::Signed),
    ]);
    goto(
        &after_fill,
        vec![
            hl(&af[0]),
            hl(&af[1]),
            hl(&af[2]),
            hl(&af[3]),
            hl(&af[4]),
            hl(&af[5]),
            hl(&af[7]),
            hl(&i_start),
        ],
        shift_loop.clone(),
    );
    let sl = block_vars(&shift_loop);
    let i_lt_n = emit_bin(
        &shift_loop,
        "int_lt",
        hl(&sl[7]),
        hl(&sl[1]),
        "lt",
        LowLevelType::Bool,
    );
    let sl_body = blk(vec![
        v("needle", kind.ptr_lltype.clone()),
        v("len_needle", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("gap", LowLevelType::Signed),
        v("table", table_list_ptr_lltype()),
        v("i", LowLevelType::Signed),
    ]);
    let sl_done = blk(vec![
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("gap", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("table", table_list_ptr_lltype()),
    ]);
    let slb = block_vars(&sl_body);
    let sld = block_vars(&sl_done);
    branch(
        &shift_loop,
        i_lt_n,
        vec![
            hl(&sl[0]),
            hl(&sl[1]),
            hl(&sl[2]),
            hl(&sl[3]),
            hl(&sl[4]),
            hl(&sl[5]),
            hl(&sl[6]),
            hl(&sl[7]),
        ],
        sl_body.clone(),
        vec![hl(&sl[2]), hl(&sl[3]), hl(&sl[5]), hl(&sl[4]), hl(&sl[6])],
        sl_done.clone(),
    );
    let ch = emit_getitem(&kind, &sl_body, &slb[0], hl(&slb[7]));
    let o = emit_ord(&kind, &sl_body, ch);
    let idx = emit_bin(
        &sl_body,
        "int_and",
        hl(&o),
        signed_const(TWOWAY_TABLE_MASK),
        "idx",
        LowLevelType::Signed,
    );
    let n1 = emit_bin(
        &sl_body,
        "int_sub",
        hl(&slb[1]),
        signed_const(1),
        "n1",
        LowLevelType::Signed,
    );
    let shift = emit_bin(
        &sl_body,
        "int_sub",
        hl(&n1),
        hl(&slb[7]),
        "shift",
        LowLevelType::Signed,
    );
    emit_direct_call(
        &sl_body,
        table_ops.setitem.clone(),
        vec![hl(&slb[6]), hl(&idx), hl(&shift)],
        "v",
        LowLevelType::Void,
    );
    let i_inc2 = emit_bin(
        &sl_body,
        "int_add",
        hl(&slb[7]),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    goto(
        &sl_body,
        vec![
            hl(&slb[0]),
            hl(&slb[1]),
            hl(&slb[2]),
            hl(&slb[3]),
            hl(&slb[4]),
            hl(&slb[5]),
            hl(&slb[6]),
            hl(&i_inc2),
        ],
        shift_loop.clone(),
    );
    let tup = emit_tuple5(
        &sl_done,
        hl(&sld[0]),
        hl(&sld[1]),
        hl(&sld[2]),
        hl(&sld[3]),
        hl(&sld[4]),
    )?;
    ret(&sl_done, hl(&tup), graph.returnblock.clone());
    Ok(finish_graph(
        graph,
        name,
        vec!["needle".into(), "len_needle".into()],
        false,
    ))
}

fn emit_table_shift(
    kind: &StringKind,
    block: &BlockRef,
    getitem: &Hlvalue,
    value: &Variable,
    table: &Variable,
    window_last: &Variable,
) -> (Variable, Variable) {
    let ch = emit_getitem(kind, block, value, hl(window_last));
    let o = emit_ord(kind, block, ch);
    let idx = emit_bin(
        block,
        "int_and",
        hl(&o),
        signed_const(TWOWAY_TABLE_MASK),
        "idx",
        LowLevelType::Signed,
    );
    let shift = emit_table_get(block, getitem, table, hl(&idx));
    let window_last2 = emit_bin(
        block,
        "int_add",
        hl(window_last),
        hl(&shift),
        "window_last",
        LowLevelType::Signed,
    );
    (shift, window_last2)
}

/// `rstring.py` `_two_way`.
pub(crate) fn build_two_way_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype)?;
    let table_getitem = twoway_table_ops(rtyper)?.getitem;
    let value = v("value", kind.ptr_lltype.clone());
    let base = v("base", LowLevelType::Signed);
    let n = v("n", LowLevelType::Signed);
    let needle = v("needle", kind.ptr_lltype.clone());
    let m = v("m", LowLevelType::Signed);
    let cut = v("cut", LowLevelType::Signed);
    let period = v("period", LowLevelType::Signed);
    let gap = v("gap", LowLevelType::Signed);
    let is_periodic = v("is_periodic", LowLevelType::Bool);
    let table = v("table", table_list_ptr_lltype());
    let startblock = blk(vec![
        value.clone(),
        base.clone(),
        n.clone(),
        needle.clone(),
        m.clone(),
        cut.clone(),
        period.clone(),
        gap.clone(),
        is_periodic.clone(),
        table.clone(),
    ]);
    let return_var = v("result", LowLevelType::Signed);
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));
    let haystack_end = emit_bin(
        &startblock,
        "int_add",
        hl(&base),
        hl(&n),
        "haystack_end",
        LowLevelType::Signed,
    );
    let m1 = emit_bin(
        &startblock,
        "int_sub",
        hl(&m),
        signed_const(1),
        "m1",
        LowLevelType::Signed,
    );
    let window_last0 = emit_bin(
        &startblock,
        "int_add",
        hl(&base),
        hl(&m1),
        "window_last",
        LowLevelType::Signed,
    );

    let p_init = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("skip_horspool", LowLevelType::Bool),
    ]);
    let np_init = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("gap", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
    ]);
    branch(
        &startblock,
        is_periodic,
        vec![
            hl(&value),
            hl(&needle),
            hl(&table),
            hl(&base),
            hl(&m),
            hl(&cut),
            hl(&period),
            hl(&haystack_end),
            hl(&window_last0),
            signed_const(0),
            bool_const(false),
        ],
        p_init.clone(),
        vec![
            hl(&value),
            hl(&needle),
            hl(&table),
            hl(&base),
            hl(&m),
            hl(&cut),
            hl(&period),
            hl(&gap),
            hl(&haystack_end),
            hl(&window_last0),
        ],
        np_init.clone(),
    );

    build_two_way_periodic(&kind, &table_getitem, &p_init, graph.returnblock.clone())?;
    build_two_way_nonperiodic(&kind, &table_getitem, &np_init, graph.returnblock.clone())?;
    Ok(finish_graph(
        graph,
        name,
        vec![
            "value".into(),
            "base".into(),
            "n".into(),
            "needle".into(),
            "m".into(),
            "cut".into(),
            "period".into(),
            "gap".into(),
            "is_periodic".into(),
            "table".into(),
        ],
        false,
    ))
}

fn pcarry(ins: &[Variable]) -> Vec<Hlvalue> {
    ins.iter().map(hl).collect()
}

fn build_two_way_periodic(
    kind: &StringKind,
    table_getitem: &Hlvalue,
    p_init: &BlockRef,
    returnblock: BlockRef,
) -> Result<(), TyperError> {
    let pin = block_vars(p_init);
    let loop_cond = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("skip_horspool", LowLevelType::Bool),
    ]);
    goto(&p_init, pcarry(&pin), loop_cond.clone());
    let lc = block_vars(&loop_cond);
    let still = emit_bin(
        &loop_cond,
        "int_lt",
        hl(&lc[8]),
        hl(&lc[7]),
        "still",
        LowLevelType::Bool,
    );
    let body = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("skip_horspool", LowLevelType::Bool),
    ]);
    branch(
        &loop_cond,
        still,
        pcarry(&lc),
        body.clone(),
        vec![signed_const(-1)],
        returnblock.clone(),
    );
    let bd = block_vars(&body);
    let not_skip = emit_un(
        &body,
        "bool_not",
        hl(&bd[10]),
        "not_skip",
        LowLevelType::Bool,
    );
    let hp = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
    ]);
    let after_hp = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
    ]);
    branch(
        &body,
        not_skip,
        vec![
            hl(&bd[0]),
            hl(&bd[1]),
            hl(&bd[2]),
            hl(&bd[3]),
            hl(&bd[4]),
            hl(&bd[5]),
            hl(&bd[6]),
            hl(&bd[7]),
            hl(&bd[8]),
            hl(&bd[9]),
        ],
        hp.clone(),
        vec![
            hl(&bd[0]),
            hl(&bd[1]),
            hl(&bd[2]),
            hl(&bd[3]),
            hl(&bd[4]),
            hl(&bd[5]),
            hl(&bd[6]),
            hl(&bd[7]),
            hl(&bd[8]),
            hl(&bd[9]),
        ],
        after_hp.clone(),
    );
    emit_horspool_loop(kind, table_getitem, &hp, &after_hp, &returnblock, 10)?;

    let ah = block_vars(&after_hp);
    let m1 = emit_bin(
        &after_hp,
        "int_sub",
        hl(&ah[4]),
        signed_const(1),
        "m1",
        LowLevelType::Signed,
    );
    let window = emit_bin(
        &after_hp,
        "int_sub",
        hl(&ah[8]),
        hl(&m1),
        "window",
        LowLevelType::Signed,
    );
    let cut_gt_mem = emit_bin(
        &after_hp,
        "int_gt",
        hl(&ah[5]),
        hl(&ah[9]),
        "gt",
        LowLevelType::Bool,
    );
    let i_cut = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("window", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    let i_mem = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("window", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    let pack_i = |ins: &[Variable], window: &Variable, i: Hlvalue| {
        vec![
            hl(&ins[0]),
            hl(&ins[1]),
            hl(&ins[2]),
            hl(&ins[3]),
            hl(&ins[4]),
            hl(&ins[5]),
            hl(&ins[6]),
            hl(&ins[7]),
            hl(&ins[8]),
            hl(&ins[9]),
            hl(window),
            i,
        ]
    };
    branch(
        &after_hp,
        cut_gt_mem,
        pack_i(&ah, &window, hl(&ah[5])),
        i_cut.clone(),
        pack_i(&ah, &window, hl(&ah[9])),
        i_mem.clone(),
    );
    let right_cond = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("window", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    goto(&i_cut, pcarry(&block_vars(&i_cut)), right_cond.clone());
    goto(&i_mem, pcarry(&block_vars(&i_mem)), right_cond.clone());
    let rc = block_vars(&right_cond);
    let i_lt_m = emit_bin(
        &right_cond,
        "int_lt",
        hl(&rc[11]),
        hl(&rc[4]),
        "lt",
        LowLevelType::Bool,
    );
    let right_body = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("window", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    let left_init = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("window", LowLevelType::Signed),
    ]);
    branch(
        &right_cond,
        i_lt_m,
        pcarry(&rc),
        right_body.clone(),
        vec![
            hl(&rc[0]),
            hl(&rc[1]),
            hl(&rc[2]),
            hl(&rc[3]),
            hl(&rc[4]),
            hl(&rc[5]),
            hl(&rc[6]),
            hl(&rc[7]),
            hl(&rc[8]),
            hl(&rc[9]),
            hl(&rc[10]),
        ],
        left_init.clone(),
    );
    let rb = block_vars(&right_body);
    let pos = emit_bin(
        &right_body,
        "int_add",
        hl(&rb[10]),
        hl(&rb[11]),
        "pos",
        LowLevelType::Signed,
    );
    let nch = emit_getitem(kind, &right_body, &rb[1], hl(&rb[11]));
    let vch = emit_getitem(kind, &right_body, &rb[0], hl(&pos));
    let ne = emit_bin(
        &right_body,
        kind.ne_op,
        hl(&nch),
        hl(&vch),
        "ne",
        LowLevelType::Bool,
    );
    let r_mis = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    let rm = block_vars(&r_mis);
    let i_inc = emit_bin(
        &right_body,
        "int_add",
        hl(&rb[11]),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    branch(
        &right_body,
        ne,
        vec![
            hl(&rb[0]),
            hl(&rb[1]),
            hl(&rb[2]),
            hl(&rb[3]),
            hl(&rb[4]),
            hl(&rb[5]),
            hl(&rb[6]),
            hl(&rb[7]),
            hl(&rb[8]),
            hl(&rb[11]),
        ],
        r_mis.clone(),
        vec![
            hl(&rb[0]),
            hl(&rb[1]),
            hl(&rb[2]),
            hl(&rb[3]),
            hl(&rb[4]),
            hl(&rb[5]),
            hl(&rb[6]),
            hl(&rb[7]),
            hl(&rb[8]),
            hl(&rb[9]),
            hl(&rb[10]),
            hl(&i_inc),
        ],
        right_cond.clone(),
    );
    let delta = emit_bin(
        &r_mis,
        "int_sub",
        hl(&rm[9]),
        hl(&rm[5]),
        "d",
        LowLevelType::Signed,
    );
    let delta1 = emit_bin(
        &r_mis,
        "int_add",
        hl(&delta),
        signed_const(1),
        "d1",
        LowLevelType::Signed,
    );
    let wl2 = emit_bin(
        &r_mis,
        "int_add",
        hl(&rm[8]),
        hl(&delta1),
        "wl",
        LowLevelType::Signed,
    );
    goto(
        &r_mis,
        vec![
            hl(&rm[0]),
            hl(&rm[1]),
            hl(&rm[2]),
            hl(&rm[3]),
            hl(&rm[4]),
            hl(&rm[5]),
            hl(&rm[6]),
            hl(&rm[7]),
            hl(&wl2),
            signed_const(0),
            bool_const(false),
        ],
        loop_cond.clone(),
    );

    let li = block_vars(&left_init);
    let left_cond = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("window", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    goto(
        &left_init,
        vec![
            hl(&li[0]),
            hl(&li[1]),
            hl(&li[2]),
            hl(&li[3]),
            hl(&li[4]),
            hl(&li[5]),
            hl(&li[6]),
            hl(&li[7]),
            hl(&li[8]),
            hl(&li[9]),
            hl(&li[10]),
            hl(&li[9]),
        ],
        left_cond.clone(),
    );
    let lcc = block_vars(&left_cond);
    let i_lt_cut = emit_bin(
        &left_cond,
        "int_lt",
        hl(&lcc[11]),
        hl(&lcc[5]),
        "lt",
        LowLevelType::Bool,
    );
    let left_body = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("window", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    let matched = blk(vec![
        v("base", LowLevelType::Signed),
        v("window", LowLevelType::Signed),
    ]);
    branch(
        &left_cond,
        i_lt_cut,
        pcarry(&lcc),
        left_body.clone(),
        vec![hl(&lcc[3]), hl(&lcc[10])],
        matched.clone(),
    );
    let lb = block_vars(&left_body);
    let pos2 = emit_bin(
        &left_body,
        "int_add",
        hl(&lb[10]),
        hl(&lb[11]),
        "pos",
        LowLevelType::Signed,
    );
    let nch2 = emit_getitem(kind, &left_body, &lb[1], hl(&lb[11]));
    let vch2 = emit_getitem(kind, &left_body, &lb[0], hl(&pos2));
    let ne2 = emit_bin(
        &left_body,
        kind.ne_op,
        hl(&nch2),
        hl(&vch2),
        "ne",
        LowLevelType::Bool,
    );
    let l_mis = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
    ]);
    let i_inc2 = emit_bin(
        &left_body,
        "int_add",
        hl(&lb[11]),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    branch(
        &left_body,
        ne2,
        vec![
            hl(&lb[0]),
            hl(&lb[1]),
            hl(&lb[2]),
            hl(&lb[3]),
            hl(&lb[4]),
            hl(&lb[5]),
            hl(&lb[6]),
            hl(&lb[7]),
            hl(&lb[8]),
            hl(&lb[9]),
        ],
        l_mis.clone(),
        vec![
            hl(&lb[0]),
            hl(&lb[1]),
            hl(&lb[2]),
            hl(&lb[3]),
            hl(&lb[4]),
            hl(&lb[5]),
            hl(&lb[6]),
            hl(&lb[7]),
            hl(&lb[8]),
            hl(&lb[9]),
            hl(&lb[10]),
            hl(&i_inc2),
        ],
        left_cond.clone(),
    );
    let lm = block_vars(&l_mis);
    let wl_p = emit_bin(
        &l_mis,
        "int_add",
        hl(&lm[8]),
        hl(&lm[6]),
        "wl",
        LowLevelType::Signed,
    );
    let mem2 = emit_bin(
        &l_mis,
        "int_sub",
        hl(&lm[4]),
        hl(&lm[6]),
        "mem",
        LowLevelType::Signed,
    );
    let ge_end = emit_bin(
        &l_mis,
        "int_ge",
        hl(&wl_p),
        hl(&lm[7]),
        "ge",
        LowLevelType::Bool,
    );
    let l_end = blk(Vec::new());
    let l_shift = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
    ]);
    branch(
        &l_mis,
        ge_end,
        vec![],
        l_end.clone(),
        vec![
            hl(&lm[0]),
            hl(&lm[1]),
            hl(&lm[2]),
            hl(&lm[3]),
            hl(&lm[4]),
            hl(&lm[5]),
            hl(&lm[6]),
            hl(&lm[7]),
            hl(&wl_p),
            hl(&mem2),
        ],
        l_shift.clone(),
    );
    ret(&l_end, signed_const(-1), returnblock.clone());
    let ls = block_vars(&l_shift);
    let (shift, _) = emit_table_shift(kind, &l_shift, table_getitem, &ls[0], &ls[2], &ls[8]);
    let shift_nz = emit_un(
        &l_shift,
        "int_is_true",
        hl(&shift),
        "nz",
        LowLevelType::Bool,
    );
    let do_jump = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
        v("shift", LowLevelType::Signed),
    ]);
    let skip_next = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("memory", LowLevelType::Signed),
    ]);
    branch(
        &l_shift,
        shift_nz,
        vec![
            hl(&ls[0]),
            hl(&ls[1]),
            hl(&ls[2]),
            hl(&ls[3]),
            hl(&ls[4]),
            hl(&ls[5]),
            hl(&ls[6]),
            hl(&ls[7]),
            hl(&ls[8]),
            hl(&ls[9]),
            hl(&shift),
        ],
        do_jump.clone(),
        vec![
            hl(&ls[0]),
            hl(&ls[1]),
            hl(&ls[2]),
            hl(&ls[3]),
            hl(&ls[4]),
            hl(&ls[5]),
            hl(&ls[6]),
            hl(&ls[7]),
            hl(&ls[8]),
            hl(&ls[9]),
        ],
        skip_next.clone(),
    );
    let sn = block_vars(&skip_next);
    goto(
        &skip_next,
        vec![
            hl(&sn[0]),
            hl(&sn[1]),
            hl(&sn[2]),
            hl(&sn[3]),
            hl(&sn[4]),
            hl(&sn[5]),
            hl(&sn[6]),
            hl(&sn[7]),
            hl(&sn[8]),
            hl(&sn[9]),
            bool_const(true),
        ],
        loop_cond.clone(),
    );
    let dj = block_vars(&do_jump);
    let cut_gt = emit_bin(
        &do_jump,
        "int_gt",
        hl(&dj[5]),
        hl(&dj[9]),
        "gt",
        LowLevelType::Bool,
    );
    let mj_cut = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("shift", LowLevelType::Signed),
        v("mj", LowLevelType::Signed),
    ]);
    let mj_mem = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("shift", LowLevelType::Signed),
        v("mj", LowLevelType::Signed),
    ]);
    branch(
        &do_jump,
        cut_gt,
        vec![
            hl(&dj[0]),
            hl(&dj[1]),
            hl(&dj[2]),
            hl(&dj[3]),
            hl(&dj[4]),
            hl(&dj[5]),
            hl(&dj[6]),
            hl(&dj[7]),
            hl(&dj[8]),
            hl(&dj[10]),
            hl(&dj[5]),
        ],
        mj_cut.clone(),
        vec![
            hl(&dj[0]),
            hl(&dj[1]),
            hl(&dj[2]),
            hl(&dj[3]),
            hl(&dj[4]),
            hl(&dj[5]),
            hl(&dj[6]),
            hl(&dj[7]),
            hl(&dj[8]),
            hl(&dj[10]),
            hl(&dj[9]),
        ],
        mj_mem.clone(),
    );
    let apply_jump = |block: &BlockRef| {
        let ins = block_vars(block);
        let mem_jump = emit_bin(
            block,
            "int_sub",
            hl(&ins[10]),
            hl(&ins[5]),
            "mj",
            LowLevelType::Signed,
        );
        let mem_jump = emit_bin(
            block,
            "int_add",
            hl(&mem_jump),
            signed_const(1),
            "mem_jump",
            LowLevelType::Signed,
        );
        let sh_gt = emit_bin(
            block,
            "int_gt",
            hl(&ins[9]),
            hl(&mem_jump),
            "gt",
            LowLevelType::Bool,
        );
        let take_sh = blk(vec![
            v("value", kind.ptr_lltype.clone()),
            v("needle", kind.ptr_lltype.clone()),
            v("table", table_list_ptr_lltype()),
            v("base", LowLevelType::Signed),
            v("m", LowLevelType::Signed),
            v("cut", LowLevelType::Signed),
            v("period", LowLevelType::Signed),
            v("haystack_end", LowLevelType::Signed),
            v("window_last", LowLevelType::Signed),
            v("delta", LowLevelType::Signed),
        ]);
        let take_mj = blk(vec![
            v("value", kind.ptr_lltype.clone()),
            v("needle", kind.ptr_lltype.clone()),
            v("table", table_list_ptr_lltype()),
            v("base", LowLevelType::Signed),
            v("m", LowLevelType::Signed),
            v("cut", LowLevelType::Signed),
            v("period", LowLevelType::Signed),
            v("haystack_end", LowLevelType::Signed),
            v("window_last", LowLevelType::Signed),
            v("delta", LowLevelType::Signed),
        ]);
        branch(
            block,
            sh_gt,
            vec![
                hl(&ins[0]),
                hl(&ins[1]),
                hl(&ins[2]),
                hl(&ins[3]),
                hl(&ins[4]),
                hl(&ins[5]),
                hl(&ins[6]),
                hl(&ins[7]),
                hl(&ins[8]),
                hl(&ins[9]),
            ],
            take_sh.clone(),
            vec![
                hl(&ins[0]),
                hl(&ins[1]),
                hl(&ins[2]),
                hl(&ins[3]),
                hl(&ins[4]),
                hl(&ins[5]),
                hl(&ins[6]),
                hl(&ins[7]),
                hl(&ins[8]),
                hl(&mem_jump),
            ],
            take_mj.clone(),
        );
        let finish = |b: &BlockRef| {
            let vs = block_vars(b);
            let wl = emit_bin(
                b,
                "int_add",
                hl(&vs[8]),
                hl(&vs[9]),
                "wl",
                LowLevelType::Signed,
            );
            goto(
                b,
                vec![
                    hl(&vs[0]),
                    hl(&vs[1]),
                    hl(&vs[2]),
                    hl(&vs[3]),
                    hl(&vs[4]),
                    hl(&vs[5]),
                    hl(&vs[6]),
                    hl(&vs[7]),
                    hl(&wl),
                    signed_const(0),
                    bool_const(false),
                ],
                loop_cond.clone(),
            );
        };
        finish(&take_sh);
        finish(&take_mj);
    };
    apply_jump(&mj_cut);
    apply_jump(&mj_mem);

    let mt = block_vars(&matched);
    let res = emit_bin(
        &matched,
        "int_sub",
        hl(&mt[1]),
        hl(&mt[0]),
        "res",
        LowLevelType::Signed,
    );
    ret(&matched, hl(&res), returnblock);
    Ok(())
}

fn clone_block_like(ins: &[Variable]) -> BlockRef {
    blk(ins.iter().map(Variable::copy).collect())
}

fn emit_horspool_loop(
    kind: &StringKind,
    table_getitem: &Hlvalue,
    hp: &BlockRef,
    after_hp: &BlockRef,
    returnblock: &BlockRef,
    ncarry: usize,
) -> Result<(), TyperError> {
    let ins = block_vars(hp);
    debug_assert_eq!(ins.len(), ncarry);
    let (shift, wl2) = emit_table_shift(kind, hp, table_getitem, &ins[0], &ins[2], &ins[8]);
    let z = emit_bin(
        hp,
        "int_eq",
        hl(&shift),
        signed_const(0),
        "z",
        LowLevelType::Bool,
    );
    let check_end = clone_block_like(&ins);
    let mut to_after: Vec<Hlvalue> = ins.iter().map(hl).collect();
    to_after[8] = hl(&wl2);
    let to_check = to_after.clone();
    branch(
        hp,
        z,
        to_after,
        after_hp.clone(),
        to_check,
        check_end.clone(),
    );
    let ce = block_vars(&check_end);
    let ge = emit_bin(
        &check_end,
        "int_ge",
        hl(&ce[8]),
        hl(&ce[7]),
        "ge",
        LowLevelType::Bool,
    );
    branch(
        &check_end,
        ge,
        vec![signed_const(-1)],
        returnblock.clone(),
        pcarry(&ce),
        hp.clone(),
    );
    let _ = shift;
    Ok(())
}

fn build_two_way_nonperiodic(
    kind: &StringKind,
    table_getitem: &Hlvalue,
    np_init: &BlockRef,
    returnblock: BlockRef,
) -> Result<(), TyperError> {
    let pin = block_vars(np_init);
    // pin: value, needle, table, base, m, cut, period, gap, haystack_end, window_last
    let per_lt_gap = emit_bin(
        np_init,
        "int_lt",
        hl(&pin[6]),
        hl(&pin[7]),
        "lt",
        LowLevelType::Bool,
    );
    let use_gap = clone_block_like(&pin);
    let keep_per = clone_block_like(&pin);
    let mut to_gap = pcarry(&pin);
    to_gap[6] = hl(&pin[7]);
    branch(
        np_init,
        per_lt_gap,
        to_gap,
        use_gap.clone(),
        pcarry(&pin),
        keep_per.clone(),
    );
    let gje_join = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("gap", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("gap_jump_end", LowLevelType::Signed),
    ]);
    let enter_gje = |block: &BlockRef| {
        let ins = block_vars(block);
        let gje = emit_bin(
            block,
            "int_add",
            hl(&ins[5]),
            hl(&ins[7]),
            "gje",
            LowLevelType::Signed,
        );
        let gt_m = emit_bin(
            block,
            "int_gt",
            hl(&gje),
            hl(&ins[4]),
            "gt",
            LowLevelType::Bool,
        );
        let take_m = clone_block_like(&block_vars(&gje_join));
        let take_g = clone_block_like(&block_vars(&gje_join));
        let pack = |gje_v: Hlvalue| {
            vec![
                hl(&ins[0]),
                hl(&ins[1]),
                hl(&ins[2]),
                hl(&ins[3]),
                hl(&ins[4]),
                hl(&ins[5]),
                hl(&ins[6]),
                hl(&ins[7]),
                hl(&ins[8]),
                hl(&ins[9]),
                gje_v,
            ]
        };
        branch(
            block,
            gt_m,
            pack(hl(&ins[4])),
            take_m.clone(),
            pack(hl(&gje)),
            take_g.clone(),
        );
        goto(&take_m, pcarry(&block_vars(&take_m)), gje_join.clone());
        goto(&take_g, pcarry(&block_vars(&take_g)), gje_join.clone());
    };
    enter_gje(&use_gap);
    enter_gje(&keep_per);

    let loop_cond = clone_block_like(&block_vars(&gje_join));
    goto(&gje_join, pcarry(&block_vars(&gje_join)), loop_cond.clone());
    let lc = block_vars(&loop_cond);
    let still = emit_bin(
        &loop_cond,
        "int_lt",
        hl(&lc[9]),
        hl(&lc[8]),
        "still",
        LowLevelType::Bool,
    );
    let hp = clone_block_like(&lc);
    branch(
        &loop_cond,
        still,
        pcarry(&lc),
        hp.clone(),
        vec![signed_const(-1)],
        returnblock.clone(),
    );
    let _after_hp = clone_block_like(&lc);
    // horspool uses window_last at [8] and haystack_end at [7]. Remap:
    // current: 0v 1n 2t 3b 4m 5cut 6per 7gap 8he 9wl 10gje
    // horspool wants he at 7, wl at 8. Shuffle for the skip loop.
    let hp_shuf = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("gap", LowLevelType::Signed),
        v("gap_jump_end", LowLevelType::Signed),
    ]);
    let hps = block_vars(&hp);
    goto(
        &hp,
        vec![
            hl(&hps[0]),
            hl(&hps[1]),
            hl(&hps[2]),
            hl(&hps[3]),
            hl(&hps[4]),
            hl(&hps[5]),
            hl(&hps[6]),
            hl(&hps[8]),
            hl(&hps[9]),
            hl(&hps[7]),
            hl(&hps[10]),
        ],
        hp_shuf.clone(),
    );
    let after_shuf = clone_block_like(&block_vars(&hp_shuf));
    emit_horspool_loop(kind, table_getitem, &hp_shuf, &after_shuf, &returnblock, 11)?;
    let ash = block_vars(&after_shuf);
    // unshuffle: 0v 1n 2t 3b 4m 5cut 6per 7he 8wl 9gap 10gje -> original
    let after = clone_block_like(&lc);
    goto(
        &after_shuf,
        vec![
            hl(&ash[0]),
            hl(&ash[1]),
            hl(&ash[2]),
            hl(&ash[3]),
            hl(&ash[4]),
            hl(&ash[5]),
            hl(&ash[6]),
            hl(&ash[9]),
            hl(&ash[7]),
            hl(&ash[8]),
            hl(&ash[10]),
        ],
        after.clone(),
    );
    let ah = block_vars(&after);
    let m1 = emit_bin(
        &after,
        "int_sub",
        hl(&ah[4]),
        signed_const(1),
        "m1",
        LowLevelType::Signed,
    );
    let window = emit_bin(
        &after,
        "int_sub",
        hl(&ah[9]),
        hl(&m1),
        "window",
        LowLevelType::Signed,
    );
    let scan = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("table", table_list_ptr_lltype()),
        v("base", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("gap", LowLevelType::Signed),
        v("haystack_end", LowLevelType::Signed),
        v("window_last", LowLevelType::Signed),
        v("gap_jump_end", LowLevelType::Signed),
        v("window", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ]);
    goto(
        &after,
        vec![
            hl(&ah[0]),
            hl(&ah[1]),
            hl(&ah[2]),
            hl(&ah[3]),
            hl(&ah[4]),
            hl(&ah[5]),
            hl(&ah[6]),
            hl(&ah[7]),
            hl(&ah[8]),
            hl(&ah[9]),
            hl(&ah[10]),
            hl(&window),
            hl(&ah[5]),
        ],
        scan.clone(),
    );

    // early: while i < gap_jump_end
    emit_np_scan(
        kind,
        &scan,
        10, // limit index gap_jump_end
        NpMismatch::Gap,
        &loop_cond,
        &returnblock,
        true,
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
enum NpMismatch {
    Gap,
    Late,
    Left,
}

fn emit_np_mismatch(block: &BlockRef, ins: &[Variable], how: NpMismatch) -> Variable {
    match how {
        NpMismatch::Gap => emit_bin(
            block,
            "int_add",
            hl(&ins[9]),
            hl(&ins[7]),
            "wl",
            LowLevelType::Signed,
        ),
        NpMismatch::Late => {
            let d = emit_bin(
                block,
                "int_sub",
                hl(&ins[12]),
                hl(&ins[5]),
                "d",
                LowLevelType::Signed,
            );
            let d1 = emit_bin(
                block,
                "int_add",
                hl(&d),
                signed_const(1),
                "d1",
                LowLevelType::Signed,
            );
            emit_bin(
                block,
                "int_add",
                hl(&ins[9]),
                hl(&d1),
                "wl",
                LowLevelType::Signed,
            )
        }
        NpMismatch::Left => emit_bin(
            block,
            "int_add",
            hl(&ins[9]),
            hl(&ins[6]),
            "wl",
            LowLevelType::Signed,
        ),
    }
}

/// Non-periodic half scans. `early` is the first of three scans; it
/// chains into the late and left scans on completion.
fn emit_np_scan(
    kind: &StringKind,
    entry: &BlockRef,
    limit_idx: usize,
    on_mismatch: NpMismatch,
    loop_cond: &BlockRef,
    returnblock: &BlockRef,
    is_early: bool,
) -> Result<(), TyperError> {
    let ins0 = block_vars(entry);
    let i_lt = emit_bin(
        entry,
        "int_lt",
        hl(&ins0[12]),
        hl(&ins0[limit_idx]),
        "lt",
        LowLevelType::Bool,
    );
    let body = clone_block_like(&ins0);
    let next = clone_block_like(&ins0);
    branch(
        entry,
        i_lt,
        pcarry(&ins0),
        body.clone(),
        pcarry(&ins0),
        next.clone(),
    );
    let bd = block_vars(&body);
    let pos = emit_bin(
        &body,
        "int_add",
        hl(&bd[11]),
        hl(&bd[12]),
        "pos",
        LowLevelType::Signed,
    );
    let nch = emit_getitem(kind, &body, &bd[1], hl(&bd[12]));
    let vch = emit_getitem(kind, &body, &bd[0], hl(&pos));
    let ne = emit_bin(
        &body,
        kind.ne_op,
        hl(&nch),
        hl(&vch),
        "ne",
        LowLevelType::Bool,
    );
    let mis = clone_block_like(&bd);
    let i_inc = emit_bin(
        &body,
        "int_add",
        hl(&bd[12]),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    let mut cont = pcarry(&bd);
    cont[12] = hl(&i_inc);
    branch(&body, ne, pcarry(&bd), mis.clone(), cont, entry.clone());
    let ms = block_vars(&mis);
    let wl = emit_np_mismatch(&mis, &ms, on_mismatch);
    let back = vec![
        hl(&ms[0]),
        hl(&ms[1]),
        hl(&ms[2]),
        hl(&ms[3]),
        hl(&ms[4]),
        hl(&ms[5]),
        hl(&ms[6]),
        hl(&ms[7]),
        hl(&ms[8]),
        hl(&wl),
        hl(&ms[10]),
    ];
    goto(&mis, back, loop_cond.clone());

    let nx = block_vars(&next);
    if is_early {
        let late = clone_block_like(&nx);
        let mut to_late = pcarry(&nx);
        to_late[12] = hl(&nx[10]); // i = gap_jump_end
        goto(&next, to_late, late.clone());
        emit_np_scan(
            kind,
            &late,
            4, // m
            NpMismatch::Late,
            loop_cond,
            returnblock,
            false,
        )?;
        // after late we need left scan; handled inside the recursive false path
        // by chaining — see below.
    } else if limit_idx == 4 {
        // late complete → left scan i = 0, limit = cut (5)
        let left = clone_block_like(&nx);
        let mut to_left = pcarry(&nx);
        to_left[12] = signed_const(0);
        goto(&next, to_left, left.clone());
        emit_np_scan(
            kind,
            &left,
            5,
            NpMismatch::Left,
            loop_cond,
            returnblock,
            false,
        )?;
    } else {
        // left complete → return window - base
        let res = emit_bin(
            &next,
            "int_sub",
            hl(&nx[11]),
            hl(&nx[3]),
            "res",
            LowLevelType::Signed,
        );
        ret(&next, hl(&res), returnblock.clone());
    }
    Ok(())
}

/// `rstring.py` `_two_way_count`.
pub(crate) fn build_two_way_count_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype.clone())?;
    let value = v("value", kind.ptr_lltype.clone());
    let base = v("base", LowLevelType::Signed);
    let n = v("n", LowLevelType::Signed);
    let needle = v("needle", kind.ptr_lltype.clone());
    let m = v("m", LowLevelType::Signed);
    let cut = v("cut", LowLevelType::Signed);
    let period = v("period", LowLevelType::Signed);
    let gap = v("gap", LowLevelType::Signed);
    let is_periodic = v("is_periodic", LowLevelType::Bool);
    let table = v("table", table_list_ptr_lltype());
    let startblock = blk(vec![
        value.clone(),
        base.clone(),
        n.clone(),
        needle.clone(),
        m.clone(),
        cut.clone(),
        period.clone(),
        gap.clone(),
        is_periodic.clone(),
        table.clone(),
    ]);
    let return_var = v("result", LowLevelType::Signed);
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));
    let loopb = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("base", LowLevelType::Signed),
        v("n", LowLevelType::Signed),
        v("needle", kind.ptr_lltype.clone()),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("gap", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("table", table_list_ptr_lltype()),
        v("index", LowLevelType::Signed),
        v("count", LowLevelType::Signed),
    ]);
    goto(
        &startblock,
        vec![
            hl(&value),
            hl(&base),
            hl(&n),
            hl(&needle),
            hl(&m),
            hl(&cut),
            hl(&period),
            hl(&gap),
            hl(&is_periodic),
            hl(&table),
            signed_const(0),
            signed_const(0),
        ],
        loopb.clone(),
    );
    let ins = block_vars(&loopb);
    let two_way = {
        let ptr = ptr_lltype;
        cached_helper_funcptr(
            rtyper,
            "_two_way",
            two_way_args(&kind),
            LowLevelType::Signed,
            move |rtyper| build_two_way_helper_graph(rtyper, "_two_way", ptr),
        )?
    };
    let base_i = emit_bin(
        &loopb,
        "int_add",
        hl(&ins[1]),
        hl(&ins[10]),
        "base_i",
        LowLevelType::Signed,
    );
    let n_i = emit_bin(
        &loopb,
        "int_sub",
        hl(&ins[2]),
        hl(&ins[10]),
        "n_i",
        LowLevelType::Signed,
    );
    let result = emit_direct_call(
        &loopb,
        two_way,
        vec![
            hl(&ins[0]),
            hl(&base_i),
            hl(&n_i),
            hl(&ins[3]),
            hl(&ins[4]),
            hl(&ins[5]),
            hl(&ins[6]),
            hl(&ins[7]),
            hl(&ins[8]),
            hl(&ins[9]),
        ],
        "result",
        LowLevelType::Signed,
    );
    let is_miss = emit_bin(
        &loopb,
        "int_eq",
        hl(&result),
        signed_const(-1),
        "miss",
        LowLevelType::Bool,
    );
    let cont = blk(vec![
        v("value", kind.ptr_lltype.clone()),
        v("base", LowLevelType::Signed),
        v("n", LowLevelType::Signed),
        v("needle", kind.ptr_lltype.clone()),
        v("m", LowLevelType::Signed),
        v("cut", LowLevelType::Signed),
        v("period", LowLevelType::Signed),
        v("gap", LowLevelType::Signed),
        v("is_periodic", LowLevelType::Bool),
        v("table", table_list_ptr_lltype()),
        v("index", LowLevelType::Signed),
        v("count", LowLevelType::Signed),
        v("result", LowLevelType::Signed),
    ]);
    let mut to_cont = pcarry(&ins);
    to_cont.push(hl(&result));
    branch(
        &loopb,
        is_miss,
        vec![hl(&ins[11])],
        graph.returnblock.clone(),
        to_cont,
        cont.clone(),
    );
    let ct = block_vars(&cont);
    let count1 = emit_bin(
        &cont,
        "int_add",
        hl(&ct[11]),
        signed_const(1),
        "count",
        LowLevelType::Signed,
    );
    let step = emit_bin(
        &cont,
        "int_add",
        hl(&ct[12]),
        hl(&ct[4]),
        "step",
        LowLevelType::Signed,
    );
    let index1 = emit_bin(
        &cont,
        "int_add",
        hl(&ct[10]),
        hl(&step),
        "index",
        LowLevelType::Signed,
    );
    goto(
        &cont,
        vec![
            hl(&ct[0]),
            hl(&ct[1]),
            hl(&ct[2]),
            hl(&ct[3]),
            hl(&ct[4]),
            hl(&ct[5]),
            hl(&ct[6]),
            hl(&ct[7]),
            hl(&ct[8]),
            hl(&ct[9]),
            hl(&index1),
            hl(&count1),
        ],
        loopb.clone(),
    );
    Ok(finish_graph(
        graph,
        name,
        vec![
            "value".into(),
            "base".into(),
            "n".into(),
            "needle".into(),
            "m".into(),
            "cut".into(),
            "period".into(),
            "gap".into(),
            "is_periodic".into(),
            "table".into(),
        ],
        false,
    ))
}

/// `rstring.py` `_default_find`.
pub(crate) fn build_default_find_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_bmh_find_helper_graph(rtyper, name, ptr_lltype, false)
}

/// `rstring.py` `_adaptive_find`. `m / 4` is `int_rshift(m, 2)` so
/// the elidable calldescr stays MemoryError-only (`int_floordiv` can
/// raise ZeroDivisionError).
pub(crate) fn build_adaptive_find_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_bmh_find_helper_graph(rtyper, name, ptr_lltype, true)
}

fn bmh_prep_vars(kind: &StringKind, adaptive: bool) -> Vec<Variable> {
    let mut vars = bmh_search_vars(kind, adaptive);
    vars[13] = v("j", LowLevelType::Signed);
    vars
}

fn bmh_search_vars(kind: &StringKind, adaptive: bool) -> Vec<Variable> {
    let mut vars = vec![
        v("value", kind.ptr_lltype.clone()),
        v("needle", kind.ptr_lltype.clone()),
        v("base", LowLevelType::Signed),
        v("n", LowLevelType::Signed),
        v("m", LowLevelType::Signed),
        v("mode", LowLevelType::Signed),
        v("w", LowLevelType::Signed),
        v("mlast", LowLevelType::Signed),
        v("last", kind.elem_lltype.clone()),
        v("nul", kind.elem_lltype.clone()),
        v("mask", LowLevelType::Signed),
        v("gap", LowLevelType::Signed),
        v("count", LowLevelType::Signed),
        v("i", LowLevelType::Signed),
    ];
    if adaptive {
        vars.push(v("hits", LowLevelType::Signed));
    }
    vars
}

fn goto_search(block: &BlockRef, ins: &[Variable], i: Variable, search: &BlockRef, adaptive: bool) {
    let mut args = vec![
        hl(&ins[0]),
        hl(&ins[1]),
        hl(&ins[2]),
        hl(&ins[3]),
        hl(&ins[4]),
        hl(&ins[5]),
        hl(&ins[6]),
        hl(&ins[7]),
        hl(&ins[8]),
        hl(&ins[9]),
        hl(&ins[10]),
        hl(&ins[11]),
        hl(&ins[12]),
        hl(&i),
    ];
    if adaptive {
        args.push(hl(&ins[14]));
    }
    goto(block, args, search.clone());
}

fn emit_bloom_skip(
    kind: &StringKind,
    bloom: &Hlvalue,
    block: &BlockRef,
    search: &BlockRef,
    use_gap: bool,
    adaptive: bool,
) {
    let ins = block_vars(block);
    let ncarry = ins.len();
    let t0 = emit_bin(
        block,
        "int_add",
        hl(&ins[2]),
        hl(&ins[7]),
        "t",
        LowLevelType::Signed,
    );
    let t1 = emit_bin(
        block,
        "int_add",
        hl(&t0),
        hl(&ins[13]),
        "t",
        LowLevelType::Signed,
    );
    let la = emit_bin(
        block,
        "int_add",
        hl(&t1),
        signed_const(1),
        "la",
        LowLevelType::Signed,
    );
    let nval = emit_len(kind, block, &ins[0]);
    let in_range = emit_bin(
        block,
        "int_lt",
        hl(&la),
        hl(&nval),
        "in_range",
        LowLevelType::Bool,
    );
    let mut take_c_vars: Vec<Variable> = ins.iter().map(Variable::copy).collect();
    take_c_vars.push(v("la", LowLevelType::Signed));
    let take_c = blk(take_c_vars);
    let take_nul = clone_block_like(&ins);
    let mut after_vars: Vec<Variable> = ins.iter().map(Variable::copy).collect();
    after_vars.push(v("c", kind.elem_lltype.clone()));
    let after = blk(after_vars);
    let mut true_args = pcarry(&ins);
    true_args.push(hl(&la));
    branch(
        block,
        in_range,
        true_args,
        take_c.clone(),
        pcarry(&ins),
        take_nul.clone(),
    );
    let tc = block_vars(&take_c);
    let ch = emit_getitem(kind, &take_c, &tc[0], hl(&tc[ncarry]));
    let mut to_after_c = pcarry(&tc[..ncarry]);
    to_after_c.push(hl(&ch));
    goto(&take_c, to_after_c, after.clone());
    let tn = block_vars(&take_nul);
    let mut to_after_n = pcarry(&tn);
    to_after_n.push(hl(&tn[9]));
    goto(&take_nul, to_after_n, after.clone());
    let af = block_vars(&after);
    let br = emit_direct_call(
        &after,
        bloom.clone(),
        vec![hl(&af[10]), hl(&af[ncarry])],
        "br",
        LowLevelType::Signed,
    );
    let zero = emit_un(&after, "int_is_zero", hl(&br), "z", LowLevelType::Bool);
    let add_m = clone_block_like(&ins);
    let bump = clone_block_like(&ins);
    if use_gap {
        let add_gap = clone_block_like(&ins);
        branch(
            &after,
            zero,
            pcarry(&af[..ncarry]),
            add_m.clone(),
            pcarry(&af[..ncarry]),
            add_gap.clone(),
        );
        let ag = block_vars(&add_gap);
        let i_g = emit_bin(
            &add_gap,
            "int_add",
            hl(&ag[13]),
            hl(&ag[11]),
            "i",
            LowLevelType::Signed,
        );
        goto_search(&add_gap, &ag, i_g, &bump, adaptive);
    } else {
        branch(
            &after,
            zero,
            pcarry(&af[..ncarry]),
            add_m.clone(),
            pcarry(&af[..ncarry]),
            bump.clone(),
        );
    }
    let am = block_vars(&add_m);
    let i_m = emit_bin(
        &add_m,
        "int_add",
        hl(&am[13]),
        hl(&am[4]),
        "i",
        LowLevelType::Signed,
    );
    goto_search(&add_m, &am, i_m, &bump, adaptive);
    let bp = block_vars(&bump);
    let i_n = emit_bin(
        &bump,
        "int_add",
        hl(&bp[13]),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    goto_search(&bump, &bp, i_n, search, adaptive);
}

fn build_bmh_find_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    adaptive: bool,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype.clone())?;
    let value = v("value", kind.ptr_lltype.clone());
    let base = v("base", LowLevelType::Signed);
    let n = v("n", LowLevelType::Signed);
    let needle = v("needle", kind.ptr_lltype.clone());
    let m = v("m", LowLevelType::Signed);
    let mode = v("mode", LowLevelType::Signed);
    let startblock = blk(vec![
        value.clone(),
        base.clone(),
        n.clone(),
        needle.clone(),
        m.clone(),
        mode.clone(),
    ]);
    let return_var = v("result", LowLevelType::Signed);
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));
    let bloom_add = funcptr_bloom_add(rtyper, &kind)?;
    let bloom = funcptr_bloom(rtyper, &kind)?;
    let nul = emit_nul(&kind, &startblock);
    let w = emit_bin(
        &startblock,
        "int_sub",
        hl(&n),
        hl(&m),
        "w",
        LowLevelType::Signed,
    );
    let mlast = emit_bin(
        &startblock,
        "int_sub",
        hl(&m),
        signed_const(1),
        "mlast",
        LowLevelType::Signed,
    );
    let last = emit_getitem(&kind, &startblock, &needle, hl(&mlast));
    let prep = blk(bmh_prep_vars(&kind, adaptive));
    let mut prep_args = vec![
        hl(&value),
        hl(&needle),
        hl(&base),
        hl(&n),
        hl(&m),
        hl(&mode),
        hl(&w),
        hl(&mlast),
        hl(&last),
        hl(&nul),
        signed_const(0),
        hl(&mlast),
        signed_const(0),
        signed_const(0),
    ];
    if adaptive {
        prep_args.push(signed_const(0));
    }
    goto(&startblock, prep_args, prep.clone());

    let pv = block_vars(&prep);
    let j_lt = emit_bin(
        &prep,
        "int_lt",
        hl(&pv[13]),
        hl(&pv[7]),
        "j_lt",
        LowLevelType::Bool,
    );
    let pbody = clone_block_like(&pv);
    let after_prep = clone_block_like(&pv);
    branch(
        &prep,
        j_lt,
        pcarry(&pv),
        pbody.clone(),
        pcarry(&pv),
        after_prep.clone(),
    );
    let pb = block_vars(&pbody);
    let ch = emit_getitem(&kind, &pbody, &pb[1], hl(&pb[13]));
    let mask1 = emit_direct_call(
        &pbody,
        bloom_add.clone(),
        vec![hl(&pb[10]), hl(&ch)],
        "mask",
        LowLevelType::Signed,
    );
    let is_last = emit_bin(
        &pbody,
        kind.eq_op,
        hl(&ch),
        hl(&pb[8]),
        "is_last",
        LowLevelType::Bool,
    );
    let gap_upd = blk({
        let mut vars: Vec<Variable> = pb.iter().map(Variable::copy).collect();
        vars[10] = v("mask", LowLevelType::Signed);
        vars
    });
    let gap_keep = clone_block_like(&block_vars(&gap_upd));
    let t = emit_bin(
        &pbody,
        "int_sub",
        hl(&pb[7]),
        hl(&pb[13]),
        "t",
        LowLevelType::Signed,
    );
    let new_gap = emit_bin(
        &pbody,
        "int_sub",
        hl(&t),
        signed_const(1),
        "gap",
        LowLevelType::Signed,
    );
    let mut to_upd = pcarry(&pb);
    to_upd[10] = hl(&mask1);
    let to_keep = to_upd.clone();
    to_upd[11] = hl(&new_gap);
    branch(
        &pbody,
        is_last,
        to_upd,
        gap_upd.clone(),
        to_keep,
        gap_keep.clone(),
    );
    let continue_prep = |block: &BlockRef| {
        let ins = block_vars(block);
        let j1 = emit_bin(
            block,
            "int_add",
            hl(&ins[13]),
            signed_const(1),
            "j",
            LowLevelType::Signed,
        );
        let mut args = pcarry(&ins);
        args[13] = hl(&j1);
        goto(block, args, prep.clone());
    };
    continue_prep(&gap_upd);
    continue_prep(&gap_keep);

    let ap = block_vars(&after_prep);
    let mask2 = emit_direct_call(
        &after_prep,
        bloom_add,
        vec![hl(&ap[10]), hl(&ap[8])],
        "mask",
        LowLevelType::Signed,
    );
    let search = blk(bmh_search_vars(&kind, adaptive));
    let mut search_args = pcarry(&ap);
    search_args[10] = hl(&mask2);
    search_args[13] = signed_const(0);
    goto(&after_prep, search_args, search.clone());

    let sv = block_vars(&search);
    let i_le = emit_bin(
        &search,
        "int_le",
        hl(&sv[13]),
        hl(&sv[6]),
        "i_le",
        LowLevelType::Bool,
    );
    let sbody = clone_block_like(&sv);
    let done = clone_block_like(&sv);
    branch(
        &search,
        i_le,
        pcarry(&sv),
        sbody.clone(),
        pcarry(&sv),
        done.clone(),
    );

    let sb = block_vars(&sbody);
    let idx0 = emit_bin(
        &sbody,
        "int_add",
        hl(&sb[2]),
        hl(&sb[7]),
        "t",
        LowLevelType::Signed,
    );
    let idx = emit_bin(
        &sbody,
        "int_add",
        hl(&idx0),
        hl(&sb[13]),
        "idx",
        LowLevelType::Signed,
    );
    let hay = emit_getitem(&kind, &sbody, &sb[0], hl(&idx));
    let eq_last = emit_bin(
        &sbody,
        kind.eq_op,
        hl(&hay),
        hl(&sb[8]),
        "eq_last",
        LowLevelType::Bool,
    );
    let mut inner_vars: Vec<Variable> = sb.iter().map(Variable::copy).collect();
    inner_vars.push(v("j", LowLevelType::Signed));
    let inner = blk(inner_vars);
    let skip_no_gap = clone_block_like(&sb);
    let mut inner_args = pcarry(&sb);
    inner_args.push(signed_const(0));
    branch(
        &sbody,
        eq_last,
        inner_args,
        inner.clone(),
        pcarry(&sb),
        skip_no_gap.clone(),
    );
    emit_bloom_skip(&kind, &bloom, &skip_no_gap, &search, false, adaptive);

    let iv = block_vars(&inner);
    let j_lt_m = emit_bin(
        &inner,
        "int_lt",
        hl(&iv[iv.len() - 1]),
        hl(&iv[7]),
        "j_lt",
        LowLevelType::Bool,
    );
    let ibody = clone_block_like(&iv);
    let full = clone_block_like(&sb);
    branch(
        &inner,
        j_lt_m,
        pcarry(&iv),
        ibody.clone(),
        pcarry(&iv[..sb.len()]),
        full.clone(),
    );
    let ib = block_vars(&ibody);
    let jv = &ib[ib.len() - 1];
    let pos0 = emit_bin(
        &ibody,
        "int_add",
        hl(&ib[2]),
        hl(&ib[13]),
        "t",
        LowLevelType::Signed,
    );
    let pos = emit_bin(
        &ibody,
        "int_add",
        hl(&pos0),
        hl(jv),
        "pos",
        LowLevelType::Signed,
    );
    let hv = emit_getitem(&kind, &ibody, &ib[0], hl(&pos));
    let nd = emit_getitem(&kind, &ibody, &ib[1], hl(jv));
    let ne = emit_bin(
        &ibody,
        kind.ne_op,
        hl(&hv),
        hl(&nd),
        "ne",
        LowLevelType::Bool,
    );
    let mismatch = clone_block_like(&iv);
    let j_inc_b = clone_block_like(&iv);
    branch(
        &ibody,
        ne,
        pcarry(&ib),
        mismatch.clone(),
        pcarry(&ib),
        j_inc_b.clone(),
    );
    let ji = block_vars(&j_inc_b);
    let j1 = emit_bin(
        &j_inc_b,
        "int_add",
        hl(&ji[ji.len() - 1]),
        signed_const(1),
        "j",
        LowLevelType::Signed,
    );
    let mut back = pcarry(&ji);
    *back.last_mut().unwrap() = hl(&j1);
    goto(&j_inc_b, back, inner.clone());

    if adaptive {
        emit_adaptive_mismatch(
            rtyper,
            &kind,
            &bloom,
            &mismatch,
            &search,
            graph.returnblock.clone(),
        )?;
    } else {
        let ms = block_vars(&mismatch);
        let skip = clone_block_like(&sb);
        goto(&mismatch, pcarry(&ms[..sb.len()]), skip.clone());
        emit_bloom_skip(&kind, &bloom, &skip, &search, true, false);
    }

    let fv = block_vars(&full);
    let not_count = emit_bin(
        &full,
        "int_ne",
        hl(&fv[5]),
        signed_const(SEARCH_COUNT),
        "not_count",
        LowLevelType::Bool,
    );
    let ret_i = clone_block_like(&fv);
    let count_hit = clone_block_like(&fv);
    branch(
        &full,
        not_count,
        pcarry(&fv),
        ret_i.clone(),
        pcarry(&fv),
        count_hit.clone(),
    );
    let ri = block_vars(&ret_i);
    ret(&ret_i, hl(&ri[13]), graph.returnblock.clone());
    let chit = block_vars(&count_hit);
    let count1 = emit_bin(
        &count_hit,
        "int_add",
        hl(&chit[12]),
        signed_const(1),
        "count",
        LowLevelType::Signed,
    );
    let i_mlast = emit_bin(
        &count_hit,
        "int_add",
        hl(&chit[13]),
        hl(&chit[7]),
        "i",
        LowLevelType::Signed,
    );
    let i_next = emit_bin(
        &count_hit,
        "int_add",
        hl(&i_mlast),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    let mut cargs = pcarry(&chit);
    cargs[12] = hl(&count1);
    cargs[13] = hl(&i_next);
    goto(&count_hit, cargs, search.clone());

    let dv = block_vars(&done);
    let not_count_d = emit_bin(
        &done,
        "int_ne",
        hl(&dv[5]),
        signed_const(SEARCH_COUNT),
        "not_count",
        LowLevelType::Bool,
    );
    let ret_m1 = clone_block_like(&dv);
    let ret_c = clone_block_like(&dv);
    branch(
        &done,
        not_count_d,
        pcarry(&dv),
        ret_m1.clone(),
        pcarry(&dv),
        ret_c.clone(),
    );
    ret(&ret_m1, signed_const(-1), graph.returnblock.clone());
    let rc = block_vars(&ret_c);
    ret(&ret_c, hl(&rc[12]), graph.returnblock.clone());

    Ok(finish_graph(
        graph,
        name,
        vec![
            "value".into(),
            "base".into(),
            "n".into(),
            "needle".into(),
            "m".into(),
            "mode".into(),
        ],
        false,
    ))
}

fn emit_adaptive_mismatch(
    rtyper: &RPythonTyper,
    kind: &StringKind,
    bloom: &Hlvalue,
    mismatch: &BlockRef,
    search: &BlockRef,
    returnblock: BlockRef,
) -> Result<(), TyperError> {
    let ms = block_vars(mismatch);
    let nsearch = ms.len() - 1;
    let hits1 = emit_bin(
        mismatch,
        "int_add",
        hl(&ms[14]),
        hl(&ms[nsearch]),
        "t",
        LowLevelType::Signed,
    );
    let hits1 = emit_bin(
        mismatch,
        "int_add",
        hl(&hits1),
        signed_const(1),
        "hits",
        LowLevelType::Signed,
    );
    let m4 = emit_bin(
        mismatch,
        "int_rshift",
        hl(&ms[4]),
        signed_const(2),
        "m4",
        LowLevelType::Signed,
    );
    let hits_gt = emit_bin(
        mismatch,
        "int_gt",
        hl(&hits1),
        hl(&m4),
        "hits_gt",
        LowLevelType::Bool,
    );
    let check_far = clone_block_like(&ms[..nsearch]);
    let skip = clone_block_like(&ms[..nsearch]);
    let mut to_check = pcarry(&ms[..nsearch]);
    to_check[14] = hl(&hits1);
    let to_skip = to_check.clone();
    branch(
        mismatch,
        hits_gt,
        to_check,
        check_far.clone(),
        to_skip,
        skip.clone(),
    );
    let cf = block_vars(&check_far);
    let wi = emit_bin(
        &check_far,
        "int_sub",
        hl(&cf[6]),
        hl(&cf[13]),
        "wi",
        LowLevelType::Signed,
    );
    let far = emit_bin(
        &check_far,
        "int_gt",
        hl(&wi),
        signed_const(2000),
        "far",
        LowLevelType::Bool,
    );
    let switch_b = clone_block_like(&cf);
    branch(
        &check_far,
        far,
        pcarry(&cf),
        switch_b.clone(),
        pcarry(&cf),
        skip.clone(),
    );
    emit_adaptive_switch_to_twoway(rtyper, kind, &switch_b, returnblock)?;
    emit_bloom_skip(kind, bloom, &skip, search, true, true);
    Ok(())
}

fn with_twoway_fields(base: &[Variable]) -> Vec<Variable> {
    let mut vars: Vec<Variable> = base.iter().map(Variable::copy).collect();
    vars.push(v("cut", LowLevelType::Signed));
    vars.push(v("period", LowLevelType::Signed));
    vars.push(v("gap", LowLevelType::Signed));
    vars.push(v("is_periodic", LowLevelType::Bool));
    vars.push(v("table", table_list_ptr_lltype()));
    vars
}

fn emit_adaptive_switch_to_twoway(
    rtyper: &RPythonTyper,
    kind: &StringKind,
    block: &BlockRef,
    returnblock: BlockRef,
) -> Result<(), TyperError> {
    let ins = block_vars(block);
    let n = ins.len();
    let prep = funcptr_twoway_preprocess(rtyper, kind)?;
    let two = funcptr_two_way(rtyper, kind)?;
    let two_count = funcptr_two_way_count(rtyper, kind)?;
    let tup = emit_direct_call(
        block,
        prep,
        vec![hl(&ins[1]), hl(&ins[4])],
        "tup",
        tuple5_ptr_lltype(),
    );
    let (cut, period, gap, is_periodic, table) = unpack_tuple5(block, &tup);
    let not_count = emit_bin(
        block,
        "int_ne",
        hl(&ins[5]),
        signed_const(SEARCH_COUNT),
        "not_count",
        LowLevelType::Bool,
    );
    let find_b = blk(with_twoway_fields(&ins));
    let count_b = blk(with_twoway_fields(&ins));
    let mut args = pcarry(&ins);
    args.extend([
        hl(&cut),
        hl(&period),
        hl(&gap),
        hl(&is_periodic),
        hl(&table),
    ]);
    branch(
        block,
        not_count,
        args.clone(),
        find_b.clone(),
        args,
        count_b.clone(),
    );

    let fb = block_vars(&find_b);
    let base_i = emit_bin(
        &find_b,
        "int_add",
        hl(&fb[2]),
        hl(&fb[13]),
        "base_i",
        LowLevelType::Signed,
    );
    let n_i = emit_bin(
        &find_b,
        "int_sub",
        hl(&fb[3]),
        hl(&fb[13]),
        "n_i",
        LowLevelType::Signed,
    );
    let res = emit_direct_call(
        &find_b,
        two,
        vec![
            hl(&fb[0]),
            hl(&base_i),
            hl(&n_i),
            hl(&fb[1]),
            hl(&fb[4]),
            hl(&fb[n]),
            hl(&fb[n + 1]),
            hl(&fb[n + 2]),
            hl(&fb[n + 3]),
            hl(&fb[n + 4]),
        ],
        "res",
        LowLevelType::Signed,
    );
    let is_m1 = emit_bin(
        &find_b,
        "int_eq",
        hl(&res),
        signed_const(-1),
        "is_m1",
        LowLevelType::Bool,
    );
    let miss = clone_block_like(&ins);
    let mut hit_vars: Vec<Variable> = ins.iter().map(Variable::copy).collect();
    hit_vars.push(v("res", LowLevelType::Signed));
    let hit = blk(hit_vars);
    let mut hit_args = pcarry(&fb[..n]);
    hit_args.push(hl(&res));
    branch(
        &find_b,
        is_m1,
        pcarry(&fb[..n]),
        miss.clone(),
        hit_args,
        hit.clone(),
    );
    ret(&miss, signed_const(-1), returnblock.clone());
    let hv = block_vars(&hit);
    let pos = emit_bin(
        &hit,
        "int_add",
        hl(&hv[n]),
        hl(&hv[13]),
        "pos",
        LowLevelType::Signed,
    );
    ret(&hit, hl(&pos), returnblock.clone());

    let cb = block_vars(&count_b);
    let base_i = emit_bin(
        &count_b,
        "int_add",
        hl(&cb[2]),
        hl(&cb[13]),
        "base_i",
        LowLevelType::Signed,
    );
    let n_i = emit_bin(
        &count_b,
        "int_sub",
        hl(&cb[3]),
        hl(&cb[13]),
        "n_i",
        LowLevelType::Signed,
    );
    let res = emit_direct_call(
        &count_b,
        two_count,
        vec![
            hl(&cb[0]),
            hl(&base_i),
            hl(&n_i),
            hl(&cb[1]),
            hl(&cb[4]),
            hl(&cb[n]),
            hl(&cb[n + 1]),
            hl(&cb[n + 2]),
            hl(&cb[n + 3]),
            hl(&cb[n + 4]),
        ],
        "res",
        LowLevelType::Signed,
    );
    let total = emit_bin(
        &count_b,
        "int_add",
        hl(&res),
        hl(&cb[12]),
        "total",
        LowLevelType::Signed,
    );
    ret(&count_b, hl(&total), returnblock);
    Ok(())
}

/// `rstring.py` `_search` — translation-time `isinstance` dispatch.
///
/// `value_can_be_none` is the caller's `SomeString.can_be_None`, forwarded
/// to `_search_elidable` / `_search_normal`.
pub(crate) fn build_search_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    value_can_be_none: bool,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype)?;
    let value = v("value", kind.ptr_lltype.clone());
    let other = v("other", kind.ptr_lltype.clone());
    let start = v("start", LowLevelType::Signed);
    let end = v("end", LowLevelType::Signed);
    let mode = v("mode", LowLevelType::Signed);
    let startblock = blk(vec![
        value.clone(),
        other.clone(),
        start.clone(),
        end.clone(),
        mode.clone(),
    ]);
    let return_var = v("result", LowLevelType::Signed);
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));
    let callee = if kind.is_unicode() {
        funcptr_search_normal(rtyper, &kind, value_can_be_none)?
    } else {
        funcptr_search_elidable(rtyper, &kind, value_can_be_none)?
    };
    let res = emit_direct_call(
        &startblock,
        callee,
        vec![hl(&value), hl(&other), hl(&start), hl(&end), hl(&mode)],
        "res",
        LowLevelType::Signed,
    );
    ret(&startblock, hl(&res), graph.returnblock.clone());
    Ok(finish_graph(
        graph,
        name,
        vec![
            "value".into(),
            "other".into(),
            "start".into(),
            "end".into(),
            "mode".into(),
        ],
        false,
    ))
}

/// `rstring.py` `_search_elidable`. `@jit.elidable` stamps
/// `_elidable_function_`; the body is `direct_call _search_normal`.
///
/// `value_can_be_none` is forwarded to `_search_normal`.
pub(crate) fn build_search_elidable_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    value_can_be_none: bool,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype)?;
    let value = v("value", kind.ptr_lltype.clone());
    let other = v("other", kind.ptr_lltype.clone());
    let start = v("start", LowLevelType::Signed);
    let end = v("end", LowLevelType::Signed);
    let mode = v("mode", LowLevelType::Signed);
    let startblock = blk(vec![
        value.clone(),
        other.clone(),
        start.clone(),
        end.clone(),
        mode.clone(),
    ]);
    let return_var = v("result", LowLevelType::Signed);
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));
    let callee = funcptr_search_normal(rtyper, &kind, value_can_be_none)?;
    let res = emit_direct_call(
        &startblock,
        callee,
        vec![hl(&value), hl(&other), hl(&start), hl(&end), hl(&mode)],
        "res",
        LowLevelType::Signed,
    );
    ret(&startblock, hl(&res), graph.returnblock.clone());
    Ok(finish_graph(
        graph,
        name,
        vec![
            "value".into(),
            "other".into(),
            "start".into(),
            "end".into(),
            "mode".into(),
        ],
        true,
    ))
}

/// `rstring.py` `_search_normal`.
///
/// `assert value is not None` is emitted only when `value_can_be_none`.
/// After `LLHelpers.ll_search` asserts, knowntypedata has
/// `can_be_None=False`, so `transform_dead_code` (`transform.py`) cuts
/// the raise link; that route passes `false`.
pub(crate) fn build_search_normal_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    value_can_be_none: bool,
) -> Result<PyGraph, TyperError> {
    let kind = StringKind::from_ptr(ptr_lltype.clone())?;
    let value = v("value", kind.ptr_lltype.clone());
    let other = v("other", kind.ptr_lltype.clone());
    let start = v("start", LowLevelType::Signed);
    let end = v("end", LowLevelType::Signed);
    let mode = v("mode", LowLevelType::Signed);
    let startblock = blk(vec![
        value.clone(),
        other.clone(),
        start.clone(),
        end.clone(),
        mode.clone(),
    ]);
    let return_var = v("result", LowLevelType::Signed);
    let graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), hl(&return_var));
    let returnblock = graph.returnblock.clone();

    let voem = || {
        vec![
            v("value", kind.ptr_lltype.clone()),
            v("other", kind.ptr_lltype.clone()),
            v("start", LowLevelType::Signed),
            v("end", LowLevelType::Signed),
            v("mode", LowLevelType::Signed),
        ]
    };
    // `rstring.py` `_search_normal`: `assert value is not None`, cut
    // when the caller annotation has `can_be_None=False`.
    let body = if value_can_be_none {
        let after_none = blk(voem());
        close_assert_ptr_not_none(
            &startblock,
            &value,
            vec![hl(&value), hl(&other), hl(&start), hl(&end), hl(&mode)],
            after_none.clone(),
            graph.exceptblock.clone(),
            exception_args("AssertionError")?,
        );
        after_none
    } else {
        startblock.clone()
    };
    let clamp_s = blk(voem());
    let an = block_vars(&body);
    let start_neg = emit_bin(
        &body,
        "int_lt",
        hl(&an[2]),
        signed_const(0),
        "start_neg",
        LowLevelType::Bool,
    );
    branch(
        &body,
        start_neg,
        vec![
            hl(&an[0]),
            hl(&an[1]),
            signed_const(0),
            hl(&an[3]),
            hl(&an[4]),
        ],
        clamp_s.clone(),
        vec![hl(&an[0]), hl(&an[1]), hl(&an[2]), hl(&an[3]), hl(&an[4])],
        clamp_s.clone(),
    );

    let cs = block_vars(&clamp_s);
    let nfull = emit_len(&kind, &clamp_s, &cs[0]);
    let end_big = emit_bin(
        &clamp_s,
        "int_gt",
        hl(&cs[3]),
        hl(&nfull),
        "end_big",
        LowLevelType::Bool,
    );
    let clamp_e = blk(voem());
    branch(
        &clamp_s,
        end_big,
        vec![hl(&cs[0]), hl(&cs[1]), hl(&cs[2]), hl(&nfull), hl(&cs[4])],
        clamp_e.clone(),
        pcarry(&cs),
        clamp_e.clone(),
    );

    let ce = block_vars(&clamp_e);
    let start_gt = emit_bin(
        &clamp_e,
        "int_gt",
        hl(&ce[2]),
        hl(&ce[3]),
        "start_gt",
        LowLevelType::Bool,
    );
    let empty = blk(voem());
    let nonempty = blk(voem());
    branch(
        &clamp_e,
        start_gt,
        pcarry(&ce),
        empty.clone(),
        pcarry(&ce),
        nonempty.clone(),
    );
    emit_count_or_m1(&empty, returnblock.clone());

    let ne = block_vars(&nonempty);
    let n = emit_bin(
        &nonempty,
        "int_sub",
        hl(&ne[3]),
        hl(&ne[2]),
        "n",
        LowLevelType::Signed,
    );
    let m = emit_len(&kind, &nonempty, &ne[1]);
    let m_eq_0 = emit_bin(
        &nonempty,
        "int_eq",
        hl(&m),
        signed_const(0),
        "m_eq_0",
        LowLevelType::Bool,
    );
    let mut nm_vars = voem();
    nm_vars.push(v("n", LowLevelType::Signed));
    nm_vars.push(v("m", LowLevelType::Signed));
    // `checkgraph` `definevar` (`flowspace/model.py`): each block's
    // inputargs are a fresh Variable, even when names and lltypes match.
    let empty_m = clone_block_like(&nm_vars);
    let has_m = clone_block_like(&nm_vars);
    let mut nm_args = pcarry(&ne);
    nm_args.push(hl(&n));
    nm_args.push(hl(&m));
    branch(
        &nonempty,
        m_eq_0,
        nm_args.clone(),
        empty_m.clone(),
        nm_args,
        has_m.clone(),
    );
    emit_empty_needle(&empty_m, returnblock.clone());

    let hm = block_vars(&has_m);
    let w = emit_bin(
        &has_m,
        "int_sub",
        hl(&hm[5]),
        hl(&hm[6]),
        "w",
        LowLevelType::Signed,
    );
    let w_neg = emit_bin(
        &has_m,
        "int_lt",
        hl(&w),
        signed_const(0),
        "w_neg",
        LowLevelType::Bool,
    );
    let too_small = clone_block_like(&nm_vars);
    let mut sized_vars: Vec<Variable> = nm_vars.iter().map(Variable::copy).collect();
    sized_vars.push(v("w", LowLevelType::Signed));
    let sized = blk(sized_vars);
    branch(
        &has_m,
        w_neg,
        pcarry(&hm),
        too_small.clone(),
        {
            let mut a = pcarry(&hm);
            a.push(hl(&w));
            a
        },
        sized.clone(),
    );
    emit_count_or_m1(&too_small, returnblock.clone());

    let sz = block_vars(&sized);
    let not_rfind = emit_bin(
        &sized,
        "int_ne",
        hl(&sz[4]),
        signed_const(SEARCH_RFIND),
        "not_rfind",
        LowLevelType::Bool,
    );
    let forward = clone_block_like(&sz);
    let rfind_b = clone_block_like(&sz);
    branch(
        &sized,
        not_rfind,
        pcarry(&sz),
        forward.clone(),
        pcarry(&sz),
        rfind_b.clone(),
    );
    emit_forward_search(rtyper, &kind, &forward, returnblock.clone())?;
    // Reverse BMH is inline (`rstring.py` `_search_normal` after
    // `mode != SEARCH_RFIND`). The graph contains a loop, so
    // `look_inside_graph` refuses it (`unsafe_loopy_graphs`).
    emit_rfind_search(rtyper, &kind, &rfind_b, returnblock)?;
    Ok(finish_graph(
        graph,
        name,
        vec![
            "value".into(),
            "other".into(),
            "start".into(),
            "end".into(),
            "mode".into(),
        ],
        false,
    ))
}

fn emit_count_or_m1(block: &BlockRef, returnblock: BlockRef) {
    let ins = block_vars(block);
    let is_count = emit_bin(
        block,
        "int_eq",
        hl(&ins[4]),
        signed_const(SEARCH_COUNT),
        "is_count",
        LowLevelType::Bool,
    );
    let z = clone_block_like(&ins);
    let m1 = clone_block_like(&ins);
    branch(
        block,
        is_count,
        pcarry(&ins),
        z.clone(),
        pcarry(&ins),
        m1.clone(),
    );
    ret(&z, signed_const(0), returnblock.clone());
    ret(&m1, signed_const(-1), returnblock);
}

fn emit_empty_needle(block: &BlockRef, returnblock: BlockRef) {
    let ins = block_vars(block);
    let is_count = emit_bin(
        block,
        "int_eq",
        hl(&ins[4]),
        signed_const(SEARCH_COUNT),
        "is_count",
        LowLevelType::Bool,
    );
    let count_b = clone_block_like(&ins);
    let rest = clone_block_like(&ins);
    branch(
        block,
        is_count,
        pcarry(&ins),
        count_b.clone(),
        pcarry(&ins),
        rest.clone(),
    );
    let cb = block_vars(&count_b);
    let span = emit_bin(
        &count_b,
        "int_sub",
        hl(&cb[3]),
        hl(&cb[2]),
        "span",
        LowLevelType::Signed,
    );
    let npos = emit_bin(
        &count_b,
        "int_add",
        hl(&span),
        signed_const(1),
        "npos",
        LowLevelType::Signed,
    );
    ret(&count_b, hl(&npos), returnblock.clone());
    let rs = block_vars(&rest);
    let is_rfind = emit_bin(
        &rest,
        "int_eq",
        hl(&rs[4]),
        signed_const(SEARCH_RFIND),
        "is_rfind",
        LowLevelType::Bool,
    );
    let rfind_b = clone_block_like(&rs);
    let find_b = clone_block_like(&rs);
    branch(
        &rest,
        is_rfind,
        pcarry(&rs),
        rfind_b.clone(),
        pcarry(&rs),
        find_b.clone(),
    );
    let rb = block_vars(&rfind_b);
    ret(&rfind_b, hl(&rb[3]), returnblock.clone());
    let fb = block_vars(&find_b);
    ret(&find_b, hl(&fb[2]), returnblock);
}

fn with_res(base: &[Variable]) -> Vec<Variable> {
    let mut vars: Vec<Variable> = base.iter().map(Variable::copy).collect();
    vars.push(v("res", LowLevelType::Signed));
    vars
}

fn emit_after_forward_res(block: &BlockRef, res: Variable, returnblock: BlockRef) {
    let ins = block_vars(block);
    let is_count = emit_bin(
        block,
        "int_eq",
        hl(&ins[4]),
        signed_const(SEARCH_COUNT),
        "is_count",
        LowLevelType::Bool,
    );
    let count_b = blk(with_res(&ins));
    let adj = blk(with_res(&ins));
    let mut args = pcarry(&ins);
    args.push(hl(&res));
    branch(
        block,
        is_count,
        args.clone(),
        count_b.clone(),
        args,
        adj.clone(),
    );
    let cb = block_vars(&count_b);
    ret(&count_b, hl(&cb[cb.len() - 1]), returnblock.clone());
    let av = block_vars(&adj);
    let resv = av[av.len() - 1].clone();
    let is_m1 = emit_bin(
        &adj,
        "int_eq",
        hl(&resv),
        signed_const(-1),
        "is_m1",
        LowLevelType::Bool,
    );
    let miss = clone_block_like(&ins);
    let hit = blk(with_res(&ins));
    branch(
        &adj,
        is_m1,
        pcarry(&av[..ins.len()]),
        miss.clone(),
        pcarry(&av),
        hit.clone(),
    );
    ret(&miss, signed_const(-1), returnblock.clone());
    let hv = block_vars(&hit);
    let pos = emit_bin(
        &hit,
        "int_add",
        hl(&hv[2]),
        hl(&hv[hv.len() - 1]),
        "pos",
        LowLevelType::Signed,
    );
    ret(&hit, hl(&pos), returnblock);
}

fn emit_forward_search(
    rtyper: &RPythonTyper,
    kind: &StringKind,
    block: &BlockRef,
    returnblock: BlockRef,
) -> Result<(), TyperError> {
    let ins = block_vars(block);
    let default_find = funcptr_default_find(rtyper, kind)?;
    let adaptive_find = funcptr_adaptive_find(rtyper, kind)?;
    let prep = funcptr_twoway_preprocess(rtyper, kind)?;
    let two = funcptr_two_way(rtyper, kind)?;
    let two_count = funcptr_two_way_count(rtyper, kind)?;
    // `_search` forward dispatch, short-circuit:
    // `n < 2500` → `_default_find`;
    // else `m < 100 and n < 30000` → `_default_find`;
    // else `m < 6` → `_default_find`;
    // else `(m >> 2) * 3 < (n >> 2)` → two-way, else `_adaptive_find`.
    let n_small = emit_bin(
        block,
        "int_lt",
        hl(&ins[5]),
        signed_const(2500),
        "n_small",
        LowLevelType::Bool,
    );
    let def_b = clone_block_like(&ins);
    let n_big = clone_block_like(&ins);
    branch(
        block,
        n_small,
        pcarry(&ins),
        def_b.clone(),
        pcarry(&ins),
        n_big.clone(),
    );
    emit_call_default_find(&def_b, &default_find, returnblock.clone());

    let nb = block_vars(&n_big);
    let m_lt_100 = emit_bin(
        &n_big,
        "int_lt",
        hl(&nb[6]),
        signed_const(100),
        "m_lt_100",
        LowLevelType::Bool,
    );
    let m_mid = clone_block_like(&nb);
    let m_big = clone_block_like(&nb);
    branch(
        &n_big,
        m_lt_100,
        pcarry(&nb),
        m_mid.clone(),
        pcarry(&nb),
        m_big.clone(),
    );
    let mm = block_vars(&m_mid);
    let n_lt_30k = emit_bin(
        &m_mid,
        "int_lt",
        hl(&mm[5]),
        signed_const(30000),
        "n_lt_30k",
        LowLevelType::Bool,
    );
    let def2 = clone_block_like(&mm);
    let rest = clone_block_like(&mm);
    branch(
        &m_mid,
        n_lt_30k,
        pcarry(&mm),
        def2.clone(),
        pcarry(&mm),
        rest.clone(),
    );
    emit_call_default_find(&def2, &default_find, returnblock.clone());
    // Join onto `m_big` so the third disjunct `m < 6` runs after
    // `n < 2500` and `m < 100 and n < 30000` both failed (`_search`).
    goto(&rest, pcarry(&block_vars(&rest)), m_big.clone());

    let mb = block_vars(&m_big);
    let m_lt_6 = emit_bin(
        &m_big,
        "int_lt",
        hl(&mb[6]),
        signed_const(6),
        "m_lt_6",
        LowLevelType::Bool,
    );
    let def3 = clone_block_like(&mb);
    let rest2 = clone_block_like(&mb);
    branch(
        &m_big,
        m_lt_6,
        pcarry(&mb),
        def3.clone(),
        pcarry(&mb),
        rest2.clone(),
    );
    emit_call_default_find(&def3, &default_find, returnblock.clone());

    emit_twoway_or_adaptive(
        kind,
        &rest2,
        &prep,
        &two,
        &two_count,
        &adaptive_find,
        returnblock,
    )
}

fn emit_call_default_find(block: &BlockRef, default_find: &Hlvalue, returnblock: BlockRef) {
    let ins = block_vars(block);
    let res = emit_direct_call(
        block,
        default_find.clone(),
        vec![
            hl(&ins[0]),
            hl(&ins[2]),
            hl(&ins[5]),
            hl(&ins[1]),
            hl(&ins[6]),
            hl(&ins[4]),
        ],
        "res",
        LowLevelType::Signed,
    );
    emit_after_forward_res(block, res, returnblock);
}

fn emit_twoway_or_adaptive(
    kind: &StringKind,
    block: &BlockRef,
    prep: &Hlvalue,
    two: &Hlvalue,
    two_count: &Hlvalue,
    adaptive_find: &Hlvalue,
    returnblock: BlockRef,
) -> Result<(), TyperError> {
    let ins = block_vars(block);
    let m4 = emit_bin(
        block,
        "int_rshift",
        hl(&ins[6]),
        signed_const(2),
        "m4",
        LowLevelType::Signed,
    );
    let n4 = emit_bin(
        block,
        "int_rshift",
        hl(&ins[5]),
        signed_const(2),
        "n4",
        LowLevelType::Signed,
    );
    let m43 = emit_bin(
        block,
        "int_mul",
        hl(&m4),
        signed_const(3),
        "m43",
        LowLevelType::Signed,
    );
    let use_tw = emit_bin(
        block,
        "int_lt",
        hl(&m43),
        hl(&n4),
        "use_tw",
        LowLevelType::Bool,
    );
    let tw = clone_block_like(&ins);
    let ad = clone_block_like(&ins);
    branch(
        block,
        use_tw,
        pcarry(&ins),
        tw.clone(),
        pcarry(&ins),
        ad.clone(),
    );
    let av = block_vars(&ad);
    let res = emit_direct_call(
        &ad,
        adaptive_find.clone(),
        vec![
            hl(&av[0]),
            hl(&av[2]),
            hl(&av[5]),
            hl(&av[1]),
            hl(&av[6]),
            hl(&av[4]),
        ],
        "res",
        LowLevelType::Signed,
    );
    emit_after_forward_res(&ad, res, returnblock.clone());

    let tv = block_vars(&tw);
    let tup = emit_direct_call(
        &tw,
        prep.clone(),
        vec![hl(&tv[1]), hl(&tv[6])],
        "tup",
        tuple5_ptr_lltype(),
    );
    let (cut, period, gap, is_periodic, table) = unpack_tuple5(&tw, &tup);
    let is_count = emit_bin(
        &tw,
        "int_eq",
        hl(&tv[4]),
        signed_const(SEARCH_COUNT),
        "is_count",
        LowLevelType::Bool,
    );
    let count_b = blk(with_twoway_fields(&tv));
    let find_b = blk(with_twoway_fields(&tv));
    let mut args = pcarry(&tv);
    args.extend([
        hl(&cut),
        hl(&period),
        hl(&gap),
        hl(&is_periodic),
        hl(&table),
    ]);
    branch(
        &tw,
        is_count,
        args.clone(),
        count_b.clone(),
        args,
        find_b.clone(),
    );
    let n = tv.len();
    let cb = block_vars(&count_b);
    let res = emit_direct_call(
        &count_b,
        two_count.clone(),
        vec![
            hl(&cb[0]),
            hl(&cb[2]),
            hl(&cb[5]),
            hl(&cb[1]),
            hl(&cb[6]),
            hl(&cb[n]),
            hl(&cb[n + 1]),
            hl(&cb[n + 2]),
            hl(&cb[n + 3]),
            hl(&cb[n + 4]),
        ],
        "res",
        LowLevelType::Signed,
    );
    ret(&count_b, hl(&res), returnblock.clone());
    let fb = block_vars(&find_b);
    let res = emit_direct_call(
        &find_b,
        two.clone(),
        vec![
            hl(&fb[0]),
            hl(&fb[2]),
            hl(&fb[5]),
            hl(&fb[1]),
            hl(&fb[6]),
            hl(&fb[n]),
            hl(&fb[n + 1]),
            hl(&fb[n + 2]),
            hl(&fb[n + 3]),
            hl(&fb[n + 4]),
        ],
        "res",
        LowLevelType::Signed,
    );
    emit_after_forward_res(&find_b, res, returnblock);
    let _ = kind;
    Ok(())
}

fn emit_rfind_search(
    rtyper: &RPythonTyper,
    kind: &StringKind,
    block: &BlockRef,
    returnblock: BlockRef,
) -> Result<(), TyperError> {
    let ins = block_vars(block);
    let bloom_add = funcptr_bloom_add(rtyper, kind)?;
    let bloom = funcptr_bloom(rtyper, kind)?;
    let mlast = emit_bin(
        block,
        "int_sub",
        hl(&ins[6]),
        signed_const(1),
        "mlast",
        LowLevelType::Signed,
    );
    let ch0 = emit_getitem(kind, block, &ins[1], signed_const(0));
    let mask0 = emit_direct_call(
        block,
        bloom_add.clone(),
        vec![signed_const(0), hl(&ch0)],
        "mask",
        LowLevelType::Signed,
    );
    let mut prep_vars = ins.iter().map(Variable::copy).collect::<Vec<_>>();
    prep_vars.extend([
        v("mlast", LowLevelType::Signed),
        v("skip", LowLevelType::Signed),
        v("mask", LowLevelType::Signed),
        v("ch0", kind.elem_lltype.clone()),
        v("i", LowLevelType::Signed),
    ]);
    let prep = blk(prep_vars);
    let mut prep_args = pcarry(&ins);
    prep_args.extend([hl(&mlast), hl(&mlast), hl(&mask0), hl(&ch0), hl(&mlast)]);
    goto(block, prep_args, prep.clone());

    let pv = block_vars(&prep);
    let i_gt = emit_bin(
        &prep,
        "int_gt",
        hl(&pv[pv.len() - 1]),
        signed_const(0),
        "i_gt",
        LowLevelType::Bool,
    );
    let pbody = clone_block_like(&pv);
    let after = clone_block_like(&pv);
    branch(
        &prep,
        i_gt,
        pcarry(&pv),
        pbody.clone(),
        pcarry(&pv),
        after.clone(),
    );
    let pb = block_vars(&pbody);
    let i = pb[pb.len() - 1].clone();
    let ch = emit_getitem(kind, &pbody, &pb[1], hl(&i));
    let mask1 = emit_direct_call(
        &pbody,
        bloom_add,
        vec![hl(&pb[pb.len() - 3]), hl(&ch)],
        "mask",
        LowLevelType::Signed,
    );
    let eq0 = emit_bin(
        &pbody,
        kind.eq_op,
        hl(&ch),
        hl(&pb[pb.len() - 2]),
        "eq0",
        LowLevelType::Bool,
    );
    let skip_upd = clone_block_like(&pb);
    let skip_keep = clone_block_like(&pb);
    let new_skip = emit_bin(
        &pbody,
        "int_sub",
        hl(&i),
        signed_const(1),
        "skip",
        LowLevelType::Signed,
    );
    let mut to_upd = pcarry(&pb);
    to_upd[pb.len() - 4] = hl(&new_skip);
    to_upd[pb.len() - 3] = hl(&mask1);
    let mut to_keep = pcarry(&pb);
    to_keep[pb.len() - 3] = hl(&mask1);
    branch(
        &pbody,
        eq0,
        to_upd,
        skip_upd.clone(),
        to_keep,
        skip_keep.clone(),
    );
    let dec_i = |block: &BlockRef| {
        let b = block_vars(block);
        let i1 = emit_bin(
            block,
            "int_sub",
            hl(&b[b.len() - 1]),
            signed_const(1),
            "i",
            LowLevelType::Signed,
        );
        let mut args = pcarry(&b);
        *args.last_mut().unwrap() = hl(&i1);
        goto(block, args, prep.clone());
    };
    dec_i(&skip_upd);
    dec_i(&skip_keep);

    let av = block_vars(&after);
    let i0 = emit_bin(
        &after,
        "int_add",
        hl(&av[2]),
        hl(&av[7]),
        "t",
        LowLevelType::Signed,
    );
    let i0 = emit_bin(
        &after,
        "int_add",
        hl(&i0),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    let search = clone_block_like(&av);
    let mut sargs = pcarry(&av);
    *sargs.last_mut().unwrap() = hl(&i0);
    goto(&after, sargs, search.clone());

    let sv = block_vars(&search);
    let im1 = emit_bin(
        &search,
        "int_sub",
        hl(&sv[sv.len() - 1]),
        signed_const(1),
        "im1",
        LowLevelType::Signed,
    );
    let cont = emit_bin(
        &search,
        "int_ge",
        hl(&im1),
        hl(&sv[2]),
        "cont",
        LowLevelType::Bool,
    );
    let sbody = clone_block_like(&sv);
    let done = clone_block_like(&sv);
    branch(
        &search,
        cont,
        pcarry(&sv),
        sbody.clone(),
        pcarry(&sv),
        done.clone(),
    );
    ret(&done, signed_const(-1), returnblock.clone());

    let sb = block_vars(&sbody);
    let i1 = emit_bin(
        &sbody,
        "int_sub",
        hl(&sb[sb.len() - 1]),
        signed_const(1),
        "i",
        LowLevelType::Signed,
    );
    let hay = emit_getitem(kind, &sbody, &sb[0], hl(&i1));
    let eq = emit_bin(
        &sbody,
        kind.eq_op,
        hl(&hay),
        hl(&sb[sb.len() - 2]),
        "eq",
        LowLevelType::Bool,
    );
    let mut inner_vars: Vec<Variable> = sb.iter().map(Variable::copy).collect();
    let i_slot = inner_vars.len() - 1;
    inner_vars[i_slot] = v("i", LowLevelType::Signed);
    inner_vars.push(v("j", LowLevelType::Signed));
    let inner = blk(inner_vars);
    let miss_ch = clone_block_like(&sb);
    let mut inner_args = pcarry(&sb);
    *inner_args.last_mut().unwrap() = hl(&i1);
    inner_args.push(hl(&sb[sb.len() - 5]));
    let mut miss_args = pcarry(&sb);
    *miss_args.last_mut().unwrap() = hl(&i1);
    branch(
        &sbody,
        eq,
        inner_args,
        inner.clone(),
        miss_args,
        miss_ch.clone(),
    );
    emit_rfind_bloom_skip(kind, &bloom, &miss_ch, &search, false);

    let iv = block_vars(&inner);
    let j_gt = emit_bin(
        &inner,
        "int_gt",
        hl(&iv[iv.len() - 1]),
        signed_const(0),
        "j_gt",
        LowLevelType::Bool,
    );
    let ibody = clone_block_like(&iv);
    let found = clone_block_like(&sb);
    branch(
        &inner,
        j_gt,
        pcarry(&iv),
        ibody.clone(),
        pcarry(&iv[..sb.len()]),
        found.clone(),
    );
    let fv = block_vars(&found);
    ret(&found, hl(&fv[fv.len() - 1]), returnblock);

    let ib = block_vars(&ibody);
    let j = ib[ib.len() - 1].clone();
    let pos = emit_bin(
        &ibody,
        "int_add",
        hl(&ib[ib.len() - 2]),
        hl(&j),
        "pos",
        LowLevelType::Signed,
    );
    let hv = emit_getitem(kind, &ibody, &ib[0], hl(&pos));
    let nd = emit_getitem(kind, &ibody, &ib[1], hl(&j));
    let ne = emit_bin(
        &ibody,
        kind.ne_op,
        hl(&hv),
        hl(&nd),
        "ne",
        LowLevelType::Bool,
    );
    let mismatch = clone_block_like(&ib);
    let jdec = clone_block_like(&ib);
    branch(
        &ibody,
        ne,
        pcarry(&ib),
        mismatch.clone(),
        pcarry(&ib),
        jdec.clone(),
    );
    let jd = block_vars(&jdec);
    let j1 = emit_bin(
        &jdec,
        "int_sub",
        hl(&jd[jd.len() - 1]),
        signed_const(1),
        "j",
        LowLevelType::Signed,
    );
    let mut back = pcarry(&jd);
    *back.last_mut().unwrap() = hl(&j1);
    goto(&jdec, back, inner.clone());

    let ms = block_vars(&mismatch);
    let skipb = clone_block_like(&sb);
    goto(&mismatch, pcarry(&ms[..sb.len()]), skipb.clone());
    emit_rfind_bloom_skip(kind, &bloom, &skipb, &search, true);
    Ok(())
}

fn emit_rfind_bloom_skip(
    kind: &StringKind,
    bloom: &Hlvalue,
    block: &BlockRef,
    search: &BlockRef,
    use_skip: bool,
) {
    let ins = block_vars(block);
    let i = ins[ins.len() - 1].clone();
    let im1 = emit_bin(
        block,
        "int_sub",
        hl(&i),
        signed_const(1),
        "im1",
        LowLevelType::Signed,
    );
    let ge0 = emit_bin(
        block,
        "int_ge",
        hl(&im1),
        signed_const(0),
        "ge0",
        LowLevelType::Bool,
    );
    let chk = clone_block_like(&ins);
    let noskip = clone_block_like(&ins);
    branch(
        block,
        ge0,
        pcarry(&ins),
        chk.clone(),
        pcarry(&ins),
        noskip.clone(),
    );
    let cv = block_vars(&chk);
    let im1 = emit_bin(
        &chk,
        "int_sub",
        hl(&cv[cv.len() - 1]),
        signed_const(1),
        "im1",
        LowLevelType::Signed,
    );
    let ch = emit_getitem(kind, &chk, &cv[0], hl(&im1));
    let br = emit_direct_call(
        &chk,
        bloom.clone(),
        vec![hl(&cv[cv.len() - 3]), hl(&ch)],
        "br",
        LowLevelType::Signed,
    );
    let zero = emit_un(&chk, "int_is_zero", hl(&br), "z", LowLevelType::Bool);
    let sub_m = clone_block_like(&cv);
    let other = clone_block_like(&cv);
    branch(
        &chk,
        zero,
        pcarry(&cv),
        sub_m.clone(),
        pcarry(&cv),
        other.clone(),
    );
    let sm = block_vars(&sub_m);
    let i2 = emit_bin(
        &sub_m,
        "int_sub",
        hl(&sm[sm.len() - 1]),
        hl(&sm[6]),
        "i",
        LowLevelType::Signed,
    );
    let mut a = pcarry(&sm);
    *a.last_mut().unwrap() = hl(&i2);
    goto(&sub_m, a, search.clone());
    if use_skip {
        let ov = block_vars(&other);
        let i2 = emit_bin(
            &other,
            "int_sub",
            hl(&ov[ov.len() - 1]),
            hl(&ov[ov.len() - 4]),
            "i",
            LowLevelType::Signed,
        );
        let mut a = pcarry(&ov);
        *a.last_mut().unwrap() = hl(&i2);
        goto(&other, a, search.clone());
    } else {
        goto(&other, pcarry(&block_vars(&other)), search.clone());
    }
    if use_skip {
        let nv = block_vars(&noskip);
        let i2 = emit_bin(
            &noskip,
            "int_sub",
            hl(&nv[nv.len() - 1]),
            hl(&nv[nv.len() - 4]),
            "i",
            LowLevelType::Signed,
        );
        let mut a = pcarry(&nv);
        *a.last_mut().unwrap() = hl(&i2);
        goto(&noskip, a, search.clone());
    } else {
        goto(&noskip, pcarry(&block_vars(&noskip)), search.clone());
    }
}

/// Convert an rtyped search helper into a rich-`OpKind` graph for
/// `CallControl::getcalldescr`. Extends `effect_graph_from_rtyped_helper`
/// with `malloc` / `malloc_varsize` / `getfield` / `setfield` /
/// `cast_unichar_to_int` / `cast_int_to_unichar` / `ptr_nonzero` /
/// `ptr_iszero`, which that converter rejects. List helpers
/// (`ll_newlist` / `ll_append` / `ll_setitem_fast` / `ll_getitem_fast`)
/// are `direct_call`. Nested `malloc_varsize` of a list `items` array
/// is `NewArray` keyed by the array element type. `ptr_nonzero` is
/// `ll_search`'s `assert value is not None` / `assert other is not None`
/// (`rstr.py` `ll_search`).
pub(crate) fn effect_graph_from_search_helper(graph: &FunctionGraph) -> ModelGraph {
    type FlowBlockKey = *const RefCell<Block>;

    let mut out = ModelGraph::new(graph.name.clone());
    let mut block_map: HashMap<FlowBlockKey, BlockId> = HashMap::new();
    block_map.insert(Rc::as_ptr(&graph.startblock), out.startblock);
    block_map.insert(Rc::as_ptr(&graph.returnblock), out.returnblock);
    block_map.insert(Rc::as_ptr(&graph.exceptblock), out.exceptblock);

    let flow_blocks = graph.iterblocks();
    for fb in &flow_blocks {
        let key = Rc::as_ptr(fb);
        let id = *block_map.entry(key).or_insert_with(|| out.create_block());
        let inputargs: Vec<Variable> = fb
            .borrow()
            .inputargs
            .iter()
            .filter_map(|hlv| match hlv {
                Hlvalue::Variable(v) => Some(v.clone()),
                Hlvalue::Constant(_) => None,
            })
            .collect();
        out.block_mut(id).inputargs = inputargs;
    }
    let return_inputargs: Vec<Variable> = graph
        .returnblock
        .borrow()
        .inputargs
        .iter()
        .filter_map(|hlv| match hlv {
            Hlvalue::Variable(v) => Some(v.clone()),
            Hlvalue::Constant(_) => None,
        })
        .collect();
    let return_id = out.returnblock;
    out.block_mut(return_id).inputargs = return_inputargs;

    let start_id = out.startblock;
    let start_inputs = out.block(start_id).inputargs.clone();
    for var in start_inputs {
        let ty = search_value_type_of(&var);
        let name = if var.renamed() {
            var.name_prefix()
        } else {
            "arg".to_string()
        };
        out.push_op_with_result_var(
            start_id,
            OpKind::Input {
                name,
                ty,
                class_root: None,
            },
            var,
        );
    }

    for fb in &flow_blocks {
        let id = block_map[&Rc::as_ptr(fb)];
        let fb_ref = fb.borrow();
        for op in &fb_ref.operations {
            search_transduce_op(&mut out, id, op);
        }
        let exitswitch = if fb_ref.canraise() {
            Some(ExitSwitch::LastException)
        } else {
            match &fb_ref.exitswitch {
                Some(Hlvalue::Variable(v)) => Some(ExitSwitch::Value(v.clone())),
                _ => None,
            }
        };
        let exits: Vec<ModelLink> = fb_ref
            .exits
            .iter()
            .map(|link_ref| {
                let link = link_ref.borrow();
                let target = link
                    .target
                    .as_ref()
                    .expect("search helper link has no target block");
                let target_id = block_map[&Rc::as_ptr(target)];
                let args: Vec<LinkArg> = link
                    .args
                    .iter()
                    .map(|arg| {
                        search_linkarg(arg.as_ref().expect("search helper link arg is undefined"))
                    })
                    .collect();
                let exitcase = match &link.exitcase {
                    Some(Hlvalue::Constant(c)) => match &c.value {
                        ConstValue::Bool(b) => Some(ExitCase::Bool(*b)),
                        other => Some(ExitCase::Const(other.clone())),
                    },
                    _ => None,
                };
                let mut model_link = ModelLink::new_mixed(args, target_id, exitcase);
                model_link.last_exception = link.last_exception.as_ref().map(search_linkarg);
                model_link.last_exc_value = link.last_exc_value.as_ref().map(search_linkarg);
                model_link
            })
            .collect();
        drop(fb_ref);
        if exitswitch.is_some() || !exits.is_empty() {
            out.set_control_flow_metadata(id, exitswitch, exits);
        }
    }

    if let Some(func) = &graph.func {
        if func._elidable_function_ {
            out.push_hint("elidable");
        }
        if func._jit_look_inside_ == Some(false) {
            out.push_hint("dont_look_inside");
        } else if let Some(look) = func._jit_look_inside_ {
            out.func.jit_look_inside = Some(look);
        }
        if func._jit_unroll_safe_ {
            out.func.unroll_safe = true;
        }
        out.func.oopspec = func.oopspec.clone();
        if func.oopspec.is_some() {
            out.func.oopspec_argnames = startblock_oopspec_argnames(graph);
        }
    }
    if out
        .block(out.returnblock)
        .inputargs
        .first()
        .is_some_and(|ret| ModelGraph::concretetype_of(ret) == ConcreteType::Signed)
    {
        out.return_type = Some("i64".into());
    }
    out
}

/// Bare startblock inputarg names for `FuncEffects.oopspec_argnames`.
/// Same source as `OpKind::Input { name }` (`Variable::name_prefix` when
/// renamed); strip the intern trailing `_` so the names match harvested
/// `#[oopspec]` argnames (no numbering suffix).
fn startblock_oopspec_argnames(graph: &FunctionGraph) -> Vec<String> {
    graph
        .startblock
        .borrow()
        .inputargs
        .iter()
        .filter_map(|hlv| match hlv {
            Hlvalue::Variable(v) => {
                let name = if v.renamed() {
                    v.name_prefix()
                } else {
                    "arg".to_string()
                };
                Some(name.strip_suffix('_').unwrap_or(&name).to_string())
            }
            Hlvalue::Constant(_) => None,
        })
        .collect()
}

fn search_value_type_of(var: &Variable) -> ValueType {
    match ModelGraph::concretetype_of(var) {
        ConcreteType::Signed => ValueType::Int,
        ConcreteType::GcRef => ValueType::Ref(None),
        ConcreteType::Float => ValueType::Float,
        ConcreteType::Void => ValueType::Void,
        ConcreteType::Unknown => ValueType::Int,
    }
}

fn search_array_type_id(base: &Variable) -> Option<String> {
    let arr = match base.concretetype() {
        Some(LowLevelType::Ptr(ptr)) => match ptr.TO {
            PtrTarget::Array(arr) => Some(arr),
            _ => None,
        },
        Some(LowLevelType::Array(arr)) => Some(*arr),
        _ => None,
    };
    Some(array_type_id_of_lltype(
        &arr.unwrap_or_else(|| Array::gc(LowLevelType::Char)),
    ))
}

fn search_linkarg(hlv: &Hlvalue) -> LinkArg {
    match hlv {
        Hlvalue::Variable(v) => LinkArg::Value(v.clone()),
        Hlvalue::Constant(c) => LinkArg::Const(c.clone()),
    }
}

fn search_expect_var(hlv: &Hlvalue) -> Variable {
    match hlv {
        Hlvalue::Variable(v) => v.clone(),
        Hlvalue::Constant(c) => {
            panic!("effect_graph_from_search_helper: expected a Variable, found constant {c:?}")
        }
    }
}

/// Runtime call operands, dropping Void. Void is compile-time only.
///
/// `jtransform.py Transformer.make_three_lists` skips `kind == 'void'`
/// (`add_in_correct_list`); `call.py CallControl.getcalldescr` builds
/// `NON_VOID_ARGS` the same way. `_rewrite_op_cond_call` routes extra
/// args through `rewrite_call` → `make_three_lists`.
fn search_non_void_args(args: &[Hlvalue]) -> impl Iterator<Item = &Hlvalue> {
    args.iter().filter(|hlv| match hlv {
        Hlvalue::Variable(v) => !matches!(v.concretetype(), Some(LowLevelType::Void)),
        Hlvalue::Constant(c) => !matches!(c.concretetype, Some(LowLevelType::Void)),
    })
}

fn search_materialize(out: &mut ModelGraph, block: BlockId, hlv: &Hlvalue) -> Variable {
    match hlv {
        Hlvalue::Variable(v) => v.clone(),
        Hlvalue::Constant(c) => {
            let (kind, lltype) = match &c.value {
                ConstValue::Int(n) => (OpKind::ConstInt(*n), LowLevelType::Signed),
                ConstValue::UInt(n) => (OpKind::ConstUInt(*n), LowLevelType::Unsigned),
                ConstValue::Bool(b) => (OpKind::ConstBool(*b), LowLevelType::Bool),
                ConstValue::AddressOffset(_) => (OpKind::ConstInt(0), LowLevelType::Signed),
                ConstValue::ByteStr(bytes) => {
                    let n = i64::from(bytes.first().copied().unwrap_or(0));
                    (OpKind::ConstInt(n), LowLevelType::Signed)
                }
                other => {
                    panic!("effect_graph_from_search_helper: cannot materialise constant {other:?}")
                }
            };
            let result = out
                .push_op_var(block, kind, true)
                .expect("constant op produces a result var");
            result.set_concretetype(Some(lltype));
            result
        }
    }
}

fn search_callee_funcptr(hlv: &Hlvalue) -> (String, Option<usize>) {
    use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
    let Hlvalue::Constant(c) = hlv else {
        panic!("effect_graph_from_search_helper: direct_call callee is not a constant");
    };
    let ConstValue::LLPtr(ptr) = &c.value else {
        panic!(
            "effect_graph_from_search_helper: direct_call callee is not an LLPtr: {:?}",
            c.value
        );
    };
    match ptr._obj0_value() {
        Ok(Some(_ptr_obj::Func(func))) => (func._name.clone(), func.graph),
        other => {
            panic!("effect_graph_from_search_helper: direct_call callee is not a Func: {other:?}")
        }
    }
}

fn search_field_name(hlv: &Hlvalue) -> String {
    match hlv {
        Hlvalue::Constant(c) => match &c.value {
            ConstValue::ByteStr(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            other => panic!(
                "effect_graph_from_search_helper: field-name operand is not a ByteStr: {other:?}"
            ),
        },
        Hlvalue::Variable(_) => {
            panic!("effect_graph_from_search_helper: field-name operand is a Variable")
        }
    }
}

fn search_owner_root(base: &Variable) -> Option<String> {
    match base.concretetype()? {
        LowLevelType::Ptr(ptr) => match ptr.TO {
            PtrTarget::Struct(s) => Some(s._name),
            _ => None,
        },
        _ => None,
    }
}

fn search_malloc_owner(hlv: &Hlvalue) -> String {
    match hlv {
        Hlvalue::Constant(c) => match &c.value {
            ConstValue::LowLevelType(lltype) => match &**lltype {
                LowLevelType::Struct(s) => s._name.clone(),
                other => panic!(
                    "effect_graph_from_search_helper: malloc type operand is not a Struct: {other:?}"
                ),
            },
            other => panic!(
                "effect_graph_from_search_helper: malloc first operand is not a LowLevelType const: {other:?}"
            ),
        },
        Hlvalue::Variable(_) => {
            panic!("effect_graph_from_search_helper: malloc first operand is a Variable")
        }
    }
}

fn search_transduce_op(out: &mut ModelGraph, block: BlockId, op: &SpaceOperation) {
    match op.opname.as_str() {
        "getsubstruct" => {
            let string_var = search_expect_var(&op.args[0]);
            let chars_var = search_expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand: string_var,
                    result_ty: search_value_type_of(&chars_var),
                },
                chars_var,
            );
        }
        "getarraysize" => {
            let base = search_expect_var(&op.args[0]);
            let array_type_id = search_array_type_id(&base);
            let result = search_expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::ArrayLen {
                    base,
                    array_type_id,
                    nolength: false,
                },
                result,
            );
        }
        "getarrayitem" => {
            let base = search_expect_var(&op.args[0]);
            let array_type_id = search_array_type_id(&base);
            let index = search_materialize(out, block, &op.args[1]);
            let result = search_expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty: search_value_type_of(&result),
                    array_type_id,
                    nolength: false,
                    pure: false,
                },
                result,
            );
        }
        "setarrayitem" => {
            let base = search_expect_var(&op.args[0]);
            let array_type_id = search_array_type_id(&base);
            let index = search_materialize(out, block, &op.args[1]);
            let value = search_linkarg(&op.args[2]);
            let item_ty = match &op.args[2] {
                Hlvalue::Variable(v) => search_value_type_of(v),
                Hlvalue::Constant(_) => ValueType::Int,
            };
            out.push_op_var(
                block,
                OpKind::ArrayWrite {
                    base,
                    index,
                    value,
                    item_ty,
                    array_type_id,
                    nolength: false,
                },
                false,
            );
        }
        "direct_call" => {
            let (callee, _) = search_callee_funcptr(&op.args[0]);
            let result = search_expect_var(&op.result);
            // `graphanalyze.py GraphAnalyzer.analyze`: a `direct_call`
            // whose funcobj has a graph is followed; a graphless funcobj
            // is `analyze_external_call`. The graphless callee reached
            // here is the `majit_rlib::jit::isconstant` oopspec
            // (`rlib/jit.py isconstant`); emit the ordinary Call so
            // analyzers classify the builtin target.
            let target = CallTarget::function_path(callee.split("::").filter(|s| !s.is_empty()));
            let args: Vec<_> = search_non_void_args(&op.args[1..])
                .map(|a| search_materialize(out, block, a))
                .collect();
            out.push_op_with_result_var(
                block,
                OpKind::Call {
                    target,
                    args: crate::model::call_args(args),
                    result_ty: search_value_type_of(&result),
                },
                result,
            );
        }
        "jit_conditional_call" => {
            // Same interpreter-facing Call `lower_graph` emits.
            // Analyzers treat `oopspec("jit.conditional_call")` as a
            // builtin; callee effects reach through the
            // `we_are_jitted()`-false `direct_call`.
            assert!(
                op.args.len() >= 2,
                "effect_graph_from_search_helper: jit_conditional_call needs cond and fnptr"
            );
            let cond = search_materialize(out, block, &op.args[0]);
            let (callee, _) = search_callee_funcptr(&op.args[1]);
            let extra_args: Vec<_> = search_non_void_args(&op.args[2..])
                .map(|a| search_materialize(out, block, a))
                .collect();
            let result = search_expect_var(&op.result);
            crate::codewriter::jtransform_opname::emit_conditional_call_oopspec(
                out, block, cond, &callee, extra_args, result,
            );
        }
        "jit_conditional_call_value" => {
            // Same interpreter-facing Call `lower_graph` emits.
            // Analyzers treat `oopspec("jit.conditional_call_value")` as a
            // builtin; callee effects reach through the
            // `we_are_jitted()`-false `direct_call`.
            assert!(
                op.args.len() >= 2,
                "effect_graph_from_search_helper: jit_conditional_call_value needs value and fnptr"
            );
            let value = search_materialize(out, block, &op.args[0]);
            let (callee, _) = search_callee_funcptr(&op.args[1]);
            let extra_args: Vec<_> = search_non_void_args(&op.args[2..])
                .map(|a| search_materialize(out, block, a))
                .collect();
            let result = search_expect_var(&op.result);
            crate::codewriter::jtransform_opname::emit_conditional_call_value_oopspec(
                out, block, value, &callee, extra_args, result,
            );
        }
        "malloc" => {
            let owner = search_malloc_owner(&op.args[0]);
            let result = search_expect_var(&op.result);
            out.push_op_with_result_var(block, OpKind::New { owner }, result);
        }
        "malloc_varsize" => {
            let length = search_materialize(out, block, &op.args[2]);
            let result = search_expect_var(&op.result);
            let array_type_id = search_array_type_id(&result);
            out.push_op_with_result_var(
                block,
                OpKind::NewArray {
                    length,
                    item_ty: ValueType::Int,
                    array_type_id,
                },
                result,
            );
        }
        "getfield" => {
            let base = search_expect_var(&op.args[0]);
            let field_name = search_field_name(&op.args[1]);
            let owner_root = search_owner_root(&base);
            let result = search_expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new(field_name, owner_root),
                    ty: search_value_type_of(&result),
                    pure: false,
                },
                result,
            );
        }
        "setfield" => {
            let base = search_expect_var(&op.args[0]);
            let field_name = search_field_name(&op.args[1]);
            let owner_root = search_owner_root(&base);
            let value = search_linkarg(&op.args[2]);
            let ty = match &op.args[2] {
                Hlvalue::Variable(v) => search_value_type_of(v),
                Hlvalue::Constant(_) => ValueType::Int,
            };
            out.push_op_var(
                block,
                OpKind::FieldWrite {
                    base,
                    field: FieldDescriptor::new(field_name, owner_root),
                    value,
                    ty,
                },
                false,
            );
        }
        name @ ("cast_ptr_to_adr"
        | "cast_adr_to_ptr"
        | "cast_primitive"
        | "cast_bool_to_int"
        | "cast_char_to_int"
        | "cast_int_to_char"
        | "cast_unichar_to_int"
        | "cast_int_to_unichar") => {
            let operand = search_materialize(out, block, &op.args[0]);
            let result = search_expect_var(&op.result);
            let _ = name;
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand,
                    result_ty: search_value_type_of(&result),
                },
                result,
            );
        }
        name @ ("int_neg" | "int_invert" | "int_is_true" | "int_is_zero" | "bool_not"
        | "ptr_nonzero" | "ptr_iszero") => {
            let operand = search_materialize(out, block, &op.args[0]);
            let result = search_expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: name.to_string(),
                    operand,
                    result_ty: search_value_type_of(&result),
                },
                result,
            );
        }
        name @ ("int_add" | "int_sub" | "int_mul" | "int_and" | "int_or" | "int_xor"
        | "int_rshift" | "int_lshift" | "int_lt" | "int_le" | "int_eq" | "int_ne"
        | "int_gt" | "int_ge" | "char_eq" | "char_ne" | "unichar_eq" | "unichar_ne") => {
            let lhs = search_materialize(out, block, &op.args[0]);
            let rhs = search_materialize(out, block, &op.args[1]);
            let result = search_expect_var(&op.result);
            let bare = name.split_once('_').map(|(_, rest)| rest).unwrap_or(name);
            out.push_op_with_result_var(
                block,
                OpKind::BinOp {
                    op: bare.to_string(),
                    lhs,
                    rhs,
                    result_ty: search_value_type_of(&result),
                },
                result,
            );
        }
        name @ ("gc_id" | "gc_fq_register" | "debug_assert" | "gc_writebarrier") => {
            // `effect_graph_from_rtyped_helper`: UnaryOp keeps the llop
            // name so analyzers consult `ll_operations()`. Extra Variable
            // operands stay materialised; Constant extras are already values.
            let Some((last, rest)) = op.args.split_last() else {
                panic!("effect_graph_from_search_helper: {name} has no operand");
            };
            for arg in rest {
                if matches!(arg, Hlvalue::Variable(_)) {
                    let _ = search_materialize(out, block, arg);
                }
            }
            let operand = search_materialize(out, block, last);
            let result = search_expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: name.to_string(),
                    operand,
                    result_ty: search_value_type_of(&result),
                },
                result,
            );
        }
        other => {
            // `graphanalyze.py GraphAnalyzer.analyze` follows only
            // `direct_call` / `indirect_call`; every other llop is
            // `analyze_simple_operation` with the `lloperation.py` flags.
            // UnaryOp keeps the llop name so analyzers consult
            // `ll_operations()`, matching `effect_graph_from_rtyped_helper`.
            if other != "direct_call"
                && other != "indirect_call"
                && crate::translator::rtyper::lltypesystem::lloperation::ll_operations()
                    .contains_key(other)
            {
                let Some((last, rest)) = op.args.split_last() else {
                    panic!("effect_graph_from_search_helper: {other} has no operand");
                };
                for arg in rest {
                    if matches!(arg, Hlvalue::Variable(_)) {
                        let _ = search_materialize(out, block, arg);
                    }
                }
                let operand = search_materialize(out, block, last);
                let result = search_expect_var(&op.result);
                out.push_op_with_result_var(
                    block,
                    OpKind::UnaryOp {
                        op: other.to_string(),
                        operand,
                        result_ty: search_value_type_of(&result),
                    },
                    result,
                );
            } else {
                panic!("effect_graph_from_search_helper: unsupported opname {other:?}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::checkgraph;
    use crate::translator::rtyper::lltypesystem::rstr::{
        STRPTR, UNICODEPTR, build_ll_strhash_helper_graph, funcobj_from_funcptr_constant,
    };
    use std::collections::HashMap;
    use std::rc::Rc;

    fn direct_call_names(graph: &FunctionGraph) -> Vec<String> {
        graph
            .iterblocks()
            .into_iter()
            .flat_map(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .filter(|op| op.opname == "direct_call")
                    .filter_map(|op| {
                        op.args
                            .first()
                            .and_then(funcobj_from_funcptr_constant)
                            .map(|func| func._name)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn search_helper_graphs_build_for_str_and_unicode() {
        for ptr in [STRPTR.clone(), UNICODEPTR.clone()] {
            let ann = RPythonAnnotator::new(None, None, None, false);
            let rtyper = RPythonTyper::new(&ann);
            let normal =
                build_search_normal_helper_graph(&rtyper, "_search_normal", ptr.clone(), false)
                    .expect("_search_normal");
            let names = direct_call_names(&normal.graph.borrow());
            for expected in [
                "_default_find",
                "_twoway_preprocess",
                "_two_way",
                "_two_way_count",
                "_adaptive_find",
                "bloom_add",
                "bloom",
            ] {
                assert!(
                    names.iter().any(|n| n == expected),
                    "_search_normal calls {expected}: {names:?}"
                );
            }

            let elidable =
                build_search_elidable_helper_graph(&rtyper, "_search_elidable", ptr.clone(), false)
                    .expect("_search_elidable");
            assert!(elidable.func._elidable_function_);
            assert_eq!(
                direct_call_names(&elidable.graph.borrow()),
                vec!["_search_normal".to_string()]
            );

            let search =
                build_search_helper_graph(&rtyper, "_search", ptr.clone(), false).expect("_search");
            let expected = if ptr == *STRPTR {
                "_search_elidable"
            } else {
                "_search_normal"
            };
            assert_eq!(
                direct_call_names(&search.graph.borrow()),
                vec![expected.to_string()]
            );
            assert!(!search.func._elidable_function_);

            build_bloom_add_helper_graph(&rtyper, "bloom_add", ptr.clone()).expect("bloom_add");
            build_bloom_helper_graph(&rtyper, "bloom", ptr.clone()).expect("bloom");
            build_lex_search_helper_graph(&rtyper, "_lex_search", ptr.clone())
                .expect("_lex_search");
            build_factorize_helper_graph(&rtyper, "_factorize", ptr.clone()).expect("_factorize");
            let prep =
                build_twoway_preprocess_helper_graph(&rtyper, "_twoway_preprocess", ptr.clone())
                    .expect("_twoway_preprocess");
            let prep_names = direct_call_names(&prep.graph.borrow());
            assert!(
                prep_names.iter().any(|n| n == "_factorize"),
                "_twoway_preprocess calls _factorize"
            );
            assert!(
                prep_names.iter().any(|n| n == "ll_newlist"),
                "_twoway_preprocess calls ll_newlist: {prep_names:?}"
            );
            assert!(
                prep_names.iter().any(|n| n == "ll_append"),
                "_twoway_preprocess calls ll_append: {prep_names:?}"
            );
            build_two_way_helper_graph(&rtyper, "_two_way", ptr.clone()).expect("_two_way");
            let count = build_two_way_count_helper_graph(&rtyper, "_two_way_count", ptr.clone())
                .expect("_two_way_count");
            assert_eq!(
                direct_call_names(&count.graph.borrow()),
                vec!["_two_way".to_string()]
            );
            let def_find = build_default_find_helper_graph(&rtyper, "_default_find", ptr.clone())
                .expect("_default_find");
            let def_names = direct_call_names(&def_find.graph.borrow());
            assert!(def_names.iter().any(|n| n == "bloom_add"));
            assert!(def_names.iter().any(|n| n == "bloom"));
            let ad_find = build_adaptive_find_helper_graph(&rtyper, "_adaptive_find", ptr.clone())
                .expect("_adaptive_find");
            let ad_names = direct_call_names(&ad_find.graph.borrow());
            assert!(ad_names.iter().any(|n| n == "_twoway_preprocess"));
            assert!(ad_names.iter().any(|n| n == "_two_way"));
            assert!(ad_names.iter().any(|n| n == "_two_way_count"));

            let effect = effect_graph_from_search_helper(&prep.graph.borrow());
            assert!(
                effect.iter_block_ops().any(|(_, op)| matches!(
                    &op.kind,
                    OpKind::Call { target, .. }
                        if target.path_segments().is_some_and(|s| s.last() == Some(&"ll_newlist"))
                )),
                "_twoway_preprocess effect graph calls ll_newlist"
            );
        }
    }

    /// `_ll_list_resize_ge` effect graph keeps `direct_call` as `Call` and
    /// `jit_conditional_call` as the `jit.conditional_call` oopspec Call.
    /// Analyzers see those ops; there is no `same_as` stand-in and no
    /// unary invented from the llop's first variable argument.
    #[test]
    fn ll_list_resize_ge_effect_graph_has_no_fabricated_ops() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        build_twoway_preprocess_helper_graph(&rtyper, "_twoway_preprocess", STRPTR.clone())
            .expect("_twoway_preprocess");
        let graphs = ann.translator.graphs.borrow();
        let ge = graphs
            .iter()
            .find(|g| g.borrow().name == "_ll_list_resize_ge")
            .expect("_twoway_preprocess mints _ll_list_resize_ge");
        let effect = effect_graph_from_search_helper(&ge.borrow());
        for (_, op) in effect.iter_block_ops() {
            match &op.kind {
                OpKind::UnaryOp { op, .. } => {
                    assert_ne!(op, "same_as", "no same_as stand-in: {op:?}");
                    assert_ne!(
                        op, "jit_conditional_call",
                        "jit_conditional_call is the oopspec Call, not a unary"
                    );
                }
                _ => {}
            }
        }
        assert!(
            effect.iter_block_ops().any(|(_, op)| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target.path_segments().is_some_and(|s| s
                        == ["majit_rlib", "jit", "conditional_call3"])
            )),
            "effect graph has the jit.conditional_call oopspec Call"
        );
    }

    /// `ll_strhash` inlines `conditional_call_elidable` to
    /// `jit_conditional_call_value`. The effect graph emits the same
    /// interpreter-facing `conditional_call_elidable1` Call `lower_graph`
    /// emits.
    #[test]
    fn ll_strhash_effect_graph_emits_conditional_call_elidable1() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_strhash_helper_graph(
            &rtyper,
            "ll_strhash",
            STRPTR.clone(),
            "_ll_strhash",
            "_hash_string",
        )
        .expect("ll_strhash");
        let effect = effect_graph_from_search_helper(&helper.graph.borrow());
        let leaves: Vec<_> = effect
            .iter_block_ops()
            .filter_map(|(_, op)| match &op.kind {
                OpKind::Call { target, args, .. } => {
                    let segs = target.path_segments()?;
                    if segs == ["majit_rlib", "jit", "conditional_call_elidable1"] {
                        Some(args.len())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            leaves,
            vec![3],
            "effect graph emits conditional_call_elidable1 (value, fn, s), got {leaves:?}"
        );
    }

    #[test]
    fn search_helper_graphs_pass_checkgraph() {
        for ptr in [STRPTR.clone(), UNICODEPTR.clone()] {
            let ann = RPythonAnnotator::new(None, None, None, false);
            let rtyper = RPythonTyper::new(&ann);
            let graphs = vec![
                (
                    "bloom_add",
                    build_bloom_add_helper_graph(&rtyper, "bloom_add", ptr.clone())
                        .expect("bloom_add"),
                ),
                (
                    "bloom",
                    build_bloom_helper_graph(&rtyper, "bloom", ptr.clone()).expect("bloom"),
                ),
                (
                    "_lex_search",
                    build_lex_search_helper_graph(&rtyper, "_lex_search", ptr.clone())
                        .expect("_lex_search"),
                ),
                (
                    "_factorize",
                    build_factorize_helper_graph(&rtyper, "_factorize", ptr.clone())
                        .expect("_factorize"),
                ),
                (
                    "_twoway_preprocess",
                    build_twoway_preprocess_helper_graph(
                        &rtyper,
                        "_twoway_preprocess",
                        ptr.clone(),
                    )
                    .expect("_twoway_preprocess"),
                ),
                (
                    "_two_way",
                    build_two_way_helper_graph(&rtyper, "_two_way", ptr.clone()).expect("_two_way"),
                ),
                (
                    "_two_way_count",
                    build_two_way_count_helper_graph(&rtyper, "_two_way_count", ptr.clone())
                        .expect("_two_way_count"),
                ),
                (
                    "_default_find",
                    build_default_find_helper_graph(&rtyper, "_default_find", ptr.clone())
                        .expect("_default_find"),
                ),
                (
                    "_adaptive_find",
                    build_adaptive_find_helper_graph(&rtyper, "_adaptive_find", ptr.clone())
                        .expect("_adaptive_find"),
                ),
                (
                    "_search",
                    build_search_helper_graph(&rtyper, "_search", ptr.clone(), false)
                        .expect("_search"),
                ),
                (
                    "_search_elidable",
                    build_search_elidable_helper_graph(
                        &rtyper,
                        "_search_elidable",
                        ptr.clone(),
                        false,
                    )
                    .expect("_search_elidable"),
                ),
                (
                    "_search_normal",
                    build_search_normal_helper_graph(&rtyper, "_search_normal", ptr.clone(), false)
                        .expect("_search_normal"),
                ),
            ];
            for (name, py) in graphs {
                assert_eq!(py.graph.borrow().name, name);
                checkgraph(&py.graph.borrow());
            }
        }
    }

    fn search_normal_has_assertionerror_path(graph: &FunctionGraph) -> bool {
        let start = graph.startblock.borrow();
        if !start
            .operations
            .first()
            .is_some_and(|op| op.opname == "ptr_nonzero")
        {
            return false;
        }
        start.exits.iter().any(|link| {
            let link = link.borrow();
            let is_false = matches!(
                &link.exitcase,
                Some(Hlvalue::Constant(c)) if c.value == ConstValue::Bool(false)
            );
            let to_except = link
                .target
                .as_ref()
                .is_some_and(|target| Rc::ptr_eq(target, &graph.exceptblock));
            let assertionerror = matches!(
                link.args.first().and_then(|arg| arg.as_ref()),
                Some(Hlvalue::Constant(Constant {
                    value: ConstValue::HostObject(obj),
                    ..
                })) if obj.qualname() == "AssertionError"
            );
            is_false && to_except && assertionerror
        })
    }

    /// `_search_normal` (`rstring.py`): `assert value is not None` is
    /// cut when the caller annotation has `can_be_None=False`
    /// (`transform_dead_code`).
    #[test]
    fn search_normal_assert_follows_value_can_be_none() {
        for ptr in [STRPTR.clone(), UNICODEPTR.clone()] {
            for value_can_be_none in [true, false] {
                let ann = RPythonAnnotator::new(None, None, None, false);
                let rtyper = RPythonTyper::new(&ann);
                let py = build_search_normal_helper_graph(
                    &rtyper,
                    "_search_normal",
                    ptr.clone(),
                    value_can_be_none,
                )
                .expect("_search_normal");
                checkgraph(&py.graph.borrow());
                assert_eq!(
                    search_normal_has_assertionerror_path(&py.graph.borrow()),
                    value_can_be_none,
                    "value_can_be_none={value_can_be_none}"
                );
            }
        }
    }

    fn malloc_varsize_is_signed_array_of_len(op: &SpaceOperation, len: i64) -> bool {
        if op.opname != "malloc_varsize" {
            return false;
        }
        let Some(Hlvalue::Constant(ty_c)) = op.args.first() else {
            return false;
        };
        let ConstValue::LowLevelType(lltype) = &ty_c.value else {
            return false;
        };
        let is_signed_array = matches!(
            &**lltype,
            LowLevelType::Array(arr) if arr.OF == LowLevelType::Signed
        );
        let is_len = matches!(
            op.args.get(2),
            Some(Hlvalue::Constant(c)) if matches!(c.value, ConstValue::Int(n) if n == len)
        );
        is_signed_array && is_len
    }

    #[test]
    fn twoway_preprocess_table_is_list_newlist_plus_append() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let prep =
            build_twoway_preprocess_helper_graph(&rtyper, "_twoway_preprocess", STRPTR.clone())
                .expect("_twoway_preprocess");
        let graph = prep.graph.borrow();
        let names = direct_call_names(&graph);
        assert!(
            names.iter().any(|n| n == "ll_newlist"),
            "comprehension starts with ll_newlist: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "ll_append"),
            "comprehension appends not_found_shift: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "ll_setitem_fast"),
            "setitem loop is ll_setitem_fast (nonneg): {names:?}"
        );
        let has_raw_table = graph.iterblocks().into_iter().any(|block| {
            block
                .borrow()
                .operations
                .iter()
                .any(|op| malloc_varsize_is_signed_array_of_len(op, TWOWAY_TABLE_SIZE))
        });
        assert!(
            !has_raw_table,
            "preprocess graph must not malloc_varsize GcArray(Signed) of length {TWOWAY_TABLE_SIZE}"
        );
        checkgraph(&graph);
    }

    fn hl_i64(env: &HashMap<u64, i64>, hlv: &Hlvalue) -> Option<i64> {
        match hlv {
            Hlvalue::Constant(c) => match c.value {
                ConstValue::Int(n) => Some(n),
                ConstValue::Bool(b) => Some(i64::from(b)),
                _ => None,
            },
            Hlvalue::Variable(v) => env.get(&v.id()).copied(),
        }
    }

    fn eval_int_op(opname: &str, args: &[Option<i64>]) -> Option<i64> {
        match (opname, args) {
            ("int_lt", [Some(a), Some(b)]) => Some(i64::from(*a < *b)),
            ("int_gt", [Some(a), Some(b)]) => Some(i64::from(*a > *b)),
            ("int_eq", [Some(a), Some(b)]) => Some(i64::from(*a == *b)),
            ("int_ne", [Some(a), Some(b)]) => Some(i64::from(*a != *b)),
            ("int_le", [Some(a), Some(b)]) => Some(i64::from(*a <= *b)),
            ("int_ge", [Some(a), Some(b)]) => Some(i64::from(*a >= *b)),
            ("int_add", [Some(a), Some(b)]) => Some(*a + *b),
            ("int_sub", [Some(a), Some(b)]) => Some(*a - *b),
            ("int_mul", [Some(a), Some(b)]) => Some(*a * *b),
            ("int_rshift", [Some(a), Some(b)]) => Some(*a >> *b),
            ("int_lshift", [Some(a), Some(b)]) => Some(*a << *b),
            ("int_and", [Some(a), Some(b)]) => Some(*a & *b),
            ("int_or", [Some(a), Some(b)]) => Some(*a | *b),
            ("bool_not", [Some(a)]) => Some(i64::from(*a == 0)),
            ("int_is_zero", [Some(a)]) => Some(i64::from(*a == 0)),
            ("int_is_true", [Some(a)]) => Some(i64::from(*a != 0)),
            _ => None,
        }
    }

    fn follow_forward_dispatch(graph: &FunctionGraph, n: i64, m: i64) -> String {
        let start = graph
            .iterblocks()
            .into_iter()
            .find(|block| {
                block.borrow().operations.iter().any(|op| {
                    op.opname == "int_lt"
                        && matches!(
                            op.args.get(1),
                            Some(Hlvalue::Constant(c)) if matches!(c.value, ConstValue::Int(2500))
                        )
                })
            })
            .expect("n < 2500 dispatch block");
        let start_vars = block_vars(&start);
        let mut env: HashMap<u64, i64> = HashMap::new();
        env.insert(start_vars[5].id(), n);
        env.insert(start_vars[6].id(), m);
        if start_vars.len() > 7 {
            env.insert(start_vars[7].id(), n - m);
        }
        let mut cur = start;
        for _ in 0..64 {
            let operations = cur.borrow().operations.clone();
            for op in &operations {
                if op.opname == "direct_call"
                    && let Some(func) = op.args.first().and_then(funcobj_from_funcptr_constant)
                {
                    match func._name.as_str() {
                        "_default_find" | "_twoway_preprocess" | "_adaptive_find" | "_two_way"
                        | "_two_way_count" => return func._name,
                        _ => {}
                    }
                }
                let args: Vec<Option<i64>> = op.args.iter().map(|a| hl_i64(&env, a)).collect();
                if let (Some(val), Hlvalue::Variable(v)) =
                    (eval_int_op(&op.opname, &args), &op.result)
                {
                    env.insert(v.id(), val);
                }
            }
            let exitswitch = cur.borrow().exitswitch.clone();
            let exits = cur.borrow().exits.clone();
            let link = match exitswitch {
                Some(Hlvalue::Variable(sw)) => {
                    let cond = *env
                        .get(&sw.id())
                        .unwrap_or_else(|| panic!("unbound dispatch cond {}", sw.name()));
                    let want = cond != 0;
                    exits
                        .into_iter()
                        .find(|link| {
                            matches!(
                                &link.borrow().exitcase,
                                Some(Hlvalue::Constant(c))
                                    if matches!(c.value, ConstValue::Bool(b) if b == want)
                            )
                        })
                        .expect("bool exit")
                }
                None if exits.len() == 1 => exits.into_iter().next().unwrap(),
                other => panic!("unexpected exitswitch {other:?}"),
            };
            let (target, args) = {
                let l = link.borrow();
                (l.target.clone().expect("link.target"), l.args.clone())
            };
            if Rc::ptr_eq(&target, &graph.returnblock) {
                panic!("dispatch returned before a search helper call (n={n}, m={m})");
            }
            let tvars = block_vars(&target);
            let mut next = HashMap::new();
            for (arg, tv) in args.iter().zip(tvars.iter()) {
                let Some(arg) = arg else {
                    continue;
                };
                if let Some(val) = hl_i64(&env, arg) {
                    next.insert(tv.id(), val);
                }
            }
            env = next;
            cur = target;
        }
        panic!("dispatch walk exceeded 64 blocks (n={n}, m={m})");
    }

    #[test]
    fn search_normal_forward_dispatch_matches_upstream_thresholds() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let py = build_search_normal_helper_graph(&rtyper, "_search_normal", STRPTR.clone(), false)
            .expect("_search_normal");
        let graph = py.graph.borrow();
        assert_eq!(
            follow_forward_dispatch(&graph, 40000, 3),
            "_default_find",
            "n=40000 m=3 takes `_search` `m < 6` → `_default_find`"
        );
        assert_eq!(
            follow_forward_dispatch(&graph, 40000, 200),
            "_twoway_preprocess",
            "n=40000 m=200 takes two-way"
        );
        assert_eq!(
            follow_forward_dispatch(&graph, 1000, 50),
            "_default_find",
            "n=1000 m=50 takes `_search` `n < 2500` → `_default_find`"
        );
    }
}
