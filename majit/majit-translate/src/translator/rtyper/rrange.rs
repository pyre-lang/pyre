//! RPython `rpython/rtyper/rrange.py` + `lltypesystem/rrange.py` —
//! `AbstractRangeRepr` slice covering the `len(range(...))` / indexing /
//! iteration lowering for constant-step ranges (`range(n)` / `range(a,
//! b)` / `range(a, b, k)` for constant `k != 0`), the forms
//! `builtin_range` mints for the common call shapes
//! (`annotator/builtin.rs`).
//!
//! A `range()` result that is never mutated annotates as a `SomeList`
//! carrying a non-`None` `range_step` (`annotator/listdef.rs`); its
//! repr is NOT array-backed (`FixedSizeListRepr`) but an immutable
//! `GcStruct("range", start, stop)` (`lltypesystem/rrange.py:51-57`).
//!
//! Landed: `rtype_len` for any step (`ll_rangelen1` for `step == 1`, the
//! general `ll_rangelen` / `_ll_rangelen` floor-division otherwise),
//! `rtype_getitem` for all four `(checkidx, nonneg)` combinations
//! (`ll_rangeitem_nonneg` / `ll_rangeitem` and their `_checked` variants,
//! sharing the `_ll_rangelen` length core), the `RANGEST` variable-step
//! `_getstep` (`rtype_len` / `rtype_getitem`), and the `RangeIteratorRepr`
//! (`make_iterator_repr` / `newiter` / `rtype_next`) for both the
//! constant-step `RANGEITER` (`ll_rangenext_up` / `_down`) and the
//! variable-step `RANGESTITER` (`ll_rangeiter` step-field copy +
//! `ll_rangenext_updown`).
//!
//! `rtype_builtin_range` also covers the real-list arm (`ll_range2list`) by
//! reusing the graph-built resized-list helpers from `rlist`.

#![allow(non_camel_case_types)]

use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, Ptr, PtrTarget, Struct};
use crate::translator::rtyper::lltypesystem::rstr::sub_helper_funcptr_constant;
use crate::translator::rtyper::rlist::{
    ListIteratorRepr, ListLayout, ListRepr, build_ll_newlist_helper_graph,
    build_ll_setitem_fast_helper_graph,
};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState, gc_flavor_const};
use crate::translator::rtyper::rtuple::TupleRepr;
use crate::translator::rtyper::rtyper::{
    ConvertedTo, GenopResult, HighLevelOp, LowLevelFunction, RPythonTyper, constant_with_lltype,
    exception_args, helper_pygraph_from_graph, variable_with_lltype,
};
use std::collections::HashMap;
use std::sync::Arc;

fn rrange_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!("rrange.{name} helper surface deferred"))
}

fn signed_const(n: i64) -> Hlvalue {
    constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed)
}

fn bool_const(b: bool) -> Hlvalue {
    constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool)
}

fn void_field(f: &str) -> Hlvalue {
    constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void)
}

/// RPython `_ll_rangelen(start, stop, step)`.
fn _ll_rangelen(start: i64, stop: i64, step: i64) -> i64 {
    let mut result = if step > 0 {
        (stop - start + (step - 1)) / step
    } else {
        (start - stop - (step + 1)) / (-step)
    };
    if result < 0 {
        result = 0;
    }
    result
}

/// RPython `ll_rangelen(l, step)` for any carrier exposing start/stop fields.
pub fn ll_rangelen(start: i64, stop: i64, step: i64) -> i64 {
    _ll_rangelen(start, stop, step)
}

/// RPython `ll_rangelen1(l)` for any carrier exposing start/stop fields.
pub fn ll_rangelen1(start: i64, stop: i64) -> i64 {
    (stop - start).max(0)
}

/// RPython `ll_rangeitem_nonneg(func, l, index, step)` in the `dum_nocheck`
/// case; the checkidx branch is represented by `checked=true`.
pub fn ll_rangeitem_nonneg(
    start: i64,
    stop: i64,
    index: i64,
    step: i64,
    checked: bool,
) -> Result<i64, TyperError> {
    if checked && index >= _ll_rangelen(start, stop, step) {
        return Err(TyperError::message("IndexError"));
    }
    Ok(start + index * step)
}

/// RPython `ll_rangeitem(func, l, index, step)`.
pub fn ll_rangeitem(
    start: i64,
    stop: i64,
    mut index: i64,
    step: i64,
    checked: bool,
) -> Result<i64, TyperError> {
    if checked {
        let length = _ll_rangelen(start, stop, step);
        if index < 0 {
            index += length;
        }
        if index < 0 || index >= length {
            return Err(TyperError::message("IndexError"));
        }
    } else if index < 0 {
        index += _ll_rangelen(start, stop, step);
    }
    Ok(start + index * step)
}

/// RPython `rtype_builtin_range(hop)` (`rrange.py`) — lowers
/// `range(...)`:
///
/// ```python
/// def rtype_builtin_range(hop):
///     vstep = hop.inputconst(Signed, 1)
///     if hop.nb_args == 1:
///         vstart = hop.inputconst(Signed, 0)
///         vstop, = hop.inputargs(Signed)
///     elif hop.nb_args == 2:
///         vstart, vstop = hop.inputargs(Signed, Signed)
///     else:
///         vstart, vstop, vstep = hop.inputargs(Signed, Signed, Signed)
///         if isinstance(vstep, Constant) and vstep.value == 0:
///             raise TyperError("range cannot have a const step of zero")
///     if isinstance(hop.r_result, AbstractRangeRepr):
///         if hop.r_result.step != 0:
///             c_rng = hop.inputconst(Void, hop.r_result.RANGE)
///             hop.exception_is_here()
///             return hop.gendirectcall(hop.r_result.ll_newrange, c_rng, vstart, vstop)
///         else:
///             hop.exception_is_here()
///             return hop.gendirectcall(hop.r_result.ll_newrangest, vstart, vstop, vstep)
///     else:
///         ... ll_range2list ...
/// ```
///
/// The `AbstractRangeRepr` result arm is ported: the constant-step arm
/// (`step != 0`) `direct_call`s `ll_newrange` (`malloc RANGE` + setfield
/// start/stop, `RANGE` folded into the builder), and the variable-step arm
/// (`step == 0`) `direct_call`s `ll_newrangest` (`malloc RANGEST` + setfield
/// start/stop/step, guarded by the runtime `step == 0` → `ValueError`).
///
/// The real-list arm (`ll_range2list`, `rrange.py`) builds the result
/// via the resized [`ListRepr`] helper graphs: `ll_newlist(length)`, followed
/// by a runtime fill loop that calls `ll_setitem_fast(idx, start)`.
pub fn rtype_builtin_range(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    let vstep_one = constant_with_lltype(ConstValue::Int(1), LowLevelType::Signed);
    let (vstart, vstop, vstep) = match hop.nb_args() {
        1 => {
            let args = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Signed)])?;
            (
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
                args[0].clone(),
                vstep_one,
            )
        }
        2 => {
            let args = hop.inputargs(vec![
                ConvertedTo::LowLevelType(&LowLevelType::Signed),
                ConvertedTo::LowLevelType(&LowLevelType::Signed),
            ])?;
            (args[0].clone(), args[1].clone(), vstep_one)
        }
        _ => {
            let args = hop.inputargs(vec![
                ConvertedTo::LowLevelType(&LowLevelType::Signed),
                ConvertedTo::LowLevelType(&LowLevelType::Signed),
                ConvertedTo::LowLevelType(&LowLevelType::Signed),
            ])?;
            if let Hlvalue::Constant(c) = &args[2]
                && matches!(c.value, ConstValue::Int(0))
            {
                return Err(TyperError::message(
                    "range cannot have a const step of zero",
                ));
            }
            (args[0].clone(), args[1].clone(), args[2].clone())
        }
    };

    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_builtin_range: r_result not populated"))?;
    let any: &dyn std::any::Any = r_result.as_ref();
    let Some(range) = any.downcast_ref::<AbstractRangeRepr>() else {
        let Some(r_list) = any.downcast_ref::<ListRepr>() else {
            return Err(rrange_deferred(
                "rtype_builtin_range result is neither AbstractRangeRepr nor ListRepr",
            ));
        };
        let ptr_lltype = r_list.lowleveltype().clone();
        let item_lltype = r_list.item_lowleveltype();
        hop.exception_is_here()?;
        let newlist_fn = {
            let ptr = ptr_lltype.clone();
            let item = item_lltype.clone();
            hop.rtyper.lowlevel_helper_function_with_builder(
                "ll_newlist".to_string(),
                vec![LowLevelType::Signed],
                ptr_lltype.clone(),
                move |_rtyper, _args, _result| {
                    build_ll_newlist_helper_graph(
                        "ll_newlist",
                        ListLayout::Resized,
                        ptr.clone(),
                        item.clone(),
                    )
                },
            )?
        };
        let setitem_fn = if item_lltype == LowLevelType::Void {
            None
        } else {
            let ptr = ptr_lltype.clone();
            let item = item_lltype.clone();
            Some(hop.rtyper.lowlevel_helper_function_with_builder(
                "ll_setitem_fast".to_string(),
                vec![
                    ptr_lltype.clone(),
                    LowLevelType::Signed,
                    item_lltype.clone(),
                ],
                LowLevelType::Void,
                move |_rtyper, _args, _result| {
                    build_ll_setitem_fast_helper_graph("ll_setitem_fast", ptr.clone(), item.clone())
                },
            )?)
        };
        let helper = {
            let ptr = ptr_lltype.clone();
            let item = item_lltype.clone();
            let newlist = newlist_fn.clone();
            let setitem = setitem_fn.clone();
            hop.rtyper.lowlevel_helper_function_with_builder(
                "ll_range2list".to_string(),
                vec![
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                ptr_lltype,
                move |rtyper, _args, _result| {
                    build_ll_range2list_helper_graph(
                        rtyper,
                        "ll_range2list",
                        ptr.clone(),
                        item.clone(),
                        &newlist,
                        setitem.as_ref(),
                    )
                },
            )?
        };
        return hop.gendirectcall(&helper, vec![vstart, vstop, vstep]);
    };
    hop.exception_is_here()?;
    let range_lltype = range.lltype.clone();
    if range.step != 0 {
        let rl = range_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_newrange".to_string(),
            vec![LowLevelType::Signed, LowLevelType::Signed],
            range_lltype,
            move |_rtyper, _args, _result| {
                build_ll_newrange_helper_graph("ll_newrange", rl.clone())
            },
        )?;
        hop.gendirectcall(&helper, vec![vstart, vstop])
    } else {
        let rl = range_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_newrangest".to_string(),
            vec![
                LowLevelType::Signed,
                LowLevelType::Signed,
                LowLevelType::Signed,
            ],
            range_lltype,
            move |_rtyper, _args, _result| {
                build_ll_newrangest_helper_graph("ll_newrangest", rl.clone())
            },
        )?;
        hop.gendirectcall(&helper, vec![vstart, vstop, vstep])
    }
}

/// RPython `rtype_builtin_xrange = rtype_builtin_range`.
pub use rtype_builtin_range as rtype_builtin_xrange;

/// RPython `ll_range2list(LIST, start, stop, step)`; the Rust carrier is the
/// immutable integer payload the helper would write through `ll_setitem_fast`.
pub fn ll_range2list(start: i64, stop: i64, step: i64) -> Result<Vec<i64>, TyperError> {
    if step == 0 {
        return Err(TyperError::message("ValueError"));
    }
    let length = _ll_rangelen(start, stop, step);
    let mut out = Vec::with_capacity(length as usize);
    let mut value = start;
    for _ in 0..length {
        out.push(value);
        value += step;
    }
    Ok(out)
}

/// Synthesise `ll_range2list(LIST, start, stop, step)` (`rrange.py`):
/// zero-step raises `ValueError`, then the helper allocates `LIST.ll_newlist`
/// with `_ll_rangelen(start, stop, step)` and fills it by repeated
/// `ll_setitem_fast(idx, start); start += step; idx += 1`.
fn build_ll_range2list_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    newlist: &LowLevelFunction,
    setitem_fast: Option<&LowLevelFunction>,
) -> Result<PyGraph, TyperError> {
    let rangelen_const = underscore_rangelen_funcptr(rtyper)?;
    let newlist_const = sub_helper_funcptr_constant(rtyper, newlist)?;
    let setitem_const = match setitem_fast {
        Some(func) => Some(sub_helper_funcptr_constant(rtyper, func)?),
        None => None,
    };

    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let stop_arg = variable_with_lltype("stop", LowLevelType::Signed);
    let step_arg = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(stop_arg.clone()),
        Hlvalue::Variable(step_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(step_arg.clone()), signed_const(0)],
        Hlvalue::Variable(is_zero.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));

    let alloc_start = variable_with_lltype("start", LowLevelType::Signed);
    let alloc_stop = variable_with_lltype("stop", LowLevelType::Signed);
    let alloc_step = variable_with_lltype("step", LowLevelType::Signed);
    let alloc_block = Block::shared(vec![
        Hlvalue::Variable(alloc_start.clone()),
        Hlvalue::Variable(alloc_stop.clone()),
        Hlvalue::Variable(alloc_step.clone()),
    ]);

    startblock.closeblock(vec![
        Link::new(
            exception_args("ValueError")?,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(start_arg),
                Hlvalue::Variable(stop_arg),
                Hlvalue::Variable(step_arg),
            ],
            Some(alloc_block.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let length = variable_with_lltype("length", LowLevelType::Signed);
    alloc_block
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(rangelen_const),
                Hlvalue::Variable(alloc_start.clone()),
                Hlvalue::Variable(alloc_stop),
                Hlvalue::Variable(alloc_step.clone()),
            ],
            Hlvalue::Variable(length.clone()),
        ));
    let list = variable_with_lltype("l", ptr_lltype.clone());
    alloc_block
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(newlist_const),
                Hlvalue::Variable(length.clone()),
            ],
            Hlvalue::Variable(list.clone()),
        ));

    if item_lltype == LowLevelType::Void {
        alloc_block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(list)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    } else {
        let loop_l = variable_with_lltype("l", ptr_lltype.clone());
        let loop_idx = variable_with_lltype("idx", LowLevelType::Signed);
        let loop_length = variable_with_lltype("length", LowLevelType::Signed);
        let loop_current = variable_with_lltype("start", LowLevelType::Signed);
        let loop_step = variable_with_lltype("step", LowLevelType::Signed);
        let loop_block = Block::shared(vec![
            Hlvalue::Variable(loop_l.clone()),
            Hlvalue::Variable(loop_idx.clone()),
            Hlvalue::Variable(loop_length.clone()),
            Hlvalue::Variable(loop_current.clone()),
            Hlvalue::Variable(loop_step.clone()),
        ]);
        alloc_block.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(list),
                    signed_const(0),
                    Hlvalue::Variable(length),
                    Hlvalue::Variable(alloc_start),
                    Hlvalue::Variable(alloc_step),
                ],
                Some(loop_block.clone()),
                None,
            )
            .into_ref(),
        ]);

        let should_continue = variable_with_lltype("continue", LowLevelType::Bool);
        loop_block.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(loop_idx.clone()),
                Hlvalue::Variable(loop_length.clone()),
            ],
            Hlvalue::Variable(should_continue.clone()),
        ));
        loop_block.borrow_mut().exitswitch = Some(Hlvalue::Variable(should_continue));

        let fill_l = variable_with_lltype("l", ptr_lltype.clone());
        let fill_idx = variable_with_lltype("idx", LowLevelType::Signed);
        let fill_length = variable_with_lltype("length", LowLevelType::Signed);
        let fill_current = variable_with_lltype("start", LowLevelType::Signed);
        let fill_step = variable_with_lltype("step", LowLevelType::Signed);
        let fill_block = Block::shared(vec![
            Hlvalue::Variable(fill_l.clone()),
            Hlvalue::Variable(fill_idx.clone()),
            Hlvalue::Variable(fill_length.clone()),
            Hlvalue::Variable(fill_current.clone()),
            Hlvalue::Variable(fill_step.clone()),
        ]);
        loop_block.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(loop_l.clone()),
                    Hlvalue::Variable(loop_idx),
                    Hlvalue::Variable(loop_length),
                    Hlvalue::Variable(loop_current),
                    Hlvalue::Variable(loop_step),
                ],
                Some(fill_block.clone()),
                Some(bool_const(true)),
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(loop_l)],
                Some(graph.returnblock.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        ]);

        let setitem_void = variable_with_lltype("v", LowLevelType::Void);
        fill_block.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(
                    setitem_const.expect("non-Void range list must have ll_setitem_fast"),
                ),
                Hlvalue::Variable(fill_l.clone()),
                Hlvalue::Variable(fill_idx.clone()),
                Hlvalue::Variable(fill_current.clone()),
            ],
            Hlvalue::Variable(setitem_void),
        ));
        let next_current = variable_with_lltype("start", LowLevelType::Signed);
        fill_block.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(fill_current),
                Hlvalue::Variable(fill_step.clone()),
            ],
            Hlvalue::Variable(next_current.clone()),
        ));
        let next_idx = variable_with_lltype("idx", LowLevelType::Signed);
        fill_block.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(fill_idx), signed_const(1)],
            Hlvalue::Variable(next_idx.clone()),
        ));
        fill_block.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(fill_l),
                    Hlvalue::Variable(next_idx),
                    Hlvalue::Variable(fill_length),
                    Hlvalue::Variable(next_current),
                    Hlvalue::Variable(fill_step),
                ],
                Some(loop_block),
                None,
            )
            .into_ref(),
        ]);
    }

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["start".to_string(), "stop".to_string(), "step".to_string()],
        func,
    ))
}

/// Lightweight carrier for `ll_rangenext_*` tests and deferred iterator reprs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeIter {
    pub next: i64,
    pub stop: i64,
    pub step: i64,
}

/// RPython `class AbstractRangeIteratorRepr(IteratorRepr)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbstractRangeIteratorRepr {
    pub step: i64,
    pub lowleveltype: LowLevelType,
}

/// RPython `ll_rangenext_up(iter, step)`.
pub fn ll_rangenext_up(iter: &mut RangeIter, step: i64) -> Result<i64, TyperError> {
    let next = iter.next;
    if next >= iter.stop {
        return Err(TyperError::message("StopIteration"));
    }
    iter.next = next + step;
    Ok(next)
}

/// RPython `ll_rangenext_down(iter, step)`.
pub fn ll_rangenext_down(iter: &mut RangeIter, step: i64) -> Result<i64, TyperError> {
    let next = iter.next;
    if next <= iter.stop {
        return Err(TyperError::message("StopIteration"));
    }
    iter.next = next + step;
    Ok(next)
}

/// RPython `ll_rangenext_updown(iter)`.
pub fn ll_rangenext_updown(iter: &mut RangeIter) -> Result<i64, TyperError> {
    let step = iter.step;
    if step > 0 {
        ll_rangenext_up(iter, step)
    } else {
        ll_rangenext_down(iter, step)
    }
}

/// RPython `class EnumerateIteratorRepr(IteratorRepr)` (`rrange.py`):
///
/// ```python
/// class EnumerateIteratorRepr(IteratorRepr):
///     def __init__(self, r_baseiter, const_startindex):
///         self.r_baseiter = r_baseiter
///         self.lowleveltype = r_baseiter.lowleveltype
///         if not hasattr(r_baseiter, 'll_getnextindex'):
///             raise TyperError("not implemented for now: enumerate(x) where x "
///                              "is not a regular list (got %r)" % (r_baseiter,))
///         self.ll_getnextindex = r_baseiter.ll_getnextindex
///         self.const_startindex = const_startindex
/// ```
///
/// Only [`ListIteratorRepr`] carries `ll_getnextindex`, so the `hasattr`
/// test is a downcast to it.
#[derive(Debug)]
pub struct EnumerateIteratorRepr {
    state: ReprState,
    pub r_baseiter: Arc<dyn Repr>,
    lowleveltype: LowLevelType,
    pub const_startindex: Option<i64>,
}

impl EnumerateIteratorRepr {
    pub fn new(
        r_baseiter: Arc<dyn Repr>,
        const_startindex: Option<i64>,
    ) -> Result<Self, TyperError> {
        let any_r: &dyn std::any::Any = r_baseiter.as_ref();
        if any_r.downcast_ref::<ListIteratorRepr>().is_none() {
            return Err(TyperError::message(format!(
                "not implemented for now: enumerate(x) where x is not a regular list (got {})",
                r_baseiter.class_name()
            )));
        }
        let lowleveltype = r_baseiter.lowleveltype().clone();
        Ok(EnumerateIteratorRepr {
            state: ReprState::new(),
            r_baseiter,
            lowleveltype,
            const_startindex,
        })
    }

    fn base_list_iter(&self) -> &ListIteratorRepr {
        let any_r: &dyn std::any::Any = self.r_baseiter.as_ref();
        any_r
            .downcast_ref::<ListIteratorRepr>()
            .expect("checked in EnumerateIteratorRepr::new")
    }
}

impl Repr for EnumerateIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "EnumerateIteratorRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::EnumerateIteratorRepr
    }

    /// RPython `IteratorRepr.rtype_iter(self, hop)` (rmodel.py) —
    /// `iter(iter(x)) <==> iter(x)`.
    fn rtype_iter(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(Some(vlist[0].clone()))
    }

    /// RPython `IteratorRepr.rtype_method_next(self, hop)` (rmodel.py).
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            "next" => self.rtype_next(hop),
            other => Err(TyperError::message(format!(
                "missing EnumerateIteratorRepr.rtype_method_{other}"
            ))),
        }
    }

    /// RPython `EnumerateIteratorRepr.rtype_next(self, hop)` (`rrange.py`):
    ///
    /// ```python
    /// def rtype_next(self, hop):
    ///     v_enumerate, = hop.inputargs(self)
    ///     v_index = hop.gendirectcall(self.ll_getnextindex, v_enumerate)
    ///     if self.const_startindex is not None:
    ///         v_index = hop.llops.genop(
    ///             "int_add",
    ///             [v_index, hop.llops.genconst(self.const_startindex)],
    ///             resulttype=v_index.concretetype)
    ///     hop2 = hop.copy()
    ///     hop2.args_r = [self.r_baseiter]
    ///     r_item_src = self.r_baseiter.external_item_repr
    ///     r_item_dst = hop.r_result.items_r[1]
    ///     v_item = self.r_baseiter.rtype_next(hop2)
    ///     v_item = hop.llops.convertvar(v_item, r_item_src, r_item_dst)
    ///     return hop.r_result.newtuple(hop.llops, hop.r_result,
    ///                                  [v_index, v_item])
    /// ```
    fn rtype_next(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_enumerate = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let r_baseiter = self.base_list_iter();
        let mut v_index = r_baseiter.gen_ll_getnextindex(hop, v_enumerate[0].clone())?;
        if let Some(start) = self.const_startindex {
            v_index = hop
                .genop(
                    "int_add",
                    vec![v_index, signed_const(start)],
                    GenopResult::LLType(LowLevelType::Signed),
                )
                .ok_or_else(|| TyperError::message("int_add produced no result"))?;
        }
        let hop2 = hop.copy();
        *hop2.args_r.borrow_mut() = vec![Some(self.r_baseiter.clone())];
        let r_item_src = r_baseiter.external_item_repr().clone();
        let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
            TyperError::message("EnumerateIteratorRepr.rtype_next: r_result missing")
        })?;
        let any_r: &dyn std::any::Any = r_result.as_ref();
        let r_tuple = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
            TyperError::message("EnumerateIteratorRepr.rtype_next: r_result is not a TupleRepr")
        })?;
        let r_item_dst = r_tuple.items_r[1].clone();
        let v_item = self
            .r_baseiter
            .rtype_next(&hop2)?
            .ok_or_else(|| TyperError::message("base iterator rtype_next returned Void"))?;
        let v_item =
            hop.llops
                .borrow_mut()
                .convertvar(v_item, r_item_src.as_ref(), r_item_dst.as_ref())?;
        Ok(Some(TupleRepr::newtuple(
            &mut hop.llops.borrow_mut(),
            r_tuple,
            vec![v_index, v_item],
        )?))
    }
}

/// RPython `rtype_builtin_enumerate(hop)` (`rrange.py`):
///
/// ```python
/// def rtype_builtin_enumerate(hop):
///     hop.exception_cannot_occur()
///     hop2 = hop.copy()
///     hop2.args_r = [hop.args_r[0]]
///     hop2.args_v = [hop.args_v[0]]
///     hop2.args_s = [hop.args_s[0]]
///     return hop.r_result.r_baseiter.newiter(hop2)
/// ```
pub fn rtype_builtin_enumerate(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    hop.exception_cannot_occur()?;
    let hop2 = hop.copy();
    hop2.args_r.borrow_mut().truncate(1);
    hop2.args_v.borrow_mut().truncate(1);
    hop2.args_s.borrow_mut().truncate(1);
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_builtin_enumerate: r_result missing"))?;
    let any_r: &dyn std::any::Any = r_result.as_ref();
    let r_enum = any_r
        .downcast_ref::<EnumerateIteratorRepr>()
        .ok_or_else(|| {
            TyperError::message("rtype_builtin_enumerate: r_result is not an EnumerateIteratorRepr")
        })?;
    r_enum.r_baseiter.newiter(&hop2)
}

/// RPython `class AbstractRangeRepr(Repr)` (`rrange.py`), with the
/// lltypesystem concrete fields stored here until the concrete module grows
/// its own wrapper type:
///
/// ```python
/// class AbstractRangeRepr(Repr):
///     def __init__(self, step):
///         self.step = step
///         if step != 0:
///             self.lowleveltype = self.RANGE
///         else:
///             self.lowleveltype = self.RANGEST
/// ```
///
/// where (`lltypesystem/rrange.py`):
///
/// ```python
/// self.RANGE = Ptr(GcStruct("range", ("start", Signed), ("stop", Signed),
///                           ..., hints = {'immutable': True}))
/// ```
///
/// The `RANGEST` (variable-step `("start", "stop", "step")`) shape and
/// the `adtmeths` / iterator surface are deferred.
#[derive(Debug)]
pub struct AbstractRangeRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// `self.step` (`rrange.py:12`) — the constant range step. `0`
    /// signals upstream's "variable step" (`RANGEST`).
    step: i64,
}

impl AbstractRangeRepr {
    /// `AbstractRangeRepr.__init__(self, step)` — picks `RANGE`
    /// (constant step) or `RANGEST` (variable step) as the low-level
    /// type. Both are immutable `GcStruct("range", ...)`.
    pub fn new(step: i64) -> Result<Self, TyperError> {
        let signed = LowLevelType::Signed;
        let fields = if step != 0 {
            vec![
                ("start".to_string(), signed.clone()),
                ("stop".to_string(), signed),
            ]
        } else {
            vec![
                ("start".to_string(), signed.clone()),
                ("stop".to_string(), signed.clone()),
                ("step".to_string(), signed),
            ]
        };
        let st = Struct::gc_with_hints(
            "range",
            fields,
            vec![("immutable".to_string(), ConstValue::Bool(true))],
        );
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(st),
        }));
        Ok(AbstractRangeRepr {
            state: ReprState::new(),
            lltype,
            step,
        })
    }

    /// RPython `AbstractRangeRepr._getstep(self, v_rng, hop)`
    /// (`rrange.py:18-20`):
    ///
    /// ```python
    /// def _getstep(self, v_rng, hop):
    ///     return hop.genop(self.getfield_opname,
    ///             [v_rng, hop.inputconst(Void, 'step')], resulttype=Signed)
    /// ```
    ///
    /// `getfield_opname` is `"getfield"` (`lltypesystem/rrange.py`),
    /// reading the runtime `step` field of a variable-step `RANGEST`.
    fn _getstep(&self, v_rng: Hlvalue, hop: &HighLevelOp) -> Result<Hlvalue, TyperError> {
        hop.genop(
            "getfield",
            vec![
                v_rng,
                constant_with_lltype(ConstValue::byte_str("step"), LowLevelType::Void),
            ],
            GenopResult::LLType(LowLevelType::Signed),
        )
        .ok_or_else(|| TyperError::message("AbstractRangeRepr._getstep: genop returned no result"))
    }
}

impl Repr for AbstractRangeRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "RangeRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::RangeRepr
    }

    /// RPython `AbstractRangeRepr.rtype_len(self, hop)`
    /// (`rrange.py:22-30`):
    ///
    /// ```python
    /// def rtype_len(self, hop):
    ///     v_rng, = hop.inputargs(self)
    ///     if self.step == 1:
    ///         return hop.gendirectcall(ll_rangelen1, v_rng)
    ///     elif self.step != 0:
    ///         v_step = hop.inputconst(Signed, self.step)
    ///     else:
    ///         v_step = self._getstep(v_rng, hop)
    ///     return hop.gendirectcall(ll_rangelen, v_rng, v_step)
    /// ```
    ///
    /// `step == 1` → `ll_rangelen1`; any other constant `step != 0` →
    /// `ll_rangelen` with the step baked as the `inputconst(Signed,
    /// self.step)` arg; the variable-step `RANGEST` (`step == 0`) reads its
    /// runtime `step` via `_getstep` and feeds it to the same `ll_rangelen`
    /// (`_ll_rangelen`'s floor-division via `int_floordiv`).
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        let mut args = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        if self.step == 1 {
            let helper = hop.rtyper.lowlevel_helper_function_with_builder(
                "ll_rangelen1".to_string(),
                vec![ptr_lltype],
                LowLevelType::Signed,
                move |_rtyper, _args, _result| {
                    build_ll_rangelen1_helper_graph("ll_rangelen1", ptr_for_builder.clone())
                },
            )?;
            return hop.gendirectcall(&helper, args);
        }
        // v_step: inputconst for a constant step, `_getstep` for variable.
        let v_step = if self.step != 0 {
            constant_with_lltype(ConstValue::Int(self.step), LowLevelType::Signed)
        } else {
            self._getstep(args[0].clone(), hop)?
        };
        args.push(v_step);
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_rangelen".to_string(),
            vec![ptr_lltype, LowLevelType::Signed],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_rangelen_helper_graph("ll_rangelen", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, args)
    }

    /// RPython `pair(AbstractRangeRepr, IntegerRepr).rtype_getitem`
    /// (`rrange.py:34-50`):
    ///
    /// ```python
    /// def rtype_getitem((r_rng, r_int), hop):
    ///     if hop.has_implicit_exception(IndexError):
    ///         spec = dum_checkidx
    ///     else:
    ///         spec = dum_nocheck
    ///     v_func = hop.inputconst(Void, spec)
    ///     v_lst, v_index = hop.inputargs(r_rng, Signed)
    ///     if r_rng.step != 0:
    ///         cstep = hop.inputconst(Signed, r_rng.step)
    ///     else:
    ///         cstep = r_rng._getstep(v_lst, hop)
    ///     if hop.args_s[1].nonneg:
    ///         llfn = ll_rangeitem_nonneg
    ///     else:
    ///         llfn = ll_rangeitem
    ///     hop.exception_is_here()
    ///     return hop.gendirectcall(llfn, v_func, v_lst, v_index, cstep)
    /// ```
    ///
    /// `spec` (`dum_checkidx` / `dum_nocheck`) and `hop.args_s[1].nonneg`
    /// select one of four `func`-folded helpers via [`rangeitem_helper`]: the
    /// nonneg `dum_nocheck` fast path `ll_rangeitem_nonneg`
    /// (`l.start + index * step`, `rrange.py:74-77`), the negative-index
    /// `ll_rangeitem` (`rrange.py`), the bound-checked nonneg
    /// `ll_rangeitem_nonneg` (`rrange.py`), or the full checked
    /// `ll_rangeitem` (`rrange.py`). A constant step is baked as
    /// `inputconst(Signed, self.step)`; a variable-step `RANGEST`
    /// (`step == 0`) reads its runtime `step` via `_getstep`.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::SomeValue;
        let checkidx = hop.has_implicit_exception("IndexError");
        let s1 = hop.args_s.borrow().get(1).cloned().ok_or_else(|| {
            TyperError::message("AbstractRangeRepr.rtype_getitem: args_s[1] missing")
        })?;
        let nonneg = match &s1 {
            SomeValue::Integer(i) => i.nonneg,
            other => {
                return Err(TyperError::message(format!(
                    "AbstractRangeRepr.rtype_getitem: args_s[1] must be SomeInteger, got {other:?}"
                )));
            }
        };
        let mut args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ])?;
        // cstep: inputconst for a constant step, `_getstep` for variable.
        let cstep = if self.step != 0 {
            constant_with_lltype(ConstValue::Int(self.step), LowLevelType::Signed)
        } else {
            self._getstep(args[0].clone(), hop)?
        };
        args.push(cstep);
        hop.exception_is_here()?;
        let helper = rangeitem_helper(&hop.rtyper, checkidx, nonneg, self.lltype.clone())?;
        hop.gendirectcall(&helper, args)
    }

    /// RPython `RangeRepr.make_iterator_repr(self, variant=None)`
    /// (`lltypesystem/rrange.py`):
    ///
    /// ```python
    /// def make_iterator_repr(self, variant=None):
    ///     if variant is not None:
    ///         raise TyperError("unsupported %r iterator over a range list" %
    ///                          (variant,))
    ///     return RangeIteratorRepr(self)
    /// ```
    ///
    /// Both the constant-step (`step != 0` → `RANGEITER`) and the
    /// variable-step (`step == 0` → `RANGESTITER`) iterators are built by
    /// [`RangeIteratorRepr::new`]. `foldable` is unused: a range iterator
    /// synthesises its values arithmetically, so there is no element load to
    /// fold (unlike the list iterator).
    fn make_iterator_repr(
        &self,
        variant: &[String],
        _foldable: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        if !variant.is_empty() {
            return Err(TyperError::message(format!(
                "unsupported {variant:?} iterator over a range list"
            )));
        }
        Ok(Arc::new(RangeIteratorRepr::new(self)?))
    }
}

/// Synthesise the `ll_rangelen1` helper graph (`rrange.py`):
///
/// ```python
/// def ll_rangelen1(l):
///     result = l.stop - l.start
///     if result < 0:
///         result = 0
///     return result
/// ```
///
/// 2-block CFG:
/// - **start**: `start = getfield(l, 'start'); stop = getfield(l, 'stop');
///   result = int_sub(stop, start); neg = int_lt(result, 0)`. Switch on
///   `neg`: True → returnblock with const `0`; False → returnblock with
///   `result`.
pub(crate) fn build_ll_rangelen1_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("l", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let field_const = |f: &str| constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);

    // start block: result = l.stop - l.start; neg = result < 0.
    let v_start = variable_with_lltype("start", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(arg.clone()), field_const("start")],
        Hlvalue::Variable(v_start.clone()),
    ));
    let v_stop = variable_with_lltype("stop", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(arg), field_const("stop")],
        Hlvalue::Variable(v_stop.clone()),
    ));
    let v_result = variable_with_lltype("result", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(v_stop), Hlvalue::Variable(v_start)],
        Hlvalue::Variable(v_result.clone()),
    ));
    let v_neg = variable_with_lltype("neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(v_result.clone()), signed_const(0)],
        Hlvalue::Variable(v_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_neg));

    // True (result < 0): clamp to 0.
    let true_link = Link::new(
        vec![signed_const(0)],
        Some(graph.returnblock.clone()),
        Some(bool_const(true)),
    )
    .into_ref();
    // False: return result unchanged.
    let false_link = Link::new(
        vec![Hlvalue::Variable(v_result)],
        Some(graph.returnblock.clone()),
        Some(bool_const(false)),
    )
    .into_ref();
    startblock.closeblock(vec![true_link, false_link]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string()],
        func,
    ))
}

/// Synthesise the general `ll_rangelen` helper graph (`rrange.py`
/// delegating to `_ll_rangelen`, `rrange.py`):
///
/// ```python
/// def ll_rangelen(l, step):
///     return _ll_rangelen(l.start, l.stop, step)
///
/// def _ll_rangelen(start, stop, step):
///     if step > 0:
///         result = (stop - start + (step - 1)) // step
///     else:
///         result = (start - stop - (step + 1)) // (-step)
///     if result < 0:
///         result = 0
///     return result
/// ```
///
/// 4-block CFG. `step` is a runtime arg (the `inputconst(Signed,
/// self.step)` the caller appends), so both sign arms are emitted and the
/// branch folds once the constant step is inlined. The floor-divisions
/// run on non-negative operands, so `int_floordiv` (truncating) matches
/// Python's `//`:
/// - **start** `(l, step)`: `start = getfield(l, 'start'); stop =
///   getfield(l, 'stop'); pos = int_gt(step, 0)`. Switch on `pos`: True →
///   pos_arm, False → neg_arm; both carry `start, stop, step`.
/// - **pos_arm** `(start, stop, step)`: `result = int_floordiv(int_add(
///   int_sub(stop, start), int_sub(step, 1)), step)` → clamp.
/// - **neg_arm** `(start, stop, step)`: `result = int_floordiv(int_sub(
///   int_sub(start, stop), int_add(step, 1)), int_neg(step))` → clamp.
/// - **clamp** `(result)`: `neg = int_lt(result, 0)`; True →
///   returnblock(`0`), False → returnblock(`result`).
pub(crate) fn build_ll_rangelen_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let step_arg = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(step_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // start = l.start; stop = l.stop.
    let v_start = variable_with_lltype("start", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg.clone()), void_field("start")],
        Hlvalue::Variable(v_start.clone()),
    ));
    let v_stop = variable_with_lltype("stop", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg), void_field("stop")],
        Hlvalue::Variable(v_stop.clone()),
    ));

    emit_rangelen_body(
        &graph.returnblock,
        &startblock,
        Hlvalue::Variable(v_start),
        Hlvalue::Variable(v_stop),
        Hlvalue::Variable(step_arg),
    );

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "step".to_string()],
        func,
    ))
}

/// Synthesise the `_ll_rangelen(start, stop, step)` helper graph
/// (`rrange.py`) — the shared length core that `ll_rangelen`
/// (`rrange.py`) and the checked / negative-index `ll_rangeitem`
/// variants `direct_call`. `start`/`stop`/`step` are runtime args; the body is
/// [`emit_rangelen_body`].
pub(crate) fn build_underscore_ll_rangelen_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let stop_arg = variable_with_lltype("stop", LowLevelType::Signed);
    let step_arg = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(stop_arg.clone()),
        Hlvalue::Variable(step_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    emit_rangelen_body(
        &graph.returnblock,
        &startblock,
        Hlvalue::Variable(start_arg),
        Hlvalue::Variable(stop_arg),
        Hlvalue::Variable(step_arg),
    );

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["start".to_string(), "stop".to_string(), "step".to_string()],
        func,
    ))
}

/// Append the `_ll_rangelen(start, stop, step)` body (`rrange.py`) onto
/// `startblock`, whose `start`/`stop`/`step` operands are already available.
/// Branches on `int_gt(step, 0)`, runs the matching floor-division arm
/// `(stop - start + (step - 1)) // step` or `(start - stop - (step + 1)) //
/// (-step)`, then clamps a negative result to `0`, linking the length into
/// `returnblock`. The floor-divisions run on non-negative operands, so
/// `int_floordiv` (truncating) matches Python's `//`.
fn emit_rangelen_body(
    returnblock: &BlockRef,
    startblock: &BlockRef,
    start: Hlvalue,
    stop: Hlvalue,
    step: Hlvalue,
) {
    // pos = step > 0.
    let v_pos = variable_with_lltype("pos", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![step.clone(), signed_const(0)],
        Hlvalue::Variable(v_pos.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_pos));

    // pos_arm / neg_arm params: (start, stop, step).
    let p_start = variable_with_lltype("start", LowLevelType::Signed);
    let p_stop = variable_with_lltype("stop", LowLevelType::Signed);
    let p_step = variable_with_lltype("step", LowLevelType::Signed);
    let pos_arm = Block::shared(vec![
        Hlvalue::Variable(p_start.clone()),
        Hlvalue::Variable(p_stop.clone()),
        Hlvalue::Variable(p_step.clone()),
    ]);
    let n_start = variable_with_lltype("start", LowLevelType::Signed);
    let n_stop = variable_with_lltype("stop", LowLevelType::Signed);
    let n_step = variable_with_lltype("step", LowLevelType::Signed);
    let neg_arm = Block::shared(vec![
        Hlvalue::Variable(n_start.clone()),
        Hlvalue::Variable(n_stop.clone()),
        Hlvalue::Variable(n_step.clone()),
    ]);

    // clamp param: (result).
    let c_result = variable_with_lltype("result", LowLevelType::Signed);
    let clamp = Block::shared(vec![Hlvalue::Variable(c_result.clone())]);

    startblock.closeblock(vec![
        Link::new(
            vec![start.clone(), stop.clone(), step.clone()],
            Some(pos_arm.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![start, stop, step],
            Some(neg_arm.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // pos_arm: result = (stop - start + (step - 1)) // step.
    let pos_result = variable_with_lltype("result", LowLevelType::Signed);
    {
        let mut b = pos_arm.borrow_mut();
        let diff = variable_with_lltype("diff", LowLevelType::Signed);
        b.operations.push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(p_stop.clone()),
                Hlvalue::Variable(p_start.clone()),
            ],
            Hlvalue::Variable(diff.clone()),
        ));
        let sm1 = variable_with_lltype("sm1", LowLevelType::Signed);
        b.operations.push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(p_step.clone()), signed_const(1)],
            Hlvalue::Variable(sm1.clone()),
        ));
        let num = variable_with_lltype("num", LowLevelType::Signed);
        b.operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(diff), Hlvalue::Variable(sm1)],
            Hlvalue::Variable(num.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "int_floordiv",
            vec![Hlvalue::Variable(num), Hlvalue::Variable(p_step)],
            Hlvalue::Variable(pos_result.clone()),
        ));
    }
    pos_arm.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(pos_result)],
            Some(clamp.clone()),
            None,
        )
        .into_ref(),
    ]);

    // neg_arm: result = (start - stop - (step + 1)) // (-step).
    let neg_result = variable_with_lltype("result", LowLevelType::Signed);
    {
        let mut b = neg_arm.borrow_mut();
        let diff = variable_with_lltype("diff", LowLevelType::Signed);
        b.operations.push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(n_start.clone()),
                Hlvalue::Variable(n_stop.clone()),
            ],
            Hlvalue::Variable(diff.clone()),
        ));
        let sp1 = variable_with_lltype("sp1", LowLevelType::Signed);
        b.operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(n_step.clone()), signed_const(1)],
            Hlvalue::Variable(sp1.clone()),
        ));
        let num = variable_with_lltype("num", LowLevelType::Signed);
        b.operations.push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(diff), Hlvalue::Variable(sp1)],
            Hlvalue::Variable(num.clone()),
        ));
        let nstep = variable_with_lltype("nstep", LowLevelType::Signed);
        b.operations.push(SpaceOperation::new(
            "int_neg",
            vec![Hlvalue::Variable(n_step)],
            Hlvalue::Variable(nstep.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "int_floordiv",
            vec![Hlvalue::Variable(num), Hlvalue::Variable(nstep)],
            Hlvalue::Variable(neg_result.clone()),
        ));
    }
    neg_arm.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(neg_result)],
            Some(clamp.clone()),
            None,
        )
        .into_ref(),
    ]);

    // clamp: if result < 0: result = 0.
    let v_neg = variable_with_lltype("neg", LowLevelType::Bool);
    clamp.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(c_result.clone()), signed_const(0)],
        Hlvalue::Variable(v_neg.clone()),
    ));
    clamp.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_neg));
    clamp.closeblock(vec![
        Link::new(
            vec![signed_const(0)],
            Some(returnblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(c_result)],
            Some(returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);
}

/// Synthesise the `ll_rangeitem_nonneg` helper graph (`rrange.py`)
/// for the `dum_nocheck` case (the `dum_checkidx` IndexError branch is
/// folded out):
///
/// ```python
/// def ll_rangeitem_nonneg(func, l, index, step):
///     ...
///     return l.start + index * step
/// ```
///
/// Single block: `start = getfield(l, 'start'); prod = int_mul(index,
/// step); result = int_add(start, prod)`. `step` is a runtime arg (the
/// `inputconst(Signed, self.step)` the caller appends).
pub(crate) fn build_ll_rangeitem_nonneg_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let step_arg = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(index_arg.clone()),
        Hlvalue::Variable(step_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_result = emit_range_formula(
        &startblock,
        &l_arg,
        Hlvalue::Variable(index_arg),
        Hlvalue::Variable(step_arg),
    );
    startblock.closeblock(vec![
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
        vec!["l".to_string(), "index".to_string(), "step".to_string()],
        func,
    ))
}

/// Push the `l.start + index * step` tail (`rrange.py:77`) onto `block` and
/// return the Signed result var: `start = getfield(l, 'start'); prod =
/// int_mul(index, step); result = int_add(start, prod)`.
fn emit_range_formula(block: &BlockRef, l: &Variable, index: Hlvalue, step: Hlvalue) -> Variable {
    let v_start = variable_with_lltype("start", LowLevelType::Signed);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l.clone()), void_field("start")],
        Hlvalue::Variable(v_start.clone()),
    ));
    let v_prod = variable_with_lltype("prod", LowLevelType::Signed);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "int_mul",
        vec![index, step],
        Hlvalue::Variable(v_prod.clone()),
    ));
    let v_result = variable_with_lltype("result", LowLevelType::Signed);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(v_start), Hlvalue::Variable(v_prod)],
        Hlvalue::Variable(v_result.clone()),
    ));
    v_result
}

/// Build (or retrieve cached) the `_ll_rangelen` sub-helper and return a
/// funcptr `Constant` to `direct_call` it from the checked / negative-index
/// `ll_rangeitem` variants (`rrange.py`).
fn underscore_rangelen_funcptr(rtyper: &RPythonTyper) -> Result<Constant, TyperError> {
    let inner = rtyper.lowlevel_helper_function_with_builder(
        "_ll_rangelen".to_string(),
        vec![
            LowLevelType::Signed,
            LowLevelType::Signed,
            LowLevelType::Signed,
        ],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| build_underscore_ll_rangelen_helper_graph("_ll_rangelen"),
    )?;
    sub_helper_funcptr_constant(rtyper, &inner)
}

/// Push `length = _ll_rangelen(l.start, l.stop, step)` onto `block` and return
/// the Signed length var: `start = getfield(l, 'start'); stop = getfield(l,
/// 'stop'); length = direct_call(_ll_rangelen, start, stop, step)`.
fn emit_range_length(
    block: &BlockRef,
    l: &Variable,
    step: Hlvalue,
    rangelen: &Constant,
) -> Variable {
    let v_start = variable_with_lltype("start", LowLevelType::Signed);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l.clone()), void_field("start")],
        Hlvalue::Variable(v_start.clone()),
    ));
    let v_stop = variable_with_lltype("stop", LowLevelType::Signed);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l.clone()), void_field("stop")],
        Hlvalue::Variable(v_stop.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(rangelen.clone()),
            Hlvalue::Variable(v_start),
            Hlvalue::Variable(v_stop),
            step,
        ],
        Hlvalue::Variable(length.clone()),
    ));
    length
}

/// `ll_rangeitem(func, l, index, step)` with `func is dum_nocheck`
/// (`rrange.py:79-90`, else arm): negative index, no bound check —
/// `if index < 0: index += _ll_rangelen(l.start, l.stop, step); return l.start
/// + index * step`. 3-block CFG (start → block_neg_fix → block_dispatch)
/// forwarding the possibly-fixed index to the inline `l.start + index*step`.
fn build_ll_rangeitem_neg_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let rangelen = underscore_rangelen_funcptr(rtyper)?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let step = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(step.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_fix = variable_with_lltype("l", ptr_lltype.clone());
    let i_fix = variable_with_lltype("index", LowLevelType::Signed);
    let step_fix = variable_with_lltype("step", LowLevelType::Signed);
    let block_neg_fix = Block::shared(vec![
        Hlvalue::Variable(l_fix.clone()),
        Hlvalue::Variable(i_fix.clone()),
        Hlvalue::Variable(step_fix.clone()),
    ]);

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let step_disp = variable_with_lltype("step", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
        Hlvalue::Variable(step_disp.clone()),
    ]);

    // start: is_neg = int_lt(index, 0); branch.
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l.clone()),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(step.clone()),
            ],
            Some(block_neg_fix.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(i),
                Hlvalue::Variable(step),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_neg_fix: length = _ll_rangelen(...); i_fixed = index + length.
    let length = emit_range_length(
        &block_neg_fix,
        &l_fix,
        Hlvalue::Variable(step_fix.clone()),
        &rangelen,
    );
    let i_fixed = variable_with_lltype("index", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_fix), Hlvalue::Variable(length)],
            Hlvalue::Variable(i_fixed.clone()),
        ));
    block_neg_fix.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_fix),
                Hlvalue::Variable(i_fixed),
                Hlvalue::Variable(step_fix),
            ],
            Some(block_dispatch.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_dispatch: result = l.start + index*step.
    let result = emit_range_formula(
        &block_dispatch,
        &l_disp,
        Hlvalue::Variable(i_disp),
        Hlvalue::Variable(step_disp),
    );
    block_dispatch.closeblock(vec![
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
        vec!["l".to_string(), "index".to_string(), "step".to_string()],
        func,
    ))
}

/// `ll_rangeitem_nonneg(func, l, index, step)` with `func is dum_checkidx`
/// (`rrange.py:74-77`): nonneg index, bound check — `if index >=
/// _ll_rangelen(l.start, l.stop, step): raise IndexError; return l.start +
/// index * step`. 2-block CFG plus `graph.exceptblock`.
fn build_ll_rangeitem_nonneg_checked_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let rangelen = underscore_rangelen_funcptr(rtyper)?;
    let exc_args = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let step = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(step.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let step_disp = variable_with_lltype("step", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
        Hlvalue::Variable(step_disp.clone()),
    ]);

    // start: length = _ll_rangelen(...); oob = int_ge(index, length); branch.
    let length = emit_range_length(&startblock, &l, Hlvalue::Variable(step.clone()), &rangelen);
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(i.clone()), Hlvalue::Variable(length)],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(i),
                Hlvalue::Variable(step),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_dispatch: result = l.start + index*step.
    let result = emit_range_formula(
        &block_dispatch,
        &l_disp,
        Hlvalue::Variable(i_disp),
        Hlvalue::Variable(step_disp),
    );
    block_dispatch.closeblock(vec![
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
        vec!["l".to_string(), "index".to_string(), "step".to_string()],
        func,
    ))
}

/// `ll_rangeitem(func, l, index, step)` with `func is dum_checkidx`
/// (`rrange.py`, then arm): `length = _ll_rangelen(l.start, l.stop,
/// step); if index < 0: index += length; if index < 0 or index >= length:
/// raise IndexError; return l.start + index * step`. 5-block CFG (start →
/// block_add → block_check_low → block_check_high → block_dispatch) plus
/// `graph.exceptblock`.
fn build_ll_rangeitem_checked_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let rangelen = underscore_rangelen_funcptr(rtyper)?;
    let exc_low = exception_args("IndexError")?;
    let exc_high = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let step = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(step.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_add params: (l, index, step, length).
    let l_add = variable_with_lltype("l", ptr_lltype.clone());
    let i_add = variable_with_lltype("index", LowLevelType::Signed);
    let step_add = variable_with_lltype("step", LowLevelType::Signed);
    let len_add = variable_with_lltype("length", LowLevelType::Signed);
    let block_add = Block::shared(vec![
        Hlvalue::Variable(l_add.clone()),
        Hlvalue::Variable(i_add.clone()),
        Hlvalue::Variable(step_add.clone()),
        Hlvalue::Variable(len_add.clone()),
    ]);

    // block_check_low params: (l, index, step, length).
    let l_lo = variable_with_lltype("l", ptr_lltype.clone());
    let i_lo = variable_with_lltype("index", LowLevelType::Signed);
    let step_lo = variable_with_lltype("step", LowLevelType::Signed);
    let len_lo = variable_with_lltype("length", LowLevelType::Signed);
    let block_check_low = Block::shared(vec![
        Hlvalue::Variable(l_lo.clone()),
        Hlvalue::Variable(i_lo.clone()),
        Hlvalue::Variable(step_lo.clone()),
        Hlvalue::Variable(len_lo.clone()),
    ]);

    // block_check_high params: (l, index, step, length).
    let l_hi = variable_with_lltype("l", ptr_lltype.clone());
    let i_hi = variable_with_lltype("index", LowLevelType::Signed);
    let step_hi = variable_with_lltype("step", LowLevelType::Signed);
    let len_hi = variable_with_lltype("length", LowLevelType::Signed);
    let block_check_high = Block::shared(vec![
        Hlvalue::Variable(l_hi.clone()),
        Hlvalue::Variable(i_hi.clone()),
        Hlvalue::Variable(step_hi.clone()),
        Hlvalue::Variable(len_hi.clone()),
    ]);

    // block_dispatch params: (l, index, step).
    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let step_disp = variable_with_lltype("step", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
        Hlvalue::Variable(step_disp.clone()),
    ]);

    // start: length = _ll_rangelen(...); is_neg = int_lt(index, 0); branch.
    let length = emit_range_length(&startblock, &l, Hlvalue::Variable(step.clone()), &rangelen);
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l.clone()),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(step.clone()),
                Hlvalue::Variable(length.clone()),
            ],
            Some(block_add.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(i),
                Hlvalue::Variable(step),
                Hlvalue::Variable(length),
            ],
            Some(block_check_low.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_add: index += length. -> block_check_low.
    let i_added = variable_with_lltype("index", LowLevelType::Signed);
    block_add.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(i_add), Hlvalue::Variable(len_add.clone())],
        Hlvalue::Variable(i_added.clone()),
    ));
    block_add.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_add),
                Hlvalue::Variable(i_added),
                Hlvalue::Variable(step_add),
                Hlvalue::Variable(len_add),
            ],
            Some(block_check_low.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_check_low: if index < 0: raise IndexError; else check_high.
    let lo = variable_with_lltype("lo", LowLevelType::Bool);
    block_check_low
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![Hlvalue::Variable(i_lo.clone()), signed_const(0)],
            Hlvalue::Variable(lo.clone()),
        ));
    block_check_low.borrow_mut().exitswitch = Some(Hlvalue::Variable(lo));
    block_check_low.closeblock(vec![
        Link::new(
            exc_low,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l_lo),
                Hlvalue::Variable(i_lo),
                Hlvalue::Variable(step_lo),
                Hlvalue::Variable(len_lo),
            ],
            Some(block_check_high.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_check_high: if index >= length: raise IndexError; else dispatch.
    let hi = variable_with_lltype("hi", LowLevelType::Bool);
    block_check_high
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![Hlvalue::Variable(i_hi.clone()), Hlvalue::Variable(len_hi)],
            Hlvalue::Variable(hi.clone()),
        ));
    block_check_high.borrow_mut().exitswitch = Some(Hlvalue::Variable(hi));
    block_check_high.closeblock(vec![
        Link::new(
            exc_high,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l_hi),
                Hlvalue::Variable(i_hi),
                Hlvalue::Variable(step_hi),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_dispatch: result = l.start + index*step.
    let result = emit_range_formula(
        &block_dispatch,
        &l_disp,
        Hlvalue::Variable(i_disp),
        Hlvalue::Variable(step_disp),
    );
    block_dispatch.closeblock(vec![
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
        vec!["l".to_string(), "index".to_string(), "step".to_string()],
        func,
    ))
}

/// Build (or retrieve cached) the range-getitem helper for the `(checkidx,
/// nonneg)` combination (`rrange.py:34-50`), selecting the `dum_nocheck`
/// nonneg fast path (`ll_rangeitem_nonneg`), the `dum_nocheck` negative-index
/// `ll_rangeitem`, the `dum_checkidx` nonneg `ll_rangeitem_nonneg`, or the full
/// checked `ll_rangeitem`. `func` is folded at build time, so each combination
/// gets its own helper name to keep the `(name, args, result)` cache keys
/// distinct.
fn rangeitem_helper(
    rtyper: &RPythonTyper,
    checkidx: bool,
    nonneg: bool,
    ptr_lltype: LowLevelType,
) -> Result<LowLevelFunction, TyperError> {
    let name = match (checkidx, nonneg) {
        (false, true) => "ll_rangeitem_nonneg",
        (false, false) => "ll_rangeitem",
        (true, true) => "ll_rangeitem_nonneg_checked",
        (true, false) => "ll_rangeitem_checked",
    };
    let name_owned = name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    rtyper.lowlevel_helper_function_with_builder(
        name.to_string(),
        vec![ptr_lltype, LowLevelType::Signed, LowLevelType::Signed],
        LowLevelType::Signed,
        move |rtyper_inner, _args, _result| match (checkidx, nonneg) {
            (false, true) => {
                build_ll_rangeitem_nonneg_helper_graph(&name_owned, ptr_for_builder.clone())
            }
            (false, false) => build_ll_rangeitem_neg_helper_graph(
                rtyper_inner,
                &name_owned,
                ptr_for_builder.clone(),
            ),
            (true, true) => build_ll_rangeitem_nonneg_checked_helper_graph(
                rtyper_inner,
                &name_owned,
                ptr_for_builder.clone(),
            ),
            (true, false) => build_ll_rangeitem_checked_helper_graph(
                rtyper_inner,
                &name_owned,
                ptr_for_builder.clone(),
            ),
        },
    )
}

/// RPython `class RangeIteratorRepr(AbstractRangeIteratorRepr)`
/// (`rrange.py:145-156` + `lltypesystem/rrange.py`), covering both  allow-line-citation
/// the constant-step (`step != 0`, `RANGEITER`) and variable-step
/// (`step == 0`, `RANGESTITER`) shapes.
///
/// ```python
/// class AbstractRangeIteratorRepr(IteratorRepr):
///     def __init__(self, r_rng):
///         self.r_rng = r_rng
///         if r_rng.step != 0:
///             self.lowleveltype = r_rng.RANGEITER
///         else:
///             self.lowleveltype = r_rng.RANGESTITER
/// ```
///
/// where (`lltypesystem/rrange.py`) `RANGEITER = Ptr(GcStruct("range",
/// ("next", Signed), ("stop", Signed)))`. Like
/// [`super::rtuple::Length1TupleIteratorRepr`], pyre collapses the
/// abstract/concrete split into one concrete repr.
///
/// Following [`super::rlist::ListIteratorRepr`], the iterator stores the
/// *data* its `ll_rangeiter` helper consumes (`range_lltype` + `step`)
/// rather than the source `r_rng` repr object: `make_iterator_repr` is a
/// `&self` method on the range repr and cannot reproduce the `Arc<dyn
/// Repr>` upstream's `self.r_rng` field holds. The `newiter` conversion
/// target is therefore `hop.args_r[0]` (the operand's own range repr), so
/// `convertvar`'s repr-identity short-circuit (`rtyper.py`) fires —
/// there is no `RangeRepr -> RangeRepr` conversion, so a rebuilt /
/// non-identical range repr would fail to convert.
#[derive(Debug)]
pub struct RangeIteratorRepr {
    /// `self.r_rng.lowleveltype` — the source `RANGE` / `RANGEST` struct
    /// the `ll_rangeiter` helper reads `start` / `stop` (/ `step`) from.
    range_lltype: LowLevelType,
    /// `self.r_rng.step` (`rrange.py:147`) — the range step. Nonzero
    /// selects `ll_rangenext_up` / `_down` (baked as the advance arg); `0`
    /// is variable step, selecting `ll_rangenext_updown` (reads
    /// `iter.step`) and the `RANGESTITER` iter struct shape.
    step: i64,
    /// `self.lowleveltype = r_rng.RANGEITER` / `RANGESTITER`
    /// (`lltypesystem/rrange.py`) — `Ptr(GcStruct("range", ("next",
    /// Signed), ("stop", Signed)[, ("step", Signed)]))`. The `step` field
    /// is present only for the variable-step `RANGESTITER`.
    lowleveltype: LowLevelType,
    state: ReprState,
}

impl RangeIteratorRepr {
    /// RPython `AbstractRangeIteratorRepr.__init__(self, r_rng)`
    /// (`rrange.py`): `RANGEITER` (`step != 0`) or `RANGESTITER`
    /// (`step == 0`, with the extra runtime `step` field).
    pub fn new(r_rng: &AbstractRangeRepr) -> Result<Self, TyperError> {
        // RANGEITER = GcStruct("range", ("next", Signed), ("stop", Signed));
        // RANGESTITER adds ("step", Signed). Mutable (no immutable hint):
        // `ll_rangenext_*` writes `iter.next`.
        let signed = LowLevelType::Signed;
        let mut fields = vec![
            ("next".to_string(), signed.clone()),
            ("stop".to_string(), signed.clone()),
        ];
        if r_rng.step == 0 {
            fields.push(("step".to_string(), signed));
        }
        let st = Struct::gc("range", fields);
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(st),
        }));
        Ok(RangeIteratorRepr {
            range_lltype: r_rng.lowleveltype().clone(),
            step: r_rng.step,
            lowleveltype,
            state: ReprState::new(),
        })
    }

    /// Whether this iterator's struct carries a runtime `step` field
    /// (the variable-step `RANGESTITER`).
    fn is_variable_step(&self) -> bool {
        self.step == 0
    }
}

impl Repr for RangeIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "RangeIteratorRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::RangeIteratorRepr
    }

    /// RPython `IteratorRepr.rtype_iter(self, hop)` (rmodel.py) —
    /// `iter(iter(x)) <==> iter(x)`: an iterator is its own iterator, so
    /// the op is the identity on the receiver (mirroring
    /// [`super::rlist::ListIteratorRepr::rtype_iter`]).
    fn rtype_iter(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(Some(vlist[0].clone()))
    }

    /// RPython `IteratorRepr.rtype_method_next(self, hop)`
    /// (rmodel.py) — `iter.next()` delegates to `rtype_next`.
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            "next" => self.rtype_next(hop),
            other => Err(TyperError::message(format!(
                "missing RangeIteratorRepr.rtype_method_{other}"
            ))),
        }
    }

    /// RPython `AbstractRangeIteratorRepr.newiter(self, hop)`
    /// (`rrange.py:153-156`):
    ///
    /// ```python
    /// def newiter(self, hop):
    ///     v_rng, = hop.inputargs(self.r_rng)
    ///     citerptr = hop.inputconst(Void, self.lowleveltype)
    ///     return hop.gendirectcall(self.ll_rangeiter, citerptr, v_rng)
    /// ```
    ///
    /// As with [`super::rlist::ListIteratorRepr::newiter`], the iterator
    /// low-level type is baked into the helper builder rather than threaded
    /// as a `citerptr` Void const, and the `hop.inputargs(self.r_rng)`
    /// conversion target is the operand's own range repr (`hop.args_r[0]`):
    /// `convertvar` short-circuits only on repr-object identity, and there
    /// is no `RangeRepr -> RangeRepr` conversion, so the operand repr — not
    /// a rebuilt copy — must be the target. The baked range struct lltype
    /// comes from the iterator's self-contained `range_lltype`.
    fn newiter(&self, hop: &HighLevelOp) -> RTypeResult {
        let r_rng = {
            let args_r = hop.args_r.borrow();
            args_r.first().and_then(|o| o.clone()).ok_or_else(|| {
                TyperError::message("RangeIteratorRepr.newiter: arg0 repr missing")
            })?
        };
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_rng.as_ref())])?;
        let range_lltype = self.range_lltype.clone();
        let iter_lltype = self.lowleveltype.clone();
        let range_for_builder = range_lltype.clone();
        let iter_for_builder = iter_lltype.clone();
        let variable_step = self.is_variable_step();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_rangeiter".to_string(),
            vec![range_lltype],
            iter_lltype,
            move |_rtyper, _args, _result| {
                build_ll_rangeiter_helper_graph(
                    "ll_rangeiter",
                    range_for_builder.clone(),
                    iter_for_builder.clone(),
                    variable_step,
                )
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// RPython `AbstractRangeIteratorRepr.rtype_next(self, hop)`
    /// (`rrange.py:158-170`):
    ///
    /// ```python
    /// def rtype_next(self, hop):
    ///     v_iter, = hop.inputargs(self)
    ///     args = hop.inputconst(Signed, self.r_rng.step),
    ///     if self.r_rng.step > 0:
    ///         llfn = ll_rangenext_up
    ///     elif self.r_rng.step < 0:
    ///         llfn = ll_rangenext_down
    ///     else:
    ///         llfn = ll_rangenext_updown
    ///         args = ()
    ///     hop.has_implicit_exception(StopIteration)
    ///     hop.exception_is_here()
    ///     return hop.gendirectcall(llfn, v_iter, *args)
    /// ```
    ///
    /// `step > 0` → `ll_rangenext_up`, `step < 0` → `ll_rangenext_down`
    /// (both bake the constant step as the advance arg); `step == 0`
    /// (variable) → `ll_rangenext_updown`, which reads `iter.step` and
    /// takes no extra arg.
    fn rtype_next(&self, hop: &HighLevelOp) -> RTypeResult {
        let mut args = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        // upstream: `args = inputconst(Signed, step),` for nonzero step,
        // `args = ()` for variable step (updown reads iter.step).
        if !self.is_variable_step() {
            args.push(constant_with_lltype(
                ConstValue::Int(self.step),
                LowLevelType::Signed,
            ));
        }
        hop.has_implicit_exception("StopIteration");
        hop.exception_is_here()?;
        let iter_lltype = self.lowleveltype.clone();
        let iter_for_builder = iter_lltype.clone();
        if self.is_variable_step() {
            let helper = hop.rtyper.lowlevel_helper_function_with_builder(
                "ll_rangenext_updown".to_string(),
                vec![iter_lltype],
                LowLevelType::Signed,
                move |_rtyper, _args, _result| {
                    build_ll_rangenext_updown_helper_graph(
                        "ll_rangenext_updown",
                        iter_for_builder.clone(),
                    )
                },
            )?;
            return hop.gendirectcall(&helper, args);
        }
        let name = if self.step > 0 {
            "ll_rangenext_up"
        } else {
            "ll_rangenext_down"
        };
        let up = self.step > 0;
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            name.to_string(),
            vec![iter_lltype, LowLevelType::Signed],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_rangenext_helper_graph(name, iter_for_builder.clone(), up)
            },
        )?;
        hop.gendirectcall(&helper, args)
    }
}

/// Synthesise the `ll_rangeiter` helper graph
/// (`lltypesystem/rrange.py`):
///
/// ```python
/// def ll_rangeiter(ITERPTR, rng):
///     iter = malloc(ITERPTR.TO)
///     iter.next = rng.start
///     iter.stop = rng.stop
///     if ITERPTR.TO is RANGESTITER:
///         iter.step = rng.step
///     return iter
/// ```
///
/// Single block: `start = getfield(rng, 'start'); stop = getfield(rng,
/// 'stop'); iter = malloc(RANGEITER, flavor=gc); setfield(iter, 'next',
/// start); setfield(iter, 'stop', stop)`. `ITERPTR` is baked as
/// `iter_lltype`; for the variable-step `RANGESTITER` (`variable_step`)
/// the `iter.step = rng.step` copy is also emitted.
pub(crate) fn build_ll_rangeiter_helper_graph(
    name: &str,
    range_lltype: LowLevelType,
    iter_lltype: LowLevelType,
    variable_step: bool,
) -> Result<PyGraph, TyperError> {
    let rng_arg = variable_with_lltype("rng", range_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(rng_arg.clone())]);
    let return_var = variable_with_lltype("result", iter_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let LowLevelType::Ptr(ptr) = &iter_lltype else {
        return Err(TyperError::message(
            "build_ll_rangeiter_helper_graph: iter lltype is not Ptr",
        ));
    };
    let inner_struct = match &ptr.TO {
        PtrTarget::Struct(body) => body.clone(),
        other => {
            return Err(TyperError::message(format!(
                "build_ll_rangeiter_helper_graph: Ptr target must be Struct, got {other:?}"
            )));
        }
    };

    let field_const = |f: &str| constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void);

    // start = rng.start; stop = rng.stop.
    let v_start = variable_with_lltype("start", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(rng_arg.clone()), field_const("start")],
        Hlvalue::Variable(v_start.clone()),
    ));
    let v_stop = variable_with_lltype("stop", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(rng_arg.clone()), field_const("stop")],
        Hlvalue::Variable(v_stop.clone()),
    ));

    // iter = malloc(RANGEITER, flavor=gc).
    let c_type = Constant::with_concretetype(
        ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(inner_struct)))),
        LowLevelType::Void,
    );
    let c_flags = gc_flavor_const()?;
    let v_iter = variable_with_lltype("iter", iter_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![Hlvalue::Constant(c_type), c_flags],
        Hlvalue::Variable(v_iter.clone()),
    ));

    // iter.next = start; iter.stop = stop.
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            field_const("next"),
            Hlvalue::Variable(v_start),
        ],
        Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            field_const("stop"),
            Hlvalue::Variable(v_stop),
        ],
        Hlvalue::Variable(variable_with_lltype("v1", LowLevelType::Void)),
    ));

    // if ITERPTR.TO is RANGESTITER: iter.step = rng.step.
    if variable_step {
        let v_step = variable_with_lltype("step", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(rng_arg), field_const("step")],
            Hlvalue::Variable(v_step.clone()),
        ));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(v_iter.clone()),
                field_const("step"),
                Hlvalue::Variable(v_step),
            ],
            Hlvalue::Variable(variable_with_lltype("v2", LowLevelType::Void)),
        ));
    } else {
        let _ = rng_arg;
    }

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
        vec!["rng".to_string()],
        func,
    ))
}

/// Push `l = malloc(<ptr_lltype>.TO, flavor=gc)` onto `block` and return the
/// gc pointer var.
fn emit_gc_malloc(
    block: &BlockRef,
    ptr_lltype: &LowLevelType,
    ctx: &str,
) -> Result<Variable, TyperError> {
    let LowLevelType::Ptr(ptr) = ptr_lltype else {
        return Err(TyperError::message(format!("{ctx}: lltype is not Ptr")));
    };
    let PtrTarget::Struct(inner_struct) = &ptr.TO else {
        return Err(TyperError::message(format!(
            "{ctx}: Ptr target must be Struct"
        )));
    };
    let c_type = Constant::with_concretetype(
        ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(
            inner_struct.clone(),
        )))),
        LowLevelType::Void,
    );
    let c_flags = gc_flavor_const()?;
    let v_l = variable_with_lltype("l", ptr_lltype.clone());
    block.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![Hlvalue::Constant(c_type), c_flags],
        Hlvalue::Variable(v_l.clone()),
    ));
    Ok(v_l)
}

/// Push `l.<field> = value` (a Void-result `setfield`) onto `block`.
fn emit_void_setfield(block: &BlockRef, l: &Variable, field: &str, value: Hlvalue) {
    block.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![Hlvalue::Variable(l.clone()), void_field(field), value],
        Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
    ));
}

/// Synthesise the `ll_newrange(RANGE, start, stop)` helper graph
/// (`lltypesystem/rrange.py`):
///
/// ```python
/// def ll_newrange(RANGE, start, stop):
///     l = malloc(RANGE.TO)
///     l.start = start
///     l.stop = stop
///     return l
/// ```
///
/// The `RANGE` type is folded into the builder, so the helper takes
/// `(start, stop)`. Single block: `malloc RANGE.TO (flavor=gc)` + setfield
/// start/stop.
pub(crate) fn build_ll_newrange_helper_graph(
    name: &str,
    range_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let stop_arg = variable_with_lltype("stop", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(stop_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", range_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_l = emit_gc_malloc(&startblock, &range_lltype, "build_ll_newrange_helper_graph")?;
    emit_void_setfield(&startblock, &v_l, "start", Hlvalue::Variable(start_arg));
    emit_void_setfield(&startblock, &v_l, "stop", Hlvalue::Variable(stop_arg));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_l)],
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
        vec!["start".to_string(), "stop".to_string()],
        func,
    ))
}

/// Synthesise the `ll_newrangest(start, stop, step)` helper graph
/// (`lltypesystem/rrange.py`):
///
/// ```python
/// def ll_newrangest(start, stop, step):
///     if step == 0:
///         raise ValueError
///     l = malloc(RANGEST)
///     l.start = start
///     l.stop = stop
///     l.step = step
///     return l
/// ```
///
/// 2-block CFG plus `graph.exceptblock`: **start** guards `int_eq(step, 0)`
/// → raise `ValueError`, else → **build** (`malloc RANGEST` + setfield
/// start/stop/step → return).
pub(crate) fn build_ll_newrangest_helper_graph(
    name: &str,
    rangest_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let stop_arg = variable_with_lltype("stop", LowLevelType::Signed);
    let step_arg = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(stop_arg.clone()),
        Hlvalue::Variable(step_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", rangest_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("ValueError")?;

    // build block params: (start, stop, step).
    let b_start = variable_with_lltype("start", LowLevelType::Signed);
    let b_stop = variable_with_lltype("stop", LowLevelType::Signed);
    let b_step = variable_with_lltype("step", LowLevelType::Signed);
    let build = Block::shared(vec![
        Hlvalue::Variable(b_start.clone()),
        Hlvalue::Variable(b_stop.clone()),
        Hlvalue::Variable(b_step.clone()),
    ]);

    // start: is_zero = int_eq(step, 0); branch.
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(step_arg.clone()), signed_const(0)],
        Hlvalue::Variable(is_zero.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(start_arg),
                Hlvalue::Variable(stop_arg),
                Hlvalue::Variable(step_arg),
            ],
            Some(build.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // build: l = malloc(RANGEST); l.start/stop/step = ...; return l.
    let v_l = emit_gc_malloc(&build, &rangest_lltype, "build_ll_newrangest_helper_graph")?;
    emit_void_setfield(&build, &v_l, "start", Hlvalue::Variable(b_start));
    emit_void_setfield(&build, &v_l, "stop", Hlvalue::Variable(b_stop));
    emit_void_setfield(&build, &v_l, "step", Hlvalue::Variable(b_step));
    build.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_l)],
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
        vec!["start".to_string(), "stop".to_string(), "step".to_string()],
        func,
    ))
}

/// Synthesise the `ll_rangenext_up` / `ll_rangenext_down` helper graph
/// (`rrange.py:172-184`):
///
/// ```python
/// def ll_rangenext_up(iter, step):    # `_down` flips `>=` to `<=`
///     next = iter.next
///     if next >= iter.stop:
///         raise StopIteration
///     iter.next = next + step
///     return next
/// ```
///
/// 2-block CFG. `up` selects `int_ge` (`next >= stop`) vs `int_le`
/// (`next <= stop`) for the StopIteration guard:
/// - **start**: `next = getfield(iter, 'next'); stop = getfield(iter,
///   'stop'); atend = int_ge/int_le(next, stop)`. Switch on `atend`:
///   True → exceptblock (StopIteration); False → cont, carrying `iter,
///   next, step`.
/// - **cont**: `newnext = int_add(next, step); setfield(iter, 'next',
///   newnext)`; return `next`.
pub(crate) fn build_ll_rangenext_helper_graph(
    name: &str,
    iter_lltype: LowLevelType,
    up: bool,
) -> Result<PyGraph, TyperError> {
    let iter_arg = variable_with_lltype("iter", iter_lltype.clone());
    let step_arg = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(iter_arg.clone()),
        Hlvalue::Variable(step_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let field_const = |f: &str| constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);

    // next = iter.next; stop = iter.stop; atend = next >=/<= stop.
    let v_next = variable_with_lltype("next", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(iter_arg.clone()), field_const("next")],
        Hlvalue::Variable(v_next.clone()),
    ));
    let v_stop = variable_with_lltype("stop", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(iter_arg.clone()), field_const("stop")],
        Hlvalue::Variable(v_stop.clone()),
    ));
    let v_atend = variable_with_lltype("atend", LowLevelType::Bool);
    let cmp_op = if up { "int_ge" } else { "int_le" };
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        cmp_op,
        vec![Hlvalue::Variable(v_next.clone()), Hlvalue::Variable(v_stop)],
        Hlvalue::Variable(v_atend.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_atend));

    // cont block args: (iter, next, step).
    let c_iter = variable_with_lltype("iter", iter_lltype);
    let c_next = variable_with_lltype("next", LowLevelType::Signed);
    let c_step = variable_with_lltype("step", LowLevelType::Signed);
    let cont = Block::shared(vec![
        Hlvalue::Variable(c_iter.clone()),
        Hlvalue::Variable(c_next.clone()),
        Hlvalue::Variable(c_step.clone()),
    ]);

    let exc_args = exception_args("StopIteration")?;
    startblock.closeblock(vec![
        // atend == True: raise StopIteration.
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        // atend == False: advance and return.
        Link::new(
            vec![
                Hlvalue::Variable(iter_arg),
                Hlvalue::Variable(v_next),
                Hlvalue::Variable(step_arg),
            ],
            Some(cont.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // newnext = next + step; iter.next = newnext.
    let v_newnext = variable_with_lltype("newnext", LowLevelType::Signed);
    {
        let mut b = cont.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(c_next.clone()), Hlvalue::Variable(c_step)],
            Hlvalue::Variable(v_newnext.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(c_iter),
                field_const("next"),
                Hlvalue::Variable(v_newnext),
            ],
            Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
        ));
    }
    cont.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c_next)],
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
        vec!["iter".to_string(), "step".to_string()],
        func,
    ))
}

/// Build one ascending/descending guard block for
/// [`build_ll_rangenext_updown_helper_graph`]. `cmp_op` selects `int_ge`
/// (ascending, `next >= stop`) vs `int_le` (descending, `next <= stop`):
/// on the at-end branch the guard raises `StopIteration`; otherwise it
/// advances by passing `(iter, next, step)` to `advance`.
fn build_rangenext_guard_block(
    guard: &BlockRef,
    b_iter: Variable,
    b_step: Variable,
    cmp_op: &str,
    advance: &BlockRef,
    exceptblock: &BlockRef,
) -> Result<(), TyperError> {
    let field_const = |f: &str| constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);

    let v_next = variable_with_lltype("next", LowLevelType::Signed);
    let v_stop = variable_with_lltype("stop", LowLevelType::Signed);
    let v_atend = variable_with_lltype("atend", LowLevelType::Bool);
    {
        let mut g = guard.borrow_mut();
        g.operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(b_iter.clone()), field_const("next")],
            Hlvalue::Variable(v_next.clone()),
        ));
        g.operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(b_iter.clone()), field_const("stop")],
            Hlvalue::Variable(v_stop.clone()),
        ));
        g.operations.push(SpaceOperation::new(
            cmp_op,
            vec![Hlvalue::Variable(v_next.clone()), Hlvalue::Variable(v_stop)],
            Hlvalue::Variable(v_atend.clone()),
        ));
        g.exitswitch = Some(Hlvalue::Variable(v_atend));
    }

    let exc_args = exception_args("StopIteration")?;
    guard.closeblock(vec![
        // atend == True: raise StopIteration.
        Link::new(exc_args, Some(exceptblock.clone()), Some(bool_const(true))).into_ref(),
        // atend == False: advance and return.
        Link::new(
            vec![
                Hlvalue::Variable(b_iter),
                Hlvalue::Variable(v_next),
                Hlvalue::Variable(b_step),
            ],
            Some(advance.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);
    Ok(())
}

/// Synthesise the `ll_rangenext_updown` helper graph
/// (`rrange.py:186-191`):
///
/// ```python
/// def ll_rangenext_updown(iter):
///     step = iter.step
///     if step > 0:
///         return ll_rangenext_up(iter, step)
///     else:
///         return ll_rangenext_down(iter, step)
/// ```
///
/// 4-block CFG. The variable-step iterator carries `step` in its struct,
/// so it is read at runtime and its sign selects the advance direction.
/// The `ll_rangenext_up` / `ll_rangenext_down` bodies are inlined behind
/// the sign branch (mirroring how [`build_ll_rangenext_helper_graph`]
/// inlines the constant-step body) rather than re-dispatched as direct
/// calls:
/// - **start** `(iter)`: `step = getfield(iter, 'step'); pos =
///   int_gt(step, 0)`. Switch on `pos`: True → up_guard, False →
///   down_guard; both carry `iter, step`.
/// - **up_guard** / **down_guard** `(iter, step)`: built by
///   [`build_rangenext_guard_block`] with `int_ge` / `int_le`.
/// - **advance** `(iter, next, step)`: `newnext = int_add(next, step);
///   setfield(iter, 'next', newnext)`; return `next`.
pub(crate) fn build_ll_rangenext_updown_helper_graph(
    name: &str,
    iter_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let iter_arg = variable_with_lltype("iter", iter_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(iter_arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let field_const = |f: &str| constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);

    // step = iter.step; pos = step > 0.
    let v_step = variable_with_lltype("step", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(iter_arg.clone()), field_const("step")],
        Hlvalue::Variable(v_step.clone()),
    ));
    let v_pos = variable_with_lltype("pos", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![
            Hlvalue::Variable(v_step.clone()),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(v_pos.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_pos));

    // up_guard / down_guard block params: (iter, step).
    let up_iter = variable_with_lltype("iter", iter_lltype.clone());
    let up_step = variable_with_lltype("step", LowLevelType::Signed);
    let up_guard = Block::shared(vec![
        Hlvalue::Variable(up_iter.clone()),
        Hlvalue::Variable(up_step.clone()),
    ]);
    let down_iter = variable_with_lltype("iter", iter_lltype.clone());
    let down_step = variable_with_lltype("step", LowLevelType::Signed);
    let down_guard = Block::shared(vec![
        Hlvalue::Variable(down_iter.clone()),
        Hlvalue::Variable(down_step.clone()),
    ]);

    // advance block params: (iter, next, step).
    let adv_iter = variable_with_lltype("iter", iter_lltype);
    let adv_next = variable_with_lltype("next", LowLevelType::Signed);
    let adv_step = variable_with_lltype("step", LowLevelType::Signed);
    let advance = Block::shared(vec![
        Hlvalue::Variable(adv_iter.clone()),
        Hlvalue::Variable(adv_next.clone()),
        Hlvalue::Variable(adv_step.clone()),
    ]);

    startblock.closeblock(vec![
        // step > 0: ascending guard.
        Link::new(
            vec![
                Hlvalue::Variable(iter_arg.clone()),
                Hlvalue::Variable(v_step.clone()),
            ],
            Some(up_guard.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        // step <= 0: descending guard.
        Link::new(
            vec![Hlvalue::Variable(iter_arg), Hlvalue::Variable(v_step)],
            Some(down_guard.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    build_rangenext_guard_block(
        &up_guard,
        up_iter,
        up_step,
        "int_ge",
        &advance,
        &graph.exceptblock,
    )?;
    build_rangenext_guard_block(
        &down_guard,
        down_iter,
        down_step,
        "int_le",
        &advance,
        &graph.exceptblock,
    )?;

    // newnext = next + step; iter.next = newnext; return next.
    let v_newnext = variable_with_lltype("newnext", LowLevelType::Signed);
    {
        let mut b = advance.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(adv_next.clone()),
                Hlvalue::Variable(adv_step),
            ],
            Hlvalue::Variable(v_newnext.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(adv_iter),
                field_const("next"),
                Hlvalue::Variable(v_newnext),
            ],
            Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
        ));
    }
    advance.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(adv_next)],
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rtyper::RPythonTyper;

    #[test]
    fn rangerepr_step1_lowleveltype_is_immutable_range_gcstruct() {
        let rr = AbstractRangeRepr::new(1).unwrap();
        assert_eq!(rr.repr_class_id(), ReprClassId::RangeRepr);
        match rr.lowleveltype() {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(st) => {
                    // RANGE = GcStruct("range", start, stop) — two Signed fields.
                    assert_eq!(st._names_without_voids(), vec!["start", "stop"]);
                }
                other => panic!("AbstractRangeRepr lltype TO not Struct: {other:?}"),
            },
            other => panic!("AbstractRangeRepr lltype not Ptr: {other:?}"),
        }
    }

    #[test]
    fn rangerepr_variable_step_lowleveltype_is_rangest_gcstruct() {
        // step == 0 → RANGEST with the extra `step` field.
        let rr = AbstractRangeRepr::new(0).unwrap();
        match rr.lowleveltype() {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(st) => {
                    assert_eq!(st._names_without_voids(), vec!["start", "stop", "step"]);
                }
                other => panic!("not Struct: {other:?}"),
            },
            other => panic!("not Ptr: {other:?}"),
        }
    }

    #[test]
    fn build_ll_rangelen1_helper_graph_synthesizes_2_block_cfg() {
        let ptr = AbstractRangeRepr::new(1).unwrap().lowleveltype().clone();
        let g = build_ll_rangelen1_helper_graph("ll_rangelen1", ptr).unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getfield", "getfield", "int_sub", "int_lt"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);
    }

    /// The general `ll_rangelen` branches on `int_gt(step, 0)` into the
    /// ascending / descending floor-division arms (each ending in
    /// `int_floordiv`), then clamps the negative result to zero.
    #[test]
    fn build_ll_rangelen_helper_graph_branches_on_step_sign_with_floordiv() {
        let ptr = AbstractRangeRepr::new(2).unwrap().lowleveltype().clone();
        let g = build_ll_rangelen_helper_graph("ll_rangelen", ptr).unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        // start = l.start; stop = l.stop; pos = step > 0.
        assert_eq!(ops, vec!["getfield", "getfield", "int_gt"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);
        // (l, step) inputargs.
        assert_eq!(startblock.inputargs.len(), 2);

        // Both sign arms end in int_floordiv.
        for link in startblock.exits.iter() {
            let arm = link.borrow().target.clone().unwrap();
            let body = arm.borrow();
            let last = body.operations.last().map(|op| op.opname.as_str());
            assert_eq!(last, Some("int_floordiv"));
        }
    }

    #[test]
    fn build_ll_rangeitem_nonneg_helper_graph_synthesizes_getfield_mul_add() {
        let ptr = AbstractRangeRepr::new(1).unwrap().lowleveltype().clone();
        let g = build_ll_rangeitem_nonneg_helper_graph("ll_rangeitem_nonneg", ptr).unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        // start = l.start; prod = index * step; result = start + prod.
        assert_eq!(ops, vec!["getfield", "int_mul", "int_add"]);
        assert!(startblock.exitswitch.is_none());
        assert_eq!(startblock.exits.len(), 1);
        // (l, index, step) inputargs.
        assert_eq!(startblock.inputargs.len(), 3);
    }

    /// rrange.py constant-step + nonneg branch — `getitem` on a
    /// `AbstractRangeRepr` lowers to a `direct_call` of `ll_rangeitem_nonneg`
    /// (`start + index*step`), preceded by `hop.exception_is_here()`, with
    /// the constant step appended as the trailing arg.
    #[test]
    fn rangerepr_getitem_nonneg_const_step_emits_direct_call_to_ll_rangeitem_nonneg() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        // step == 2 (constant step != 1) — the formula handles any step.
        let range_repr: Arc<AbstractRangeRepr> =
            Arc::new(AbstractRangeRepr::new(2).expect("AbstractRangeRepr::new(2)"));
        let range_lltype = range_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_rng = Variable::new();
        v_rng.set_concretetype(Some(range_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_rng), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        // args_s[0] (the range sequence) is not inspected by rtype_getitem
        // (only args_s[1].nonneg); a generic SomeList stands in.
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ false,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(range_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = range_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("range getitem nonneg: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(ops._called_exception_is_here_or_cannot_occur);
        // direct_call args: funcptr, v_rng, v_index, cstep (the baked step).
        assert_eq!(ops.ops[0].args.len(), 4);
        let Hlvalue::Constant(cstep) = &ops.ops[0].args[3] else {
            panic!("expected Constant step as last direct_call arg");
        };
        assert!(matches!(cstep.value, ConstValue::Int(2)));
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_rangeitem_nonneg"),
            "expected 'll_rangeitem_nonneg' in {dbg}"
        );
    }

    /// rrange.py:22-30 — `len()` on a constant-step `range` with `step !=
    /// 1` lowers to a `direct_call` of the general `ll_rangelen` with the
    /// constant step baked as the trailing arg.
    #[test]
    fn rangerepr_len_const_step_emits_direct_call_to_ll_rangelen() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        // step == 2 (constant step != 1) → general ll_rangelen.
        let range_repr: Arc<AbstractRangeRepr> =
            Arc::new(AbstractRangeRepr::new(2).expect("AbstractRangeRepr::new(2)"));
        let range_lltype = range_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_rng = Variable::new();
        v_rng.set_concretetype(Some(range_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "len".to_string(),
                vec![Hlvalue::Variable(v_rng)],
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
            .push(Some(range_repr.clone() as Arc<dyn Repr>));

        let result = range_repr
            .rtype_len(&hop)
            .unwrap_or_else(|err| panic!("range rtype_len(step=2): {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        // direct_call args: funcptr, v_rng, cstep (the baked step).
        assert_eq!(ops.ops[0].args.len(), 3);
        let Hlvalue::Constant(cstep) = &ops.ops[0].args[2] else {
            panic!("expected Constant step as last direct_call arg");
        };
        assert!(matches!(cstep.value, ConstValue::Int(2)));
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_rangelen"),
            "expected 'll_rangelen' in {dbg}"
        );
    }

    /// rrange.py — `len()` on a variable-step `RANGEST` reads the
    /// runtime `step` via `_getstep` (a `getfield`) and passes it to
    /// `ll_rangelen`, so the trailing direct_call arg is the getfield
    /// result variable, not a baked constant.
    #[test]
    fn rangerepr_len_variable_step_reads_getstep_then_ll_rangelen() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        // step == 0 (variable) → RANGEST; _getstep reads iter.step.
        let range_repr: Arc<AbstractRangeRepr> =
            Arc::new(AbstractRangeRepr::new(0).expect("AbstractRangeRepr::new(0)"));
        let range_lltype = range_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_rng = Variable::new();
        v_rng.set_concretetype(Some(range_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "len".to_string(),
                vec![Hlvalue::Variable(v_rng)],
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
            .push(Some(range_repr.clone() as Arc<dyn Repr>));

        let result = range_repr
            .rtype_len(&hop)
            .unwrap_or_else(|err| panic!("range rtype_len(step=0): {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        // _getstep getfield, then the ll_rangelen direct_call.
        assert_eq!(ops.ops.len(), 2);
        assert_eq!(ops.ops[0].opname, "getfield");
        assert_eq!(ops.ops[1].opname, "direct_call");
        // direct_call args: funcptr, v_rng, v_step (the getfield result).
        assert_eq!(ops.ops[1].args.len(), 3);
        assert!(matches!(ops.ops[1].args[2], Hlvalue::Variable(_)));
        let Hlvalue::Constant(c) = &ops.ops[1].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_rangelen"),
            "expected 'll_rangelen' in {dbg}"
        );
    }

    /// Drive `rtype_getitem` for a given `(implicit IndexError, nonneg)` and
    /// return the funcptr-`Constant` debug string of the emitted `direct_call`.
    fn getitem_helper_name_for(checkidx: bool, nonneg: bool) -> String {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let range_repr: Arc<AbstractRangeRepr> =
            Arc::new(AbstractRangeRepr::new(1).expect("AbstractRangeRepr::new(1)"));
        let range_lltype = range_repr.lowleveltype().clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_rng = Variable::new();
        v_rng.set_concretetype(Some(range_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        // An IndexError exceptionlink makes `has_implicit_exception("IndexError")`
        // (the `dum_checkidx` selector) return true.
        let exc_links = if checkidx {
            let exitblock = std::rc::Rc::new(std::cell::RefCell::new(Block::new(vec![])));
            let cls_index = crate::flowspace::model::HOST_ENV
                .lookup_exception_class("IndexError")
                .unwrap();
            vec![std::rc::Rc::new(std::cell::RefCell::new(Link::new(
                vec![],
                Some(exitblock),
                Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                    cls_index,
                )))),
            )))]
        } else {
            Vec::new()
        };
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_rng), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            exc_links,
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ false,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(nonneg, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(range_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);
        let result = range_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("range getitem ({checkidx}, {nonneg}): {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        let last = ops.ops.last().expect("expected at least one op");
        assert_eq!(last.opname, "direct_call");
        let Hlvalue::Constant(c) = &last.args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        format!("{:?}", c.value)
    }

    /// All four `(checkidx, nonneg)` combinations lower to a `direct_call` of
    /// the matching `func`-folded helper (`rrange.py:34-50`).
    #[test]
    fn rangerepr_getitem_selects_func_folded_helper_per_combination() {
        let nocheck_nonneg = getitem_helper_name_for(false, true);
        assert!(
            nocheck_nonneg.contains("ll_rangeitem_nonneg") && !nocheck_nonneg.contains("checked"),
            "expected 'll_rangeitem_nonneg' in {nocheck_nonneg}"
        );
        let nocheck_neg = getitem_helper_name_for(false, false);
        assert!(
            nocheck_neg.contains("ll_rangeitem") && !nocheck_neg.contains("nonneg"),
            "expected 'll_rangeitem' in {nocheck_neg}"
        );
        let checked_nonneg = getitem_helper_name_for(true, true);
        assert!(
            checked_nonneg.contains("ll_rangeitem_nonneg_checked"),
            "expected 'll_rangeitem_nonneg_checked' in {checked_nonneg}"
        );
        let checked_neg = getitem_helper_name_for(true, false);
        assert!(
            checked_neg.contains("ll_rangeitem_checked") && !checked_neg.contains("nonneg"),
            "expected 'll_rangeitem_checked' in {checked_neg}"
        );
    }

    /// `_ll_rangelen(start, stop, step)` takes the three Signed operands
    /// directly (no `getfield`) and runs the sign-branch floor-division core:
    /// the startblock switches on `int_gt(step, 0)` into two arms each ending
    /// in `int_floordiv`.
    #[test]
    fn build_underscore_ll_rangelen_helper_graph_has_signed_params_and_floordiv() {
        let g = build_underscore_ll_rangelen_helper_graph("_ll_rangelen").unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        // (start, stop, step) inputargs; no getfield — the first op is the
        // sign test int_gt.
        assert_eq!(startblock.inputargs.len(), 3);
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["int_gt"]);
        assert_eq!(startblock.exits.len(), 2);
        for link in startblock.exits.iter() {
            let arm = link.borrow().target.clone().unwrap();
            let body = arm.borrow();
            let last = body.operations.last().map(|op| op.opname.as_str());
            assert_eq!(last, Some("int_floordiv"));
        }
    }

    /// The full checked `ll_rangeitem` (`dum_checkidx`, negative-index) reads
    /// the length once via `direct_call(_ll_rangelen, ...)` and raises
    /// `IndexError` from both the `index < 0` and `index >= length` guards
    /// (two links into the graph's exceptblock).
    #[test]
    fn build_ll_rangeitem_checked_helper_graph_raises_indexerror_from_both_guards() {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let ptr = AbstractRangeRepr::new(1).unwrap().lowleveltype().clone();
        let g =
            build_ll_rangeitem_checked_helper_graph(&rtyper, "ll_rangeitem_checked", ptr).unwrap();
        let inner = g.graph.borrow();

        // Exactly one `direct_call` to `_ll_rangelen` across the whole graph.
        let mut rangelen_calls = 0usize;
        // Exactly two links target the exceptblock (the two IndexError guards).
        let mut raise_links = 0usize;
        for block in inner.iterblocks() {
            for op in block.borrow().operations.iter() {
                if op.opname == "direct_call"
                    && let Some(Hlvalue::Constant(c)) = op.args.first()
                    && format!("{:?}", c.value).contains("_ll_rangelen")
                {
                    rangelen_calls += 1;
                }
            }
            for link in block.borrow().exits.iter() {
                if let Some(target) = link.borrow().target.as_ref()
                    && std::rc::Rc::ptr_eq(target, &inner.exceptblock)
                {
                    raise_links += 1;
                }
            }
        }
        assert_eq!(rangelen_calls, 1, "expected a single _ll_rangelen call");
        assert_eq!(raise_links, 2, "expected two IndexError-raising links");
    }

    /// Drive `rtype_builtin_range` for `range(<nb_args> Signed args)` whose
    /// result repr is `RangeRepr(step)` and return the funcptr-`Constant`
    /// debug string of the emitted `direct_call`.
    fn range_builtin_call_funcptr(nb_args: usize, step: i64) -> String {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let range_repr: Arc<AbstractRangeRepr> =
            Arc::new(AbstractRangeRepr::new(step).expect("AbstractRangeRepr::new"));
        let range_lltype = range_repr.lowleveltype().clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));

        let mut call_args = Vec::new();
        for _ in 0..nb_args {
            let v = Variable::new();
            v.set_concretetype(Some(LowLevelType::Signed));
            call_args.push(Hlvalue::Variable(v));
        }
        let v_result = Variable::new();
        v_result.set_concretetype(Some(range_lltype));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                call_args.clone(),
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(call_args);
        for _ in 0..nb_args {
            hop.args_s
                .borrow_mut()
                .push(SomeValue::Integer(SomeInteger::new(false, false)));
            hop.args_r
                .borrow_mut()
                .push(Some(signed_repr() as Arc<dyn Repr>));
        }
        *hop.r_result.borrow_mut() = Some(range_repr as Arc<dyn Repr>);

        let result = rtype_builtin_range(&hop, &HashMap::new())
            .unwrap_or_else(|err| panic!("rtype_builtin_range({nb_args}, {step}): {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        let last = ops.ops.last().expect("expected a direct_call op");
        assert_eq!(last.opname, "direct_call");
        let Hlvalue::Constant(c) = &last.args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        format!("{:?}", c.value)
    }

    fn range_builtin_list_call_funcptr(nb_args: usize) -> String {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let item_repr: Arc<dyn Repr> = signed_repr();
        let list_repr: Arc<ListRepr> =
            Arc::new(ListRepr::new(&rtyper, item_repr).expect("ListRepr::new"));
        let list_lltype = list_repr.lowleveltype().clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));

        let mut call_args = Vec::new();
        for _ in 0..nb_args {
            let v = Variable::new();
            v.set_concretetype(Some(LowLevelType::Signed));
            call_args.push(Hlvalue::Variable(v));
        }
        let v_result = Variable::new();
        v_result.set_concretetype(Some(list_lltype));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                call_args.clone(),
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(call_args);
        for _ in 0..nb_args {
            hop.args_s
                .borrow_mut()
                .push(SomeValue::Integer(SomeInteger::new(false, false)));
            hop.args_r
                .borrow_mut()
                .push(Some(signed_repr() as Arc<dyn Repr>));
        }
        *hop.r_result.borrow_mut() = Some(list_repr as Arc<dyn Repr>);

        let result = rtype_builtin_range(&hop, &HashMap::new())
            .unwrap_or_else(|err| panic!("rtype_builtin_range list arm: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        let last = ops.ops.last().expect("expected a direct_call op");
        assert_eq!(last.opname, "direct_call");
        let Hlvalue::Constant(c) = &last.args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        format!("{:?}", c.value)
    }

    /// A constant-step `range(a, b)` lowers to a `direct_call` of `ll_newrange`
    /// (`rrange.py`); a variable-step result lowers to `ll_newrangest`
    /// (`rrange.py:114`).
    #[test]
    fn rtype_builtin_range_selects_newrange_vs_newrangest_by_step() {
        let const_step = range_builtin_call_funcptr(2, 1);
        assert!(
            const_step.contains("ll_newrange") && !const_step.contains("ll_newrangest"),
            "expected 'll_newrange' in {const_step}"
        );
        let var_step = range_builtin_call_funcptr(3, 0);
        assert!(
            var_step.contains("ll_newrangest"),
            "expected 'll_newrangest' in {var_step}"
        );
    }

    #[test]
    fn rtype_builtin_range_list_result_calls_ll_range2list() {
        let list_call = range_builtin_list_call_funcptr(3);
        assert!(
            list_call.contains("ll_range2list"),
            "expected 'll_range2list' in {list_call}"
        );
    }

    /// `ll_newrange(start, stop)` mallocs the RANGE struct and sets both
    /// fields: a single block of `malloc` + 2×`setfield`.
    #[test]
    fn build_ll_newrange_helper_graph_mallocs_and_sets_start_stop() {
        let ptr = AbstractRangeRepr::new(1).unwrap().lowleveltype().clone();
        let g = build_ll_newrange_helper_graph("ll_newrange", ptr).unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["malloc", "setfield", "setfield"]);
        assert_eq!(startblock.inputargs.len(), 2);
        assert!(startblock.exitswitch.is_none());
    }

    /// `ll_newrangest(start, stop, step)` guards `int_eq(step, 0)` → raise
    /// `ValueError`, else mallocs RANGEST and sets all three fields.
    #[test]
    fn build_ll_newrangest_helper_graph_guards_zero_step_then_mallocs() {
        let ptr = AbstractRangeRepr::new(0).unwrap().lowleveltype().clone();
        let g = build_ll_newrangest_helper_graph("ll_newrangest", ptr).unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["int_eq"]);
        assert!(startblock.exitswitch.is_some());
        // One exit raises ValueError (links to exceptblock), the other builds.
        let mut raise_links = 0usize;
        let mut build_target = None;
        for link in startblock.exits.iter() {
            let target = link.borrow().target.clone().unwrap();
            if std::rc::Rc::ptr_eq(&target, &inner.exceptblock) {
                raise_links += 1;
            } else {
                build_target = Some(target);
            }
        }
        assert_eq!(raise_links, 1, "expected one ValueError-raising link");
        let build = build_target.expect("expected a build block");
        let build_block = build.borrow();
        let build_ops: Vec<&str> = build_block
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            build_ops,
            vec!["malloc", "setfield", "setfield", "setfield"]
        );
    }

    /// `make_iterator_repr` mints a `RangeIteratorRepr` whose lowleveltype
    /// is the `RANGEITER` struct (`next`, `stop`) for constant step and the
    /// `RANGESTITER` struct (`next`, `stop`, `step`) for variable step. A
    /// non-None variant is rejected (`rrange.py:64-66`).
    #[test]
    fn make_iterator_repr_yields_rangeiter_and_rangestiter_structs() {
        let r_rng = AbstractRangeRepr::new(1).unwrap();
        let it = r_rng.make_iterator_repr(&[], false).unwrap();
        assert_eq!(it.class_name(), "RangeIteratorRepr");
        assert_eq!(it.repr_class_id(), ReprClassId::RangeIteratorRepr);
        match it.lowleveltype() {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(st) => {
                    assert_eq!(st._names_without_voids(), vec!["next", "stop"]);
                }
                other => panic!("RANGEITER TO not Struct: {other:?}"),
            },
            other => panic!("RANGEITER not Ptr: {other:?}"),
        }
        // unsupported variant → TyperError.
        assert!(
            r_rng
                .make_iterator_repr(&["reversed".to_string()], false)
                .is_err()
        );

        // variable-step (step == 0) → RANGESTITER with the extra `step` field.
        let rst = AbstractRangeRepr::new(0).unwrap();
        let it = rst.make_iterator_repr(&[], false).unwrap();
        match it.lowleveltype() {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(st) => {
                    assert_eq!(st._names_without_voids(), vec!["next", "stop", "step"]);
                }
                other => panic!("RANGESTITER TO not Struct: {other:?}"),
            },
            other => panic!("RANGESTITER not Ptr: {other:?}"),
        }
    }

    /// `ll_rangeiter` single block: `getfield start; getfield stop;
    /// malloc RANGEITER; setfield next; setfield stop`.
    #[test]
    fn build_ll_rangeiter_helper_graph_synthesizes_getfields_malloc_setfields() {
        let r_rng = AbstractRangeRepr::new(1).unwrap();
        let iter_lltype = RangeIteratorRepr::new(&r_rng)
            .unwrap()
            .lowleveltype()
            .clone();
        let range_lltype = r_rng.lowleveltype().clone();
        let g = build_ll_rangeiter_helper_graph("ll_rangeiter", range_lltype, iter_lltype, false)
            .unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec!["getfield", "getfield", "malloc", "setfield", "setfield"]
        );
        assert!(startblock.exitswitch.is_none());
        assert_eq!(startblock.exits.len(), 1);
        assert_eq!(startblock.inputargs.len(), 1);
    }

    /// Variable-step `ll_rangeiter` additionally copies `iter.step =
    /// rng.step` (the `if ITERPTR.TO is RANGESTITER` branch,
    /// lltypesystem/rrange.py), so the block carries a third
    /// `getfield` + `setfield` pair.
    #[test]
    fn build_ll_rangeiter_helper_graph_variable_step_copies_step_field() {
        let r_rng = AbstractRangeRepr::new(0).unwrap();
        let iter_lltype = RangeIteratorRepr::new(&r_rng)
            .unwrap()
            .lowleveltype()
            .clone();
        let range_lltype = r_rng.lowleveltype().clone();
        let g = build_ll_rangeiter_helper_graph("ll_rangeiter", range_lltype, iter_lltype, true)
            .unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec![
                "getfield", "getfield", "malloc", "setfield", "setfield", "getfield", "setfield"
            ]
        );
        assert!(startblock.exitswitch.is_none());
        assert_eq!(startblock.exits.len(), 1);
    }

    /// `ll_rangenext_updown` reads `iter.step`, branches on `int_gt(step,
    /// 0)`, and inlines the ascending (`int_ge`) / descending (`int_le`)
    /// guards into a shared advance block (rrange.py:186-191).
    #[test]
    fn build_ll_rangenext_updown_helper_graph_branches_on_step_sign() {
        let r_rng = AbstractRangeRepr::new(0).unwrap();
        let iter_lltype = RangeIteratorRepr::new(&r_rng)
            .unwrap()
            .lowleveltype()
            .clone();
        let g = build_ll_rangenext_updown_helper_graph("ll_rangenext_updown", iter_lltype).unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        // step = iter.step; pos = step > 0.
        assert_eq!(ops, vec!["getfield", "int_gt"]);
        assert!(startblock.exitswitch.is_some());
        // True → up_guard, False → down_guard.
        assert_eq!(startblock.exits.len(), 2);
        // single `iter` inputarg.
        assert_eq!(startblock.inputargs.len(), 1);

        // The two guard targets compare with int_ge / int_le.
        let mut cmps: Vec<String> = startblock
            .exits
            .iter()
            .map(|link| {
                let target = link.borrow().target.clone().unwrap();
                let body = target.borrow();
                body.operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .find(|name| name == "int_ge" || name == "int_le")
                    .unwrap()
            })
            .collect();
        cmps.sort();
        assert_eq!(cmps, vec!["int_ge".to_string(), "int_le".to_string()]);
    }

    /// `ll_rangenext_up` start block guards with `int_ge` and branches to
    /// the StopIteration (exceptblock) and advance (cont) exits;
    /// `ll_rangenext_down` flips the guard to `int_le`.
    #[test]
    fn build_ll_rangenext_helper_graph_guards_with_int_ge_up_int_le_down() {
        let r_rng = AbstractRangeRepr::new(1).unwrap();
        let iter_lltype = RangeIteratorRepr::new(&r_rng)
            .unwrap()
            .lowleveltype()
            .clone();

        for (up, cmp) in [(true, "int_ge"), (false, "int_le")] {
            let name = if up {
                "ll_rangenext_up"
            } else {
                "ll_rangenext_down"
            };
            let g = build_ll_rangenext_helper_graph(name, iter_lltype.clone(), up).unwrap();
            let inner = g.graph.borrow();
            let startblock = inner.startblock.borrow();
            let ops: Vec<&str> = startblock
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(ops, vec!["getfield", "getfield", cmp]);
            assert!(startblock.exitswitch.is_some());
            assert_eq!(startblock.exits.len(), 2);
            // (iter, step) inputargs.
            assert_eq!(startblock.inputargs.len(), 2);
        }
    }

    /// Drive constant-step `RangeIteratorRepr::rtype_next` and assert it
    /// lowers to a single `direct_call` of `expected_name` with the baked
    /// constant `step` appended, preceded by `hop.exception_is_here()`
    /// (rrange.py:158-170).
    fn assert_rtype_next_emits_direct_call(step: i64, expected_name: &str) {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let r_rng = AbstractRangeRepr::new(step).unwrap();
        let iter_repr: Arc<RangeIteratorRepr> = Arc::new(RangeIteratorRepr::new(&r_rng).unwrap());
        let iter_lltype = iter_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_iter = Variable::new();
        v_iter.set_concretetype(Some(iter_lltype));
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
        // Non-constant binding so inputarg reaches convertvar (identity, no-op).
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r
            .borrow_mut()
            .push(Some(iter_repr.clone() as Arc<dyn Repr>));

        let result = iter_repr
            .rtype_next(&hop)
            .unwrap_or_else(|err| panic!("range rtype_next: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(ops._called_exception_is_here_or_cannot_occur);
        // direct_call args: funcptr, v_iter, cstep (the baked step).
        assert_eq!(ops.ops[0].args.len(), 3);
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains(expected_name),
            "expected '{expected_name}' in {dbg}"
        );
        let Hlvalue::Constant(cstep) = &ops.ops[0].args[2] else {
            panic!("expected Constant step as last direct_call arg");
        };
        assert!(matches!(cstep.value, ConstValue::Int(s) if s == step));
    }

    /// rrange.py `step > 0` lowers to `ll_rangenext_up` with the
    /// baked positive step.
    #[test]
    fn rangeiter_rtype_next_emits_direct_call_to_ll_rangenext_up() {
        assert_rtype_next_emits_direct_call(1, "ll_rangenext_up");
    }

    /// rrange.py `step < 0` lowers to `ll_rangenext_down` with the
    /// baked negative step.
    #[test]
    fn rangeiter_rtype_next_emits_direct_call_to_ll_rangenext_down() {
        assert_rtype_next_emits_direct_call(-1, "ll_rangenext_down");
    }

    /// rmodel.py `IteratorRepr.rtype_iter` — `iter()` on a range
    /// iterator is the identity (returns the iterator unchanged, emits no
    /// op), and the iterator carries its own `RangeIteratorRepr` class id.
    #[test]
    fn rangeiter_rtype_iter_returns_the_iterator_itself() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        let r_rng = AbstractRangeRepr::new(1).unwrap();
        let iter_repr: Arc<RangeIteratorRepr> = Arc::new(RangeIteratorRepr::new(&r_rng).unwrap());
        assert_eq!(iter_repr.repr_class_id(), ReprClassId::RangeIteratorRepr);
        let iter_lltype = iter_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
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

        let result = iter_repr
            .rtype_iter(&hop)
            .unwrap_or_else(|err| panic!("range rtype_iter: {err:?}"));
        // identity: the iterator is returned unchanged, no op emitted.
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_eq!(llops.borrow().ops.len(), 0);

        // `it.next()` method call routes to rtype_next.
        let method_err = iter_repr.rtype_method("unknown_method", &hop);
        assert!(method_err.is_err());
    }

    fn enumerate_list_iterator(
        rtyper: &std::rc::Rc<RPythonTyper>,
        start: Option<i64>,
    ) -> Arc<dyn Repr> {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeIterator, SomeList, SomeValue};
        let ldef = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        let s_iter = SomeValue::Iterator(SomeIterator::new_with_enumerate_start(
            SomeValue::List(SomeList::new(ldef)),
            vec!["enumerate".to_string()],
            start.map(ConstValue::Int),
        ));
        crate::translator::rtyper::rmodel::rtyper_makerepr(&s_iter, rtyper)
            .expect("rtyper_makerepr enumerate iterator")
    }

    /// The typer holds its annotator weakly: keep both alive.
    fn live_rtyper() -> (
        std::rc::Rc<crate::annotator::annrpython::RPythonAnnotator>,
        std::rc::Rc<RPythonTyper>,
    ) {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        (ann, rtyper)
    }

    /// rmodel.py `SomeIterator.rtyper_makerepr`: the `("enumerate", start)`
    /// variant wraps the container's plain iterator and shares its
    /// lowleveltype.
    #[test]
    fn makerepr_enumerate_iterator_wraps_the_list_iterator() {
        let (_ann, rtyper) = live_rtyper();
        let repr = enumerate_list_iterator(&rtyper, None);
        assert_eq!(repr.class_name(), "EnumerateIteratorRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::EnumerateIteratorRepr);
        let any_r: &dyn std::any::Any = repr.as_ref();
        let r_enum = any_r
            .downcast_ref::<EnumerateIteratorRepr>()
            .expect("EnumerateIteratorRepr");
        assert_eq!(r_enum.r_baseiter.class_name(), "ListIteratorRepr");
        assert_eq!(repr.lowleveltype(), r_enum.r_baseiter.lowleveltype());
        assert_eq!(r_enum.const_startindex, None);
    }

    /// rrange.py `EnumerateIteratorRepr.rtype_next`: the index is read
    /// (`ll_getnextindex`) before the base iterator steps (`ll_listnext`),
    /// a constant start is added to it, and the pair is packed with
    /// `newtuple`.
    #[test]
    fn enumerate_rtype_next_reads_the_index_before_listnext() {
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtuple::TupleRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let (_ann, rtyper) = live_rtyper();
        let repr = enumerate_list_iterator(&rtyper, Some(1));
        let r_tuple: Arc<dyn Repr> = Arc::new(
            TupleRepr::new(
                &rtyper,
                vec![
                    signed_repr() as Arc<dyn Repr>,
                    signed_repr() as Arc<dyn Repr>,
                ],
            )
            .unwrap(),
        );
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_iter = Variable::new();
        v_iter.set_concretetype(Some(repr.lowleveltype().clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(r_tuple.lowleveltype().clone()));
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
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(repr.clone()));
        *hop.r_result.borrow_mut() = Some(r_tuple);

        let result = repr
            .rtype_next(&hop)
            .unwrap_or_else(|err| panic!("enumerate rtype_next: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        let names: Vec<String> = ops
            .ops
            .iter()
            .map(|op| match (op.opname.as_str(), op.args.first()) {
                ("direct_call", Some(Hlvalue::Constant(c))) => {
                    let dbg = format!("{:?}", c.value);
                    if dbg.contains("ll_getnextindex") {
                        "ll_getnextindex".to_string()
                    } else if dbg.contains("ll_listnext") {
                        "ll_listnext".to_string()
                    } else {
                        "direct_call".to_string()
                    }
                }
                (name, _) => name.to_string(),
            })
            .collect();
        let at = |name: &str| names.iter().position(|n| n == name);
        let getnextindex = at("ll_getnextindex").expect("ll_getnextindex call");
        let add = at("int_add").expect("constant start added");
        let listnext = at("ll_listnext").expect("ll_listnext call");
        assert!(getnextindex < add && add < listnext, "{names:?}");
        assert!(
            names[listnext..].iter().any(|n| n == "malloc"),
            "newtuple after the item: {names:?}"
        );
    }

    /// rrange.py `rtype_builtin_enumerate`: `enumerate(lst)` is the base
    /// list iterator's `newiter` (`ll_listiter`) and cannot raise.
    #[test]
    fn rtype_builtin_enumerate_is_the_list_newiter() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let (_ann, rtyper) = live_rtyper();
        let r_enum = enumerate_list_iterator(&rtyper, None);
        let s_list = SomeValue::List(SomeList::new(ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        )));
        let r_list = crate::translator::rtyper::rmodel::rtyper_makerepr(&s_list, &rtyper)
            .expect("list repr");
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(r_list.lowleveltype().clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(r_enum.lowleveltype().clone()));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![Hlvalue::Variable(v_list.clone())],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_list));
        hop.args_s.borrow_mut().push(s_list);
        hop.args_r.borrow_mut().push(Some(r_list));
        *hop.r_result.borrow_mut() = Some(r_enum);

        let result = rtype_builtin_enumerate(&hop, &HashMap::new())
            .unwrap_or_else(|err| panic!("rtype_builtin_enumerate: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        let last = ops.ops.last().expect("a direct_call op");
        assert_eq!(last.opname, "direct_call");
        let Hlvalue::Constant(c) = &last.args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        assert!(format!("{:?}", c.value).contains("ll_listiter"));
    }
}
