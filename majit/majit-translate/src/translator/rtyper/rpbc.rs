//! Polymorphic Bound Callable (PBC) representation — lowers indirect
//! method calls on `dyn Trait` receivers.
//!
//! RPython equivalent:
//! `rpython/rtyper/rpbc.py FunctionReprBase.call`. RPython's
//! rtyper materialises the funcptr Variable via
//! `convert_to_concrete_llfn` and then emits
//! `genop('indirect_call', [funcptr, *args, c_graphs])` carrying the
//! full row-of-graphs as the trailing Constant. Pyre's rtyper-equivalent
//! does the same shape over `OpKind::Call { target: CallTarget::Indirect
//! { trait_root, method_name }, .. }` sites: insert a `VtableMethodPtr`
//! to materialise the funcptr Variable, then replace the original Call
//! with an `OpKind::IndirectCall` carrying funcptr + args + the family
//! `graphs` (full `c_graphs`, not yet filtered by JIT candidates —
//! that filtering happens later in `call.py::graphs_from(op)`).
//!
//! After this pass, no `CallTarget::Indirect` survives in the graph;
//! `jtransform` only sees `OpKind::IndirectCall` for indirect dispatch.

#![allow(private_interfaces)]
#![cfg_attr(test, allow(non_snake_case))]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::annotator::argument::ArgumentsForTranslation;
use crate::annotator::bookkeeper::{Bookkeeper, PositionKey};
use crate::annotator::description::{CallFamily, CallTableRow, DescEntry, DescKey};
use crate::annotator::model::{AnnotatorError, DescKind, SomeObjectTrait, SomePBC};
use crate::call::CallControl;
use crate::flowspace::argument::CallShape;
use crate::flowspace::pygraph::PyGraph;
use crate::model::{
    BlockId, CallTarget, FunctionGraph as JitFunctionGraph, OpKind, SpaceOperation, ValueType,
};
use crate::parse::CallPath;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, DelayedPointer, FuncType, MallocFlavor, Ptr as LLPtr, PtrTarget, Struct,
    malloc as ll_malloc, nullptr as ll_nullptr,
};
use crate::translator::rtyper::rclass;
use crate::translator::rtyper::rmodel::warning;
use crate::translator::rtyper::rtyper::RPythonTyper;
// `lower_indirect_calls` records the inserted `VtableMethodPtr` funcptr
// as Signed for downstream regalloc / flatten via
// `FunctionGraph::set_concretetype_of_inline(&var, ConcreteType::Signed)` —
// the rtyper-orthodox `v.concretetype = lltype` write
// (`rpython/rtyper/rtyper.py setconcretetype`).

/// RPython `ConcreteCallTableRow(dict)` (rpbc.py).
#[derive(Clone, Debug)]
pub struct ConcreteCallTableRow {
    pub row: HashMap<DescKey, _ptr>,
    pub fntype: FuncType,
    pub attrname: Option<String>,
}

type ConcreteCallTableRowRef = Rc<RefCell<ConcreteCallTableRow>>;

impl ConcreteCallTableRow {
    /// RPython `ConcreteCallTableRow.from_row(cls, rtyper, row)`.
    pub fn from_row(rtyper: &RPythonTyper, row: &CallTableRow) -> Result<Self, TyperError> {
        let mut concrete_row = HashMap::new();
        let mut last_llfn = None;
        for (funcdesc, graph) in row {
            let llfn = rtyper.getcallable(graph)?;
            last_llfn = Some(llfn.clone());
            concrete_row.insert(*funcdesc, llfn);
        }
        let fntype = match last_llfn
            .expect("ConcreteCallTableRow::from_row requires a non-empty row")
            ._TYPE
            .TO
            .clone()
        {
            PtrTarget::Func(func) => func,
            other => panic!("ConcreteCallTableRow expects Func ptr, got {other:?}"),
        };
        Ok(ConcreteCallTableRow {
            row: concrete_row,
            fntype,
            attrname: None,
        })
    }
}

impl PartialEq for ConcreteCallTableRow {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
    }
}

impl Eq for ConcreteCallTableRow {}

/// RPython `LLCallTable(object)` (rpbc.py).
#[derive(Clone, Debug, Default)]
pub struct LLCallTable {
    /// Upstream `self.table` keyed by `(shape, index)`.
    pub table: HashMap<(CallShape, usize), ConcreteCallTableRowRef>,
    /// Upstream `self.uniquerows`.
    pub uniquerows: Vec<ConcreteCallTableRowRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallTableLookupError;

impl LLCallTable {
    pub fn lookup(
        &self,
        row: &ConcreteCallTableRow,
    ) -> Result<(usize, ConcreteCallTableRow), CallTableLookupError> {
        for (existingindex, existingrow) in self.uniquerows.iter().enumerate() {
            let existingrow = existingrow.borrow();
            if row.fntype != existingrow.fntype {
                continue;
            }
            let mut mismatched = false;
            for (funcdesc, llfn) in &row.row {
                if let Some(existing_llfn) = existingrow.row.get(funcdesc)
                    && llfn != existing_llfn
                {
                    mismatched = true;
                    break;
                }
            }
            if mismatched {
                continue;
            }
            let mut merged = ConcreteCallTableRow {
                row: row.row.clone(),
                fntype: row.fntype.clone(),
                attrname: row.attrname.clone(),
            };
            for (desc, llfn) in &existingrow.row {
                merged.row.insert(*desc, llfn.clone());
            }
            if merged.row.len() == row.row.len() + existingrow.row.len() {
                // no common funcdesc, not a match
            } else {
                return Ok((existingindex, merged));
            }
        }
        Err(CallTableLookupError)
    }

    pub fn add(&mut self, row: ConcreteCallTableRow) {
        match self.lookup(&row) {
            Err(CallTableLookupError) => self.uniquerows.push(Rc::new(RefCell::new(row))),
            Ok((index, merged)) => {
                if merged == *self.uniquerows[index].borrow() {
                    return;
                }
                self.uniquerows.remove(index);
                self.add(merged);
            }
        }
    }
}

/// RPython `build_concrete_calltable(rtyper, callfamily)` (rpbc.py).
pub fn build_concrete_calltable(
    rtyper: &RPythonTyper,
    callfamily: &CallFamily,
) -> Result<LLCallTable, TyperError> {
    let mut llct = LLCallTable::default();
    let mut concreterows: HashMap<(CallShape, usize), ConcreteCallTableRow> = HashMap::new();

    for (shape, rows) in &callfamily.calltables {
        for (index, row) in rows.iter().enumerate() {
            let concreterow = ConcreteCallTableRow::from_row(rtyper, row)?;
            concreterows.insert((shape.clone(), index), concreterow.clone());
            llct.add(concreterow);
        }
    }

    for ((shape, index), row) in concreterows {
        let (existingindex, biggerrow) = llct.lookup(&row).expect("row must exist in LLCallTable");
        let canonical_row = Rc::clone(&llct.uniquerows[existingindex]);
        assert!(biggerrow == *canonical_row.borrow());
        llct.table.insert((shape, index), canonical_row);
    }

    if llct.uniquerows.len() == 1 {
        llct.uniquerows[0].borrow_mut().attrname = None;
    } else {
        for (finalindex, row) in llct.uniquerows.iter_mut().enumerate() {
            row.borrow_mut().attrname = Some(format!("variant{finalindex}"));
        }
    }

    Ok(llct)
}

/// RPython `small_cand(rtyper, s_pbc)` (rpbc.py).
///
/// ```python
/// def small_cand(rtyper, s_pbc):
///     if 1 < len(s_pbc.descriptions) < rtyper.getconfig().translation.withsmallfuncsets:
///         callfamily = s_pbc.any_description().getcallfamily()
///         llct = get_concrete_calltable(rtyper, callfamily)
///         if (len(llct.uniquerows) == 1 and
///                 (not s_pbc.subset_of or small_cand(rtyper, s_pbc.subset_of))):
///             return True
///     return False
/// ```
///
/// `withsmallfuncsets = 0` is upstream's default; with that value,
/// the strict `< withsmallfuncsets` bound is never satisfied and the
/// helper returns `false` without probing `callfamily`.
pub fn small_cand(rtyper: &RPythonTyper, s_pbc: &SomePBC) -> Result<bool, TyperError> {
    let withsmallfuncsets = rtyper
        .getconfig()
        .map(|cfg| cfg.translation.withsmallfuncsets)
        .unwrap_or(0);
    let nb = s_pbc.descriptions.len();
    if !(1 < nb && nb < withsmallfuncsets) {
        return Ok(false);
    }
    let Some(sample) = s_pbc.any_description() else {
        return Ok(false);
    };
    let sample_ref = sample.as_function().ok_or_else(|| {
        TyperError::message("small_cand: any_description() is not a FunctionDesc")
    })?;
    let callfamily = sample_ref.borrow().base.getcallfamily().map_err(|e| {
        TyperError::message(format!(
            "small_cand: getcallfamily failed: {}",
            e.msg.as_deref().unwrap_or("<no message>")
        ))
    })?;
    let llct = get_concrete_calltable(rtyper, &callfamily).map_err(|e| {
        TyperError::message(format!(
            "small_cand: get_concrete_calltable failed: {}",
            e.msg.as_deref().unwrap_or("<no message>")
        ))
    })?;
    if llct.uniquerows.len() != 1 {
        return Ok(false);
    }
    match &s_pbc.subset_of {
        None => Ok(true),
        Some(parent) => small_cand(rtyper, parent),
    }
}

/// RPython `get_concrete_calltable(rtyper, callfamily)` (rpbc.py).
pub fn get_concrete_calltable(
    rtyper: &RPythonTyper,
    callfamily: &Rc<RefCell<CallFamily>>,
) -> Result<LLCallTable, AnnotatorError> {
    let key = Rc::as_ptr(callfamily) as usize;
    {
        let concrete_calltables = rtyper.concrete_calltables.borrow();
        if let Some((llct, oldsize)) = concrete_calltables.get(&key) {
            if *oldsize != callfamily.borrow().total_calltable_size {
                return Err(AnnotatorError::new("call table was unexpectedly extended"));
            }
            return Ok(llct.clone());
        }
    }

    let llct = build_concrete_calltable(rtyper, &callfamily.borrow())
        .map_err(|e| AnnotatorError::new(e.to_string()))?;
    let oldsize = callfamily.borrow().total_calltable_size;
    rtyper
        .concrete_calltables
        .borrow_mut()
        .insert(key, (llct.clone(), oldsize));
    Ok(llct)
}

/// RPython `FunctionReprBase.call()` row-selection prefix
/// (rpbc.py).
#[derive(Clone, Debug)]
pub(crate) struct SelectedCallFamilyRow {
    pub(crate) shape: CallShape,
    pub(crate) index: usize,
    pub(crate) row_of_graphs: CallTableRow,
    pub(crate) anygraph: Rc<PyGraph>,
}

/// Select the `(shape, index)` row witness for a `SomePBC` call family,
/// mirroring the first half of `FunctionReprBase.call()`:
///
/// * `descs = list(s_pbc.descriptions)`
/// * `shape, index = self.callfamily.find_row(...)`
/// * `row_of_graphs = self.callfamily.calltables[shape][index]`
/// * `anygraph = row_of_graphs.itervalues().next()`
pub(crate) fn select_call_family_row(
    bookkeeper: &Rc<Bookkeeper>,
    callfamily: &Rc<RefCell<CallFamily>>,
    s_pbc: &SomePBC,
    args: &ArgumentsForTranslation,
    op_key: Option<PositionKey>,
) -> Result<SelectedCallFamilyRow, AnnotatorError> {
    let descs: Vec<_> = s_pbc.descriptions.values().cloned().collect();
    let (shape, index) = callfamily
        .borrow()
        .find_row(bookkeeper, &descs, args, op_key)?;
    let row_of_graphs = callfamily
        .borrow()
        .calltables
        .get(&shape)
        .and_then(|table| table.get(index))
        .cloned()
        .ok_or_else(|| AnnotatorError::new("FunctionReprBase.call: missing calltable row"))?;
    let anygraph = row_of_graphs
        .values()
        .next()
        .cloned()
        .ok_or_else(|| AnnotatorError::new("FunctionReprBase.call: empty row_of_graphs"))?;
    Ok(SelectedCallFamilyRow {
        shape,
        index,
        row_of_graphs,
        anygraph,
    })
}

/// Walk every `OpKind::Call` whose target is `CallTarget::Indirect` and
/// rewrite it into the RPython-orthodox pair:
///
/// 1. `OpKind::VtableMethodPtr { receiver, trait_root, method_name }`
///    → produces `funcptr: Variable` (Int kind)
/// 2. `OpKind::IndirectCall { funcptr, args, graphs, result_ty }`
///    → mirrors RPython `indirect_call(funcptr, *args, c_graphs)`
///
/// `args` remain the full call argument list, including the receiver.
/// RPython's `convert_to_concrete_llfn` materialises the funcptr but the
/// eventual `indirect_call` still receives `self, ...` as ordinary args.
pub fn lower_indirect_calls(graph: &mut JitFunctionGraph, call_control: &CallControl) {
    lower_indirect_calls_with(graph, call_control, true);
}

/// What [`lower_indirect_calls_with`] reads to fill an indirect call's
/// `c_graphs` row: the call family of each `(trait_root, method_name)`, the
/// builtin-wrapper family, and the result type the family declares.
pub(crate) trait IndirectCallFamilies {
    fn all_impls_for_indirect(&self, trait_root: &str, method_name: &str) -> Vec<CallPath>;
    fn builtin_wrapper_indirect_graphs(&self) -> &[CallPath];
    fn declared_result_type_for_indirect(
        &self,
        trait_root: &str,
        method_name: &str,
    ) -> Option<majit_ir::value::Type>;
}

impl IndirectCallFamilies for CallControl {
    fn all_impls_for_indirect(&self, trait_root: &str, method_name: &str) -> Vec<CallPath> {
        CallControl::all_impls_for_indirect(self, trait_root, method_name)
    }

    fn builtin_wrapper_indirect_graphs(&self) -> &[CallPath] {
        CallControl::builtin_wrapper_indirect_graphs(self)
    }

    fn declared_result_type_for_indirect(
        &self,
        trait_root: &str,
        method_name: &str,
    ) -> Option<majit_ir::value::Type> {
        CallControl::declared_result_type_for_indirect(self, trait_root, method_name)
    }
}

/// Same as [`lower_indirect_calls`]. `collapse_unresolved` rewrites an
/// empty lookup to the unknown family (`graphs: None`). The in-place pass
/// passes `false` and leaves that marker: the one empty family it still
/// sees is not registered on any translated graph.
pub(crate) fn lower_indirect_calls_with(
    graph: &mut JitFunctionGraph,
    call_control: &dyn IndirectCallFamilies,
    collapse_unresolved: bool,
) {
    // Generated gateway wrappers enter the MIR graph as a plain function-
    // pointer `IndirectCall` with `Some([])` as a deferred PBC-family marker.
    // At rtype time CallControl owns both the translated graphs and the
    // linker-resolved wrapper addresses, so fill the same `c_graphs` list
    // `FunctionReprBase.call()` appends in rpbc.py.
    let builtin_wrappers = call_control.builtin_wrapper_indirect_graphs();
    for block in &mut graph.blocks {
        for op in &mut block.operations {
            if let OpKind::IndirectCall {
                graphs, family_key, ..
            } = &mut op.kind
            {
                if let Some((trait_root, method_name)) = family_key.clone() {
                    let family = call_control.all_impls_for_indirect(&trait_root, &method_name);
                    if !family.is_empty() {
                        *graphs = Some(family);
                        *family_key = None;
                    } else if collapse_unresolved {
                        // An empty lookup cannot be told from "not
                        // registered yet". `None` is the unknown family.
                        // The in-place pass leaves the marker when asked
                        // not to collapse, so a later registration can
                        // still fill it.
                        *graphs = None;
                        *family_key = None;
                    }
                    continue;
                }
                if !graphs.as_deref().is_some_and(<[_]>::is_empty) {
                    continue;
                }
                // Filling the marker with an empty wrapper set would leave
                // `Some([])`, which every family analyzer reads as "the
                // family is known and has no members" and folds to its
                // bottom result. An unfilled marker is an unknown family
                // (`None`) when collapsing; otherwise it stays a marker.
                if !builtin_wrappers.is_empty() && collapse_unresolved {
                    *graphs = Some(builtin_wrappers.to_vec());
                } else if collapse_unresolved {
                    *graphs = None;
                }
            }
        }
    }

    // Collect the (block, op_index) sites first so the rewrite below
    // can mutate the graph without aliasing the borrow.
    let sites: Vec<(usize, usize)> = graph
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(bid, block)| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(move |(oi, op)| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::Indirect { .. },
                        ..
                    } => Some((bid, oi)),
                    _ => None,
                })
        })
        .collect();

    // Process in reverse so earlier indices stay valid as later sites
    // grow by 1 op (the inserted `VtableMethodPtr`).
    for (bid, oi) in sites.into_iter().rev() {
        let block_id = BlockId(bid);
        let op = graph.blocks[bid].operations[oi].clone();
        let (target, args, mut result_ty, result) = match op.kind {
            OpKind::Call {
                target,
                args,
                result_ty,
            } => (target, args, result_ty, op.result),
            _ => unreachable!("site filter mismatch"),
        };
        let (trait_root, method_name) = match target {
            CallTarget::Indirect {
                trait_root,
                method_name,
            } => (trait_root, method_name),
            _ => unreachable!("site filter mismatch"),
        };
        let family = call_control.all_impls_for_indirect(&trait_root, &method_name);
        // An empty lookup is an unknown family. Collapsing it to `None`
        // on the shared graph makes effect analysis report random effects
        // for a call the analyzers previously did not enumerate. Leave the
        // `CallTarget::Indirect` in place unless this pass is collapsing.
        if family.is_empty() && !collapse_unresolved {
            continue;
        }
        // `rpbc.py FunctionReprBase.call` assigns the call result's
        // concretetype from the selected call-family row.  Use that family
        // result here as well: a dependency LLBC can expose a transparent
        // scalar wrapper only as an opaque declaration, while a local family
        // member still supplies the complete translated signature.
        let declared = call_control.declared_result_type_for_indirect(&trait_root, &method_name);
        if let Some(declared) = declared {
            result_ty = match declared {
                majit_ir::value::Type::Int => ValueType::Int,
                majit_ir::value::Type::Ref => ValueType::Ref(None),
                majit_ir::value::Type::Float => ValueType::Float,
                majit_ir::value::Type::Void => ValueType::Void,
            };
            if let Some(result) = &result {
                JitFunctionGraph::set_concretetype_of_inline(
                    result,
                    crate::codewriter::type_state::valuetype_to_concrete(&result_ty),
                );
            }
        }
        let receiver_var = args
            .first()
            .unwrap_or_else(|| {
                panic!(
                    "dyn-Trait method call must have a receiver arg \
                     (trait_root={trait_root:?} method={method_name:?} graph={:?})",
                    graph.name,
                )
            })
            .clone();
        // RPython rclass.py getclsfield (condensed into a single op).
        let funcptr_var = rclass::class_get_method_ptr(
            graph,
            block_id,
            oi,
            receiver_var.into_variable(),
            trait_root.clone(),
            method_name.clone(),
        );
        // RPython rpbc.py:216 `c_graphs = row_of_graphs.values()` — full
        // family without JIT candidate filtering.
        //
        // `None` means "unknown family" — matches
        // `rpython/translator/backendopt/graphanalyze.py:117`, where
        // `graphs is None` short-circuits family analyzers to
        // `top_result()` (conservative: canraise, can_invalidate, etc.
        // default to True). `Some(vec![])` would instead be treated as
        // "empty family → No/false" by every analyzer, which is unsafe
        // when external/unregistered impls might still be called.
        // Pyre's source-only parser only sees `#[trait_method]`-
        // registered impls; if none are registered for a
        // `(trait, method)` family, we don't know whether that's
        // because the family is truly empty or because the impls live
        // outside the analyzed sources, so we take the conservative
        // `None` path.
        let graphs = if family.is_empty() {
            None
        } else {
            Some(family)
        };
        // The original Call op is now at index `oi + 1` because
        // `class_get_method_ptr` inserted `VtableMethodPtr` at `oi`.
        graph.blocks[bid].operations[oi + 1] = SpaceOperation {
            result,
            kind: OpKind::IndirectCall {
                funcptr: funcptr_var,
                args: crate::model::call_arg_vars(&args),
                graphs,
                family_key: None,
                result_ty,
            },
        };
    }
}

/// Debug-build invariant: after `lower_indirect_calls` runs, no
/// `CallTarget::Indirect` may survive in the graph. Callers can run
/// this assert to catch missed sites early.
#[cfg(debug_assertions)]
pub fn assert_no_indirect_call_targets(graph: &JitFunctionGraph) {
    for (bid, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            if let OpKind::Call {
                target: CallTarget::Indirect { .. },
                ..
            } = &op.kind
            {
                panic!(
                    "post-lowering invariant violation: \
                     CallTarget::Indirect survived at block {bid} op {oi}: {:?}",
                    op
                );
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::argument::ArgumentsForTranslation;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::description::{DescEntry, FunctionDesc};
    use crate::annotator::model::{SomeInteger, SomePBC, SomeValue};
    use crate::flowspace::model::{
        Block as FlowBlock, BlockRefExt as FlowBlockRefExt, ConstValue as FlowConstValue,
        Constant as FlowConstant, FunctionGraph as FlowFunctionGraph, GraphFunc as FlowGraphFunc,
        Hlvalue as FlowHlvalue, Link as FlowLink, Variable as FlowVariable,
    };
    use crate::model::{OpKind, ValueType};
    use crate::translator::rtyper::lltypesystem::lltype::{_ptr, FuncType, functionptr};

    fn make_rtyper() -> (Rc<RPythonAnnotator>, RPythonTyper) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        (ann, rtyper)
    }

    pub(crate) fn make_pygraph(name: &str) -> Rc<PyGraph> {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let arg_var = FlowVariable::named("x");
        arg_var.set_concretetype(Some(LowLevelType::Signed));
        arg_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let arg = FlowHlvalue::Variable(arg_var);
        let startblock = FlowBlock::shared(vec![arg.clone()]);
        let ret_var = FlowVariable::new();
        ret_var.set_concretetype(Some(LowLevelType::Void));
        ret_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let graph = FlowFunctionGraph::with_return_var(
            name,
            startblock.clone(),
            FlowHlvalue::Variable(ret_var),
        );
        let link = Rc::new(std::cell::RefCell::new(FlowLink::new(
            vec![arg],
            Some(graph.returnblock.clone()),
            None,
        )));
        startblock.closeblock(vec![link]);
        Rc::new(PyGraph {
            graph: Rc::new(std::cell::RefCell::new(graph)),
            func: FlowGraphFunc::new(
                name,
                FlowConstant::new(FlowConstValue::Dict(Default::default())),
            ),
            signature: std::cell::RefCell::new(crate::flowspace::argument::Signature::new(
                vec!["x".to_string()],
                None,
                None,
            )),
            defaults: std::cell::RefCell::new(None),
            access_directly: std::cell::Cell::new(false),
        })
    }

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    fn int_sig(names: &[&str]) -> crate::flowspace::argument::Signature {
        crate::flowspace::argument::Signature::new(
            names.iter().map(|name| name.to_string()).collect(),
            None,
            None,
        )
    }

    fn fake_llfn(name: &str, arity: usize) -> _ptr {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        functionptr(
            FuncType {
                args: vec![LowLevelType::Signed; arity],
                result: LowLevelType::Void,
            },
            name,
            None,
            None,
        )
    }

    #[test]
    fn build_concrete_calltable_merges_overlapping_rows() {
        let (_ann, rtyper) = make_rtyper();
        let mut family = CallFamily::new(DescKey(1));
        let shared = make_pygraph("shared");
        let mut row0 = CallTableRow::new();
        row0.insert(DescKey(10), shared.clone());
        row0.insert(DescKey(11), make_pygraph("left_only"));
        let mut row1 = CallTableRow::new();
        row1.insert(DescKey(10), shared);
        row1.insert(DescKey(12), make_pygraph("right_only"));
        family.calltables.insert(
            CallShape {
                shape_cnt: 1,
                shape_keys: Vec::new(),
                shape_star: false,
            },
            vec![row0, row1],
        );

        let llct = build_concrete_calltable(&rtyper, &family).unwrap();
        assert_eq!(llct.uniquerows.len(), 1);
        let merged = llct.uniquerows[0].borrow();
        assert_eq!(merged.row.len(), 3);
        assert_eq!(merged.attrname, None);
        assert_eq!(llct.table.len(), 2);
        assert_eq!(llct.table.values().next().unwrap().borrow().row.len(), 3);
    }

    #[test]
    fn build_concrete_calltable_table_rows_alias_uniquerows() {
        let (_ann, rtyper) = make_rtyper();
        let mut family = CallFamily::new(DescKey(1));
        let mut row0 = CallTableRow::new();
        row0.insert(DescKey(10), make_pygraph("a"));
        family.calltables.insert(
            CallShape {
                shape_cnt: 1,
                shape_keys: Vec::new(),
                shape_star: false,
            },
            vec![row0],
        );

        let llct = build_concrete_calltable(&rtyper, &family).unwrap();
        let shared = llct.table.values().next().expect("table row missing");
        assert!(Rc::ptr_eq(shared, &llct.uniquerows[0]));
    }

    #[test]
    fn build_concrete_calltable_assigns_variant_names_to_multiple_uniquerows() {
        let (_ann, rtyper) = make_rtyper();
        let mut family = CallFamily::new(DescKey(1));
        let mut row0 = CallTableRow::new();
        row0.insert(DescKey(10), make_pygraph("a"));
        let mut row1 = CallTableRow::new();
        row1.insert(DescKey(11), make_pygraph("b"));
        family.calltables.insert(
            CallShape {
                shape_cnt: 1,
                shape_keys: Vec::new(),
                shape_star: false,
            },
            vec![row0, row1],
        );

        let llct = build_concrete_calltable(&rtyper, &family).unwrap();
        assert_eq!(llct.uniquerows.len(), 2);
        let mut attrnames: Vec<_> = llct
            .uniquerows
            .iter()
            .map(|row| row.borrow().attrname.clone().unwrap())
            .collect();
        attrnames.sort();
        assert_eq!(
            attrnames,
            vec!["variant0".to_string(), "variant1".to_string()]
        );
        let mut table_attrnames: Vec<_> = llct
            .table
            .values()
            .map(|row| row.borrow().attrname.clone().unwrap())
            .collect();
        table_attrnames.sort();
        assert_eq!(
            table_attrnames,
            vec!["variant0".to_string(), "variant1".to_string()]
        );
    }

    #[test]
    fn llcalltable_lookup_rejects_rows_with_mismatched_fntype() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let mut llct = LLCallTable::default();
        llct.uniquerows
            .push(Rc::new(RefCell::new(ConcreteCallTableRow {
                row: HashMap::from([(DescKey(10), fake_llfn("a", 1))]),
                fntype: FuncType {
                    args: vec![LowLevelType::Signed],
                    result: LowLevelType::Void,
                },
                attrname: None,
            })));

        let row = ConcreteCallTableRow {
            row: HashMap::from([(DescKey(10), fake_llfn("a", 1))]),
            fntype: FuncType {
                args: vec![LowLevelType::Signed, LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            attrname: None,
        };

        assert!(llct.lookup(&row).is_err());
    }

    #[test]
    fn small_cand_returns_false_for_singleton_descriptions() {
        let (_ann, rtyper) = make_rtyper();
        let bk = bk();
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            bk,
            None,
            "single",
            int_sig(&["x"]),
            None,
            None,
        )));
        // `1 < len(descs)` is `1 < 1` → false → early return.
        let s_pbc_one = SomePBC::new(vec![DescEntry::function(fd)], false);
        assert!(!small_cand(&rtyper, &s_pbc_one).unwrap());
    }

    #[test]
    fn small_cand_returns_false_when_withsmallfuncsets_is_zero() {
        let (_ann, rtyper) = make_rtyper();
        let bk = bk();
        let fd_a = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            "a",
            int_sig(&["x"]),
            None,
            None,
        )));
        let fd_b = Rc::new(RefCell::new(FunctionDesc::new(
            bk,
            None,
            "b",
            int_sig(&["x"]),
            None,
            None,
        )));
        let s_pbc = SomePBC::new(
            vec![DescEntry::function(fd_a), DescEntry::function(fd_b)],
            false,
        );
        // Default withsmallfuncsets=0 → `2 < 0` is false → early return false.
        assert!(!small_cand(&rtyper, &s_pbc).unwrap());
    }

    #[test]
    fn get_concrete_calltable_caches_and_rejects_extension() {
        let (_ann, rtyper) = make_rtyper();
        let family = Rc::new(RefCell::new(CallFamily::new(DescKey(1))));
        let mut row0 = CallTableRow::new();
        row0.insert(DescKey(10), make_pygraph("a"));
        family.borrow_mut().calltables.insert(
            CallShape {
                shape_cnt: 1,
                shape_keys: Vec::new(),
                shape_star: false,
            },
            vec![row0],
        );
        family.borrow_mut().total_calltable_size = 1;

        let first = get_concrete_calltable(&rtyper, &family).unwrap();
        let second = get_concrete_calltable(&rtyper, &family).unwrap();
        assert_eq!(first.uniquerows.len(), second.uniquerows.len());
        assert_eq!(rtyper.concrete_calltables.borrow().len(), 1);

        family.borrow_mut().total_calltable_size = 2;
        let err = get_concrete_calltable(&rtyper, &family).unwrap_err();
        assert!(
            err.to_string()
                .contains("call table was unexpectedly extended")
        );
    }

    #[test]
    fn select_call_family_row_matches_find_row_and_returns_witness_graph() {
        let bk = bk();
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            "f",
            int_sig(&["x"]),
            None,
            None,
        )));
        let cached_graph = make_pygraph("f_graph");
        fd.borrow().cache.borrow_mut().insert(
            crate::annotator::description::GraphCacheKey::None,
            cached_graph.clone(),
        );
        let args = ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            std::slice::from_ref(&fd),
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        let s_pbc = SomePBC::new(vec![DescEntry::function(fd.clone())], false);
        let callfamily = fd.borrow().base.getcallfamily().unwrap();

        let selected = select_call_family_row(&bk, &callfamily, &s_pbc, &args, None).unwrap();

        assert_eq!(selected.shape.shape_cnt, 1);
        assert_eq!(selected.index, 0);
        assert_eq!(selected.row_of_graphs.len(), 1);
        assert!(Rc::ptr_eq(&selected.anygraph.graph, &cached_graph.graph));
    }

    fn build_indirect_graph() -> crate::model::FunctionGraph {
        let mut graph = crate::model::FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "h".to_string(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: crate::model::CallTarget::indirect("Handler", "run"),
                args: crate::model::call_args(vec![receiver_var]),
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);
        graph
    }

    /// Boundary: pre-lowering graph has `CallTarget::Indirect`;
    /// post-lowering graph has `VtableMethodPtr + IndirectCall` and
    /// zero `Indirect` targets.  Mirrors `rpbc.py call` emit shape.
    /// Runs through the cfg(test)-gated legacy annotate / resolve_types
    /// pair because `function_graph_to_flowspace` rejects
    /// `CallTarget::Indirect` (it is the rclass-rewrite pre-image,
    /// not a flowspace operand) — the production path lowers it
    /// away first.
    #[test]
    fn lower_indirect_calls_eliminates_call_target_indirect() {
        use crate::call::CallControl;
        use crate::model::CallTarget;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;

        let mut cc = CallControl::new();
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "A",
            crate::model::FunctionGraph::new("A::run"),
        );
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "B",
            crate::model::FunctionGraph::new("B::run"),
        );
        cc.find_all_graphs_for_tests();

        let mut graph = build_indirect_graph();

        // Pre-lowering: a CallTarget::Indirect exists.
        let pre_indirect = graph.blocks[graph.startblock.0]
            .operations
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Indirect { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(pre_indirect, 1, "expected 1 Indirect Call pre-lowering");

        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

        // Post-lowering: invariant — zero Indirect targets.
        #[cfg(debug_assertions)]
        assert_no_indirect_call_targets(&graph);

        // Post-lowering: exactly one VtableMethodPtr and one IndirectCall.
        let ops = &graph.blocks[graph.startblock.0].operations;
        let vtable_ptr_count = ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::VtableMethodPtr { .. }))
            .count();
        let indirect_call_count = ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::IndirectCall { .. }))
            .count();
        assert_eq!(vtable_ptr_count, 1);
        assert_eq!(indirect_call_count, 1);
    }

    /// `ClassRepr.getclsfield` produces a real vtable `getfield`, and
    /// `FunctionReprBase.call` only appends the PBC graph family before its
    /// `indirect_call`.  A MIR dynamic-call operand already contains that
    /// field projection, so rtyping must retain it rather than replace it with
    /// pyre's name-only `VtableMethodPtr` pseudo-op (which no backend can
    /// execute).
    #[test]
    fn existing_vtable_field_read_keeps_its_funcptr_and_only_fills_family() {
        use crate::call::CallControl;
        use crate::model::{FieldDescriptor, ValueType};

        let mut cc = CallControl::new();
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "A",
            crate::model::FunctionGraph::new("A::run"),
        );

        let mut graph = crate::model::FunctionGraph::new("caller");
        let receiver = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "vtable".to_string(),
                    ty: ValueType::Ref(Some("Handler::{vtable}".to_string())),
                    class_root: Some("Handler::{vtable}".to_string()),
                },
                true,
            )
            .unwrap();
        let funcptr = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: receiver,
                    field: FieldDescriptor::new(
                        "method_run",
                        Some("Handler::{vtable}".to_string()),
                    ),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph
            .push_op_var(
                graph.startblock,
                OpKind::IndirectCall {
                    funcptr: funcptr.clone(),
                    args: Vec::new(),
                    graphs: None,
                    family_key: Some(("Handler".to_string(), "run".to_string())),
                    result_ty: ValueType::Void,
                },
                true,
            )
            .unwrap();

        lower_indirect_calls(&mut graph, &cc);

        let ops = &graph.blocks[graph.startblock.0].operations;
        assert!(
            !ops.iter()
                .any(|op| matches!(op.kind, OpKind::VtableMethodPtr { .. })),
            "an existing getfield must not be replaced by VtableMethodPtr",
        );
        let indirect = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::IndirectCall {
                    funcptr: actual,
                    graphs,
                    family_key,
                    ..
                } => Some((actual, graphs, family_key)),
                _ => None,
            })
            .expect("indirect call survives rtyping");
        assert_eq!(indirect.0, &funcptr);
        assert_eq!(indirect.1.as_ref().map(Vec::len), Some(1));
        assert!(indirect.2.is_none(), "rtyper must consume the family key");
    }

    #[test]
    fn indirect_family_result_retypes_opaque_caller_result() {
        use crate::call::CallControl;
        use crate::codewriter::type_state::ConcreteType;

        let mut cc = CallControl::new();
        cc.register_trait_method(
            "pop",
            Some("Storage"),
            "<default methods of Storage>",
            crate::model::FunctionGraph::new("Storage::pop").with_return_type("i64"),
        );

        let mut graph = crate::model::FunctionGraph::new("caller");
        let receiver = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "storage".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::indirect("Storage", "pop"),
                    args: crate::model::call_args(vec![receiver]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        lower_indirect_calls(&mut graph, &cc);

        assert_eq!(
            JitFunctionGraph::concretetype_of(&result),
            ConcreteType::Signed
        );
        assert!(
            graph.blocks[graph.startblock.0]
                .operations
                .iter()
                .any(|op| matches!(
                    &op.kind,
                    OpKind::IndirectCall {
                        result_ty: ValueType::Int,
                        ..
                    }
                ))
        );
    }

    /// Regression: inherent (non-trait) method calls —
    /// `CallTarget::Method` targets — must pass through
    /// `lower_indirect_calls` unchanged.  `rpbc.py call` only
    /// rewrites the indirect-call dispatch (`s_pbc.callfamily`);
    /// inherent method calls are statically resolved upstream by
    /// the rtyper.
    #[test]
    fn inherent_method_unchanged_by_lowering() {
        use crate::call::CallControl;
        use crate::model::CallTarget;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;

        let mut cc = CallControl::new();
        cc.register_function_graph(
            crate::parse::CallPath::from_segments(["Foo", "bar"]),
            crate::model::FunctionGraph::new("Foo::bar"),
        );

        let mut graph = crate::model::FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "f".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::method("bar", Some("Foo".to_string())),
                args: crate::model::call_args(vec![receiver_var]),
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let pre_ops_len = graph.blocks[graph.startblock.0].operations.len();
        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

        // Same op count; the Call op survives untouched.
        let ops = &graph.blocks[graph.startblock.0].operations;
        assert_eq!(ops.len(), pre_ops_len);
        let method_count = ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Method { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(method_count, 1, "Method target must survive lowering");
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::VtableMethodPtr { .. })),
            "inherent call must not produce VtableMethodPtr"
        );
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::IndirectCall { .. })),
            "inherent call must not produce IndirectCall"
        );
    }

    /// Build a graph holding one deferred builtin-wrapper marker —
    /// `IndirectCall { graphs: Some([]) }`, the shape `front/mir.rs`
    /// emits for a gateway-wrapper call it cannot resolve yet.
    fn graph_with_deferred_wrapper_marker() -> JitFunctionGraph {
        let mut graph = JitFunctionGraph::new("outer");
        let funcptr_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "fp".to_string(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::IndirectCall {
                funcptr: funcptr_var,
                args: vec![],
                graphs: Some(Vec::new()),
                family_key: None,
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);
        graph
    }

    fn sole_indirect_call_family(graph: &JitFunctionGraph) -> Option<Vec<crate::parse::CallPath>> {
        let mut found = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::IndirectCall { graphs, .. } => Some(graphs.clone()),
                _ => None,
            });
        let family = found.next().expect("marker op must survive lowering");
        assert!(found.next().is_none(), "expected exactly one IndirectCall");
        family
    }

    /// The deferred builtin-wrapper marker is a request to enumerate a
    /// family, not a claim that the family is empty.  When `CallControl`
    /// has no wrappers to fill it with, the request went unanswered, and
    /// the op must say so — `graphs: None`, the same "unknown family"
    /// spelling `lower_indirect_calls` gives an unregistered
    /// `(trait, method)` pair.  Leaving `Some([])` would instead assert a
    /// closed, empty family, which every downstream analyzer folds to its
    /// bottom result (`graphanalyze.py:117-121` takes `None` to top).
    ///
    /// The registered-wrapper case is the non-vacuity control: it shows
    /// the fill still runs and still reaches the marker, so the `None`
    /// above is this rule and not a skipped pass.
    #[test]
    fn unfillable_wrapper_marker_lowers_to_unknown_family() {
        use crate::call::CallControl;
        use crate::parse::CallPath;

        // Control — the fill is engaged and rewrites the marker.
        let wrapper = CallPath::from_segments(["helpers", crate::runtime_names::symbols::WRAP_ADD]);
        let mut with_wrappers = CallControl::new();
        with_wrappers.register_function_graph(
            wrapper.clone(),
            JitFunctionGraph::new(crate::runtime_names::symbols::WRAP_ADD),
        );
        with_wrappers.register_function_fnaddr(wrapper.clone(), 0x1234);
        assert_eq!(
            with_wrappers.builtin_wrapper_indirect_graphs().to_vec(),
            vec![wrapper.clone()],
            "control needs exactly one registered wrapper"
        );

        let mut filled = graph_with_deferred_wrapper_marker();
        lower_indirect_calls(&mut filled, &with_wrappers);
        assert_eq!(
            sole_indirect_call_family(&filled),
            Some(vec![wrapper]),
            "a fillable marker must be filled with the wrapper family"
        );

        // Invariant — with nothing to enumerate, the family is unknown.
        let without_wrappers = CallControl::new();
        assert!(
            without_wrappers
                .builtin_wrapper_indirect_graphs()
                .is_empty(),
            "the invariant below needs an empty wrapper registry"
        );

        let mut unfilled = graph_with_deferred_wrapper_marker();
        lower_indirect_calls(&mut unfilled, &without_wrappers);
        assert_eq!(
            sole_indirect_call_family(&unfilled),
            None,
            "an unfillable marker is an unknown family, not an empty one"
        );
    }
}

// rpbc.py FunctionReprBase — full PBC Repr hierarchy
//
// Concrete classes ported below mirror upstream's hierarchy:
//   FunctionReprBase  (rpbc.py)  — shared base, composed-in
//   FunctionRepr      (rpbc.py)  — single concrete function PBC
//   FunctionsPBCRepr  (rpbc.py)  — multi-function PBC
//   SmallFunctionSetPBCRepr (rpbc.py)
//   SingleFrozenPBCRepr     (rpbc.py)
//   MultipleUnrelatedFrozenPBCRepr (rpbc.py)
//   MultipleFrozenPBCRepr   (rpbc.py)
//   MethodOfFrozenPBCRepr   (rpbc.py)
//   ClassesPBCRepr          (rclass-adjacent PBC dispatch)
//   MethodsPBCRepr          (method PBC dispatch)
//
// Pyre composes rather than inherits — see [`FunctionReprBase`] for the
// composition layout; each concrete struct embeds the shared base as a
// field and overrides the lowleveltype + dispatch body.  Pair-type arms
// (rpbc.py `__extend__(pairtype(...))`) live alongside their
// owning concrete reprs below.

use crate::flowspace::model::{ConstValue, Constant, GraphKey, Hlvalue};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{CanBeNull, Repr, ReprState};
use std::rc::Weak;
use std::sync::Arc;

/// RPython `class FunctionReprBase(Repr)` (rpbc.py).
///
/// Shared state for every function-PBC repr. Concrete subclasses
/// override `lowleveltype` and the call-dispatch `call()` body —
/// upstream has `FunctionRepr`, `FunctionsPBCRepr`, and
/// `SmallFunctionSetPBCRepr` subclassing this.
///
/// Pyre composes rather than inherits: concrete reprs (for now only
/// [`FunctionRepr`]) embed [`FunctionReprBase`] as a field so the
/// shared fields live in one place while each repr declares its own
/// `Repr` trait impl.
#[derive(Debug)]
pub struct FunctionReprBase {
    /// RPython `self.rtyper = rtyper` (rpbc.py:179). Weak because the
    /// rtyper owns the strong reference and each repr lives inside
    /// `RPythonTyper.reprs` cache.
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.s_pbc = s_pbc` (rpbc.py:180). Stored by value;
    /// upstream keeps the original Python reference, pyre clones the
    /// structured `SomePBC`.
    pub s_pbc: SomePBC,
    /// RPython `self.callfamily = s_pbc.any_description().getcallfamily()`
    /// (rpbc.py:181). Kept optional because test harnesses may build a
    /// `SomePBC` whose descriptions have no attached callfamily yet —
    /// upstream raises in that case; pyre defers the raise until the
    /// dispatcher that actually needs the callfamily fires.
    pub callfamily: Option<Rc<RefCell<CallFamily>>>,
}

impl FunctionReprBase {
    /// RPython `FunctionReprBase.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py).
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        // upstream: `self.callfamily =
        // s_pbc.any_description().getcallfamily()`.
        //
        // If `any_description()` returns `None` the PBC is empty — upstream
        // would have tripped the `assert len(descs) > 0` in the ctor.
        // `as_function()` yields the underlying FunctionDesc for both
        // FunctionDesc and MemoDesc (MemoDesc is-a FunctionDesc,
        // description.py). MethodDesc / FrozenDesc /
        // MethodOfFrozenDesc / ClassDesc also expose `getcallfamily` via
        // the base `Desc`, but only FunctionRepr needs the stored handle
        // today so those routes stay unpopulated until their concrete
        // reprs land.
        let callfamily = match s_pbc.any_description().and_then(|e| e.as_function()) {
            Some(rc) => rc.borrow().base.getcallfamily().ok(),
            None => None,
        };
        Ok(FunctionReprBase {
            rtyper: Rc::downgrade(rtyper),
            s_pbc,
            callfamily,
        })
    }

    /// RPython `FunctionReprBase.get_s_callable(self)` (rpbc.py).
    pub fn get_s_callable(&self) -> &SomePBC {
        &self.s_pbc
    }

    /// RPython `FunctionReprBase.get_s_signatures(self, shape)`
    /// (rpbc.py).
    ///
    /// ```python
    /// def get_s_signatures(self, shape):
    ///     funcdesc = self.s_pbc.any_description()
    ///     return funcdesc.get_s_signatures(shape)
    /// ```
    ///
    /// Only meaningful when the pbc's representative description is a
    /// `FunctionDesc`; other `DescEntry` variants return `None` since
    /// upstream's `get_s_signatures` method lives on `FunctionDesc` only.
    pub fn get_s_signatures(
        &self,
        shape: &CallShape,
    ) -> Result<
        Vec<(
            Vec<crate::annotator::model::SomeValue>,
            crate::annotator::model::SomeValue,
        )>,
        TyperError,
    > {
        let Some(entry) = self.s_pbc.any_description() else {
            return Err(TyperError::message(
                "FunctionReprBase.get_s_signatures: s_pbc is empty",
            ));
        };
        // MemoDesc is-a FunctionDesc (description.py), so accept its
        // wrapped base via `as_function()`.
        let Some(rc) = entry.as_function() else {
            return Err(TyperError::message(
                "FunctionReprBase.get_s_signatures: representative is not a FunctionDesc",
            ));
        };
        rc.borrow()
            .get_s_signatures(shape)
            .map_err(|e| TyperError::message(e.to_string()))
    }

    // Deferred ports — the remaining members of upstream
    // `FunctionReprBase` (rpbc.py) are:
    //
    //   * `rtype_simple_call(self, hop)`
    //   * `rtype_call_args(self, hop)`
    //   * `call(self, opname, hop)` — row selection + callparse +
    //     gencall
    //
    // All three depend on upstream `rpython/rtyper/callparse.py` and
    // `HighLevelOp.exception_is_here()` (ExceptionData). The row-
    // selection half of `call()` is already available as the free
    // function `select_call_family_row`; the rest
    // lands with the ExceptionData + callparse ports.
}

/// RPython `class FunctionRepr(FunctionReprBase)` (rpbc.py).
///
/// The repr for a statically known constant function — `SomePBC` with
/// a single `FunctionDesc` and `can_be_None=False`. `lowleveltype` is
/// `Void` because the callable is identity-encoded at compile time;
/// `convert_desc` / `convert_const` both return `None` (no runtime
/// value), and calls route through `convert_to_concrete_llfn` (port
/// pending).
#[derive(Debug)]
pub struct FunctionRepr {
    base: FunctionReprBase,
    state: ReprState,
    lltype: LowLevelType,
    /// Upgrades to the `Arc` `new` returned, so `get_r_implfunc` can
    /// hand back `(self, 0)` as `Arc<dyn Repr>`.
    self_weak: std::sync::Weak<FunctionRepr>,
}

impl FunctionRepr {
    /// RPython `FunctionRepr(rtyper, s_pbc)` — inherits `__init__` from
    /// `FunctionReprBase` and sets `lowleveltype = Void` as a
    /// class-level attribute (rpbc.py).
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Arc<Self>, TyperError> {
        let base = FunctionReprBase::new(rtyper, s_pbc)?;
        Ok(Arc::new_cyclic(|weak| FunctionRepr {
            base,
            state: ReprState::new(),
            lltype: LowLevelType::Void,
            self_weak: weak.clone(),
        }))
    }

    /// Access to the embedded [`FunctionReprBase`] state.
    pub fn base(&self) -> &FunctionReprBase {
        &self.base
    }

    /// RPython `FunctionRepr.convert_to_concrete_llfn(v, shape, index,
    /// llop)` (rpbc.py).
    ///
    /// ```python
    /// def convert_to_concrete_llfn(self, v, shape, index, llop):
    ///     assert v.concretetype == Void
    ///     funcdesc, = self.s_pbc.descriptions
    ///     row_of_one_graph = self.callfamily.calltables[shape][index]
    ///     graph = row_of_one_graph[funcdesc]
    ///     llfn = self.rtyper.getcallable(graph)
    ///     return inputconst(typeOf(llfn), llfn)
    /// ```
    ///
    /// `llop` is part of the upstream signature for subclasses that
    /// materialise funcptrs via `get_specfunc_row`; `FunctionRepr` is
    /// `Void`-typed so it returns a compile-time `Constant` directly
    /// without consuming `llop`.
    pub fn convert_to_concrete_llfn(
        &self,
        v: &crate::flowspace::model::Hlvalue,
        shape: &CallShape,
        index: usize,
        _llops: &Rc<RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
    ) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        // upstream: `assert v.concretetype == Void`.
        let v_ty = match v {
            crate::flowspace::model::Hlvalue::Variable(var) => var.concretetype(),
            crate::flowspace::model::Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        if v_ty != Some(LowLevelType::Void) {
            return Err(TyperError::message(
                "FunctionRepr.convert_to_concrete_llfn: v.concretetype != Void",
            ));
        }

        // upstream: `funcdesc, = self.s_pbc.descriptions`.
        if self.base.s_pbc.descriptions.len() != 1 {
            return Err(TyperError::message(
                "FunctionRepr.convert_to_concrete_llfn: expected a single FunctionDesc",
            ));
        }
        let funcdesc_entry = self
            .base
            .s_pbc
            .descriptions
            .values()
            .next()
            .expect("len checked");
        // `build_calltable_row` keys the row on `desc.rowkey()` (the
        // stable `Desc.identity`), not on pointer identity — so
        // lookup here has to follow the same route.
        let funcdesc_rowkey = funcdesc_entry
            .rowkey()
            .map_err(|e| TyperError::message(e.to_string()))?;

        // upstream: `self.callfamily.calltables[shape][index]`.
        let callfamily = self.base.callfamily.as_ref().ok_or_else(|| {
            TyperError::message("FunctionRepr.convert_to_concrete_llfn: callfamily not available")
        })?;
        let family_ref = callfamily.borrow();
        let table = family_ref.calltables.get(shape).ok_or_else(|| {
            TyperError::message("FunctionRepr.convert_to_concrete_llfn: shape not in calltables")
        })?;
        let row = table.get(index).ok_or_else(|| {
            TyperError::message("FunctionRepr.convert_to_concrete_llfn: row index out of range")
        })?;

        // upstream: `graph = row_of_one_graph[funcdesc]`.
        let graph = row.get(&funcdesc_rowkey).cloned().ok_or_else(|| {
            TyperError::message(
                "FunctionRepr.convert_to_concrete_llfn: funcdesc not in row_of_one_graph",
            )
        })?;

        // upstream: `llfn = self.rtyper.getcallable(graph); return
        //            inputconst(typeOf(llfn), llfn)`.
        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message(
                "FunctionRepr.convert_to_concrete_llfn: rtyper weak reference dropped",
            )
        })?;
        build_llfn_constant(&rtyper, &graph)
    }

    /// RPython `FunctionRepr.get_unique_llfn(self)` (rpbc.py).
    ///
    /// ```python
    /// def get_unique_llfn(self):
    ///     funcdesc, = self.s_pbc.descriptions
    ///     tables = []        # find the simple call in the calltable
    ///     for shape, table in self.callfamily.calltables.items():
    ///         if not shape[1] and not shape[2]:
    ///             tables.append(table)
    ///     if len(tables) != 1:
    ///         raise TyperError("cannot pass a function with various call shapes")
    ///     table, = tables
    ///     graphs = []
    ///     for row in table:
    ///         if funcdesc in row:
    ///             graphs.append(row[funcdesc])
    ///     if not graphs:
    ///         raise TyperError("cannot pass here a function that is not called")
    ///     graph = graphs[0]
    ///     if graphs != [graph] * len(graphs):
    ///         raise TyperError("cannot pass a specialized function here")
    ///     llfn = self.rtyper.getcallable(graph)
    ///     return inputconst(typeOf(llfn), llfn)
    /// ```
    pub fn get_unique_llfn(&self) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        // upstream: `funcdesc, = self.s_pbc.descriptions`.
        if self.base.s_pbc.descriptions.len() != 1 {
            return Err(TyperError::message(
                "FunctionRepr.get_unique_llfn: expected a single FunctionDesc",
            ));
        }
        let funcdesc_entry = self
            .base
            .s_pbc
            .descriptions
            .values()
            .next()
            .expect("len checked");
        // upstream `funcdesc in row` / `row[funcdesc]` — calltable rows
        // are keyed by `rowkey()` (stable identity), not by
        // pointer-derived `desc_key()`.
        let funcdesc_rowkey = funcdesc_entry
            .rowkey()
            .map_err(|e| TyperError::message(e.to_string()))?;

        let callfamily = self.base.callfamily.as_ref().ok_or_else(|| {
            TyperError::message("FunctionRepr.get_unique_llfn: callfamily not available")
        })?;
        let family_ref = callfamily.borrow();

        // upstream: walk `self.callfamily.calltables.items()` and keep
        // every table whose shape has no kwargs and no `*args`. `shape[1]`
        // is `shape_keys` and `shape[2]` is `shape_star`.
        let mut tables: Vec<&Vec<CallTableRow>> = Vec::new();
        for (shape, table) in &family_ref.calltables {
            if shape.shape_keys.is_empty() && !shape.shape_star {
                tables.push(table);
            }
        }
        if tables.len() != 1 {
            return Err(TyperError::message(
                "cannot pass a function with various call shapes",
            ));
        }
        let table = tables[0];

        // upstream: `graphs = [row[funcdesc] for row in table if
        //                      funcdesc in row]`.
        let mut graphs: Vec<Rc<PyGraph>> = Vec::new();
        for row in table {
            if let Some(graph) = row.get(&funcdesc_rowkey) {
                graphs.push(graph.clone());
            }
        }
        if graphs.is_empty() {
            return Err(TyperError::message(
                "cannot pass here a function that is not called",
            ));
        }
        let first = graphs[0].clone();
        // upstream: `if graphs != [graph] * len(graphs)` — every entry
        // must be the same graph (by identity).
        if graphs.iter().any(|g| !Rc::ptr_eq(g, &first)) {
            return Err(TyperError::message(
                "cannot pass a specialized function here",
            ));
        }

        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("FunctionRepr.get_unique_llfn: rtyper weak reference dropped")
        })?;
        build_llfn_constant(&rtyper, &first)
    }

    /// RPython `FunctionRepr.get_concrete_llfn(s_pbc, args_s, op)`
    /// (rpbc.py).
    ///
    /// ```python
    /// def get_concrete_llfn(self, s_pbc, args_s, op):
    ///     bk = self.rtyper.annotator.bookkeeper
    ///     funcdesc, = s_pbc.descriptions
    ///     with bk.at_position(None):
    ///         argspec = simple_args(args_s)
    ///         graph = funcdesc.get_graph(argspec, op)
    ///     llfn = self.rtyper.getcallable(graph)
    ///     return inputconst(typeOf(llfn), llfn)
    /// ```
    ///
    /// `op` is upstream's SpaceOperation identity key; the Rust port
    /// threads it as `Option<PositionKey>` to match
    /// [`FunctionDesc::get_graph`](crate::annotator::description::FunctionDesc::get_graph).
    pub(crate) fn get_concrete_llfn(
        &self,
        s_pbc: &SomePBC,
        args_s: Vec<crate::annotator::model::SomeValue>,
        op_key: Option<PositionKey>,
    ) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        // upstream: `funcdesc, = s_pbc.descriptions`.
        if s_pbc.descriptions.len() != 1 {
            return Err(TyperError::message(
                "FunctionRepr.get_concrete_llfn: expected a single FunctionDesc",
            ));
        }
        // MemoDesc is-a FunctionDesc (description.py): `as_function()`
        // yields the wrapped base for both.
        let funcdesc_rc = match s_pbc
            .descriptions
            .values()
            .next()
            .expect("len checked")
            .as_function()
        {
            Some(rc) => rc,
            None => {
                return Err(TyperError::message(
                    "FunctionRepr.get_concrete_llfn: single desc is not a FunctionDesc",
                ));
            }
        };

        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("FunctionRepr.get_concrete_llfn: rtyper weak reference dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("FunctionRepr.get_concrete_llfn: annotator weak reference dropped")
        })?;

        // upstream: `with bk.at_position(None): argspec = simple_args(args_s);
        //            graph = funcdesc.get_graph(argspec, op)`.
        let graph = {
            let _guard = annotator.bookkeeper.at_position(None);
            let argspec = crate::annotator::argument::simple_args(args_s);
            funcdesc_rc
                .borrow()
                .get_graph(&argspec, op_key)
                .map_err(|e| TyperError::message(e.to_string()))?
        };

        build_llfn_constant(&rtyper, &graph)
    }

    /// RPython `FunctionReprBase.call(self, hop)` (rpbc.py).
    /// FunctionRepr override — delegates to the shared free helper
    /// [`pbc_call_via_concrete_llfn`] with `FunctionRepr`'s
    /// `convert_to_concrete_llfn`.
    pub fn call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> Result<Option<Hlvalue>, TyperError> {
        pbc_call_via_concrete_llfn(&self.base, self, hop, |v, shape, idx, llops| {
            self.convert_to_concrete_llfn(v, shape, idx, llops)
        })
    }
}

/// Shared body of upstream `FunctionReprBase.call(self, hop)`
/// (rpbc.py):
///
/// ```python
/// def call(self, hop):
///     bk = self.rtyper.annotator.bookkeeper
///     args = hop.spaceop.build_args(hop.args_s[1:])
///     s_pbc = hop.args_s[0]   # possibly more precise than self.s_pbc
///     descs = list(s_pbc.descriptions)
///     shape, index = self.callfamily.find_row(bk, descs, args, hop.spaceop)
///     row_of_graphs = self.callfamily.calltables[shape][index]
///     anygraph = row_of_graphs.itervalues().next()
///     vfn = hop.inputarg(self, arg=0)
///     vlist = [self.convert_to_concrete_llfn(vfn, shape, index, hop.llops)]
///     vlist += callparse.callparse(self.rtyper, anygraph, hop)
///     rresult = callparse.getrresult(self.rtyper, anygraph)
///     hop.exception_is_here()
///     if isinstance(vlist[0], Constant):
///         v = hop.genop('direct_call', vlist, resulttype=rresult)
///     else:
///         vlist.append(hop.inputconst(Void, row_of_graphs.values()))
///         v = hop.genop('indirect_call', vlist, resulttype=rresult)
///     if hop.r_result is impossible_repr:
///         return None
///     else:
///         return hop.llops.convertvar(v, rresult, hop.r_result)
/// ```
///
/// Pyre composes rather than inherits — upstream's `call` lives on
/// `FunctionReprBase` and dispatches through virtual
/// `convert_to_concrete_llfn`. The Rust port encodes the polymorphism
/// as a `FnOnce` parameter so each concrete repr (`FunctionRepr`,
/// `FunctionsPBCRepr`, …) can plug in its own
/// `convert_to_concrete_llfn` body without losing the line-by-line
/// match against upstream `FunctionReprBase.call`.
pub(super) fn pbc_call_via_concrete_llfn<F>(
    base: &FunctionReprBase,
    self_repr: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    convert_to_concrete_llfn: F,
) -> Result<Option<Hlvalue>, TyperError>
where
    F: FnOnce(
        &Hlvalue,
        &CallShape,
        usize,
        &Rc<RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
    ) -> Result<Hlvalue, TyperError>,
{
    use crate::annotator::bookkeeper::build_args_for_op;
    use crate::annotator::model::SomeValue;
    use crate::translator::rtyper::callparse::{self, RResult};
    use crate::translator::rtyper::rmodel::impossible_repr;
    use crate::translator::rtyper::rtyper::GenopResult;

    // upstream: `bk = self.rtyper.annotator.bookkeeper`.
    let rtyper = base.rtyper.upgrade().ok_or_else(|| {
        TyperError::message("FunctionReprBase.call: rtyper weak reference dropped")
    })?;
    let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
        TyperError::message("FunctionReprBase.call: annotator weak reference dropped")
    })?;
    let bookkeeper = annotator.bookkeeper.clone();

    // upstream: `args = hop.spaceop.build_args(hop.args_s[1:])`.
    //
    // Pyre's analogue is `build_args_for_op(opname, args_s)` —
    // bookkeeper.rs — which mirrors the upstream
    // `CallOp.build_args` polymorphic dispatch
    // (`flowspace/operation.py SimpleCall simple_call`,
    // `:699 call_args`).
    let args_s_full = hop.args_s.borrow().clone();
    if args_s_full.is_empty() {
        return Err(TyperError::message(
            "FunctionReprBase.call: hop.args_s must contain the receiver",
        ));
    }
    // `hop.args_s` carries concrete `SomeValue`s by the time the
    // rtyper runs (annotator's fixpoint has converged), so wrap each
    // entry as `Some(_)` for the post-Item-#100 `build_args_for_op`
    // signature.
    let args_s_opt: Vec<Option<super::super::super::annotator::model::SomeValue>> =
        args_s_full[1..].iter().cloned().map(Some).collect();
    let args = build_args_for_op(&hop.spaceop.opname, &args_s_opt)
        .map_err(|e| TyperError::message(e.to_string()))?;

    // upstream: `s_pbc = hop.args_s[0]; descs = list(s_pbc.descriptions);
    //            shape, index = self.callfamily.find_row(bk, descs, args, hop.spaceop)
    //            row_of_graphs = self.callfamily.calltables[shape][index]
    //            anygraph = row_of_graphs.itervalues().next()`.
    //
    // Reuses `select_call_family_row` which mirrors the
    // four lines above. `op_key` is `None` because `HighLevelOp` does
    // not carry its enclosing block/graph identity — upstream caches
    // `find_row` keyed on `hop.spaceop` Python identity. The cache
    // miss is benign: `find_row` recomputes deterministically.
    let s_pbc = match args_s_full.first().cloned() {
        Some(SomeValue::PBC(pbc)) => pbc,
        Some(other) => {
            return Err(TyperError::message(format!(
                "FunctionReprBase.call: hop.args_s[0] is not a SomePBC: {other:?}"
            )));
        }
        None => unreachable!("emptiness checked above"),
    };
    let callfamily = base
        .callfamily
        .as_ref()
        .ok_or_else(|| TyperError::message("FunctionReprBase.call: callfamily not available"))?;
    // upstream `rpbc.py:204` passes `hop.spaceop` to `find_row`; pyre
    // carries the equivalent (graph + block + op_index identity) on
    // [`HighLevelOp::position_key`], stamped by `highlevelops_with_graph`
    // during `specialize_block`. Falls back to `None` for test-only
    // hops constructed via the bare [`HighLevelOp::new`] ctor — those
    // never reach a real callfamily lookup.
    let op_key = hop.position_key.borrow().clone();
    let row = select_call_family_row(&bookkeeper, callfamily, &s_pbc, &args, op_key)
        .map_err(|e| TyperError::message(e.to_string()))?;

    // upstream: `vfn = hop.inputarg(self, arg=0)`.
    let vfn = hop.inputarg(self_repr, 0)?;

    // upstream: `vlist = [self.convert_to_concrete_llfn(vfn, shape, index, hop.llops)]`.
    //
    // Pass the `Rc<RefCell<LowLevelOpList>>` directly so the multi-row
    // `FunctionsPBCRepr` branch can `borrow_mut()` to emit the `getfield`
    // op (rpbc.py). Single-row paths take a no-op pass-through
    // and never touch the inner buffer.
    let vlist0 = convert_to_concrete_llfn(&vfn, &row.shape, row.index, &hop.llops)?;
    let mut vlist: Vec<Hlvalue> = vec![vlist0];

    // upstream: `vlist += callparse.callparse(self.rtyper, anygraph, hop)`.
    vlist.extend(callparse::callparse(&rtyper, &row.anygraph, hop, None)?);

    // upstream: `rresult = callparse.getrresult(self.rtyper, anygraph)`.
    let rresult = callparse::getrresult(&rtyper, &row.anygraph)?;

    // upstream: `hop.exception_is_here()`.
    hop.exception_is_here()?;

    // upstream:
    //     if isinstance(vlist[0], Constant):
    //         v = hop.genop('direct_call', vlist, resulttype=rresult)
    //     else:
    //         vlist.append(hop.inputconst(Void, row_of_graphs.values()))
    //         v = hop.genop('indirect_call', vlist, resulttype=rresult)
    let resulttype = match &rresult {
        RResult::Repr(r) => GenopResult::Repr(r.clone()),
        // upstream `resulttype=lltype.Void` enters the LLType arm
        // (vresult.concretetype = Void; returns `Some(vresult)`), not
        // `GenopResult::Void` (which would return `None`).
        RResult::Void => GenopResult::LLType(LowLevelType::Void),
    };
    let v = if matches!(&vlist[0], Hlvalue::Constant(_)) {
        hop.genop("direct_call", vlist, resulttype)
    } else {
        // upstream `rpbc.py:216`:
        //     vlist.append(hop.inputconst(Void, row_of_graphs.values()))
        // pyre carries the candidate graphs as a `ConstValue::Graphs`
        // (graph identities via `GraphKey::as_usize`) since `Constant`
        // cannot hold raw `Rc<RefCell<FunctionGraph>>` without breaking
        // the wider `Sync`/`Hash` invariants (see `ConstValue::Graphs`
        // doc on `flowspace/model.rs`). Sorted so the constant is
        // deterministic across HashMap iteration orders — order has no
        // semantic meaning at the indirect_call protocol level
        // (`lower_indirect_calls` in this file still recovers the
        // graphs from `OpKind::Call { CallTarget::Indirect, .. }`,
        // which is the authoritative channel).
        let mut graph_ids: Vec<usize> = row
            .row_of_graphs
            .values()
            .map(|g| GraphKey::of(&g.graph).as_usize())
            .collect();
        graph_ids.sort_unstable();
        vlist.push(Hlvalue::Constant(
            crate::translator::rtyper::rtyper::HighLevelOp::inputconst(
                &LowLevelType::Void,
                &ConstValue::Graphs(graph_ids),
            )?,
        ));
        hop.genop("indirect_call", vlist, resulttype)
    };

    // upstream:
    //     if hop.r_result is impossible_repr:
    //         return None
    //     else:
    //         return hop.llops.convertvar(v, rresult, hop.r_result)
    let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
        TyperError::message("FunctionReprBase.call: hop.r_result not initialised")
    })?;
    // upstream `r is impossible_repr` is a Python identity check;
    // pyre stores the singleton `Arc<VoidRepr>` so the underlying
    // `VoidRepr` address is stable — compare via raw pointer through
    // `&dyn Repr` (matches the pattern at `rtyper.rs`
    // `Arc::ptr_eq`).
    let imp = impossible_repr();
    let imp_ptr = (imp.as_ref() as *const _ as *const ()) as usize;
    let r_result_ptr = (r_result.as_ref() as *const dyn Repr as *const ()) as usize;
    if r_result_ptr == imp_ptr {
        return Ok(None);
    }
    // `genop` always returns `Some(Variable)` because we passed a
    // typed `resulttype`; the `None` case (upstream `resulttype=None`)
    // does not apply here.
    let v_h = v.ok_or_else(|| {
        TyperError::message("FunctionReprBase.call: genop returned None despite typed resulttype")
    })?;
    let rresult_repr: std::sync::Arc<dyn Repr> = match rresult {
        RResult::Repr(r) => r,
        // Void rresult only reaches convertvar when r_result is also
        // impossible_repr — which already short-circuited above.
        // Defensive fallback: treat as the impossible_repr singleton
        // so identity in convertvar holds.
        RResult::Void => impossible_repr() as std::sync::Arc<dyn Repr>,
    };
    let converted =
        hop.llops
            .borrow_mut()
            .convertvar(v_h, rresult_repr.as_ref(), r_result.as_ref())?;
    Ok(Some(converted))
}

/// Build an `inputconst(typeOf(llfn), llfn)` Constant from an
/// `Rc<PyGraph>`, mirroring upstream's boilerplate tail of all three
/// `FunctionRepr` llfn helpers.
fn build_llfn_constant(
    rtyper: &Rc<RPythonTyper>,
    graph: &Rc<PyGraph>,
) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
    let func_ptr = rtyper.getcallable(graph)?;
    let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
    let c = crate::translator::rtyper::rmodel::inputconst_from_lltype(
        &func_ptr_type,
        &crate::flowspace::model::ConstValue::LLPtr(Box::new(func_ptr)),
    )?;
    Ok(crate::flowspace::model::Hlvalue::Constant(c))
}

impl Repr for FunctionRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "FunctionRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::FunctionRepr
    }

    /// RPython `FunctionReprBase.get_r_implfunc(self)` (rpbc.py) —
    /// upstream `return self, 0`. Inherited by `FunctionRepr` via the
    /// `FunctionReprBase` base class.
    fn get_r_implfunc(&self) -> Result<(Arc<dyn Repr>, usize), TyperError> {
        let arc = self.self_weak.upgrade().ok_or_else(|| {
            TyperError::message("FunctionRepr.get_r_implfunc: repr Arc was dropped")
        })?;
        Ok((arc, 0))
    }

    /// RPython `FunctionReprBase.s_pbc` (rpbc.py) — exposed via the
    /// trait so `MethodsPBCRepr.redispatch_call` (rpbc.py) can
    /// supply `subset_of=r_func.s_pbc` when narrowing per-call SomePBC.
    fn pbc_s_pbc(&self) -> Option<&SomePBC> {
        Some(&self.base.s_pbc)
    }

    fn get_s_signatures(
        &self,
        shape: &crate::flowspace::argument::CallShape,
    ) -> Result<
        Vec<(
            Vec<crate::annotator::model::SomeValue>,
            crate::annotator::model::SomeValue,
        )>,
        TyperError,
    > {
        self.base.get_s_signatures(shape)
    }

    /// RPython `FunctionRepr.convert_desc(self, funcdesc)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_desc(self, funcdesc):
    ///     return None
    /// ```
    ///
    /// Returns the "no runtime value" sentinel as a Void-typed
    /// `Constant(None, Void)`. Upstream's `None` return gets wrapped
    /// by `convert_desc_or_const` into a `Constant(None, Void)` at the
    /// call site in `rclass.setup_vtable`.
    fn convert_desc(&self, _desc: &DescEntry) -> Result<Constant, TyperError> {
        Ok(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    }

    /// RPython `FunctionRepr.convert_const(self, value)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     return None
    /// ```
    ///
    /// Mirrors `convert_desc`: no runtime value, Void-typed `Constant`.
    fn convert_const(&self, _value: &ConstValue) -> Result<Constant, TyperError> {
        Ok(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    }

    /// RPython `FunctionReprBase.rtype_simple_call(self, hop)`
    /// (rpbc.py) — `return self.call(hop)`. Inherited by
    /// `FunctionRepr` via `class FunctionRepr(FunctionReprBase)`.
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }

    /// RPython `FunctionReprBase.rtype_call_args(self, hop)`
    /// (rpbc.py) — `return self.call(hop)`. Inherited by
    /// `FunctionRepr` via `class FunctionRepr(FunctionReprBase)`.
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }
}

/// RPython `class FunctionsPBCRepr(CanBeNull, FunctionReprBase)`
/// (rpbc.py).
///
/// ```python
/// class FunctionsPBCRepr(CanBeNull, FunctionReprBase):
///     def __init__(self, rtyper, s_pbc):
///         FunctionReprBase.__init__(self, rtyper, s_pbc)
///         llct = get_concrete_calltable(self.rtyper, self.callfamily)
///         self.concretetable = llct.table
///         self.uniquerows = llct.uniquerows
///         if len(llct.uniquerows) == 1:
///             row = llct.uniquerows[0]
///             self.lowleveltype = row.fntype
///         else:
///             self.lowleveltype = self.setup_specfunc()
///         self.funccache = {}
/// ```
///
/// Pyre lands both the single-row case (one uniquerow → `lowleveltype =
/// row.fntype`, rpbc.py) and the multi-row `setup_specfunc`
/// branch (rpbc.py → `Self::setup_specfunc(...)` below). The
/// `CanBeNull` mixin (upstream rmodel.py — `rtype_bool` via
/// `ptr_nonzero`) is wired through `Self::rtype_bool` →
/// `can_be_null_rtype_bool`.
#[derive(Debug)]
pub struct FunctionsPBCRepr {
    /// Inherited state from `FunctionReprBase` (rpbc.py).
    pub base: FunctionReprBase,
    /// RPython `self.concretetable = llct.table` (rpbc.py).
    pub concretetable: HashMap<(CallShape, usize), ConcreteCallTableRowRef>,
    /// RPython `self.uniquerows = llct.uniquerows` (rpbc.py).
    pub uniquerows: Vec<ConcreteCallTableRowRef>,
    /// RPython `self.lowleveltype = row.fntype` (rpbc.py) or the
    /// deferred `self.setup_specfunc()` result (rpbc.py). Wrapped
    /// in `Ptr(...)` so the type is a function pointer as upstream —
    /// `row.fntype` is a `FuncType` and the pointer wrap mirrors
    /// `typeOf(llfn)` which returns `Ptr(FuncType)`.
    lltype: LowLevelType,
    /// RPython `self.funccache = {}` (rpbc.py). Keyed on
    /// `DescKey` (pointer-identity) matching `convert_desc`'s
    /// upstream `self.funccache[funcdesc]` lookup.
    pub funccache: RefCell<HashMap<DescKey, Constant>>,
    state: ReprState,
    /// Upgrades to the `Arc` `new` returned, so `get_r_implfunc` can
    /// hand back `(self, 0)` as `Arc<dyn Repr>`.
    self_weak: std::sync::Weak<FunctionsPBCRepr>,
}

impl FunctionsPBCRepr {
    /// RPython `FunctionsPBCRepr(rtyper, s_pbc)` (rpbc.py).
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Arc<Self>, TyperError> {
        let base = FunctionReprBase::new(rtyper, s_pbc)?;
        let callfamily = base.callfamily.clone().ok_or_else(|| {
            TyperError::message("FunctionsPBCRepr: sample FunctionDesc has no callfamily")
        })?;
        // upstream: `llct = get_concrete_calltable(self.rtyper, self.callfamily)`.
        let llct = get_concrete_calltable(rtyper, &callfamily)
            .map_err(|e| TyperError::message(e.to_string()))?;
        // upstream rpbc.py:232-239:
        //     if len(llct.uniquerows) == 1:
        //         row = llct.uniquerows[0]
        //         self.lowleveltype = row.fntype
        //     else:
        //         self.lowleveltype = self.setup_specfunc()
        let lltype = if llct.uniquerows.len() == 1 {
            let row = llct.uniquerows[0].borrow();
            LowLevelType::Ptr(Box::new(LLPtr {
                TO: PtrTarget::Func(row.fntype.clone()),
            }))
        } else {
            Self::setup_specfunc(&llct.uniquerows)?
        };
        Ok(Arc::new_cyclic(|weak| FunctionsPBCRepr {
            base,
            concretetable: llct.table,
            uniquerows: llct.uniquerows,
            lltype,
            funccache: RefCell::new(HashMap::new()),
            state: ReprState::new(),
            self_weak: weak.clone(),
        }))
    }

    /// RPython `FunctionsPBCRepr.setup_specfunc(self)` (rpbc.py):
    ///
    /// ```python
    /// def setup_specfunc(self):
    ///     fields = []
    ///     for row in self.uniquerows:
    ///         fields.append((row.attrname, row.fntype))
    ///     kwds = {'hints': {'immutable': True, 'static_immutable': True}}
    ///     return Ptr(Struct('specfunc', *fields, **kwds))
    /// ```
    ///
    /// Each multi-row `uniquerow` carries a unique `variant{N}` attrname
    /// stamped in [`get_concrete_calltable`]; we
    /// surface a `TyperError::message` if attrname is `None` to catch
    /// the impossible single-row-leaks-into-multi-row case.
    fn setup_specfunc(uniquerows: &[ConcreteCallTableRowRef]) -> Result<LowLevelType, TyperError> {
        let mut fields: Vec<(String, LowLevelType)> = Vec::with_capacity(uniquerows.len());
        for row_ref in uniquerows {
            let row = row_ref.borrow();
            let attrname = row.attrname.clone().ok_or_else(|| {
                TyperError::message(
                    "FunctionsPBCRepr.setup_specfunc: multi-row uniquerow has no \
                     attrname (get_concrete_calltable should have stamped variant{N})",
                )
            })?;
            let fnptr_ty = LowLevelType::Ptr(Box::new(LLPtr {
                TO: PtrTarget::Func(row.fntype.clone()),
            }));
            fields.push((attrname, fnptr_ty));
        }
        // upstream `kwds = {'hints': {'immutable': True, 'static_immutable': True}}`
        let hints = vec![
            ("immutable".to_string(), ConstValue::Bool(true)),
            ("static_immutable".to_string(), ConstValue::Bool(true)),
        ];
        let struct_t = Struct::with_hints("specfunc", fields, hints);
        Ok(LowLevelType::Ptr(Box::new(LLPtr {
            TO: PtrTarget::Struct(struct_t),
        })))
    }

    /// RPython `FunctionsPBCRepr.get_specfunc_row(self, llop, v, c_rowname,
    /// resulttype)` (rpbc.py):
    ///
    /// ```python
    /// def get_specfunc_row(self, llop, v, c_rowname, resulttype):
    ///     return llop.genop('getfield', [v, c_rowname], resulttype=resulttype)
    /// ```
    fn get_specfunc_row(
        &self,
        llop: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
        v: &Hlvalue,
        attrname: &str,
        resulttype: LowLevelType,
    ) -> Result<Hlvalue, TyperError> {
        let cname = Constant::with_concretetype(ConstValue::byte_str(attrname), LowLevelType::Void);
        let var = llop
            .genop(
                "getfield",
                vec![v.clone(), Hlvalue::Constant(cname)],
                crate::translator::rtyper::rtyper::GenopResult::LLType(resulttype),
            )
            .ok_or_else(|| {
                TyperError::message(
                    "FunctionsPBCRepr.get_specfunc_row: genop(getfield) returned None \
                     despite typed resulttype",
                )
            })?;
        Ok(Hlvalue::Variable(var))
    }

    /// RPython `FunctionsPBCRepr.convert_to_concrete_llfn(self, v,
    /// shape, index, llop)` (rpbc.py):
    ///
    /// ```python
    /// def convert_to_concrete_llfn(self, v, shape, index, llop):
    ///     assert v.concretetype == self.lowleveltype
    ///     if len(self.uniquerows) == 1:
    ///         return v
    ///     else:
    ///         row = self.concretetable[shape, index]
    ///         cname = inputconst(Void, row.attrname)
    ///         return self.get_specfunc_row(llop, v, cname, row.fntype)
    /// ```
    ///
    /// Both arms ported:
    ///   * Single-row (rpbc.py): returns `v` unchanged — the
    ///     funcptr is already the `Ptr(FuncType)` lowleveltype.
    ///   * Multi-row (rpbc.py): looks up the row in
    ///     `concretetable[shape, index]`, then emits
    ///     `getfield(v, c_rowname)` via [`Self::get_specfunc_row`] to
    ///     pull the per-variant funcptr out of the `specfunc` struct.
    pub fn convert_to_concrete_llfn(
        &self,
        v: &Hlvalue,
        shape: &CallShape,
        index: usize,
        llops: &Rc<RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
    ) -> Result<Hlvalue, TyperError> {
        // upstream: `assert v.concretetype == self.lowleveltype`.
        let v_ty = match v {
            Hlvalue::Variable(var) => var.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        if v_ty.as_ref() != Some(&self.lltype) {
            return Err(TyperError::message(format!(
                "FunctionsPBCRepr.convert_to_concrete_llfn: v.concretetype \
                 {:?} != self.lowleveltype {:?}",
                v_ty, self.lltype,
            )));
        }
        if self.uniquerows.len() == 1 {
            // upstream: `return v`.
            return Ok(v.clone());
        }
        // upstream rpbc.py:308-312:
        //     row = self.concretetable[shape, index]
        //     cname = inputconst(Void, row.attrname)
        //     return self.get_specfunc_row(llop, v, cname, row.fntype)
        let row_ref = self
            .concretetable
            .get(&(shape.clone(), index))
            .cloned()
            .ok_or_else(|| {
                TyperError::message(format!(
                    "FunctionsPBCRepr.convert_to_concrete_llfn: \
                     concretetable has no row for (shape={shape:?}, index={index})"
                ))
            })?;
        let row = row_ref.borrow();
        let attrname = row.attrname.clone().ok_or_else(|| {
            TyperError::message(
                "FunctionsPBCRepr.convert_to_concrete_llfn: \
                 multi-row concretetable entry has no attrname",
            )
        })?;
        let resulttype = LowLevelType::Ptr(Box::new(LLPtr {
            TO: PtrTarget::Func(row.fntype.clone()),
        }));
        let mut llop = llops.borrow_mut();
        self.get_specfunc_row(&mut llop, v, &attrname, resulttype)
    }

    /// RPython `FunctionReprBase.call(self, hop)` (rpbc.py).
    /// FunctionsPBCRepr override — delegates to the shared free helper
    /// [`pbc_call_via_concrete_llfn`] with `FunctionsPBCRepr`'s
    /// `convert_to_concrete_llfn`. For the single-row case the funcptr
    /// arrives as a `Variable` from `inputarg(self, 0)` (the arg
    /// carries `lowleveltype = Ptr(FuncType(...))`), so the dispatch
    /// takes the `indirect_call` branch.
    pub fn call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> Result<Option<Hlvalue>, TyperError> {
        pbc_call_via_concrete_llfn(&self.base, self, hop, |v, shape, idx, llops| {
            self.convert_to_concrete_llfn(v, shape, idx, llops)
        })
    }
}

impl Repr for FunctionsPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "FunctionsPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::FunctionsPBCRepr
    }

    /// RPython `FunctionReprBase.get_r_implfunc(self)` (rpbc.py) —
    /// upstream `return self, 0`. Inherited by `FunctionsPBCRepr` via the
    /// `FunctionReprBase` base class.
    fn get_r_implfunc(&self) -> Result<(Arc<dyn Repr>, usize), TyperError> {
        let arc = self.self_weak.upgrade().ok_or_else(|| {
            TyperError::message("FunctionsPBCRepr.get_r_implfunc: repr Arc was dropped")
        })?;
        Ok((arc, 0))
    }

    /// RPython `FunctionReprBase.s_pbc` (rpbc.py) — exposed via the
    /// trait so `MethodsPBCRepr.redispatch_call` (rpbc.py) can
    /// supply `subset_of=r_func.s_pbc` when narrowing per-call SomePBC.
    fn pbc_s_pbc(&self) -> Option<&SomePBC> {
        Some(&self.base.s_pbc)
    }

    fn get_s_signatures(
        &self,
        shape: &crate::flowspace::argument::CallShape,
    ) -> Result<
        Vec<(
            Vec<crate::annotator::model::SomeValue>,
            crate::annotator::model::SomeValue,
        )>,
        TyperError,
    > {
        self.base.get_s_signatures(shape)
    }

    /// RPython `FunctionsPBCRepr.convert_desc(self, funcdesc)`
    /// (rpbc.py).
    ///
    /// Both arms ported:
    ///   * Single-row (rpbc.py): returns the `_ptr` stored at
    ///     `row[funcdesc]`, or a fresh distinct-identity `_ptr` carrying
    ///     `DelayedPointer` for the "funcdesc missing from the row"
    ///     case (upstream's `rffi.cast(self.lowleveltype, ~len(funccache))`
    ///     int-encoded sentinel).
    ///   * Multi-row (rpbc.py): allocates a `specfunc` Struct
    ///     via `ll_malloc(immortal=True)` and `setattr`s each
    ///     `(attrname, llfn)` from `llfns`.
    ///
    /// Result is cached in `funccache` keyed by `DescKey` (rpbc.py).
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        let desc_rowkey = desc
            .rowkey()
            .map_err(|e| TyperError::message(e.to_string()))?;

        // upstream: `try: return self.funccache[funcdesc]`.
        if let Some(cached) = self.funccache.borrow().get(&desc_rowkey) {
            return Ok(cached.clone());
        }

        // upstream rpbc.py:261-271:
        //     llfns = {}
        //     found_anything = False
        //     for row in self.uniquerows:
        //         if funcdesc in row:
        //             llfn = row[funcdesc]
        //             found_anything = True
        //         else:
        //             llfn = nullptr(row.fntype.TO)
        //         llfns[row.attrname] = llfn
        let mut llfns: Vec<(Option<String>, _ptr)> = Vec::with_capacity(self.uniquerows.len());
        let mut last_llfn: Option<_ptr> = None;
        let mut found_anything = false;
        for row_ref in &self.uniquerows {
            let row = row_ref.borrow();
            let llfn = match row.row.get(&desc_rowkey) {
                Some(ptr) => {
                    found_anything = true;
                    ptr.clone()
                }
                None => ll_nullptr(LowLevelType::Func(Box::new(row.fntype.clone())))
                    .map_err(TyperError::message)?,
            };
            last_llfn = Some(llfn.clone());
            llfns.push((row.attrname.clone(), llfn));
        }

        // upstream rpbc.py:272-285:
        //     if len(self.uniquerows) == 1:
        //         if found_anything:
        //             result = llfn
        //         else:
        //             # extremely rare case
        //             result = rffi.cast(self.lowleveltype, ~len(self.funccache))
        //     else:
        //         result = self.create_specfunc()
        //         for attrname, llfn in llfns.items():
        //             setattr(result, attrname, llfn)
        let result_ptr: _ptr = if self.uniquerows.len() == 1 {
            if found_anything {
                last_llfn.expect("single-row loop must populate last_llfn when found_anything")
            } else {
                // upstream rpbc.py:275-280:
                //     # extremely rare case, shown only sometimes by
                //     # test_bug_callfamily: don't emit NULL, because
                //     # that would be interpreted as equal to None...
                //     # It should never be called anyway.
                //     result = rffi.cast(self.lowleveltype, ~len(self.funccache))
                //
                // Upstream's `~len(funccache)` casts an integer to a
                // function pointer; the value is unique per call, so
                // any two missing-from-row descs get distinct
                // non-NULL sentinels (so equality checks against None
                // and against each other resolve correctly).
                //
                // TODO: pyre's parity equivalent is
                // a fresh `_ptr` carrying [`DelayedPointer`] as the
                // `_obj0` slot. `_ptr`'s PartialEq (lltype.rs)
                // falls back to `_identity` for any non-Ok(Some)
                // `_obj0`, and each `_ptr::new` allocation gets a
                // fresh identity. The result: never-NULL,
                // distinct-per-call, never dereferenceable — same
                // observable semantics as upstream's int-encoded cast
                // without needing the llmemory/rffi int-to-ptr
                // machinery.
                //
                // **Convergence path** (TODO:
                // queue): once `rffi.cast` (rpython/rtyper/lltypesystem/
                // rffi.py) and the `llmemory.cast_int_to_adr` chain
                // (rpython/rtyper/lltypesystem/llmemory.py) port lands
                // — both are int-to-ptr primitives currently absent
                // from `lltypesystem/lltype.rs` — replace this branch
                // with the byte-for-byte `rffi.cast(self.lowleveltype,
                // !self.funccache.borrow().len())` call. The
                // `DelayedPointer` shim is then dead code and may be
                // deleted from this site (other DelayedPointer use
                // sites stand on their own). Test coverage to add at
                // that point: a port of `test_bug_callfamily`
                // (rpython/rtyper/test/test_rpbc.py).
                let LowLevelType::Ptr(boxed) = &self.lltype else {
                    return Err(TyperError::message(
                        "FunctionsPBCRepr.convert_desc: single-row lltype is not Ptr",
                    ));
                };
                _ptr::new_with_solid(
                    (**boxed).clone(),
                    Err(DelayedPointer),
                    /* _solid = */ true,
                )
            }
        } else {
            // upstream `self.create_specfunc()` ≡ `malloc(self.lowleveltype.TO,
            // immortal=True)` (rpbc.py). Here `self.lltype` is
            // `Ptr(Struct('specfunc', ...))`, so the `.TO` peels back to
            // the Struct.
            let LowLevelType::Ptr(struct_ptr) = &self.lltype else {
                return Err(TyperError::message(
                    "FunctionsPBCRepr.convert_desc: multi-row lltype is not Ptr(Struct)",
                ));
            };
            let PtrTarget::Struct(struct_t) = &struct_ptr.TO else {
                return Err(TyperError::message(
                    "FunctionsPBCRepr.convert_desc: multi-row lltype.TO is not Struct",
                ));
            };
            let struct_lltype = LowLevelType::Struct(Box::new(struct_t.clone()));
            let mut specfunc_ptr = ll_malloc(struct_lltype, None, MallocFlavor::Raw, true)
                .map_err(TyperError::message)?;
            // upstream `for attrname, llfn in llfns.items(): setattr(result, attrname, llfn)`.
            for (attrname_opt, llfn) in &llfns {
                let attrname = attrname_opt.as_deref().ok_or_else(|| {
                    TyperError::message(
                        "FunctionsPBCRepr.convert_desc: multi-row uniquerow has no attrname",
                    )
                })?;
                let llfn_value =
                    crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Ptr(Box::new(
                        llfn.clone(),
                    ));
                specfunc_ptr
                    .setattr(attrname, llfn_value)
                    .map_err(TyperError::message)?;
            }
            specfunc_ptr
        };

        // upstream `self.funccache[funcdesc] = result; return result`.
        let c = crate::translator::rtyper::rmodel::inputconst_from_lltype(
            &self.lltype,
            &ConstValue::LLPtr(Box::new(result_ptr)),
        )?;
        self.funccache.borrow_mut().insert(desc_rowkey, c.clone());
        Ok(c)
    }

    /// RPython `CanBeNull.rtype_bool(self, hop)` (rmodel.py) —
    /// `FunctionsPBCRepr` inherits this via `class FunctionsPBCRepr(CanBeNull,
    /// FunctionReprBase)` (rpbc.py). Delegates to
    /// [`crate::translator::rtyper::rmodel::can_be_null_rtype_bool`] which
    /// emits the constant fast-path or a `ptr_nonzero` guard.
    fn rtype_bool(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        crate::translator::rtyper::rmodel::can_be_null_rtype_bool(self, hop)
    }

    /// RPython `FunctionsPBCRepr.convert_const(self, value)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if isinstance(value, types.MethodType) and value.im_self is None:
    ///         value = value.im_func  # unbound method -> bare function
    ///     elif isinstance(value, staticmethod):
    ///         value = value.__get__(42)  # staticmethod unwrap
    ///     if value is None:
    ///         null = nullptr(self.lowleveltype.TO)
    ///         return null
    ///     funcdesc = self.rtyper.annotator.bookkeeper.getdesc(value)
    ///     return self.convert_desc(funcdesc)
    /// ```
    ///
    /// `types.MethodType` with `im_self is None` is a Python 2 holdover
    /// — pyre's `HostObject::BoundMethod` always carries non-None
    /// `self_obj`, so the unbound-method unwrap never fires here. The
    /// staticmethod unwrap is wired through
    /// [`HostObject::staticmethod_func`] and runs BEFORE the None
    /// check so the rare `staticmethod(None)` case behaves as upstream
    /// would (rpbc.py).
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        // upstream rpbc.py:291-292 staticmethod unwrap fires BEFORE the
        // None check; keep that order. The borrow-stable holder is
        // needed because `host.staticmethod_func()` returns a borrow,
        // and we need an owned value rebind to drop into the None /
        // getdesc paths below.
        let unwrapped: ConstValue;
        let value: &ConstValue = match value {
            ConstValue::HostObject(host) if host.is_staticmethod() => {
                let func = host.staticmethod_func().ok_or_else(|| {
                    TyperError::message(
                        "FunctionsPBCRepr.convert_const: is_staticmethod \
                         host without staticmethod_func",
                    )
                })?;
                unwrapped = ConstValue::HostObject(func.clone());
                &unwrapped
            }
            other => other,
        };

        if matches!(value, ConstValue::None) {
            let LowLevelType::Ptr(ptr) = &self.lltype else {
                return Err(TyperError::message(
                    "FunctionsPBCRepr.convert_const: lowleveltype is not Ptr",
                ));
            };
            let null_ptr = ptr.as_ref().clone()._defl();
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(null_ptr)),
                self.lltype.clone(),
            ));
        }
        // upstream rpbc.py:296-297:
        //     funcdesc = self.rtyper.annotator.bookkeeper.getdesc(value)
        //     return self.convert_desc(funcdesc)
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "FunctionsPBCRepr.convert_const: non-None value is not a \
                 HostObject (got {value:?}); upstream `bookkeeper.getdesc` \
                 only accepts callable host objects",
            )));
        };
        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("FunctionsPBCRepr.convert_const: rtyper has been dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("FunctionsPBCRepr.convert_const: annotator has been dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }

    /// RPython `FunctionReprBase.rtype_simple_call(self, hop)`
    /// (rpbc.py) — `return self.call(hop)`. Inherited by
    /// `FunctionsPBCRepr` via `class FunctionsPBCRepr(CanBeNull,
    /// FunctionReprBase)`.
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }

    /// RPython `FunctionReprBase.rtype_call_args(self, hop)`
    /// (rpbc.py) — `return self.call(hop)`. Inherited by
    /// `FunctionsPBCRepr` via `class FunctionsPBCRepr(CanBeNull,
    /// FunctionReprBase)`.
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }
}

/// RPython `class SmallFunctionSetPBCRepr(FunctionReprBase)`
/// (rpbc.py).
///
/// ```python
/// class SmallFunctionSetPBCRepr(FunctionReprBase):
///     def __init__(self, rtyper, s_pbc):
///         FunctionReprBase.__init__(self, rtyper, s_pbc)
///         llct = get_concrete_calltable(self.rtyper, self.callfamily)
///         assert len(llct.uniquerows) == 1
///         self.lowleveltype = Char
///         self.pointer_repr = FunctionsPBCRepr(rtyper, s_pbc)
///         self._conversion_tables = {}
///         self._compression_function = None
///         self._dispatch_cache = {}
/// ```
///
/// Char-indexed compact PBC of multiple FunctionDescs (chosen by
/// `small_cand`). Each desc maps to a distinct `chr(i)` index; runtime
/// dispatch reads `c_pointer_table[v_int]` to recover a function
/// pointer and `direct_call`s it.
///
/// `_conversion_tables` cache (rpbc.py + rpbc.py): keyed
/// on the *identity* of the target `SmallFunctionSetPBCRepr` (Python
/// `r_from._conversion_tables[r_to]` is a dict-by-object-identity
/// lookup). The Rust port uses the raw pointer address of the
/// `&SmallFunctionSetPBCRepr` receiver as the key; this is sound
/// because every live `SmallFunctionSetPBCRepr` is owned by an
/// `Arc<dyn Repr>` slot in `RPythonTyper`'s per-`SomePBC` repr cache
/// (rtyper.rs `getrepr` table) which outlives the cache, and
/// `pair_..._convert_from_to` only inspects (does not store) the
/// pointer.
#[derive(Debug)]
pub struct SmallFunctionSetPBCRepr {
    /// RPython `FunctionReprBase.__init__` (rpbc.py). Carries
    /// `rtyper`, `s_pbc`, `callfamily`.
    pub base: FunctionReprBase,
    /// RPython `self.pointer_repr = FunctionsPBCRepr(rtyper, s_pbc)`
    /// (rpbc.py). The wider repr that the small-funcset pulls
    /// per-desc function pointers from.
    pub pointer_repr: Arc<FunctionsPBCRepr>,
    /// RPython `self.descriptions = list(self.s_pbc.descriptions)`
    /// (rpbc.py:413), with `None` inserted at index 0 when
    /// `self.s_pbc.can_be_None` (rpbc.py:414-415). Populated by
    /// `_setup_repr`.
    pub descriptions: RefCell<Vec<Option<DescEntry>>>,
    /// RPython `self.c_pointer_table = inputconst(Ptr(POINTER_TABLE),
    /// pointer_table)` (rpbc.py). Built by `_setup_repr` from a
    /// `malloc(POINTER_TABLE, len(descriptions), immortal=True)` array
    /// pre-filled with the per-desc / `convert_const(None)` function
    /// pointers. Read by `pairtype(SmallFunctionSetPBCRepr,
    /// FunctionsPBCRepr).convert_from_to` (rpbc.py) and by the
    /// dispatcher (Step B.3).
    pub c_pointer_table: RefCell<Option<Constant>>,
    /// RPython `self._dispatch_cache = {}` (rpbc.py). Caches the
    /// per-`(shape, index, argtypes, resulttype)` dispatcher
    /// `Constant` produced by [`SmallFunctionSetPBCRepr::dispatcher`].
    /// Each entry holds the function pointer at the dispatcher graph's
    /// `getfunctionptr` typed result.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub _dispatch_cache:
        RefCell<HashMap<(CallShape, usize, Vec<LowLevelType>, LowLevelType), Constant>>,
    /// RPython `self._compression_function = None` (rpbc.py).
    /// Caches the synthesized `ll_compress` helper graph's function
    /// pointer Constant. The first
    /// `pair(FunctionsPBCRepr, SmallFunctionSetPBCRepr).convert_from_to`
    /// hit builds the helper via [`SmallFunctionSetPBCRepr::compression_function`]
    /// and stashes the resulting `Constant` here.
    pub _compression_function: RefCell<Option<Constant>>,
    /// RPython `self._conversion_tables = {}` (rpbc.py). Caches
    /// `conversion_table(r_from=self, r_to)` results keyed on the
    /// *identity* of the target `SmallFunctionSetPBCRepr`. `Some(c)`
    /// stores the `Char Array` indexing constant; `None` records an
    /// identity conversion (upstream's `r = None` branch when
    /// `l == range(len(r_from.descriptions))`, rpbc.py) so
    /// the caller skips the `getarrayitem` and returns `v` unchanged.
    /// Key = pointer address of the target repr (see struct doc above
    /// for the lifetime contract).
    pub _conversion_tables: RefCell<HashMap<usize, Option<Constant>>>,
    /// RPython `self.lowleveltype = Char` (rpbc.py). Stored
    /// explicitly because `Repr::lowleveltype` returns `&LowLevelType`.
    lltype: LowLevelType,
    state: ReprState,
    /// Upgrades to the `Arc` `new` returned, so `get_r_implfunc` can
    /// hand back `(self, 0)` as `Arc<dyn Repr>`.
    self_weak: std::sync::Weak<SmallFunctionSetPBCRepr>,
}

impl SmallFunctionSetPBCRepr {
    /// RPython `SmallFunctionSetPBCRepr.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py).
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Arc<Self>, TyperError> {
        // upstream rpbc.py — `FunctionReprBase.__init__(...)`.
        let base = FunctionReprBase::new(rtyper, s_pbc.clone())?;
        // upstream rpbc.py:396-397 —
        //   `llct = get_concrete_calltable(self.rtyper, self.callfamily)`
        //   `assert len(llct.uniquerows) == 1`.
        let callfamily = base.callfamily.as_ref().ok_or_else(|| {
            TyperError::message(
                "SmallFunctionSetPBCRepr: callfamily missing — small_cand should have \
                 rejected a non-callable PBC",
            )
        })?;
        let llct = get_concrete_calltable(rtyper, callfamily)
            .map_err(|e| TyperError::message(e.to_string()))?;
        if llct.uniquerows.len() != 1 {
            return Err(TyperError::message(format!(
                "SmallFunctionSetPBCRepr: expected len(uniquerows) == 1, got {}",
                llct.uniquerows.len()
            )));
        }
        // upstream rpbc.py:399 — `self.pointer_repr =
        //                           FunctionsPBCRepr(rtyper, s_pbc)`.
        let pointer_repr = FunctionsPBCRepr::new(rtyper, s_pbc)?;
        Ok(Arc::new_cyclic(|weak| SmallFunctionSetPBCRepr {
            base,
            pointer_repr,
            descriptions: RefCell::new(Vec::new()),
            c_pointer_table: RefCell::new(None),
            // upstream rpbc.py — `self._dispatch_cache = {}`.
            _dispatch_cache: RefCell::new(HashMap::new()),
            // upstream rpbc.py — `self._compression_function = None`.
            _compression_function: RefCell::new(None),
            // upstream rpbc.py — `self._conversion_tables = {}`.
            _conversion_tables: RefCell::new(HashMap::new()),
            // upstream rpbc.py — `self.lowleveltype = Char`.
            lltype: LowLevelType::Char,
            state: ReprState::new(),
            self_weak: weak.clone(),
        }))
    }

    /// RPython `SmallFunctionSetPBCRepr._invent_dispatcher_name(self, row)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def _invent_dispatcher_name(self, row):
    ///     import os
    ///     names = [value.name.rsplit(".", 1)[-1] for value in row.itervalues()]
    ///     commonprefix = os.path.commonprefix(names) # bit silly, but works well
    ///
    ///     if not commonprefix:
    ///         commonprefix = sorted(names)[0] + "_etc" # just pick one
    ///     return "dispatcher_" + commonprefix
    /// ```
    pub fn _invent_dispatcher_name(row: &CallTableRow) -> String {
        // upstream `for value in row.itervalues()` — `value` is a graph;
        // `value.name` is the graph's qualified name. Take the last
        // dotted segment (`name.rsplit(".", 1)[-1]`).
        let mut names: Vec<String> = row
            .values()
            .map(|graph| {
                let full = graph.func.name.clone();
                full.rsplit_once('.')
                    .map(|(_, last)| last.to_string())
                    .unwrap_or(full)
            })
            .collect();

        // upstream `commonprefix = os.path.commonprefix(names)` — the
        // longest character-prefix shared by all entries (no path
        // semantics; Python's `os.path.commonprefix` is a string-prefix
        // operation despite its name).
        let commonprefix = if names.is_empty() {
            String::new()
        } else {
            let mut prefix: String = names[0].clone();
            for name in names.iter().skip(1) {
                let new_len = prefix
                    .chars()
                    .zip(name.chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                prefix.truncate(
                    prefix
                        .char_indices()
                        .nth(new_len)
                        .map(|(i, _)| i)
                        .unwrap_or(prefix.len()),
                );
                if prefix.is_empty() {
                    break;
                }
            }
            prefix
        };

        // upstream `if not commonprefix: commonprefix = sorted(names)[0] +
        // "_etc"`.
        let commonprefix = if commonprefix.is_empty() {
            names.sort();
            match names.into_iter().next() {
                Some(first) => format!("{first}_etc"),
                None => String::from("etc"),
            }
        } else {
            commonprefix
        };

        // upstream: `return "dispatcher_" + commonprefix`.
        format!("dispatcher_{commonprefix}")
    }

    /// RPython `SmallFunctionSetPBCRepr.dispatcher(self, shape, index,
    /// argtypes, resulttype)` (rpbc.py):
    ///
    /// ```python
    /// def dispatcher(self, shape, index, argtypes, resulttype):
    ///     key = shape, index, tuple(argtypes), resulttype
    ///     if key in self._dispatch_cache:
    ///         return self._dispatch_cache[key]
    ///     graph = self.make_dispatcher(shape, index, argtypes, resulttype)
    ///     self.rtyper.annotator.translator.graphs.append(graph)
    ///     ll_ret = getfunctionptr(graph)
    ///     c_ret = self._dispatch_cache[key] = inputconst(typeOf(ll_ret), ll_ret)
    ///     return c_ret
    /// ```
    pub fn dispatcher(
        &self,
        shape: &CallShape,
        index: usize,
        argtypes: &[LowLevelType],
        resulttype: &LowLevelType,
    ) -> Result<Constant, TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype;

        // upstream: `key = shape, index, tuple(argtypes), resulttype`.
        let key = (shape.clone(), index, argtypes.to_vec(), resulttype.clone());
        // upstream: `if key in self._dispatch_cache: return
        //                          self._dispatch_cache[key]`.
        if let Some(cached) = self._dispatch_cache.borrow().get(&key) {
            return Ok(cached.clone());
        }
        // upstream: `graph = self.make_dispatcher(shape, index, argtypes,
        //                                          resulttype)`.
        let graph = self.make_dispatcher(shape, index, argtypes, resulttype)?;
        // upstream: `self.rtyper.annotator.translator.graphs.append(graph)`.
        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.dispatcher: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.dispatcher: annotator weak ref dropped")
        })?;
        annotator.translator.graphs.borrow_mut().push(graph.clone());
        // upstream: `ll_ret = getfunctionptr(graph)`. The dispatcher's
        // inputargs / returnvar all carry their concretetype directly
        // from `make_dispatcher`'s `set_concretetype` calls, so the
        // closure simply pulls the recorded concretetype off each
        // Hlvalue.
        let ll_ret = lltype::getfunctionptr(&graph, |v| match v {
            Hlvalue::Variable(var) => var.concretetype().ok_or_else(|| {
                TyperError::message(
                    "SmallFunctionSetPBCRepr.dispatcher: dispatcher graph \
                     argument missing concretetype",
                )
            }),
            Hlvalue::Constant(c) => c.concretetype.clone().ok_or_else(|| {
                TyperError::message(
                    "SmallFunctionSetPBCRepr.dispatcher: dispatcher graph \
                     constant argument missing concretetype",
                )
            }),
        })?;
        // upstream: `c_ret = self._dispatch_cache[key] =
        //                       inputconst(typeOf(ll_ret), ll_ret)`.
        let llfn_type = LowLevelType::Ptr(Box::new(lltype::typeOf(&ll_ret)));
        let c_ret = Constant::with_concretetype(ConstValue::LLPtr(Box::new(ll_ret)), llfn_type);
        self._dispatch_cache.borrow_mut().insert(key, c_ret.clone());
        Ok(c_ret)
    }

    /// RPython `SmallFunctionSetPBCRepr.make_dispatcher(self, shape,
    /// index, argtypes, resulttype)` (rpbc.py):
    ///
    /// ```python
    /// def make_dispatcher(self, shape, index, argtypes, resulttype):
    ///     inputargs = [varoftype(t) for t in [Char] + argtypes]
    ///     startblock = Block(inputargs)
    ///     startblock.exitswitch = inputargs[0]
    ///     graph = FunctionGraph("dispatcher", startblock, varoftype(resulttype))
    ///     row_of_graphs = self.callfamily.calltables[shape][index]
    ///     links = []
    ///     descs = list(self.s_pbc.descriptions)
    ///     if self.s_pbc.can_be_None:
    ///         descs.insert(0, None)
    ///     for desc in descs:
    ///         if desc is None:
    ///             continue
    ///         args_v = [varoftype(t) for t in argtypes]
    ///         b = Block(args_v)
    ///         llfn = self.rtyper.getcallable(row_of_graphs[desc])
    ///         v_fn = inputconst(typeOf(llfn), llfn)
    ///         v_result = varoftype(resulttype)
    ///         b.operations.append(
    ///             SpaceOperation("direct_call", [v_fn] + args_v, v_result))
    ///         b.closeblock(Link([v_result], graph.returnblock))
    ///         i = self.descriptions.index(desc)
    ///         links.append(Link(inputargs[1:], b, chr(i)))
    ///         links[-1].llexitcase = chr(i)
    ///     startblock.closeblock(*links)
    ///     graph.name = self._invent_dispatcher_name(row_of_graphs)
    ///     return graph
    /// ```
    pub fn make_dispatcher(
        &self,
        shape: &CallShape,
        index: usize,
        argtypes: &[LowLevelType],
        resulttype: &LowLevelType,
    ) -> Result<crate::flowspace::model::GraphRef, TyperError> {
        use crate::flowspace::model::{
            Block, BlockRefExt, FunctionGraph, Link, LinkRef, SpaceOperation, Variable,
        };
        use crate::translator::rtyper::lltypesystem::lltype;

        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.make_dispatcher: rtyper weak ref dropped")
        })?;

        // RPython `varoftype(t)` — a fresh `Variable` carrying
        // `concretetype = t`. `Variable::with_concretetype` does not
        // exist on pyre's `Variable`; build via `Variable::new()` +
        // `set_concretetype`.
        fn fresh_typed_var(t: &LowLevelType) -> Variable {
            let v = Variable::new();
            v.set_concretetype(Some(t.clone()));
            v
        }

        // upstream: `inputargs = [varoftype(t) for t in [Char] + argtypes]`.
        let mut inputargs: Vec<Variable> = Vec::with_capacity(1 + argtypes.len());
        inputargs.push(fresh_typed_var(&LowLevelType::Char));
        for t in argtypes {
            inputargs.push(fresh_typed_var(t));
        }
        let input_hl: Vec<Hlvalue> = inputargs.iter().cloned().map(Hlvalue::Variable).collect();

        // upstream: `startblock = Block(inputargs);
        //           startblock.exitswitch = inputargs[0]`.
        let startblock = Block::shared(input_hl.clone());
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(inputargs[0].clone()));

        // upstream: `graph = FunctionGraph("dispatcher", startblock,
        //                                    varoftype(resulttype))`.
        let return_var = fresh_typed_var(resulttype);
        let mut graph = FunctionGraph::with_return_var(
            "dispatcher",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );

        // upstream: `row_of_graphs = self.callfamily.calltables[shape][index]`.
        let callfamily = self.base.callfamily.as_ref().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.make_dispatcher: callfamily missing")
        })?;
        let row_of_graphs = callfamily
            .borrow()
            .calltables
            .get(shape)
            .and_then(|t| t.get(index))
            .cloned()
            .ok_or_else(|| {
                TyperError::message(
                    "SmallFunctionSetPBCRepr.make_dispatcher: calltable row missing",
                )
            })?;

        // upstream rpbc.py:459-462:
        //   descs = list(self.s_pbc.descriptions)
        //   if self.s_pbc.can_be_None:
        //       descs.insert(0, None)
        //   for desc in descs:
        //       if desc is None:
        //           continue
        //
        // The upstream `descs` is built from `self.s_pbc.descriptions`
        // (this dispatcher's own PBC), NOT `self.descriptions` (which
        // may aliase a parent's wider list under `subset_of`). The
        // can_be_None None insertion is a no-op for the dispatcher
        // since the inner loop skips None — but the exit-case index
        // `i = self.descriptions.index(desc)` still resolves against
        // self.descriptions (the +1 shift from the None prefix lives
        // there, populated by `_setup_repr`).
        let s_pbc_descs: Vec<DescEntry> = self.base.s_pbc.descriptions.values().cloned().collect();
        let self_descriptions = self.descriptions.borrow().clone();
        let mut start_links: Vec<LinkRef> = Vec::new();
        for desc in &s_pbc_descs {
            // upstream: `args_v = [varoftype(t) for t in argtypes]`.
            let args_v: Vec<Variable> = argtypes.iter().map(fresh_typed_var).collect();
            let args_v_hl: Vec<Hlvalue> = args_v.iter().cloned().map(Hlvalue::Variable).collect();

            // upstream: `b = Block(args_v)`.
            let b = Block::shared(args_v_hl.clone());

            // upstream: `llfn = self.rtyper.getcallable(row_of_graphs[desc])`.
            // `row_of_graphs` is built by `build_calltable_row`, which keys
            // each entry on `desc.rowkey()` (description.py); look it
            // up by the same key. For the FunctionDescs in a SmallFunction-
            // SetPBCRepr `rowkey()` is the desc's own identity, matching
            // upstream where `rowkey() is self`.
            let row_key = desc
                .rowkey()
                .map_err(|e| TyperError::message(e.to_string()))?;
            let target_graph = row_of_graphs.get(&row_key).ok_or_else(|| {
                TyperError::message(format!(
                    "SmallFunctionSetPBCRepr.make_dispatcher: row_of_graphs missing \
                     entry for {desc:?}"
                ))
            })?;
            let llfn = rtyper.getcallable(target_graph)?;
            // upstream: `v_fn = inputconst(typeOf(llfn), llfn)`.
            let llfn_type = LowLevelType::Ptr(Box::new(lltype::typeOf(&llfn)));
            let v_fn = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(llfn)),
                llfn_type,
            ));

            // upstream: `v_result = varoftype(resulttype)`.
            let v_result = fresh_typed_var(resulttype);

            // upstream: `b.operations.append(
            //   SpaceOperation("direct_call", [v_fn] + args_v, v_result))`.
            let mut call_args: Vec<Hlvalue> = Vec::with_capacity(1 + args_v_hl.len());
            call_args.push(v_fn);
            call_args.extend(args_v_hl.iter().cloned());
            b.borrow_mut().operations.push(SpaceOperation::new(
                "direct_call",
                call_args,
                Hlvalue::Variable(v_result.clone()),
            ));

            // upstream: `b.closeblock(Link([v_result], graph.returnblock))`.
            let link_to_return = Rc::new(RefCell::new(Link::new(
                vec![Hlvalue::Variable(v_result)],
                Some(graph.returnblock.clone()),
                None,
            )));
            b.closeblock(vec![link_to_return]);

            // upstream: `i = self.descriptions.index(desc);
            //           links.append(Link(inputargs[1:], b, chr(i)));
            //           links[-1].llexitcase = chr(i)`.
            //
            // The `chr(i)` exitcase mirrors `convert_desc`'s Char-typed
            // ByteStr-of-length-1 encoding (lltype.rs enforces
            // `LowLevelType::Char ⇔ ConstValue::ByteStr(s) if s.len()==1`).
            //
            // `i` resolves against `self.descriptions` (NOT the local
            // `s_pbc_descs`) — under `subset_of` sharing, this carries
            // the parent's wider list and so the dispatcher's exit
            // cases align with the parent's c_pointer_table indexing.
            let target_key = desc.desc_key();
            let i = self_descriptions
                .iter()
                .position(|slot| match slot {
                    Some(d) => d.desc_key() == target_key,
                    None => false,
                })
                .ok_or_else(|| {
                    TyperError::message(format!(
                        "SmallFunctionSetPBCRepr.make_dispatcher: descriptions.index({desc:?}) \
                         missing — _setup_repr did not include this desc"
                    ))
                })?;
            if i > u8::MAX as usize {
                return Err(TyperError::message(format!(
                    "SmallFunctionSetPBCRepr.make_dispatcher: index {i} exceeds \
                     Char range (256)"
                )));
            }
            let exitcase = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::byte_str(vec![i as u8]),
                LowLevelType::Char,
            ));
            let mut start_link = Link::new(
                input_hl[1..].to_vec(),
                Some(b.clone()),
                Some(exitcase.clone()),
            );
            // upstream: `links[-1].llexitcase = chr(i)`.
            start_link.llexitcase = Some(exitcase);
            start_links.push(Rc::new(RefCell::new(start_link)));
        }

        // upstream: `startblock.closeblock(*links)`.
        startblock.closeblock(start_links);

        // upstream: `graph.name = self._invent_dispatcher_name(row_of_graphs)`.
        graph.name = Self::_invent_dispatcher_name(&row_of_graphs);

        // Wrap as `GraphRef` (= `Rc<RefCell<FunctionGraph>>`) for the
        // pyre `translator.graphs` registry and for `getfunctionptr`.
        // The dispatcher graph is synthesized — no host-side Python
        // function backing — so we deliberately do not wrap it in a
        // `PyGraph` (which expects a real `GraphFunc + HostCode`).
        Ok(graph.into_ref())
    }

    /// RPython `compression_function(r_set)` (rpbc.py).
    ///
    /// ```python
    /// def compression_function(r_set):
    ///     if r_set._compression_function is None:
    ///         table = []
    ///         for i, p in enumerate(r_set.c_pointer_table.value):
    ///             table.append((chr(i), p))
    ///         last_c, last_p = table[-1]
    ///         unroll_table = unrolling_iterable(table[:-1])
    ///
    ///         def ll_compress(fnptr):
    ///             for c, p in unroll_table:
    ///                 if fnptr == p:
    ///                     return c
    ///             else:
    ///                 ll_assert(fnptr == last_p, "unexpected function pointer")
    ///                 return last_c
    ///         r_set._compression_function = ll_compress
    ///     return r_set._compression_function
    /// ```
    ///
    /// Caches the synthesized helper graph's function-pointer
    /// `Constant`. Pyre returns the `Constant` directly (RPython
    /// returns the python-level closure and lets the rtyper resolve it
    /// to a graph via `gendirectcall`); the difference is mechanical —
    /// the cache holds the same logical artifact, materialised earlier.
    pub fn compression_function(&self) -> Result<Constant, TyperError> {
        if let Some(cached) = self._compression_function.borrow().clone() {
            return Ok(cached);
        }
        let graph = self.make_compression_function()?;
        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message(
                "SmallFunctionSetPBCRepr.compression_function: rtyper weak ref dropped",
            )
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message(
                "SmallFunctionSetPBCRepr.compression_function: annotator weak ref dropped",
            )
        })?;
        // upstream: `r_set._compression_function = ll_compress` — the
        // helper closure is appended to the translator's graph set so
        // downstream codewriting picks it up like any rtyped function.
        annotator.translator.graphs.borrow_mut().push(graph.clone());
        // Materialise the function-pointer Constant. The dispatcher
        // helper uses the same pattern (see `dispatcher`).
        use crate::translator::rtyper::lltypesystem::lltype;
        let ll_ret = lltype::getfunctionptr(&graph, |v| match v {
            Hlvalue::Variable(var) => var.concretetype().ok_or_else(|| {
                TyperError::message(
                    "SmallFunctionSetPBCRepr.compression_function: ll_compress \
                     argument missing concretetype",
                )
            }),
            Hlvalue::Constant(c) => c.concretetype.clone().ok_or_else(|| {
                TyperError::message(
                    "SmallFunctionSetPBCRepr.compression_function: ll_compress \
                     constant argument missing concretetype",
                )
            }),
        })?;
        let llfn_type = LowLevelType::Ptr(Box::new(lltype::typeOf(&ll_ret)));
        let c_ret = Constant::with_concretetype(ConstValue::LLPtr(Box::new(ll_ret)), llfn_type);
        *self._compression_function.borrow_mut() = Some(c_ret.clone());
        Ok(c_ret)
    }

    /// Build the `ll_compress` helper graph synthesized by
    /// [`SmallFunctionSetPBCRepr::compression_function`]. Mirrors
    /// upstream's `unrolling_iterable(table[:-1])` linear cascade —
    /// each non-last entry produces an `(fnptr == p)` test that
    /// branches to a Char-returning block; the final block returns
    /// `chr(last)` unconditionally (upstream's `ll_assert(fnptr ==
    /// last_p)` is debug-only, no equivalent OpKind in pyre).
    fn make_compression_function(&self) -> Result<crate::flowspace::model::GraphRef, TyperError> {
        use crate::flowspace::model::{
            Block, BlockRefExt, FunctionGraph, Link, LinkRef, SpaceOperation, Variable,
        };

        // Pull the c_pointer_table entries — upstream reads
        // `r_set.c_pointer_table.value` (the raw `_array`).
        let c_pointer_table = self.c_pointer_table.borrow().clone().ok_or_else(|| {
            TyperError::message(
                "SmallFunctionSetPBCRepr.make_compression_function: \
                 c_pointer_table not populated — _setup_repr must have run",
            )
        })?;
        let array_value: Vec<Constant> = {
            let ConstValue::LLPtr(ptr) = &c_pointer_table.value else {
                return Err(TyperError::message(
                    "SmallFunctionSetPBCRepr.make_compression_function: \
                     c_pointer_table is not LLPtr",
                ));
            };
            // Pull each entry as a typed Constant (Ptr lowleveltype
            // matching pointer_repr).
            use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, LowLevelValue};
            let item_lltype = self.pointer_repr.lowleveltype().clone();
            let _ptr_obj::Array(array) = ptr._obj().map_err(|e| {
                TyperError::message(format!(
                    "SmallFunctionSetPBCRepr.make_compression_function: \
                     c_pointer_table ptr resolution failed: {e:?}"
                ))
            })?
            else {
                return Err(TyperError::message(
                    "SmallFunctionSetPBCRepr.make_compression_function: \
                     c_pointer_table ptr does not point at Array",
                ));
            };
            let mut entries: Vec<Constant> = Vec::new();
            let n = array.getbounds().1;
            for i in 0..n {
                let entry = array.getitem(i).ok_or_else(|| {
                    TyperError::message(format!(
                        "SmallFunctionSetPBCRepr.make_compression_function: \
                         c_pointer_table entry {i} missing"
                    ))
                })?;
                let LowLevelValue::Ptr(entry_ptr) = entry else {
                    return Err(TyperError::message(format!(
                        "SmallFunctionSetPBCRepr.make_compression_function: \
                         c_pointer_table entry {i} is not Ptr"
                    )));
                };
                entries.push(Constant::with_concretetype(
                    ConstValue::LLPtr(entry_ptr),
                    item_lltype.clone(),
                ));
            }
            entries
        };

        if array_value.is_empty() {
            return Err(TyperError::message(
                "SmallFunctionSetPBCRepr.make_compression_function: \
                 c_pointer_table is empty — cannot synthesize ll_compress",
            ));
        }

        fn fresh_typed_var(t: &LowLevelType) -> Variable {
            let v = Variable::new();
            v.set_concretetype(Some(t.clone()));
            v
        }

        let item_lltype = self.pointer_repr.lowleveltype().clone();

        // upstream `def ll_compress(fnptr): ...` — single Ptr-typed input.
        let arg_fnptr = fresh_typed_var(&item_lltype);
        let return_var = fresh_typed_var(&LowLevelType::Char);

        let mut graph = FunctionGraph::with_return_var(
            "ll_compress",
            // placeholder startblock; we replace it below.
            Block::shared(vec![Hlvalue::Variable(arg_fnptr.clone())]),
            Hlvalue::Variable(return_var),
        );

        // Strategy: build N test blocks chained by False branch, each
        // True branch returns chr(i). The Nth (last) block has no test
        // — it just returns chr(N-1). This is the structural parity
        // expansion of upstream's `unrolling_iterable(table[:-1])` +
        // `ll_assert(fnptr == last_p)` final fallthrough.
        //
        // Build the chain back-to-front so each test block can name
        // its False successor.
        let n = array_value.len();

        // Final block: just close to returnblock with chr(n-1).
        let final_idx = n - 1;
        if final_idx > u8::MAX as usize {
            return Err(TyperError::message(format!(
                "SmallFunctionSetPBCRepr.make_compression_function: \
                 last index {final_idx} exceeds Char range (256)"
            )));
        }
        let final_chr = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::byte_str(vec![final_idx as u8]),
            LowLevelType::Char,
        ));
        // The final block needs an inputarg matching what the previous
        // block's False-link will pass — we pass `fnptr` along (even
        // though it's unused) for uniformity.
        let final_input = fresh_typed_var(&item_lltype);
        let final_block = Block::shared(vec![Hlvalue::Variable(final_input)]);
        final_block.closeblock(vec![Rc::new(RefCell::new(Link::new(
            vec![final_chr],
            Some(graph.returnblock.clone()),
            None,
        )))]);

        // Walk i = N-2 .. 0, building each test block whose False arm
        // points at the next block in the chain.
        let mut next_block = final_block;
        for i in (0..n - 1).rev() {
            let idx_chr = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::byte_str(vec![i as u8]),
                LowLevelType::Char,
            ));

            // True-arm block: just return chr(i).
            let true_input = fresh_typed_var(&item_lltype);
            let true_block = Block::shared(vec![Hlvalue::Variable(true_input)]);
            true_block.closeblock(vec![Rc::new(RefCell::new(Link::new(
                vec![idx_chr],
                Some(graph.returnblock.clone()),
                None,
            )))]);

            // Test block:
            //   v_eq = ptr_eq(fnptr, p_i); exitswitch = v_eq
            //   True -> true_block; False -> next_block
            let test_input = fresh_typed_var(&item_lltype);
            let test_block = Block::shared(vec![Hlvalue::Variable(test_input.clone())]);
            let v_eq = fresh_typed_var(&LowLevelType::Bool);
            test_block.borrow_mut().operations.push(SpaceOperation::new(
                "ptr_eq",
                vec![
                    Hlvalue::Variable(test_input.clone()),
                    Hlvalue::Constant(array_value[i].clone()),
                ],
                Hlvalue::Variable(v_eq.clone()),
            ));
            test_block.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_eq));

            let true_case = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            ));
            let false_case = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            ));

            let mut true_link = Link::new(
                vec![Hlvalue::Variable(test_input.clone())],
                Some(true_block),
                Some(true_case.clone()),
            );
            true_link.llexitcase = Some(true_case);
            let mut false_link = Link::new(
                vec![Hlvalue::Variable(test_input)],
                Some(next_block.clone()),
                Some(false_case.clone()),
            );
            false_link.llexitcase = Some(false_case);
            test_block.closeblock(vec![
                Rc::new(RefCell::new(true_link)),
                Rc::new(RefCell::new(false_link)),
            ]);

            next_block = test_block;
        }

        // Wire startblock = first test block via a Link that passes
        // `fnptr` from the graph's inputargs in.
        let startblock_inputs: Vec<Hlvalue> = vec![Hlvalue::Variable(arg_fnptr.clone())];
        let startblock = Block::shared(startblock_inputs);
        let entry_link: LinkRef = Rc::new(RefCell::new(Link::new(
            vec![Hlvalue::Variable(arg_fnptr.clone())],
            Some(next_block),
            None,
        )));
        startblock.closeblock(vec![entry_link]);
        graph.startblock = startblock;

        Ok(graph.into_ref())
    }

    /// RPython `SmallFunctionSetPBCRepr.convert_desc(self, funcdesc)`
    /// (rpbc.py): `return chr(self.descriptions.index(funcdesc))`.
    ///
    /// `self.descriptions` is populated by `_setup_repr`; calling
    /// `convert_desc` before setup runs surfaces a structured
    /// TyperError so the missing prerequisite is explicit.
    pub fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        let descriptions = self.descriptions.borrow();
        if descriptions.is_empty() {
            return Err(TyperError::message(
                "SmallFunctionSetPBCRepr.convert_desc: descriptions not populated — \
                 _setup_repr must have run",
            ));
        }
        let target_key = desc.desc_key();
        let idx = descriptions
            .iter()
            .position(|slot| match slot {
                Some(d) => d.desc_key() == target_key,
                None => false,
            })
            .ok_or_else(|| {
                TyperError::message(format!(
                    "SmallFunctionSetPBCRepr.convert_desc: {desc:?} not in descriptions"
                ))
            })?;
        if idx > u8::MAX as usize {
            return Err(TyperError::message(format!(
                "SmallFunctionSetPBCRepr.convert_desc: index {idx} exceeds Char range \
                 (256) — small_cand invariant violated"
            )));
        }
        // upstream `chr(i)` — Char-typed Constant. Pyre's lltype.rs
        // enforces `LowLevelType::Char ⇔ ConstValue::ByteStr(s) if
        // s.len()==1`, so `chr` materialises as a single-byte ByteStr.
        Ok(Constant::with_concretetype(
            ConstValue::byte_str(vec![idx as u8]),
            LowLevelType::Char,
        ))
    }

    /// RPython `SmallFunctionSetPBCRepr.call(self, hop)` (rpbc.py):
    ///
    /// ```python
    /// def call(self, hop):
    ///     bk = self.rtyper.annotator.bookkeeper
    ///     args = hop.spaceop.build_args(hop.args_s[1:])
    ///     s_pbc = hop.args_s[0]   # possibly more precise than self.s_pbc
    ///     descs = list(s_pbc.descriptions)
    ///     shape, index = self.callfamily.find_row(bk, descs, args, hop.spaceop)
    ///     row_of_graphs = self.callfamily.calltables[shape][index]
    ///     anygraph = row_of_graphs.itervalues().next()  # pick any witness
    ///     vlist = [hop.inputarg(self, arg=0)]
    ///     vlist += callparse.callparse(self.rtyper, anygraph, hop)
    ///     rresult = callparse.getrresult(self.rtyper, anygraph)
    ///     hop.exception_is_here()
    ///     v_dispatcher = self.dispatcher(shape, index,
    ///             [v.concretetype for v in vlist[1:]], rresult.lowleveltype)
    ///     v_result = hop.genop('direct_call', [v_dispatcher] + vlist,
    ///                          resulttype=rresult)
    ///     return hop.llops.convertvar(v_result, rresult, hop.r_result)
    /// ```
    pub fn call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> Result<Option<Hlvalue>, TyperError> {
        use crate::annotator::bookkeeper::build_args_for_op;
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::callparse::{self, RResult};
        use crate::translator::rtyper::rtyper::GenopResult;

        // upstream: `bk = self.rtyper.annotator.bookkeeper`.
        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.call: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.call: annotator weak ref dropped")
        })?;
        let bookkeeper = annotator.bookkeeper.clone();

        // upstream: `args = hop.spaceop.build_args(hop.args_s[1:])`.
        let args_s_full = hop.args_s.borrow().clone();
        if args_s_full.is_empty() {
            return Err(TyperError::message(
                "SmallFunctionSetPBCRepr.call: hop.args_s must contain the receiver",
            ));
        }
        // `hop.args_s` carries concrete `SomeValue`s by the time the
        // rtyper runs (annotator's fixpoint has converged), so wrap
        // each entry as `Some(_)` for the post-Item-#100
        // `build_args_for_op` signature.
        let args_s_opt: Vec<Option<SomeValue>> =
            args_s_full[1..].iter().cloned().map(Some).collect();
        let args = build_args_for_op(&hop.spaceop.opname, &args_s_opt)
            .map_err(|e| TyperError::message(e.to_string()))?;

        // upstream: `s_pbc = hop.args_s[0]; descs = list(s_pbc.descriptions);
        //           shape, index = self.callfamily.find_row(bk, descs, args,
        //                                                    hop.spaceop);
        //           row_of_graphs = self.callfamily.calltables[shape][index];
        //           anygraph = row_of_graphs.itervalues().next()`.
        let s_pbc = match args_s_full.first().cloned() {
            Some(SomeValue::PBC(pbc)) => pbc,
            other => {
                return Err(TyperError::message(format!(
                    "SmallFunctionSetPBCRepr.call: hop.args_s[0] is not a SomePBC: {other:?}"
                )));
            }
        };
        let callfamily = self.base.callfamily.as_ref().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.call: callfamily not available")
        })?;
        let row = select_call_family_row(&bookkeeper, callfamily, &s_pbc, &args, None)
            .map_err(|e| TyperError::message(e.to_string()))?;

        // upstream: `vlist = [hop.inputarg(self, arg=0)]`.
        let mut vlist: Vec<Hlvalue> = vec![hop.inputarg(self as &dyn Repr, 0)?];

        // upstream: `vlist += callparse.callparse(self.rtyper, anygraph, hop)`.
        vlist.extend(callparse::callparse(&rtyper, &row.anygraph, hop, None)?);

        // upstream: `rresult = callparse.getrresult(self.rtyper, anygraph)`.
        let rresult = callparse::getrresult(&rtyper, &row.anygraph)?;

        // upstream: `hop.exception_is_here()`.
        hop.exception_is_here()?;

        // upstream: `v_dispatcher = self.dispatcher(shape, index,
        //           [v.concretetype for v in vlist[1:]], rresult.lowleveltype)`.
        let argtypes: Vec<LowLevelType> = vlist[1..]
            .iter()
            .map(|v| match v {
                Hlvalue::Variable(var) => var.concretetype().ok_or_else(|| {
                    TyperError::message(
                        "SmallFunctionSetPBCRepr.call: vlist Variable missing concretetype",
                    )
                }),
                Hlvalue::Constant(c) => c.concretetype.clone().ok_or_else(|| {
                    TyperError::message(
                        "SmallFunctionSetPBCRepr.call: vlist Constant missing concretetype",
                    )
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result_lltype = rresult.lowleveltype();
        let v_dispatcher = self.dispatcher(&row.shape, row.index, &argtypes, &result_lltype)?;

        // upstream: `v_result = hop.genop('direct_call', [v_dispatcher] + vlist,
        //                                  resulttype=rresult)`.
        let mut call_args: Vec<Hlvalue> = Vec::with_capacity(1 + vlist.len());
        call_args.push(Hlvalue::Constant(v_dispatcher));
        call_args.extend(vlist);
        let v_result = match &rresult {
            RResult::Repr(r) => hop.genop("direct_call", call_args, GenopResult::Repr(r.clone())),
            RResult::Void => hop.genop("direct_call", call_args, GenopResult::Void),
        };

        // upstream: `return hop.llops.convertvar(v_result, rresult, hop.r_result)`.
        //
        // The Void return arm is a no-op (Constant Void); convertvar
        // collapses to identity. The non-Void arm runs `convertvar` to
        // align the dispatcher's return-type Repr with `hop.r_result`.
        let Some(v) = v_result else {
            return Ok(None);
        };
        let r_result = hop.r_result.borrow().clone();
        match (&rresult, r_result) {
            (RResult::Repr(r_from), Some(r_to)) => {
                let mut llops = hop.llops.borrow_mut();
                llops
                    .convertvar(v, r_from.as_ref(), r_to.as_ref())
                    .map(Some)
            }
            // Void return or missing r_result — pass through.
            _ => Ok(Some(v)),
        }
    }
}

impl Repr for SmallFunctionSetPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "SmallFunctionSetPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::SmallFunctionSetPBCRepr
    }

    /// RPython `FunctionReprBase.s_pbc` (rpbc.py) — exposed via the
    /// trait so `MethodsPBCRepr.redispatch_call` can supply
    /// `subset_of=r_func.s_pbc` when the methodname maps to a
    /// SmallFunctionSetPBCRepr.
    fn pbc_s_pbc(&self) -> Option<&SomePBC> {
        Some(&self.base.s_pbc)
    }

    fn get_s_signatures(
        &self,
        shape: &crate::flowspace::argument::CallShape,
    ) -> Result<
        Vec<(
            Vec<crate::annotator::model::SomeValue>,
            crate::annotator::model::SomeValue,
        )>,
        TyperError,
    > {
        self.base.get_s_signatures(shape)
    }

    /// RPython `SmallFunctionSetPBCRepr._setup_repr(self)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def _setup_repr(self):
    ///     if self.s_pbc.subset_of:
    ///         assert self.s_pbc.can_be_None == self.s_pbc.subset_of.can_be_None
    ///         r = self.rtyper.getrepr(self.s_pbc.subset_of)
    ///         if r is not self:
    ///             r.setup()
    ///             self.descriptions = r.descriptions
    ///             self.c_pointer_table = r.c_pointer_table
    ///             return
    ///     self.descriptions = list(self.s_pbc.descriptions)
    ///     if self.s_pbc.can_be_None:
    ///         self.descriptions.insert(0, None)
    ///     POINTER_TABLE = Array(self.pointer_repr.lowleveltype,
    ///                           hints={'nolength': True, 'immutable': True,
    ///                                  'static_immutable': True})
    ///     pointer_table = malloc(POINTER_TABLE, len(self.descriptions),
    ///                            immortal=True)
    ///     for i, desc in enumerate(self.descriptions):
    ///         if desc is not None:
    ///             pointer_table[i] = self.pointer_repr.convert_desc(desc)
    ///         else:
    ///             pointer_table[i] = self.pointer_repr.convert_const(None)
    ///     self.c_pointer_table = inputconst(Ptr(POINTER_TABLE), pointer_table)
    /// ```
    fn _setup_repr(&self) -> Result<(), TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype::{
            self as lltype, Array, LowLevelValue, MallocFlavor,
        };

        // upstream rpbc.py:405-412 — subset_of share branch.
        if let Some(subset_of) = self.base.s_pbc.subset_of.as_deref() {
            // upstream `assert self.s_pbc.can_be_None ==
            //           self.s_pbc.subset_of.can_be_None`.
            if self.base.s_pbc.can_be_none != subset_of.can_be_none {
                return Err(TyperError::message(
                    "SmallFunctionSetPBCRepr._setup_repr: can_be_None mismatch \
                     between s_pbc and s_pbc.subset_of",
                ));
            }
            // upstream `r = self.rtyper.getrepr(self.s_pbc.subset_of)`.
            let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
                TyperError::message("SmallFunctionSetPBCRepr._setup_repr: rtyper weak ref dropped")
            })?;
            let r = rtyper.getrepr(&crate::annotator::model::SomeValue::PBC(subset_of.clone()))?;
            // upstream `if r is not self`: pointer-identity comparison
            // through the trait-object box. `r` is `Arc<dyn Repr>`; if
            // its inner pointer matches `self`'s address, the pbc maps
            // back to the same SmallFunctionSetPBCRepr (rare degenerate
            // case where subset_of points at the original PBC).
            let r_self_addr = self as *const SmallFunctionSetPBCRepr as *const () as usize;
            let r_addr = Arc::as_ptr(&r) as *const () as usize;
            if r_self_addr != r_addr {
                Repr::setup(r.as_ref())?;
                let r_set = (r.as_ref() as &dyn std::any::Any)
                    .downcast_ref::<SmallFunctionSetPBCRepr>()
                    .ok_or_else(|| {
                        TyperError::message(
                            "SmallFunctionSetPBCRepr._setup_repr: subset_of repr \
                             is not a SmallFunctionSetPBCRepr",
                        )
                    })?;
                // upstream `self.descriptions = r.descriptions;
                //           self.c_pointer_table = r.c_pointer_table; return`.
                *self.descriptions.borrow_mut() = r_set.descriptions.borrow().clone();
                *self.c_pointer_table.borrow_mut() = r_set.c_pointer_table.borrow().clone();
                return Ok(());
            }
        }

        // upstream rpbc.py — `self.descriptions = list(self.s_pbc.descriptions)`.
        let mut descriptions: Vec<Option<DescEntry>> = self
            .base
            .s_pbc
            .descriptions
            .values()
            .cloned()
            .map(Some)
            .collect();
        // upstream rpbc.py:414-415 — `if self.s_pbc.can_be_None:
        //                                self.descriptions.insert(0, None)`.
        if self.base.s_pbc.can_be_none {
            descriptions.insert(0, None);
        }
        *self.descriptions.borrow_mut() = descriptions.clone();

        // `SmallFunctionSetPBCRepr._setup_repr` defines `POINTER_TABLE = Array(
        //   self.pointer_repr.lowleveltype, hints={...})`.
        let item_type = self.pointer_repr.lowleveltype().clone();
        let array_type = Array::with_hints(
            item_type,
            vec![
                ("nolength".to_string(), ConstValue::Bool(true)),
                ("immutable".to_string(), ConstValue::Bool(true)),
                ("static_immutable".to_string(), ConstValue::Bool(true)),
            ],
        );
        let array_lltype = LowLevelType::Array(Box::new(array_type));

        // upstream rpbc.py:419-420 — `pointer_table =
        //   malloc(POINTER_TABLE, len(self.descriptions), immortal=True)`.
        // `flavor='gc'` is upstream's malloc default; `immortal=True`
        // means the allocation is solid and never collected.
        let mut pointer_table = lltype::malloc(
            array_lltype.clone(),
            Some(descriptions.len()),
            MallocFlavor::Gc,
            true,
        )
        .map_err(TyperError::message)?;

        // upstream rpbc.py:421-425 —
        //   for i, desc in enumerate(self.descriptions):
        //       if desc is not None:
        //           pointer_table[i] = self.pointer_repr.convert_desc(desc)
        //       else:
        //           pointer_table[i] = self.pointer_repr.convert_const(None)
        for (i, slot) in descriptions.iter().enumerate() {
            let entry_const = match slot {
                Some(desc) => (self.pointer_repr.as_ref() as &dyn Repr).convert_desc(desc)?,
                None => {
                    (self.pointer_repr.as_ref() as &dyn Repr).convert_const(&ConstValue::None)?
                }
            };
            // The Constant produced by FunctionsPBCRepr.{convert_desc,
            // convert_const(None)} carries `ConstValue::LLPtr(_ptr)`.
            // `_array.setitem` consumes a `LowLevelValue`, so unwrap
            // the LLPtr and re-wrap as `LowLevelValue::Ptr`.
            let ConstValue::LLPtr(ptr) = entry_const.value else {
                return Err(TyperError::message(format!(
                    "SmallFunctionSetPBCRepr._setup_repr: pointer_repr produced \
                     non-LLPtr entry at index {i}: {:?}",
                    entry_const.value
                )));
            };
            pointer_table
                .setitem(i, LowLevelValue::Ptr(ptr))
                .map_err(TyperError::message)?;
        }

        // upstream rpbc.py:426 — `self.c_pointer_table =
        //   inputconst(Ptr(POINTER_TABLE), pointer_table)`. The Ptr
        //   target wraps the just-built `array_lltype` so downstream
        //   `getarrayitem` resolves the element type via `Ptr(TO=Array)`.
        let ptr_type = LowLevelType::Ptr(Box::new(
            lltype::Ptr::from_container_type(array_lltype.clone()).map_err(TyperError::message)?,
        ));
        *self.c_pointer_table.borrow_mut() = Some(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(pointer_table)),
            ptr_type,
        ));
        Ok(())
    }

    /// RPython `SmallFunctionSetPBCRepr.convert_desc(self, funcdesc)`
    /// (rpbc.py) — thin Repr-trait forwarder.
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        SmallFunctionSetPBCRepr::convert_desc(self, desc)
    }

    /// RPython `SmallFunctionSetPBCRepr.convert_const(self, value)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if isinstance(value, types.MethodType) and value.im_self is None:
    ///         value = value.im_func   # unbound method -> bare function
    ///     if value is None:
    ///         assert self.descriptions[0] is None
    ///         return chr(0)
    ///     funcdesc = self.rtyper.annotator.bookkeeper.getdesc(value)
    ///     return self.convert_desc(funcdesc)
    /// ```
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        // upstream: `if value is None: assert descriptions[0] is None;
        //              return chr(0)`.
        if matches!(value, ConstValue::None) {
            let descriptions = self.descriptions.borrow();
            if descriptions.first().and_then(|s| s.as_ref()).is_some() {
                return Err(TyperError::message(
                    "SmallFunctionSetPBCRepr.convert_const(None): descriptions[0] \
                     is not None — can_be_None invariant violated",
                ));
            }
            return Ok(Constant::with_concretetype(
                ConstValue::byte_str(vec![0u8]),
                LowLevelType::Char,
            ));
        }
        // upstream: `if isinstance(value, types.MethodType) and
        //              value.im_self is None: value = value.im_func`.
        //
        // The unbound-method case (Python 2 holdover) does not occur in
        // pyre's lattice — `HostObject::BoundMethod` always carries a
        // non-None `self_obj`, so the upstream condition `im_self is
        // None` is never true. Route bound methods through
        // `bookkeeper.getdesc` directly: it handles MethodDesc
        // resolution, and pre-emptively rewriting to the underlying
        // function would be over-porting (a bound method desc is not a
        // bare-function desc).
        let host_obj = match value {
            ConstValue::HostObject(h) => h.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "SmallFunctionSetPBCRepr.convert_const: expected HostObject or \
                     None, got {other:?}"
                )));
            }
        };
        // upstream: `funcdesc = bk.getdesc(value); return self.convert_desc(funcdesc)`.
        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.convert_const: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.convert_const: annotator weak ref dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(&host_obj)
            .map_err(|e| TyperError::message(e.to_string()))?;
        SmallFunctionSetPBCRepr::convert_desc(self, &desc)
    }

    /// RPython `SmallFunctionSetPBCRepr.special_uninitialized_value(self)`
    /// (rpbc.py): `return chr(0xFF)`.
    fn special_uninitialized_value(&self) -> Option<ConstValue> {
        Some(ConstValue::byte_str(vec![0xFFu8]))
    }

    /// RPython `SmallFunctionSetPBCRepr.rtype_bool(self, hop)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def rtype_bool(self, hop):
    ///     if not self.s_pbc.can_be_None:
    ///         return inputconst(Bool, True)
    ///     else:
    ///         v1, = hop.inputargs(self)
    ///         return hop.genop('char_ne', [v1, inputconst(Char, '\000')],
    ///                      resulttype=Bool)
    /// ```
    fn rtype_bool(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp};

        // upstream: `if not self.s_pbc.can_be_None: return inputconst(Bool, True)`.
        if !self.base.s_pbc.can_be_none {
            return Ok(Some(Hlvalue::Constant(HighLevelOp::inputconst(
                ConvertedTo::LowLevelType(&LowLevelType::Bool),
                &ConstValue::Bool(true),
            )?)));
        }
        // upstream: `v1, = hop.inputargs(self);
        //           return hop.genop('char_ne', [v1, inputconst(Char, '\000')],
        //                            resulttype=Bool)`.
        let v_args = hop.inputargs(vec![ConvertedTo::Repr(self as &dyn Repr)])?;
        let v1 = v_args
            .into_iter()
            .next()
            .expect("inputargs returns 1 value for 1 ConvertedTo");
        let c_zero = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::byte_str(vec![0u8]),
            LowLevelType::Char,
        ));
        Ok(hop.genop(
            "char_ne",
            vec![v1, c_zero],
            GenopResult::LLType(LowLevelType::Bool),
        ))
    }

    /// RPython `FunctionReprBase.get_r_implfunc(self)` (rpbc.py) —
    /// inherited via `class SmallFunctionSetPBCRepr(FunctionReprBase)`:
    /// `return self, 0`.
    fn get_r_implfunc(&self) -> Result<(Arc<dyn Repr>, usize), TyperError> {
        let arc = self.self_weak.upgrade().ok_or_else(|| {
            TyperError::message("SmallFunctionSetPBCRepr.get_r_implfunc: repr Arc was dropped")
        })?;
        Ok((arc, 0))
    }

    /// RPython `SmallFunctionSetPBCRepr.rtype_simple_call(self, hop)`
    /// inherited via `FunctionReprBase`: `return self.call(hop)`.
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }

    /// `FunctionsPBCRepr` does not override `rtype_call_args` either —
    /// same `call(hop)` body for both upstream call shapes.
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }
}

/// RPython `compression_function(r_set)` (rpbc.py).
pub fn compression_function(r_set: &SmallFunctionSetPBCRepr) -> Result<Constant, TyperError> {
    r_set.compression_function()
}

/// RPython `pairtype(FunctionRepr, SmallFunctionSetPBCRepr)
///                  .convert_from_to((r_ptr, r_set), v, llops)`
/// (rpbc.py):
///
/// ```python
/// def convert_from_to((r_ptr, r_set), v, llops):
///     desc, = r_ptr.s_pbc.descriptions
///     return inputconst(Char, r_set.convert_desc(desc))
/// ```
///
/// FunctionRepr always carries a single-FunctionDesc PBC (rpbc.py).
/// The Char-typed Constant produced by `r_set.convert_desc(desc)` is
/// returned wrapped as an `Hlvalue::Constant` — `same_lowleveltype_*`
/// is *not* used here because the Function source value is discarded.
pub(super) fn pair_function_repr_small_function_set_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    _v: &Hlvalue,
    _llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let r_ptr = (r_from as &dyn std::any::Any)
        .downcast_ref::<FunctionRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_function_repr_small_function_set_convert_from_to: \
                 r_from is not FunctionRepr",
            )
        })?;
    let r_set = (r_to as &dyn std::any::Any)
        .downcast_ref::<SmallFunctionSetPBCRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_function_repr_small_function_set_convert_from_to: \
                 r_to is not SmallFunctionSetPBCRepr",
            )
        })?;
    // upstream: `desc, = r_ptr.s_pbc.descriptions` — single-desc PBC
    // unpacking. FunctionRepr's `__init__` enforces `len == 1` so the
    // unpack is safe.
    let descriptions = &r_ptr.base.s_pbc.descriptions;
    if descriptions.len() != 1 {
        return Err(TyperError::message(format!(
            "pair_function_repr_small_function_set_convert_from_to: \
             FunctionRepr.s_pbc must have exactly 1 description, got {}",
            descriptions.len()
        )));
    }
    let desc = descriptions.values().next().expect("len 1 checked");
    // upstream: `return inputconst(Char, r_set.convert_desc(desc))`.
    let char_const = SmallFunctionSetPBCRepr::convert_desc(r_set, desc)?;
    Ok(Some(Hlvalue::Constant(char_const)))
}

/// RPython `pairtype(SmallFunctionSetPBCRepr, FunctionsPBCRepr)
///                  .convert_from_to((r_set, r_ptr), v, llops)`
/// (rpbc.py):
///
/// ```python
/// def convert_from_to((r_set, r_ptr), v, llops):
///     assert v.concretetype is Char
///     v_int = llops.genop('cast_char_to_int', [v], resulttype=Signed)
///     return llops.genop('getarrayitem', [r_set.c_pointer_table, v_int],
///                         resulttype=r_ptr.lowleveltype)
/// ```
///
/// Cast the source `Char` index to `Signed`, then index the
/// pre-built `c_pointer_table` to retrieve the function pointer at
/// the matching `r_ptr.lowleveltype`. `r_set._setup_repr` populates
/// `c_pointer_table`; if it has not yet run we surface a structured
/// TyperError.
///
/// RPython `pairtype(FunctionRepr, FunctionsPBCRepr)
///                  .convert_from_to((r_fpbc1, r_fpbc2), v, llops)`
/// (rpbc.py):
///
/// ```python
/// def convert_from_to((r_fpbc1, r_fpbc2), v, llops):
///     return inputconst(r_fpbc2, r_fpbc1.s_pbc.const)
/// ```
///
/// `r_fpbc1.s_pbc.const` is the host-side `pyobj` of the single
/// FunctionDesc carried by FunctionRepr (rpbc.py enforces
/// `len(descriptions) == 1` and `not can_be_None` so the SomePBC has
/// a `const_box`). `inputconst(r_fpbc2, pyobj)` routes through
/// `r_fpbc2.convert_const(pyobj)` to materialise the Ptr-typed
/// `Constant` for the FunctionsPBCRepr lowleveltype.
pub(super) fn pair_function_repr_functions_pbc_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    _v: &Hlvalue,
    _llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let r_fpbc1 = (r_from as &dyn std::any::Any)
        .downcast_ref::<FunctionRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_function_repr_functions_pbc_convert_from_to: r_from is not \
                 FunctionRepr",
            )
        })?;
    // upstream: `r_fpbc1.s_pbc.const` — pull the SomePBC's const_box
    // value (a host pyobj wrapped as a `Constant`).
    let s_pbc_const_value = r_fpbc1
        .base
        .s_pbc
        .base
        .const_box
        .as_ref()
        .map(|c| c.value.clone())
        .ok_or_else(|| {
            TyperError::message(
                "pair_function_repr_functions_pbc_convert_from_to: FunctionRepr \
                 s_pbc has no const — single-FunctionDesc !can_be_None invariant \
                 violated",
            )
        })?;
    // upstream: `return inputconst(r_fpbc2, r_fpbc1.s_pbc.const)`. Pyre's
    // `convert_const` is the dispatcher — for FunctionsPBCRepr that
    // handles the host-pyobj → Ptr-typed Constant conversion.
    let result = r_to.convert_const(&s_pbc_const_value)?;
    Ok(Some(Hlvalue::Constant(result)))
}

/// RPython `pairtype(MultipleFrozenPBCRepr, FunctionRepr)
///                  .convert_from_to((r_frozen1, r_fn2), v, llops)`
/// (rpbc.py):
///
/// ```python
/// def convert_from_to((r_frozen1, r_fn2), v, llops):
///     if r_fn2.lowleveltype is Void:
///         value = r_fn2.s_pbc.const
///         return Constant(value, Void)
///     return NotImplemented
/// ```
///
/// The target is a constant function, so its repr carries no runtime
/// value and the source pointer is dropped: the conversion materialises
/// the target's own constant at `Void`. Unlike the
/// `pairtype(FunctionsPBCRepr, FunctionRepr)` sibling — which yields
/// `inputconst(Void, None)` — upstream puts `s_pbc.const` in the
/// `Constant` here, and builds it directly rather than through
/// `FunctionRepr.convert_const`, whose answer is always `None`.
pub(super) fn pair_multiple_frozen_pbc_function_repr_convert_from_to(
    r_to: &dyn Repr,
) -> Result<Option<Hlvalue>, TyperError> {
    let r_fn2 = (r_to as &dyn std::any::Any)
        .downcast_ref::<FunctionRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_multiple_frozen_pbc_function_repr_convert_from_to: r_to is not \
                 FunctionRepr",
            )
        })?;
    // upstream: `if r_fn2.lowleveltype is Void:` … `return NotImplemented`.
    // `FunctionRepr.lowleveltype` is the class attribute `Void`
    // (rpbc.py), so the guard normally holds; a repr that answers
    // otherwise falls through to the next pair_mro entry.
    if !matches!(r_fn2.lowleveltype(), LowLevelType::Void) {
        return Ok(None);
    }
    // upstream: `value = r_fn2.s_pbc.const`. FunctionRepr is the repr for
    // a constant function, so the SomePBC carries a const_box.
    let value = r_fn2
        .base
        .s_pbc
        .base
        .const_box
        .as_ref()
        .map(|c| c.value.clone())
        .ok_or_else(|| {
            TyperError::message(
                "pair_multiple_frozen_pbc_function_repr_convert_from_to: FunctionRepr \
                 s_pbc has no const — the constant-function invariant is violated",
            )
        })?;
    // upstream: `return Constant(value, Void)`.
    let c = crate::translator::rtyper::rmodel::inputconst_from_lltype(&LowLevelType::Void, &value)?;
    Ok(Some(Hlvalue::Constant(c)))
}

pub(super) fn pair_small_function_set_functions_pbc_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    use crate::translator::rtyper::rtyper::GenopResult;

    let r_set = (r_from as &dyn std::any::Any)
        .downcast_ref::<SmallFunctionSetPBCRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_small_function_set_functions_pbc_convert_from_to: \
                 r_from is not SmallFunctionSetPBCRepr",
            )
        })?;
    let r_ptr = (r_to as &dyn std::any::Any)
        .downcast_ref::<FunctionsPBCRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_small_function_set_functions_pbc_convert_from_to: \
                 r_to is not FunctionsPBCRepr",
            )
        })?;

    let c_pointer_table = r_set.c_pointer_table.borrow().clone().ok_or_else(|| {
        TyperError::message(
            "pair_small_function_set_functions_pbc_convert_from_to: \
                 r_set.c_pointer_table not populated — _setup_repr must have run",
        )
    })?;

    // upstream: `v_int = llops.genop('cast_char_to_int', [v],
    //                                resulttype=Signed)`.
    let v_int = llops
        .genop(
            "cast_char_to_int",
            vec![v.clone()],
            GenopResult::LLType(LowLevelType::Signed),
        )
        .expect("cast_char_to_int with Signed result yields a Variable");

    // upstream: `return llops.genop('getarrayitem',
    //                                [r_set.c_pointer_table, v_int],
    //                                resulttype=r_ptr.lowleveltype)`.
    let v_result = llops
        .genop(
            "getarrayitem",
            vec![Hlvalue::Constant(c_pointer_table), Hlvalue::Variable(v_int)],
            GenopResult::LLType(r_ptr.lowleveltype().clone()),
        )
        .expect("getarrayitem with non-Void result yields a Variable");
    Ok(Some(Hlvalue::Variable(v_result)))
}

/// RPython `conversion_table(r_from, r_to)` (rpbc.py) +
/// `pairtype(SmallFunctionSetPBCRepr,
/// SmallFunctionSetPBCRepr).convert_from_to` (rpbc.py).
///
/// ```python
/// def conversion_table(r_from, r_to):
///     if r_to in r_from._conversion_tables:
///         return r_from._conversion_tables[r_to]
///     else:
///         t = malloc(Array(Char, hints={'nolength': True, 'immutable': True,
///                                       'static_immutable': True}),
///                    len(r_from.descriptions), immortal=True)
///         l = []
///         for i, d in enumerate(r_from.descriptions):
///             if d in r_to.descriptions:
///                 j = r_to.descriptions.index(d)
///                 l.append(j)
///                 t[i] = chr(j)
///             else:
///                 l.append(None)
///         if l == range(len(r_from.descriptions)):
///             r = None
///         else:
///             r = inputconst(typeOf(t), t)
///         r_from._conversion_tables[r_to] = r
///         return r
///
/// class __extend__(pairtype(SmallFunctionSetPBCRepr, SmallFunctionSetPBCRepr)):
///     def convert_from_to((r_from, r_to), v, llops):
///         c_table = conversion_table(r_from, r_to)
///         if c_table:
///             assert v.concretetype is Char
///             v_int = llops.genop('cast_char_to_int', [v], resulttype=Signed)
///             return llops.genop('getarrayitem', [c_table, v_int],
///                                resulttype=Char)
///         else:
///             return v
/// ```
///
/// `_conversion_tables` cache (rpbc.py + rpbc.py): keyed
/// on the identity of `r_to` (pointer address of the target
/// `SmallFunctionSetPBCRepr`). `Some(c_table)` reuses the cached
/// `Char Array` constant; `None` records an identity conversion
/// (upstream's `r = None` branch) so callers can return `v` directly.
pub fn conversion_table(
    r_from: &SmallFunctionSetPBCRepr,
    r_to: &SmallFunctionSetPBCRepr,
) -> Result<Option<Constant>, TyperError> {
    use crate::translator::rtyper::lltypesystem::lltype::{
        self as lltype, Array, LowLevelValue, MallocFlavor,
    };

    // upstream rpbc.py conversion_table — `if r_to in r_from._conversion_tables:
    //     return r_from._conversion_tables[r_to]`. Cache key = identity
    // of `r_to`; pointer address mirrors Python's dict-by-id semantics.
    let cache_key = (r_to as *const SmallFunctionSetPBCRepr) as usize;
    if let Some(cached) = r_from._conversion_tables.borrow().get(&cache_key).cloned() {
        return Ok(cached);
    }

    let from_descs = r_from.descriptions.borrow().clone();
    let to_descs = r_to.descriptions.borrow().clone();
    if from_descs.is_empty() || to_descs.is_empty() {
        return Err(TyperError::message(
            "conversion_table: descriptions not populated — _setup_repr must have run",
        ));
    }

    // upstream rpbc.py:581-588 — `for i, d in enumerate(r_from.descriptions):
    //   if d in r_to.descriptions: j = r_to.descriptions.index(d); l.append(j); t[i] = chr(j)
    //   else: l.append(None)`.
    //
    // `l[i] == i` for every i ⇔ identity (upstream's `l == range(len)`).
    let mut l: Vec<Option<usize>> = Vec::with_capacity(from_descs.len());
    let mut all_identity = from_descs.len() == to_descs.len();
    for (i, slot) in from_descs.iter().enumerate() {
        let j_opt = match slot {
            Some(d) => {
                let target_key = d.desc_key();
                to_descs.iter().position(|s| match s {
                    Some(td) => td.desc_key() == target_key,
                    None => false,
                })
            }
            // None entry (can_be_None) — upstream's set membership
            // would match the corresponding None in r_to.
            None => to_descs.iter().position(|s| s.is_none()),
        };
        if j_opt != Some(i) {
            all_identity = false;
        }
        l.push(j_opt);
    }

    // upstream rpbc.py:589-594 — `if l == range(len): r = None; else: r = inputconst(...)`.
    if all_identity {
        // upstream rpbc.py:593 — `r_from._conversion_tables[r_to] = None`.
        r_from
            ._conversion_tables
            .borrow_mut()
            .insert(cache_key, None);
        return Ok(None);
    }

    // upstream rpbc.py:578-580 — `t = malloc(Array(Char, hints=...),
    //   len(r_from.descriptions), immortal=True)`.
    let array_type = Array::with_hints(
        LowLevelType::Char,
        vec![
            ("nolength".to_string(), ConstValue::Bool(true)),
            ("immutable".to_string(), ConstValue::Bool(true)),
            ("static_immutable".to_string(), ConstValue::Bool(true)),
        ],
    );
    let array_lltype = LowLevelType::Array(Box::new(array_type));
    let mut t = lltype::malloc(
        array_lltype.clone(),
        Some(from_descs.len()),
        MallocFlavor::Gc,
        true,
    )
    .map_err(TyperError::message)?;

    // upstream rpbc.py — `t[i] = chr(j)` only when d ∈ r_to.descriptions.
    for (i, j_opt) in l.iter().enumerate() {
        if let Some(j) = j_opt {
            if *j > u8::MAX as usize {
                return Err(TyperError::message(format!(
                    "conversion_table: index {j} exceeds Char range (256)"
                )));
            }
            t.setitem(i, LowLevelValue::Char(char::from(*j as u8)))
                .map_err(TyperError::message)?;
        }
        // else: leave the slot at the malloc default — upstream relies
        // on the d∉r_to branch never being reached at runtime (the
        // narrower r_from is constructed to be a subset of r_to in
        // practice). If the lookup ever fires for such a slot it
        // surfaces as undefined behaviour upstream too.
    }

    // upstream rpbc.py — `r = inputconst(typeOf(t), t)`.
    let ptr_type = LowLevelType::Ptr(Box::new(
        lltype::Ptr::from_container_type(array_lltype.clone()).map_err(TyperError::message)?,
    ));
    let c_table = Constant::with_concretetype(ConstValue::LLPtr(Box::new(t)), ptr_type);

    // upstream rpbc.py — `r_from._conversion_tables[r_to] = r`.
    r_from
        ._conversion_tables
        .borrow_mut()
        .insert(cache_key, Some(c_table.clone()));

    Ok(Some(c_table))
}

/// RPython `pairtype(SmallFunctionSetPBCRepr,
/// SmallFunctionSetPBCRepr).convert_from_to` (rpbc.py).
pub(super) fn pair_small_function_set_small_function_set_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let r_from_set = (r_from as &dyn std::any::Any)
        .downcast_ref::<SmallFunctionSetPBCRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_small_function_set_small_function_set_convert_from_to: \
                 r_from is not SmallFunctionSetPBCRepr",
            )
        })?;
    let r_to_set = (r_to as &dyn std::any::Any)
        .downcast_ref::<SmallFunctionSetPBCRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_small_function_set_small_function_set_convert_from_to: \
                 r_to is not SmallFunctionSetPBCRepr",
            )
        })?;

    let c_table = conversion_table(r_from_set, r_to_set)?;
    small_function_set_apply_conversion(c_table.as_ref(), v, llops)
}

/// RPython `pairtype(SmallFunctionSetPBCRepr,
/// SmallFunctionSetPBCRepr).convert_from_to` (rpbc.py) emit
/// half — `c_table is None` ⇒ identity (`return v`); `c_table is
/// Some(...)` ⇒ `cast_char_to_int` + `getarrayitem`. Shared between
/// the cache-hit and cache-miss paths so the `_conversion_tables`
/// lookup short-circuits straight to the same emission shape.
fn small_function_set_apply_conversion(
    c_table: Option<&Constant>,
    v: &Hlvalue,
    llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    use crate::translator::rtyper::rtyper::GenopResult;

    let Some(c_table) = c_table else {
        // upstream rpbc.py:606-607 — `else: return v`.
        return Ok(Some(v.clone()));
    };

    // upstream rpbc.py:601 — `assert v.concretetype is Char`.
    let v_ty = match v {
        Hlvalue::Variable(var) => var.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    };
    if v_ty.as_ref() != Some(&LowLevelType::Char) {
        return Err(TyperError::message(format!(
            "pair_small_function_set_small_function_set_convert_from_to: \
             v.concretetype must be Char, got {:?}",
            v_ty,
        )));
    }

    // upstream rpbc.py:602-603 — `v_int = llops.genop('cast_char_to_int',
    //   [v], resulttype=Signed)`.
    let v_int = llops
        .genop(
            "cast_char_to_int",
            vec![v.clone()],
            GenopResult::LLType(LowLevelType::Signed),
        )
        .expect("cast_char_to_int with Signed result yields a Variable");

    // upstream rpbc.py:604-605 — `return llops.genop('getarrayitem',
    //   [c_table, v_int], resulttype=Char)`.
    let v_result = llops
        .genop(
            "getarrayitem",
            vec![Hlvalue::Constant(c_table.clone()), Hlvalue::Variable(v_int)],
            GenopResult::LLType(LowLevelType::Char),
        )
        .expect("getarrayitem with Char result yields a Variable");
    Ok(Some(Hlvalue::Variable(v_result)))
}

/// RPython `pairtype(FunctionsPBCRepr,
/// SmallFunctionSetPBCRepr).convert_from_to` (rpbc.py) +
/// `compression_function(r_set)` (rpbc.py).
///
/// ```python
/// class __extend__(pairtype(FunctionsPBCRepr, SmallFunctionSetPBCRepr)):
///     def convert_from_to((r_ptr, r_set), v, llops):
///         ll_compress = compression_function(r_set)
///         return llops.gendirectcall(ll_compress, v)
/// ```
///
/// Looks up (or builds and caches) the `ll_compress` helper graph
/// pointer via [`SmallFunctionSetPBCRepr::compression_function`] and
/// emits a `direct_call` with the function pointer + the v argument.
pub(super) fn pair_functions_pbc_small_function_set_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    use crate::translator::rtyper::rtyper::GenopResult;

    let _r_ptr = (r_from as &dyn std::any::Any)
        .downcast_ref::<FunctionsPBCRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_functions_pbc_small_function_set_convert_from_to: \
                 r_from is not FunctionsPBCRepr",
            )
        })?;
    let r_set = (r_to as &dyn std::any::Any)
        .downcast_ref::<SmallFunctionSetPBCRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_functions_pbc_small_function_set_convert_from_to: \
                 r_to is not SmallFunctionSetPBCRepr",
            )
        })?;

    // upstream: `ll_compress = compression_function(r_set)`.
    let ll_compress_const = r_set.compression_function()?;

    // upstream: `return llops.gendirectcall(ll_compress, v)`.
    //
    // `gendirectcall` emits a `direct_call` with the function pointer
    // followed by the python-level arguments — here just `v`.
    let v_result = llops
        .genop(
            "direct_call",
            vec![Hlvalue::Constant(ll_compress_const), v.clone()],
            GenopResult::LLType(LowLevelType::Char),
        )
        .expect("direct_call with Char result yields a Variable");
    Ok(Some(Hlvalue::Variable(v_result)))
}

/// RPython `class SingleFrozenPBCRepr(Repr)` (rpbc.py).
///
/// Void-typed repr for a single-FrozenDesc `SomePBC` with
/// `can_be_None=False`. `convert_desc` asserts desc-identity with the
/// stored `frozendesc` then emits `Constant(None, Void)`; `convert_const`
/// is the same sentinel (upstream `return None`).
#[derive(Debug)]
pub struct SingleFrozenPBCRepr {
    /// RPython `self.frozendesc = frozendesc` (rpbc.py).
    pub frozendesc: DescEntry,
    state: ReprState,
    lltype: LowLevelType,
}

impl SingleFrozenPBCRepr {
    /// RPython `SingleFrozenPBCRepr.__init__(self, frozendesc)`
    /// (rpbc.py).
    pub fn new(frozendesc: DescEntry) -> Self {
        SingleFrozenPBCRepr {
            frozendesc,
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        }
    }
}

impl Repr for SingleFrozenPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "SingleFrozenPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::SingleFrozenPBCRepr
    }

    /// RPython `SingleFrozenPBCRepr.convert_desc(self, frozendesc)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_desc(self, frozendesc):
    ///     assert frozendesc is self.frozendesc
    ///     return object()  # lowleveltype is Void
    /// ```
    ///
    /// Pyre's identity check uses [`DescEntry::desc_key`] which wraps
    /// `Rc::as_ptr` — equivalent to upstream's `frozendesc is
    /// self.frozendesc`. The sentinel is emitted as a `Constant(None,
    /// Void)` so `convert_desc_or_const` gets a typed return instead
    /// of upstream's untagged Python `object()`.
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        if desc.desc_key() != self.frozendesc.desc_key() {
            return Err(TyperError::message(
                "SingleFrozenPBCRepr.convert_desc: frozendesc identity mismatch",
            ));
        }
        Ok(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    }

    /// RPython `SingleFrozenPBCRepr.convert_const(self, value)`
    /// (rpbc.py): upstream `return None`. Mirrors
    /// `FunctionRepr.convert_const` — Void sentinel.
    fn convert_const(&self, _value: &ConstValue) -> Result<Constant, TyperError> {
        Ok(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    }

    /// RPython `SingleFrozenPBCRepr.rtype_getattr(_, hop)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def rtype_getattr(_, hop):
    ///     if not hop.s_result.is_constant():
    ///         raise TyperError("getattr on a constant PBC returns a non-constant")
    ///     return hop.inputconst(hop.r_result, hop.s_result.const)
    /// ```
    ///
    /// Structurally identical to
    /// [`MultipleUnrelatedFrozenPBCRepr::rtype_getattr`] — Void-typed
    /// reprs carry no runtime fields, so `getattr` is only valid when
    /// the annotator can hand back a constant.
    fn rtype_getattr(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        let s_const = hop
            .s_result
            .borrow()
            .as_ref()
            .and_then(|s| s.const_())
            .cloned();
        let Some(value) = s_const else {
            return Err(TyperError::message(
                "getattr on a constant PBC returns a non-constant",
            ));
        };
        let r_result = hop
            .r_result
            .borrow()
            .clone()
            .ok_or_else(|| TyperError::message("HighLevelOp.r_result is not set"))?;
        let c = crate::translator::rtyper::rtyper::HighLevelOp::inputconst(&*r_result, &value)?;
        Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)))
    }
}

/// RPython `class MultipleFrozenPBCReprBase(CanBeNull, Repr)`
/// (rpbc.py).
///
/// Rust keeps the two concrete subclasses as independent structs, but this
/// trait carries the shared base-class behavior they both inherit upstream:
/// `convert_const`, `get_ll_eq_function`, and the `CanBeNull` marker.
pub trait MultipleFrozenPBCReprBase: CanBeNull {
    /// Access to the `self.rtyper` back-edge stored by both subclasses.
    fn rtyper_weak(&self) -> &Weak<RPythonTyper>;

    /// Subclass-specific `null_instance()` converted to a typed Constant.
    fn null_instance_constant(&self) -> Result<Constant, TyperError>;

    /// Shared body for RPython
    /// `MultipleFrozenPBCReprBase.convert_const(self, pbc)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_const(self, pbc):
    ///     if pbc is None:
    ///         return self.null_instance()
    ///     frozendesc = self.rtyper.annotator.bookkeeper.getdesc(pbc)
    ///     return self.convert_desc(frozendesc)
    /// ```
    fn convert_const_multiple_frozen_pbc_base(
        &self,
        value: &ConstValue,
    ) -> Result<Constant, TyperError> {
        if matches!(value, ConstValue::None) {
            return self.null_instance_constant();
        }
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "{}.convert_const: non-None value must be a HostObject \
                 (frozen pbc), got {value:?}",
                self.class_name()
            )));
        };
        let rtyper = self.rtyper_weak().upgrade().ok_or_else(|| {
            TyperError::message(format!(
                "{}.convert_const: rtyper weak ref dropped",
                self.class_name()
            ))
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message(format!(
                "{}.convert_const: annotator weak ref dropped",
                self.class_name()
            ))
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        Repr::convert_desc(self, &desc)
    }

    /// RPython `MultipleFrozenPBCReprBase.get_ll_eq_function(self)`.
    fn get_ll_eq_function_multiple_frozen_pbc_base(
        &self,
        _rtyper: &RPythonTyper,
    ) -> Result<Option<crate::translator::rtyper::rtyper::LowLevelFunction>, TyperError> {
        Ok(None)
    }
}

/// RPython `class MultipleUnrelatedFrozenPBCRepr(MultipleFrozenPBCReprBase)`
/// (rpbc.py).
///
/// Representation for a `SomePBC` of frozen PBCs that have no common
/// access set. The only operation upstream allows is `is` comparison,
/// lowered to `adr_eq` at pairtype dispatch.
///
/// ```python
/// class MultipleUnrelatedFrozenPBCRepr(MultipleFrozenPBCReprBase):
///     lowleveltype = llmemory.Address
///     EMPTY = Struct('pbc', hints={'immutable': True, 'static_immutable': True})
///
///     def __init__(self, rtyper):
///         self.rtyper = rtyper
///         self.converted_pbc_cache = {}
///
///     def null_instance(self):
///         return llmemory.Address._defl()
/// ```
///
/// Pyre lowleveltype is [`LowLevelType::Address`]. `null_instance`
/// returns the NULL fakeaddress sentinel; `convert_desc` /
/// `convert_const` / `convert_pbc` / `create_instance` /
/// `rtype_getattr` are all ported line-by-line from
/// `rpbc.py`. `Bookkeeper::getdesc` + `rtyper.getrepr` +
/// `_address::Fake` (fakeaddress) are wired through; the only
/// remaining limitation is that `convert_desc` requires the
/// per-frozendesc `r = rtyper.getrepr(SomePBC([d]))` to produce an
/// `LLPtr` — desc lattices that route to a non-pointer repr surface
/// the structural mismatch as a `TyperError`.
#[derive(Debug)]
pub struct MultipleUnrelatedFrozenPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:682).
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.converted_pbc_cache = {}` (rpbc.py).
    /// Keyed by `DescEntry::desc_key()` for pointer-identity semantics
    /// matching upstream's dict-by-Python-identity. Value is the
    /// `_address` produced by `convert_pbc(pbcptr)` —
    /// `LowLevelValue::Address(_)` so callers can surface it
    /// directly as a Constant.
    pub converted_pbc_cache: RefCell<
        HashMap<
            crate::annotator::description::DescKey,
            crate::translator::rtyper::lltypesystem::lltype::_address,
        >,
    >,
    state: ReprState,
    lltype: LowLevelType,
}

impl MultipleUnrelatedFrozenPBCRepr {
    /// RPython `MultipleUnrelatedFrozenPBCRepr.__init__(self, rtyper)`
    /// (rpbc.py).
    pub fn new(rtyper: &Rc<RPythonTyper>) -> Self {
        MultipleUnrelatedFrozenPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            converted_pbc_cache: RefCell::new(HashMap::new()),
            state: ReprState::new(),
            lltype: LowLevelType::Address,
        }
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.create_instance(self)`
    /// (rpbc.py):
    ///
    /// ```python
    /// EMPTY = Struct('pbc', hints={'immutable': True, 'static_immutable': True})
    /// def create_instance(self):
    ///     return malloc(self.EMPTY, immortal=True)
    /// ```
    ///
    /// Allocates a fresh empty `Struct('pbc', ...)` carrier. Used by
    /// [`Self::convert_desc`] when the matching SomePBC desc's repr
    /// is `Void` (the desc carries no fields, so a placeholder
    /// allocation is needed for `fakeaddress` to point at).
    pub fn create_instance(&self) -> Result<_ptr, TyperError> {
        // upstream `Struct('pbc', hints={'immutable': True,
        // 'static_immutable': True})` — fields-empty struct, Raw flavor
        // (default for `Struct(...)` without `Gc` prefix).
        let body = crate::translator::rtyper::lltypesystem::lltype::Struct::with_hints(
            "pbc",
            vec![],
            vec![
                ("immutable".into(), ConstValue::Bool(true)),
                ("static_immutable".into(), ConstValue::Bool(true)),
            ],
        );
        let body_ty = LowLevelType::Struct(Box::new(body));
        crate::translator::rtyper::lltypesystem::lltype::malloc(
            body_ty,
            None,
            crate::translator::rtyper::lltypesystem::lltype::MallocFlavor::Raw,
            true,
        )
        .map_err(TyperError::message)
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.convert_pbc(self, pbcptr)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_pbc(self, pbcptr):
    ///     return llmemory.fakeaddress(pbcptr)
    /// ```
    ///
    /// Wraps a live `_ptr` in [`_address::Fake`] so it flows through
    /// `Address`-typed slots. Used by [`Self::convert_desc`] after
    /// the per-frozendesc repr produced its `_ptr` value.
    pub fn convert_pbc(
        &self,
        pbcptr: _ptr,
    ) -> crate::translator::rtyper::lltypesystem::lltype::_address {
        crate::translator::rtyper::lltypesystem::lltype::_address::Fake(Box::new(pbcptr))
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.null_instance(self)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def null_instance(self):
    ///     return llmemory.Address._defl()
    /// ```
    ///
    /// Returns the NULL fakeaddress sentinel. Consumers use this for
    /// the `can_be_None` fallback in `convert_const` and for pairtype
    /// dispatch defaults.
    pub fn null_instance(&self) -> Constant {
        let value = match LowLevelType::Address._defl() {
            crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Address(addr) => {
                ConstValue::LLAddress(addr)
            }
            other => {
                unreachable!("Address._defl() must yield LowLevelValue::Address, got {other:?}")
            }
        };
        Constant::with_concretetype(value, LowLevelType::Address)
    }
}

impl CanBeNull for MultipleUnrelatedFrozenPBCRepr {}

impl MultipleFrozenPBCReprBase for MultipleUnrelatedFrozenPBCRepr {
    fn rtyper_weak(&self) -> &Weak<RPythonTyper> {
        &self.rtyper
    }

    fn null_instance_constant(&self) -> Result<Constant, TyperError> {
        Ok(self.null_instance())
    }
}

impl Repr for MultipleUnrelatedFrozenPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "MultipleUnrelatedFrozenPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::MultipleUnrelatedFrozenPBCRepr
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.convert_desc(self, frozendesc)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_desc(self, frozendesc):
    ///     try:
    ///         return self.converted_pbc_cache[frozendesc]
    ///     except KeyError:
    ///         r = self.rtyper.getrepr(annmodel.SomePBC([frozendesc]))
    ///         if r.lowleveltype is Void:
    ///             pbc = self.create_instance()
    ///         else:
    ///             pbc = r.convert_desc(frozendesc)
    ///         convpbc = self.convert_pbc(pbc)
    ///         self.converted_pbc_cache[frozendesc] = convpbc
    ///         return convpbc
    /// ```
    ///
    /// Routes the frozen desc through its concrete per-desc repr,
    /// extracts the `_ptr`, and wraps it via `convert_pbc` (=
    /// `fakeaddress`) into the `Address`-typed cache.
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        let desc_key = desc.desc_key();
        if let Some(cached) = self.converted_pbc_cache.borrow().get(&desc_key) {
            return Ok(Constant::with_concretetype(
                ConstValue::LLAddress(cached.clone()),
                LowLevelType::Address,
            ));
        }

        let DescEntry::Frozen(_) = desc else {
            return Err(TyperError::message(format!(
                "MultipleUnrelatedFrozenPBCRepr.convert_desc: non-Frozen desc {desc:?}"
            )));
        };

        // upstream `r = self.rtyper.getrepr(annmodel.SomePBC([frozendesc]))`.
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message(
                "MultipleUnrelatedFrozenPBCRepr.convert_desc: rtyper weak ref dropped",
            )
        })?;
        let s_pbc = SomePBC::new(vec![desc.clone()], false);
        let r = rtyper.getrepr(&crate::annotator::model::SomeValue::PBC(s_pbc))?;

        // upstream `if r.lowleveltype is Void: pbc = self.create_instance()
        // else: pbc = r.convert_desc(frozendesc)`.
        let pbc = if matches!(r.lowleveltype(), LowLevelType::Void) {
            self.create_instance()?
        } else {
            let converted = r.convert_desc(desc)?;
            match converted.value {
                ConstValue::LLPtr(p) => *p,
                other => {
                    return Err(TyperError::message(format!(
                        "MultipleUnrelatedFrozenPBCRepr.convert_desc: per-desc repr \
                         {} returned non-LLPtr value {other:?}",
                        r.class_name()
                    )));
                }
            }
        };

        // upstream `convpbc = self.convert_pbc(pbc); self.converted_pbc_cache[frozendesc] = convpbc`.
        let convpbc = self.convert_pbc(pbc);
        self.converted_pbc_cache
            .borrow_mut()
            .insert(desc_key, convpbc.clone());
        Ok(Constant::with_concretetype(
            ConstValue::LLAddress(convpbc),
            LowLevelType::Address,
        ))
    }

    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        self.convert_const_multiple_frozen_pbc_base(value)
    }

    fn get_ll_eq_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<crate::translator::rtyper::rtyper::LowLevelFunction>, TyperError> {
        self.get_ll_eq_function_multiple_frozen_pbc_base(rtyper)
    }

    fn rtype_bool(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        crate::translator::rtyper::rmodel::can_be_null_rtype_bool(self, hop)
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.rtype_getattr(_, hop)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def rtype_getattr(_, hop):
    ///     if not hop.s_result.is_constant():
    ///         raise TyperError("getattr on a constant PBC returns a non-constant")
    ///     return hop.inputconst(hop.r_result, hop.s_result.const)
    /// ```
    ///
    /// A MU repr carries no structural fields, so upstream only allows
    /// getattr when the annotator has already proved the result constant.
    fn rtype_getattr(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        let s_const = hop
            .s_result
            .borrow()
            .as_ref()
            .and_then(|s| s.const_())
            .cloned();
        let Some(value) = s_const else {
            return Err(TyperError::message(
                "getattr on a constant PBC returns a non-constant",
            ));
        };
        let r_result = hop
            .r_result
            .borrow()
            .clone()
            .ok_or_else(|| TyperError::message("HighLevelOp.r_result is not set"))?;
        let c = crate::translator::rtyper::rtyper::HighLevelOp::inputconst(&*r_result, &value)?;
        Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)))
    }
}

/// RPython `pairtype(MultipleUnrelatedFrozenPBCRepr,
/// MultipleUnrelatedFrozenPBCRepr).rtype_is_` (rpbc.py).
///
/// ```python
/// def rtype_is_((robj1, robj2), hop):
///     if isinstance(robj1, MultipleUnrelatedFrozenPBCRepr):
///         r = robj1
///     else:
///         r = robj2
///     vlist = hop.inputargs(r, r)
///     return hop.genop('adr_eq', vlist, resulttype=Bool)
/// ```
///
/// The same `rtype_is_` body backs `pairtype(MU, MU)`,
/// `pairtype(MU, SingleFrozen)`, and `pairtype(SingleFrozen, MU)`. The
/// receiver `r` is whichever operand is the MU repr, and BOTH operands
/// are converted to `r` via `inputargs(r, r)`. A `SingleFrozen` operand
/// is a single prebuilt constant, so its `inputarg` routes through
/// `inputconst(r, value)` -> `MU.convert_const` (fakeaddress for a live
/// frozendesc, NULL address for `None`); no `convert_from_to` is needed.
pub fn pair_mu_mu_rtype_is_(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
    use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult};
    // upstream: `if isinstance(robj1, MultipleUnrelatedFrozenPBCRepr):
    //              r = robj1; else: r = robj2`.
    let r = if matches!(
        r1.repr_class_id(),
        ReprClassId::MultipleUnrelatedFrozenPBCRepr
    ) {
        r1
    } else {
        r2
    };
    // upstream: `vlist = hop.inputargs(r, r)`.
    let v_list = hop.inputargs(vec![ConvertedTo::Repr(r), ConvertedTo::Repr(r)])?;
    Ok(hop
        .genop("adr_eq", v_list, GenopResult::LLType(LowLevelType::Bool))
        .expect("adr_eq with Bool result returns a value"))
}

/// RPython `pairtype(FunctionReprBase, FunctionReprBase).rtype_is_`
/// (rpbc.py):
///
/// ```python
/// def rtype_is_((robj1, robj2), hop):
///     if hop.s_result.is_constant():
///         return inputconst(Bool, hop.s_result.const)
///     s_pbc = annmodel.unionof(robj1.s_pbc, robj2.s_pbc)
///     r_pbc = hop.rtyper.getrepr(s_pbc)
///     v1, v2 = hop.inputargs(r_pbc, r_pbc)
///     assert v1.concretetype == v2.concretetype
///     if v1.concretetype == Char:
///         return hop.genop('char_eq', [v1, v2], resulttype=Bool)
///     elif isinstance(v1.concretetype, Ptr):
///         return hop.genop('ptr_eq', [v1, v2], resulttype=Bool)
///     else:
///         raise TyperError("unknown type %r" % (v1.concretetype,))
/// ```
///
/// `robj1` / `robj2` come from any pair of FunctionReprBase analogues
/// (FunctionRepr / FunctionsPBCRepr / SmallFunctionSetPBCRepr); pyre
/// reads each side's `pbc_s_pbc()` (the trait accessor mirroring
/// upstream `FunctionReprBase.s_pbc`) and union-resolves them.
pub fn pair_function_repr_base_rtype_is_(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
    use crate::annotator::model::{SomeObjectTrait, SomeValue, unionof};
    use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp};

    // upstream: `if hop.s_result.is_constant(): return
    //              inputconst(Bool, hop.s_result.const)`.
    if let Some(s_result) = hop.s_result.borrow().clone()
        && s_result.is_constant()
        && let Some(cv) = s_result.const_().cloned()
    {
        return Ok(Hlvalue::Constant(HighLevelOp::inputconst(
            ConvertedTo::LowLevelType(&LowLevelType::Bool),
            &cv,
        )?));
    }

    // upstream: `s_pbc = annmodel.unionof(robj1.s_pbc, robj2.s_pbc);
    //           r_pbc = hop.rtyper.getrepr(s_pbc)`.
    let s_pbc1 = r1.pbc_s_pbc().ok_or_else(|| {
        TyperError::message("pair_function_repr_base_rtype_is_: r1 has no FunctionReprBase.s_pbc")
    })?;
    let s_pbc2 = r2.pbc_s_pbc().ok_or_else(|| {
        TyperError::message("pair_function_repr_base_rtype_is_: r2 has no FunctionReprBase.s_pbc")
    })?;
    let sv1 = SomeValue::PBC(s_pbc1.clone());
    let sv2 = SomeValue::PBC(s_pbc2.clone());
    let s_union = unionof([&sv1, &sv2]).map_err(|e| TyperError::message(e.to_string()))?;
    let r_pbc = hop.rtyper.getrepr(&s_union)?;

    // upstream: `v1, v2 = hop.inputargs(r_pbc, r_pbc)`.
    let v_list = hop.inputargs(vec![
        ConvertedTo::Repr(r_pbc.as_ref()),
        ConvertedTo::Repr(r_pbc.as_ref()),
    ])?;
    let mut v_iter = v_list.into_iter();
    let v1 = v_iter.next().expect("inputargs returns 2 values");
    let v2 = v_iter.next().expect("inputargs returns 2 values");

    // upstream: `if v1.concretetype == Char: char_eq;
    //           elif Ptr: ptr_eq; else raise`.
    let v1_ct = match &v1 {
        Hlvalue::Variable(var) => var.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
    .ok_or_else(|| {
        TyperError::message("pair_function_repr_base_rtype_is_: v1 has no concretetype")
    })?;
    let opname = match v1_ct {
        LowLevelType::Char => "char_eq",
        LowLevelType::Ptr(_) => "ptr_eq",
        other => {
            return Err(TyperError::message(format!(
                "pair_function_repr_base_rtype_is_: unknown type {other:?}"
            )));
        }
    };
    Ok(hop
        .genop(
            opname,
            vec![v1, v2],
            GenopResult::LLType(LowLevelType::Bool),
        )
        .expect("char_eq/ptr_eq with Bool result returns a value"))
}

/// RPython `class MultipleFrozenPBCRepr(MultipleFrozenPBCReprBase)`
/// (rpbc.py).
///
/// Representation for a SomePBC of frozen PBCs that share a common
/// `ClassAttrFamily`-style access set. The vtable layout is a custom
/// `Struct('pbc', ...)` whose fields are one per attribute the access
/// set tracks; each frozen desc materialises into an immortal
/// allocation of that struct, with attribute values written eagerly
/// from the matching `frozendesc.attrcache`.
///
/// ```python
/// class MultipleFrozenPBCRepr(MultipleFrozenPBCReprBase):
///     def __init__(self, rtyper, access_set):
///         self.rtyper = rtyper
///         self.access_set = access_set
///         self.pbc_type = ForwardReference()
///         self.lowleveltype = Ptr(self.pbc_type)
///         self.pbc_cache = {}
///
///     def _setup_repr(self):
///         llfields = self._setup_repr_fields()
///         kwds = {'hints': {'immutable': True, 'static_immutable': True}}
///         self.pbc_type.become(Struct('pbc', *llfields, **kwds))
///     ...
/// ```
#[derive(Debug)]
pub struct MultipleFrozenPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:732).
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.access_set = access_set` (rpbc.py). `None` is the
    /// no-attrs arm (`getFrozenPBCRepr`'s `access is None` case): a
    /// zero-field `Struct('pbc')` with no per-attr membership check
    /// (rpbc.py, 770).
    pub access_set: Option<Rc<RefCell<crate::annotator::description::FrozenAttrFamily>>>,
    /// RPython `self.pbc_type = ForwardReference()` (rpbc.py).
    /// Stored as a clone alongside [`Self::lltype`] so callers can
    /// observe the resolved struct after `_setup_repr` runs `become`
    /// against either reference (Rust's `ForwardReference` shares its
    /// `Arc<Mutex<…>>` slot across clones).
    pbc_type: crate::translator::rtyper::lltypesystem::lltype::ForwardReference,
    /// RPython `self.lowleveltype = Ptr(self.pbc_type)` (rpbc.py).
    lltype: LowLevelType,
    /// RPython `self.pbc_cache = {}` (rpbc.py). Caches the
    /// per-frozendesc materialised `_ptr` so repeated `convert_desc`
    /// calls return identity-equal pointers and `convert_const` cycles
    /// terminate.
    pbc_cache: RefCell<HashMap<crate::annotator::description::DescKey, _ptr>>,
    /// RPython `self.fieldmap` populated by `_setup_repr_fields`
    /// (rpbc.py). Maps each tracked attrname to its mangled struct
    /// field name + the per-attr Repr for the value type.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    fieldmap: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    state: ReprState,
}

impl MultipleFrozenPBCRepr {
    /// RPython `MultipleFrozenPBCRepr.__init__(self, rtyper, access_set)`
    /// (rpbc.py).
    pub fn new(
        rtyper: &Rc<RPythonTyper>,
        access_set: Option<Rc<RefCell<crate::annotator::description::FrozenAttrFamily>>>,
    ) -> Self {
        let pbc_type = crate::translator::rtyper::lltypesystem::lltype::ForwardReference::new();
        let pbc_type_clone = pbc_type.clone();
        let lltype = LowLevelType::Ptr(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::Ptr {
                TO: crate::translator::rtyper::lltypesystem::lltype::PtrTarget::ForwardReference(
                    pbc_type_clone,
                ),
            },
        ));
        MultipleFrozenPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            access_set,
            pbc_type,
            lltype,
            pbc_cache: RefCell::new(HashMap::new()),
            fieldmap: RefCell::new(HashMap::new()),
            state: ReprState::new(),
        }
    }

    /// RPython `MultipleFrozenPBCRepr._setup_repr_fields(self)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def _setup_repr_fields(self):
    ///     fields = []
    ///     self.fieldmap = {}
    ///     if self.access_set is not None:
    ///         attrlist = self.access_set.attrs.keys()
    ///         attrlist.sort()
    ///         for attr in attrlist:
    ///             s_value = self.access_set.attrs[attr]
    ///             r_value = self.rtyper.getrepr(s_value)
    ///             mangled_name = mangle('pbc', attr)
    ///             fields.append((mangled_name, r_value.lowleveltype))
    ///             self.fieldmap[attr] = mangled_name, r_value
    ///     return fields
    /// ```
    fn setup_repr_fields(&self) -> Result<Vec<(String, LowLevelType)>, TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("MultipleFrozenPBCRepr._setup_repr_fields: rtyper weak ref dropped")
        })?;
        // upstream `if self.access_set is not None:` (rpbc.py) — the
        // no-attrs arm leaves `fields`/`fieldmap` empty, giving a
        // zero-field `Struct('pbc')`.
        let Some(access_set) = &self.access_set else {
            self.fieldmap.borrow_mut().clear();
            return Ok(Vec::new());
        };
        // upstream `attrlist = self.access_set.attrs.keys(); attrlist.sort()`.
        let attrs_snapshot: Vec<(String, crate::annotator::model::SomeValue)> = {
            let caf = access_set.borrow();
            let mut attrs: Vec<(String, crate::annotator::model::SomeValue)> = caf
                .attrs
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            attrs.sort_by(|a, b| a.0.cmp(&b.0));
            attrs
        };
        let mut fields = Vec::with_capacity(attrs_snapshot.len());
        let mut new_fieldmap: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        for (attr, s_value) in attrs_snapshot {
            let r_value = rtyper.getrepr(&s_value)?;
            let mangled_name = crate::translator::rtyper::rmodel::mangle("pbc", &attr);
            fields.push((mangled_name.clone(), r_value.lowleveltype().clone()));
            new_fieldmap.insert(attr, (mangled_name, r_value));
        }
        *self.fieldmap.borrow_mut() = new_fieldmap;
        Ok(fields)
    }

    /// RPython `MultipleFrozenPBCRepr.create_instance(self)` (rpbc.py):
    ///
    /// ```python
    /// def create_instance(self):
    ///     return malloc(self.pbc_type, immortal=True)
    /// ```
    pub fn create_instance(&self) -> Result<_ptr, TyperError> {
        let resolved = self.pbc_type.resolved().ok_or_else(|| {
            TyperError::message(
                "MultipleFrozenPBCRepr.create_instance: pbc_type ForwardReference \
                 not resolved — call setup() first",
            )
        })?;
        crate::translator::rtyper::lltypesystem::lltype::malloc(
            resolved,
            None,
            crate::translator::rtyper::lltypesystem::lltype::MallocFlavor::Raw,
            true,
        )
        .map_err(TyperError::message)
    }

    /// RPython `MultipleFrozenPBCRepr.null_instance(self)` (rpbc.py):
    ///
    /// ```python
    /// def null_instance(self):
    ///     return nullptr(self.pbc_type)
    /// ```
    pub fn null_instance(&self) -> Result<_ptr, TyperError> {
        let LowLevelType::Ptr(ptr) = &self.lltype else {
            return Err(TyperError::message(
                "MultipleFrozenPBCRepr.null_instance: lowleveltype is not Ptr",
            ));
        };
        Ok(ptr.as_ref().clone()._defl())
    }

    /// RPython `MultipleFrozenPBCRepr.getfield(self, vpbc, attr, llops)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def getfield(self, vpbc, attr, llops):
    ///     mangled_name, r_value = self.fieldmap[attr]
    ///     cmangledname = inputconst(Void, mangled_name)
    ///     return llops.genop('getfield', [vpbc, cmangledname], resulttype=r_value)
    /// ```
    pub fn getfield(
        &self,
        vpbc: Hlvalue,
        attr: &str,
        llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
    ) -> Result<crate::flowspace::model::Variable, TyperError> {
        let (mangled_name, r_value) =
            self.fieldmap.borrow().get(attr).cloned().ok_or_else(|| {
                TyperError::message(format!(
                    "MultipleFrozenPBCRepr.getfield: attr {attr:?} not in fieldmap"
                ))
            })?;
        let cname =
            Constant::with_concretetype(ConstValue::byte_str(mangled_name), LowLevelType::Void);
        Ok(llops
            .genop(
                "getfield",
                vec![vpbc, Hlvalue::Constant(cname)],
                crate::translator::rtyper::rtyper::GenopResult::LLType(
                    r_value.lowleveltype().clone(),
                ),
            )
            .expect("getfield with non-Void result yields a Variable"))
    }
}

impl CanBeNull for MultipleFrozenPBCRepr {}

impl MultipleFrozenPBCReprBase for MultipleFrozenPBCRepr {
    fn rtyper_weak(&self) -> &Weak<RPythonTyper> {
        &self.rtyper
    }

    fn null_instance_constant(&self) -> Result<Constant, TyperError> {
        let null = self.null_instance()?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(null)),
            self.lltype.clone(),
        ))
    }
}

impl Repr for MultipleFrozenPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "MultipleFrozenPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::MultipleFrozenPBCRepr
    }

    /// RPython `MultipleFrozenPBCRepr._setup_repr(self)` (rpbc.py):
    ///
    /// ```python
    /// def _setup_repr(self):
    ///     llfields = self._setup_repr_fields()
    ///     kwds = {'hints': {'immutable': True, 'static_immutable': True}}
    ///     self.pbc_type.become(Struct('pbc', *llfields, **kwds))
    /// ```
    fn _setup_repr(&self) -> Result<(), TyperError> {
        let llfields = self.setup_repr_fields()?;
        let body = crate::translator::rtyper::lltypesystem::lltype::Struct::with_hints(
            "pbc",
            llfields,
            vec![
                ("immutable".into(), ConstValue::Bool(true)),
                ("static_immutable".into(), ConstValue::Bool(true)),
            ],
        );
        self.pbc_type
            .r#become(LowLevelType::Struct(Box::new(body)))
            .map_err(TyperError::message)?;
        Ok(())
    }

    /// RPython `MultipleFrozenPBCRepr.convert_desc(self, frozendesc)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_desc(self, frozendesc):
    ///     if (self.access_set is not None and
    ///             frozendesc not in self.access_set.descs):
    ///         raise TyperError("not found in PBC access set: %r" % (frozendesc,))
    ///     try:
    ///         return self.pbc_cache[frozendesc]
    ///     except KeyError:
    ///         self.setup()
    ///         result = self.create_instance()
    ///         self.pbc_cache[frozendesc] = result
    ///         for attr, (mangled_name, r_value) in self.fieldmap.items():
    ///             if r_value.lowleveltype is Void:
    ///                 continue
    ///             try:
    ///                 thisattrvalue = frozendesc.attrcache[attr]
    ///             except KeyError:
    ///                 if frozendesc.warn_missing_attribute(attr):
    ///                     warning("Desc %r has no attribute %r" % (frozendesc, attr))
    ///                 continue
    ///             llvalue = r_value.convert_const(thisattrvalue)
    ///             setattr(result, mangled_name, llvalue)
    ///         return result
    /// ```
    ///
    /// Cache-before-init order matches upstream rpbc.py (parity
    /// with the TupleRepr / InstanceRepr fixes from the recent reviewer
    /// pass): the `_ptr` is inserted into `pbc_cache` before any
    /// recursive `r_value.convert_const(...)` so cyclic frozen graphs
    /// terminate via the cached entry.
    fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        // upstream `if (self.access_set is not None and
        // frozendesc not in self.access_set.descs)` — the membership
        // check is skipped in the `None` (no-attrs) arm.
        //
        // upstream's `convert_desc` does not require a FrozenDesc: in the
        // `access_set is None` arm the fieldmap is empty and the desc is
        // consumed only as a cache key, so an uncallable FunctionDesc
        // routed here by `getFrozenPBCRepr` is accepted and materialises
        // an empty zero-field instance. Only the `access_set is Some` arm
        // reads `attrcache`, so a FrozenDesc is required there.
        //
        // PRE-EXISTING-DEVIATION: pyre keys
        // `FrozenAttrFamily.descs` on `FrozenDesc.base.identity`
        // (counter-based DescKey) because that is the key
        // `frozenpbc_attr_families` UnionFind uses; the parallel
        // `DescEntry::desc_key()` returns the `Rc::as_ptr`-based
        // identity used elsewhere. Look up by `base.identity` so
        // membership matches what `mergeattrfamilies` populated.
        let fd_rc = match desc {
            crate::annotator::description::DescEntry::Frozen(fd_rc) => Some(fd_rc),
            _ => None,
        };
        if let Some(access_set) = &self.access_set {
            let fd_rc = fd_rc.ok_or_else(|| {
                TyperError::message(format!(
                    "MultipleFrozenPBCRepr.convert_desc: access_set=Some requires a \
                     Frozen desc, got {desc:?}"
                ))
            })?;
            let identity_key = fd_rc.borrow().base.identity;
            if !access_set.borrow().descs.contains_key(&identity_key) {
                return Err(TyperError::message(format!(
                    "MultipleFrozenPBCRepr.convert_desc: {desc:?} not found in PBC access set"
                )));
            }
        }
        let desc_key = desc.desc_key();

        // upstream `return self.pbc_cache[frozendesc]` fast path.
        if let Some(cached) = self.pbc_cache.borrow().get(&desc_key) {
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(cached.clone())),
                self.lltype.clone(),
            ));
        }

        // upstream `self.setup()`. Repr::setup is idempotent.
        Repr::setup(self as &dyn Repr)?;

        // upstream `result = self.create_instance(); self.pbc_cache[frozendesc] = result`.
        let result = self.create_instance()?;
        self.pbc_cache.borrow_mut().insert(desc_key, result);

        // upstream loop over fieldmap, writing each attr's converted
        // value into the cached pointer through brief borrow_mut bursts.
        // `fieldmap` is non-empty only in the `access_set is Some` arm,
        // where the desc is a FrozenDesc (guarded above); `None` when
        // access_set is None so the loop below never runs.
        let frozendesc = fd_rc.cloned();
        let fieldmap_snapshot: Vec<(String, String, Arc<dyn Repr>)> = self
            .fieldmap
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.0.clone(), v.1.clone()))
            .collect();
        for (attr, mangled_name, r_value) in fieldmap_snapshot {
            if matches!(r_value.lowleveltype(), LowLevelType::Void) {
                continue;
            }
            // A non-empty fieldmap only occurs on the `access_set is Some`
            // arm, where `desc` was validated as a FrozenDesc above.
            let frozendesc = frozendesc.as_ref().ok_or_else(|| {
                TyperError::message(
                    "MultipleFrozenPBCRepr.convert_desc: non-empty fieldmap requires a Frozen desc",
                )
            })?;
            // upstream `frozendesc.attrcache[attr]`; missing attrs
            // consult `warn_missing_attribute` and emit the same rtyper
            // warning before leaving the field unset.
            let attrvalue = match frozendesc.borrow().attrcache.borrow().get(&attr) {
                Some(v) => v.clone(),
                None => {
                    if frozendesc
                        .borrow()
                        .warn_missing_attribute(&attr)
                        .map_err(|err| TyperError::message(err.to_string()))?
                    {
                        warning(&format!(
                            "Desc {:?} has no attribute {:?}",
                            frozendesc, attr
                        ));
                    }
                    continue;
                }
            };
            let item_const = r_value.convert_const(&attrvalue)?;
            let llval = crate::translator::rtyper::rclass::constant_to_lowlevel_value(&item_const)?;
            let mut cache = self.pbc_cache.borrow_mut();
            let result_in_cache = cache.get_mut(&desc_key).ok_or_else(|| {
                TyperError::message(
                    "MultipleFrozenPBCRepr.convert_desc: cached pointer disappeared mid-init",
                )
            })?;
            result_in_cache
                .setattr(&mangled_name, llval)
                .map_err(TyperError::message)?;
        }

        // upstream `return result` — clone out the cached entry as a
        // Constant.
        let cached = self
            .pbc_cache
            .borrow()
            .get(&desc_key)
            .cloned()
            .ok_or_else(|| {
                TyperError::message(
                    "MultipleFrozenPBCRepr.convert_desc: cached pointer missing on return",
                )
            })?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(cached)),
            self.lltype.clone(),
        ))
    }

    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        self.convert_const_multiple_frozen_pbc_base(value)
    }

    fn get_ll_eq_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<crate::translator::rtyper::rtyper::LowLevelFunction>, TyperError> {
        self.get_ll_eq_function_multiple_frozen_pbc_base(rtyper)
    }

    fn rtype_bool(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        crate::translator::rtyper::rmodel::can_be_null_rtype_bool(self, hop)
    }

    /// RPython `MultipleFrozenPBCRepr.rtype_getattr(self, hop)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def rtype_getattr(self, hop):
    ///     if hop.s_result.is_constant():
    ///         return hop.inputconst(hop.r_result, hop.s_result.const)
    ///     attr = hop.args_s[1].const
    ///     vpbc, vattr = hop.inputargs(self, Void)
    ///     v_res = self.getfield(vpbc, attr, hop.llops)
    ///     mangled_name, r_res = self.fieldmap[attr]
    ///     return hop.llops.convertvar(v_res, r_res, hop.r_result)
    /// ```
    fn rtype_getattr(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::{ConvertedTo, HighLevelOp};

        // upstream const-result fast path.
        let const_fast_path: Option<(Arc<dyn Repr>, ConstValue)> = {
            let s_result_borrow = hop.s_result.borrow();
            let r_result_borrow = hop.r_result.borrow();
            match (s_result_borrow.as_ref(), r_result_borrow.as_ref()) {
                (Some(s_result), Some(r_result)) => {
                    s_result.const_().map(|cv| (r_result.clone(), cv.clone()))
                }
                _ => None,
            }
        };
        if let Some((r_result, const_val)) = const_fast_path {
            let c = HighLevelOp::inputconst(ConvertedTo::Repr(r_result.as_ref()), &const_val)?;
            return Ok(Some(Hlvalue::Constant(c)));
        }

        // upstream `attr = hop.args_s[1].const`.
        let attr: String = {
            let args_s = hop.args_s.borrow();
            let s_attr = args_s.get(1).cloned().ok_or_else(|| {
                TyperError::message("MultipleFrozenPBCRepr.rtype_getattr: missing args_s[1]")
            })?;
            match s_attr.const_().cloned() {
                Some(value) => value.as_text().unwrap_or("<non-string>").to_string(),
                other => {
                    return Err(TyperError::message(format!(
                        "MultipleFrozenPBCRepr.rtype_getattr: non-constant attribute name (got {other:?})"
                    )));
                }
            }
        };

        // upstream `vpbc, vattr = hop.inputargs(self, Void)`.
        let v_args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Void),
        ])?;
        let vpbc = v_args[0].clone();

        // upstream `v_res = self.getfield(vpbc, attr, hop.llops)`.
        let v_res = {
            let mut llops = hop.llops.borrow_mut();
            self.getfield(vpbc, &attr, &mut llops)?
        };

        // upstream `mangled_name, r_res = self.fieldmap[attr]; return
        // hop.llops.convertvar(v_res, r_res, hop.r_result)`.
        let r_res = self
            .fieldmap
            .borrow()
            .get(&attr)
            .map(|(_, r)| r.clone())
            .ok_or_else(|| {
                TyperError::message(format!(
                    "MultipleFrozenPBCRepr.rtype_getattr: attr {attr:?} not in fieldmap"
                ))
            })?;
        let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
            TyperError::message("MultipleFrozenPBCRepr.rtype_getattr: hop.r_result missing")
        })?;
        let result = {
            let mut llops = hop.llops.borrow_mut();
            llops.convertvar(Hlvalue::Variable(v_res), r_res.as_ref(), r_result.as_ref())?
        };
        Ok(Some(result))
    }
}

/// RPython `adjust_shape(hop2, s_shape)` (rpbc.py):
///
/// ```python
/// def adjust_shape(hop2, s_shape):
///     new_shape = (s_shape.const[0]+1,) + s_shape.const[1:]
///     c_shape = Constant(new_shape)
///     s_shape = hop2.rtyper.annotator.bookkeeper.immutablevalue(new_shape)
///     hop2.v_s_insertfirstarg(c_shape, s_shape) # reinsert adjusted shape
/// ```
///
/// `s_shape.const` is a `(shape_cnt, shape_keys, shape_star)` tuple
/// constant carried by `call_args` ops; the helper bumps `shape_cnt`
/// by 1 to account for the bound-self arg that
/// [`MethodOfFrozenPBCRepr::redispatch_call`] /
/// [`MethodsPBCRepr::redispatch_call`] (rpbc.py, :1195) prepend.
pub fn adjust_shape(
    hop2: &crate::translator::rtyper::rtyper::HighLevelOp,
    s_shape: &crate::annotator::model::SomeValue,
) -> Result<(), TyperError> {
    use crate::annotator::model::SomeValue;
    use crate::flowspace::model::Constant as FlowConstant;

    // upstream: `s_shape.const` — pull the underlying tuple ConstValue.
    let shape_const = s_shape
        .const_()
        .cloned()
        .ok_or_else(|| TyperError::message("adjust_shape: s_shape is not a Constant"))?;
    let items = match &shape_const {
        ConstValue::Tuple(items) => items,
        other => {
            return Err(TyperError::message(format!(
                "adjust_shape: s_shape.const is not a Tuple: {other:?}"
            )));
        }
    };
    if items.len() != 3 {
        return Err(TyperError::message(
            "adjust_shape: shape tuple must have 3 elements",
        ));
    }
    // upstream: `(s_shape.const[0]+1,) + s_shape.const[1:]`.
    let bumped_cnt = match &items[0] {
        ConstValue::Int(n) => ConstValue::Int(n + 1),
        other => {
            return Err(TyperError::message(format!(
                "adjust_shape: shape_cnt is not Int: {other:?}"
            )));
        }
    };
    let new_shape = ConstValue::Tuple(vec![bumped_cnt, items[1].clone(), items[2].clone()]);
    // upstream: `c_shape = Constant(new_shape)` — Void-typed constant
    // (matches the encoding `bookkeeper::call_shape_from_const` reads).
    let c_shape = Hlvalue::Constant(FlowConstant::with_concretetype(
        new_shape.clone(),
        LowLevelType::Void,
    ));
    // upstream: `s_shape = hop2.rtyper.annotator.bookkeeper.immutablevalue(new_shape)`.
    let bookkeeper = hop2
        .rtyper
        .annotator
        .upgrade()
        .ok_or_else(|| TyperError::message("adjust_shape: annotator weak ref dropped"))?
        .bookkeeper
        .clone();
    let s_new_shape = bookkeeper
        .immutablevalue(&new_shape)
        .map_err(|e| TyperError::message(e.to_string()))?;
    // upstream: `hop2.v_s_insertfirstarg(c_shape, s_shape)`.
    hop2.v_s_insertfirstarg(c_shape, s_new_shape)?;
    // Suppress unused-import warning for SomeValue when only used in
    // the parameter type.
    let _ = SomeValue::Impossible;
    Ok(())
}

/// RPython `class MethodOfFrozenPBCRepr(Repr)` (rpbc.py).
///
/// ```python
/// class MethodOfFrozenPBCRepr(Repr):
///     """Representation selected for a PBC of method object(s) of frozen PBCs.
///     It assumes that all methods are the same function bound to different PBCs.
///     The low-level representation can then be a pointer to that PBC."""
///
///     def __init__(self, rtyper, s_pbc):
///         self.rtyper = rtyper
///         funcdescs = set([desc.funcdesc for desc in s_pbc.descriptions])
///         assert len(funcdescs) == 1
///         self.funcdesc = funcdescs.pop()
///         if s_pbc.can_be_none():
///             raise TyperError(...)
///         im_selves = [desc.frozendesc for desc in s_pbc.descriptions]
///         self.s_im_self = annmodel.SomePBC(im_selves)
///         self.r_im_self = rtyper.getrepr(self.s_im_self)
///         self.lowleveltype = self.r_im_self.lowleveltype
/// ```
#[derive(Debug)]
pub struct MethodOfFrozenPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:850).
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.funcdesc = funcdescs.pop()` (rpbc.py) — the
    /// single shared underlying funcdesc across all bound methods in
    /// `s_pbc`. Carries the whole `FuncDescEntry` so a bound memo method
    /// keeps its MemoDesc identity (upstream `desc.funcdesc` is the
    /// MemoDesc, compared with `is`); `func()` is the FunctionDesc view.
    pub funcdesc: crate::annotator::description::FuncDescEntry,
    /// RPython `self.s_im_self = SomePBC(im_selves)` (rpbc.py:867) —
    /// the PBC of the bound `frozendesc`s.
    pub s_im_self: SomePBC,
    /// RPython `self.r_im_self = rtyper.getrepr(self.s_im_self)`
    /// (rpbc.py:868).
    pub r_im_self: std::sync::Arc<dyn Repr>,
    /// RPython `self.lowleveltype = self.r_im_self.lowleveltype`
    /// (rpbc.py:869) — pointer-to-bound-PBC. Stored explicitly because
    /// `Repr::lowleveltype` returns `&LowLevelType` and we cannot
    /// borrow through `r_im_self`.
    lltype: LowLevelType,
    state: ReprState,
}

impl MethodOfFrozenPBCRepr {
    /// RPython `MethodOfFrozenPBCRepr.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py).
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        // upstream: `funcdescs = set([desc.funcdesc for desc in
        //                              s_pbc.descriptions]); assert
        //            len(funcdescs) == 1; self.funcdesc =
        //            funcdescs.pop()`.
        let mut funcdesc_set: std::collections::BTreeMap<
            crate::annotator::description::DescKey,
            crate::annotator::description::FuncDescEntry,
        > = std::collections::BTreeMap::new();
        let mut frozendescs: Vec<crate::annotator::description::DescEntry> = Vec::new();
        for entry in s_pbc.descriptions.values() {
            let mof = entry.as_method_of_frozen().ok_or_else(|| {
                TyperError::message(
                    "MethodOfFrozenPBCRepr: every description must be a MethodOfFrozenDesc",
                )
            })?;
            let mof_b = mof.borrow();
            // upstream `set([desc.funcdesc ...])` keys on the funcdesc
            // identity — the MemoDesc itself for a memo, not its base.
            let fd_key = mof_b.funcdesc.desc_key();
            funcdesc_set
                .entry(fd_key)
                .or_insert_with(|| mof_b.funcdesc.clone());
            frozendescs.push(crate::annotator::description::DescEntry::Frozen(
                mof_b.frozendesc.clone(),
            ));
        }
        if funcdesc_set.len() != 1 {
            return Err(TyperError::message(
                "MethodOfFrozenPBCRepr: all bound methods must share a single FunctionDesc",
            ));
        }
        let funcdesc = funcdesc_set.into_iter().next().expect("len checked").1;

        // upstream: `if s_pbc.can_be_none(): raise TyperError(...)`.
        if s_pbc.can_be_none {
            return Err(TyperError::message(
                "unsupported: variable of type method-of-frozen-PBC or None",
            ));
        }

        // upstream: `im_selves = [desc.frozendesc for desc in
        //                          s_pbc.descriptions];
        //            self.s_im_self = SomePBC(im_selves);
        //            self.r_im_self = rtyper.getrepr(self.s_im_self);
        //            self.lowleveltype = self.r_im_self.lowleveltype`.
        let s_im_self = SomePBC::new(frozendescs, false);
        let r_im_self =
            rtyper.getrepr(&crate::annotator::model::SomeValue::PBC(s_im_self.clone()))?;
        let lltype = r_im_self.lowleveltype().clone();

        Ok(MethodOfFrozenPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            funcdesc,
            s_im_self,
            r_im_self,
            lltype,
            state: ReprState::new(),
        })
    }

    /// RPython `MethodOfFrozenPBCRepr.convert_desc(self, mdesc)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_desc(self, mdesc):
    ///     if mdesc.funcdesc is not self.funcdesc:
    ///         raise TyperError("not a method bound on %r: %r" % ...)
    ///     return self.r_im_self.convert_desc(mdesc.frozendesc)
    /// ```
    pub fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        let mof = desc.as_method_of_frozen().ok_or_else(|| {
            TyperError::message(
                "MethodOfFrozenPBCRepr.convert_desc: desc is not a MethodOfFrozenDesc",
            )
        })?;
        let mof_b = mof.borrow();
        // upstream: `if mdesc.funcdesc is not self.funcdesc: raise` —
        // identity on the funcdesc (the MemoDesc itself for a memo).
        if mof_b.funcdesc.desc_key() != self.funcdesc.desc_key() {
            return Err(TyperError::message(format!(
                "not a method bound on {:?}: {:?}",
                self.funcdesc.func().borrow().name,
                mof_b.funcdesc.func().borrow().name,
            )));
        }
        // upstream: `return self.r_im_self.convert_desc(mdesc.frozendesc)`.
        let frozen_entry =
            crate::annotator::description::DescEntry::Frozen(mof_b.frozendesc.clone());
        self.r_im_self.convert_desc(&frozen_entry)
    }

    /// RPython `MethodOfFrozenPBCRepr.redispatch_call(self, hop,
    /// call_args)` (rpbc.py).
    ///
    /// ```python
    /// def redispatch_call(self, hop, call_args):
    ///     s_function = annmodel.SomePBC([self.funcdesc])
    ///     hop2 = hop.copy()
    ///     hop2.args_s[0] = self.s_im_self
    ///     hop2.args_r[0] = self.r_im_self
    ///     if isinstance(hop2.args_v[0], Constant):
    ///         boundmethod = hop2.args_v[0].value
    ///         hop2.args_v[0] = Constant(boundmethod.im_self)
    ///     if call_args:
    ///         hop2.swap_fst_snd_args()
    ///         _, s_shape = hop2.r_s_popfirstarg()
    ///         adjust_shape(hop2, s_shape)
    ///     c = Constant("obscure-don't-use-me")
    ///     hop2.v_s_insertfirstarg(c, s_function)
    ///     return hop2.dispatch()
    /// ```
    pub fn redispatch_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
        call_args: bool,
    ) -> Result<Option<Hlvalue>, TyperError> {
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::Constant as FlowConstant;

        // upstream: `s_function = SomePBC([self.funcdesc])` — the memo
        // wrapper survives as the desc, not its unwrapped base.
        let s_function = SomeValue::PBC(SomePBC::new(
            vec![DescEntry::Func(self.funcdesc.clone())],
            false,
        ));
        // upstream: `hop2 = hop.copy()`.
        let hop2 = hop.copy();
        // upstream: `hop2.args_s[0] = self.s_im_self;
        //            hop2.args_r[0] = self.r_im_self`.
        hop2.args_s.borrow_mut()[0] = SomeValue::PBC(self.s_im_self.clone());
        hop2.args_r.borrow_mut()[0] = Some(self.r_im_self.clone());

        // upstream rpbc.py:900-902:
        //     if isinstance(hop2.args_v[0], Constant):
        //         boundmethod = hop2.args_v[0].value
        //         hop2.args_v[0] = Constant(boundmethod.im_self)
        //
        // `hop2.args_v[0].value` is the host-side bound method object;
        // `.im_self` is the receiver instance. Pyre's
        // `HostObject::BoundMethod` carries the receiver as `self_obj`
        // (`flowspace/model.rs`), exposed via `bound_method_self()`. A
        // non-`HostObject` Constant or a `HostObject` that is not a
        // bound method surfaces as a structured TyperError — upstream
        // would AttributeError on `.im_self`.
        let new_first_const: Option<FlowConstant> = {
            let args_v = hop2.args_v.borrow();
            match args_v.first() {
                Some(Hlvalue::Constant(c)) => match &c.value {
                    ConstValue::HostObject(host_obj) => {
                        let self_obj = host_obj.bound_method_self().ok_or_else(|| {
                            TyperError::message(
                                "MethodOfFrozenPBCRepr.redispatch_call: \
                                     Constant arg[0] HostObject is not a bound \
                                     method (no im_self)"
                                    .to_string(),
                            )
                        })?;
                        Some(FlowConstant::new(ConstValue::HostObject(self_obj.clone())))
                    }
                    other => {
                        return Err(TyperError::message(format!(
                            "MethodOfFrozenPBCRepr.redispatch_call: Constant \
                             arg[0] value is not a HostObject (got {other:?})"
                        )));
                    }
                },
                _ => None,
            }
        };
        if let Some(new_const) = new_first_const {
            hop2.args_v.borrow_mut()[0] = Hlvalue::Constant(new_const);
        }

        // upstream: `if call_args: swap_fst_snd_args(); _, s_shape =
        //                          r_s_popfirstarg(); adjust_shape(hop2, s_shape)`.
        if call_args {
            hop2.swap_fst_snd_args();
            let (_r, s_shape) = hop2.r_s_popfirstarg();
            adjust_shape(&hop2, &s_shape)?;
        }

        // upstream: `c = Constant("obscure-don't-use-me")`. Void-typed
        // sentinel — the hop2 path never dereferences it, but the
        // dispatcher needs *some* args_v[0] to feed
        // `hop.inputarg(self_repr, 0)`. Upstream stores no
        // concretetype on this Constant; pyre tags Void to satisfy
        // genop's concretetype assertion downstream.
        let c = Hlvalue::Constant(FlowConstant::with_concretetype(
            ConstValue::byte_str("obscure-don't-use-me"),
            LowLevelType::Void,
        ));
        // upstream: `hop2.v_s_insertfirstarg(c, s_function)`.
        hop2.v_s_insertfirstarg(c, s_function)?;
        // upstream: `return hop2.dispatch()`.
        hop2.dispatch()
    }
}

impl Repr for MethodOfFrozenPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "MethodOfFrozenPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::MethodOfFrozenPBCRepr
    }

    /// RPython `MethodOfFrozenPBCRepr.get_r_implfunc(self)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def get_r_implfunc(self):
    ///     r_func = self.rtyper.getrepr(self.get_s_callable())
    ///     return r_func, 1
    /// ```
    fn get_r_implfunc(&self) -> Result<(Arc<dyn Repr>, usize), TyperError> {
        // upstream: `r_func = self.rtyper.getrepr(self.get_s_callable())`.
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("MethodOfFrozenPBCRepr.get_r_implfunc: rtyper weak ref dropped")
        })?;
        // upstream `get_s_callable` is `FunctionReprBase.get_s_callable`
        // (rpbc.py) — `return self.s_pbc`. For
        // MethodOfFrozenPBCRepr the equivalent is the funcdesc-derived
        // SomePBC: a single-FunctionDesc PBC (no can_be_None).
        let s_callable = SomePBC::new(vec![DescEntry::Func(self.funcdesc.clone())], false);
        let r_func = rtyper.getrepr(&crate::annotator::model::SomeValue::PBC(s_callable))?;
        // upstream: `return r_func, 1` — the `1` is the arg-position
        // offset (skip the bound `self`).
        Ok((r_func, 1))
    }

    /// RPython `MethodOfFrozenPBCRepr.get_s_callable(self)` (rpbc.py):
    /// `return SomePBC([self.funcdesc])`. This is the underlying
    /// function, not the bound-method PBC.
    fn get_s_callable(&self) -> Option<SomePBC> {
        Some(SomePBC::new(
            vec![DescEntry::Func(self.funcdesc.clone())],
            false,
        ))
    }

    /// RPython `MethodOfFrozenPBCRepr.convert_desc` (rpbc.py) —
    /// thin Repr-trait forwarder; the body lives on the inherent impl.
    fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        MethodOfFrozenPBCRepr::convert_desc(self, desc)
    }

    /// RPython `MethodOfFrozenPBCRepr.convert_const(self, method)`
    /// RPython `MethodOfFrozenPBCRepr.convert_const(self, method)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_const(self, method):
    ///     return self.convert_desc(
    ///         self.rtyper.annotator.bookkeeper.getdesc(method))
    /// ```
    ///
    /// `bookkeeper.getdesc` (bookkeeper.rs) recognises
    /// `HostObject::BoundMethod` and returns a
    /// `DescEntry::MethodOfFrozen` when the bound `self` is a frozen PBC
    /// (the only kind this repr accepts) — `convert_desc` then
    /// re-validates the funcdesc identity guard.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let host_obj = match value {
            ConstValue::HostObject(h) => h.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "MethodOfFrozenPBCRepr.convert_const: expected \
                     HostObject(BoundMethod), got {other:?}"
                )));
            }
        };
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("MethodOfFrozenPBCRepr.convert_const: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("MethodOfFrozenPBCRepr.convert_const: annotator weak ref dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(&host_obj)
            .map_err(|e| TyperError::message(e.to_string()))?;
        MethodOfFrozenPBCRepr::convert_desc(self, &desc)
    }

    /// RPython `MethodOfFrozenPBCRepr.rtype_simple_call(self, hop)`
    /// (rpbc.py) — `return self.redispatch_call(hop, call_args=False)`.
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.redispatch_call(hop, false)
    }

    /// RPython `MethodOfFrozenPBCRepr.rtype_call_args(self, hop)`
    /// (rpbc.py) — `return self.redispatch_call(hop, call_args=True)`.
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.redispatch_call(hop, true)
    }
}

/// RPython `class ClassesPBCRepr(Repr)` (rpbc.py).
///
/// PBC repr for a `SomePBC` whose kind is `DescKind::Class`. Upstream:
///
/// ```python
/// def __init__(self, rtyper, s_pbc):
///     self.rtyper = rtyper
///     self.s_pbc = s_pbc
///     if s_pbc.is_constant():
///         self.lowleveltype = Void
///     else:
///         self.lowleveltype = self.getlowleveltype()
///
/// def getlowleveltype(self):
///     return CLASSTYPE
/// ```
///
/// Both arms of `__init__` are now ported; `getlowleveltype()` returns
/// `CLASSTYPE` directly and is independent of `init_vtable`. The
/// downstream `convert_desc` non-Void path still routes to
/// `getruntime` which is blocked on `ClassRepr.init_vtable`.
#[derive(Debug)]
pub struct ClassesPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:924). Weak backref.
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.s_pbc = s_pbc` (rpbc.py:925).
    pub s_pbc: SomePBC,
    state: ReprState,
    /// RPython `self.lowleveltype` — `Void` on the constant branch,
    /// `CLASSTYPE` (Ptr(OBJECT_VTABLE)) otherwise via
    /// [`Self::getlowleveltype`].
    lltype: LowLevelType,
}

fn object_vtable_field_type(
    name: &str,
) -> Result<crate::translator::rtyper::lltypesystem::lltype::LowLevelType, TyperError> {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::rclass::OBJECT_VTABLE;

    let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
        return Err(TyperError::message(
            "OBJECT_VTABLE must be a ForwardReference",
        ));
    };
    let LowLevelType::Struct(body) = fwd
        .resolved()
        .ok_or_else(|| TyperError::message("OBJECT_VTABLE forward reference is unresolved"))?
    else {
        return Err(TyperError::message(
            "OBJECT_VTABLE must resolve to a Struct",
        ));
    };
    body._flds
        .get(name)
        .cloned()
        .ok_or_else(|| TyperError::message(format!("OBJECT_VTABLE has no field {name:?}")))
}

/// RPython `pairtype(ClassesPBCRepr, ClassRepr).convert_from_to` (rpbc.py).
///
/// `RootClassRepr` is a `ClassRepr` upstream. Both pyre class reprs use
/// [`ReprClassId::Repr`], so the dispatcher selects this handler only
/// when the target's concrete type is `ClassRepr` or `RootClassRepr`.
pub(crate) fn pair_classes_pbc_class_convert_from_to(
    r_from: &dyn crate::translator::rtyper::rmodel::Repr,
    r_to: &dyn crate::translator::rtyper::rmodel::Repr,
    v: &crate::flowspace::model::Hlvalue,
    llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    use crate::flowspace::model::Hlvalue;
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::rmodel::{Repr, inputconst};
    use crate::translator::rtyper::rtyper::GenopResult;

    let raw = r_from as *const dyn Repr as *const ();
    let r_clspbc = unsafe { &*(raw as *const ClassesPBCRepr) };
    if r_clspbc.lowleveltype() == r_to.lowleveltype() {
        return Ok(Some(v.clone()));
    }
    if matches!(r_clspbc.lowleveltype(), LowLevelType::Void) {
        let value = r_clspbc.s_pbc.base.const_box.as_ref().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr→ClassRepr: Void class PBC has no constant")
        })?;
        let constant = inputconst(r_to, &value.value)?;
        return Ok(Some(Hlvalue::Constant(constant)));
    }
    Repr::setup(r_to)?;
    let casted = llops.genop(
        "cast_pointer",
        vec![v.clone()],
        GenopResult::LLType(r_to.lowleveltype().clone()),
    );
    Ok(casted.map(Hlvalue::Variable))
}

impl ClassesPBCRepr {
    /// RPython `ClassesPBCRepr.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py). Both constant and non-constant arms ported.
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        // upstream rpbc.py — `if s_pbc.is_constant():`. The pyre
        // `SomePBC::is_constant` mirrors model.py which only
        // sets `const_box` when `len==1 && !can_be_none &&
        // desc.pyobj is not None`, so a single-desc PBC whose
        // ClassDesc has no live pyobj falls through to the non-constant
        // arm.
        let lltype = if s_pbc.is_constant() {
            LowLevelType::Void
        } else {
            // upstream rpbc.py — `self.lowleveltype = self.getlowleveltype()`.
            Self::getlowleveltype()
        };
        Ok(ClassesPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            s_pbc,
            state: ReprState::new(),
            lltype,
        })
    }

    /// RPython `ClassesPBCRepr.getlowleveltype(self)` (rpbc.py).
    ///
    /// ```python
    /// def getlowleveltype(self):
    ///     return CLASSTYPE
    /// ```
    pub fn getlowleveltype() -> LowLevelType {
        crate::translator::rtyper::rclass::CLASSTYPE.clone()
    }

    /// RPython `ClassesPBCRepr.get_access_set(self, attrname)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def get_access_set(self, attrname):
    ///     classdescs = list(self.s_pbc.descriptions)
    ///     access = classdescs[0].queryattrfamily(attrname)
    ///     for classdesc in classdescs[1:]:
    ///         access1 = classdesc.queryattrfamily(attrname)
    ///         assert access1 is access       # XXX not implemented
    ///     if access is None:
    ///         raise rclass.MissingRTypeAttribute(attrname)
    ///     commonbase = access.commonbase
    ///     class_repr = rclass.getclassrepr(self.rtyper, commonbase)
    ///     return access, class_repr
    /// ```
    ///
    /// Returns the `(ClassAttrFamily, ClassRepr)` pair that hosts the
    /// shared vtable slot for `attrname`. The `ClassAttrFamily.commonbase`
    /// is populated by
    /// [`crate::translator::rtyper::normalizecalls::merge_classpbc_getattr_into_classdef`];
    /// missing means `merge_classpbc_getattr_into_classdef` was either
    /// never run or skipped this family because it has fewer than two
    /// distinct ClassDescs (single-class PBCs read attrs through their
    /// concrete `ClassRepr` directly, not through this dispatch).
    pub fn get_access_set(
        &self,
        attrname: &str,
    ) -> Result<
        (
            Rc<RefCell<crate::annotator::description::ClassAttrFamily>>,
            crate::translator::rtyper::rclass::ClassReprArc,
        ),
        TyperError,
    > {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.get_access_set: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.get_access_set: annotator weak ref dropped")
        })?;

        // upstream `classdescs = list(self.s_pbc.descriptions)`. Pyre
        // stores `descriptions: BTreeMap<DescKey, DescEntry>` so
        // iteration is deterministic — the "first" desc is therefore
        // well-defined for the assert-all-equal walk below.
        let mut class_descs: Vec<Rc<RefCell<crate::annotator::classdesc::ClassDesc>>> = Vec::new();
        for entry in self.s_pbc.descriptions.values() {
            if let DescEntry::Class(rc) = entry {
                class_descs.push(rc.clone());
            } else {
                return Err(TyperError::message(format!(
                    "ClassesPBCRepr.get_access_set: non-Class desc {entry:?} in s_pbc",
                )));
            }
        }
        let Some((first, rest)) = class_descs.split_first() else {
            return Err(TyperError::message(
                "ClassesPBCRepr.get_access_set: empty descriptions",
            ));
        };

        // upstream `access = classdescs[0].queryattrfamily(attrname); for
        // classdesc in classdescs[1:]: access1 = classdesc.queryattrfamily(attrname);
        // assert access1 is access`.
        let access = crate::annotator::classdesc::ClassDesc::queryattrfamily(first, attrname);
        for other in rest {
            let other_access =
                crate::annotator::classdesc::ClassDesc::queryattrfamily(other, attrname);
            match (&access, &other_access) {
                (Some(a), Some(b)) if Rc::ptr_eq(a, b) => {}
                _ => {
                    return Err(TyperError::message(format!(
                        "ClassesPBCRepr.get_access_set: descs disagree on \
                         attrfamily for {attrname:?} — upstream marks this \
                         path as 'XXX not implemented'"
                    )));
                }
            }
        }
        // upstream `if access is None: raise rclass.MissingRTypeAttribute(attrname)`.
        let access = access.ok_or_else(|| {
            TyperError::missing_rtype_operation(format!(
                "ClassesPBCRepr.get_access_set: no access set for attribute \
                 {attrname:?} (rclass.MissingRTypeAttribute)"
            ))
        })?;

        // upstream `commonbase = access.commonbase`. Populated by
        // `normalizecalls.merge_classpbc_getattr_into_classdef`
        // (normalizecalls.py).
        let commonbase = access.borrow().commonbase.clone().ok_or_else(|| {
            TyperError::message(
                "ClassesPBCRepr.get_access_set: ClassAttrFamily.commonbase missing — \
                     merge_classpbc_getattr_into_classdef must run before rtyping",
            )
        })?;

        // upstream `class_repr = rclass.getclassrepr(self.rtyper, commonbase)`.
        let class_repr =
            crate::translator::rtyper::rclass::getclassrepr_arc(&rtyper, Some(&commonbase))?;
        let _ = annotator; // silence unused after future-proofing the upgrade() guard.
        Ok((access, class_repr))
    }

    /// RPython `ClassesPBCRepr._instantiate_runtime_class` (rpbc.py).
    ///
    /// ```python
    /// def _instantiate_runtime_class(self, hop, vtypeptr, r_instance):
    ///     graphs = []
    ///     for desc in self.s_pbc.descriptions:
    ///         classdef = desc.getclassdef(None)
    ///         assert hasattr(classdef, 'my_instantiate_graph')
    ///         graphs.append(classdef.my_instantiate_graph)
    ///     c_graphs = hop.inputconst(Void, graphs)
    ///     c_name = hop.inputconst(Void, 'instantiate')
    ///     v_instantiate = hop.genop('getfield', [vtypeptr, c_name],
    ///                              resulttype=OBJECT_VTABLE.instantiate)
    ///     v_inst = hop.genop('indirect_call', [v_instantiate, c_graphs],
    ///                        resulttype=OBJECTPTR)
    ///     return hop.genop('cast_pointer', [v_inst], resulttype=r_instance)
    /// ```
    ///
    /// `r_instance` is the low-level result type. `rtype_instantiate`
    /// passes `hop.r_result.lowleveltype`.
    pub fn _instantiate_runtime_class(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
        vtypeptr: crate::flowspace::model::Hlvalue,
        r_instance: &LowLevelType,
    ) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        use crate::annotator::classdesc::ClassDesc;
        use crate::flowspace::model::{ConstValue, GraphKey, Hlvalue};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{GenopResult, HighLevelOp};

        let mut graph_ids = Vec::new();
        for entry in self.s_pbc.descriptions.values() {
            let DescEntry::Class(desc) = entry else {
                return Err(TyperError::message(format!(
                    "ClassesPBCRepr._instantiate_runtime_class: non-Class desc {entry:?}"
                )));
            };
            let classdef =
                ClassDesc::getclassdef(desc, ()).map_err(|e| TyperError::message(e.to_string()))?;
            let graph = classdef
                .borrow()
                .my_instantiate_graph
                .clone()
                .ok_or_else(|| {
                    TyperError::message(
                        "ClassesPBCRepr._instantiate_runtime_class: classdef has no \
                         my_instantiate_graph",
                    )
                })?;
            graph_ids.push(GraphKey::of(&graph).as_usize());
        }
        let c_graphs =
            HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::Graphs(graph_ids))?;
        let c_name =
            HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::byte_str("instantiate"))?;
        let instantiate_type = object_vtable_field_type("instantiate")?;
        let v_instantiate = hop
            .genop(
                "getfield",
                vec![vtypeptr, Hlvalue::Constant(c_name)],
                GenopResult::LLType(instantiate_type),
            )
            .ok_or_else(|| {
                TyperError::message(
                    "ClassesPBCRepr._instantiate_runtime_class: getfield returned no result",
                )
            })?;
        let v_inst = hop
            .genop(
                "indirect_call",
                vec![v_instantiate, Hlvalue::Constant(c_graphs)],
                GenopResult::LLType(crate::translator::rtyper::rclass::OBJECTPTR.clone()),
            )
            .ok_or_else(|| {
                TyperError::message(
                    "ClassesPBCRepr._instantiate_runtime_class: indirect_call returned no result",
                )
            })?;
        hop.genop(
            "cast_pointer",
            vec![v_inst],
            GenopResult::LLType(r_instance.clone()),
        )
        .ok_or_else(|| {
            TyperError::message(
                "ClassesPBCRepr._instantiate_runtime_class: cast_pointer returned no result",
            )
        })
    }

    /// RPython `ClassesPBCRepr.replace_class_with_inst_arg(self, hop,
    /// v_inst, s_inst, call_args)` (rpbc.py):
    ///
    /// ```python
    /// def replace_class_with_inst_arg(self, hop, v_inst, s_inst, call_args):
    ///     hop2 = hop.copy()
    ///     hop2.r_s_popfirstarg()   # discard the class pointer argument
    ///     if call_args:
    ///         _, s_shape = hop2.r_s_popfirstarg() # temporarely remove shape
    ///         hop2.v_s_insertfirstarg(v_inst, s_inst)  # add 'instance'
    ///         adjust_shape(hop2, s_shape)
    ///     else:
    ///         hop2.v_s_insertfirstarg(v_inst, s_inst)  # add 'instance'
    ///     return hop2
    /// ```
    fn replace_class_with_inst_arg(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
        v_inst: crate::flowspace::model::Hlvalue,
        s_inst: crate::annotator::model::SomeValue,
        call_args: bool,
    ) -> Result<crate::translator::rtyper::rtyper::HighLevelOp, TyperError> {
        // upstream: `hop2 = hop.copy(); hop2.r_s_popfirstarg()`.
        let hop2 = hop.copy();
        hop2.r_s_popfirstarg();
        if call_args {
            let (_, s_shape) = hop2.r_s_popfirstarg();
            hop2.v_s_insertfirstarg(v_inst, s_inst)?;
            adjust_shape(&hop2, &s_shape)?;
        } else {
            hop2.v_s_insertfirstarg(v_inst, s_inst)?;
        }
        Ok(hop2)
    }

    /// RPython `ClassesPBCRepr.redispatch_call(self, hop, call_args)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def redispatch_call(self, hop, call_args):
    ///     s_instance = hop.s_result
    ///     r_instance = hop.r_result
    ///     if len(self.s_pbc.descriptions) == 1:
    ///         if self.lowleveltype is not Void:
    ///             assert 0, "XXX None-or-1-class instantation not implemented"
    ///         assert isinstance(s_instance, annmodel.SomeInstance)
    ///         classdef = s_instance.classdef
    ///         s_init = classdef.classdesc.s_read_attribute('__init__')
    ///         v_init = Constant("init-func-dummy")   # this value not really used
    ///         if (isinstance(s_init, annmodel.SomeImpossibleValue) and
    ///             classdef.classdesc.is_exception_class() and
    ///             classdef.has_no_attrs()):
    ///             r_instance = rclass.getinstancerepr(hop.rtyper, classdef)
    ///             example = r_instance.get_reusable_prebuilt_instance()
    ///             hop.exception_cannot_occur()
    ///             return hop.inputconst(r_instance.lowleveltype, example)
    ///         v_instance = rclass.rtype_new_instance(hop.rtyper, classdef,
    ///                                                hop.llops, hop)
    ///         if isinstance(v_instance, tuple):
    ///             v_instance, must_call_init = v_instance
    ///             if not must_call_init:
    ///                 return v_instance
    ///     else:
    ///         ...                         # multiple possible classes
    ///     if isinstance(s_init, annmodel.SomeImpossibleValue):
    ///         assert hop.nb_args == 1, "arguments passed to __init__, but no __init__!"
    ///         hop.exception_cannot_occur()
    ///     else:
    ///         ...                         # dispatch to __init__
    ///     return v_instance
    /// ```
    ///
    /// The single-class `Void` arm is ported, including a class that has
    /// `__init__` (`replace_class_with_inst_arg` then `hop2.dispatch`).
    /// The multi-class arm and the non-`Void` single-class arm (upstream
    /// `assert 0`) raise a descriptive error citing the lines they cover.
    pub fn redispatch_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
        call_args: bool,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::model::SomeValue;

        // Shapes outside the ported single-class / Void / no-`__init__` /
        // fieldless / numbered / unit-ctor envelope return an
        // "unimplemented operation" error so the dual-gate classifies
        // them as known-unported (`cutover.rs`'s `is_known_unported`) and
        // falls back to the legacy walker — exactly the graceful skip the
        // base `Repr::rtype_simple_call` produced before this override.
        let unported = |reason: &str| -> TyperError {
            TyperError::missing_rtype_operation(format!(
                "unimplemented operation: 'simple_call' on <ClassesPBCRepr Void> — {reason}",
            ))
        };

        // upstream `if len(self.s_pbc.descriptions) == 1:`.
        if self.s_pbc.descriptions.len() != 1 {
            return Err(unported("multi-class instantiation (rpbc.py:1043-1056)"));
        }
        // upstream `if self.lowleveltype is not Void: assert 0, "XXX
        // None-or-1-class instantation not implemented"`.
        if !matches!(self.lltype, LowLevelType::Void) {
            return Err(unported(
                "single non-Void class instantiation (rpbc.py:1037-1038)",
            ));
        }
        // upstream `s_instance = hop.s_result` then
        // `assert isinstance(s_instance, annmodel.SomeInstance);
        // classdef = s_instance.classdef`.
        let s_instance = hop
            .s_result
            .borrow()
            .clone()
            .ok_or_else(|| unported("s_result is not a SomeInstance"))?;
        let classdef = match &s_instance {
            SomeValue::Instance(si) => match &si.classdef {
                Some(cd) => cd.clone(),
                None => return Err(unported("object-only instance (classdef=None)")),
            },
            _ => return Err(unported("s_result is not a SomeInstance")),
        };
        // upstream `s_init = classdef.classdesc.s_read_attribute('__init__')`.
        let classdesc = classdef.borrow().classdesc.clone();
        let s_init = ClassDesc::s_read_attribute(&classdesc, "__init__")
            .map_err(|e| TyperError::message(e.to_string()))?;
        let init_is_impossible = matches!(s_init, SomeValue::Impossible);

        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.redispatch_call: rtyper weak ref dropped")
        })?;

        // Ported envelope gate: `classdef.minid` unset ⇒
        // `assign_inheritance_ids` has not
        //    numbered this class, so `getvtable` (rclass.py) cannot
        //    bake the subclass range.  The session prologue numbers only
        //    the struct-root + standard-exception prefix; enum-variant
        //    transparent-ctor classes are interned lazily by the flowspace
        //    adapter AFTER that pass, so they reach here unnumbered.  Since
        //    `assign_inheritance_ids` is now append-stable (skip-if-numbered
        //    + append-only), number such a class on demand below.
        // Field-bearing classes route through `new_instance`'s
        // class-default initialisation loop (rclass.py) like
        // any other; a field without a class-level default keeps the
        // malloc zero-init.
        if classdef.borrow().minid.is_none() {
            // Number this mid-session-minted classdef on demand.
            // `assign_inheritance_ids` is append-stable, so re-running it
            // appends this leaf's bracket after the existing maximum
            // without shifting any already-baked prefix id.  A classdef
            // that is not append-safe (a fresh subclass of an
            // already-numbered class) stays unnumbered and skips below.
            let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
                TyperError::message("ClassesPBCRepr.redispatch_call: annotator weak ref dropped")
            })?;
            crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&annotator);
            if classdef.borrow().minid.is_none() {
                return Err(unported("class not numbered (assign_inheritance_ids)"));
            }
        }

        // upstream `ClassesPBCRepr.redispatch_call` — a class with
        // `__init__` allocates, then `hop2.dispatch()` runs the init
        // function as `simple_call(initfunc, instance, args...)`.
        if !init_is_impossible {
            let v_instance = {
                let mut llops = hop.llops.borrow_mut();
                crate::translator::rtyper::rclass::rtype_new_instance(
                    &rtyper,
                    Some(&classdef),
                    &mut llops,
                    Some(hop),
                    false,
                )?
            };
            // upstream `v_init = Constant("init-func-dummy")` — the
            // value is not read; `inputarg` re-converts it through the
            // init function's repr.
            let v_init = crate::flowspace::model::Hlvalue::Constant(
                crate::flowspace::model::Constant::with_concretetype(
                    crate::flowspace::model::ConstValue::byte_str("init-func-dummy"),
                    crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
                ),
            );
            let hop2 =
                self.replace_class_with_inst_arg(hop, v_instance.clone(), s_instance, call_args)?;
            hop2.v_s_insertfirstarg(v_init, s_init)?;
            let s_none = crate::annotator::model::s_none();
            let r_none = rtyper.getrepr(&s_none)?;
            *hop2.s_result.borrow_mut() = Some(s_none);
            *hop2.r_result.borrow_mut() = Some(r_none);
            hop2.dispatch()?;
            return Ok(Some(v_instance));
        }

        // upstream rpbc.py:1024-1035 — simple built-in exception special
        // case: no `__init__`, an exception class, no attrs ⇒ return the
        // same prebuilt instance and ignore any constructor arguments.
        if classdesc.borrow().is_exception_class() && ClassDef::has_no_attrs(&classdef) {
            let r_instance = crate::translator::rtyper::rclass::getinstancerepr(
                &rtyper,
                Some(&classdef),
                crate::translator::rtyper::rclass::Flavor::Gc,
            )?;
            let example = r_instance.get_reusable_prebuilt_instance()?;
            hop.exception_cannot_occur()?;
            let cst = Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(example)),
                r_instance.lowleveltype().clone(),
            );
            return Ok(Some(Hlvalue::Constant(cst)));
        }

        // upstream rpbc.py:1049-1050 — `assert hop.nb_args == 1,
        // "arguments passed to __init__, but no __init__!"`.  Upstream
        // asserts after `rtype_new_instance`; the graceful-skip port
        // checks before so a skipped shape leaves no ops behind.  The
        // gate must stay BELOW the exception shortcut above — upstream
        // returns the prebuilt instance and "ignore[s] any arguments
        // passed to the constructor" before reaching this assert, so
        // exception constructors called with arguments are supported.
        if hop.nb_args() != 1 {
            return Err(unported("constructor arguments with no __init__"));
        }

        // upstream `v_instance = rclass.rtype_new_instance(hop.rtyper,
        // classdef, hop.llops, hop)`.
        let v_instance = {
            let mut llops = hop.llops.borrow_mut();
            crate::translator::rtyper::rclass::rtype_new_instance(
                &rtyper,
                Some(&classdef),
                &mut llops,
                Some(hop),
                false,
            )?
        };
        // upstream rpbc.py — no `__init__` ⇒ the malloc cannot
        // raise.
        hop.exception_cannot_occur()?;
        // upstream `return v_instance`.
        Ok(Some(v_instance))
    }
}

/// RPython `ll_cls_hash(cls)` (rpbc.py).
pub fn ll_cls_hash(cls: &_ptr) -> Result<i64, TyperError> {
    crate::translator::rtyper::lltypesystem::lltype::cast_ptr_to_int(cls)
        .map_err(TyperError::message)
}

impl Repr for ClassesPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ClassesPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::ClassesPBCRepr
    }

    /// RPython `ClassesPBCRepr.convert_desc(self, desc)` (rpbc.py:
    /// :950-957).
    ///
    /// ```python
    /// def convert_desc(self, desc):
    ///     if desc not in self.s_pbc.descriptions:
    ///         raise TyperError("%r not in %r" % (desc, self))
    ///     if self.lowleveltype is Void:
    ///         return None
    ///     subclassdef = desc.getuniqueclassdef()
    ///     r_subclass = rclass.getclassrepr(self.rtyper, subclassdef)
    ///     return r_subclass.getruntime(self.lowleveltype)
    /// ```
    ///
    /// Both Void and non-Void arms ported. The non-Void arm walks
    /// `desc.getuniqueclassdef()` → `getclassrepr_arc(rtyper, classdef)`
    /// → `ClassReprArc::getruntime(self.lowleveltype)` to materialise
    /// the vtable pointer constant.
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        if !self.s_pbc.descriptions.contains_key(&desc.desc_key()) {
            return Err(TyperError::message(format!(
                "ClassesPBCRepr.convert_desc: {desc:?} not in {:?}",
                self.s_pbc
            )));
        }
        if matches!(self.lltype, LowLevelType::Void) {
            return Ok(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            ));
        }
        // upstream: `subclassdef = desc.getuniqueclassdef();
        //           r_subclass = rclass.getclassrepr(self.rtyper, subclassdef);
        //           return r_subclass.getruntime(self.lowleveltype)`.
        let DescEntry::Class(class_rc) = desc else {
            return Err(TyperError::message(format!(
                "ClassesPBCRepr.convert_desc: non-Class desc {desc:?} reached the \
                 non-Void arm",
            )));
        };
        let subclassdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc)
            .map_err(|e| TyperError::message(e.to_string()))?;
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.convert_desc: rtyper weak ref dropped")
        })?;
        let r_subclass =
            crate::translator::rtyper::rclass::getclassrepr_arc(&rtyper, Some(&subclassdef))?;
        let vtable_ptr = r_subclass.getruntime(&self.lltype)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(vtable_ptr)),
            self.lltype.clone(),
        ))
    }

    /// RPython `ClassesPBCRepr.convert_const(self, cls)` (rpbc.py).
    ///
    /// ```python
    /// def convert_const(self, cls):
    ///     if cls is None:
    ///         if self.lowleveltype is Void:
    ///             return None
    ///         else:
    ///             T = self.lowleveltype
    ///             return nullptr(T.TO)
    ///     bk = self.rtyper.annotator.bookkeeper
    ///     classdesc = bk.getdesc(cls)
    ///     return self.convert_desc(classdesc)
    /// ```
    ///
    /// All three arms ported:
    /// - `cls is None` + `Void` → `Constant(None, Void)`
    /// - `cls is None` + non-Void → `Constant(LLPtr(_defl), Ptr(...))`
    /// - non-None → `bk.getdesc(cls); self.convert_desc(desc)` —
    ///   `convert_desc` walks `getuniqueclassdef` → `getclassrepr_arc`
    ///   → `getruntime(self.lowleveltype)` to materialise the vtable
    ///   pointer constant.
    ///   RPython `ClassesPBCRepr.rtype_getattr(self, hop)`:
    ///
    /// ```python
    /// def rtype_getattr(self, hop):
    ///     if hop.s_result.is_constant():
    ///         return hop.inputconst(hop.r_result, hop.s_result.const)
    ///     else:
    ///         attr = hop.args_s[1].const
    ///         if attr == '__name__':
    ///             from rpython.rtyper.lltypesystem import rstr
    ///             class_repr = self.rtyper.rootclass_repr
    ///             vcls, vattr = hop.inputargs(class_repr, Void)
    ///             cname = inputconst(Void, 'name')
    ///             return hop.genop('getfield', [vcls, cname],
    ///                              resulttype = Ptr(rstr.STR))
    ///         access_set, class_repr = self.get_access_set(attr)
    ///         vcls, vattr = hop.inputargs(class_repr, Void)
    ///         v_res = class_repr.getpbcfield(vcls, access_set, attr, hop.llops)
    ///         s_res = access_set.s_value
    ///         r_res = self.rtyper.getrepr(s_res)
    ///         return hop.llops.convertvar(v_res, r_res, hop.r_result)
    /// ```
    ///
    /// The `__name__` arm is ported line-by-line: it pulls the rooted
    /// `ClassRepr` from `rtyper.rootclass_repr`, calls
    /// `hop.inputargs(class_repr, Void)` for the (`vcls`, `vattr`)
    /// pair, and emits the upstream `getfield` op into `hop.llops`
    /// with `STRPTR` (= `Ptr(rstr.STR)`) as the result type. The
    /// downstream non-`__name__` access-set arm reuses the same
    /// `getpbcfield` shape via `class_repr.get_access_set`.
    fn rtype_getattr(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rclass::ClassReprArc;
        use crate::translator::rtyper::rtyper::{ConvertedTo, HighLevelOp};

        // upstream `if hop.s_result.is_constant(): return hop.inputconst(hop.r_result, hop.s_result.const)`.
        let const_fast_path: Option<(Arc<dyn Repr>, ConstValue)> = {
            let s_result_borrow = hop.s_result.borrow();
            let r_result_borrow = hop.r_result.borrow();
            match (s_result_borrow.as_ref(), r_result_borrow.as_ref()) {
                (Some(s_result), Some(r_result)) => {
                    s_result.const_().map(|cv| (r_result.clone(), cv.clone()))
                }
                _ => None,
            }
        };
        if let Some((r_result, const_val)) = const_fast_path {
            let c = HighLevelOp::inputconst(ConvertedTo::Repr(r_result.as_ref()), &const_val)?;
            return Ok(Some(Hlvalue::Constant(c)));
        }

        // upstream `attr = hop.args_s[1].const`.
        let attr: String = {
            let args_s = hop.args_s.borrow();
            let s_attr = args_s.get(1).cloned().ok_or_else(|| {
                TyperError::message(
                    "ClassesPBCRepr.rtype_getattr: missing attribute argument args_s[1]",
                )
            })?;
            match s_attr.const_().cloned() {
                Some(value) => value.as_text().unwrap_or("<non-string>").to_string(),
                other => {
                    return Err(TyperError::message(format!(
                        "ClassesPBCRepr.rtype_getattr: non-constant attribute name \
                         (got {other:?})"
                    )));
                }
            }
        };

        // upstream `if attr == '__name__':
        //     class_repr = self.rtyper.rootclass_repr
        //     vcls, vattr = hop.inputargs(class_repr, Void)
        //     cname = inputconst(Void, 'name')
        //     return hop.genop('getfield', [vcls, cname], resulttype=Ptr(rstr.STR))`.
        if attr == "__name__" {
            let rtyper = self.rtyper.upgrade().ok_or_else(|| {
                TyperError::message(
                    "ClassesPBCRepr.rtype_getattr('__name__'): rtyper weak ref dropped",
                )
            })?;
            let root_repr = rtyper.rootclass_repr.borrow().clone().ok_or_else(|| {
                TyperError::message(
                    "ClassesPBCRepr.rtype_getattr('__name__'): rootclass_repr is None — \
                         call RPythonTyper::initialize_exceptiondata first",
                )
            })?;
            let root_repr_arc: Arc<dyn Repr> = root_repr.clone();
            let v_args = hop.inputargs(vec![
                ConvertedTo::Repr(root_repr_arc.as_ref()),
                ConvertedTo::LowLevelType(&LowLevelType::Void),
            ])?;
            let vcls = v_args[0].clone();
            let cname =
                Constant::with_concretetype(ConstValue::byte_str("name"), LowLevelType::Void);
            let strptr = crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone();
            let v_res = {
                let mut llops = hop.llops.borrow_mut();
                llops
                    .genop(
                        "getfield",
                        vec![vcls, Hlvalue::Constant(cname)],
                        crate::translator::rtyper::rtyper::GenopResult::LLType(strptr),
                    )
                    .expect("getfield with non-Void result yields a Variable")
            };
            return Ok(Some(Hlvalue::Variable(v_res)));
        }

        // upstream `access_set, class_repr = self.get_access_set(attr)`.
        let (access_set, class_repr) = self.get_access_set(&attr)?;
        let class_repr_inst = match &class_repr {
            ClassReprArc::Inst(inst) => inst.clone(),
            ClassReprArc::Root(_) => {
                return Err(TyperError::message(
                    "ClassesPBCRepr.rtype_getattr: commonbase resolved to RootClassRepr — \
                     unexpected; root has no pbcfields",
                ));
            }
        };
        let class_repr_arc: Arc<dyn Repr> = class_repr.as_repr();

        // upstream `vcls, vattr = hop.inputargs(class_repr, Void)`.
        let v_args = hop.inputargs(vec![
            ConvertedTo::Repr(class_repr_arc.as_ref()),
            ConvertedTo::LowLevelType(&LowLevelType::Void),
        ])?;
        let vcls = v_args[0].clone();

        // upstream `v_res = class_repr.getpbcfield(vcls, access_set, attr, hop.llops)`.
        let access_set_id = Rc::as_ptr(&access_set) as usize;
        let v_res = {
            let mut llops = hop.llops.borrow_mut();
            class_repr_inst.getpbcfield(vcls, access_set_id, &attr, &mut llops)?
        };

        // upstream `s_res = access_set.s_value; r_res = self.rtyper.getrepr(s_res)`.
        let s_res = access_set.borrow().s_value.clone();
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.rtype_getattr: rtyper weak ref dropped")
        })?;
        let r_res = rtyper.getrepr(&s_res)?;

        // upstream `return hop.llops.convertvar(v_res, r_res, hop.r_result)`.
        let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
            TyperError::message(
                "ClassesPBCRepr.rtype_getattr: hop.r_result missing on the non-const arm",
            )
        })?;
        let result = {
            let mut llops = hop.llops.borrow_mut();
            llops.convertvar(Hlvalue::Variable(v_res), r_res.as_ref(), r_result.as_ref())?
        };
        Ok(Some(result))
    }

    /// RPython `ClassesPBCRepr.rtype_simple_call(self, hop)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def rtype_simple_call(self, hop):
    ///     return self.redispatch_call(hop, call_args=False)
    /// ```
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.redispatch_call(hop, false)
    }

    /// RPython `ClassesPBCRepr.rtype_call_args(self, hop)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def rtype_call_args(self, hop):
    ///     return self.redispatch_call(hop, call_args=True)
    /// ```
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.redispatch_call(hop, true)
    }

    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if matches!(value, ConstValue::None) {
            // upstream `if self.lowleveltype is Void: return None`.
            if matches!(self.lltype, LowLevelType::Void) {
                return Ok(Constant::with_concretetype(
                    ConstValue::None,
                    LowLevelType::Void,
                ));
            }
            // upstream `T = self.lowleveltype; return nullptr(T.TO)`.
            let LowLevelType::Ptr(ptr) = &self.lltype else {
                return Err(TyperError::message(format!(
                    "ClassesPBCRepr.convert_const: lowleveltype is neither Void nor Ptr, got {:?}",
                    self.lltype,
                )));
            };
            let null_ptr = ptr.as_ref().clone()._defl();
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(null_ptr)),
                self.lltype.clone(),
            ));
        }
        // upstream `bk = self.rtyper.annotator.bookkeeper;
        //          classdesc = bk.getdesc(cls);
        //          return self.convert_desc(classdesc)`.
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "ClassesPBCRepr.convert_const: non-None value must be a HostObject \
                 carrying a class, got {value:?}",
            )));
        };
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.convert_const: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.convert_const: annotator weak ref dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }
}

/// RPython `getFrozenPBCRepr(rtyper, s_pbc)` (rpbc.py).
///
/// ```python
/// def getFrozenPBCRepr(rtyper, s_pbc):
///     descs = list(s_pbc.descriptions)
///     assert len(descs) >= 1
///     if len(descs) == 1 and not s_pbc.can_be_None:
///         return SingleFrozenPBCRepr(descs[0])
///     else:
///         access = descs[0].queryattrfamily()
///         for desc in descs[1:]:
///             access1 = desc.queryattrfamily()
///             if access1 is not access:
///                 try:
///                     return rtyper.pbc_reprs['unrelated']
///                 except KeyError:
///                     result = MultipleUnrelatedFrozenPBCRepr(rtyper)
///                     rtyper.pbc_reprs['unrelated'] = result
///                     return result
///         try:
///             return rtyper.pbc_reprs[access]
///         except KeyError:
///             result = MultipleFrozenPBCRepr(rtyper, access)
///             rtyper.pbc_reprs[access] = result
///             rtyper.add_pendingsetup(result)
///             return result
/// ```
///
/// Pyre's port handles:
/// * single-desc + `!can_be_None` → [`SingleFrozenPBCRepr`],
/// * multi-desc with any two descs whose `queryattrfamily` differ →
///   [`MultipleUnrelatedFrozenPBCRepr`] keyed in
///   `rtyper.pbc_reprs[PbcReprKey::Unrelated]` (singleton cache,
///   matching upstream's `'unrelated'` dict slot),
/// * multi-desc with a common `Some(access)` set →
///   [`MultipleFrozenPBCRepr`] keyed in
///   `rtyper.pbc_reprs[PbcReprKey::Access(access_key)]`, registered
///   via `rtyper.add_pendingsetup`,
/// * multi-desc whose common access set is `None` (every desc has no
///   attrfamily) → [`MultipleFrozenPBCRepr`] with `access_set = None`,
///   a zero-field `Struct('pbc')` keyed in
///   `rtyper.pbc_reprs[PbcReprKey::AccessNone]` (rpbc.py + 758).
///
/// Called from both `somepbc_rtyper_makerepr` DescKind::Frozen and the
/// Function-kind uncallable branch — `FunctionDesc.queryattrfamily` is
/// the base `None` stub, so multi-FunctionDesc always appears "common
/// access set = None" and routes to the `None`-access `MultipleFrozenPBCRepr`
/// arm.
#[allow(non_snake_case)]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
)]
pub fn getFrozenPBCRepr(
    rtyper: &Rc<RPythonTyper>,
    s_pbc: &SomePBC,
) -> Result<std::sync::Arc<dyn Repr>, TyperError> {
    if s_pbc.descriptions.is_empty() {
        return Err(TyperError::message(
            "getFrozenPBCRepr: empty descriptions (rpbc.py:612 assertion)",
        ));
    }
    if s_pbc.descriptions.len() == 1 && !s_pbc.can_be_none {
        let frozendesc = s_pbc
            .descriptions
            .values()
            .next()
            .expect("SomePBC guarantees non-empty descriptions")
            .clone();
        return Ok(std::sync::Arc::new(SingleFrozenPBCRepr::new(frozendesc)));
    }
    // upstream rpbc.py:616-625 — walk descs, collect each
    // `queryattrfamily()` (Python-object identity). If any two differ,
    // route to MultipleUnrelatedFrozenPBCRepr.
    //
    // FrozenDesc.queryattrfamily returns `Option<Rc<RefCell<FrozenAttrFamily>>>`;
    // FunctionDesc (via base `Desc`) returns `Option<()>` = None. Pyre
    // collapses both to `Option<usize>` keyed on `Rc::as_ptr`.
    let access_keys: Vec<Option<usize>> = s_pbc
        .descriptions
        .values()
        .map(|d| match d {
            DescEntry::Frozen(fd) => fd
                .borrow()
                .queryattrfamily()
                .map(|family| Rc::as_ptr(&family) as usize),
            _ => None,
        })
        .collect();
    let first = access_keys.first().copied().unwrap_or(None);
    let all_same = access_keys.iter().all(|key| *key == first);
    if !all_same {
        // upstream rpbc.py:621-624 — `rtyper.pbc_reprs['unrelated']`
        // singleton cache. Fetch if present, otherwise insert a fresh
        // repr and return it.
        use crate::translator::rtyper::rtyper::PbcReprKey;
        if let Some(cached) = rtyper.pbc_reprs.borrow().get(&PbcReprKey::Unrelated) {
            return Ok(cached.clone());
        }
        let fresh: std::sync::Arc<dyn Repr> =
            std::sync::Arc::new(MultipleUnrelatedFrozenPBCRepr::new(rtyper));
        rtyper
            .pbc_reprs
            .borrow_mut()
            .insert(PbcReprKey::Unrelated, fresh.clone());
        return Ok(fresh);
    }
    // upstream rpbc.py:626-632 — common-access-set arm.
    //
    // ```python
    // try:
    //     return rtyper.pbc_reprs[access]
    // except KeyError:
    //     result = MultipleFrozenPBCRepr(rtyper, access)
    //     rtyper.pbc_reprs[access] = result
    //     rtyper.add_pendingsetup(result)
    //     return result
    // ```
    //
    // All descs share `first` (verified above): the access family is
    // either `Some(usize)` or `None`. `None` means every desc has no
    // attrfamily (uncallable multi-desc function/frozen PBCs) —
    // `MultipleFrozenPBCRepr(rtyper, None)` lowers to a zero-field
    // `Struct('pbc')` (`_setup_repr_fields` returns `[]`, rpbc.py),
    // cached under the `pbc_reprs[None]` singleton slot.
    use crate::translator::rtyper::rtyper::PbcReprKey;
    let repr_key = match first {
        Some(access_key) => PbcReprKey::Access(access_key),
        None => PbcReprKey::AccessNone,
    };
    if let Some(cached) = rtyper.pbc_reprs.borrow().get(&repr_key) {
        return Ok(cached.clone());
    }
    // Fetch the live FrozenAttrFamily shared by all descs (verified via
    // `first`). `None` when the shared access set is absent — the
    // zero-field `access_set=None` arm.
    let access_family = match first {
        Some(_) => Some(
            s_pbc
                .descriptions
                .values()
                .find_map(|d| match d {
                    DescEntry::Frozen(fd) => fd.borrow().queryattrfamily(),
                    _ => None,
                })
                .ok_or_else(|| {
                    TyperError::message(
                        "getFrozenPBCRepr: invariant — first desc's queryattrfamily \
                         returned Some(key) but no descs surface a family",
                    )
                })?,
        ),
        None => None,
    };
    let fresh: std::sync::Arc<dyn Repr> =
        std::sync::Arc::new(MultipleFrozenPBCRepr::new(rtyper, access_family));
    rtyper
        .pbc_reprs
        .borrow_mut()
        .insert(repr_key, fresh.clone());
    rtyper.add_pendingsetup(fresh.clone());
    Ok(fresh)
}

/// RPython `class MethodsPBCRepr(Repr)` (rpbc.py).
///
/// ```python
/// class MethodsPBCRepr(Repr):
///     """Representation selected for a PBC of MethodDescs.
///     It assumes that all the methods come from the same name and have
///     been read from instances with a common base."""
///
///     def __init__(self, rtyper, s_pbc):
///         self.rtyper = rtyper
///         self.s_pbc = s_pbc
///         mdescs = list(s_pbc.descriptions)
///         methodname = mdescs[0].name
///         classdef = mdescs[0].selfclassdef
///         flags    = mdescs[0].flags
///         for mdesc in mdescs[1:]:
///             if mdesc.name != methodname: raise TyperError(...)
///             if mdesc.flags != flags:     raise TyperError(...)
///             classdef = classdef.commonbase(mdesc.selfclassdef)
///             if classdef is None:         raise TyperError(...)
///         self.methodname = methodname
///         self.classdef = classdef.get_owner(methodname)
///         self.s_im_self = annmodel.SomeInstance(self.classdef, flags=flags)
///         self.r_im_self = rclass.getinstancerepr(rtyper, self.classdef)
///         self.lowleveltype = self.r_im_self.lowleveltype
/// ```
#[derive(Debug)]
pub struct MethodsPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:1132).
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.s_pbc = s_pbc` (rpbc.py:1133) — original PBC of
    /// MethodDesc descriptions.
    pub s_pbc: SomePBC,
    /// RPython `self.methodname = methodname` (rpbc.py) — the
    /// shared method name across all `mdescs`.
    pub methodname: String,
    /// RPython `self.classdef = classdef.get_owner(methodname)`
    /// (rpbc.py:1152) — the owner ClassDef for `methodname` after
    /// `commonbase`-folding all `mdesc.selfclassdef`.
    pub classdef: Rc<RefCell<crate::annotator::classdesc::ClassDef>>,
    /// RPython `self.s_im_self = SomeInstance(self.classdef,
    /// flags=flags)` (rpbc.py:1154). Rust stores the constructed
    /// `SomeInstance` directly so `redispatch_call` can write it into
    /// `hop2.args_s[0]` without rebuilding it.
    pub s_im_self: crate::annotator::model::SomeInstance,
    /// RPython `self.r_im_self = getinstancerepr(rtyper, self.classdef)`
    /// (rpbc.py:1155). Stored as concrete `Arc<InstanceRepr>` so
    /// `redispatch_call` can call the InstanceRepr-specific
    /// `getfield('__class__')` / `rclass()` accessors directly.
    pub r_im_self: Arc<rclass::InstanceRepr>,
    /// RPython `self.lowleveltype = self.r_im_self.lowleveltype`
    /// (rpbc.py:1156) — the bound-`self` pointer's low-level type.
    /// Stored explicitly because `Repr::lowleveltype` returns
    /// `&LowLevelType` and we cannot borrow through `r_im_self`.
    lltype: LowLevelType,
    state: ReprState,
}

impl MethodsPBCRepr {
    /// RPython `MethodsPBCRepr.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py).
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        use crate::annotator::classdesc::ClassDef;

        // upstream: `mdescs = list(s_pbc.descriptions)`.
        let mut mdescs = Vec::with_capacity(s_pbc.descriptions.len());
        for entry in s_pbc.descriptions.values() {
            let md = entry.as_method().ok_or_else(|| {
                TyperError::message("MethodsPBCRepr: every description must be a MethodDesc")
            })?;
            mdescs.push(md);
        }
        if mdescs.is_empty() {
            return Err(TyperError::message(
                "MethodsPBCRepr: s_pbc must contain at least one MethodDesc",
            ));
        }

        // upstream: `methodname = mdescs[0].name; classdef =
        //            mdescs[0].selfclassdef; flags = mdescs[0].flags`.
        let methodname = mdescs[0].borrow().name.clone();
        let flags = mdescs[0].borrow().flags.clone();

        // The rust port stores `selfclassdef` as `ClassDefKey` (pointer
        // identity) instead of a live `Rc<RefCell<ClassDef>>` to avoid
        // descriptor-graph reference cycles; resolve the live handle via
        // `bookkeeper.lookup_classdef` here. None means "unbound method"
        // — invalid as a `MethodsPBCRepr` seed.
        let annotator = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MethodsPBCRepr.new: annotator weak ref dropped"))?;
        let bookkeeper = annotator.bookkeeper.clone();
        let resolve_classdef = |md: &Rc<RefCell<crate::annotator::description::MethodDesc>>|
            -> Result<Rc<RefCell<ClassDef>>, TyperError> {
            let key = md.borrow().selfclassdef.ok_or_else(|| {
                TyperError::message(format!(
                    "MethodsPBCRepr: unbound MethodDesc {:?} has no selfclassdef",
                    md.borrow().name,
                ))
            })?;
            bookkeeper.lookup_classdef(key).ok_or_else(|| {
                TyperError::message(format!(
                    "MethodsPBCRepr: unknown ClassDefKey {key:?}"
                ))
            })
        };
        let mut classdef = resolve_classdef(&mdescs[0])?;

        // upstream: `for mdesc in mdescs[1:]: ...`.
        for mdesc in mdescs.iter().skip(1) {
            let mdesc_b = mdesc.borrow();
            // upstream: `if mdesc.name != methodname: raise TyperError(...)`.
            if mdesc_b.name != methodname {
                return Err(TyperError::message(format!(
                    "cannot find a unique name under which the methods can \
                     be found: {:?}",
                    mdescs
                        .iter()
                        .map(|m| m.borrow().name.clone())
                        .collect::<Vec<_>>()
                )));
            }
            // upstream: `if mdesc.flags != flags: raise TyperError(...)`.
            if mdesc_b.flags != flags {
                return Err(TyperError::message(format!(
                    "inconsistent 'flags': {:?} versus {:?}",
                    mdesc_b.flags, flags
                )));
            }
            drop(mdesc_b);
            // upstream: `classdef = classdef.commonbase(mdesc.selfclassdef);
            //            if classdef is None: raise TyperError(...)`.
            let other = resolve_classdef(mdesc)?;
            classdef = ClassDef::commonbase(&classdef, &other).ok_or_else(|| {
                TyperError::message(format!(
                    "mixing methods coming from instances of classes with no \
                     common base: {:?}",
                    mdescs
                        .iter()
                        .map(|m| m.borrow().name.clone())
                        .collect::<Vec<_>>()
                ))
            })?;
        }

        // upstream: `self.classdef = classdef.get_owner(methodname)`.
        let owner = ClassDef::get_owner(&classdef, &methodname).ok_or_else(|| {
            TyperError::message(format!(
                "MethodsPBCRepr: classdef.get_owner({methodname:?}) returned None — \
                 method not declared anywhere in MRO"
            ))
        })?;

        // upstream: `self.s_im_self = SomeInstance(self.classdef, flags=flags)`.
        let s_im_self =
            crate::annotator::model::SomeInstance::new(Some(owner.clone()), false, flags);

        // upstream: `self.r_im_self = getinstancerepr(rtyper, self.classdef)`
        // — `getinstancerepr`'s `default_flavor` parameter (rclass.py)
        // defaults to `'gc'`.
        let r_im_self = rclass::getinstancerepr(rtyper, Some(&owner), rclass::Flavor::Gc)?;

        // upstream: `self.lowleveltype = self.r_im_self.lowleveltype`.
        let lltype = r_im_self.lowleveltype().clone();

        Ok(MethodsPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            s_pbc,
            methodname,
            classdef: owner,
            s_im_self,
            r_im_self,
            lltype,
            state: ReprState::new(),
        })
    }

    /// RPython `MethodsPBCRepr.add_instance_arg_to_hop(self, hop, call_args)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def add_instance_arg_to_hop(self, hop, call_args):
    ///     hop2 = hop.copy()
    ///     hop2.args_s[0] = self.s_im_self
    ///     hop2.args_r[0] = self.r_im_self
    ///     if call_args:
    ///         hop2.swap_fst_snd_args()
    ///         _, s_shape = hop2.r_s_popfirstarg()
    ///         adjust_shape(hop2, s_shape)
    ///     return hop2
    /// ```
    fn add_instance_arg_to_hop(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
        call_args: bool,
    ) -> Result<crate::translator::rtyper::rtyper::HighLevelOp, TyperError> {
        use crate::annotator::model::SomeValue;

        // upstream: `hop2 = hop.copy()`.
        let hop2 = hop.copy();
        // upstream: `hop2.args_s[0] = self.s_im_self`.
        hop2.args_s.borrow_mut()[0] = SomeValue::Instance(self.s_im_self.clone());
        // upstream: `hop2.args_r[0] = self.r_im_self`.
        hop2.args_r.borrow_mut()[0] = Some(self.r_im_self.clone() as Arc<dyn Repr>);

        // upstream: `if call_args: ... adjust_shape(hop2, s_shape)`.
        if call_args {
            hop2.swap_fst_snd_args();
            let (_r, s_shape) = hop2.r_s_popfirstarg();
            adjust_shape(&hop2, &s_shape)?;
        }
        Ok(hop2)
    }

    /// RPython `MethodsPBCRepr.redispatch_call(self, hop, call_args)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def redispatch_call(self, hop, call_args):
    ///     r_class = self.r_im_self.rclass
    ///     mangled_name, r_func = r_class.clsfields[self.methodname]
    ///     assert isinstance(r_func, FunctionReprBase)
    ///     # s_func = r_func.s_pbc -- not precise enough, see
    ///     # test_precise_method_call_1.  Build a more precise one...
    ///     funcdescs = [desc.funcdesc for desc in hop.args_s[0].descriptions]
    ///     s_func = annmodel.SomePBC(funcdescs, subset_of=r_func.s_pbc)
    ///     v_im_self = hop.inputarg(self, arg=0)
    ///     v_cls = self.r_im_self.getfield(v_im_self, '__class__', hop.llops)
    ///     v_func = r_class.getclsfield(v_cls, self.methodname, hop.llops)
    ///
    ///     hop2 = self.add_instance_arg_to_hop(hop, call_args)
    ///     hop2.v_s_insertfirstarg(v_func, s_func)   # insert 'function'
    ///
    ///     if (type(hop2.args_r[0]) is SmallFunctionSetPBCRepr and
    ///             type(r_func) is FunctionsPBCRepr):
    ///         hop2.args_r[0] = FunctionsPBCRepr(self.rtyper, s_func)
    ///     else:
    ///         hop2.args_v[0] = hop2.llops.convertvar(
    ///             hop2.args_v[0], r_func, hop2.args_r[0])
    ///
    ///     # now hop2 looks like simple_call(function, self, args...)
    ///     return hop2.dispatch()
    /// ```
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    pub fn redispatch_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
        call_args: bool,
    ) -> Result<Option<Hlvalue>, TyperError> {
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::rclass::ClassReprArc;

        // upstream: `r_class = self.r_im_self.rclass`.
        let r_class_arc = self.r_im_self.rclass().ok_or_else(|| {
            TyperError::message(
                "MethodsPBCRepr.redispatch_call: r_im_self.rclass not populated — \
                 setup() must have run",
            )
        })?;
        // upstream MethodsPBCRepr precondition: classdef is not None
        // (rpbc.py), so r_class is always a `ClassRepr`, never
        // `RootClassRepr`.
        let r_class = match r_class_arc {
            ClassReprArc::Inst(class_repr) => class_repr,
            ClassReprArc::Root(_) => {
                return Err(TyperError::message(
                    "MethodsPBCRepr.redispatch_call: rclass resolves to RootClassRepr — \
                     classdef is unexpectedly None",
                ));
            }
        };

        // upstream: `mangled_name, r_func = r_class.clsfields[self.methodname]`.
        let r_func = r_class
            .clsfields()
            .get(&self.methodname)
            .map(|(_mangled, r)| r.clone())
            .ok_or_else(|| {
                TyperError::message(format!(
                    "MethodsPBCRepr.redispatch_call: methodname {:?} not in \
                     r_class.clsfields — _setup_repr did not register it",
                    self.methodname
                ))
            })?;

        // upstream: `funcdescs = [desc.funcdesc for desc in
        //                          hop.args_s[0].descriptions]`.
        let s_pbc_call = match hop.args_s.borrow().first().cloned() {
            Some(SomeValue::PBC(pbc)) => pbc,
            other => {
                return Err(TyperError::message(format!(
                    "MethodsPBCRepr.redispatch_call: hop.args_s[0] is not a SomePBC: \
                     {other:?}"
                )));
            }
        };
        let mut funcdesc_entries = Vec::with_capacity(s_pbc_call.descriptions.len());
        for entry in s_pbc_call.descriptions.values() {
            let md = entry.as_method().ok_or_else(|| {
                TyperError::message(
                    "MethodsPBCRepr.redispatch_call: hop.args_s[0] description is not \
                     a MethodDesc",
                )
            })?;
            funcdesc_entries.push(DescEntry::Func(md.borrow().funcdesc.clone()));
        }

        // upstream: `s_func = SomePBC(funcdescs, subset_of=r_func.s_pbc)`.
        // `Repr::pbc_s_pbc()` returns `Some(&self.base.s_pbc)` for
        // `FunctionRepr` / `FunctionsPBCRepr` (rpbc.rs `impl Repr for
        // FunctionRepr` / `impl Repr for FunctionsPBCRepr`); other PBC
        // reprs return `None`, mirroring upstream's
        // `AttributeError`-on-non-FunctionReprBase semantics.
        let subset_of = r_func.pbc_s_pbc().cloned().map(Box::new);
        let s_func = SomeValue::PBC(SomePBC::with_subset(funcdesc_entries, false, subset_of));

        // upstream: `v_im_self = hop.inputarg(self, arg=0)`.
        let v_im_self = hop.inputarg(self as &dyn Repr, 0)?;

        // upstream: `v_cls = self.r_im_self.getfield(v_im_self,
        //                                            '__class__', hop.llops)`.
        let v_cls = {
            let mut llops = hop.llops.borrow_mut();
            self.r_im_self.getfield(
                v_im_self,
                "__class__",
                &mut llops,
                false,
                &Default::default(),
            )?
        };
        // upstream: `v_func = r_class.getclsfield(v_cls,
        //                                         self.methodname, hop.llops)`.
        let v_func = {
            let mut llops = hop.llops.borrow_mut();
            r_class.getclsfield(Hlvalue::Variable(v_cls), &self.methodname, &mut llops)?
        };

        // upstream: `hop2 = self.add_instance_arg_to_hop(hop, call_args)`.
        let hop2 = self.add_instance_arg_to_hop(hop, call_args)?;

        // upstream: `hop2.v_s_insertfirstarg(v_func, s_func)`.
        hop2.v_s_insertfirstarg(Hlvalue::Variable(v_func), s_func)?;

        // upstream:
        //     if (type(hop2.args_r[0]) is SmallFunctionSetPBCRepr and
        //             type(r_func) is FunctionsPBCRepr):
        //         hop2.args_r[0] = FunctionsPBCRepr(self.rtyper, s_func)
        //     else:
        //         hop2.args_v[0] = hop2.llops.convertvar(
        //             hop2.args_v[0], r_func, hop2.args_r[0])
        //
        // The Small→Functions short-circuit prevents
        // `getrepr(SomePBC(funcdescs, subset_of=r_func.s_pbc))` from
        // shrinking the per-call repr to a `SmallFunctionSetPBCRepr`
        // (which would lose the function-pointer typing the call site
        // expects). Replace the resolved repr with a fresh
        // `FunctionsPBCRepr(rtyper, s_func)` instead.
        let new_args_r = hop2.args_r.borrow()[0].clone().ok_or_else(|| {
            TyperError::message(
                "MethodsPBCRepr.redispatch_call: hop2.args_r[0] missing after \
                 v_s_insertfirstarg",
            )
        })?;
        if matches!(
            new_args_r.repr_class_id(),
            ReprClassId::SmallFunctionSetPBCRepr
        ) && matches!(r_func.repr_class_id(), ReprClassId::FunctionsPBCRepr)
        {
            // upstream: `hop2.args_r[0] = FunctionsPBCRepr(self.rtyper, s_func)`.
            //   Re-derive `s_func` SomePBC from the inserted-arg's
            //   SomeValue so the new FunctionsPBCRepr matches the
            //   narrowed funcdescs.
            let s_func_pbc = match hop2.args_s.borrow().first().cloned() {
                Some(SomeValue::PBC(p)) => p,
                other => {
                    return Err(TyperError::message(format!(
                        "MethodsPBCRepr.redispatch_call: hop2.args_s[0] is not a \
                         SomePBC after v_s_insertfirstarg: {other:?}"
                    )));
                }
            };
            let rtyper = self.rtyper.upgrade().ok_or_else(|| {
                TyperError::message("MethodsPBCRepr.redispatch_call: rtyper weak ref dropped")
            })?;
            let new_repr: Arc<dyn Repr> = FunctionsPBCRepr::new(&rtyper, s_func_pbc)?;
            hop2.args_r.borrow_mut()[0] = Some(new_repr);
        } else {
            // upstream `else` arm: re-`convertvar` the freshly-inserted
            // `v_func` from `r_func` to whatever `getrepr(s_func)`
            // produced.
            let v_after_convert = {
                let arg_v0 = hop2.args_v.borrow()[0].clone();
                let mut llops = hop2.llops.borrow_mut();
                llops.convertvar(arg_v0, r_func.as_ref(), new_args_r.as_ref())?
            };
            hop2.args_v.borrow_mut()[0] = v_after_convert;
        }

        // upstream: `return hop2.dispatch()`.
        hop2.dispatch()
    }

    /// RPython `MethodsPBCRepr.get_r_implfunc(self)` (rpbc.py):
    ///
    /// ```python
    /// def get_r_implfunc(self):
    ///     r_class = self.r_im_self.rclass
    ///     mangled_name, r_func = r_class.clsfields[self.methodname]
    ///     return r_func, 1
    /// ```
    ///
    /// The trait method `get_r_implfunc` calls this. The upstream
    /// Python `r_func` reference outlives the call, but Rust's
    /// `&dyn Repr` cannot escape the short-lived `RefCell::borrow()`
    /// guard on `clsfields`. Returning the `Arc<dyn Repr>` clone
    /// preserves the same identity (the Arc is shared with the cache).
    pub fn get_r_implfunc_arc_impl(&self) -> Result<(Arc<dyn Repr>, usize), TyperError> {
        use crate::translator::rtyper::rclass::ClassReprArc;

        // upstream: `r_class = self.r_im_self.rclass`.
        let r_class_arc = self.r_im_self.rclass().ok_or_else(|| {
            TyperError::message(
                "MethodsPBCRepr.get_r_implfunc: r_im_self.rclass not populated — \
                 setup() must have run",
            )
        })?;
        let r_class = match r_class_arc {
            ClassReprArc::Inst(class_repr) => class_repr,
            ClassReprArc::Root(_) => {
                return Err(TyperError::message(
                    "MethodsPBCRepr.get_r_implfunc: rclass resolves to RootClassRepr",
                ));
            }
        };
        // upstream: `mangled_name, r_func = r_class.clsfields[self.methodname];
        //           return r_func, 1`.
        let r_func = r_class
            .clsfields()
            .get(&self.methodname)
            .map(|(_mangled, r)| r.clone())
            .ok_or_else(|| {
                TyperError::message(format!(
                    "MethodsPBCRepr.get_r_implfunc: methodname {:?} not in \
                     r_class.clsfields",
                    self.methodname
                ))
            })?;
        Ok((r_func, 1))
    }

    /// RPython `MethodsPBCRepr.get_method_from_instance(self, r_inst,
    /// v_inst, llops)` (rpbc.py):
    ///
    /// ```python
    /// def get_method_from_instance(self, r_inst, v_inst, llops):
    ///     # The 'self' might have to be cast to a parent class
    ///     # (as shown for example in test_rclass/test_method_both_A_and_B)
    ///     return llops.convertvar(v_inst, r_inst, self.r_im_self)
    /// ```
    ///
    /// Called from `InstanceRepr.rtype_getattr` (rclass.py) on
    /// the method-dispatch branch. Pyre's `convertvar` lives on the
    /// active `LowLevelOpList` (`rtyper.rs`); pass it through.
    pub fn get_method_from_instance(
        &self,
        r_inst: &dyn Repr,
        v_inst: Hlvalue,
        llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
    ) -> Result<Hlvalue, TyperError> {
        llops.convertvar(v_inst, r_inst, self.r_im_self.as_ref())
    }
}

impl Repr for MethodsPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "MethodsPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::MethodsPBCRepr
    }

    /// RPython `MethodsPBCRepr.get_r_implfunc(self)` (rpbc.py).
    /// The returned `r_func` is the clsfields entry, a different repr.
    fn get_r_implfunc(&self) -> Result<(Arc<dyn Repr>, usize), TyperError> {
        self.get_r_implfunc_arc_impl()
    }

    /// RPython `MethodsPBCRepr.get_s_callable(self)` (rpbc.py):
    /// `return self.s_pbc`.
    fn get_s_callable(&self) -> Option<SomePBC> {
        Some(self.s_pbc.clone())
    }

    /// RPython `MethodsPBCRepr.convert_const(self, method)`
    /// (rpbc.py):
    ///
    /// ```python
    /// def convert_const(self, method):
    ///     if method is None:
    ///         return nullptr(self.lowleveltype.TO)
    ///     if getattr(method, 'im_func', None) is None:
    ///         raise TyperError("not a bound method: %r" % method)
    ///     return self.r_im_self.convert_const(method.im_self)
    /// ```
    ///
    /// Body delegates to `InstanceRepr.convert_const` for the bound-self
    /// arm; the `None` arm returns a null pointer typed at
    /// `self.lowleveltype` (= `r_im_self.lowleveltype`).
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        // upstream: `if method is None: return nullptr(self.lowleveltype.TO)`.
        //   `self.lowleveltype == r_im_self.lowleveltype`, so reuse
        //   `InstanceRepr::null_instance` — produces a null `_ptr`
        //   whose container matches the target Ptr.
        if matches!(value, ConstValue::None) {
            let null = self.r_im_self.null_instance()?;
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(null)),
                self.lltype.clone(),
            ));
        }
        // upstream: `if getattr(method, 'im_func', None) is None:
        //              raise TyperError("not a bound method: %r" % method)`.
        let host_obj = match value {
            ConstValue::HostObject(h) => h.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "MethodsPBCRepr.convert_const: expected HostObject(BoundMethod) \
                     or None, got {other:?}"
                )));
            }
        };
        if !host_obj.is_bound_method() {
            return Err(TyperError::message(format!(
                "not a bound method: {:?}",
                host_obj.qualname()
            )));
        }
        // upstream: `return self.r_im_self.convert_const(method.im_self)`.
        let im_self = host_obj.bound_method_self().cloned().ok_or_else(|| {
            TyperError::message("MethodsPBCRepr.convert_const: bound method has no self_obj")
        })?;
        let im_self_const = ConstValue::HostObject(im_self);
        (self.r_im_self.as_ref() as &dyn Repr).convert_const(&im_self_const)
    }

    /// RPython `MethodsPBCRepr.rtype_simple_call(self, hop)`
    /// (rpbc.py).
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.redispatch_call(hop, false)
    }

    /// RPython `MethodsPBCRepr.rtype_call_args(self, hop)`
    /// (rpbc.py).
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.redispatch_call(hop, true)
    }
}

/// RPython `SomePBC.rtyper_makerepr` (rpbc.py), the
/// `DescKind`-keyed factory. Each `DescKind` arm dispatches to the
/// matching Repr constructor:
///
///   * `Function`: `len(descriptions) == 1 && !can_be_None` →
///     [`FunctionRepr`]; small multi-desc → [`SmallFunctionSetPBCRepr`]
///     (chosen by `small_cand`); larger multi-desc →
///     [`FunctionsPBCRepr`]; uncallable → [`getFrozenPBCRepr`].
///   * `Class` → [`ClassesPBCRepr`] (constant + non-constant arms).
///   * `Method` → [`MethodsPBCRepr`].
///   * `MethodOfFrozen` → [`MethodOfFrozenPBCRepr`].
///   * `Frozen` → [`getFrozenPBCRepr`] (Single / MultipleUnrelated /
///     Multiple arms).
///
/// All callable arms route through `FunctionReprBase.call`, which is
/// ported via the `callparse` module (rpbc.py). The remaining
/// `MissingRTypeOperation` paths are scoped to specific edge cases
/// documented at each repr (e.g. `getFrozenPBCRepr` `None`-access
/// arm, `FunctionsPBCRepr` multi-row `setup_specfunc` corners).
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
)]
pub fn somepbc_rtyper_makerepr(
    s_pbc: &SomePBC,
    rtyper: &Rc<RPythonTyper>,
) -> Result<std::sync::Arc<dyn Repr>, TyperError> {
    let kind = s_pbc.get_kind().map_err(|e| {
        TyperError::message(format!(
            "SomePBC.rtyper_makerepr: {}",
            e.msg.as_deref().unwrap_or("<no message>")
        ))
    })?;
    match kind {
        // MemoDesc is-a FunctionDesc; a memo PBC builds the same repr
        // (its `as_function()` yields the wrapped base).
        DescKind::Function | DescKind::Memo => {
            if s_pbc.descriptions.len() == 1 && !s_pbc.can_be_none {
                Ok(FunctionRepr::new(rtyper, s_pbc.clone())? as std::sync::Arc<dyn Repr>)
            } else {
                // rpbc.py:42-49 — sample = self.any_description();
                // callfamily = sample.querycallfamily(); if callfamily
                // and callfamily.total_calltable_size > 0:
                //     getRepr = FunctionsPBCRepr
                //     if small_cand(rtyper, self):
                //         getRepr = SmallFunctionSetPBCRepr
                // else:
                //     getRepr = getFrozenPBCRepr
                let sample = s_pbc.any_description().ok_or_else(|| {
                    TyperError::message("SomePBC.rtyper_makerepr: empty descriptions")
                })?;
                let callfamily = sample
                    .as_function()
                    .ok_or_else(|| {
                        TyperError::message(
                            "SomePBC.rtyper_makerepr: Function-kind sample is not a FunctionDesc",
                        )
                    })?
                    .borrow()
                    .base
                    .querycallfamily();
                let callable = callfamily
                    .as_ref()
                    .map(|cf| cf.borrow().total_calltable_size > 0)
                    .unwrap_or(false);
                if callable {
                    if small_cand(rtyper, s_pbc)? {
                        // rpbc.py — `getRepr = SmallFunctionSetPBCRepr`.
                        Ok(SmallFunctionSetPBCRepr::new(rtyper, s_pbc.clone())?
                            as std::sync::Arc<dyn Repr>)
                    } else {
                        Ok(FunctionsPBCRepr::new(rtyper, s_pbc.clone())?
                            as std::sync::Arc<dyn Repr>)
                    }
                } else {
                    // rpbc.py:49 — uncallable Function-kind PBCs route
                    // through `getFrozenPBCRepr`. The factory dispatches
                    // to `SingleFrozenPBCRepr` (single-desc + !can_be_None),
                    // `MultipleUnrelatedFrozenPBCRepr` (cached at
                    // `pbc_reprs[Unrelated]`), `MultipleFrozenPBCRepr`
                    // (cached at `pbc_reprs[Access(_)]`), or — when every
                    // desc has no attrfamily — the `None`-access
                    // `MultipleFrozenPBCRepr` zero-field struct PBC
                    // (cached at `pbc_reprs[AccessNone]`).
                    getFrozenPBCRepr(rtyper, s_pbc)
                }
            }
        }
        DescKind::Class => {
            // rpbc.py — `getRepr = ClassesPBCRepr`. Both
            // constant (lowleveltype = Void) and non-constant
            // (lowleveltype = CLASSTYPE) arms are ported in
            // `ClassesPBCRepr::new`.
            Ok(
                std::sync::Arc::new(ClassesPBCRepr::new(rtyper, s_pbc.clone())?)
                    as std::sync::Arc<dyn Repr>,
            )
        }
        DescKind::Method => {
            // rpbc.py — `getRepr = MethodsPBCRepr`.
            Ok(
                std::sync::Arc::new(MethodsPBCRepr::new(rtyper, s_pbc.clone())?)
                    as std::sync::Arc<dyn Repr>,
            )
        }
        DescKind::Frozen => getFrozenPBCRepr(rtyper, s_pbc),
        DescKind::MethodOfFrozen => {
            // rpbc.py — `getRepr = MethodOfFrozenPBCRepr`.
            Ok(
                std::sync::Arc::new(MethodOfFrozenPBCRepr::new(rtyper, s_pbc.clone())?)
                    as std::sync::Arc<dyn Repr>,
            )
        }
    }
}

#[cfg(test)]
mod pbc_repr_tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::description::{DescEntry, FunctionDesc};
    use crate::flowspace::argument::Signature;
    use std::cell::RefCell as StdRefCell;

    fn make_rtyper() -> (Rc<RPythonAnnotator>, Rc<RPythonTyper>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().unwrap();
        (ann, rtyper)
    }

    fn function_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
        DescEntry::function(Rc::new(StdRefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            name,
            Signature::new(vec![], None, None),
            None,
            None,
        ))))
    }

    #[test]
    fn function_repr_has_void_lowleveltype_and_pbc_tag() {
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        assert_eq!(r.class_name(), "FunctionRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::FunctionRepr);
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_function_desc_without_none_returns_function_repr() {
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "FunctionRepr");
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_function_desc_with_can_be_none_returns_multiple_frozen() {
        // Uncallable (no callfamily wired), `can_be_None=true` drops out of
        // the SingleFrozenPBCRepr fast-path → getFrozenPBCRepr multi-desc
        // branch → the `access is None` no-attrs arm builds a zero-field
        // MultipleFrozenPBCRepr (rpbc.py + 758).
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], true);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
    }

    #[test]
    fn somepbc_rtyper_makerepr_multi_function_descs_without_callfamily_uses_getfrozenpbcrepr() {
        // Multi-FunctionDesc PBC without an attached callfamily → upstream
        // rpbc.py routes to getFrozenPBCRepr, whose no-attrs
        // (`access is None`) arm yields a zero-field MultipleFrozenPBCRepr.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let g = function_entry(&ann.bookkeeper, "g");
        let s_pbc = SomePBC::new(vec![f, g], false);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
    }

    #[test]
    fn somepbc_rtyper_makerepr_multi_function_descs_with_callfamily_uses_functionspbcrepr() {
        // Multi-FunctionDesc PBC whose sample has total_calltable_size > 0 →
        // upstream rpbc.py routes to FunctionsPBCRepr. The single
        // uniquerow branch (rpbc.py) produces lowleveltype = row.fntype
        // wrapped as Ptr(Func).
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        // Pre-populate each FunctionDesc's graph cache so consider_call_site
        // can build the call table without touching a missing pyobj.
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        let args = ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        assert!(
            fd_f.borrow()
                .base
                .getcallfamily()
                .unwrap()
                .borrow()
                .total_calltable_size
                > 0,
            "sanity: consider_call_site should have grown total_calltable_size",
        );

        let s_pbc = SomePBC::new(
            vec![DescEntry::function(fd_f), DescEntry::function(fd_g)],
            false,
        );
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "FunctionsPBCRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::FunctionsPBCRepr);
        // Single uniquerow → Ptr(Func(...)) as upstream `typeOf(llfn)`.
        assert!(matches!(
            r.lowleveltype(),
            LowLevelType::Ptr(box_ptr) if matches!(box_ptr.TO, PtrTarget::Func(_))
        ));
    }

    /// Build a multi-uniquerow `FunctionsPBCRepr` for the
    /// `setup_specfunc` / multi-row tests. The shape:
    ///   1. `consider_call_site` populates a CallFamily with one row
    ///      that contains both `fd_f` and `fd_g`.
    ///   2. We then `calltable_add_row` a second row whose descs are
    ///      disjoint from the first, so `build_concrete_calltable` 's
    ///      merge step (rpbc.rs LLCallTable::lookup) cannot fold them
    ///      together — `llct.uniquerows.len()` becomes 2.
    ///
    /// Returns `(repr, fd_f, fd_g, rtyper)` for downstream
    /// `convert_desc` / `convert_to_concrete_llfn` tests. The `rtyper`
    /// is returned alongside so the caller keeps the strong `Rc` alive
    /// — the repr only holds a `Weak<RPythonTyper>` and would deny
    /// upgrade once the helper's local Rc drops. The repr is the `Arc`
    /// `FunctionsPBCRepr::new` built, so `get_r_implfunc` can upgrade it.
    fn build_multi_row_functions_pbc_repr() -> (
        Arc<FunctionsPBCRepr>,
        Rc<StdRefCell<crate::annotator::description::FunctionDesc>>,
        Rc<StdRefCell<crate::annotator::description::FunctionDesc>>,
        Rc<RPythonTyper>,
    ) {
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::{CallTableRow, DescKey, GraphCacheKey};
        use crate::annotator::model::{SomeInteger, SomeValue};

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        let args = ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        // Inject a second row with descs disjoint from row0 so
        // build_concrete_calltable produces 2 uniquerows. Reuses the
        // shape consider_call_site populated.
        {
            let family = fd_f.borrow().base.getcallfamily().unwrap();
            let mut family_mut = family.borrow_mut();
            let shape = family_mut
                .calltables
                .keys()
                .next()
                .cloned()
                .expect("consider_call_site must populate at least one shape");
            let mut row2 = CallTableRow::new();
            // DescKey value matters only for hash-table identity here —
            // any value not aliasing fd_f / fd_g works.
            row2.insert(DescKey(99_999), super::tests::make_pygraph("h_graph"));
            family_mut.calltable_add_row(shape, row2);
        }

        let s_pbc = SomePBC::new(
            vec![
                DescEntry::function(fd_f.clone()),
                DescEntry::function(fd_g.clone()),
            ],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();
        (r, fd_f, fd_g, rtyper)
    }

    #[test]
    fn functions_pbc_repr_setup_specfunc_lowleveltype_is_ptr_specfunc_struct() {
        // rpbc.py:235-247 — when len(uniquerows) > 1 the constructor
        // calls `setup_specfunc()` which builds
        // `Ptr(Struct('specfunc', (attr0, fntype0), (attr1, fntype1),
        // ..., hints={'immutable': True, 'static_immutable': True}))`.
        // pyre's port (`setup_specfunc` in this file) emits the same
        // shape; `build_concrete_calltable` stamps each uniquerow
        // with `variant{N}` attrnames.
        let (r, _fd_f, _fd_g, _rtyper) = build_multi_row_functions_pbc_repr();
        assert_eq!(
            r.uniquerows.len(),
            2,
            "test fixture must drive multi-row branch"
        );

        let LowLevelType::Ptr(boxed) = r.lowleveltype() else {
            panic!(
                "multi-row lowleveltype must be Ptr, got {:?}",
                r.lowleveltype()
            );
        };
        let PtrTarget::Struct(struct_t) = &boxed.TO else {
            panic!(
                "multi-row lowleveltype.TO must be Struct, got {:?}",
                boxed.TO
            );
        };
        assert_eq!(
            struct_t._name, "specfunc",
            "upstream rpbc.py:247 names the struct 'specfunc'"
        );
        assert_eq!(
            struct_t._names.len(),
            2,
            "specfunc must have one field per uniquerow; got names {:?}",
            struct_t._names,
        );
        let mut names_sorted = struct_t._names.clone();
        names_sorted.sort();
        assert_eq!(
            names_sorted,
            vec!["variant0".to_string(), "variant1".to_string()],
            "build_concrete_calltable stamps `variant{{N}}` attrnames",
        );
        for name in &struct_t._names {
            let fld = struct_t._flds.get(name).expect("specfunc field present");
            assert!(
                matches!(fld, LowLevelType::Ptr(p) if matches!(p.TO, PtrTarget::Func(_))),
                "specfunc field {name} must be Ptr(Func), got {fld:?}",
            );
        }

        // hints from setup_specfunc.
        use crate::flowspace::model::ConstValue;
        assert!(
            matches!(
                struct_t._hints.get("immutable"),
                Some(ConstValue::Bool(true))
            ),
            "setup_specfunc must set hints['immutable']=True",
        );
        assert!(
            matches!(
                struct_t._hints.get("static_immutable"),
                Some(ConstValue::Bool(true))
            ),
            "setup_specfunc must set hints['static_immutable']=True",
        );
    }

    #[test]
    fn functions_pbc_repr_convert_desc_multi_row_returns_specfunc_struct_constant() {
        // rpbc.py — when len(uniquerows) > 1, convert_desc
        // calls `create_specfunc()` (≡ ll_malloc(specfunc_struct,
        // immortal=True)) and `setattr(result, attrname, llfn)` for
        // each (attrname, llfn) in llfns. The Constant returned has
        // the multi-row Ptr(Struct) concretetype, value is LLPtr to
        // the populated specfunc.
        use crate::flowspace::model::ConstValue;
        let (r, fd_f, _fd_g, _rtyper) = build_multi_row_functions_pbc_repr();
        let desc_f = DescEntry::function(fd_f);

        let c1 = r.convert_desc(&desc_f).unwrap();
        assert_eq!(
            c1.concretetype.as_ref(),
            Some(r.lowleveltype()),
            "Constant.concretetype must match the repr's multi-row Ptr(Struct)",
        );
        assert!(
            matches!(c1.value, ConstValue::LLPtr(_)),
            "Constant.value must wrap a _ptr to the specfunc struct; got {:?}",
            c1.value,
        );

        // rpbc.py caches the result. Second call returns the
        // same Constant (pointer-identity-stable on the inner _ptr).
        let c2 = r.convert_desc(&desc_f).unwrap();
        let (ConstValue::LLPtr(p1), ConstValue::LLPtr(p2)) = (&c1.value, &c2.value) else {
            panic!("expected LLPtr values for both convert_desc calls");
        };
        assert_eq!(
            p1, p2,
            "funccache must return the same _ptr identity on re-lookup"
        );
    }

    #[test]
    fn functions_pbc_repr_convert_to_concrete_llfn_multi_row_emits_getfield() {
        // rpbc.py:308-312 — when len(uniquerows) > 1,
        // convert_to_concrete_llfn looks up the row at
        // (shape, index) in concretetable, then emits
        // `getfield(v, c_rowname) -> row.fntype` (rpbc.rs
        // get_specfunc_row). Validates the multi-row dispatch path's
        // emitted op shape end-to-end.
        use crate::flowspace::model::{ConstValue, Hlvalue, Variable};
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        use std::cell::RefCell as StdRef;

        let (r, _fd_f, _fd_g, rtyper) = build_multi_row_functions_pbc_repr();

        // Pick the first concretetable entry (shape, index=0) and its
        // attrname — we don't care which row, only that the emitted
        // getfield names that exact variant.
        let ((shape, index), row_ref) = r
            .concretetable
            .iter()
            .next()
            .map(|(k, v)| (k.clone(), v.clone()))
            .expect("multi-row concretetable must be populated");
        let row = row_ref.borrow();
        let expected_attrname = row
            .attrname
            .clone()
            .expect("multi-row uniquerow must carry a variant{N} attrname");
        let expected_resulttype = LowLevelType::Ptr(Box::new(LLPtr {
            TO: PtrTarget::Func(row.fntype.clone()),
        }));
        drop(row);

        // Build a Variable carrying the repr's multi-row lowleveltype
        // (Ptr(Struct('specfunc', ...))) — upstream `assert
        // v.concretetype == self.lowleveltype` (rpbc.py:301).
        let v_var = Variable::new();
        v_var.set_concretetype(Some(r.lowleveltype().clone()));
        let v = Hlvalue::Variable(v_var);

        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper, None)));
        let result = r
            .convert_to_concrete_llfn(&v, &shape, index, &llops)
            .expect("multi-row convert_to_concrete_llfn must succeed");

        // The returned value's concretetype is the row's fntype
        // wrapped in Ptr(Func) — upstream `resulttype = row.fntype`
        // is a `Ptr(Func)`.
        match &result {
            Hlvalue::Variable(var) => assert_eq!(
                var.concretetype(),
                Some(expected_resulttype.clone()),
                "convert_to_concrete_llfn return must carry the row's fntype as Ptr(Func)",
            ),
            other => panic!("expected Variable result from getfield, got {other:?}"),
        }

        // get_specfunc_row emits `getfield(v, c_rowname)` —
        // assert the op shape: opname + first arg is v + second arg
        // is a Constant carrying the variant name as bytes.
        let llops_borrow = llops.borrow();
        let getfield_op = llops_borrow
            .ops
            .iter()
            .find(|op| op.opname == "getfield")
            .expect("multi-row dispatch must emit a getfield op");
        assert_eq!(
            getfield_op.args.len(),
            2,
            "getfield op signature: (v, c_rowname); got {} args",
            getfield_op.args.len(),
        );
        let Hlvalue::Constant(rowname_const) = &getfield_op.args[1] else {
            panic!("getfield's second arg must be a Constant carrying the attrname");
        };
        match &rowname_const.value {
            ConstValue::ByteStr(bytes) => assert_eq!(
                bytes.as_slice(),
                expected_attrname.as_bytes(),
                "getfield's c_rowname must equal the chosen uniquerow's attrname",
            ),
            other => panic!("getfield c_rowname must be ByteStr, got {other:?}"),
        }
    }

    #[test]
    fn function_repr_convert_desc_returns_void_none_constant() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f.clone()], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();

        // rpbc.py — convert_desc returns None; pyre wraps it in
        // a Void-typed Constant so call sites get a typed value.
        let c = r.convert_desc(&f).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));
    }

    #[test]
    fn function_repr_convert_const_returns_void_none_constant() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();

        // rpbc.py — convert_const returns None regardless of
        // the input value (a constant function PBC has no runtime
        // payload).
        let c = r.convert_const(&ConstValue::Int(123)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));
    }

    fn frozen_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
        use crate::annotator::description::FrozenDesc;
        use crate::flowspace::model::HostObject;
        // `Frozen`-kind descriptions require a concrete Python-side
        // object; use a module carrier since upstream frozen descs
        // typically wrap module/class singletons.
        let pyobj = HostObject::new_module(name);
        DescEntry::Frozen(Rc::new(StdRefCell::new(
            FrozenDesc::new(bk.clone(), pyobj).expect("FrozenDesc::new"),
        )))
    }

    fn method_of_frozen_entry(
        bk: &Rc<Bookkeeper>,
        funcdesc: &Rc<StdRefCell<crate::annotator::description::FunctionDesc>>,
        frozen_name: &str,
    ) -> DescEntry {
        use crate::annotator::description::{FrozenDesc, MethodOfFrozenDesc};
        use crate::flowspace::model::HostObject;
        let pyobj = HostObject::new_module(frozen_name);
        let frozendesc = Rc::new(StdRefCell::new(FrozenDesc::new(bk.clone(), pyobj).unwrap()));
        DescEntry::MethodOfFrozen(Rc::new(StdRefCell::new(MethodOfFrozenDesc::new(
            bk.clone(),
            crate::annotator::description::FuncDescEntry::plain(funcdesc.clone()),
            frozendesc,
        ))))
    }

    #[test]
    fn method_of_frozen_pbc_repr_new_succeeds_for_single_methodoffrozen_pbc() {
        // rpbc.py __init__ — shared FunctionDesc + can_be_None=False
        // builds the repr; lowleveltype = r_im_self.lowleveltype.
        // Single-MethodOfFrozenDesc PBC routes r_im_self through
        // SingleFrozenPBCRepr (Void-typed).
        let (ann, rtyper) = make_rtyper();
        let Some(fd) = function_entry(&ann.bookkeeper, "f").as_function() else {
            unreachable!()
        };
        let mof = method_of_frozen_entry(&ann.bookkeeper, &fd, "frozen0");
        let s_pbc = SomePBC::new(vec![mof], false);
        let r = MethodOfFrozenPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(r.class_name(), "MethodOfFrozenPBCRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::MethodOfFrozenPBCRepr);
        // r_im_self is `SingleFrozenPBCRepr` → Void.
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        assert_eq!(r.lowleveltype(), r.r_im_self.lowleveltype());
        // The shared funcdesc on the repr is the same Rc as the
        // one bound on the MethodOfFrozenDesc.
        assert!(Rc::ptr_eq(&r.funcdesc.func(), &fd));
        let s_callable = r.get_s_callable().expect("underlying function");
        assert_eq!(s_callable.descriptions.len(), 1);
        assert!(
            s_callable
                .descriptions
                .values()
                .next()
                .unwrap()
                .as_function()
                .is_some()
        );
        let (r_func, nimplicit) = r.get_r_implfunc().expect("owned impl func");
        assert_eq!(nimplicit, 1);
        assert!(
            r_func
                .get_s_signatures(&crate::flowspace::argument::CallShape {
                    shape_cnt: 1,
                    shape_keys: Vec::new(),
                    shape_star: false,
                })
                .is_ok()
                || r_func.pbc_s_pbc().is_some()
        );
    }

    // The "mixed funcdescs" rejection branch (rpbc.py:851-853 `assert
    // len(funcdescs) == 1`) is enforced one layer up: `SomePBC::new`
    // (annotator/model.rs) panics with "AnnotatorError: You
    // can't mix a set of methods on a frozen PBC in RPython that are
    // different underlying functions" before the rtyper repr ever
    // sees the SomePBC. The redundant `funcdesc_set.len() != 1`
    // guard inside `MethodOfFrozenPBCRepr::new` is defensive only —
    // the upstream `assert` is the load-bearing check. No separate
    // test is added here because the SomePBC-level enforcement is
    // already exercised by `annotator/model::tests`.

    #[test]
    fn method_of_frozen_pbc_repr_convert_desc_routes_through_r_im_self() {
        // rpbc.py convert_desc — `if mdesc.funcdesc is not self.funcdesc:
        // raise; return self.r_im_self.convert_desc(mdesc.frozendesc)`.
        // The success path returns the Constant produced by the
        // bound-frozendesc's repr (single-frozen → Void None sentinel).
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let Some(fd) = function_entry(&ann.bookkeeper, "f").as_function() else {
            unreachable!()
        };
        let mof = method_of_frozen_entry(&ann.bookkeeper, &fd, "frozen0");
        let s_pbc = SomePBC::new(vec![mof.clone()], false);
        let r = MethodOfFrozenPBCRepr::new(&rtyper, s_pbc).unwrap();
        let c = r.convert_desc(&mof).unwrap();
        // Single-frozen `r_im_self` → Void-typed `None` sentinel
        // matches `SingleFrozenPBCRepr.convert_desc` parity.
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));
    }

    #[test]
    fn method_of_frozen_pbc_repr_convert_const_requires_hostobject_argument() {
        // rpbc.py — `convert_const(method)` calls
        // `bookkeeper.getdesc(method)` on a host-side bound method.
        // pyre's `bookkeeper.getdesc` only accepts `HostObject`s, so the
        // ported `convert_const` rejects non-`HostObject` ConstValues
        // before invoking `getdesc`.
        let (ann, rtyper) = make_rtyper();
        let Some(fd) = function_entry(&ann.bookkeeper, "f").as_function() else {
            unreachable!()
        };
        let mof = method_of_frozen_entry(&ann.bookkeeper, &fd, "frozen0");
        let s_pbc = SomePBC::new(vec![mof], false);
        let r = MethodOfFrozenPBCRepr::new(&rtyper, s_pbc).unwrap();
        let err = r
            .convert_const(&crate::flowspace::model::ConstValue::Int(42))
            .unwrap_err();
        assert!(
            err.to_string().contains("expected HostObject"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn method_of_frozen_pbc_repr_convert_desc_rejects_foreign_funcdesc() {
        // rpbc.py:879-880 — "not a method bound on %r". Build the
        // repr around `f`, then call convert_desc with a MethodOf-
        // FrozenDesc whose funcdesc is `g`.
        let (ann, rtyper) = make_rtyper();
        let Some(fd_f) = function_entry(&ann.bookkeeper, "f").as_function() else {
            unreachable!()
        };
        let Some(fd_g) = function_entry(&ann.bookkeeper, "g").as_function() else {
            unreachable!()
        };
        let mof_f = method_of_frozen_entry(&ann.bookkeeper, &fd_f, "frozen0");
        let mof_g = method_of_frozen_entry(&ann.bookkeeper, &fd_g, "frozen0");
        let s_pbc = SomePBC::new(vec![mof_f], false);
        let r = MethodOfFrozenPBCRepr::new(&rtyper, s_pbc).unwrap();
        let err = r.convert_desc(&mof_g).unwrap_err();
        assert!(
            err.to_string().contains("not a method bound on"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn somepbc_rtyper_makerepr_method_of_frozen_kind_routes_to_method_of_frozen_pbc_repr() {
        // rpbc.py:57-58 — `elif issubclass(kind, MethodOfFrozenDesc):
        //                  getRepr = MethodOfFrozenPBCRepr`.
        use crate::annotator::model::SomeValue;
        let (ann, rtyper) = make_rtyper();
        let Some(fd) = function_entry(&ann.bookkeeper, "f").as_function() else {
            unreachable!()
        };
        let mof = method_of_frozen_entry(&ann.bookkeeper, &fd, "frozen0");
        let s_pbc = SomePBC::new(vec![mof], false);
        let r = rtyper.getrepr(&SomeValue::PBC(s_pbc)).unwrap();
        assert_eq!(r.repr_class_id(), ReprClassId::MethodOfFrozenPBCRepr);
    }

    fn method_entry(
        bk: &Rc<Bookkeeper>,
        funcdesc: &Rc<StdRefCell<crate::annotator::description::FunctionDesc>>,
        classdef: &Rc<StdRefCell<crate::annotator::classdesc::ClassDef>>,
        name: &str,
    ) -> DescEntry {
        use crate::annotator::description::{ClassDefKey, MethodDesc};
        let cd_key = ClassDefKey::from_classdef(classdef);
        DescEntry::Method(Rc::new(StdRefCell::new(MethodDesc::new(
            bk.clone(),
            crate::annotator::description::FuncDescEntry::plain(funcdesc.clone()),
            cd_key,
            Some(cd_key),
            name,
            std::collections::BTreeMap::new(),
        ))))
    }

    fn classdef_for(
        ann: &Rc<crate::annotator::annrpython::RPythonAnnotator>,
        name: &str,
    ) -> Rc<StdRefCell<crate::annotator::classdesc::ClassDef>> {
        use crate::flowspace::model::HostObject;
        let host = HostObject::new_class(name, vec![]);
        let DescEntry::Class(class_rc) = ann.bookkeeper.getdesc(&host).expect("bk.getdesc") else {
            unreachable!()
        };
        crate::annotator::classdesc::ClassDesc::getuniqueclassdef(&class_rc)
            .expect("getuniqueclassdef")
    }

    /// Insert a bare `Attribute(name)` into `classdef.attrs` so
    /// `ClassDef::get_owner(classdef, name)` (`classdesc.py`)
    /// returns the classdef. Bypasses `add_source_for_attribute`'s
    /// reflow / validation since these tests only need the dict membership.
    fn register_attr(classdef: &Rc<StdRefCell<crate::annotator::classdesc::ClassDef>>, name: &str) {
        use crate::annotator::classdesc::Attribute;
        let attr = Attribute::new(name);
        classdef.borrow_mut().attrs.insert(name.to_string(), attr);
    }

    #[test]
    fn methods_pbc_repr_new_succeeds_for_homogeneous_method_pbc() {
        // rpbc.py __init__ — single MethodDesc PBC builds a repr whose
        // methodname matches, classdef is the owner, and lowleveltype
        // tracks `r_im_self.lowleveltype`.
        let (ann, rtyper) = make_rtyper();
        let Some(fd) = function_entry(&ann.bookkeeper, "f").as_function() else {
            unreachable!()
        };
        let cdef = classdef_for(&ann, "C");
        register_attr(&cdef, "method_a");
        let m = method_entry(&ann.bookkeeper, &fd, &cdef, "method_a");
        let s_pbc = SomePBC::new(vec![m], false);
        let r = MethodsPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(r.class_name(), "MethodsPBCRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::MethodsPBCRepr);
        assert_eq!(r.methodname, "method_a");
        // upstream: `self.classdef = classdef.get_owner(methodname)` —
        // for a single-MethodDesc PBC with the method registered on
        // `C` itself, `get_owner` returns `C`.
        assert!(Rc::ptr_eq(&r.classdef, &cdef));
        // upstream: `self.lowleveltype = self.r_im_self.lowleveltype`.
        assert_eq!(r.lowleveltype(), r.r_im_self.lowleveltype());
    }

    #[test]
    fn methods_pbc_repr_new_rejects_mismatched_method_names() {
        // rpbc.py:1138-1142 — two MethodDescs with different `name`s
        // raise "cannot find a unique name under which the methods can
        // be found". Pyre surfaces it as a structured TyperError.
        let (ann, rtyper) = make_rtyper();
        let Some(fd_a) = function_entry(&ann.bookkeeper, "fa").as_function() else {
            unreachable!()
        };
        let Some(fd_b) = function_entry(&ann.bookkeeper, "fb").as_function() else {
            unreachable!()
        };
        let cdef = classdef_for(&ann, "C");
        register_attr(&cdef, "method_a");
        register_attr(&cdef, "method_b");
        let m_a = method_entry(&ann.bookkeeper, &fd_a, &cdef, "method_a");
        let m_b = method_entry(&ann.bookkeeper, &fd_b, &cdef, "method_b");
        let s_pbc = SomePBC::new(vec![m_a, m_b], false);
        let err = MethodsPBCRepr::new(&rtyper, s_pbc).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot find a unique name under which the methods"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn methods_pbc_repr_convert_const_none_returns_null_pointer() {
        // rpbc.py convert_const — `if method is None: return
        //                       nullptr(self.lowleveltype.TO)`.
        // The MethodsPBCRepr's lowleveltype tracks `r_im_self.lowleveltype`,
        // so the produced LLPtr's concretetype matches the target Ptr.
        let (ann, rtyper) = make_rtyper();
        let Some(fd) = function_entry(&ann.bookkeeper, "f").as_function() else {
            unreachable!()
        };
        let cdef = classdef_for(&ann, "PbcConvertConstC");
        register_attr(&cdef, "method_a");
        let m = method_entry(&ann.bookkeeper, &fd, &cdef, "method_a");
        let s_pbc = SomePBC::new(vec![m], false);
        let r = MethodsPBCRepr::new(&rtyper, s_pbc).unwrap();
        // r_im_self.setup() must run so null_instance can resolve the
        // object_type ForwardReference.
        crate::translator::rtyper::rmodel::Repr::setup(
            r.r_im_self.as_ref() as &dyn crate::translator::rtyper::rmodel::Repr
        )
        .expect("setup r_im_self");

        let c = r
            .convert_const(&crate::flowspace::model::ConstValue::None)
            .expect("convert_const(None)");
        assert_eq!(c.concretetype.as_ref(), Some(r.lowleveltype()));
        let crate::flowspace::model::ConstValue::LLPtr(ptr) = &c.value else {
            panic!(
                "MethodsPBCRepr.convert_const(None) must produce LLPtr, got {:?}",
                c.value
            );
        };
        assert!(!ptr.nonzero(), "null pointer must be null");
    }

    #[test]
    fn methods_pbc_repr_convert_const_rejects_non_bound_method() {
        // rpbc.py:1162 — `if getattr(method, 'im_func', None) is None:
        //                   raise TyperError("not a bound method: %r" % method)`.
        let (ann, rtyper) = make_rtyper();
        let Some(fd) = function_entry(&ann.bookkeeper, "f").as_function() else {
            unreachable!()
        };
        let cdef = classdef_for(&ann, "PbcConvertConstReject");
        register_attr(&cdef, "method_a");
        let m = method_entry(&ann.bookkeeper, &fd, &cdef, "method_a");
        let s_pbc = SomePBC::new(vec![m], false);
        let r = MethodsPBCRepr::new(&rtyper, s_pbc).unwrap();

        // Pass a non-BoundMethod HostObject (a class). MethodsPBCRepr
        // routes to the "not a bound method" branch.
        let cls = crate::flowspace::model::HostObject::new_class("Other", vec![]);
        let err = r
            .convert_const(&crate::flowspace::model::ConstValue::HostObject(cls))
            .unwrap_err();
        assert!(
            err.to_string().contains("not a bound method"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn somepbc_rtyper_makerepr_method_kind_routes_to_methods_pbc_repr() {
        // rpbc.py:53-54 — `elif issubclass(kind, MethodDesc):
        //                  getRepr = MethodsPBCRepr`.
        use crate::annotator::model::SomeValue;
        let (ann, rtyper) = make_rtyper();
        let Some(fd) = function_entry(&ann.bookkeeper, "f").as_function() else {
            unreachable!()
        };
        let cdef = classdef_for(&ann, "D");
        register_attr(&cdef, "method_x");
        let m = method_entry(&ann.bookkeeper, &fd, &cdef, "method_x");
        let s_pbc = SomePBC::new(vec![m], false);
        let r = rtyper.getrepr(&SomeValue::PBC(s_pbc)).unwrap();
        assert_eq!(r.repr_class_id(), ReprClassId::MethodsPBCRepr);
    }

    #[test]
    fn single_frozen_pbc_repr_convert_desc_identity_check_and_void_sentinel() {
        use crate::flowspace::model::ConstValue;
        let (ann, _rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let r = SingleFrozenPBCRepr::new(f.clone());

        // Same desc identity → Ok(Constant(None, Void)).
        let c = r.convert_desc(&f).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));

        // Different desc identity → rejected.
        let other = frozen_entry(&ann.bookkeeper, "frozen1");
        let err = r.convert_desc(&other).unwrap_err();
        assert!(err.to_string().contains("frozendesc identity mismatch"));
    }

    #[test]
    fn single_frozen_pbc_repr_convert_const_returns_void_none() {
        use crate::flowspace::model::ConstValue;
        let (ann, _rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let r = SingleFrozenPBCRepr::new(f);
        let c = r.convert_const(&ConstValue::Int(7)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));
    }

    #[test]
    fn multiple_unrelated_frozen_pbc_repr_has_address_lowleveltype() {
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        assert_eq!(r.lowleveltype(), &LowLevelType::Address);
        assert_eq!(r.class_name(), "MultipleUnrelatedFrozenPBCRepr");
        assert_eq!(
            r.repr_class_id(),
            ReprClassId::MultipleUnrelatedFrozenPBCRepr
        );
    }

    #[test]
    fn multiple_unrelated_frozen_pbc_repr_null_instance_emits_null_address_constant() {
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let null = r.null_instance();
        assert_eq!(null.concretetype, Some(LowLevelType::Address));
        assert!(matches!(
            null.value,
            ConstValue::LLAddress(crate::translator::rtyper::lltypesystem::lltype::_address::Null)
        ));
    }

    #[test]
    fn multiple_unrelated_frozen_pbc_repr_convert_const_none_returns_null_instance() {
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let c = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Address));
        assert!(matches!(
            c.value,
            ConstValue::LLAddress(crate::translator::rtyper::lltypesystem::lltype::_address::Null)
        ));
    }

    /// Non-HostObject non-None values surface a structured TyperError
    /// — upstream rpbc.py convert_const only feeds HostObjects on the
    /// non-None path, so non-HostObjects are upstream-impossible and
    /// the Rust port rejects them up-front.
    #[test]
    fn multiple_unrelated_frozen_pbc_repr_convert_const_non_hostobject_rejected() {
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let err = r.convert_const(&ConstValue::Int(7)).unwrap_err();
        assert!(
            err.to_string().contains("must be a HostObject"),
            "rejection message must be specific, got {err}"
        );
    }

    /// `MultipleUnrelatedFrozenPBCRepr.convert_desc(frozendesc)`
    /// (rpbc.py) routes the desc through its per-desc repr,
    /// extracts the `_ptr`, wraps it in `_address::Fake`, and caches
    /// the result keyed on `DescEntry::desc_key()`. This test drives
    /// the `r.lowleveltype is Void` arm — `SingleFrozenPBCRepr` is
    /// the per-desc repr for an isolated FrozenDesc and its lltype is
    /// `Void`, so `convert_desc` falls back to `create_instance()`.
    #[test]
    fn multiple_unrelated_frozen_pbc_repr_convert_desc_routes_through_void_per_desc_repr() {
        let (ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let c = r
            .convert_desc(&f)
            .expect("convert_desc Void per-desc repr arm");
        assert_eq!(c.concretetype, Some(LowLevelType::Address));
        let ConstValue::LLAddress(addr) = &c.value else {
            panic!("expected LLAddress, got {:?}", c.value);
        };
        assert!(
            matches!(
                addr,
                crate::translator::rtyper::lltypesystem::lltype::_address::Fake(_)
            ),
            "Void per-desc arm wraps a fresh `pbc` instance via fakeaddress"
        );

        // Repeated convert_desc on the same desc returns the cached value.
        let c2 = r.convert_desc(&f).unwrap();
        assert_eq!(c.value, c2.value);
    }

    #[test]
    fn getFrozenPBCRepr_single_desc_without_can_be_none_returns_single_frozen_repr() {
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = getFrozenPBCRepr(&rtyper, &s_pbc).unwrap();
        assert_eq!(r.class_name(), "SingleFrozenPBCRepr");
    }

    #[test]
    fn getFrozenPBCRepr_multi_desc_common_none_access_set_returns_multiple_frozen() {
        // Two FrozenDescs without any `getattrfamily` touch share the
        // same `queryattrfamily() == None` result, so upstream routes them
        // to the `access is None` arm — a zero-field MultipleFrozenPBCRepr
        // (rpbc.py + 758). The None arm is a singleton cached at
        // `pbc_reprs[None]`.
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let g = frozen_entry(&ann.bookkeeper, "frozen1");
        let s_pbc = SomePBC::new(vec![f, g], false);
        let r1 = getFrozenPBCRepr(&rtyper, &s_pbc).unwrap();
        assert_eq!(r1.class_name(), "MultipleFrozenPBCRepr");
        let r2 = getFrozenPBCRepr(&rtyper, &s_pbc).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(&r1, &r2),
            "the None-access MultipleFrozenPBCRepr must be a cached singleton"
        );
    }

    #[test]
    fn multiple_frozen_pbc_repr_none_access_convert_desc_accepts_function_desc() {
        // An uncallable multi-FunctionDesc PBC (each `queryattrfamily() ==
        // None`) is routed by `getFrozenPBCRepr` to the zero-field
        // None-access arm. upstream `MultipleFrozenPBCRepr.convert_desc`
        // (rpbc.py) does not require a FrozenDesc, so converting a
        // FunctionDesc there must succeed: the fieldmap is empty, so nothing
        // is read off the desc and an empty `pbc` instance is materialised.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "func0");
        let g = function_entry(&ann.bookkeeper, "func1");
        let s_pbc = SomePBC::new(vec![f.clone(), g], false);
        let r = getFrozenPBCRepr(&rtyper, &s_pbc).unwrap();
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
        Repr::setup(r.as_ref()).expect("setup");
        let c = r
            .convert_desc(&f)
            .expect("None-access convert_desc must accept a FunctionDesc");
        assert!(
            matches!(c.value, ConstValue::LLPtr(_)),
            "expected a zero-field pbc instance pointer, got {:?}",
            c.value
        );
    }

    #[test]
    fn mu_repr_rtype_getattr_requires_constant_s_result() {
        // rpbc.py rtype_getattr — getattr on a constant PBC must yield a
        // constant; non-constant s_result raises TyperError.
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (_ann, rtyper) = make_rtyper();
        let mu = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let spaceop = SpaceOperation::new(
            OpKind::GetAttr.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // s_result is None (non-constant) → TyperError.
        hop.s_result
            .replace(Some(crate::annotator::model::SomeValue::Impossible));
        hop.r_result.replace(Some(
            std::sync::Arc::new(MultipleUnrelatedFrozenPBCRepr::new(&rtyper))
                as std::sync::Arc<dyn Repr>,
        ));
        let err = mu.rtype_getattr(&hop).unwrap_err();
        assert!(
            err.to_string()
                .contains("getattr on a constant PBC returns a non-constant"),
            "unexpected: {err}",
        );
    }

    #[test]
    fn pair_mu_mu_rtype_is_emits_adr_eq_against_bool_result() {
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let _ = ann;

        // Single MU repr instance shared across both arg slots and the
        // helper call so `inputargs`'s no-op convert path fires.
        let mu_arc: std::sync::Arc<dyn Repr> =
            std::sync::Arc::new(MultipleUnrelatedFrozenPBCRepr::new(&rtyper));

        // Address-typed Variables — inputargs reads args_v / args_r.
        use crate::flowspace::model::Constant;
        let make_address_var = || {
            let v = Variable::new();
            v.set_concretetype(Some(LowLevelType::Address));
            v
        };
        let v1 = Hlvalue::Variable(make_address_var());
        let v2 = Hlvalue::Variable(make_address_var());
        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Bool));
        let spaceop = SpaceOperation::new(
            OpKind::Is.opname(),
            vec![v1.clone(), v2.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Use non-const SomeValue so inputarg skips the const_ shortcut
        // and reaches convertvar directly. Semantically upstream passes
        // SomePBC here; the unit test only cares about the op surface.
        hop.args_v.borrow_mut().push(v1);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(mu_arc.clone()));
        hop.args_v.borrow_mut().push(v2);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(mu_arc.clone()));

        let result = pair_mu_mu_rtype_is_(&*mu_arc, &*mu_arc, &hop).unwrap();
        // adr_eq is emitted as a ll op with Bool result.
        let ops = hop.llops.borrow();
        let adr_eq = ops.ops.iter().find(|op| op.opname == "adr_eq");
        assert!(
            adr_eq.is_some(),
            "expected adr_eq op to be emitted, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>(),
        );
        // Result is a variable pinned to Bool.
        match result {
            Hlvalue::Variable(v) => {
                assert_eq!(v.concretetype(), Some(LowLevelType::Bool));
            }
            Hlvalue::Constant(c) => panic!("unexpected constant result {c:?}"),
        }
        // Silence the unused-const warning for the Constant import.
        let _ = Constant::new(crate::flowspace::model::ConstValue::None);
    }

    #[test]
    fn getFrozenPBCRepr_unrelated_branch_reuses_rtyper_pbc_reprs_singleton() {
        // upstream rpbc.py caches a single MultipleUnrelatedFrozenPBCRepr
        // under `rtyper.pbc_reprs['unrelated']`. Two calls with
        // different SomePBCs (both landing in the unrelated branch)
        // must return the same Arc.
        let (ann, rtyper) = make_rtyper();

        let (f1, g1) = (
            frozen_entry(&ann.bookkeeper, "a"),
            frozen_entry(&ann.bookkeeper, "b"),
        );
        let (DescEntry::Frozen(a_rc), DescEntry::Frozen(b_rc)) = (f1.clone(), g1.clone()) else {
            unreachable!()
        };
        let _ = a_rc.borrow().getattrfamily().unwrap();
        let _ = b_rc.borrow().getattrfamily().unwrap();
        let s_pbc_1 = SomePBC::new(vec![f1, g1], false);
        let r1 = getFrozenPBCRepr(&rtyper, &s_pbc_1).unwrap();

        let (f2, g2) = (
            frozen_entry(&ann.bookkeeper, "c"),
            frozen_entry(&ann.bookkeeper, "d"),
        );
        let (DescEntry::Frozen(c_rc), DescEntry::Frozen(d_rc)) = (f2.clone(), g2.clone()) else {
            unreachable!()
        };
        let _ = c_rc.borrow().getattrfamily().unwrap();
        let _ = d_rc.borrow().getattrfamily().unwrap();
        let s_pbc_2 = SomePBC::new(vec![f2, g2], false);
        let r2 = getFrozenPBCRepr(&rtyper, &s_pbc_2).unwrap();

        assert!(
            std::sync::Arc::ptr_eq(&r1, &r2),
            "rtyper.pbc_reprs['unrelated'] must return the same Arc on repeat lookups",
        );
    }

    #[test]
    fn getFrozenPBCRepr_multi_desc_unrelated_access_sets_routes_to_unrelated_repr() {
        // Give each FrozenDesc its own attrfamily (distinct Rc ptrs),
        // so `queryattrfamily()` returns distinct Python-object
        // identities → upstream rpbc.py:619-625 routes to
        // MultipleUnrelatedFrozenPBCRepr.
        let (ann, rtyper) = make_rtyper();
        let f_entry = frozen_entry(&ann.bookkeeper, "frozen0");
        let g_entry = frozen_entry(&ann.bookkeeper, "frozen1");
        let DescEntry::Frozen(f_rc) = f_entry.clone() else {
            unreachable!("frozen_entry returns DescEntry::Frozen");
        };
        let DescEntry::Frozen(g_rc) = g_entry.clone() else {
            unreachable!("frozen_entry returns DescEntry::Frozen");
        };
        // Materialize distinct FrozenAttrFamily singletons per desc.
        let fam_f = f_rc.borrow().getattrfamily().unwrap();
        let fam_g = g_rc.borrow().getattrfamily().unwrap();
        assert!(!Rc::ptr_eq(&fam_f, &fam_g));
        let s_pbc = SomePBC::new(vec![f_entry, g_entry], false);
        let r = getFrozenPBCRepr(&rtyper, &s_pbc).unwrap();
        assert_eq!(r.class_name(), "MultipleUnrelatedFrozenPBCRepr");
        assert_eq!(r.lowleveltype(), &LowLevelType::Address);
    }

    /// `getFrozenPBCRepr` for a multi-FrozenDesc PBC whose descs
    /// share an `Rc::ptr_eq`-identical `FrozenAttrFamily` (after
    /// `mergeattrfamilies`) returns a fresh [`MultipleFrozenPBCRepr`]
    /// keyed in `rtyper.pbc_reprs[Access(family_id)]`. Repeated calls
    /// reuse the cached entry.
    #[test]
    fn getFrozenPBCRepr_multi_desc_shared_family_returns_multiple_frozen_repr() {
        let (ann, rtyper) = make_rtyper();
        let f_entry = frozen_entry(&ann.bookkeeper, "frozen0");
        let g_entry = frozen_entry(&ann.bookkeeper, "frozen1");
        let DescEntry::Frozen(f_rc) = &f_entry else {
            unreachable!();
        };
        let DescEntry::Frozen(g_rc) = &g_entry else {
            unreachable!();
        };
        // Share a single FrozenAttrFamily across both descs.
        f_rc.borrow()
            .mergeattrfamilies(&[&g_rc.borrow()])
            .expect("mergeattrfamilies");
        let fam_f = f_rc.borrow().getattrfamily().unwrap();
        let fam_g = g_rc.borrow().getattrfamily().unwrap();
        assert!(
            Rc::ptr_eq(&fam_f, &fam_g),
            "post-merge both descs must share the same family Rc"
        );

        let s_pbc = SomePBC::new(vec![f_entry.clone(), g_entry.clone()], false);
        let r = getFrozenPBCRepr(&rtyper, &s_pbc).expect("MultipleFrozenPBCRepr");
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
        // lowleveltype is Ptr(ForwardReference) before setup.
        let LowLevelType::Ptr(ptr) = r.lowleveltype() else {
            panic!("MultipleFrozenPBCRepr lltype must be Ptr");
        };
        assert!(
            matches!(
                &ptr.TO,
                crate::translator::rtyper::lltypesystem::lltype::PtrTarget::ForwardReference(_)
            ),
            "lowleveltype TO must be ForwardReference until _setup_repr runs"
        );

        // Calling getFrozenPBCRepr again with an equivalent SomePBC
        // should hit the `pbc_reprs[Access(...)]` singleton cache
        // (rpbc.py).
        let s_pbc2 = SomePBC::new(vec![f_entry, g_entry], false);
        let r2 = getFrozenPBCRepr(&rtyper, &s_pbc2).expect("cached");
        assert!(
            Arc::ptr_eq(&r, &r2),
            "second call must return the same Arc<dyn Repr>"
        );
    }

    /// `MultipleFrozenPBCRepr._setup_repr` runs `_setup_repr_fields`,
    /// converts each access-set attr's `s_value` to a Repr, mangles
    /// the field name with `mangle('pbc', attr)`, and resolves
    /// `pbc_type` to a `Struct('pbc', ...)` carrying those fields.
    #[test]
    fn multiple_frozen_pbc_repr_setup_resolves_pbc_type_struct_with_mangled_fields() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        let (ann, rtyper) = make_rtyper();
        let f_entry = frozen_entry(&ann.bookkeeper, "frozen0");
        let g_entry = frozen_entry(&ann.bookkeeper, "frozen1");
        let DescEntry::Frozen(f_rc) = &f_entry else {
            unreachable!();
        };
        let DescEntry::Frozen(g_rc) = &g_entry else {
            unreachable!();
        };
        f_rc.borrow()
            .mergeattrfamilies(&[&g_rc.borrow()])
            .expect("mergeattrfamilies");
        // Seed a single Int-typed attr `x` on the family.
        {
            let fam = f_rc.borrow().getattrfamily().unwrap();
            fam.borrow_mut()
                .attrs
                .insert("x".into(), SomeValue::Integer(SomeInteger::default()));
        }

        let s_pbc = SomePBC::new(vec![f_entry, g_entry], false);
        let r = getFrozenPBCRepr(&rtyper, &s_pbc).unwrap();
        Repr::setup(r.as_ref()).expect("setup MultipleFrozenPBCRepr");

        // pbc_type ForwardReference resolved to Struct('pbc', ('pbc_x', Signed)).
        let LowLevelType::Ptr(ptr) = r.lowleveltype() else {
            panic!("MultipleFrozenPBCRepr lltype must be Ptr");
        };
        let to = ptr.TO.clone();
        let resolved = match to {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::ForwardReference(fwd) => {
                fwd.resolved().expect("pbc_type resolved after setup")
            }
            other => panic!("Ptr target must remain ForwardReference variant, got {other:?}"),
        };
        let LowLevelType::Struct(body) = resolved else {
            panic!("resolved pbc_type must be Struct");
        };
        assert_eq!(body._name, "pbc");
        assert!(
            body._flds.get("pbc_x").is_some(),
            "Struct('pbc', ...) must carry mangled `pbc_x` field"
        );
    }

    /// `MultipleFrozenPBCRepr.convert_desc(frozendesc)` with attr
    /// values seeded into `frozendesc.attrcache` returns a live
    /// `_ptr` whose struct field equals the converted value. Cache
    /// is populated keyed on the desc identity.
    #[test]
    fn multiple_frozen_pbc_repr_convert_desc_writes_attr_value_into_pbc_struct() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        let (ann, rtyper) = make_rtyper();
        let f_entry = frozen_entry(&ann.bookkeeper, "frozen0");
        let g_entry = frozen_entry(&ann.bookkeeper, "frozen1");
        let DescEntry::Frozen(f_rc) = &f_entry else {
            unreachable!();
        };
        let DescEntry::Frozen(g_rc) = &g_entry else {
            unreachable!();
        };
        f_rc.borrow()
            .mergeattrfamilies(&[&g_rc.borrow()])
            .expect("mergeattrfamilies");
        {
            let fam = f_rc.borrow().getattrfamily().unwrap();
            fam.borrow_mut()
                .attrs
                .insert("x".into(), SomeValue::Integer(SomeInteger::default()));
        }
        // Per-desc attrcache seed.
        f_rc.borrow()
            .create_new_attribute("x", ConstValue::Int(7))
            .unwrap();
        g_rc.borrow()
            .create_new_attribute("x", ConstValue::Int(11))
            .unwrap();

        let s_pbc = SomePBC::new(vec![f_entry.clone(), g_entry.clone()], false);
        let r = getFrozenPBCRepr(&rtyper, &s_pbc).unwrap();
        Repr::setup(r.as_ref()).expect("setup");

        let c_f = r.convert_desc(&f_entry).expect("convert_desc f");
        let ConstValue::LLPtr(p_f) = &c_f.value else {
            panic!("convert_desc must return LLPtr");
        };
        assert_eq!(
            p_f.getattr("pbc_x").unwrap(),
            crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Signed(7)
        );

        let c_g = r.convert_desc(&g_entry).expect("convert_desc g");
        let ConstValue::LLPtr(p_g) = &c_g.value else {
            panic!("convert_desc must return LLPtr");
        };
        assert_eq!(
            p_g.getattr("pbc_x").unwrap(),
            crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Signed(11)
        );

        // Repeated convert_desc on the same frozendesc returns the
        // identity-equal cached `_ptr`.
        let c_f2 = r.convert_desc(&f_entry).unwrap();
        let ConstValue::LLPtr(p_f2) = &c_f2.value else {
            unreachable!();
        };
        assert_eq!(p_f._hashable_identity(), p_f2._hashable_identity());
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_frozen_desc_without_none_returns_single_frozen_repr() {
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "SingleFrozenPBCRepr");
    }

    fn class_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
        use crate::annotator::classdesc::ClassDesc;
        use crate::flowspace::model::HostObject;
        let pyobj = HostObject::new_class(name, vec![]);
        DescEntry::Class(Rc::new(StdRefCell::new(ClassDesc::new_shell(
            bk,
            pyobj,
            name.to_string(),
        ))))
    }

    #[test]
    fn classes_pbc_repr_constant_branch_has_void_lowleveltype() {
        let (ann, rtyper) = make_rtyper();
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], false);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        assert_eq!(r.class_name(), "ClassesPBCRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::ClassesPBCRepr);
    }

    #[test]
    fn classes_pbc_repr_non_constant_branch_uses_classtype_lowleveltype() {
        let (ann, rtyper) = make_rtyper();
        // can_be_none=true with a single desc: upstream's
        // `s_pbc.is_constant()` returns False, so __init__ routes
        // through `getlowleveltype()` and the repr's lltype is
        // CLASSTYPE.
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(
            r.lowleveltype(),
            &crate::translator::rtyper::rclass::CLASSTYPE.clone()
        );
    }

    #[test]
    fn classes_pbc_repr_convert_desc_identity_check() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c.clone()], false);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        // Same-identity desc → Void-None sentinel.
        let out = r.convert_desc(&c).unwrap();
        assert_eq!(out.concretetype, Some(LowLevelType::Void));
        assert!(matches!(out.value, ConstValue::None));

        // Other-identity desc → TyperError.
        let other = class_entry(&ann.bookkeeper, "D");
        let err = r.convert_desc(&other).unwrap_err();
        assert!(err.to_string().contains("not in"));
    }

    #[test]
    fn classes_pbc_repr_convert_const_none_returns_void_sentinel() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], false);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        let out = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(out.concretetype, Some(LowLevelType::Void));
        assert!(matches!(out.value, ConstValue::None));

        // Non-HostObject value rejected with a structured TyperError
        // (upstream surfaces `bk.getdesc` AttributeError on the host
        // side; pyre catches the type mismatch up-front).
        let err = r.convert_const(&ConstValue::Int(7)).unwrap_err();
        assert!(err.to_string().contains("must be a HostObject"));
    }

    #[test]
    fn classes_pbc_repr_convert_const_none_on_non_void_returns_null_ptr() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        // can_be_none=true forces the non-constant arm, so lltype is
        // CLASSTYPE = Ptr(OBJECT_VTABLE) and convert_const(None) emits
        // `nullptr(T.TO)` as a `ConstValue::LLPtr` carrying the
        // `_defl()` null sentinel.
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        let out = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(
            out.concretetype.as_ref(),
            Some(&crate::translator::rtyper::rclass::CLASSTYPE.clone())
        );
        let ConstValue::LLPtr(p) = &out.value else {
            panic!(
                "non-Void None must produce ConstValue::LLPtr, got {:?}",
                out.value
            );
        };
        // _defl produces a null pointer (no underlying object).
        assert!(!p.nonzero(), "expected null pointer from _defl()");
    }

    #[test]
    fn classes_pbc_repr_convert_const_non_none_routes_through_bookkeeper() {
        use crate::flowspace::model::{ConstValue, HostObject};
        let (ann, rtyper) = make_rtyper();
        // Route the desc construction through `bk.getdesc` so the
        // desc-identity used by `s_pbc` matches the one returned when
        // `convert_const` does the roundtrip — `class_entry` builds a
        // fresh ClassDesc shell that is NOT cached in `bk.descs`.
        let host = HostObject::new_class("C", vec![]);
        let c = ann
            .bookkeeper
            .getdesc(&host)
            .expect("bk.getdesc(host) for fresh class");
        let s_pbc = SomePBC::new(vec![c], false);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        let out = r
            .convert_const(&ConstValue::HostObject(host))
            .expect("non-None HostObject must route through bk.getdesc + convert_desc");
        assert_eq!(out.concretetype, Some(LowLevelType::Void));
        assert!(matches!(out.value, ConstValue::None));
    }

    #[test]
    fn classes_pbc_repr_convert_desc_non_void_returns_vtable_pointer() {
        use crate::flowspace::model::HostObject;
        use crate::translator::rtyper::rclass::CLASSTYPE;
        let (ann, rtyper) = make_rtyper();
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        // Route the ClassDesc through `bk.getdesc` so it's cached and
        // shared between the SomePBC.descriptions seed and the
        // `getuniqueclassdef` lookup invoked by convert_desc.
        let host = HostObject::new_class("C", vec![]);
        let c = ann
            .bookkeeper
            .getdesc(&host)
            .expect("bk.getdesc fresh class");
        // Drive `getuniqueclassdef` so the classdef exists, then assign
        // minimal inheritance ids — `init_vtable → fill_vtable_root`
        // requires `classdef.minid/maxid`. The values are arbitrary
        // here (id assignment is normalizecalls.assign_inheritance_ids
        // territory).
        let DescEntry::Class(class_rc) = &c else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc)
            .expect("getuniqueclassdef");
        {
            let mut cd = classdef.borrow_mut();
            cd.minid = Some(2);
            cd.maxid = Some(3);
        }
        let s_pbc = SomePBC::new(vec![c.clone()], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();
        // Set up the ClassRepr (vtable_type ForwardReference resolves).
        let r_subclass =
            crate::translator::rtyper::rclass::getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r_subclass.as_repr()).expect("setup classrepr");

        let out = r.convert_desc(&c).expect("convert_desc non-Void");
        assert_eq!(out.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
        let ConstValue::LLPtr(p) = &out.value else {
            panic!("expected ConstValue::LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero(), "vtable pointer must be live (non-null)");
    }

    /// `ClassesPBCRepr.get_access_set(attrname)` resolves the
    /// `ClassAttrFamily` shared by the s_pbc.descriptions and returns
    /// it alongside the `ClassRepr` of the family's `commonbase`. The
    /// `commonbase` field is populated by
    /// [`merge_classpbc_getattr_into_classdef`]; this test runs that
    /// pass first to drive the live setup end-to-end.
    #[test]
    fn classes_pbc_repr_get_access_set_returns_family_and_commonbase_class_repr() {
        use crate::annotator::classdesc::ClassDesc;
        use crate::flowspace::model::HostObject;
        use crate::translator::rtyper::normalizecalls::merge_classpbc_getattr_into_classdef;

        let (ann, rtyper) = make_rtyper();

        // Build Root + Leaf classdescs sharing a ClassAttrFamily for "x".
        // Both descs must live in `bookkeeper.descs` so the merge pass
        // can resolve `family.descs` keys.
        let root_host = HostObject::new_class("pkg.Root", vec![]);
        let root_desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            root_host.clone(),
            "pkg.Root".to_string(),
        )));
        let leaf_host = HostObject::new_class("pkg.Leaf", vec![root_host.clone()]);
        let leaf_desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            leaf_host.clone(),
            "pkg.Leaf".to_string(),
        )));
        leaf_desc.borrow_mut().basedesc = Some(root_desc.clone());
        {
            let mut descs = ann.bookkeeper.descs.borrow_mut();
            descs.insert(root_host.clone(), DescEntry::Class(root_desc.clone()));
            descs.insert(leaf_host.clone(), DescEntry::Class(leaf_desc.clone()));
        }
        // Drive ClassDef creation up front — getuniqueclassdef caches
        // and `merge_classpbc_getattr_into_classdef` calls it again.
        let root_cd = ClassDesc::getuniqueclassdef(&root_desc).unwrap();
        let _leaf_cd = ClassDesc::getuniqueclassdef(&leaf_desc).unwrap();

        // Wire the ClassAttrFamily for "x" via UnionFind union.
        let root_key = root_desc.borrow().identity;
        let leaf_key = leaf_desc.borrow().identity;
        ann.bookkeeper.with_classpbc_attr_families("x", |uf| {
            uf.find(root_key);
            uf.union(root_key, leaf_key);
        });

        merge_classpbc_getattr_into_classdef(&ann).expect("merge_classpbc_getattr_into_classdef");

        // Build the ClassesPBCRepr over both descs.
        let s_pbc = SomePBC::new(
            vec![
                DescEntry::Class(root_desc.clone()),
                DescEntry::Class(leaf_desc.clone()),
            ],
            false,
        );
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        // Call get_access_set("x") and verify both halves of the tuple.
        let (access, class_repr) = r.get_access_set("x").expect("get_access_set");
        // The returned access matches what merge wrote to `family.commonbase`.
        let cb = access
            .borrow()
            .commonbase
            .clone()
            .expect("commonbase populated");
        assert!(
            Rc::ptr_eq(&cb, &root_cd),
            "commonbase must equal Root classdef"
        );
        // class_repr is the ClassRepr for Root.
        let class_repr_classdef = match &class_repr {
            crate::translator::rtyper::rclass::ClassReprArc::Inst(inst) => inst.classdef(),
            _ => panic!("commonbase Root must produce ClassReprArc::Inst"),
        };
        assert!(
            Rc::ptr_eq(&class_repr_classdef, &root_cd),
            "class_repr.classdef must equal Root"
        );
    }

    /// `rtype_getattr` const-result fast path (rpbc.py):
    /// when `hop.s_result.is_constant()` the body short-circuits to
    /// `inputconst(hop.r_result, hop.s_result.const)` BEFORE consulting
    /// the access set or the attrname. The general / `__name__` paths
    /// are never reached.
    #[test]
    fn classes_pbc_repr_rtype_getattr_const_result_short_circuits_to_inputconst() {
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{Hlvalue, HostObject, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let host = HostObject::new_class("pkg.C", vec![]);
        let desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            host.clone(),
            "pkg.C".to_string(),
        )));
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(host, DescEntry::Class(desc.clone()));
        let s_pbc = SomePBC::new(vec![DescEntry::Class(desc)], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        // Build a getattr hop where s_result is a *constant* integer —
        // should short-circuit to `inputconst` and NEVER touch
        // get_access_set (we leave the bookkeeper unwired so the
        // general path would fail loudly if reached).
        let spaceop = SpaceOperation::new(
            OpKind::GetAttr.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        let mut s_int = SomeInteger::default();
        s_int.base.const_box = Some(Constant::new(ConstValue::Int(7)));
        hop.s_result.replace(Some(SomeValue::Integer(s_int)));
        hop.r_result
            .replace(Some(
                std::sync::Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")))
                    as std::sync::Arc<dyn Repr>,
            ));

        let out = r.rtype_getattr(&hop).expect("const-result path").unwrap();
        let Hlvalue::Constant(c) = out else {
            panic!("const-result must produce a Constant, got {out:?}");
        };
        assert_eq!(c.value, ConstValue::Int(7));
    }

    /// `rtype_getattr('__name__')` (rpbc.py) routes through
    /// `rootclass_repr` and emits `genop('getfield', [vcls, 'name'],
    /// resulttype=Ptr(rstr.STR))`. Body uncloses now that
    /// `rstr.STR` + `OBJECT_VTABLE.name` have landed.
    #[test]
    fn classes_pbc_repr_rtype_getattr_name_arm_emits_getfield_into_strptr() {
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::model::{SomeString, SomeValue};
        use crate::flowspace::model::{Hlvalue, HostObject, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rmodel::SimplePointerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let host = HostObject::new_class("pkg.C", vec![]);
        let desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            host.clone(),
            "pkg.C".to_string(),
        )));
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(host, DescEntry::Class(desc.clone()));
        let s_pbc = SomePBC::new(vec![DescEntry::Class(desc.clone())], true);
        let r_arc: std::sync::Arc<dyn Repr> =
            std::sync::Arc::new(ClassesPBCRepr::new(&rtyper, s_pbc).unwrap());

        // Build a hop with non-constant s_result + args_s[1] = Str("__name__").
        let spaceop = SpaceOperation::new(
            OpKind::GetAttr.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.s_result.replace(Some(SomeValue::Impossible));
        // r_result for __name__ is `Ptr(STR)` — use SimplePointerRepr
        // built over `STRPTR`.
        let strptr = crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone();
        hop.r_result.replace(Some(
            std::sync::Arc::new(SimplePointerRepr::new(strptr.clone())) as std::sync::Arc<dyn Repr>,
        ));
        // args_s[0] = receiver (the class PBC), args_s[1] = the
        // attrname constant. Pre-set args_r[0] to the rootclass_repr
        // so `inputargs(class_repr, Void)`'s convertvar takes the
        // identity short-circuit (ClassesPBCRepr → RootClassRepr is
        // a same-lltype conversion that has no explicit pairtype
        // entry yet).
        let root_repr_arc: std::sync::Arc<dyn Repr> = rtyper
            .rootclass_repr
            .borrow()
            .clone()
            .expect("rootclass_repr must be initialised");
        let v_recv = Variable::new();
        v_recv.set_concretetype(Some(root_repr_arc.lowleveltype().clone()));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_recv));
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(root_repr_arc.clone()));
        let mut s_attr = SomeString::default();
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("__name__")));
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::byte_str("__name__"),
                LowLevelType::Void,
            )));
        hop.args_s.borrow_mut().push(SomeValue::String(s_attr));
        hop.args_r.borrow_mut().push(None);

        let out = r_arc
            .rtype_getattr(&hop)
            .expect("rtype_getattr __name__")
            .expect("non-empty result");
        let Hlvalue::Variable(_) = out else {
            panic!("__name__ arm must return a Variable carrying the genop result");
        };
        let ops = hop.llops.borrow();
        let getfield = ops
            .ops
            .iter()
            .find(|op| op.opname == "getfield")
            .expect("getfield op must be emitted");
        // args[1] is the field-name Void constant `name`.
        let Hlvalue::Constant(field_const) = &getfield.args[1] else {
            panic!("getfield arg[1] must be a Constant");
        };
        assert_eq!(field_const.value, ConstValue::byte_str("name"));
        assert_eq!(field_const.concretetype, Some(LowLevelType::Void));
    }

    /// `get_access_set` returns a structured `MissingRTypeAttribute`
    /// when no ClassAttrFamily exists for the attrname. Upstream
    /// rpbc.py raises directly; pyre surfaces it as a TyperError.
    #[test]
    fn classes_pbc_repr_get_access_set_unknown_attr_raises_missing_rtype_attribute() {
        use crate::annotator::classdesc::ClassDesc;
        use crate::flowspace::model::HostObject;

        let (ann, rtyper) = make_rtyper();
        let host = HostObject::new_class("pkg.C", vec![]);
        let desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            host.clone(),
            "pkg.C".to_string(),
        )));
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(host, DescEntry::Class(desc.clone()));

        let s_pbc = SomePBC::new(vec![DescEntry::Class(desc)], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        let err = r.get_access_set("nonexistent").unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("nonexistent"));
    }

    /// Single-class instantiation whose class has `__init__`
    /// (`ClassesPBCRepr.redispatch_call`): the lowered ops malloc the
    /// instance and `direct_call` the init graph with that instance as
    /// the first argument.
    #[test]
    fn classes_pbc_repr_simple_call_with_init_mallocs_then_direct_calls_init() {
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::classdesc::{ClassDesc, ClassDictEntry};
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInstance, SomeValue};
        use crate::flowspace::model::{BlockRefExt, Hlvalue, HostObject, SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let host = HostObject::new_class("pkg.C", vec![]);
        let class_entry = ann.bookkeeper.getdesc(&host).expect("getdesc");
        let DescEntry::Class(class_rc) = &class_entry else {
            unreachable!();
        };
        let classdef = ClassDesc::getuniqueclassdef(class_rc).expect("getuniqueclassdef");
        {
            let mut cd = classdef.borrow_mut();
            cd.minid = Some(2);
            cd.maxid = Some(3);
        }
        let s_inst = SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            Default::default(),
        ));

        let sig = crate::flowspace::argument::Signature::new(vec!["self".to_string()], None, None);
        let fd = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "__init__",
            sig,
            None,
            None,
        )));
        let init_graph = {
            let arg_var = Variable::named("self");
            arg_var.annotation.replace(Some(Rc::new(s_inst.clone())));
            let arg = Hlvalue::Variable(arg_var);
            let startblock = crate::flowspace::model::Block::shared(vec![arg.clone()]);
            let ret_var = Variable::new();
            ret_var
                .annotation
                .replace(Some(Rc::new(crate::annotator::model::s_none())));
            let graph = crate::flowspace::model::FunctionGraph::with_return_var(
                "__init__",
                startblock.clone(),
                Hlvalue::Variable(ret_var),
            );
            let link = Rc::new(StdRef::new(crate::flowspace::model::Link::new(
                vec![arg],
                Some(graph.returnblock.clone()),
                None,
            )));
            startblock.closeblock(vec![link]);
            Rc::new(PyGraph {
                graph: Rc::new(StdRef::new(graph)),
                func: crate::flowspace::model::GraphFunc::new(
                    "__init__",
                    crate::flowspace::model::Constant::new(ConstValue::Dict(Default::default())),
                ),
                signature: StdRef::new(crate::flowspace::argument::Signature::new(
                    vec!["self".to_string()],
                    None,
                    None,
                )),
                defaults: StdRef::new(None),
                access_directly: std::cell::Cell::new(false),
            })
        };
        fd.borrow()
            .cache
            .borrow_mut()
            .insert(GraphCacheKey::None, init_graph.clone());
        let call_args = ArgumentsForTranslation::new(vec![Some(s_inst.clone())], None, None);
        FunctionDesc::consider_call_site(
            std::slice::from_ref(&fd),
            &call_args,
            &crate::annotator::model::s_none(),
            None,
        )
        .expect("consider_call_site");
        class_rc.borrow_mut().classdict.insert(
            "__init__".into(),
            ClassDictEntry::Desc(DescEntry::function(fd)),
        );

        let s_pbc = SomePBC::new(vec![class_entry.clone()], false);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc.clone()).unwrap();
        let r_dyn: std::sync::Arc<dyn Repr> = std::sync::Arc::new(r);

        let class_const = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::HostObject(host),
            LowLevelType::Void,
        ));
        let result_h = Hlvalue::Variable(Variable::new());
        let spaceop = SpaceOperation::new("simple_call", vec![class_const.clone()], result_h);
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        *hop.args_v.borrow_mut() = vec![class_const];
        *hop.args_s.borrow_mut() = vec![SomeValue::PBC(s_pbc)];
        *hop.args_r.borrow_mut() = vec![Some(r_dyn.clone())];
        *hop.s_result.borrow_mut() = Some(s_inst.clone());
        *hop.r_result.borrow_mut() = Some(
            crate::translator::rtyper::rclass::getinstancerepr(
                &rtyper,
                Some(&classdef),
                crate::translator::rtyper::rclass::Flavor::Gc,
            )
            .expect("getinstancerepr") as std::sync::Arc<dyn Repr>,
        );

        let result = r_dyn
            .rtype_simple_call(&hop)
            .expect("rtype_simple_call with __init__");
        let Some(Hlvalue::Variable(inst)) = result else {
            panic!("redispatch_call must return the new instance variable, got {result:?}");
        };

        let ops = hop.llops.borrow();
        let malloc = ops
            .ops
            .iter()
            .find(|op| op.opname == "malloc")
            .expect("instance malloc");
        let Hlvalue::Variable(malloc_v) = &malloc.result else {
            panic!("malloc result must be a Variable, got {:?}", malloc.result);
        };
        assert_eq!(
            &inst, malloc_v,
            "returned instance must be the malloc result"
        );
        let direct = ops
            .ops
            .iter()
            .find(|op| op.opname == "direct_call")
            .expect("direct_call of __init__");
        assert!(
            const_points_at_graph(&direct.args[0], &init_graph),
            "direct_call target must be the __init__ graph"
        );
        let Hlvalue::Variable(call_self) = &direct.args[1] else {
            panic!(
                "direct_call first argument must be the new instance, got {:?}",
                direct.args[1]
            );
        };
        assert_eq!(
            &inst, call_self,
            "direct_call's first argument must be the malloc'd instance"
        );
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_class_desc_routes_to_classes_pbc_repr() {
        let (ann, rtyper) = make_rtyper();
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], false);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "ClassesPBCRepr");
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_frozen_desc_with_can_be_none_returns_multiple_frozen() {
        // Single FrozenDesc with `can_be_None=true` falls out of the
        // SingleFrozenPBCRepr fast-path, hits getFrozenPBCRepr's multi-desc
        // branch, and — with only one access-set (None) — routes to the
        // zero-field MultipleFrozenPBCRepr no-attrs arm.
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let s_pbc = SomePBC::new(vec![f], true);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
    }

    #[test]
    fn function_repr_convert_desc_or_const_routes_desc_through_convert_desc() {
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::rmodel::DescOrConst;
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f.clone()], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();

        // Desc arm — routes through convert_desc, which returns the
        // Void-None sentinel.
        let c1 = r
            .convert_desc_or_const(&DescOrConst::Desc(f.clone()))
            .unwrap();
        assert!(matches!(c1.value, ConstValue::None));
        assert_eq!(c1.concretetype, Some(LowLevelType::Void));

        // Const arm — routes through convert_const, same outcome.
        let c2 = r
            .convert_desc_or_const(&DescOrConst::Const(Constant::new(ConstValue::Int(7))))
            .unwrap();
        assert!(matches!(c2.value, ConstValue::None));
        assert_eq!(c2.concretetype, Some(LowLevelType::Void));
    }

    /// Build a single-FunctionDesc callfamily whose simple-call
    /// calltable contains one row pointing at a pre-cached PyGraph.
    /// Mirrors the fixture at
    /// `somepbc_rtyper_makerepr_multi_function_descs_with_callfamily_uses_functionspbcrepr`
    /// but constrained to one desc so `convert_to_concrete_llfn`,
    /// `get_unique_llfn`, and `get_concrete_llfn` all have a
    /// well-formed `(shape, index, funcdesc_key)` triple to look up.
    fn single_funcdesc_with_callfamily(
        ann: &Rc<RPythonAnnotator>,
        name: &str,
    ) -> (Rc<StdRefCell<FunctionDesc>>, CallShape, Rc<PyGraph>) {
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};

        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            name,
            sig,
            None,
            None,
        )));
        let graph_name = format!("{name}_graph");
        let pygraph = super::tests::make_pygraph(&graph_name);
        fd.borrow()
            .cache
            .borrow_mut()
            .insert(GraphCacheKey::None, pygraph.clone());

        let args = ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            None,
            None,
        );
        let shape = args.rawshape();
        FunctionDesc::consider_call_site(
            std::slice::from_ref(&fd),
            &args,
            &SomeValue::Impossible,
            None,
        )
        .expect("consider_call_site should populate the calltable");
        (fd, shape, pygraph)
    }

    fn void_variable() -> crate::flowspace::model::Hlvalue {
        let v = crate::flowspace::model::Variable::new();
        v.set_concretetype(Some(LowLevelType::Void));
        crate::flowspace::model::Hlvalue::Variable(v)
    }

    fn signed_variable() -> crate::flowspace::model::Hlvalue {
        let v = crate::flowspace::model::Variable::new();
        v.set_concretetype(Some(LowLevelType::Signed));
        crate::flowspace::model::Hlvalue::Variable(v)
    }

    fn empty_lloplist(
        rtyper: &Rc<RPythonTyper>,
    ) -> Rc<RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>> {
        Rc::new(RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ))
    }

    /// Verifies that a `Hlvalue::Constant(LLPtr(Func(...)))` names the
    /// same `PyGraph` identity as `expected`. The `_func` object stores
    /// the graph as a `GraphKey::as_usize()` pointer hash, so identity
    /// comparison runs through [`GraphKey`].
    fn const_points_at_graph(
        value: &crate::flowspace::model::Hlvalue,
        expected: &Rc<PyGraph>,
    ) -> bool {
        use crate::flowspace::model::{ConstValue, GraphKey, Hlvalue};
        use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
        let Hlvalue::Constant(c) = value else {
            return false;
        };
        let ConstValue::LLPtr(llptr) = &c.value else {
            return false;
        };
        let obj = match llptr._obj() {
            Ok(obj) => obj,
            Err(_) => return false,
        };
        let _ptr_obj::Func(func_obj) = obj else {
            return false;
        };
        match func_obj.graph {
            Some(id) => id == GraphKey::of(&expected.graph).as_usize(),
            None => false,
        }
    }

    #[test]
    fn function_repr_convert_to_concrete_llfn_returns_constant_pointing_at_callable_graph() {
        let (ann, rtyper) = make_rtyper();
        let (fd, shape, pygraph) = single_funcdesc_with_callfamily(&ann, "f");

        let s_pbc = SomePBC::new(vec![DescEntry::function(fd)], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let llop = empty_lloplist(&rtyper);

        let result = r
            .convert_to_concrete_llfn(&void_variable(), &shape, 0, &llop)
            .unwrap();
        assert!(
            const_points_at_graph(&result, &pygraph),
            "convert_to_concrete_llfn should wrap the row-of-one-graph entry as an LLPtr Constant"
        );
    }

    #[test]
    fn function_repr_convert_to_concrete_llfn_rejects_non_void_input() {
        let (ann, rtyper) = make_rtyper();
        let (fd, shape, _pygraph) = single_funcdesc_with_callfamily(&ann, "f");

        let s_pbc = SomePBC::new(vec![DescEntry::function(fd)], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let llop = empty_lloplist(&rtyper);

        let err = r
            .convert_to_concrete_llfn(&signed_variable(), &shape, 0, &llop)
            .unwrap_err();
        assert!(
            err.to_string().contains("concretetype != Void"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn function_repr_get_unique_llfn_returns_constant_for_single_simple_shape() {
        let (ann, rtyper) = make_rtyper();
        let (fd, _shape, pygraph) = single_funcdesc_with_callfamily(&ann, "f");

        let s_pbc = SomePBC::new(vec![DescEntry::function(fd)], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();

        let result = r.get_unique_llfn().unwrap();
        assert!(
            const_points_at_graph(&result, &pygraph),
            "get_unique_llfn should produce an LLPtr constant aimed at the sole simple-shape graph"
        );
    }

    #[test]
    fn function_repr_get_unique_llfn_errors_when_no_simple_shape_tables_exist() {
        // A FunctionDesc whose callfamily has no calltable entries at
        // all leaves `get_unique_llfn`'s simple-shape filter with an
        // empty `tables` list — `len(tables) != 1` trips upstream's
        // `"cannot pass a function with various call shapes"` branch.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        // Touch `getcallfamily()` so FunctionReprBase picks up an empty
        // family rather than `None` — otherwise the test would surface
        // the "callfamily not available" guard instead.
        if let Some(rc) = f.as_function() {
            rc.borrow().base.getcallfamily().unwrap();
        } else {
            unreachable!();
        }
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let err = r.get_unique_llfn().unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot pass a function with various call shapes"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn function_repr_get_concrete_llfn_returns_constant_pointing_at_specialized_graph() {
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig,
            None,
            None,
        )));
        // `get_concrete_llfn` calls `funcdesc.get_graph` which routes
        // through `default_specialize` → `maybe_star_args` → cachedgraph
        // with `GraphCacheKey::None` when there are no star args — the
        // pre-populated cache entry is what lets the test avoid a
        // missing-pyobj crash.
        let pygraph = super::tests::make_pygraph("f_graph");
        fd.borrow()
            .cache
            .borrow_mut()
            .insert(GraphCacheKey::None, pygraph.clone());

        let s_pbc = SomePBC::new(vec![DescEntry::function(fd)], false);
        let r = FunctionRepr::new(&rtyper, s_pbc.clone()).unwrap();

        let result = r
            .get_concrete_llfn(
                &s_pbc,
                vec![SomeValue::Integer(SomeInteger::default())],
                None,
            )
            .unwrap();
        assert!(
            const_points_at_graph(&result, &pygraph),
            "get_concrete_llfn should wrap funcdesc.get_graph's result as an LLPtr Constant"
        );
    }

    #[test]
    fn function_repr_simple_call_emits_direct_call_op() {
        // rpbc.py — `FunctionReprBase.call` (Rust binding on
        // `FunctionRepr`). The Void-typed `convert_to_concrete_llfn`
        // returns a Constant funcptr, so the dispatch must take the
        // `direct_call` branch and short-circuit at the
        // `r_result is impossible_repr` check (Void return graph).
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{
            ConstValue, Constant as FlowConstant, Hlvalue, SpaceOperation, Variable,
        };
        use crate::translator::rtyper::rmodel::impossible_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let (fd, _shape, _pygraph) = single_funcdesc_with_callfamily(&ann, "f");
        let s_pbc = SomePBC::new(vec![DescEntry::function(fd)], false);
        let r: std::sync::Arc<FunctionRepr> = FunctionRepr::new(&rtyper, s_pbc.clone()).unwrap();
        let r_dyn: std::sync::Arc<dyn Repr> = r.clone();

        // SpaceOperation: result_var = simple_call(receiver, c_int)
        let receiver = Variable::new();
        receiver.set_concretetype(Some(LowLevelType::Void));
        let receiver_h = Hlvalue::Variable(receiver);

        let int_const = FlowConstant::with_concretetype(ConstValue::Int(7), LowLevelType::Signed);
        let int_const_h = Hlvalue::Constant(int_const);

        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Void));
        let result_h = Hlvalue::Variable(result_var);

        let spaceop = SpaceOperation::new(
            "simple_call",
            vec![receiver_h.clone(), int_const_h.clone()],
            result_h,
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);

        // Seed args_v / args_s / args_r / r_result manually; the test
        // PyGraph carries no annotation propagation so `hop.setup()`
        // would not produce useful Reprs from the spaceop.
        *hop.args_v.borrow_mut() = vec![receiver_h, int_const_h];
        *hop.args_s.borrow_mut() = vec![
            SomeValue::PBC(s_pbc),
            SomeValue::Integer(SomeInteger::default()),
        ];
        *hop.args_r.borrow_mut() = vec![Some(r_dyn.clone()), None];
        // The test PyGraph's returnvar has annotation=Impossible, so
        // upstream rtyper would assign `impossible_repr` to r_result.
        *hop.r_result.borrow_mut() = Some(impossible_repr() as std::sync::Arc<dyn Repr>);

        // upstream rpbc.py — `r_result is impossible_repr`
        // short-circuits to None.
        let result = r.rtype_simple_call(&hop).unwrap();
        assert!(
            result.is_none(),
            "Void return + impossible_repr r_result must short-circuit to None"
        );

        // upstream rpbc.py:213 — direct_call is emitted because
        // `convert_to_concrete_llfn` returns a Constant for Void-typed
        // FunctionRepr.
        let llops = hop.llops.borrow();
        let opnames: Vec<&str> = llops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert!(
            opnames.contains(&"direct_call"),
            "expected a direct_call op in llops, got: {opnames:?}"
        );
        assert!(
            !opnames.contains(&"indirect_call"),
            "FunctionRepr (Void-typed funcptr) must not emit indirect_call, got: {opnames:?}"
        );
    }

    #[test]
    fn functions_pbc_repr_simple_call_emits_indirect_call_op() {
        // rpbc.py (`FunctionReprBase.call`) +
        // rpbc.py (`FunctionsPBCRepr.convert_to_concrete_llfn`).
        // The single-row `convert_to_concrete_llfn` returns `v`
        // unchanged — when `v` is a `Variable` (runtime funcptr),
        // the dispatch must take the `indirect_call` branch and
        // append the `c_graphs` placeholder Constant.
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{
            ConstValue, Constant as FlowConstant, Hlvalue, SpaceOperation, Variable,
        };
        use crate::translator::rtyper::rmodel::impossible_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let (fd, _shape, _pygraph) = single_funcdesc_with_callfamily(&ann, "f");
        let s_pbc = SomePBC::new(vec![DescEntry::function(fd)], false);
        let r: std::sync::Arc<FunctionsPBCRepr> =
            FunctionsPBCRepr::new(&rtyper, s_pbc.clone()).unwrap();
        let r_dyn: std::sync::Arc<dyn Repr> = r.clone();
        let funcptr_lltype = r.lowleveltype().clone();

        // SpaceOperation: result_var = simple_call(receiver_funcptr, c_int)
        // The receiver carries the PBCRepr's lowleveltype = Ptr(FuncType(...)),
        // mirroring upstream's `vfn = hop.inputarg(self, arg=0)` path
        // where `self` is the FunctionsPBCRepr.
        let receiver = Variable::new();
        receiver.set_concretetype(Some(funcptr_lltype.clone()));
        let receiver_h = Hlvalue::Variable(receiver);

        let int_const = FlowConstant::with_concretetype(ConstValue::Int(7), LowLevelType::Signed);
        let int_const_h = Hlvalue::Constant(int_const);

        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Void));
        let result_h = Hlvalue::Variable(result_var);

        let spaceop = SpaceOperation::new(
            "simple_call",
            vec![receiver_h.clone(), int_const_h.clone()],
            result_h,
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);

        // Seed args_v / args_s / args_r / r_result manually; the test
        // PyGraph carries no annotation propagation so `hop.setup()`
        // would not produce useful Reprs from the spaceop.
        *hop.args_v.borrow_mut() = vec![receiver_h, int_const_h];
        *hop.args_s.borrow_mut() = vec![
            SomeValue::PBC(s_pbc),
            SomeValue::Integer(SomeInteger::default()),
        ];
        *hop.args_r.borrow_mut() = vec![Some(r_dyn.clone()), None];
        // The test PyGraph's returnvar has annotation=Impossible, so
        // upstream rtyper would assign `impossible_repr` to r_result.
        *hop.r_result.borrow_mut() = Some(impossible_repr() as std::sync::Arc<dyn Repr>);

        // upstream rpbc.py — `r_result is impossible_repr`
        // short-circuits to None.
        let result = r.rtype_simple_call(&hop).unwrap();
        assert!(
            result.is_none(),
            "Void return + impossible_repr r_result must short-circuit to None"
        );

        // upstream rpbc.py:215-217 — indirect_call is emitted because
        // `convert_to_concrete_llfn` returned `v` (a Variable funcptr,
        // not a Constant) for the single-row FunctionsPBCRepr case.
        let llops = hop.llops.borrow();
        let opnames: Vec<&str> = llops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert!(
            opnames.contains(&"indirect_call"),
            "expected an indirect_call op in llops, got: {opnames:?}"
        );
        assert!(
            !opnames.contains(&"direct_call"),
            "FunctionsPBCRepr (Variable funcptr) must not emit direct_call, got: {opnames:?}"
        );

        // upstream rpbc.py:216 — the `c_graphs` placeholder Constant
        // is appended as the last arg of the indirect_call op.
        let indirect = llops
            .ops
            .iter()
            .find(|op| op.opname == "indirect_call")
            .expect("indirect_call op must exist");
        assert!(
            matches!(indirect.args.last(), Some(Hlvalue::Constant(_))),
            "indirect_call's last arg must be a Constant (c_graphs placeholder), \
             got: {:?}",
            indirect.args.last()
        );
    }

    #[test]
    fn single_frozen_pbc_repr_rtype_getattr_requires_constant_s_result() {
        // rpbc.py rtype_getattr — mirror of the MU test; getattr on a Void
        // PBC must surface a constant, otherwise raise.
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let r = SingleFrozenPBCRepr::new(f);
        let spaceop = SpaceOperation::new(
            OpKind::GetAttr.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.s_result
            .replace(Some(crate::annotator::model::SomeValue::Impossible));
        hop.r_result.replace(Some(
            std::sync::Arc::new(SingleFrozenPBCRepr::new(frozen_entry(
                &ann.bookkeeper,
                "frozen_r",
            ))) as std::sync::Arc<dyn Repr>,
        ));
        let err = r.rtype_getattr(&hop).unwrap_err();
        assert!(
            err.to_string()
                .contains("getattr on a constant PBC returns a non-constant"),
            "unexpected: {err}",
        );
    }

    #[test]
    fn functions_pbc_repr_convert_desc_returns_llptr_constant_and_caches_it() {
        // rpbc.py — single-row convert_desc must produce a
        // Constant wrapping the row's _ptr, and subsequent calls must
        // hit the funccache.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::ConstValue;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        let args = ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let f_entry = DescEntry::function(fd_f.clone());
        let s_pbc = SomePBC::new(vec![f_entry.clone(), DescEntry::function(fd_g)], false);
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();

        let c1 = r.convert_desc(&f_entry).unwrap();
        assert!(matches!(c1.value, ConstValue::LLPtr(_)));
        // Re-invoking should hit the funccache.
        let before = r.funccache.borrow().len();
        let c2 = r.convert_desc(&f_entry).unwrap();
        assert_eq!(c1, c2);
        assert_eq!(r.funccache.borrow().len(), before);
    }

    #[test]
    fn functions_pbc_repr_convert_const_none_returns_null_pointer_constant() {
        // rpbc.py:294-296 — `value is None` arm emits
        // `nullptr(self.lowleveltype.TO)`.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::ConstValue;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        let args = ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let s_pbc = SomePBC::new(
            vec![DescEntry::function(fd_f), DescEntry::function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();
        let c = r.convert_const(&ConstValue::None).unwrap();
        let ConstValue::LLPtr(llptr) = &c.value else {
            panic!("expected LLPtr");
        };
        assert!(!llptr.nonzero(), "convert_const(None) must be a null ptr");
        assert_eq!(c.concretetype.as_ref(), Some(r.lowleveltype()));
    }

    #[test]
    fn functions_pbc_repr_convert_const_non_host_object_surfaces_message_error() {
        // rpbc.py:296-297 funnels every non-None value through
        // `bookkeeper.getdesc`, which only accepts callable host
        // objects. Non-`HostObject` ConstValue payloads (e.g.
        // `ConstValue::Int`) cannot reach getdesc — pyre surfaces
        // a typed message error instead of attempting the lookup.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::ConstValue;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &ArgumentsForTranslation::new(
                vec![Some(SomeValue::Integer(SomeInteger::default()))],
                None,
                None,
            ),
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        let s_pbc = SomePBC::new(
            vec![DescEntry::function(fd_f), DescEntry::function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();
        let err = r.convert_const(&ConstValue::Int(7)).unwrap_err();
        assert!(
            err.to_string().contains("not a HostObject"),
            "expected non-HostObject diagnostic, got {err}",
        );
    }

    #[test]
    fn functions_pbc_repr_convert_const_host_object_dispatches_through_getdesc_to_convert_desc() {
        // rpbc.py:296-297 — the standard non-None path:
        //     funcdesc = self.rtyper.annotator.bookkeeper.getdesc(value)
        //     return self.convert_desc(funcdesc)
        //
        // Routes a `ConstValue::HostObject(UserFunction)` through
        // `bookkeeper.getdesc` (which materialises a `FunctionDesc`)
        // and then through `convert_desc`. Result: a Constant of the
        // repr's lowleveltype carrying the cached `_ptr`.
        //
        // NOTE: this test cannot reuse `build_multi_row_functions_pbc_repr`
        // because the host function and the FunctionDescs that drive
        // `consider_call_site` must share identity. Instead we
        // `bookkeeper.newfuncdesc(host)` first, then attach the
        // resulting FunctionDesc to consider_call_site.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{
            ConstValue, Constant as FlowConstant, GraphFunc, HostObject,
        };

        let (ann, rtyper) = make_rtyper();
        let host_f = HostObject::new_user_function(GraphFunc::new(
            "test::f",
            FlowConstant::new(ConstValue::Dict(Default::default())),
        ));
        let fd_rc = ann
            .bookkeeper
            .newfuncdesc(&host_f)
            .unwrap()
            .as_function()
            .unwrap();

        // Pre-populate the graph cache so consider_call_site can build
        // a row without hitting the missing pyobj path.
        fd_rc
            .borrow()
            .cache
            .borrow_mut()
            .insert(GraphCacheKey::None, super::tests::make_pygraph("f_graph"));
        FunctionDesc::consider_call_site(
            std::slice::from_ref(&fd_rc),
            &ArgumentsForTranslation::new(
                vec![Some(SomeValue::Integer(SomeInteger::default()))],
                None,
                None,
            ),
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        let s_pbc = SomePBC::new(vec![DescEntry::function(fd_rc)], false);
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();

        let host_value = ConstValue::HostObject(host_f);
        let c = r.convert_const(&host_value).unwrap();
        assert_eq!(
            c.concretetype.as_ref(),
            Some(r.lowleveltype()),
            "convert_const must hand back a Constant typed as the \
             repr's lowleveltype",
        );
        assert!(
            matches!(c.value, ConstValue::LLPtr(_)),
            "convert_const must wrap the convert_desc-produced _ptr; \
             got {:?}",
            c.value,
        );
    }

    #[test]
    fn functions_pbc_repr_rtype_bool_constant_fast_path_returns_bool_constant() {
        // rmodel.py — CanBeNull.rtype_bool fast-path: when the
        // annotator has proved the result constant, emit inputconst(Bool, ...).
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeBool, SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &ArgumentsForTranslation::new(
                vec![Some(SomeValue::Integer(SomeInteger::default()))],
                None,
                None,
            ),
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        let s_pbc = SomePBC::new(
            vec![DescEntry::function(fd_f), DescEntry::function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();

        let spaceop = SpaceOperation::new(
            OpKind::Bool.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops.clone());
        // Constant annotation → fast-path: emits a Bool constant.
        let mut sb = SomeBool::new();
        sb.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
        hop.s_result.replace(Some(SomeValue::Bool(sb)));

        let result = r
            .rtype_bool(&hop)
            .unwrap()
            .expect("fast path returns value");
        let Hlvalue::Constant(c) = result else {
            panic!("expected Bool Constant, got {result:?}");
        };
        assert_eq!(c.concretetype, Some(LowLevelType::Bool));
        assert!(matches!(c.value, ConstValue::Bool(true)));
        assert!(
            llops.borrow().ops.is_empty(),
            "fast-path should emit no ops"
        );
    }

    #[test]
    fn functions_pbc_repr_rtype_bool_non_constant_emits_ptr_nonzero() {
        // rmodel.py:259-260 — non-constant s_result path: emit a
        // ptr_nonzero op against a Bool resulttype.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &ArgumentsForTranslation::new(
                vec![Some(SomeValue::Integer(SomeInteger::default()))],
                None,
                None,
            ),
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        let s_pbc = SomePBC::new(
            vec![DescEntry::function(fd_f), DescEntry::function(fd_g)],
            false,
        );
        let r: std::sync::Arc<dyn Repr> = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();

        // Input arg: a Ptr-typed Variable (matches FunctionsPBCRepr lowleveltype).
        let arg_var = Variable::new();
        arg_var.set_concretetype(Some(r.lowleveltype().clone()));
        let arg = Hlvalue::Variable(arg_var);
        let spaceop = SpaceOperation::new(
            OpKind::Bool.opname(),
            vec![arg.clone()],
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops.clone());
        hop.args_v.borrow_mut().push(arg);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r.clone()));
        // Non-constant s_result → ptr_nonzero path.
        hop.s_result.replace(Some(SomeValue::Impossible));

        let result = r
            .rtype_bool(&hop)
            .unwrap()
            .expect("ptr_nonzero returns value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(
            ops.ops.len(),
            1,
            "ptr_nonzero should be the only emitted op"
        );
        assert_eq!(ops.ops[0].opname, "ptr_nonzero");
    }

    #[test]
    fn functions_pbc_repr_single_uniquerow_populates_concretetable_and_funccache() {
        // rpbc.py __init__ — ensure the new single-row ctor path wires
        // `concretetable`, `uniquerows`, and the funccache correctly.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        let args = ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let s_pbc = SomePBC::new(
            vec![DescEntry::function(fd_f), DescEntry::function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(
            r.uniquerows.len(),
            1,
            "two FunctionDescs sharing a shape merge into one uniquerow"
        );
        assert_eq!(r.concretetable.len(), 1);
        assert!(r.funccache.borrow().is_empty());
    }

    #[test]
    fn function_repr_get_r_implfunc_returns_self_and_zero() {
        // rpbc.py — upstream `return self, 0`. Port exposes
        // `(self_as_dyn_repr, 0)`; identity is checked by comparing
        // pointer against the concrete FunctionRepr receiver.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let (r_impl, nimplicit) = r.get_r_implfunc().unwrap();
        assert_eq!(nimplicit, 0);
        assert!(Arc::ptr_eq(&r_impl, &(Arc::clone(&r) as Arc<dyn Repr>)));
    }

    #[test]
    fn function_repr_base_get_s_signatures_delegates_to_funcdesc() {
        // rpbc.py — upstream `return funcdesc.get_s_signatures(shape)`.
        // Without a populated calltable, `FunctionDesc.get_s_signatures`
        // returns an empty Vec (its `calltables.get(shape)` miss path).
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        if let Some(rc) = f.as_function() {
            rc.borrow().base.getcallfamily().unwrap();
        }
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let sigs = r
            .base()
            .get_s_signatures(&crate::flowspace::argument::CallShape {
                shape_cnt: 0,
                shape_keys: Vec::new(),
                shape_star: false,
            })
            .unwrap();
        assert!(sigs.is_empty());
    }
}
