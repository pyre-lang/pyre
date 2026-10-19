//! RPython `rpython/rtyper/rptr.py` parity module.
//!
//! The pointer repr implementations landed in [`super::rmodel`] with
//! upstream `rptr.py` line references.  This module preserves the PyPy
//! import/file path so ports can refer to `rtyper::rptr::PtrRepr` and
//! related symbols by their original RPython home.

use crate::flowspace::model::{ConstValue, Hlvalue};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{RTypeResult, ptr_target_func};
use crate::translator::rtyper::rtyper::{GenopResult, HighLevelOp};

pub use crate::annotator::model::SomePtr;
pub use crate::flowspace::model as flowmodel;
pub use crate::translator::rtyper::llannotation::{
    SomeInteriorPtr, SomeLLADTMeth, lltype_to_annotation,
};
pub use crate::translator::rtyper::lltypesystem::lltype;
pub use crate::translator::rtyper::rint::IntegerRepr;
pub use crate::translator::rtyper::rmodel::{InteriorPtrRepr, LLADTMethRepr, PtrRepr, Repr};

/// RPython `PtrRepr.rtype_simple_call` (`rptr.py`).
///
/// A Constant callee becomes `direct_call` (and `record_extra_call` when
/// the pointer names a graph). Otherwise the hop is `indirect_call` with
/// a trailing `inputconst(Void, None)` — no known graphs — after
/// `hop.exception_is_here()`. The result type is `FUNC.RESULT`.
pub(crate) fn rtype_simple_call(r: &PtrRepr, hop: &HighLevelOp) -> RTypeResult {
    let func_type = ptr_target_func(&r.ptrtype().TO).ok_or_else(|| {
        TyperError::message(format!("calling a non-function {:?}", r.ptrtype().TO))
    })?;
    let args_r = hop.args_r.borrow().clone();
    let mut vlist = Vec::with_capacity(args_r.len() + 1);
    for (i, r_arg) in args_r.iter().enumerate() {
        let r_arg = r_arg
            .as_ref()
            .ok_or_else(|| TyperError::message("PtrRepr.rtype_simple_call missing arg repr"))?;
        vlist.push(hop.inputarg(r_arg, i)?);
    }
    let nexpected = func_type.args.len();
    let nactual = vlist.len().saturating_sub(1);
    if nactual != nexpected {
        return Err(TyperError::message(format!(
            "argcount mismatch:  expected {nexpected} got {nactual}"
        )));
    }
    let opname = if let Some(Hlvalue::Constant(c_func)) = vlist.first() {
        if let ConstValue::LLPtr(ptr) = &c_func.value
            && let Ok(lltype::_ptr_obj::Func(func)) = ptr._obj()
            && let Some(graph_id) = func.graph
        {
            hop.llops.borrow().record_extra_call_by_graph_id(graph_id)?;
        }
        "direct_call"
    } else {
        vlist.push(Hlvalue::Constant(HighLevelOp::inputconst(
            &LowLevelType::Void,
            &ConstValue::None,
        )?));
        "indirect_call"
    };
    hop.exception_is_here()?;
    Ok(hop.genop(opname, vlist, GenopResult::LLType(func_type.result.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::model::{SomeInteger, SomePtr, SomeValue};
    use crate::flowspace::model::{
        Block, BlockKey, ConstValue, Constant, FunctionGraph, Hlvalue, SpaceOperation, Variable,
    };
    use crate::model::{FunctionGraph as LegacyGraph, OpKind, ValueType};
    use crate::translator::rtyper::call_registry::CallRegistry;
    use crate::translator::rtyper::cutover::specialize_legacy_graph_with_registry_returning_value_to_var;
    use crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace;
    use crate::translator::rtyper::lltypesystem::lltype::{FuncType, LowLevelType, Ptr, PtrTarget};
    use crate::translator::rtyper::rmodel::{PtrRepr, Repr, ReprState, inputconst_from_lltype};
    use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
    use crate::translator::translator::CallGraphKey;

    #[derive(Debug)]
    struct TestRepr {
        state: ReprState,
        lltype: LowLevelType,
        class_name: &'static str,
    }

    impl TestRepr {
        fn new(lltype: LowLevelType, class_name: &'static str) -> Self {
            TestRepr {
                state: ReprState::new(),
                lltype,
                class_name,
            }
        }
    }

    impl Repr for TestRepr {
        fn lowleveltype(&self) -> &LowLevelType {
            &self.lltype
        }

        fn state(&self) -> &ReprState {
            &self.state
        }

        fn class_name(&self) -> &'static str {
            self.class_name
        }
    }

    fn live_rtyper_for_hop() -> (Rc<RPythonAnnotator>, Rc<RPythonTyper>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        (ann, rtyper)
    }

    fn empty_hop(rtyper: &Rc<RPythonTyper>, opname: &str) -> HighLevelOp {
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                opname.to_string(),
                Vec::new(),
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops,
        )
    }

    fn anchor_lock() -> parking_lot::MutexGuard<'static, ()> {
        static LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
        LOCK.lock()
    }

    #[test]
    fn rptr_module_exposes_upstream_pointer_repr_symbols() {
        fn assert_repr<T: Repr>() {}

        assert_repr::<PtrRepr>();
        assert_repr::<InteriorPtrRepr>();
        assert_repr::<LLADTMethRepr>();

        let _ptr_new: fn(lltype::Ptr) -> PtrRepr = PtrRepr::new;
        let _interior_new: fn(lltype::InteriorPtr) -> InteriorPtrRepr = InteriorPtrRepr::new;
        let _lltype_to_annotation: fn(lltype::LowLevelType) -> crate::annotator::model::SomeValue =
            lltype_to_annotation::<lltype::LowLevelType>;

        assert_eq!(
            std::any::type_name::<SomePtr>().rsplit("::").next(),
            Some("SomePtr")
        );
        assert_eq!(
            std::any::type_name::<SomeInteriorPtr>().rsplit("::").next(),
            Some("SomeInteriorPtr")
        );
        assert_eq!(
            std::any::type_name::<SomeLLADTMeth>().rsplit("::").next(),
            Some("SomeLLADTMeth")
        );
        assert_eq!(
            std::any::type_name::<IntegerRepr>().rsplit("::").next(),
            Some("IntegerRepr")
        );
    }

    #[test]
    fn ptrrepr_rtype_simple_call_emits_indirect_call_with_void_none_graphs() {
        let ptr = Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Bool,
            }),
        };
        let r_ptr = Arc::new(PtrRepr::new(ptr.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let r_signed: Arc<dyn Repr> = Arc::new(TestRepr::new(LowLevelType::Signed, "SignedRepr"));
        let r_bool: Arc<dyn Repr> = Arc::new(TestRepr::new(LowLevelType::Bool, "BoolRepr"));
        let (_ann, rtyper) = live_rtyper_for_hop();
        let hop = empty_hop(&rtyper, "simple_call");
        let v_func = Variable::new();
        v_func.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        hop.args_v.borrow_mut().extend([
            Hlvalue::Variable(v_func),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(5),
                LowLevelType::Signed,
            )),
        ]);
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr_dyn), Some(r_signed)]);
        *hop.r_result.borrow_mut() = Some(r_bool);

        let result = rtype_simple_call(&r_ptr, &hop).unwrap().unwrap();
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "indirect_call");
        assert_eq!(ops.ops[0].args.len(), 3);
        let Hlvalue::Constant(c_graphs) = &ops.ops[0].args[2] else {
            panic!("last arg must be inputconst(Void, None)");
        };
        assert_eq!(c_graphs.concretetype, Some(LowLevelType::Void));
        assert_eq!(c_graphs.value, ConstValue::None);
    }

    #[test]
    fn ptrrepr_rtype_simple_call_constant_funcptr_emits_direct_call() {
        let (ann, rtyper) = live_rtyper_for_hop();
        let helper = rtyper
            .lowlevel_helper_function("callee", vec![LowLevelType::Signed], LowLevelType::Signed)
            .expect("helper graph");
        let helper_graph = helper.graph.as_ref().expect("helper graph must exist");
        let func_ptr = rtyper.getcallable(helper_graph).expect("getcallable");
        let PtrTarget::Func(func_type) = func_ptr._TYPE.TO.clone() else {
            panic!("helper callable must be a function pointer");
        };
        let r_ptr = Arc::new(PtrRepr::new(func_ptr._TYPE.clone()));
        let r_ptr_dyn: Arc<dyn Repr> = r_ptr.clone();
        let r_signed: Arc<dyn Repr> = Arc::new(TestRepr::new(LowLevelType::Signed, "SignedRepr"));
        let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
        let c_func = inputconst_from_lltype(&func_ptr_type, &ConstValue::LLPtr(Box::new(func_ptr)))
            .expect("function ptr constant");

        let caller_start = Block::shared(Vec::new());
        let caller = Rc::new(RefCell::new(FunctionGraph::new(
            "caller",
            caller_start.clone(),
        )));
        ann.translator.graphs.borrow_mut().push(caller.clone());
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&caller_start), Some(caller.clone()));
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            Some(caller_start.clone()),
        )));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(func_type.result.clone()));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call",
                vec![
                    Hlvalue::Constant(c_func),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(5),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(r_ptr.ptrtype().clone())),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr_dyn), Some(r_signed.clone())]);
        *hop.r_result.borrow_mut() = Some(r_signed);

        let result = rtype_simple_call(&r_ptr, &hop).unwrap().unwrap();
        assert!(matches!(result, Hlvalue::Variable(_)));
        assert_eq!(llops.borrow().ops[0].opname, "direct_call");

        let callgraph = &ann.translator;
        let callgraph = callgraph.callgraph.borrow();
        let expected_key = CallGraphKey {
            caller: crate::flowspace::model::GraphKey::of(&caller),
            callee: crate::flowspace::model::GraphKey::of(&helper_graph.graph),
            tag_block: BlockKey::of(&caller_start),
            tag_index: 0,
        };
        let edge = callgraph
            .get(&expected_key)
            .expect("direct call should record callgraph edge");
        assert_eq!(
            crate::flowspace::model::GraphKey::of(&edge.caller),
            crate::flowspace::model::GraphKey::of(&caller)
        );
        assert_eq!(
            crate::flowspace::model::GraphKey::of(&edge.callee),
            crate::flowspace::model::GraphKey::of(&helper_graph.graph)
        );
    }

    #[test]
    fn stored_fn_ptr_graph_annotates_functype_result_and_rtypes_indirect_call() {
        let _lock = anchor_lock();
        let mut graph = LegacyGraph::new("call_stored_fn");
        let entry = graph.startblock;
        let f_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "f".to_string(),
                    ty: ValueType::Int,
                    class_root: Some("fn(i64) -> bool".to_string()),
                },
                true,
            )
            .unwrap();
        let x_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "x".to_string(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, f_var.clone());
        graph.push_inputarg_var(entry, x_var.clone());
        let r_var = graph
            .push_op_var(
                entry,
                OpKind::IndirectCall {
                    funcptr: f_var,
                    args: vec![x_var],
                    graphs: None,
                    family_key: None,
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(r_var.clone()));

        let registry = CallRegistry::new(Rc::new(Bookkeeper::new()));
        let lifted = function_graph_to_flowspace(&graph, &registry)
            .expect("stored fn-ptr IndirectCall must lift to simple_call");
        let start = lifted.graph.borrow().startblock.clone();
        let lifted_ops = start.borrow().operations.clone();
        let simple = lifted_ops
            .iter()
            .find(|op| op.opname == "simple_call")
            .expect("adapter lowers stored fn-ptr IndirectCall to simple_call");
        assert_eq!(simple.args.len(), 2);

        let (value_to_var, _) =
            specialize_legacy_graph_with_registry_returning_value_to_var(&graph, &registry)
                .expect("stored fn-ptr graph must annotate and rtype");
        let result_var = value_to_var.get(&r_var).expect("call result must project");
        {
            let cell = result_var.annotation.borrow();
            match cell.as_ref().map(|rc| rc.as_ref()) {
                Some(SomeValue::Bool(_)) => {}
                other => panic!(
                    "SomePtr.call returns ll_to_annotation of FuncType.RESULT, got {other:?}"
                ),
            }
        }
        assert_eq!(result_var.concretetype(), Some(LowLevelType::Bool));

        let (ann, _) = registry.ensure_session().expect("session after specialize");
        let graph_ref = ann
            .annotated
            .borrow()
            .values()
            .find_map(|g| g.clone())
            .expect("specialized subject graph");
        let start = graph_ref.borrow().startblock.clone();
        let rtyped = start.borrow().operations.clone();
        let indirect = rtyped
            .iter()
            .find(|op| op.opname == "indirect_call")
            .expect("PtrRepr.rtype_simple_call emits indirect_call");
        let last = indirect.args.last().expect("trailing graphs arg");
        let Hlvalue::Constant(c_graphs) = last else {
            panic!("last arg must be inputconst(Void, None)");
        };
        assert_eq!(c_graphs.concretetype, Some(LowLevelType::Void));
        assert_eq!(c_graphs.value, ConstValue::None);
    }
}
