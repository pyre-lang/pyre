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

    fn object_class_registry() -> CallRegistry {
        let bk = Rc::new(Bookkeeper::new());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "PyObject".to_string(),
            vec![("ob_type".to_string(), "PyObjectRef".to_string())],
        );
        bk.set_struct_fields(Rc::new(fields));
        CallRegistry::new(bk)
    }

    fn kind_of(
        value_to_var: &crate::translator::rtyper::flowspace_adapter::LegacyToTyped,
        var: &crate::flowspace::model::Variable,
    ) -> crate::codewriter::type_state::ConcreteType {
        let typed = value_to_var
            .get(var)
            .unwrap_or_else(|| panic!("value missing from value_to_var"));
        let ll = typed
            .concretetype()
            .unwrap_or_else(|| panic!("Variable.concretetype not populated"));
        crate::translator::rtyper::cutover::lowleveltype_to_concrete(&ll)
            .unwrap_or_else(|_| panic!("lltype does not project to ConcreteType"))
    }

    /// `SomePtr.call` of a `Ptr(FuncType)` whose FUNC arg is
    /// `rclass.OBJECTPTR`: a GC instance argument is taken as that
    /// instance pointer (`annlowlevel.cast_instance_to_base_ptr`
    /// adaptation in `SomePtr.call`) and the result is
    /// `ll_to_annotation(FUNC.RESULT)`. Dual-gate Match; hop is
    /// `indirect_call`; arg and result kinds are `'r'`.
    #[test]
    fn fnptr_objectptr_instance_arg_result_dual_gate_match() {
        use crate::codewriter::type_state::ConcreteType;
        use crate::translator::rtyper::cutover::{DualGateOutcome, dual_gate_check_with_registry};
        use crate::translator::rtyper::rclass::OBJECTPTR;

        let _lock = anchor_lock();
        let mut graph = LegacyGraph::new("call_objectptr_fn");
        let entry = graph.startblock;
        let f_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "f".to_string(),
                    ty: ValueType::Int,
                    class_root: Some("fn(*mut PyObject) -> *mut PyObject".to_string()),
                },
                true,
            )
            .unwrap();
        let ft = FuncType {
            args: vec![OBJECTPTR.clone()],
            result: OBJECTPTR.clone(),
        };
        f_var.set_concretetype(Some(LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Func(ft),
        }))));
        let obj_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "obj".to_string(),
                    ty: ValueType::Ref(Some("PyObject".to_string())),
                    class_root: Some("PyObject".to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, f_var.clone());
        graph.push_inputarg_var(entry, obj_var.clone());
        let r_var = graph
            .push_op_var(
                entry,
                OpKind::IndirectCall {
                    funcptr: f_var,
                    args: vec![obj_var.clone()],
                    graphs: None,
                    family_key: None,
                    result_ty: ValueType::Ref(Some("PyObject".to_string())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(r_var.clone()));

        let registry = Rc::new(object_class_registry());
        let graphs = crate::codewriter::call::GraphStore::default();
        let outcome = dual_gate_check_with_registry(&graph, &registry, &graphs).expect("dual-gate");
        let DualGateOutcome::Match {
            real_value_to_var, ..
        } = outcome
        else {
            panic!("OBJECTPTR instance-arg fn-ptr call must Match, got {outcome:?}");
        };
        assert_eq!(kind_of(&real_value_to_var, &obj_var), ConcreteType::GcRef);
        assert_eq!(kind_of(&real_value_to_var, &r_var), ConcreteType::GcRef);
        match real_value_to_var
            .get(&r_var)
            .expect("result in value_to_var")
            .annotation
            .borrow()
            .as_ref()
            .map(|rc| rc.as_ref())
        {
            Some(SomeValue::Instance(_)) => {}
            other => panic!("FUNC.RESULT OBJECTPTR is the instance annotation, got {other:?}"),
        }

        let (ann, _) = registry.ensure_session().expect("session after dual-gate");
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
        assert!(
            indirect.args.len() >= 3,
            "indirect_call is funcptr, OBJECTPTR arg, Void graphs"
        );
        let last = indirect.args.last().expect("trailing graphs arg");
        let Hlvalue::Constant(c_graphs) = last else {
            panic!("last arg must be inputconst(Void, None)");
        };
        assert_eq!(c_graphs.concretetype, Some(LowLevelType::Void));
        assert_eq!(c_graphs.value, ConstValue::None);
    }

    /// `SomePtr.call` of a `Ptr(FuncType)` whose FUNC result is
    /// `Signed`: dual-gate Match and result kind `'i'`.
    #[test]
    fn fnptr_signed_result_dual_gate_match() {
        use crate::codewriter::type_state::ConcreteType;
        use crate::translator::rtyper::cutover::{DualGateOutcome, dual_gate_check_with_registry};

        let _lock = anchor_lock();
        let mut graph = LegacyGraph::new("call_signed_fn");
        let entry = graph.startblock;
        let f_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "f".to_string(),
                    ty: ValueType::Int,
                    class_root: Some("fn(i64) -> i64".to_string()),
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
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(r_var.clone()));

        let registry = Rc::new(CallRegistry::new(Rc::new(Bookkeeper::new())));
        let graphs = crate::codewriter::call::GraphStore::default();
        let outcome = dual_gate_check_with_registry(&graph, &registry, &graphs).expect("dual-gate");
        let DualGateOutcome::Match {
            real_value_to_var, ..
        } = outcome
        else {
            panic!("Signed-result fn-ptr call must Match, got {outcome:?}");
        };
        assert_eq!(kind_of(&real_value_to_var, &r_var), ConcreteType::Signed);
        match real_value_to_var
            .get(&r_var)
            .expect("result in value_to_var")
            .annotation
            .borrow()
            .as_ref()
            .map(|rc| rc.as_ref())
        {
            Some(SomeValue::Integer(_)) => {}
            other => panic!("FUNC.RESULT Signed is SomeInteger, got {other:?}"),
        }

        let (ann, _) = registry.ensure_session().expect("session after dual-gate");
        let graph_ref = ann
            .annotated
            .borrow()
            .values()
            .find_map(|g| g.clone())
            .expect("specialized subject graph");
        let start = graph_ref.borrow().startblock.clone();
        let rtyped = start.borrow().operations.clone();
        assert!(
            rtyped.iter().any(|op| op.opname == "indirect_call"),
            "PtrRepr.rtype_simple_call emits indirect_call"
        );
    }

    #[test]
    fn array_read_expose_store_through_ptr_rtypes_to_interior_field_write_r() {
        use crate::annotator::model::SomePtr;
        use crate::codewriter::call::effect_graph_from_rtyped_helper;
        use crate::model::{ArrayGcKind, FieldDescriptor, ValueType};
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, GCREF, Ptr, PtrTarget, Struct,
        };

        let _lock = anchor_lock();
        let mut graph = LegacyGraph::new("leaf_store_through");
        let entry = graph.startblock;
        let entries = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "entries".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let index = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "i".to_string(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let value = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "v".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, entries.clone());
        graph.push_inputarg_var(entry, index.clone());
        graph.push_inputarg_var(entry, value.clone());
        let elem = graph
            .push_op_var(
                entry,
                OpKind::ArrayRead {
                    base: entries,
                    index,
                    item_ty: ValueType::Ref(None),
                    array_type_id: None,
                    nolength: false,
                    gckind: ArrayGcKind::Gc,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let exposed = graph
            .push_op_var(
                entry,
                OpKind::InteriorPtrExpose {
                    base: elem,
                    field: FieldDescriptor::new("value", Some("Entry".into())),
                    ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::StoreThroughPtr {
                ptr: exposed,
                value,
            },
            false,
        );
        graph.set_return(entry, None);

        let gcref_ptr = match GCREF.clone() {
            LowLevelType::Ptr(p) => *p,
            other => panic!("GCREF must be Ptr, got {other:?}"),
        };
        let entry_struct = Struct::new("Entry", vec![("value".into(), GCREF.clone())]);
        let array = Array::gc(LowLevelType::Struct(Box::new(entry_struct)));
        let s_entries = SomeValue::Ptr(SomePtr::new(Ptr {
            TO: PtrTarget::Array(array),
        }));
        let s_index = SomeValue::Integer(SomeInteger::new(false, false));
        let s_value = SomeValue::Ptr(SomePtr::new(gcref_ptr));

        let registry = CallRegistry::new(Rc::new(Bookkeeper::new()));
        let lifted = function_graph_to_flowspace(&graph, &registry)
            .expect("ArrayRead + InteriorPtrExpose + StoreThroughPtr must lift");
        let start_ops = lifted.graph.borrow().startblock.borrow().operations.clone();
        assert!(
            start_ops
                .iter()
                .any(|op| op.opname == "interior_ptr_expose"),
            "adapter must emit interior_ptr_expose; ops={start_ops:?}"
        );
        assert!(
            start_ops.iter().any(|op| op.opname == "store_through_ptr"),
            "adapter must emit store_through_ptr; ops={start_ops:?}"
        );

        let (ann, rtyper) = registry.ensure_session().expect("session");
        ann.build_graph_types(
            &lifted.graph,
            &[Some(s_entries), Some(s_index), Some(s_value)],
            true,
        )
        .expect("annotate ArrayRead + expose + store-through");
        rtyper
            .specialize_more_blocks()
            .expect("rtype leaf-TO store-through");

        let flow = lifted.graph.borrow();
        let rtyped_ops = flow.startblock.borrow().operations.clone();
        let rtyped_names: Vec<&str> = rtyped_ops.iter().map(|op| op.opname.as_str()).collect();
        assert!(
            !rtyped_names
                .iter()
                .any(|name| *name == "interior_ptr_expose"
                    || *name == "store_through_ptr"
                    || *name == "raw_store"),
            "rtyper must lower expose/store-through; ops={rtyped_names:?}"
        );
        assert_eq!(
            rtyped_names
                .iter()
                .filter(|name| **name == "setinteriorfield")
                .count(),
            1,
            "exactly one setinteriorfield; ops={rtyped_names:?}"
        );

        let converted = effect_graph_from_rtyped_helper(&flow);
        let writes: Vec<_> = converted
            .iter_block_ops()
            .filter_map(|(_, op)| match &op.kind {
                OpKind::InteriorFieldWrite { item_ty, field, .. } => {
                    Some((item_ty.clone(), field.name.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            writes,
            vec![(ValueType::Ref(None), "value".to_string())],
            "one InteriorFieldWrite of kind r; ops={:?}",
            converted
                .iter_block_ops()
                .map(|(_, op)| format!("{:?}", op.kind))
                .collect::<Vec<_>>()
        );
        assert!(
            converted.iter_block_ops().all(|(_, op)| !matches!(
                op.kind,
                OpKind::RawStore { .. }
                    | OpKind::StoreThroughPtr { .. }
                    | OpKind::InteriorPtrExpose { .. }
            )),
            "no RawStore / StoreThroughPtr / InteriorPtrExpose left"
        );
    }
}
