//! The hybrid consumer's helper roots use the public codewriter API.
use majit_translate::codewriter::{CodeWriter, call::CallControl, policy::DefaultJitPolicy};
use majit_translate::model::{FunctionGraph, OpKind, SpaceOperation, ValueType};
use majit_translate::{CallPath, CallTarget, GraphTransformConfig};

fn return_graph(name: &str) -> FunctionGraph {
    let mut graph = FunctionGraph::new(name);
    graph.set_return(graph.startblock, None);
    graph
}

#[test]
fn helper_roots_compile_their_callees_without_a_portal_or_unrelated_graphs() {
    let mut call_control = CallControl::new();
    let helper = CallPath::from_segments(["fixture", "helper"]);
    let callee = CallPath::from_segments(["fixture", "callee"]);
    let unrelated = CallPath::from_segments(["fixture", "engine_setup"]);
    let mut graph = return_graph("helper");
    graph
        .block_mut(graph.startblock)
        .operations
        .push(SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: callee.segments.clone(),
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: Vec::new(),
                result_ty: ValueType::Void,
            },
        });
    call_control.register_function_graph(helper.clone(), graph);
    call_control.register_function_graph(callee.clone(), return_graph("callee"));
    call_control.register_function_graph(unrelated.clone(), return_graph("engine_setup"));
    let wrapper = CallPath::from_segments(["fixture", "__majit_wrap_unrelated"]);
    call_control.register_function_graph(wrapper.clone(), return_graph("wrapper"));
    call_control.register_function_fnaddr(wrapper.clone(), 0x1234);
    assert!(
        call_control
            .builtin_wrapper_indirect_graphs()
            .contains(&wrapper)
    );
    call_control.find_helper_graphs(&mut DefaultJitPolicy::new(), &[helper.clone()]);
    let jitcodes =
        CodeWriter::new().make_jitcodes(&mut call_control, &GraphTransformConfig::default());
    assert_eq!(jitcodes.in_order.len(), 2);
    assert!(call_control.jitdrivers_sd().is_empty());
    assert!(jitcodes.by_path.contains_key(&helper));
    assert!(jitcodes.by_path.contains_key(&callee));
    assert!(!jitcodes.by_path.contains_key(&unrelated));
    assert!(!jitcodes.by_path.contains_key(&wrapper));
}

#[test]
#[should_panic(expected = "missing helper graph")]
fn helper_roots_reject_missing_graphs() {
    CallControl::new().find_helper_graphs(
        &mut DefaultJitPolicy::new(),
        &[CallPath::from_segments(["missing"])],
    );
}
