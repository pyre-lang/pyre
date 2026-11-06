//! `Option<fn>` return phi: `Some` is the fn-ptr parameter as the front
//! end spells it (`tyref_fn_ptr_input_class_root`), `None` is a live
//! spelled `null_fn` (`s_None`).
//!
//! The Input seeds `Impossible`; spelled `null_fn` is `s_None`; the phi
//! is `s_None` (or a function-PBC once a caller unions). `stamp_fn_ptr`
//! keeps `Ptr(FuncType)` on the dests; `history.getkind` of that pointer
//! is `'int'`. Publication copies `specialize_block`'s Constant onto
//! the None arms (`rtype_const_result`) before jtransform.

use majit_translate::codewriter::call::CallControl;
use majit_translate::codewriter::codewriter::CodeWriter;
use majit_translate::codewriter::jitcode::JitCode;
use majit_translate::model::{FunctionGraph, OpKind, ValueType};
use majit_translate::{CallPath, GraphTransformConfig};
use std::sync::Arc;

fn option_fn_return_phi_graph() -> FunctionGraph {
    let mut graph = FunctionGraph::new("option_fn_return_phi").with_return_type("i64");
    let (pred_some, some_args) = graph.create_block_with_arg_vars(1);
    let some_phi = some_args[0].clone();
    let (pred_split, split_args) = graph.create_block_with_arg_vars(1);
    let pred_none1 = graph.create_block();
    let pred_none2 = graph.create_block();
    let start = graph.startblock;
    let cond = graph
        .push_op_var(
            start,
            OpKind::Input {
                name: "c".into(),
                ty: ValueType::Bool,
                class_root: None,
            },
            true,
        )
        .unwrap();
    let some_fn = graph
        .push_op_var(
            start,
            OpKind::Input {
                name: "f".into(),
                ty: ValueType::Int,
                class_root: Some("fn() -> i64".into()),
            },
            true,
        )
        .unwrap();
    graph.push_inputarg_var(start, cond.clone());
    graph.push_inputarg_var(start, some_fn.clone());
    graph.set_branch(
        start,
        cond.clone(),
        pred_some,
        vec![some_fn],
        pred_split,
        vec![cond],
    );
    graph.set_return(pred_some, Some(some_phi));
    // Thread the cond so the fused dual-gate Matches. Publication
    // copies `specialize_block`'s Constant onto the None arms
    // (`rtype_const_result`) before jtransform.
    graph.set_branch(
        pred_split,
        split_args[0].clone(),
        pred_none1,
        vec![],
        pred_none2,
        vec![],
    );
    let none1 = graph.push_null_fn_ptr_with_spelling(pred_none1, Some("fn() -> i64"));
    graph.set_return(pred_none1, Some(none1));
    let none2 = graph.push_null_fn_ptr_with_spelling(pred_none2, Some("fn() -> i64"));
    graph.set_return(pred_none2, Some(none2));
    graph
}

#[test]
fn option_fn_return_phi_assembles_as_int() {
    let graph = option_fn_return_phi_graph();
    let path = CallPath::from_segments(["option_fn_return_phi"]);
    let mut callcontrol = CallControl::new();
    callcontrol.register_function_graph(path.clone(), graph.clone());
    let mut codewriter = CodeWriter::new();
    let jitcode = Arc::new(JitCode::new("option_fn_return_phi"));
    codewriter.transform_graph_to_jitcode(
        &graph,
        &path,
        &mut callcontrol,
        &GraphTransformConfig::default(),
        &jitcode,
        false,
        0,
    );
    assert_eq!(jitcode.calldescr().result_type, 'i');
}
