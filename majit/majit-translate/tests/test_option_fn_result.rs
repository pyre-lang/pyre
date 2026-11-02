//! `Option<fn>` return phi: `Some` is an int, `None` is a live `null_fn`.
//!
//! T1c-1 annotates `core::ptr::null_fn` as constant `SomePtr(Ptr(FuncType))`.
//! `follow_link` copies that shell onto the returnblock inputarg. Upstream
//! `history.getkind` / `codewriter/support.py` (via `getkind`) banks a
//! `Ptr` whose `TO` is `FuncType` (`_gckind == 'raw'`) as `'int'`. A
//! projection that maps every `SomePtr` to `Ref` commits that phi as
//! `GcRef`, so `func_result_kind` panics `FUNC.RESULT=i but CFG return
//! kind is r`.
//!
//! The graph is unfolded: return predecessors are an Int `Some` input
//! plus two live `null_fn` `None` arms (the extra `None` is the CI
//! shape: after `Int ∪ Ref` clears the phi, a later `SomePtr` rewrite
//! sticks), `return_type = Some("i64")`, driven through
//! `CodeWriter::transform_graph_to_jitcode` once. A test that
//! `set_return`s the `null_fn` result Variable itself stays green
//! without the projection fix.

use majit_translate::codewriter::call::CallControl;
use majit_translate::codewriter::codewriter::CodeWriter;
use majit_translate::codewriter::jitcode::JitCode;
use majit_translate::model::{FunctionGraph, OpKind, ValueType};
use majit_translate::{CallPath, GraphTransformConfig};
use std::sync::Arc;

fn option_fn_return_phi_graph() -> FunctionGraph {
    let mut graph = FunctionGraph::new("option_fn_return_phi").with_return_type("i64");
    let pred_some = graph.create_block();
    let pred_split = graph.create_block();
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
                class_root: None,
            },
            true,
        )
        .unwrap();
    graph.set_branch(start, cond.clone(), pred_some, vec![], pred_split, vec![]);
    graph.set_return(pred_some, Some(some_fn));
    // Reuse the startblock Input as the inner switch without threading it
    // through `pred_split` inputargs so dual-gate Skip-classifies (the
    // production `builtin_fixed_arity_fn` path is Skip). Two `null_fn`
    // `None` arms sit after the Int `Some` in block order so the phi's
    // last annotation is `SomePtr`.
    graph.set_branch(pred_split, cond, pred_none1, vec![], pred_none2, vec![]);
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
