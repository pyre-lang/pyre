//! `w_str_from_wtf8_managed` inlines `w_str_from_wtf8_managed_borrowed`.
//! The caller's `inline_call_*` reskind must match the callee terminator
//! (`jtransform.py rewrite_call` / `call.py` `RESULT == FUNC.RESULT`).

mod common;

use common::{OBJECT_LLBC, load_llbc_if_present, lower_function};
use majit_charon_reader::Llbc;
use majit_translate::codewriter::call::CallControl;
use majit_translate::codewriter::codewriter::CodeWriter;
use majit_translate::codewriter::jitcode::JitCode;
use majit_translate::codewriter::jtransform::Transformer;
use majit_translate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs;
use majit_translate::insns::{
    BC_INLINE_CALL_IR_I, BC_INLINE_CALL_IR_R, BC_INLINE_CALL_IR_V, BC_INLINE_CALL_R_I,
    BC_INLINE_CALL_R_R, BC_INLINE_CALL_R_V,
};
use majit_translate::model::{CallTarget, ConcreteType, FunctionGraph, OpKind};
use majit_translate::{CallPath, GraphTransformConfig};

fn object_llbc() -> Option<&'static Llbc> {
    load_llbc_if_present(OBJECT_LLBC)
}

fn is_borrowed(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments
            .last()
            .is_some_and(|s| s == "w_str_from_wtf8_managed_borrowed"),
        CallTarget::Method { name, .. } => name == "w_str_from_wtf8_managed_borrowed",
        _ => false,
    }
}

fn graph_result_kind(graph: &FunctionGraph) -> char {
    let Some(arg) = graph.block(graph.returnblock).inputargs.first() else {
        return 'v';
    };
    match FunctionGraph::concretetype_of(arg) {
        ConcreteType::Signed => 'i',
        ConcreteType::GcRef => 'r',
        ConcreteType::Float => 'f',
        ConcreteType::Void => 'v',
        ConcreteType::Unknown => '?',
    }
}

#[test]
fn w_str_from_wtf8_managed_inlines_borrowed_with_func_result() {
    let Some(llbc) = object_llbc() else {
        eprintln!("skipping: {OBJECT_LLBC} is missing");
        return;
    };
    let caller_name = "w_str_from_wtf8_managed";
    let callee_name = "w_str_from_wtf8_managed_borrowed";
    let caller =
        lower_function(llbc, caller_name).unwrap_or_else(|e| panic!("lower {caller_name}: {e}"));
    let callee =
        lower_function(llbc, callee_name).unwrap_or_else(|e| panic!("lower {callee_name}: {e}"));

    eprintln!(
        "CALLEE return_type={:?} class_root={:?} cfg_kind={} returnblock_args={}",
        callee.return_type,
        callee.return_class_root,
        graph_result_kind(&callee),
        callee.block(callee.returnblock).inputargs.len()
    );
    eprintln!(
        "CALLER return_type={:?} class_root={:?} cfg_kind={}",
        caller.return_type,
        caller.return_class_root,
        graph_result_kind(&caller)
    );

    let mut calls = Vec::new();
    for (bi, block) in caller.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            let OpKind::Call {
                target, result_ty, ..
            } = &op.kind
            else {
                continue;
            };
            if !is_borrowed(target) {
                continue;
            }
            let ct = op.result.as_ref().map(FunctionGraph::concretetype_of);
            calls.push((
                bi,
                oi,
                format!("{target}"),
                result_ty.clone(),
                ct,
                op.result.is_some(),
            ));
            eprintln!(
                "CALL bb{bi} op{oi} target={target} result_ty={result_ty:?} dest={:?} dest_ct={ct:?}",
                op.result.is_some()
            );
        }
    }
    assert!(
        !calls.is_empty(),
        "{caller_name} has no Call of {callee_name}\n{}",
        caller.dump()
    );

    let harvest = harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(llbc));
    let callee_path = match &caller
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .find_map(|op| match &op.kind {
            OpKind::Call { target, .. } if is_borrowed(target) => Some(target.clone()),
            _ => None,
        })
        .expect("borrowed target")
    {
        CallTarget::FunctionPath { segments, .. } => {
            CallPath::from_segments(segments.iter().map(String::as_str))
        }
        other => panic!("borrowed target {other}"),
    };
    let caller_path =
        CallPath::from_segments(["pyre_object", "unicodeobject", "w_str_from_wtf8_managed"]);
    let config = GraphTransformConfig::default();
    let mut cc = CallControl::new();
    cc.raw_alloc_flavor_structs = harvest;
    cc.register_function_graph(callee_path.clone(), callee.clone());
    cc.add_candidate_graph(callee_path.clone());
    cc.register_function_graph(caller_path.clone(), caller.clone());
    cc.add_candidate_graph(caller_path.clone());

    let rewritten = Transformer::new(&config)
        .with_callcontrol(&mut cc)
        .transform(&caller)
        .graph;
    let mut inlines = Vec::new();
    for op in rewritten.blocks.iter().flat_map(|b| b.operations.iter()) {
        let OpKind::InlineCall {
            jitcode,
            result_kind,
            ..
        } = &op.kind
        else {
            continue;
        };
        if jitcode.name != callee_name {
            continue;
        }
        let ct = op.result.as_ref().map(FunctionGraph::concretetype_of);
        inlines.push((*result_kind, ct, op.result.is_some()));
        eprintln!(
            "INLINE jitcode={} result_kind={} dest={} dest_ct={ct:?}",
            jitcode.name,
            result_kind,
            op.result.is_some()
        );
    }
    assert!(
        !inlines.is_empty(),
        "{caller_name} rewrite has no inline_call of {callee_name}"
    );

    let mut codewriter = CodeWriter::new();
    let jitcode = std::sync::Arc::new(JitCode::new("w_str_from_wtf8_managed"));
    codewriter.transform_graph_to_jitcode(
        &caller,
        &caller_path,
        &mut cc,
        &config,
        &jitcode,
        false,
        0,
    );
    let body = jitcode.body();
    let has_v = body.code.contains(&BC_INLINE_CALL_R_V) || body.code.contains(&BC_INLINE_CALL_IR_V);
    let has_i = body.code.contains(&BC_INLINE_CALL_R_I) || body.code.contains(&BC_INLINE_CALL_IR_I);
    let has_r = body.code.contains(&BC_INLINE_CALL_R_R) || body.code.contains(&BC_INLINE_CALL_IR_R);
    eprintln!(
        "ASSEMBLED has_v={has_v} has_i={has_i} has_r={has_r} resulttypes={:?}",
        body.resulttypes
    );
    assert!(
        !has_v,
        "{caller_name} must not encode inline_call_*_v of {callee_name}; code={:?} resulttypes={:?}",
        body.code, body.resulttypes
    );
    assert!(
        has_i || has_r,
        "{caller_name} must encode inline_call_*_i or _r of {callee_name}; inlines={inlines:?}"
    );
}
