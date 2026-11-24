//! Skip-arm (non-rtyped) `ord` / `unichr` must codewrite without
//! `int_ord` / `int_unichr`.
//!
//! The front emits high-level `ord` (`c as u32`) and `Call(__majit_unichr)`
//! (`char::from_u32_unchecked`). `rtype_ord` / `rtype_unichr` turn those
//! into `cast_unichar_to_int` / `cast_int_to_unichar`, which
//! `rewrite_op_cast_unichar_to_int` / `rewrite_op_cast_int_to_unichar`
//! drop. A graph that reaches the codewriter without being rtyped still
//! carries the high-level hop; jtransform must project it to the same
//! dropped cast so the assembler never names `int_ord` / `int_unichr`.
//!
//! These tests run `transform_graph` first (the skip-arm rewrite, no
//! rtyper) and then `transform_graph_to_jitcode` on the rewritten graph.
//! Driving the original hop through the dual-gate would take the Match
//! arm (`rtype_ord` already drops it) or, for UnaryOp `unichr`, panic in
//! `normalize_unary_op_name` (`operation.py` has no `unichr` unary).

use majit_translate::codewriter::call::CallControl;
use majit_translate::codewriter::codewriter::CodeWriter;
use majit_translate::codewriter::format::format_assembler;
use majit_translate::codewriter::jitcode::JitCode;
use majit_translate::flatten::{FlatOp, SSARepr};
use majit_translate::model::{CallTarget, FunctionGraph, OpKind, ValueType, call_args};
use majit_translate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use majit_translate::{CallPath, GraphSpecialization, GraphTransformConfig, transform_graph};
use std::sync::Arc;

fn leftover_ord_unichr(graph: &FunctionGraph) -> Vec<String> {
    let mut found = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::UnaryOp { op, .. } if op == "ord" || op == "unichr" => {
                    found.push(format!("unary {op}"));
                }
                OpKind::Call { target, .. } => {
                    if let CallTarget::FunctionPath { segments, .. } = target
                        && segments.as_slice() == ["__majit_unichr"]
                    {
                        found.push("call __majit_unichr".into());
                    }
                }
                _ => {}
            }
        }
    }
    found
}

fn assemble(name: &str, graph: FunctionGraph) -> String {
    // Skip-arm: jtransform sees the high-level hop, the rtyper does not.
    let transformed = transform_graph(&graph, &GraphTransformConfig::default());
    let leftover = leftover_ord_unichr(&transformed.graph);
    assert!(
        leftover.is_empty(),
        "skip-arm rewrite left {leftover:?} on {name}"
    );

    let path = CallPath::from_segments([name]);
    let mut callcontrol = CallControl::new();
    callcontrol.register_function_graph(path.clone(), transformed.graph.clone());
    let mut codewriter = CodeWriter::new();
    let jitcode = Arc::new(JitCode::new(name));
    codewriter.transform_graph_to_jitcode(
        &transformed.graph,
        &path,
        &mut callcontrol,
        &GraphTransformConfig::default(),
        &jitcode,
        false,
        0,
    );
    let ssarepr = jitcode
        .body()
        ._ssarepr
        .as_ref()
        .and_then(|repr| repr.as_any().downcast_ref::<SSARepr>())
        .expect("assembled graph keeps its SSA representation");
    let dump = format_assembler(ssarepr);
    for op in &ssarepr.insns {
        match op {
            FlatOp::Op(inner) => match &inner.kind {
                OpKind::UnaryOp { op, .. } => {
                    assert_ne!(op, "ord", "leftover UnaryOp ord becomes int_ord\n{dump}");
                    assert_ne!(
                        op, "unichr",
                        "leftover UnaryOp unichr becomes int_unichr\n{dump}"
                    );
                    assert!(
                        !op.contains("int_ord") && !op.contains("int_unichr"),
                        "assembled unary opname {op} carries int_ord/int_unichr\n{dump}"
                    );
                }
                OpKind::Call { target, .. } => {
                    if let CallTarget::FunctionPath { segments, .. } = target {
                        assert_ne!(
                            segments.as_slice(),
                            ["__majit_unichr"],
                            "leftover __majit_unichr Call\n{dump}"
                        );
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    assert!(
        !dump.contains("int_ord") && !dump.contains("int_unichr"),
        "jitcode dump names int_ord/int_unichr\n{dump}"
    );
    dump
}

fn unary_int_graph(name: &str, op: &str, operand_ll: Option<LowLevelType>) -> FunctionGraph {
    let mut graph = FunctionGraph::new(name).with_return_type("i64");
    let start = graph.startblock;
    let input = graph
        .push_op_var(
            start,
            OpKind::Input {
                name: "c".into(),
                ty: ValueType::Int,
                class_root: None,
            },
            true,
        )
        .expect("input produces a value");
    if let Some(ll) = operand_ll {
        input.set_concretetype(Some(ll));
    }
    graph.push_inputarg_var(start, input.clone());
    let result = graph
        .push_op_var(
            start,
            OpKind::UnaryOp {
                op: op.into(),
                operand: input,
                result_ty: ValueType::Int,
            },
            true,
        )
        .expect("unary produces a value");
    graph.set_return(start, Some(result));
    graph
}

#[test]
fn skip_arm_ord_codewrites_without_int_ord() {
    assemble(
        "skip_arm_ord",
        unary_int_graph("skip_arm_ord", "ord", Some(LowLevelType::UniChar)),
    );
}

#[test]
fn skip_arm_unstamped_ord_codewrites_without_int_ord() {
    assemble(
        "skip_arm_unstamped_ord",
        unary_int_graph("skip_arm_unstamped_ord", "ord", None),
    );
}

#[test]
fn skip_arm_char_ord_codewrites_without_int_ord() {
    assemble(
        "skip_arm_char_ord",
        unary_int_graph("skip_arm_char_ord", "ord", Some(LowLevelType::Char)),
    );
}

#[test]
fn skip_arm_unichr_unary_codewrites_without_int_unichr() {
    assemble(
        "skip_arm_unichr_unary",
        unary_int_graph("skip_arm_unichr_unary", "unichr", None),
    );
}

#[test]
fn skip_arm_unichr_call_codewrites_without_int_unichr() {
    let mut graph = FunctionGraph::new("skip_arm_unichr_call").with_return_type("i64");
    let start = graph.startblock;
    let input = graph
        .push_op_var(
            start,
            OpKind::Input {
                name: "n".into(),
                ty: ValueType::Int,
                class_root: None,
            },
            true,
        )
        .expect("input produces a value");
    graph.push_inputarg_var(start, input.clone());
    let result = graph
        .push_op_var(
            start,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__majit_unichr".to_string()],
                    resolved_path: None,
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                    specialization: GraphSpecialization::Regular,
                },
                args: call_args(vec![input]),
                result_ty: ValueType::Int,
            },
            true,
        )
        .expect("unichr produces a value");
    graph.set_return(start, Some(result));
    assemble("skip_arm_unichr_call", graph);
}
