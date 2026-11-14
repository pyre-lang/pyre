//! `Wtf8CodePoints::nth` → `rutf8.py` byte-position walk.
//!
//! ## Positioning
//!
//! `Wtf8::code_points` is already the string (`call.rs`). `Iterator::count`
//! on that iterator is `codepoints_in_utf8`. `Iterator::next` is the
//! for-loop `__iter_next` rewrite. Random-access `nth(n)` is none of
//! those: it is `unicodeobject.py` `_index_to_byte` without index
//! storage — step `n` times with `rutf8.py` `next_codepoint_pos` from
//! byte position 0 while the position is inside the string, then
//! `codepoint_at_pos`, else past the end.
//!
//! The walk lives in a minted look-inside helper, the same construction
//! [`crate::front::slice_ascii::register_first_non_ascii_char_helpers`]
//! uses for `rutf8.first_non_ascii_char`. The helper returns the signed
//! code point or `-1` (the same miss sentinel `first_non_ascii_char`
//! uses). Each call site wraps that word as dest-`Option` through
//! `emit_option_variant` + `resolve_option_dest_owners`, which is the
//! representation other Option-returning rewrites (`slice_first`,
//! `str.find` → `Option<usize>`) already mint, so `.expect(..).to_u32()`
//! reads the same classdef a static `Some(..)` would.

use crate::codewriter::call::CallControl;
use crate::flowspace::model::Variable;
use crate::model::{
    BlockId, CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};
use crate::parse::CallPath;

/// Linear `nth` over a utf8 string: `rutf8.py` `next_codepoint_pos` /
/// `codepoint_at_pos`.
pub(crate) const LL_WTF8_NTH: &str = "ll_wtf8_nth";

/// Mint and register the look-inside [`LL_WTF8_NTH`] helper.
pub(crate) fn register_ll_wtf8_nth_helper(cc: &mut CallControl) {
    let path = CallPath::from_segments([LL_WTF8_NTH]);
    if cc.has_function_graph(&path) {
        return;
    }
    let graph = build_ll_wtf8_nth_graph().with_return_type("i64");
    cc.register_function_graph(path, graph);
}

/// `ll_wtf8_nth(s, n)`: walk `n` code points from byte 0; return
/// `codepoint_at_pos(s, pos)` or `-1` when `n` is at or past the count.
pub(crate) fn build_ll_wtf8_nth_graph() -> FunctionGraph {
    let mut graph = FunctionGraph::new(LL_WTF8_NTH);
    let start = graph.startblock;

    let s = graph.alloc_value_var();
    graph.push_inputarg_var(start, s.clone());
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "s".to_string(),
            ty: ValueType::Str,
            class_root: None,
        },
        s.clone(),
    );

    let n = graph.alloc_value_var();
    graph.push_inputarg_var(start, n.clone());
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "n".to_string(),
            ty: ValueType::Int,
            class_root: None,
        },
        n.clone(),
    );

    let nlen = push_call(
        &mut graph,
        start,
        &["__strlen"],
        vec![s.clone()],
        ValueType::Int,
    );
    let pos = push_const_int(&mut graph, start, 0);
    let i = push_const_int(&mut graph, start, 0);

    let (loop_bb, loop_args) = graph.create_block_with_arg_vars(5);
    let (step_check_bb, step_check_args) = graph.create_block_with_arg_vars(5);
    let (step_bb, step_args) = graph.create_block_with_arg_vars(5);
    let (after_bb, after_args) = graph.create_block_with_arg_vars(3);
    let (some_bb, some_args) = graph.create_block_with_arg_vars(2);
    let (none_bb, _) = graph.create_block_with_arg_vars(0);

    graph.set_goto(start, loop_bb, vec![s, n, nlen, pos, i]);

    let s_loop = loop_args[0].clone();
    let n_loop = loop_args[1].clone();
    let nlen_loop = loop_args[2].clone();
    let pos_loop = loop_args[3].clone();
    let i_loop = loop_args[4].clone();
    let still = push_binop(&mut graph, loop_bb, "lt", i_loop.clone(), n_loop.clone());
    graph.set_branch(
        loop_bb,
        still,
        step_check_bb,
        vec![
            s_loop.clone(),
            n_loop.clone(),
            nlen_loop.clone(),
            pos_loop.clone(),
            i_loop.clone(),
        ],
        after_bb,
        vec![s_loop, nlen_loop, pos_loop],
    );

    let s_chk = step_check_args[0].clone();
    let n_chk = step_check_args[1].clone();
    let nlen_chk = step_check_args[2].clone();
    let pos_chk = step_check_args[3].clone();
    let i_chk = step_check_args[4].clone();
    let inside = push_binop(
        &mut graph,
        step_check_bb,
        "lt",
        pos_chk.clone(),
        nlen_chk.clone(),
    );
    graph.set_branch(
        step_check_bb,
        inside,
        step_bb,
        vec![s_chk, n_chk, nlen_chk, pos_chk, i_chk],
        none_bb,
        vec![],
    );

    let s_step = step_args[0].clone();
    let n_step = step_args[1].clone();
    let nlen_step = step_args[2].clone();
    let pos_step = step_args[3].clone();
    let i_step = step_args[4].clone();
    let pos_next = push_call(
        &mut graph,
        step_bb,
        &["pyre_object", "rutf8", "next_codepoint_pos"],
        vec![s_step.clone(), pos_step],
        ValueType::Int,
    );
    let one = push_const_int(&mut graph, step_bb, 1);
    let i_next = push_binop(&mut graph, step_bb, "add", i_step, one);
    graph.set_goto(
        step_bb,
        loop_bb,
        vec![s_step, n_step, nlen_step, pos_next, i_next],
    );

    let s_after = after_args[0].clone();
    let nlen_after = after_args[1].clone();
    let pos_after = after_args[2].clone();
    let inside_after = push_binop(&mut graph, after_bb, "lt", pos_after.clone(), nlen_after);
    graph.set_branch(
        after_bb,
        inside_after,
        some_bb,
        vec![s_after, pos_after],
        none_bb,
        vec![],
    );

    let s_some = some_args[0].clone();
    let pos_some = some_args[1].clone();
    let cp = push_call(
        &mut graph,
        some_bb,
        &["pyre_object", "rutf8", "codepoint_at_pos"],
        vec![s_some, pos_some],
        ValueType::Int,
    );
    graph.set_return(some_bb, Some(cp));

    let miss = push_const_int(&mut graph, none_bb, -1);
    graph.set_return(none_bb, Some(miss));

    graph
}

/// Dest `Option<T>` payload the nth diamond stores in `Some.__pos_0`.
///
/// `ClassDef.generalize_attr` / `unionof` keep one annotation per
/// attribute. A `Ref` payload is the named instance intern publishes
/// (`intern_opaque_payload_spelling` / `intern_adt_type_decl_identity`
/// for an Opaque TypeDecl, `emit_aggregate_ctor` for a Struct). The
/// ctor name is that identity so `intern_class_by_qualname` shares the
/// FORCE `__pos_0` class (`getuniqueclassdef`). A scalar / pointer
/// payload is the helper word. Never a silent word store into a struct
/// payload.
#[derive(Clone, Debug)]
pub(crate) enum NthPayload {
    Word,
    Instance {
        owner: String,
        field: Option<(String, ValueType)>,
    },
}

/// Wrap the helper word as [`NthPayload::Instance`], else return it.
pub(crate) fn emit_nth_payload(
    graph: &mut FunctionGraph,
    block: BlockId,
    value: Variable,
    payload: &NthPayload,
) -> Variable {
    let NthPayload::Instance { owner, field } = payload else {
        return value;
    };
    let shell = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(shell.clone()),
        kind: OpKind::Call {
            // Empty `owner_path`: intern under `owner`
            // (`intern_adt_type_decl_identity` / `class_identity_spelling`).
            // A crate-prefixed path mints a second ClassDef
            // (`intern_class_by_qualname` of the dotted TypeDecl path)
            // that `unionof` cannot join with the FORCE payload row.
            target: CallTarget::synthetic_transparent_struct_ctor(Vec::new(), owner.clone()),
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(owner.clone())),
        },
    });
    if let Some((field_name, field_ty)) = field {
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: shell.clone(),
                field: FieldDescriptor::new(field_name.clone(), Some(owner.clone()))
                    .with_base_is_deref(false),
                value: LinkArg::Value(value),
                ty: field_ty.clone(),
            },
        });
    }
    shell
}

fn push_call(
    graph: &mut FunctionGraph,
    block: BlockId,
    segments: &[&str],
    args: Vec<Variable>,
    result_ty: ValueType,
) -> Variable {
    let res = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path(segments.iter().copied()),
            args: crate::model::call_args(args),
            result_ty,
        },
    });
    res
}

fn push_const_int(graph: &mut FunctionGraph, block: BlockId, value: i64) -> Variable {
    let res = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::ConstInt(value),
    });
    res
}

fn push_binop(
    graph: &mut FunctionGraph,
    block: BlockId,
    op: &str,
    lhs: Variable,
    rhs: Variable,
) -> Variable {
    let res = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::BinOp {
            op: op.to_string(),
            lhs,
            rhs,
            result_ty: ValueType::Int,
        },
    });
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::call::CallControl;
    use crate::model::CallTarget;

    fn has_path(g: &FunctionGraph, leaf: &str) -> bool {
        g.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some(leaf)
                )
            })
        })
    }

    fn has_full_path(g: &FunctionGraph, want: &[&str]) -> bool {
        g.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.iter().map(String::as_str).eq(want.iter().copied())
                )
            })
        })
    }

    fn has_loop_back_edge(g: &FunctionGraph) -> bool {
        g.blocks.iter().any(|block| {
            let mut seen = vec![false; g.blocks.len()];
            let mut stack = vec![block.id];
            let mut first = true;
            while let Some(id) = stack.pop() {
                if !first && id == block.id {
                    return true;
                }
                first = false;
                if seen[id.0] {
                    continue;
                }
                seen[id.0] = true;
                for exit in &g.blocks[id.0].exits {
                    stack.push(exit.target);
                }
            }
            false
        })
    }

    #[test]
    fn helper_graph_steps_with_rutf8_and_returns_a_code_point_or_minus_one() {
        let g = build_ll_wtf8_nth_graph();
        assert!(has_path(&g, "__strlen"), "byte length is __strlen");
        assert!(
            has_full_path(&g, &["pyre_object", "rutf8", "next_codepoint_pos"]),
            "step is rutf8.py next_codepoint_pos"
        );
        assert!(
            has_full_path(&g, &["pyre_object", "rutf8", "codepoint_at_pos"]),
            "read is rutf8.py codepoint_at_pos"
        );
        assert!(
            has_loop_back_edge(&g),
            "nth walks n steps from byte position 0"
        );
        assert!(
            g.blocks.iter().any(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| matches!(&op.kind, OpKind::ConstInt(-1)))
            }),
            "past-the-end is -1"
        );
        assert!(!has_path(&g, "nth"));
        assert!(!has_full_path(&g, &["Wtf8CodePoints", "nth"]));
        assert!(!has_full_path(&g, &["Wtf8CodePoints", "next"]));
    }

    #[test]
    fn helper_is_registered_on_call_control() {
        let mut cc = CallControl::new();
        register_ll_wtf8_nth_helper(&mut cc);
        let path = CallPath::from_segments([LL_WTF8_NTH]);
        assert!(cc.has_function_graph(&path));
        register_ll_wtf8_nth_helper(&mut cc);
        assert!(cc.has_function_graph(&path));
        let g = cc
            .function_graphs()
            .get_built(&path)
            .expect("registered helper graph");
        assert_eq!(g.return_type.as_deref(), Some("i64"));
        assert!(has_full_path(
            g.as_ref(),
            &["pyre_object", "rutf8", "next_codepoint_pos"]
        ));
    }
}
