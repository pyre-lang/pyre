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
    cc.register_function_graph(path.clone(), graph);
    // rutf8.py `codepoint_position_at_index` / `codepoint_index_at_byte_position`
    // are `@jit.elidable`. An unbounded per-code-point walk is residual
    // (`look_inside_graph`) and is not unroll-safe.
    cc.mark_elidable(path);
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
            // `Wtf8CodePoints::nth(n)`: `n` is `usize` / `r_uint`.
            ty: ValueType::Unsigned,
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
    // Counter starts at 0 (`SomeInteger` signed nonneg). `n` is unsigned;
    // `rint.py` `_rtype_compare_template` allows `lt` of that pair.
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
/// attribute. A `Ref` payload whose TypeDecl is a Struct is the named
/// instance `emit_aggregate_ctor` publishes, with the helper word
/// written to the unique non-ZST field — retyped to the field's integer
/// kind (`signedness_retype_op` / `rarithmetic.py` `r_uint`) when that
/// kind is Unsigned. The ctor name is that identity so
/// `intern_class_by_qualname` shares the FORCE `__pos_0` class
/// (`getuniqueclassdef`). A scalar / pointer payload is the helper
/// word. An Opaque TypeDecl (or Alias of Opaque) cannot carry the code
/// point; `nth_payload_shape` declines. Never a silent word store into
/// a struct payload.
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
        let stored = nth_word_for_field(graph, block, value, field_ty);
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: shell.clone(),
                field: FieldDescriptor::new(field_name.clone(), Some(owner.clone()))
                    .with_base_is_deref(false),
                value: LinkArg::Value(stored),
                ty: field_ty.clone(),
            },
        });
    }
    shell
}

/// `ll_wtf8_nth` returns a signed `SomeInteger`. Store the field's integer
/// kind: `ClassDef.generalize_attr` / `unionof` reject `int ∪ r_uint`.
/// `signedness_retype_op` (`rarithmetic.py` `r_uint`) when the field is
/// Unsigned. A signed field already matches the helper word.
fn nth_word_for_field(
    graph: &mut FunctionGraph,
    block: BlockId,
    value: Variable,
    field_ty: &ValueType,
) -> Variable {
    // Helper word is signed (`ll_wtf8_nth` / `ValueType::Int`). A signed
    // field matches that kind; only Unsigned differs (`r_uint`).
    if !matches!(field_ty, ValueType::Unsigned) {
        return value;
    }
    let (kind, result) = crate::front::mir::signedness_retype_op(graph, value, false);
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(result.clone()),
        kind,
    });
    result
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
        assert!(
            g.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Input {
                            name,
                            ty: ValueType::Unsigned,
                            ..
                        } if name == "n"
                    )
                })
            }),
            "n is usize / r_uint"
        );
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

    /// Caller of `Wtf8CodePoints::nth(s, n)`: `n` is `usize`.
    fn nth_caller_graph() -> FunctionGraph {
        let mut graph = FunctionGraph::new("nth_caller");
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
                ty: ValueType::Unsigned,
                class_root: None,
            },
            n.clone(),
        );
        let result = push_call(
            &mut graph,
            start,
            &[LL_WTF8_NTH],
            vec![s, n],
            ValueType::Int,
        );
        graph.set_return(start, Some(result));
        graph
    }

    /// Call propagation: `derive_subject_inputcells` seeds `ll_wtf8_nth`
    /// from its Input ops, then a caller with `usize` `n` unions into
    /// that cell (`annrpython.py` `recursivecall` / `mergeinputargs`).
    /// A signed non-nonneg seed (`ValueType::Int`) fails
    /// `binaryop.py` `pairtype(SomeInteger, SomeInteger).union`.
    #[test]
    fn unsigned_caller_n_annotates_into_ll_wtf8_nth() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeValue, unionof};
        use crate::translator::rtyper::flowspace_adapter::derive_subject_inputcells;

        let helper = build_ll_wtf8_nth_graph();
        let caller = nth_caller_graph();
        let ann = RPythonAnnotator::new(None, None, None, false);

        let helper_cells = derive_subject_inputcells(&helper, Some(&ann.bookkeeper))
            .expect("seed ll_wtf8_nth from its Input ops");
        assert_eq!(helper_cells.len(), 2, "ll_wtf8_nth(s, n)");
        let caller_cells = derive_subject_inputcells(&caller, Some(&ann.bookkeeper))
            .expect("seed nth_caller from its Input ops");
        assert_eq!(caller_cells.len(), 2, "nth_caller(s, n)");
        match &caller_cells[1] {
            SomeValue::Integer(i) => {
                assert!(i.unsigned, "caller n is usize / r_uint");
                assert!(i.nonneg);
            }
            other => panic!("caller n is SomeInteger r_uint, got {other:?}"),
        }
        match &helper_cells[1] {
            SomeValue::Integer(i) => {
                assert!(i.unsigned, "ll_wtf8_nth n is usize / r_uint");
                assert!(i.nonneg);
            }
            other => panic!("ll_wtf8_nth n is SomeInteger r_uint, got {other:?}"),
        }
        unionof([&helper_cells[1], &caller_cells[1]]).expect(
            "caller's usize n must union into ll_wtf8_nth's n cell \
             (binaryop.py pairtype(SomeInteger, SomeInteger).union)",
        );
    }

    /// Counter `ConstInt(0)` is signed nonneg; `n` is unsigned. `lt`
    /// annotates `SomeBool` (`binaryop.py` `_compare_helper`) and rtypes
    /// through `rint.py` `_rtype_compare_template` as `uint_lt`.
    #[test]
    fn counter_lt_unsigned_n_annotates_and_rtypes() {
        use std::cell::RefCell;
        use std::rc::Rc;
        use std::sync::Arc;

        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeBool, SomeInteger, SomeValue};
        use crate::flowspace::model::{Hlvalue, SpaceOperation as FsOp, Variable as FsVar};
        use crate::translator::rtyper::call_registry::CallRegistry;
        use crate::translator::rtyper::flowspace_adapter::{
            derive_subject_inputcells, function_graph_to_flowspace,
        };
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rbool::bool_repr;
        use crate::translator::rtyper::rint::{rtype_compare_template, signed_repr, unsigned_repr};
        use crate::translator::rtyper::rmodel::Repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};

        let mut graph = FunctionGraph::new("counter_lt_unsigned_n");
        let start = graph.startblock;
        let n = graph.alloc_value_var();
        graph.push_inputarg_var(start, n.clone());
        graph.push_op_with_result_var(
            start,
            OpKind::Input {
                name: "n".to_string(),
                ty: ValueType::Unsigned,
                class_root: None,
            },
            n.clone(),
        );
        let i = push_const_int(&mut graph, start, 0);
        let still = push_binop(&mut graph, start, "lt", i, n);
        graph.set_return(start, Some(still));

        let ann = RPythonAnnotator::new(None, None, None, false);
        let registry = CallRegistry::new(ann.bookkeeper.clone());
        let lifted =
            function_graph_to_flowspace(&graph, &registry).expect("lift counter_lt_unsigned_n");
        let cells = derive_subject_inputcells(&graph, Some(&ann.bookkeeper))
            .expect("seed counter_lt_unsigned_n");
        assert_eq!(cells.len(), 1);
        let inputcells: Vec<_> = cells.into_iter().map(Some).collect();
        let ret = ann
            .build_graph_types(&lifted.graph, &inputcells, true)
            .expect("annotate i < n")
            .expect("lt result");
        assert!(
            matches!(ret, SomeValue::Bool(_)),
            "lt of signed-nonneg counter against unsigned n annotates SomeBool, got {ret:?}"
        );

        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let v_left = FsVar::new();
        v_left.set_concretetype(Some(LowLevelType::Signed));
        let v_right = FsVar::new();
        v_right.set_concretetype(Some(LowLevelType::Unsigned));
        let v_result = FsVar::new();
        v_result.set_concretetype(Some(LowLevelType::Bool));
        let hop = HighLevelOp::new(
            rtyper,
            FsOp::new(
                "lt".to_string(),
                vec![Hlvalue::Variable(v_left), Hlvalue::Variable(v_right)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(true, false)),
            SomeValue::Integer(SomeInteger::new(false, true)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(signed_repr() as Arc<dyn Repr>),
            Some(unsigned_repr() as Arc<dyn Repr>),
        ]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Bool(SomeBool::new()));
        *hop.r_result.borrow_mut() = Some(bool_repr() as Arc<dyn Repr>);

        let result =
            rtype_compare_template(&hop, "lt").expect("lt of nonneg counter vs r_uint rtypes");
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert!(
            ops.ops.iter().any(|op| op.opname == "uint_lt"),
            "rint.py _rtype_compare_template emits uint_lt, ops={:?}",
            ops.ops
                .iter()
                .map(|op| op.opname.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn helper_is_elidable_not_unroll_safe() {
        use crate::codewriter::call::AnalysisCache;
        use majit_ir::descr::{ExtraEffect, OopSpecIndex};
        use majit_ir::value::Type;

        let mut cc = CallControl::new();
        register_ll_wtf8_nth_helper(&mut cc);
        cc.find_all_graphs_for_tests();

        let path = CallPath::from_segments([LL_WTF8_NTH]);
        let target = CallTarget::function_path([LL_WTF8_NTH]);
        assert!(
            cc.is_elidable(&target),
            "ll_wtf8_nth carries func._elidable_function_"
        );
        assert!(
            !cc.has_cannot_raise_assertion(&target),
            "synthesized helper has no native fnaddr; has_cannot_raise_assertion is the executor-fold gate"
        );
        assert!(!cc.has_function_fnaddr(&path));

        let g = cc
            .function_graphs()
            .get_built(&path)
            .expect("registered helper graph");
        assert!(g.func.elidable);
        assert!(!g.func.unroll_safe);
        assert!(
            g.hints.iter().any(|h| h == "elidable"),
            "graph.hints={:?}",
            g.hints
        );
        assert!(
            !g.hints.iter().any(|h| h == "unroll_safe"),
            "unbounded per-code-point walk is not unroll-safe; hints={:?}",
            g.hints
        );

        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &crate::model::SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: target.clone(),
                    args: crate::model::call_args(vec![Variable::new(), Variable::new()]),
                    result_ty: ValueType::Int,
                },
            },
            vec![Type::Ref, Type::Int],
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
            None,
        );
        assert!(
            matches!(
                descriptor.extra_info.extraeffect,
                ExtraEffect::ElidableCannotRaise
                    | ExtraEffect::ElidableOrMemoryError
                    | ExtraEffect::ElidableCanRaise
            ),
            "ll_wtf8_nth extraeffect was {:?}",
            descriptor.extra_info.extraeffect
        );
    }
}
