//! `rstr.py` `LLHelpers.ll_join_chars`: copy a list of chars into a `STR`.
//!
//! ## Positioning
//!
//! A shared `&[u8]` is rstr `STR`. A `Vec<u8>` / `&mut [u8]` stays a list of
//! unsigned. Coercing that list into a shared byte slice is therefore a copy
//! (`bytearrayobject.py` `bytes_w` / `_join_chars`), not an identity view.
//!
//! Upstream:
//!
//! ```python
//! @jit.look_inside_iff(lambda length, chars, RES:
//!     jit.isconstant(length) and jit.isvirtual(chars))
//! def ll_join_chars(length, chars, RES):
//!     result = mallocstr(length)   # or mallocunicode
//!     for i in range(length):
//!         result.chars[i] = cast_primitive(Char, chars[i])
//!     return result
//! ```
//!
//! The front intercepts `Vec<u8>` deref / `as_slice` / `&vec[..]` when the
//! destination is a shared `&[u8]` and plants a call to this helper. The
//! helper is a translator-built rich-model graph registered on
//! `CallControl` before `find_all_graphs`, the same construction
//! [`crate::front::slice_ascii::register_first_non_ascii_char_helpers`]
//! uses. Annotator-known ops only: `__len`, `ArrayRead` → `getitem`,
//! `__majit_chr` → `builtin_chr` / `SomeChar`, and the
//! `__majit_stringbuilder_{new,append,build}` markers. The input is a
//! `Ref` so the caller's `SomeList` flows in; the result is `SomeString`.

use crate::codewriter::call::CallControl;
use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};
use crate::parse::CallPath;
use crate::runtime_names::shims::{
    CHR, STRINGBUILDER_APPEND, STRINGBUILDER_BUILD, STRINGBUILDER_NEW,
};

/// `rstr.py` `LLHelpers.ll_join_chars` specialized to `Ptr(STR)`.
pub(crate) const LL_JOIN_CHARS: &str = "ll_join_chars";

/// Inverse of [`LL_JOIN_CHARS`]: copy a `STR` into a list of unsigned
/// (`Vec<u8>`). `list(str)` yields `Char`; a `Vec<u8>` item is `r_uint`.
pub(crate) const LL_STR2UINTLIST: &str = "ll_str2uintlist";

/// Mint and register the look-inside `ll_join_chars` helper.
pub(crate) fn register_ll_join_chars_helper(cc: &mut CallControl) {
    let path = CallPath::from_segments([LL_JOIN_CHARS]);
    if cc.has_function_graph(&path) {
        return;
    }
    let mut graph = build_ll_join_chars_graph()
        .with_return_type(crate::translator::rtyper::cutover::STR_RETURN_TYPE);
    graph.return_is_str = true;
    cc.register_function_graph(path, graph);
}

/// Mint and register the look-inside inverse of [`LL_JOIN_CHARS`].
pub(crate) fn register_ll_str2uintlist_helper(cc: &mut CallControl) {
    let path = CallPath::from_segments([LL_STR2UINTLIST]);
    if cc.has_function_graph(&path) {
        return;
    }
    let mut graph = build_ll_str2uintlist_graph().with_return_type("ref");
    graph.return_is_str = false;
    graph.return_container_root = Some("Vec<u8>".to_string());
    cc.register_function_graph(path, graph);
}

/// `ll_str2uintlist(s)`: `lst = []; for i in range(len(s)): lst.append(ord(s[i]))`.
pub(crate) fn build_ll_str2uintlist_graph() -> FunctionGraph {
    let mut graph = FunctionGraph::new(LL_STR2UINTLIST);
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
    graph.block_mut(start).operations.push(SpaceOperation {
        result: Some(n.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path(["__strlen"]),
            args: crate::model::call_args(vec![s.clone()]),
            result_ty: ValueType::Int,
        },
    });
    let lst = graph.alloc_value_var();
    graph.block_mut(start).operations.push(SpaceOperation {
        result: Some(lst.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path(["vec", "Vec", "new"]),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(None),
        },
    });
    let i = graph.alloc_value_var();
    graph.block_mut(start).operations.push(SpaceOperation {
        result: Some(i.clone()),
        kind: OpKind::ConstInt(0),
    });

    let (loop_bb, loop_args) = graph.create_block_with_arg_vars(4);
    let (body_bb, body_args) = graph.create_block_with_arg_vars(4);
    let (done_bb, done_args) = graph.create_block_with_arg_vars(1);

    graph.set_goto(start, loop_bb, vec![s, n, i, lst]);

    let s_loop = loop_args[0].clone();
    let n_loop = loop_args[1].clone();
    let i_loop = loop_args[2].clone();
    let lst_loop = loop_args[3].clone();
    let lt = graph.alloc_value_var();
    graph.block_mut(loop_bb).operations.push(SpaceOperation {
        result: Some(lt.clone()),
        kind: OpKind::BinOp {
            op: "lt".to_string(),
            lhs: i_loop.clone(),
            rhs: n_loop.clone(),
            result_ty: ValueType::Int,
        },
    });
    graph.set_branch(
        loop_bb,
        lt,
        body_bb,
        vec![
            s_loop.clone(),
            n_loop.clone(),
            i_loop.clone(),
            lst_loop.clone(),
        ],
        done_bb,
        vec![lst_loop.clone()],
    );

    let s_body = body_args[0].clone();
    let n_body = body_args[1].clone();
    let i_body = body_args[2].clone();
    let lst_body = body_args[3].clone();
    let byte = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(byte.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path(["__string_byte_getitem"]),
            args: crate::model::call_args(vec![s_body.clone(), i_body.clone()]),
            result_ty: ValueType::Unsigned,
        },
    });
    let void = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(void),
        kind: OpKind::Call {
            target: CallTarget::function_path(["vec", "Vec", "push"]),
            args: crate::model::call_args(vec![lst_body.clone(), byte]),
            result_ty: ValueType::Void,
        },
    });
    let one = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(one.clone()),
        kind: OpKind::ConstInt(1),
    });
    let i_next = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(i_next.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs: i_body,
            rhs: one,
            result_ty: ValueType::Int,
        },
    });
    graph.set_goto(body_bb, loop_bb, vec![s_body, n_body, i_next, lst_body]);

    let lst_done = done_args[0].clone();
    graph.set_return(done_bb, Some(lst_done));

    graph
}

/// `ll_join_chars(chars)`: `s = mallocstr(len(chars)); s.chars[i] = chr(chars[i])`.
pub(crate) fn build_ll_join_chars_graph() -> FunctionGraph {
    let mut graph = FunctionGraph::new(LL_JOIN_CHARS);
    let start = graph.startblock;

    let chars = graph.alloc_value_var();
    graph.push_inputarg_var(start, chars.clone());
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "chars".to_string(),
            ty: ValueType::Ref(None),
            // `[u8]` is the list-of-unsigned spelling (`is_list_container_spelling`);
            // a shared `&[u8]` would seed `SomeString`.
            class_root: Some("[u8]".to_string()),
        },
        chars.clone(),
    );

    let n = graph.alloc_value_var();
    graph.block_mut(start).operations.push(SpaceOperation {
        result: Some(n.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path(["__len"]),
            args: crate::model::call_args(vec![chars.clone()]),
            result_ty: ValueType::Int,
        },
    });
    let builder = graph.alloc_value_var();
    graph.block_mut(start).operations.push(SpaceOperation {
        result: Some(builder.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path([STRINGBUILDER_NEW]),
            args: crate::model::call_args(vec![n.clone()]),
            result_ty: ValueType::StringBuilder,
        },
    });
    let i = graph.alloc_value_var();
    graph.block_mut(start).operations.push(SpaceOperation {
        result: Some(i.clone()),
        kind: OpKind::ConstInt(0),
    });

    let (loop_bb, loop_args) = graph.create_block_with_arg_vars(4);
    let (body_bb, body_args) = graph.create_block_with_arg_vars(4);
    let (done_bb, done_args) = graph.create_block_with_arg_vars(1);

    graph.set_goto(start, loop_bb, vec![chars, n, i, builder]);

    let chars_loop = loop_args[0].clone();
    let n_loop = loop_args[1].clone();
    let i_loop = loop_args[2].clone();
    let builder_loop = loop_args[3].clone();
    let lt = graph.alloc_value_var();
    graph.block_mut(loop_bb).operations.push(SpaceOperation {
        result: Some(lt.clone()),
        kind: OpKind::BinOp {
            op: "lt".to_string(),
            lhs: i_loop.clone(),
            rhs: n_loop.clone(),
            result_ty: ValueType::Int,
        },
    });
    graph.set_branch(
        loop_bb,
        lt,
        body_bb,
        vec![
            chars_loop.clone(),
            n_loop.clone(),
            i_loop.clone(),
            builder_loop.clone(),
        ],
        done_bb,
        vec![builder_loop.clone()],
    );

    let chars_body = body_args[0].clone();
    let n_body = body_args[1].clone();
    let i_body = body_args[2].clone();
    let builder_body = body_args[3].clone();
    let byte = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(byte.clone()),
        kind: OpKind::ArrayRead {
            base: chars_body.clone(),
            item_ty: ValueType::Unsigned,
            index: i_body.clone(),
            array_type_id: Some("[u8]".to_string()),
            nolength: crate::front::typestr::nolength_from_array_type_id(Some("[u8]")),
            gckind: crate::front::typestr::gckind_from_array_type_id(Some("[u8]")),
            pure: false,
        },
    });
    let ch = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(ch.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path([CHR]),
            args: crate::model::call_args(vec![byte]),
            result_ty: ValueType::Int,
        },
    });
    let void = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(void),
        kind: OpKind::Call {
            target: CallTarget::function_path([STRINGBUILDER_APPEND]),
            args: crate::model::call_args(vec![builder_body.clone(), ch]),
            result_ty: ValueType::Void,
        },
    });
    let one = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(one.clone()),
        kind: OpKind::ConstInt(1),
    });
    let i_next = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(i_next.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs: i_body,
            rhs: one,
            result_ty: ValueType::Int,
        },
    });
    graph.set_goto(
        body_bb,
        loop_bb,
        vec![chars_body, n_body, i_next, builder_body],
    );

    let builder_done = done_args[0].clone();
    let result = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
    stamp_result_as_rstr_str(&result);
    graph.block_mut(done_bb).operations.push(SpaceOperation {
        result: Some(result.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path([STRINGBUILDER_BUILD]),
            args: crate::model::call_args(vec![builder_done]),
            result_ty: ValueType::Str,
        },
    });
    graph.set_return(done_bb, Some(result));

    graph
}

fn stamp_result_as_rstr_str(var: &Variable) {
    use crate::translator::rtyper::lltypesystem::rstr::STRPTR;
    FunctionGraph::set_concretetype_of_inline(var, crate::model::ConcreteType::GcRef);
    var.set_concretetype(Some(STRPTR.clone()));
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

    fn has_loop_back_edge(g: &FunctionGraph) -> bool {
        g.blocks.iter().any(|block| {
            block.exits.iter().any(|exit| {
                g.blocks[exit.target.0]
                    .exits
                    .iter()
                    .any(|back| back.target == block.id)
            })
        })
    }

    #[test]
    fn helper_graph_copies_list_chars_into_a_str() {
        let g = build_ll_join_chars_graph();
        assert!(has_path(&g, "__len"), "list length is __len");
        assert!(
            g.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::ArrayRead {
                            array_type_id: Some(id),
                            ..
                        } if id == "[u8]"
                    )
                })
            }),
            "list byte is ArrayRead"
        );
        assert!(has_path(&g, CHR), "byte to char is builtin chr");
        assert!(
            has_path(&g, STRINGBUILDER_NEW),
            "mallocstr analogue is StringBuilder::new(n)"
        );
        assert!(has_path(&g, STRINGBUILDER_APPEND));
        assert!(has_path(&g, STRINGBUILDER_BUILD));
        assert!(has_loop_back_edge(&g), "the copy loops over chars");
        assert!(!has_path(&g, "__strlen"));
        assert!(!has_path(&g, "__string_byte_getitem"));
    }

    #[test]
    fn helper_is_registered_on_call_control() {
        let mut cc = CallControl::new();
        register_ll_join_chars_helper(&mut cc);
        let path = CallPath::from_segments([LL_JOIN_CHARS]);
        assert!(cc.has_function_graph(&path));
        register_ll_join_chars_helper(&mut cc);
        assert!(cc.has_function_graph(&path));
    }

    #[test]
    fn inverse_helper_graph_copies_str_chars_into_a_uint_list() {
        let g = build_ll_str2uintlist_graph();
        assert!(has_path(&g, "__strlen"), "string length is __strlen");
        assert!(has_path(&g, "__string_byte_getitem"), "byte is ord(s[i])");
        assert!(
            g.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if is_vec_ctor_path(segments)
                    )
                })
            }),
            "empty list is vec::Vec::new"
        );
        assert!(
            g.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if is_vec_push_path(segments)
                    )
                })
            }),
            "append is vec::Vec::push"
        );
        assert!(has_loop_back_edge(&g), "the copy loops over chars");
        assert!(!has_path(&g, "__len"));
        assert!(!g.return_is_str);
    }

    #[test]
    fn inverse_helper_is_registered_on_call_control() {
        let mut cc = CallControl::new();
        register_ll_str2uintlist_helper(&mut cc);
        let path = CallPath::from_segments([LL_STR2UINTLIST]);
        assert!(cc.has_function_graph(&path));
        register_ll_str2uintlist_helper(&mut cc);
        assert!(cc.has_function_graph(&path));
        let g = build_ll_str2uintlist_graph();
        assert!(!g.return_is_str);
    }

    fn is_vec_ctor_path(segments: &[String]) -> bool {
        segments == ["vec", "Vec", "new"]
    }

    fn is_vec_push_path(segments: &[String]) -> bool {
        segments == ["vec", "Vec", "push"]
    }
}
