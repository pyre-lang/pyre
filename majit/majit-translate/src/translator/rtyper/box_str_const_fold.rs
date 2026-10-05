//! Pre-transform fold of boxed string literals.
//!
//! String views are identity aliases in the model graph: references, pointer
//! casts, and `Wtf8::new` do not emit operations. A call that boxes such a
//! view therefore receives the string literal's `Variable` directly. Calls
//! split MIR basic blocks, so the literal definition may be in a straight-line
//! predecessor rather than in the call's block.

use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{BlockId, CallTarget, FunctionGraph, LinkArg, OpKind};

const BOX_STR_CONSTANT_PATH: [&str; 3] = [
    crate::runtime_names::crates::OBJECT,
    "unicodeobject",
    "box_str_constant",
];

fn is_box_str_constant_call(kind: &OpKind) -> Option<&LinkArg> {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = kind
    else {
        return None;
    };
    let [arg] = args.as_slice() else {
        return None;
    };
    if !segments
        .iter()
        .map(String::as_str)
        .eq(BOX_STR_CONSTANT_PATH)
    {
        return None;
    }
    Some(arg)
}

/// The bytes of a string literal, in either spelling it can have.
///
/// [`str_const_fold::fold_str_consts`](crate::translator::rtyper::str_const_fold::fold_str_consts)
/// rewrites the front's synthetic `__str_const` call to [`OpKind::ConstStr`],
/// but it runs in the codewriter — a front pass still sees the call.
pub(crate) fn str_literal_bytes(kind: &OpKind) -> Option<Vec<u8>> {
    match kind {
        OpKind::ConstStr(bytes) => Some(bytes.clone()),
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.is_empty() && segments.len() == 2 && segments[0] == "__str_const" => {
            Some(segments[1].as_bytes().to_vec())
        }
        _ => None,
    }
}

/// Resolve `value` to a string literal that dominates its use.
///
/// The walk accepts only straight-line control flow. At a block input it also
/// follows the sole incoming link, preserving the input-argument position.
/// Multiple predecessors or multiple incoming links make the value ambiguous
/// and stop the fold.
fn dominating_literal(
    graph: &FunctionGraph,
    use_block: BlockId,
    use_op_index: usize,
    value: &Variable,
) -> Option<Vec<u8>> {
    let mut block_id = use_block;
    let mut before = use_op_index;
    let mut value = value.clone();
    let mut seen = Vec::new();

    loop {
        if seen.contains(&block_id) {
            return None;
        }
        seen.push(block_id);

        let block = graph.block(block_id);
        if let Some(producer) = block.operations[..before]
            .iter()
            .rev()
            .find(|op| op.result.as_ref() == Some(&value))
        {
            return str_literal_bytes(&producer.kind);
        }

        if let Some(slot) = block.inputargs.iter().position(|arg| arg == &value) {
            let predecessors = graph.predecessors(block_id);
            let [predecessor] = predecessors.as_slice() else {
                return None;
            };
            let predecessor_block = graph.block(*predecessor);
            let mut incoming = predecessor_block
                .exits
                .iter()
                .filter(|link| link.target == block_id);
            let link = incoming.next()?;
            if incoming.next().is_some() {
                return None;
            }
            let LinkArg::Value(incoming_value) = link.args.get(slot)? else {
                return None;
            };
            value = incoming_value.clone();
            block_id = *predecessor;
            before = predecessor_block.operations.len();
            continue;
        }

        let predecessors = graph.predecessors(block_id);
        let [predecessor] = predecessors.as_slice() else {
            return None;
        };
        block_id = *predecessor;
        before = graph.block(block_id).operations.len();
    }
}

/// Replace `box_str_constant` calls over proven string literals with an
/// interned-unicode Ref constant while preserving each call's result variable.
pub fn fold_box_str_constants(graph: &mut FunctionGraph) {
    let mut rewrites = Vec::new();
    for block in &graph.blocks {
        for (op_index, op) in block.operations.iter().enumerate() {
            let Some(arg) = is_box_str_constant_call(&op.kind) else {
                continue;
            };
            let literal = match arg {
                LinkArg::Value(var) => dominating_literal(graph, block.id, op_index, var),
                LinkArg::Const(c) => match &c.value {
                    ConstValue::ByteStr(bytes) => Some(bytes.clone()),
                    ConstValue::UniStr(s) => Some(s.as_bytes().to_vec()),
                    _ => None,
                },
            };
            if let Some(bytes) = literal {
                rewrites.push((block.id, op_index, bytes));
            }
        }
    }

    for (block_id, op_index, bytes) in rewrites {
        graph.block_mut(block_id).operations[op_index].kind = OpKind::ConstInternedStr(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ValueType;

    fn str_const_call(payload: &str) -> OpKind {
        OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__str_const".to_string(), payload.to_string()],
                fun_decl_id: None,
                generic_rust_args: Vec::new(),
            },
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(None),
        }
    }

    fn box_str_constant_call(arg: Variable) -> OpKind {
        OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: BOX_STR_CONSTANT_PATH.map(str::to_string).to_vec(),
                fun_decl_id: None,
                generic_rust_args: Vec::new(),
            },
            args: crate::model::call_args(vec![arg]),
            result_ty: ValueType::Ref(None),
        }
    }

    /// A constant operand is not a `Variable`; a string constant folds
    /// directly and any other constant leaves the call alone.
    #[test]
    fn a_constant_operand_folds_without_a_producer() {
        let mut graph = FunctionGraph::new("box_const_operand");
        let entry = graph.startblock;
        let call = |value: ConstValue| OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: BOX_STR_CONSTANT_PATH.map(str::to_string).to_vec(),
                fun_decl_id: None,
                generic_rust_args: Vec::new(),
            },
            args: vec![LinkArg::from(value)],
            result_ty: ValueType::Ref(None),
        };
        graph
            .push_op_var(entry, call(ConstValue::UniStr("__len__".into())), true)
            .expect("box call must produce a value");
        graph
            .push_op_var(entry, call(ConstValue::Int(7)), true)
            .expect("box call must produce a value");

        fold_box_str_constants(&mut graph);

        assert_eq!(
            graph.block(entry).operations[0].kind,
            OpKind::ConstInternedStr(b"__len__".to_vec())
        );
        assert_eq!(
            graph.block(entry).operations[1].kind,
            call(ConstValue::Int(7))
        );
    }

    #[test]
    fn folds_frontend_literal_view_across_straight_line_blocks() {
        let mut graph = FunctionGraph::new("box_literal");
        let entry = graph.startblock;
        let literal = graph
            .push_op_var(entry, str_const_call("__instancecheck__"), true)
            .expect("string literal must produce a value");
        let call_block = graph.create_block();
        graph.set_goto(entry, call_block, vec![]);
        let boxed = graph
            .push_op_var(call_block, box_str_constant_call(literal.clone()), true)
            .expect("box call must produce a value");

        crate::translator::rtyper::str_const_fold::fold_str_consts(&mut graph);
        fold_box_str_constants(&mut graph);

        assert_eq!(
            graph.block(entry).operations[0].kind,
            OpKind::ConstStr(b"__instancecheck__".to_vec())
        );
        let folded = &graph.block(call_block).operations[0];
        assert_eq!(folded.result.as_ref(), Some(&boxed));
        assert_eq!(
            folded.kind,
            OpKind::ConstInternedStr(b"__instancecheck__".to_vec())
        );
    }

    /// The other arm of [`str_literal_bytes`]. A front pass sees the literal
    /// as an unlowered `__str_const` call, because `fold_str_consts` runs later
    /// in the codewriter — so this fold must not depend on having run it, which
    /// the test above cannot show because it runs it first.
    #[test]
    fn folds_an_unlowered_str_const_call() {
        let mut graph = FunctionGraph::new("box_literal_front");
        let entry = graph.startblock;
        let literal = graph
            .push_op_var(entry, str_const_call("__instancecheck__"), true)
            .expect("string literal must produce a value");
        let boxed = graph
            .push_op_var(entry, box_str_constant_call(literal.clone()), true)
            .expect("box call must produce a value");

        fold_box_str_constants(&mut graph);

        // Only the box call folds; its producer is left for the codewriter.
        assert_eq!(
            graph.block(entry).operations[0].kind,
            str_const_call("__instancecheck__")
        );
        let folded = &graph.block(entry).operations[1];
        assert_eq!(folded.result.as_ref(), Some(&boxed));
        assert_eq!(
            folded.kind,
            OpKind::ConstInternedStr(b"__instancecheck__".to_vec())
        );
    }

    /// A constant argument is not a literal `Variable` to trace back.
    #[test]
    fn leaves_constant_argument_call_unchanged() {
        let mut graph = FunctionGraph::new("box_constant");
        let entry = graph.startblock;
        let original = OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: BOX_STR_CONSTANT_PATH.map(str::to_string).to_vec(),
                fun_decl_id: None,
                generic_rust_args: Vec::new(),
            },
            args: vec![LinkArg::from(crate::flowspace::model::ConstValue::None)],
            result_ty: ValueType::Ref(None),
        };
        graph.push_op_var(entry, original.clone(), true);

        fold_box_str_constants(&mut graph);

        assert_eq!(graph.block(entry).operations[0].kind, original);
    }

    #[test]
    fn leaves_dynamic_argument_call_unchanged() {
        let mut graph = FunctionGraph::new("box_dynamic");
        let entry = graph.startblock;
        let input = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "value".to_string(),
                    ty: ValueType::Str,
                    class_root: None,
                },
                true,
            )
            .expect("input must produce a value");
        let original = box_str_constant_call(input);
        graph.push_op_var(entry, original.clone(), true);

        fold_box_str_constants(&mut graph);

        assert_eq!(graph.block(entry).operations[1].kind, original);
    }

    /// `dunder_overridden`'s caller boxes the literal, then passes the ref.
    #[test]
    fn folds_dunder_overridden_caller_literal() {
        let mut graph = FunctionGraph::new("dunder_overridden_caller");
        let entry = graph.startblock;
        let literal = graph
            .push_op_var(entry, str_const_call("__add__"), true)
            .expect("string literal must produce a value");
        let boxed = graph
            .push_op_var(entry, box_str_constant_call(literal), true)
            .expect("box call must produce a value");

        fold_box_str_constants(&mut graph);

        assert_eq!(
            graph.block(entry).operations[1].kind,
            OpKind::ConstInternedStr(b"__add__".to_vec())
        );
        assert_eq!(
            graph.block(entry).operations[1].result.as_ref(),
            Some(&boxed)
        );
    }

    /// `try_reflected_binary_special`'s caller boxes the reflected literal.
    #[test]
    fn folds_reflected_binary_special_caller_literal() {
        let mut graph = FunctionGraph::new("reflected_binary_special_caller");
        let entry = graph.startblock;
        let literal = graph
            .push_op_var(entry, str_const_call("__radd__"), true)
            .expect("string literal must produce a value");
        let boxed = graph
            .push_op_var(entry, box_str_constant_call(literal), true)
            .expect("box call must produce a value");

        fold_box_str_constants(&mut graph);

        assert_eq!(
            graph.block(entry).operations[1].kind,
            OpKind::ConstInternedStr(b"__radd__".to_vec())
        );
        assert_eq!(
            graph.block(entry).operations[1].result.as_ref(),
            Some(&boxed)
        );
    }

    /// `try_lookup_unaryop` is reached from a per-arm literal. Each arm has
    /// one predecessor, so both boxes fold.
    #[test]
    fn folds_lookup_unaryop_per_arm_literals() {
        let mut graph = FunctionGraph::new("lookup_unaryop_arms");
        let entry = graph.startblock;
        let cond = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "neg".to_string(),
                    ty: ValueType::Bool,
                    class_root: None,
                },
                true,
            )
            .expect("cond must produce a value");
        let neg_arm = graph.create_block();
        let pos_arm = graph.create_block();
        graph.set_branch(entry, cond.clone(), neg_arm, vec![], pos_arm, vec![]);

        let neg_lit = graph
            .push_op_var(neg_arm, str_const_call("__neg__"), true)
            .expect("neg literal");
        let neg_box = graph
            .push_op_var(neg_arm, box_str_constant_call(neg_lit), true)
            .expect("neg box");
        let pos_lit = graph
            .push_op_var(pos_arm, str_const_call("__pos__"), true)
            .expect("pos literal");
        let pos_box = graph
            .push_op_var(pos_arm, box_str_constant_call(pos_lit), true)
            .expect("pos box");

        fold_box_str_constants(&mut graph);

        assert_eq!(
            graph.block(neg_arm).operations[1].kind,
            OpKind::ConstInternedStr(b"__neg__".to_vec())
        );
        assert_eq!(
            graph.block(neg_arm).operations[1].result.as_ref(),
            Some(&neg_box)
        );
        assert_eq!(
            graph.block(pos_arm).operations[1].kind,
            OpKind::ConstInternedStr(b"__pos__".to_vec())
        );
        assert_eq!(
            graph.block(pos_arm).operations[1].result.as_ref(),
            Some(&pos_box)
        );
    }
}
