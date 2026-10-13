//! `w_dict_lookup`'s `DictStrategy::getitem` reads `method_getitem` from
//! the vtable word of `DictStrategyRef.imp`, not from the data pointer.

use std::collections::HashSet;
use std::sync::OnceLock;

use majit_charon_reader::Llbc;
use majit_translate::flowspace::model::Variable;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{
    BlockId, CallTarget, ConcreteType, FunctionGraph, OpKind, SpaceOperation, ValueType,
    VecFieldPart,
};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);

/// `None` means the artefact is absent. A checkout that has not extracted
/// LLBC skips these tests instead of panicking on the missing file.
fn object_llbc() -> Option<&'static Llbc> {
    static LLBC: OnceLock<Option<Llbc>> = OnceLock::new();
    LLBC.get_or_init(|| {
        if !std::path::Path::new(OBJECT_LLBC).is_file() {
            eprintln!("skipping: {OBJECT_LLBC} is missing; run `python3 scripts/extract-llbc.py`");
            return None;
        }
        Some(Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc"))
    })
    .as_ref()
}

fn assert_vtable_reads(graph: &FunctionGraph, method_name: &str) {
    let mut reads = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::FieldRead { base, field, .. } = &op.kind else {
                continue;
            };
            if field.name != method_name {
                continue;
            }
            reads += 1;
            let mut seen = HashSet::new();
            assert!(
                reaches_imp_fat_len(graph, block.id, base, &mut seen),
                "{method_name} base does not reach FatLen of imp\n{}",
                graph.dump()
            );
            assert_eq!(
                FunctionGraph::concretetype_of(base),
                ConcreteType::Signed,
                "vtable word must be an int so getfield_raw_r loads it"
            );
        }
    }
    assert!(reads > 0, "lowered no {method_name} read\n{}", graph.dump());
    let still_calls_strategy = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            let OpKind::Call { target, .. } = &op.kind else {
                return false;
            };
            call_names_get_strategy(target)
        })
    });
    assert!(
        !still_calls_strategy,
        "w_dict_get_strategy stayed a call, so its vtable word was dropped\n{}",
        graph.dump()
    );
    let fat_vtable_left = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::FieldRead { field, .. } => {
                field.name == "imp" && field.vec_part == Some(VecFieldPart::FatVtable)
            }
            _ => false,
        })
    });
    assert!(
        !fat_vtable_left,
        "imp.vtable GetfieldGcR next to imp.len GetfieldGcI trips make_equal_to Box.type\n{}",
        graph.dump()
    );
}

#[test]
fn w_dict_lookup_reads_getitem_from_the_vtable_word() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "w_dict_lookup").expect("lower w_dict_lookup");
    assert_vtable_reads(&graph, "method_getitem");
}

fn call_names_get_strategy(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments
            .iter()
            .any(|segment| segment == "w_dict_get_strategy"),
        CallTarget::Method { name, .. } => name == "w_dict_get_strategy",
        _ => false,
    }
}

fn reaches_imp_fat_len(
    graph: &FunctionGraph,
    block: BlockId,
    var: &Variable,
    seen: &mut HashSet<(usize, u64)>,
) -> bool {
    if !seen.insert((block.0, var.id())) {
        return false;
    }
    if let Some((def_block, op)) = producer(graph, var.id()) {
        return match &op.kind {
            OpKind::FieldRead { field, ty, .. }
                if field.name == "imp"
                    && matches!(
                        field.vec_part,
                        Some(VecFieldPart::FatLen | VecFieldPart::FatMeta)
                    )
                    && *ty == ValueType::Int
                    && FunctionGraph::concretetype_of(var) == ConcreteType::Signed =>
            {
                true
            }
            OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                reaches_imp_fat_len(graph, def_block, operand, seen)
            }
            _ => false,
        };
    }
    let Some(block_ref) = graph.blocks.get(block.0) else {
        return false;
    };
    let preds = graph.predecessors(block);
    if preds.is_empty() {
        return false;
    }
    let index = block_ref.inputargs.iter().position(|arg| arg == var);
    preds.iter().all(|pred| {
        let Some(pred_block) = graph.blocks.get(pred.0) else {
            return false;
        };
        let incoming: Vec<_> = pred_block
            .exits
            .iter()
            .filter(|link| link.target == block)
            .collect();
        if incoming.is_empty() {
            return false;
        }
        incoming.iter().all(|link| {
            let src = if let Some(index) = index {
                link.args.get(index).and_then(|arg| arg.as_variable())
            } else {
                // Dominating use: the method_* block reads the metadata
                // word without taking it as an inputarg.
                Some(var)
            };
            src.is_some_and(|src| reaches_imp_fat_len(graph, *pred, src, seen))
        })
    })
}

#[test]
fn w_dict_getitem_str_hashed_assembles_with_vtable_word() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph =
        lower_function(llbc, "w_dict_getitem_str_hashed").expect("lower w_dict_getitem_str_hashed");
    assert_vtable_reads(&graph, "method_getitem_str_hashed");
    // `join_blocks` renames the indirect call's `hash_` onto the link arg.
    // That arg has to stay defined after the fat-dyn splice, or liveness
    // panics while assembling this graph.
    let segments: Vec<String> = graph.name.split("::").map(str::to_string).collect();
    let path = majit_translate::CallPath::from_segments(segments);
    let mut callcontrol = majit_translate::codewriter::call::CallControl::new();
    callcontrol.register_function_graph(path.clone(), graph.clone());
    let mut codewriter = majit_translate::codewriter::codewriter::CodeWriter::new();
    let jitcode = std::sync::Arc::new(majit_translate::codewriter::jitcode::JitCode::new(
        "w_dict_getitem_str_hashed",
    ));
    let cfg = majit_translate::GraphTransformConfig::default();
    codewriter.transform_graph_to_jitcode(
        &graph,
        &path,
        &mut callcontrol,
        &cfg,
        &jitcode,
        false,
        0,
    );
}

fn leftover_cast_int_to_ptr_unary(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::UnaryOp { op, .. } if op == "cast_int_to_ptr"
            )
        })
    })
}

fn is_lltype_cast_int_to_ptr_call(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            segments.iter().any(|segment| segment == "lltype")
                && segments
                    .last()
                    .is_some_and(|leaf| leaf == "cast_int_to_ptr")
        }
        _ => false,
    }
}

#[test]
fn w_dict_store_cast_int_to_ptr_is_the_lltype_host_call() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "w_dict_store").expect("lower w_dict_store");
    assert_vtable_reads(&graph, "method_setitem");
    assert!(
        !leftover_cast_int_to_ptr_unary(&graph),
        "w_dict_store must not emit UnaryOp cast_int_to_ptr of the FatMeta Int load\n{}",
        graph.dump()
    );
    assert!(
        graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target, .. } if is_lltype_cast_int_to_ptr_call(target)
                )
            })
        }),
        "w_dict_store must emit lltype.cast_int_to_ptr of the FatMeta Int load\n{}",
        graph.dump()
    );
}

fn call_leaf_is(target: &CallTarget, leaf: &str) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().is_some_and(|name| name == leaf)
        }
        _ => false,
    }
}

#[test]
fn eq_w_compares_utf8_fields_with_ll_streq() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "eq_w").expect("lower eq_w");
    assert!(
        graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Call { target, .. } if call_leaf_is(target, "jit_ll_streq"))
            })
        }),
        "eq_w must call ll_streq on the two `_utf8` STRs \
         (unicodeobject.py eq_w: self._utf8 == w_other._utf8)\n{}",
        graph.dump()
    );
    let residual_payload = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            let OpKind::Call { target, .. } = &op.kind else {
                return false;
            };
            call_leaf_is(target, "utf8_payload_bytes")
        })
    });
    assert!(
        !residual_payload,
        "eq_w must not go through utf8_payload_bytes; the compare is ll_streq \
         of the two value fields\n{}",
        graph.dump()
    );
}

fn producer(graph: &FunctionGraph, var_id: u64) -> Option<(BlockId, &SpaceOperation)> {
    graph.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find(|op| {
                op.result
                    .as_ref()
                    .is_some_and(|result| result.id() == var_id)
            })
            .map(|op| (block.id, op))
    })
}
