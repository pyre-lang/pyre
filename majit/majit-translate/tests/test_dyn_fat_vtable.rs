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
    // The IndirectCall receiver is the fat DATA word (`keep_dyn_call_receiver`),
    // not a cast of the vtable word. Unused `cast_int_to_ptr` of FatLen
    // is dropped by `drop_unused_fat_len_casts`.
}

fn is_cast_int_to_ptr(kind: &OpKind) -> bool {
    match kind {
        OpKind::UnaryOp { op, .. } if op == "cast_int_to_ptr" => true,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } if segments.last().map(|s| s.as_str()) == Some("cast_int_to_ptr") => true,
        _ => false,
    }
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

fn has_cast_int_to_ptr(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| is_cast_int_to_ptr(&op.kind))
    })
}

#[test]
fn w_dict_store_has_no_cast_int_to_ptr_of_vtable_word() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "w_dict_store").expect("lower w_dict_store");
    assert_vtable_reads(&graph, "method_setitem");
    assert!(
        !has_cast_int_to_ptr(&graph),
        "leftover FatVtable users are the DATA word (Ref), not CastIntToPtr of FatMeta\n{}",
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

fn value_type_kind(ty: &ValueType) -> Option<char> {
    match ty {
        ValueType::Int
        | ValueType::Unsigned
        | ValueType::Int128
        | ValueType::UInt128
        | ValueType::SingleFloat
        | ValueType::Bool => Some('i'),
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => Some('r'),
        ValueType::Float => Some('f'),
        ValueType::Void | ValueType::State | ValueType::Unknown => None,
    }
}

fn concrete_kind(ty: ConcreteType) -> Option<char> {
    match ty {
        ConcreteType::Signed => Some('i'),
        ConcreteType::GcRef => Some('r'),
        ConcreteType::Float => Some('f'),
        ConcreteType::Void => None,
        ConcreteType::Unknown => None,
    }
}

fn var_kind(graph: &FunctionGraph, block: BlockId, var: &Variable) -> Option<char> {
    var_kind_from(graph, block, var, &mut HashSet::new())
}

fn var_kind_from(
    graph: &FunctionGraph,
    block: BlockId,
    var: &Variable,
    seen: &mut HashSet<(usize, u64)>,
) -> Option<char> {
    if !seen.insert((block.0, var.id())) {
        return None;
    }
    if let Some(kind) = concrete_kind(FunctionGraph::concretetype_of(var)) {
        return Some(kind);
    }
    if let Some((def_block, op)) = producer(graph, var.id()) {
        return match &op.kind {
            OpKind::FieldRead { ty, field, .. } => {
                if field.vec_part == Some(VecFieldPart::FatData) && *ty == ValueType::Int {
                    Some('i')
                } else {
                    value_type_kind(ty)
                }
            }
            OpKind::ConstInt(_) | OpKind::ConstUInt(_) => Some('i'),
            OpKind::ConstRefAddr(_) => Some('r'),
            OpKind::Input { ty, .. } => value_type_kind(ty),
            OpKind::UnaryOp { operand, .. } => var_kind_from(graph, def_block, operand, seen),
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().map(|s| s.as_str()) == Some("cast_int_to_ptr")
                || segments.last().map(|s| s.as_str()) == Some("same_as") =>
            {
                args.first()
                    .and_then(|arg| arg.as_variable())
                    .and_then(|src| var_kind_from(graph, def_block, src, seen))
            }
            _ => None,
        };
    }
    let block_ref = graph.blocks.get(block.0)?;
    let index = block_ref.inputargs.iter().position(|arg| arg == var)?;
    let mut kinds = Vec::new();
    for pred in graph.predecessors(block) {
        let pred_block = graph.blocks.get(pred.0)?;
        for link in pred_block.exits.iter().filter(|link| link.target == block) {
            let src = link.args.get(index)?.as_variable()?;
            kinds.push(var_kind_from(graph, pred, src, seen)?);
        }
    }
    let first = *kinds.first()?;
    kinds.iter().all(|kind| *kind == first).then_some(first)
}

fn first_input_kind(graph: &FunctionGraph) -> Option<(char, ValueType)> {
    let block = graph.blocks.get(graph.startblock.0)?;
    for op in &block.operations {
        let OpKind::Input { ty, .. } = &op.kind else {
            continue;
        };
        if matches!(ty, ValueType::Void) {
            continue;
        }
        let kind = value_type_kind(ty).or_else(|| {
            op.result
                .as_ref()
                .and_then(|var| var_kind(graph, graph.startblock, var))
        })?;
        return Some((kind, ty.clone()));
    }
    None
}

fn first_indirect_call<'a>(
    graph: &'a FunctionGraph,
) -> Option<(BlockId, &'a [Variable], Option<&'a (String, String)>)> {
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::IndirectCall {
                args, family_key, ..
            } = &op.kind
            {
                return Some((block.id, args.as_slice(), family_key.as_ref()));
            }
        }
    }
    None
}

/// Dyn `DictStrategy::length` passes the fat DATA word as Ref
/// (RPython `DictStrategy` instance). The ZST impl keeps that same
/// first inputarg so the `indirect_call` family is one FUNC type.
#[test]
fn w_dict_len_indirect_call_receiver_kind_matches_zst_length_self() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let caller = lower_function(llbc, "w_dict_len").expect("lower w_dict_len");
    let callee = lower_function(llbc, "kwargsdict::<Impl>::length")
        .expect("lower kwargsdict::<Impl>::length");
    if let Ok(path) = std::env::var("DUMP_W_DICT_LEN") {
        std::fs::write(format!("{path}.caller"), caller.dump()).expect("write w_dict_len dump");
        std::fs::write(format!("{path}.callee"), callee.dump())
            .expect("write kwargsdict length dump");
    }
    let (call_block, call_args, family) = first_indirect_call(&caller).unwrap_or_else(|| {
        panic!(
            "w_dict_len must emit IndirectCall for DictStrategy::length\n{}",
            caller.dump()
        )
    });
    assert!(
        !has_cast_int_to_ptr(&caller),
        "leftover FatVtable users are the DATA word (Ref), not CastIntToPtr of FatMeta\n{}",
        caller.dump()
    );
    assert!(
        family.is_some_and(|(trait_root, method)| {
            trait_root.contains("DictStrategy") && method == "length"
        }),
        "IndirectCall family must be DictStrategy::length, got {family:?}\n{}",
        caller.dump()
    );
    let recv = call_args
        .first()
        .unwrap_or_else(|| panic!("dyn length call must pass a receiver\n{}", caller.dump()));
    assert_ne!(
        FunctionGraph::concretetype_of(recv),
        ConcreteType::Void,
        "dyn call receiver must not be Void\n{}",
        caller.dump()
    );
    let recv_kind = var_kind(&caller, call_block, recv).unwrap_or_else(|| {
        panic!(
            "dyn call receiver has no kind (var={recv:?})\n{}",
            caller.dump()
        )
    });
    let (callee_kind, callee_ty) = first_input_kind(&callee).unwrap_or_else(|| {
        panic!(
            "kwargsdict::<Impl>::length must keep a non-Void self input\n{}",
            callee.dump()
        )
    });
    assert_eq!(
        recv_kind,
        callee_kind,
        "caller receiver kind {recv_kind} != callee first inputarg kind {callee_kind} ({callee_ty:?})\ncaller:\n{}\ncallee:\n{}",
        caller.dump(),
        callee.dump()
    );
    assert_eq!(
        recv_kind, 'r',
        "DictStrategy self is a GC instance (Ref); mixed ZST+GC family follows that kind, got {recv_kind}"
    );

    let caller_segments: Vec<String> = caller.name.split("::").map(str::to_string).collect();
    let callee_segments: Vec<String> = callee.name.split("::").map(str::to_string).collect();
    let caller_path = majit_translate::CallPath::from_segments(caller_segments);
    let callee_path = majit_translate::CallPath::from_segments(callee_segments);
    let mut callcontrol = majit_translate::codewriter::call::CallControl::new();
    callcontrol.register_function_graph(caller_path.clone(), caller.clone());
    callcontrol.register_function_graph(callee_path.clone(), callee.clone());
    let cfg = majit_translate::GraphTransformConfig::default();
    let mut codewriter = majit_translate::codewriter::codewriter::CodeWriter::new();
    let caller_jit = std::sync::Arc::new(majit_translate::codewriter::jitcode::JitCode::new(
        "w_dict_len",
    ));
    let callee_jit = std::sync::Arc::new(majit_translate::codewriter::jitcode::JitCode::new(
        "kwargsdict_length",
    ));
    codewriter.transform_graph_to_jitcode(
        &caller,
        &caller_path,
        &mut callcontrol,
        &cfg,
        &caller_jit,
        false,
        0,
    );
    codewriter.transform_graph_to_jitcode(
        &callee,
        &callee_path,
        &mut callcontrol,
        &cfg,
        &callee_jit,
        false,
        0,
    );
    let callee_args = &callee_jit.calldescr.arg_classes;
    assert!(
        callee_args.starts_with('r'),
        "ZST length impl arg_classes must start with r (DictStrategy instance self), got {callee_args:?}\n{}",
        callee.dump()
    );
    if let Ok(path) = std::env::var("DUMP_W_DICT_LEN") {
        std::fs::write(
            format!("{path}.arg_classes"),
            format!(
                "w_dict_len recv_kind={recv_kind}\nkwargsdict::<Impl>::length arg_classes={callee_args}\n"
            ),
        )
        .expect("write arg_classes dump");
    }
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
