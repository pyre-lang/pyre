//! `KwargsDictStrategy::length` is `kwargs_storage(w_dict).0.len()`.
//!
//! A one-word-item `Vec` field is the three-word header in the parent
//! (`rlist.py` LIST `("length", Signed)`). The `.len()` must be a
//! `VecFieldPart::Len` getfield of that parent, not `ll_vec_length` of
//! the first header word (the data pointer) plus `raw_load` at +16.

use std::sync::OnceLock;

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, FunctionGraph, OpKind, VecFieldPart};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);

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

fn is_ll_vec_length(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments.last().is_some_and(|leaf| {
            leaf == "ll_vec_length_r" || leaf == "ll_vec_length_i" || leaf == "ll_vec_length_f"
        }),
        _ => false,
    }
}

#[test]
fn kwargs_length_reads_the_inline_len_word() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "kwargsdict::<Impl>::length")
        .expect("lower kwargsdict::<Impl>::length");
    if let Ok(path) = std::env::var("DUMP_KWARGS_LENGTH") {
        std::fs::write(&path, graph.dump()).expect("write kwargs length dump");
    }
    assert_inline_len_of_pos0(&graph);
}

/// `kwargs_getitem_correct_indirection` is `for i in 0..keys_w.len()`.
/// The Len retarget must keep the whole-field `__pos_0` read: that field
/// also feeds the index (`retarget_fat_operand` keeps a shared fat
/// field whole for the same reason).
#[test]
fn kwargs_getitem_keeps_the_whole_field_and_reads_len() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "_orig_kwargs_getitem_correct_indirection")
        .expect("lower _orig_kwargs_getitem_correct_indirection");
    if let Ok(path) = std::env::var("DUMP_KWARGS_GETITEM") {
        std::fs::write(&path, graph.dump()).expect("write kwargs getitem dump");
    }
    assert_inline_len_of_pos0(&graph);
    assert_whole_field_of_pos0(&graph);
    assert_defined_link_args(&graph);
}

fn assert_inline_len_of_pos0(graph: &FunctionGraph) {
    let ops = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter());
    let mut len_reads = 0usize;
    let mut ll_vec_length = 0usize;
    for op in ops {
        match &op.kind {
            OpKind::FieldRead { field, .. }
                if field.vec_part == Some(VecFieldPart::Len)
                    && (field.name == "__pos_0" || field.name == "0") =>
            {
                len_reads += 1;
            }
            OpKind::Call { target, .. } if is_ll_vec_length(target) => {
                ll_vec_length += 1;
            }
            _ => {}
        }
    }
    assert!(
        len_reads > 0,
        "kwargs length must read the parent+len word (`rlist.ll_length`)\n{}",
        graph.dump()
    );
    assert_eq!(
        ll_vec_length,
        0,
        "kwargs length must not call ll_vec_length of the first Vec word\n{}",
        graph.dump()
    );
}

fn assert_whole_field_of_pos0(graph: &FunctionGraph) {
    let kept = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if (field.name == "__pos_0" || field.name == "0")
                        && field.vec_part.is_none()
            )
        });
    assert!(
        kept,
        "getitem must keep the whole-field `__pos_0` read for later index/gcarray\n{}",
        graph.dump()
    );
}

fn assert_defined_link_args(graph: &FunctionGraph) {
    use majit_translate::model::LinkArg;
    let mut defined = std::collections::HashSet::new();
    for block in &graph.blocks {
        for input in &block.inputargs {
            defined.insert(input.id());
        }
        for op in &block.operations {
            if let Some(result) = &op.result {
                defined.insert(result.id());
            }
        }
    }
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (ei, link) in block.exits.iter().enumerate() {
            for (ai, arg) in link.args.iter().enumerate() {
                let LinkArg::Value(var) = arg else {
                    continue;
                };
                assert!(
                    defined.contains(&var.id()),
                    "undefined link arg block {bi} exit {ei} slot {ai}: {var:?}\n{}",
                    graph.dump()
                );
            }
        }
    }
}
