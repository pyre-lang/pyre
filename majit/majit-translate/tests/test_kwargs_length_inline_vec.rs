//! `KwargsDictStrategy::length` is `kwargs_storage(w_dict).0.len()`.
//!
//! A one-word-item `Vec` field is the three-word header in the parent
//! (`rlist.py` LIST `("length", Signed)`). The `.len()` must be a
//! `VecFieldPart::Len` getfield of that parent, not `ll_vec_length` of
//! the first header word (the data pointer) plus `raw_load` at +16.

use std::sync::OnceLock;

use majit_charon_reader::Llbc;
use majit_translate::HostStaticAddrs;
use majit_translate::front::mir::{
    build_semantic_program_from_llbcs_with_static_addrs_and_function_names, lower_function,
};
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

/// `KwargsDictStorage::default` is `(Vec::new(), Vec::new())`.
/// `core::tuple::<Impl>::default` is not in CallRegistry; the front
/// end emits `TupleRepr.newtuple`: a shaped Tuple constructor plus
/// one setfield per empty `vec::Vec::new`.
#[test]
fn kwargs_get_empty_storage_is_newtuple_of_empty_vecs() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "kwargsdict::<Impl>::get_empty_storage")
        .expect("lower kwargsdict::<Impl>::get_empty_storage");
    if let Ok(path) = std::env::var("DUMP_KWARGS_GET_EMPTY_STORAGE") {
        std::fs::write(&path, graph.dump()).expect("write kwargs get_empty_storage dump");
    }
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    let tuple_default_calls = ops
        .iter()
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target, .. } if is_core_tuple_default_call(target)
            )
        })
        .count();
    let newtuples = ops
        .iter()
        .filter(|op| matches!(&op.kind, OpKind::NewTuple { .. }))
        .count();
    let tuple_ctors = ops
        .iter()
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, .. },
                    ..
                } if majit_ir::descr::is_shaped_tuple_name(name)
            )
        })
        .count();
    let pos_writes = ops
        .iter()
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. }
                    if field.name == "__pos_0" || field.name == "__pos_1"
            )
        })
        .count();
    let vec_news = ops
        .iter()
        .filter(|op| matches!(&op.kind, OpKind::Call { target, .. } if is_vec_new_call(target)))
        .count();
    assert_eq!(
        tuple_default_calls,
        0,
        "tuple Default must not remain a residual CallRegistry miss\n{}",
        graph.dump()
    );
    assert_eq!(
        newtuples,
        0,
        "raw NewTuple must not survive; TupleRepr.newtuple is ctor + setfield\n{}",
        graph.dump()
    );
    assert!(
        tuple_ctors > 0,
        "empty kwargs storage is a shaped Tuple constructor\n{}",
        graph.dump()
    );
    assert!(
        pos_writes >= 2,
        "each tuple item is a setfield of the empty vec\n{}",
        graph.dump()
    );
    assert!(
        vec_news >= 2,
        "each tuple element default is the RustVecRepr NewEmpty helper\n{}",
        graph.dump()
    );
}

/// `gc_alloc_young_storage_box` is one `dont_look_inside` body
/// (`monomorphize: false`). Each concrete storage shape must call its
/// own residual funcptr (`annlowlevel.specialize` / `getfunctionptr`,
/// an `__inst_` suffix); kwargs Tuple and typed RDict must not share
/// the unspecialized path.
#[test]
fn gc_alloc_young_storage_box_gets_one_residual_per_storage_shape() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &[
            "kwargsdict",
            "dictmultiobject",
            "gc_storage",
            "identitydict",
        ],
        &["get_empty_storage"],
    )
    .expect("lower get_empty_storage callers");

    let mut inst_leaves = std::collections::BTreeSet::new();
    let mut unspecialized = 0usize;
    for function in &program.functions {
        if function.name.rsplit("::").next() != Some("get_empty_storage") {
            continue;
        }
        for op in function
            .graph()
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
        {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } = &op.kind
            else {
                continue;
            };
            let Some(leaf) = segments.last() else {
                continue;
            };
            if leaf == "gc_alloc_young_storage_box" {
                unspecialized += 1;
            } else if leaf.starts_with("gc_alloc_young_storage_box__inst_") {
                inst_leaves.insert(leaf.clone());
            }
        }
    }
    assert_eq!(
        unspecialized, 0,
        "get_empty_storage must not call the unspecialized residual; inst leaves={inst_leaves:?}"
    );
    assert!(
        inst_leaves.len() >= 2,
        "kwargs Tuple and typed RDict must be distinct residual funcptrs, got {inst_leaves:?}"
    );
}

fn is_core_tuple_default_call(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            segments.join("::") == "core::tuple::<Impl>::default"
        }
        _ => false,
    }
}

/// The `RustVecRepr` `NewEmpty` helper an ordinary `Vec::new()` of
/// one-word items lowers to (`rust_vec_std_call`), not the `newlist`
/// spelling `vec::Vec::new`.
fn is_vec_new_call(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => matches!(
            majit_ir::rvec::vec_helper_for_path(&segments.join("::")),
            Some((majit_ir::rvec::VecOp::NewEmpty, _))
        ),
        _ => false,
    }
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
