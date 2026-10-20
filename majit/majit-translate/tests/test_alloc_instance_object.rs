//! Production-graph census for the two `rclass.new_instance` graphs.
//!
//! Exact `object()` mallocs `W_ObjectObject` with constant `INSTANCE_TYPE`;
//! `_getusercls` mallocs `W_ObjectObjectUser` with constant
//! `INSTANCE_USER_TYPE`. `fuse_boxing_alloc` turns each `malloc_typed_managed`
//! cluster into `new_with_vtable`, and `jtransform.py rewrite_op_setfield`
//! / `is_typeptr_getset` drops the typeptr store.

mod common;

use common::{OBJECT_LLBC, load_llbc, lower_named_with_static_addrs};
use majit_translate::HostStaticAddrs;
use majit_translate::codewriter::jtransform::{GraphTransformConfig, Transformer};
use majit_translate::model::{CallTarget, FunctionGraph, OpKind};

fn op_brief(op: &majit_translate::model::SpaceOperation) -> String {
    let result = op
        .result
        .as_ref()
        .map(|v| format!("v{}", v.id()))
        .unwrap_or_else(|| "_".into());
    match &op.kind {
        OpKind::New { owner } => format!("{result} = New {owner}"),
        OpKind::NewWithVtable { owner, vtable } => {
            format!("{result} = NewWithVtable {owner} vtable={vtable:#x}")
        }
        OpKind::FieldWrite {
            base, field, value, ..
        } => {
            let val = match value {
                majit_translate::model::LinkArg::Value(v) => format!("v{}", v.id()),
                other => format!("{other:?}"),
            };
            format!("setfield {} base=v{} val={val}", field.name, base.id(),)
        }
        OpKind::FieldRead { base, field, .. } => {
            format!("getfield {} base=v{} -> {result}", field.name, base.id())
        }
        OpKind::Call { target, args, .. } => {
            let name = match target {
                CallTarget::FunctionPath { segments, .. } => segments.join("::"),
                CallTarget::SyntheticTransparentCtor { name, .. } => format!("ctor {name}"),
                other => format!("{other:?}"),
            };
            let args = args
                .iter()
                .map(|a| match a {
                    majit_translate::model::LinkArg::Value(v) => format!("v{}", v.id()),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{result} = Call {name}({args})")
        }
        other => format!("{result} = {other:?}"),
    }
}

fn summarize(label: &str, graph: &FunctionGraph) -> String {
    let mut lines = vec![format!(
        "=== {label} {} blocks={} returnblock={} inputs={:?} ===",
        graph.name,
        graph.blocks.len(),
        graph.returnblock.0,
        graph
            .block(graph.returnblock)
            .inputargs
            .iter()
            .map(|v| format!("v{}", v.id()))
            .collect::<Vec<_>>()
    )];
    for (bi, block) in graph.blocks.iter().enumerate() {
        let has_alloc = block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::New { .. }
                    | OpKind::NewWithVtable { .. }
                    | OpKind::FieldWrite { .. }
                    | OpKind::Call { .. }
            )
        });
        if !has_alloc {
            continue;
        }
        lines.push(format!(
            "-- block {bi} inputs={:?} exits={:?}",
            block
                .inputargs
                .iter()
                .map(|v| format!("v{}", v.id()))
                .collect::<Vec<_>>(),
            block
                .exits
                .iter()
                .map(|l| format!("->{} args={:?}", l.target.0, l.args.len()))
                .collect::<Vec<_>>()
        ));
        for (oi, op) in block.operations.iter().enumerate() {
            lines.push(format!("  {bi}:{oi} {}", op_brief(op)));
        }
        for (ei, link) in block.exits.iter().enumerate() {
            let args = link
                .args
                .iter()
                .map(|a| match a {
                    majit_translate::model::LinkArg::Value(v) => format!("v{}", v.id()),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!(
                "  exit[{ei}] ->{} ({args}) case={:?}",
                link.target.0, link.exitcase
            ));
        }
        if let Some(ret) = &block.exits.first().and_then(|e| e.exitcase.as_ref()) {
            lines.push(format!("  exitcase {ret:?}"));
        }
    }
    lines.join("\n")
}

fn has_new_with_vtable(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(op.kind, OpKind::NewWithVtable { .. }))
    })
}

fn has_ob_type_store(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. } if field.name == "ob_type"
            )
        })
    })
}

fn has_nursery_raw(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("try_gc_alloc_nursery_raw")
            )
        })
    })
}

fn has_ptr_write(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("write")
                    && segments.iter().any(|s| s == "ptr")
            )
        })
    })
}

fn transform_named(leaf: &str) -> (FunctionGraph, FunctionGraph) {
    let llbc = load_llbc(OBJECT_LLBC);
    let pytypes = [
        ("pyobject::INSTANCE_TYPE", 0x1000_0000i64),
        ("pyobject::INSTANCE_USER_TYPE", 0x1000_1000i64),
    ];
    let graph = lower_named_with_static_addrs(
        llbc,
        leaf,
        HostStaticAddrs {
            pytypes: &pytypes,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("lower {leaf}: {e:?}"));
    let transformed = Transformer::new(&GraphTransformConfig::default()).transform(&graph);
    (graph, transformed.graph)
}

fn assert_new_with_vtable_and_no_typeptr_store(leaf: &str) {
    let (lowered, transformed) = transform_named(leaf);
    let dump = format!(
        "{}\n{}",
        summarize("lowered", &lowered),
        summarize("jtransform", &transformed)
    );
    assert!(
        !has_ptr_write(&transformed),
        "generic core::ptr::write must not survive as a shared callee\n{dump}"
    );
    assert!(
        has_new_with_vtable(&transformed),
        "{leaf} must lower to NewWithVtable\n{dump}"
    );
    assert!(
        !has_ob_type_store(&transformed),
        "jtransform.py rewrite_op_setfield drops typeptr on {leaf}\n{dump}"
    );
    assert!(
        !has_nursery_raw(&transformed),
        "{leaf} must not residualise try_gc_alloc_nursery_raw\n{dump}"
    );
}

#[test]
fn alloc_exact_object_instance_is_new_with_vtable() {
    assert_new_with_vtable_and_no_typeptr_store("alloc_exact_object_instance");
}

#[test]
fn alloc_user_object_instance_is_new_with_vtable() {
    assert_new_with_vtable_and_no_typeptr_store("alloc_user_object_instance");
}
