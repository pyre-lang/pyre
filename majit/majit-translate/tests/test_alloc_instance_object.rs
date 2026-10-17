//! Production-graph census for `alloc_instance_object`.
//!
//! `try_gc_alloc_nursery_raw` + `ptr::write` is `lltype.malloc(STRUCT)`.
//! `jtransform.py rewrite_op_malloc` turns that into `new_with_vtable` when
//! the STRUCT has one vtable. This graph does not: `W_ObjectObject` is one
//! STRUCT with two vtables (`INSTANCE_TYPE` tid 53 vs `INSTANCE_USER_TYPE`
//! tid 217), `fuse_boxing_alloc` does not match `try_gc_alloc_nursery_raw`,
//! and `instance_typeptr_for` is not a constant. Until `_getusercls` is its
//! own interp class (`W_ObjectObjectUserDictWeakrefable`), the residual bump
//! must keep the typeptr store.

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

#[test]
fn alloc_instance_object_stamps_typeptr_after_jtransform() {
    let llbc = load_llbc(OBJECT_LLBC);
    let pytypes = [
        ("pyobject::INSTANCE_TYPE", 0x1000_0000i64),
        ("pyobject::INSTANCE_USER_TYPE", 0x1000_1000i64),
    ];
    let graph = lower_named_with_static_addrs(
        llbc,
        "alloc_instance_object",
        HostStaticAddrs {
            pytypes: &pytypes,
            ..Default::default()
        },
    )
    .expect("lower alloc_instance_object");
    eprintln!("{}", summarize("lowered", &graph));
    let transformed = Transformer::new(&GraphTransformConfig::default()).transform(&graph);
    eprintln!("{}", summarize("jtransform", &transformed.graph));

    let fused = has_new_with_vtable(&transformed.graph);
    let residual = has_nursery_raw(&transformed.graph);
    let write = has_ptr_write(&transformed.graph);
    let dump = format!(
        "{}\n{}",
        summarize("lowered", &graph),
        summarize("jtransform", &transformed.graph)
    );
    assert!(
        !write,
        "generic core::ptr::write must not survive as a shared callee\n{dump}"
    );

    let plain_new_header_keeps_typeptr = transformed.graph.blocks.iter().any(|block| {
        let news: Vec<_> = block
            .operations
            .iter()
            .filter_map(|op| match (&op.result, &op.kind) {
                (Some(r), OpKind::New { owner }) if owner.ends_with("PyObject") => Some(r),
                _ => None,
            })
            .collect();
        news.iter().any(|base| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { base: b, field, .. }
                        if b == *base && field.name == "ob_type"
                )
            })
        })
    });
    let any_ob_type = has_ob_type_store(&transformed.graph);
    assert!(
        fused || (residual && any_ob_type),
        "alloc_instance_object must stamp the typeptr: NewWithVtable, or a \
         residual nursery bump that keeps the ob_type store. fused={fused} \
         residual={residual} any_ob_type={any_ob_type} \
         plain_new_header_keeps={plain_new_header_keeps_typeptr}\n{dump}"
    );
    assert!(
        fused || plain_new_header_keeps_typeptr || !residual,
        "plain New PyObject must keep its ob_type store\n{dump}"
    );
}
