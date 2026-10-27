//! Actual LLBC regression for the borrowed Result payload in Unicode errors.
//! The full dispatcher acceptance test separately exercises assembly with all
//! layouts and callees; this fixture isolates the source-lowering dependency.

use majit_charon_reader::Llbc;
use majit_translate::HostStaticAddrs;
use majit_translate::front::mir::{LowerContext, lower_fun_decl_with_static_addrs};
use majit_translate::model::{CallTarget, OpKind, ValueType};

#[test]
fn unicode_translate_error_borrow_chain_is_value_lowered() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    if !std::path::Path::new(path).is_file() {
        eprintln!("skipping: {path} is missing; run scripts/extract-llbc.py");
        return;
    }
    let llbc = Llbc::load(path).expect("load interpreter LLBC");
    let declaration = llbc
        .iter_local_fns()
        .find(|f| {
            f.item_meta.name_path() == "pyre_interpreter::display::unicode_translate_error_str"
        })
        .expect("unicode_translate_error_str exists in interpreter LLBC");
    let context = LowerContext::new(&llbc);
    let graph = lower_fun_decl_with_static_addrs(&context, declaration, HostStaticAddrs::default())
        .expect("lower unicode_translate_error_str, not reject the graph");
    let mut as_ref_calls = 0;
    let mut borrowed_integer_reads = 0;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        if let OpKind::Call { target, .. } = &op.kind {
            match target {
                CallTarget::Method {
                    name,
                    receiver_root,
                    ..
                } => {
                    if name == "as_ref" {
                        as_ref_calls += 1;
                    }
                    assert!(
                        !(receiver_root.as_deref() == Some("Result") && name == "expect"),
                        "Result::expect is still the unwrap rewrite: {:?}",
                        op.kind
                    );
                }
                CallTarget::FunctionPath { segments, .. }
                    if segments.last().map(String::as_str) == Some("as_ref")
                        && segments.iter().any(|s| s == "result") =>
                {
                    as_ref_calls += 1;
                }
                _ => {}
            }
        }
        if let OpKind::FieldRead { field, ty, .. } = &op.kind
            && field.name == "__pos_0"
            && field
                .owner_root
                .as_deref()
                .is_some_and(|owner| owner == "core::result::Result<i64,Wtf8Buf>::Ok")
        {
            assert_eq!(
                *ty,
                ValueType::Int,
                "read the lifted scalar, not its native address"
            );
            borrowed_integer_reads += 1;
        }
    }
    assert_eq!(
        as_ref_calls, 1,
        "caller keeps one regular call to the core as_ref body"
    );
    assert_eq!(
        borrowed_integer_reads, 1,
        "the as_ref Ok copy lives in the core body; the caller has one remaining __pos_0 Int read from nonconstant expect"
    );
}
