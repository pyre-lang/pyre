//! Actual LLBC regression for the borrowed Result payload in Unicode errors.
//! The full dispatcher acceptance test separately exercises assembly with all
//! layouts and callees; this fixture isolates the source-lowering dependency.

mod common;

use common::{INTERPRETER_LLBC, load_llbc};
use majit_translate::HostStaticAddrs;
use majit_translate::front::mir::lower_fun_decl_with_static_addrs;
use majit_translate::model::{CallTarget, OpKind, ValueType};

fn leaf_is(name: &str, prefix: &str) -> bool {
    name == prefix || name.starts_with(&format!("{prefix}__spec_"))
}

#[test]
fn unicode_translate_error_borrow_chain_is_value_lowered() {
    let path = INTERPRETER_LLBC;
    let llbc = load_llbc(path);
    let declaration = llbc
        .iter_local_fns()
        .find(|f| {
            f.item_meta.name_path() == "pyre_interpreter::display::unicode_translate_error_str"
        })
        .expect("unicode_translate_error_str exists in interpreter LLBC");
    common::with_shared_lowering(llbc, |context| {
        let graph =
            lower_fun_decl_with_static_addrs(context, declaration, HostStaticAddrs::default())
                .expect("lower unicode_translate_error_str, not reject the graph");
        let mut as_ref_calls = 0;
        let mut result_expect_calls = 0;
        let mut borrowed_integer_reads = 0;
        for op in graph.blocks.iter().flat_map(|block| &block.operations) {
            if let OpKind::Call { target, .. } = &op.kind {
                match target {
                    CallTarget::Method {
                        name,
                        receiver_root,
                        resolved_path,
                        ..
                    } => {
                        if name == "as_ref" {
                            as_ref_calls += 1;
                        }
                        if leaf_is(name, "expect") {
                            let is_result = resolved_path
                                .as_ref()
                                .is_some_and(|p| p.segments.iter().any(|s| s == "result"))
                                || receiver_root.as_deref() == Some("Result")
                                || receiver_root
                                    .as_deref()
                                    .is_some_and(|root| root.starts_with("Result<"));
                            if is_result {
                                result_expect_calls += 1;
                            }
                        }
                    }
                    CallTarget::FunctionPath { segments, .. } => {
                        if segments.last().map(String::as_str) == Some("as_ref")
                            && segments.iter().any(|s| s == "result")
                        {
                            as_ref_calls += 1;
                        }
                        if segments.iter().any(|s| s == "result")
                            && segments.last().is_some_and(|s| leaf_is(s, "expect"))
                        {
                            result_expect_calls += 1;
                        }
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
            result_expect_calls, 2,
            "caller keeps two regular calls to the core Result expect body (`as_ref().expect` and `try_from.expect`), got {result_expect_calls}"
        );
        assert_eq!(
            borrowed_integer_reads, 0,
            "expect's Ok payload FieldRead lives in the extracted core body, got {borrowed_integer_reads}"
        );
    });
}
