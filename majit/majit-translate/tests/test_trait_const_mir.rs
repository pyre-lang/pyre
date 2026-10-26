//! A selected impl's primitive `TraitConst` is that impl's literal.
//!
//! `alloc_exception_nursery` is generic over `T: GcType`. `allocate_exception`
//! calls `alloc_typed` at each concrete layout. That call copies the body
//! (`FunctionDesc.cachedgraph`) and the copy's `T::SIZE` is the impl's
//! `NamedConst`, folded by `const_eval_init_body`. A zero-arg
//! `__trait_const` call is not a registered graph.

use majit_charon_reader::Llbc;
use majit_translate::HostStaticAddrs;
use majit_translate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names;
use majit_translate::model::{CallTarget, OpKind};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-object.ullbc"
);

fn call_segments(target: &CallTarget) -> Option<&[String]> {
    match target {
        CallTarget::FunctionPath { segments, .. } => Some(segments),
        _ => None,
    }
}

#[test]
fn nursery_spec_folds_gc_type_size() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("pyre-object.ullbc is already extracted");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(&llbc),
        HostStaticAddrs::default(),
        &[],
        &[
            "alloc_exception_nursery",
            // Slim and extended layouts are both built in `allocate_exception`.
            // The nursery call itself sits in `alloc_typed`. The two
            // `dont_look_inside` constructors call `allocate_exception`.
            "allocate_exception",
            "alloc_typed",
            "w_exception_new_empty_impl",
            "w_exception_new_empty_extended_for_class",
        ],
    )
    .expect("interp exception graphs lower");

    let mut sizes = Vec::new();
    for function in &program.functions {
        for op in function
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
        {
            let OpKind::Call { target, .. } = &op.kind else {
                continue;
            };
            let Some(segments) = call_segments(target) else {
                continue;
            };
            assert!(
                !(segments.len() == 1 && segments[0] == "__trait_const"),
                "{} still calls __trait_const",
                function.name
            );
        }
        if function.name == "alloc_exception_nursery" {
            assert!(
                function.hints.iter().any(|hint| hint == "dont_look_inside"),
                "the unspecialized template still carries a Clause TraitConst"
            );
        }
        if !function.name.contains("alloc_exception_nursery__spec_") {
            continue;
        }
        assert!(
            !function.hints.iter().any(|hint| hint == "dont_look_inside"),
            "{} folded the const and stays look-inside",
            function.name
        );
        let call = function
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call { target, args, .. }
                    if call_segments(target)
                        .and_then(|segments| segments.last())
                        .is_some_and(|leaf| leaf == "try_gc_alloc_collecting_rooted") =>
                {
                    Some(args.clone())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("{} does not call the collecting allocator", function.name));
        let size_var = call
            .get(1)
            .and_then(|arg| arg.as_variable())
            .cloned()
            .unwrap_or_else(|| panic!("{} size argument is not a value", function.name));
        let size_op = function.graph().blocks.iter().find_map(|block| {
            block
                .operations
                .iter()
                .find_map(|op| (op.result.as_ref() == Some(&size_var)).then_some(op.kind.clone()))
        });
        let n = match size_op {
            Some(OpKind::ConstInt(n)) => n,
            Some(OpKind::ConstUInt(n)) => i64::try_from(n).expect("size fits i64"),
            other => panic!(
                "{} T::SIZE must be an integer constant, got {other:?}",
                function.name
            ),
        };
        assert!(n > 0, "{} folded a non-positive size {n}", function.name);
        sizes.push((function.name.clone(), n));
    }
    assert!(
        sizes.len() >= 2,
        "expected one spec copy per concrete exception layout, got {sizes:?}"
    );
    let distinct = sizes
        .iter()
        .map(|(_, n)| *n)
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        distinct.len() >= 2,
        "base and extended layouts must not share one size: {sizes:?}"
    );
}

fn is_trait_const_sentinel(target: &CallTarget) -> bool {
    call_segments(target).is_some_and(|segments| {
        segments.len() == 2 && segments[0] == "__str_const" && segments[1] == "__trait_const"
    })
}

#[test]
fn malloc_templates_with_a_clause_size_are_not_look_inside() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("pyre-object.ullbc is already extracted");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(&llbc),
        HostStaticAddrs::default(),
        &[],
        &[
            "malloc_typed",
            "malloc_typed_immortal",
            "malloc_typed_managed",
            "malloc_typed_stable",
            "w_module_new_managed",
        ],
    )
    .expect("malloc graphs lower");
    let names: Vec<_> = program
        .functions
        .iter()
        .map(|function| function.name.as_str())
        .collect();
    assert!(
        !names.iter().any(|name| name.contains("__spec_")),
        "host builtins are not specialized: {names:?}"
    );
    for name in [
        "malloc_typed",
        "malloc_typed_immortal",
        "malloc_typed_managed",
        "malloc_typed_stable",
    ] {
        let function = program
            .functions
            .iter()
            .find(|function| function.name == name)
            .unwrap_or_else(|| panic!("missing {name} in {names:?}"));
        let has_sentinel = function.graph().blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Call { target, .. } if is_trait_const_sentinel(target))
            })
        });
        // `malloc_typed` mentions `T::SIZE` only inside `debug_assert`, and
        // that operand does not survive lowering. The stable and managed
        // allocators pass it to the hook, so the sentinel stays live.
        if matches!(name, "malloc_typed_managed" | "malloc_typed_stable") {
            assert!(has_sentinel, "{name} dropped the unresolved TraitConst");
        }
        if !has_sentinel {
            continue;
        }
        assert!(
            function.hints.iter().any(|hint| hint == "dont_look_inside"),
            "{name} hints {:?}",
            function.hints
        );
        assert!(
            function
                .graph()
                .hints
                .iter()
                .any(|hint| hint == "dont_look_inside"),
            "{name} graph hints {:?}",
            function.graph().hints
        );
        assert_eq!(
            function.return_type, None,
            "{name} FUNC.RESULT must stay unstamped"
        );
    }
    let caller = program
        .functions
        .iter()
        .find(|function| function.name == "w_module_new_managed")
        .expect("w_module_new_managed");
    assert!(
        !caller.hints.iter().any(|hint| hint == "dont_look_inside"),
        "the caller has no Clause TraitConst: {:?}",
        caller.hints
    );
    let calls_template = caller.graph().blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target, .. } if call_segments(target).is_some_and(|segments| {
                    segments.last().map(String::as_str) == Some("malloc_typed_stable")
                        && !segments.iter().any(|segment| segment.contains("__spec_"))
                })
            )
        })
    });
    assert!(
        calls_template,
        "w_module_new_managed must keep calling malloc_typed_stable"
    );
}
