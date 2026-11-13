//! `for i in a..b` lowers like `rrange.py` `RangeIteratorRepr` /
//! `ll_rangenext_up`: int compare + add, no leftover `__majit_range`.
//!
//! The front rewrites the exclusive int `Range` to `__majit_range` plus
//! the `iter` / `__iter_next` bridge (`front/range_iter.rs`,
//! `front/iter_next.rs`).  `codewriter/iter_lower.rs` then scalarises
//! that loop the way `ll_rangenext_up` does, but only after `SSA_to_SSI`
//! has threaded the iterator through continue-edge predecessors.  A
//! graph whose Result/`?` rewrite leaves one undefined operand used to
//! make `ssa_to_ssi` abandon the whole graph, so the continue block
//! never carried the iterator and the reserved spelling survived as a
//! symbolic residual.

mod common;

use common::{MODULE_LLBC, load_llbc, lower_named_with_static_addrs};
use majit_translate::model::{CallTarget, ExitSwitch, FunctionGraph, OpKind};
use majit_translate::{ErrorCarrierSpec, GraphTransformConfig, HostStaticAddrs};

/// Same carrier the production prepass stamps (`pyre-jit-trace/build/prepass.rs`).
const ERROR_CARRIER: ErrorCarrierSpec<'static> = ErrorCarrierSpec {
    carrier_path: "pyre_interpreter::error::PyError",
    carrier_class: "",
    carrier_wrappers: &[],
    to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
    from_exc_object: Some(("PyError", "from_exc_object")),
};

fn production_addrs() -> HostStaticAddrs<'static> {
    HostStaticAddrs {
        error_carrier: ERROR_CARRIER,
        ..Default::default()
    }
}

fn count_markers(graph: &FunctionGraph) -> (usize, usize, usize) {
    let mut range = 0;
    let mut slice_iter = 0;
    let mut iter_next = 0;
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } = &op.kind
        {
            if segments.last().map(String::as_str) == Some("__majit_range") {
                range += 1;
            }
            if segments.last().map(String::as_str) == Some("iter")
                && segments.iter().any(|s| s == "slice")
            {
                slice_iter += 1;
            }
            if segments.last().map(String::as_str) == Some("__iter_next") {
                iter_next += 1;
            }
        }
    }
    (range, slice_iter, iter_next)
}

fn binop_count(graph: &FunctionGraph, name: &str) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == name))
        .count()
}

fn lower_ctypefunc(leaf: &str) -> FunctionGraph {
    let llbc = load_llbc(MODULE_LLBC);
    let name = format!("pyre_module::module::_cffi_backend::ctypefunc::{leaf}");
    lower_named_with_static_addrs(llbc, &name, production_addrs())
        .unwrap_or_else(|e| panic!("lower {leaf}: {e}"))
}

fn has_int_lt(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        matches!(
            &block.exitswitch,
            Some(ExitSwitch::Fused { opname, .. }) if opname == "int_lt" || opname == "lt"
        )
    }) || binop_count(graph, "lt") > 0
        || binop_count(graph, "int_lt") > 0
}

fn assert_range_scalarised(graph: &FunctionGraph, leaf: &str) {
    let (range, slice_iter, iter_next) = count_markers(graph);
    assert_eq!(
        (range, slice_iter, iter_next),
        (0, 0, 0),
        "{leaf} still carries the reserved range spelling after transform_graph \
         (range={range} slice::iter={slice_iter} __iter_next={iter_next}); \
         `ll_rangenext_up` leaves int ops only"
    );
    // `optimize_goto_if_not` fuses `int_lt` into the exitswitch
    // (`goto_if_not_int_lt`); the step stays a BinOp add.
    assert!(
        has_int_lt(graph) && binop_count(graph, "add") > 0,
        "{leaf}: `ll_rangenext_up` is `index < stop` then `index + 1`; \
         fused/binop lt={} add={}",
        has_int_lt(graph),
        binop_count(graph, "add")
    );
}

/// `W_CTypeFunc._call`'s `finally` (`release_arguments`) is
/// `@jit.unroll_safe` and loops `for i in range(mustfree_max_plus_1)`.
/// Isolated lowering has no Result rewrite, so SSI already succeeded
/// here; keep the scalarisation as a regression.
#[test]
fn release_arguments_range_scalarises_like_ll_rangenext_up() {
    let graph = lower_ctypefunc("release_arguments");
    let (range, _, _) = count_markers(&graph);
    assert!(
        range > 0,
        "release_arguments must still go through front::range_iter"
    );
    let transformed = majit_translate::codewriter::jtransform::transform_graph(
        &graph,
        &GraphTransformConfig::default(),
    );
    assert_range_scalarised(&transformed.graph, "release_arguments");
}

/// `do_call` returns `Result<_, PyError>`, so the production error
/// carrier rewrites `?` into exception edges and leaves one undefined
/// operand.  `ssa_to_ssi` used to restore the whole graph at that
/// startblock walk, which blocked iterator threading and left
/// `__majit_range` in the jitcode.
#[test]
fn do_call_range_scalarises_like_ll_rangenext_up_with_error_carrier() {
    let graph = lower_ctypefunc("do_call");
    let (range, slice_iter, iter_next) = count_markers(&graph);
    assert!(
        range > 0 && slice_iter > 0 && iter_next > 0,
        "do_call must still go through front::range_iter + iter_next \
         (range={range} slice::iter={slice_iter} __iter_next={iter_next})"
    );
    let transformed = majit_translate::codewriter::jtransform::transform_graph(
        &graph,
        &GraphTransformConfig::default(),
    );
    assert_range_scalarised(&transformed.graph, "do_call");
}
