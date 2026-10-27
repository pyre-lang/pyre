//! Residual registration of `rffi.get_raw_address_of_string`.
//!
//! `@jit.dont_look_inside` (`rffi.py get_raw_address_of_string`) keeps the
//! helper out of `look_inside_graph`. Spine B would panic on `gc_can_move`
//! / `gc_move_out_of_nursery`; the call is a residual of the native
//! `HELPER_FNADDRS` row (`lltypesystem::rffi::get_raw_address_of_string`).
//! Shape matches [`super::alloc_and_set::register`]: bind fnaddr →
//! `register_function_graph` → `add_candidate_graph_if_look_inside`.
//!
//! `_get_raw_address_buf_from_string` (`rffi.py`) has no hint. Register
//! its graph for effect analysis only: `CallControl.find_all_graphs`
//! never makes it a candidate, because the only caller is the refused
//! `get_raw_address_of_string` graph.
//!
//! Effect analysis walks the translator helper graphs:
//! `build_ll_get_raw_address_of_string_helper_graph` (`gc_can_move`,
//! `gc_move_out_of_nursery`, NUL `setarrayitem` on `STR.chars`, and a
//! `direct_call` to `_get_raw_address_buf_from_string`) and
//! `build_get_raw_address_buf_from_string_family` (`gc_id`,
//! `ll_dict_getitem` with a KeyError exit, `str2charp`, `ll_dict_setitem`,
//! `cast_opaque_ptr`, `gc_fq_register`). Native LLBC of the majit-rlib
//! body would hide those as `RawStore` / `HashMap` and miss
//! `CollectAnalyzer`.

use crate::codewriter::call::{CallControl, effect_graph_from_rtyped_helper};
use crate::model::FunctionGraph;
use crate::parse::CallPath;
use crate::runtime_names::shims;
use crate::translator::rtyper::lltypesystem::rffi::{
    build_get_raw_address_buf_from_string_family, build_ll_get_raw_address_of_string_helper_graph,
};

/// Rtyper helper name (`lowlevel_helper_function_with_builder`).
pub const LL_GET_RAW_ADDRESS_OF_STRING: &str = "ll_get_raw_address_of_string";
/// Native `HELPER_FNADDRS` leaf after crate-strip.
const NATIVE_LEAF: &str = "get_raw_address_of_string";
/// Native buf helper leaf.
const BUF_LEAF: &str = "_get_raw_address_buf_from_string";

fn native_fnaddr_key(leaf: &str) -> CallPath {
    CallPath::from_segments(["lltypesystem", "rffi", leaf])
}

#[cfg(test)]
fn helper_fnaddr_path(leaf: &str) -> String {
    format!("majit_rlib::lltypesystem::rffi::{leaf}")
}

/// True when `segments` names a `get_raw_address_of_string` residual.
///
/// Skip-spine leftover is `[__majit_get_raw_address_of_string]`; the
/// rtyped helper is `[ll_get_raw_address_of_string]`. Both bind the
/// native `HELPER_FNADDRS` row (`rffi.get_raw_address_of_string`).
pub fn is_get_raw_address_path(segments: &[String]) -> bool {
    matches!(
        segments,
        [s] if s == shims::GET_RAW_ADDRESS_OF_STRING
            || s == LL_GET_RAW_ADDRESS_OF_STRING
            || s == NATIVE_LEAF
            || s == BUF_LEAF
    )
}

/// Path of `ll_get_raw_address_of_string`, minting the residual family.
pub fn ll_get_raw_address_of_string_path(cc: &mut CallControl) -> CallPath {
    let path = CallPath::from_segments([LL_GET_RAW_ADDRESS_OF_STRING]);
    if cc.has_function_graph(&path) {
        return path;
    }
    register(cc);
    path
}

fn register(cc: &mut CallControl) {
    bind_fnaddr(cc, NATIVE_LEAF, NATIVE_LEAF);
    bind_fnaddr(cc, LL_GET_RAW_ADDRESS_OF_STRING, NATIVE_LEAF);
    bind_fnaddr(cc, shims::GET_RAW_ADDRESS_OF_STRING, NATIVE_LEAF);
    bind_fnaddr(cc, BUF_LEAF, BUF_LEAF);
    // MiniMark `move_out_of_nursery` / `_find_shadow` after
    // `gct_gc_move_out_of_nursery`. The LLOp itself is not
    // `canmallocgc`; the transformed `direct_call` allocates the
    // shadow. `mark_canmallocgc` is the graph-less stand-in
    // (`collectanalyze.py analyze_simple_operation`).
    cc.mark_canmallocgc(CallPath::from_segments([
        "majit_gc",
        "gc_move_out_of_nursery",
    ]));
    // `gc_id` (`lloperation.py`) is `canmallocgc`. `ll_dict_setitem`
    // resizes through its helper graph (`rordereddict.py`); CollectAnalyzer
    // follows that `direct_call` once the callee graph is registered.
    cc.mark_canmallocgc(CallPath::from_segments(["gc_id"]));
    register_buf_effect_graph(cc);
    register_residual_graph(cc, NATIVE_LEAF);
    register_residual_graph(cc, LL_GET_RAW_ADDRESS_OF_STRING);
    register_residual_graph(cc, shims::GET_RAW_ADDRESS_OF_STRING);
}

fn bind_fnaddr(cc: &mut CallControl, dest_leaf: &str, src_leaf: &str) {
    let dest = CallPath::from_segments([dest_leaf]);
    if cc.lookup_function_fnaddr(&dest).is_some() {
        return;
    }
    let src = native_fnaddr_key(src_leaf);
    if cc.lookup_function_fnaddr(&src).is_some() {
        cc.alias_function_fnaddr(dest, &src);
    }
}

/// `_get_raw_address_buf_from_string` (`rffi.py`) has no
/// `@jit.dont_look_inside`. `CallControl.find_all_graphs` never makes it
/// a candidate: the only caller is the refused `get_raw_address_of_string`
/// graph. Register the rtyped body (`gc_id`, `ll_dict_getitem` with a
/// KeyError miss, `str2charp`, `ll_dict_setitem`, `cast_opaque_ptr`,
/// `gc_fq_register`) and every collected callee graph
/// (`copy_string_to_raw`, `ll_dict_getitem`, `ll_dict_setitem`, and
/// the lookup/resize helpers they call) for `getcalldescr` analyzers
/// without a tracing prohibition and without
/// `add_candidate_graph_if_look_inside`.
fn register_buf_effect_graph(cc: &mut CallControl) {
    let path = CallPath::from_segments([BUF_LEAF]);
    if cc.has_function_graph(&path) {
        return;
    }
    let (py, callees) = build_get_raw_address_buf_from_string_family(BUF_LEAF)
        .expect("_get_raw_address_buf_from_string helper graph");
    let graph = effect_graph_from_rtyped_helper(&py.graph.borrow());
    cc.register_function_graph(path, graph);
    for (name, gref) in callees {
        let callee_path = CallPath::from_segments([name.as_str()]);
        if cc.has_function_graph(&callee_path) {
            continue;
        }
        let callee_graph = effect_graph_from_rtyped_helper(&gref.borrow());
        cc.register_function_graph(callee_path, callee_graph);
    }
}

fn register_residual_graph(cc: &mut CallControl, name: &str) {
    let path = CallPath::from_segments([name]);
    if cc.has_function_graph(&path) {
        return;
    }
    let mut graph = residual_helper_graph(name);
    graph.push_hint("dont_look_inside");
    cc.register_function_graph(path.clone(), graph);
    cc.add_candidate_graph_if_look_inside(path);
}

/// Body is never traced (`dont_look_inside`). One STR argument, CCHARP
/// residual ABI (integer bank).
fn residual_helper_graph(name: &str) -> FunctionGraph {
    let py = build_ll_get_raw_address_of_string_helper_graph(name)
        .expect("ll_get_raw_address_of_string helper graph");
    effect_graph_from_rtyped_helper(&py.graph.borrow())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::call::CallKind;
    use crate::flowspace::model::Variable;
    use crate::model::{CallTarget, OpKind, SpaceOperation, ValueType};

    fn mint() -> CallControl {
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr(&helper_fnaddr_path(NATIVE_LEAF), 0x1000);
        cc.register_macro_helper_trace_fnaddr(&helper_fnaddr_path(BUF_LEAF), 0x2000);
        let _ = ll_get_raw_address_of_string_path(&mut cc);
        cc
    }

    fn call_op(name: &str) -> SpaceOperation {
        SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::function_path([name]),
                args: crate::model::call_args(vec![Variable::new()]),
                result_ty: ValueType::Int,
            },
        }
    }

    fn assert_residual(cc: &CallControl, name: &str, addr: i64) {
        let path = CallPath::from_segments([name]);
        assert!(cc.has_function_graph(&path), "missing graph {name}");
        assert!(
            !cc.is_candidate(&path),
            "look_inside_graph refuses `_jit_look_inside_ = False` for {name}"
        );
        let graph = cc
            .function_graphs()
            .get(&path)
            .unwrap_or_else(|| panic!("missing helper {name}"));
        assert_eq!(graph.func.jit_look_inside, Some(false));
        assert!(graph.hints.iter().any(|h| h == "dont_look_inside"));
        let call = call_op(name);
        assert!(cc.graphs_from(&call).is_none());
        assert_eq!(cc.guess_call_kind(&call), CallKind::Residual);
        assert_eq!(cc.lookup_function_fnaddr(&path), Some(addr));
        let OpKind::Call { target, .. } = &call.kind else {
            panic!("call_op is a Call");
        };
        let binding = cc.fnaddr_binding_for_target(target);
        assert!(!binding.symbolic, "{name} residual must be executable");
        assert!(!crate::codewriter::call::is_symbolic_fnaddr(binding.addr));
        assert_eq!(binding.addr, addr);
    }

    #[test]
    fn get_raw_address_of_string_is_residual_with_real_fnaddr() {
        let cc = mint();
        assert_residual(&cc, LL_GET_RAW_ADDRESS_OF_STRING, 0x1000);
        assert_residual(&cc, shims::GET_RAW_ADDRESS_OF_STRING, 0x1000);
        assert_residual(&cc, NATIVE_LEAF, 0x1000);
        assert_eq!(
            cc.lookup_function_fnaddr(&native_fnaddr_key(NATIVE_LEAF)),
            Some(0x1000)
        );
    }

    #[test]
    fn get_raw_address_buf_from_string_is_effect_graph_not_a_candidate() {
        let cc = mint();
        let path = CallPath::from_segments([BUF_LEAF]);
        assert!(cc.has_function_graph(&path), "missing graph {BUF_LEAF}");
        assert!(
            !cc.is_candidate(&path),
            "find_all_graphs never candidates {BUF_LEAF}"
        );
        let graph = cc
            .function_graphs()
            .get(&path)
            .unwrap_or_else(|| panic!("missing helper {BUF_LEAF}"));
        assert!(
            !graph.hints.iter().any(|h| h == "dont_look_inside"),
            "{BUF_LEAF} has no dont_look_inside hint"
        );
        assert_eq!(graph.func.jit_look_inside, None);
        assert_eq!(
            cc.lookup_function_fnaddr(&native_fnaddr_key(BUF_LEAF)),
            Some(0x2000)
        );
    }

    fn calldescr(cc: &CallControl, name: &str) -> crate::codewriter::call::CallDescriptor {
        use crate::codewriter::call::AnalysisCache;
        use majit_ir::descr::OopSpecIndex;
        use majit_ir::value::Type;
        let mut cache = AnalysisCache::default();
        cc.getcalldescr(
            &call_op(name),
            vec![Type::Ref],
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        )
    }

    /// `getcalldescr` (`call.py`) on a `dont_look_inside` helper that
    /// `gc_move_out_of_nursery`s, writes the NUL, and may take the
    /// buffer-cache path: `EF_CAN_RAISE` and `can_collect`.
    #[test]
    fn get_raw_address_of_string_calldescr_collects_and_can_raise() {
        use majit_ir::descr::ExtraEffect;
        let cc = mint();
        for name in [
            LL_GET_RAW_ADDRESS_OF_STRING,
            shims::GET_RAW_ADDRESS_OF_STRING,
            NATIVE_LEAF,
        ] {
            let descr = calldescr(&cc, name);
            assert_eq!(
                descr.extra_info.extraeffect,
                ExtraEffect::CanRaise,
                "{name}"
            );
            assert!(descr.extra_info.can_collect, "{name}");
            let graph = cc
                .function_graphs()
                .get(&CallPath::from_segments([name]))
                .unwrap_or_else(|| panic!("missing helper {name}"));
            assert!(
                graph
                    .iter_block_ops()
                    .any(|(_, op)| matches!(op.kind, OpKind::ArrayWrite { .. })),
                "{name} NUL is setarrayitem: {:?}",
                graph
                    .iter_block_ops()
                    .map(|(_, op)| format!("{:?}", op.kind))
                    .collect::<Vec<_>>()
            );
            assert!(
                graph.iter_block_ops().any(|(_, op)| matches!(
                    &op.kind,
                    OpKind::Call { target, .. }
                        if matches!(
                            target,
                            CallTarget::FunctionPath { segments, .. }
                                if segments == &["majit_gc".to_string(), "gc_move_out_of_nursery".to_string()]
                        )
                )),
                "{name} calls gc_move_out_of_nursery"
            );
            assert!(
                graph.iter_block_ops().any(|(_, op)| matches!(
                    &op.kind,
                    OpKind::Call { target, .. }
                        if matches!(
                            target,
                            CallTarget::FunctionPath { segments, .. }
                                if segments == &[BUF_LEAF.to_string()]
                        )
                )),
                "{name} calls {BUF_LEAF}"
            );
        }
    }

    #[test]
    fn get_raw_address_buf_from_string_calldescr_collects_and_can_raise() {
        use majit_ir::descr::ExtraEffect;
        let cc = mint();
        let descr = calldescr(&cc, BUF_LEAF);
        assert_eq!(descr.extra_info.extraeffect, ExtraEffect::CanRaise);
        assert!(descr.extra_info.can_collect);
        let graph = cc
            .function_graphs()
            .get(&CallPath::from_segments([BUF_LEAF]))
            .expect("buf helper graph");
        assert!(
            graph.iter_block_ops().any(|(_, op)| match &op.kind {
                OpKind::Call { target, .. } => match target {
                    CallTarget::FunctionPath { segments, .. } => {
                        segments
                            .iter()
                            .any(|s| s == "ll_dict_setitem" || s == "gc_id")
                    }
                    _ => false,
                },
                OpKind::LoweredBlackholeOp { opname, .. } => opname == "gc_id",
                OpKind::New { .. } => true,
                _ => false,
            }),
            "gc_id / ll_dict_setitem collect"
        );
    }

    #[test]
    fn get_raw_address_buf_from_string_graph_has_finalizer_getitem_str2charp() {
        use crate::model::{ExitCase, ExitSwitch};
        let cc = mint();
        let graph = cc
            .function_graphs()
            .get(&CallPath::from_segments([BUF_LEAF]))
            .expect("buf helper graph");
        let kinds: Vec<String> = graph
            .iter_block_ops()
            .map(|(_, op)| format!("{:?}", op.kind))
            .collect();
        assert!(
            graph.iter_block_ops().any(|(_, op)| match &op.kind {
                OpKind::Call { target, .. } => match target {
                    CallTarget::FunctionPath { segments, .. } => {
                        segments.iter().any(|s| s == "gc_fq_register")
                    }
                    _ => false,
                },
                OpKind::LoweredBlackholeOp { opname, .. } => opname == "gc_fq_register",
                _ => false,
            }),
            "gc_fq_register: {kinds:?}"
        );
        assert!(
            graph.iter_block_ops().any(|(_, op)| match &op.kind {
                OpKind::Call { target, .. } => match target {
                    CallTarget::FunctionPath { segments, .. } => {
                        segments.iter().any(|s| s == "ll_dict_getitem")
                    }
                    _ => false,
                },
                _ => false,
            }),
            "ll_dict_getitem: {kinds:?}"
        );
        let saw_keyerror = graph.iterblocks().iter().any(|block| {
            matches!(block.exitswitch, Some(ExitSwitch::LastException))
                && block.exits.iter().any(|link| match &link.exitcase {
                    Some(ExitCase::Const(c)) => format!("{c:?}").contains("KeyError"),
                    _ => false,
                })
        });
        assert!(saw_keyerror, "KeyError exit on getitem: {kinds:?}");
        assert!(
            graph.iter_block_ops().any(|(_, op)| match &op.kind {
                OpKind::Call { target, .. } => match target {
                    CallTarget::FunctionPath { segments, .. } => {
                        segments.iter().any(|s| s == "str2charp")
                    }
                    _ => false,
                },
                OpKind::RawMalloc { .. } | OpKind::New { .. } | OpKind::NewArray { .. } => true,
                _ => false,
            }),
            "str2charp call/malloc: {kinds:?}"
        );
    }

    #[test]
    fn copy_string_to_raw_oopspec_and_convertible_callees_are_registered() {
        let cc = mint();
        let target = CallTarget::function_path(["copy_string_to_raw"]);
        assert_eq!(
            cc.get_oopspec(&target).as_deref(),
            Some("stroruni.copy_string_to_raw(src, ptrdst, srcstart, length)")
        );
        assert_eq!(
            cc.get_oopspec_argnames(&target),
            Some(vec![
                "src".to_string(),
                "ptrdst".to_string(),
                "srcstart".to_string(),
                "length".to_string(),
            ])
        );
        assert!(
            cc.has_function_graph(&CallPath::from_segments(["copy_string_to_raw"])),
            "copy_string_to_raw graph is converted and registered"
        );
        assert!(
            cc.has_function_graph(&CallPath::from_segments(["str2charp"])),
            "str2charp graph is convertible and registered"
        );
        assert!(
            cc.has_function_graph(&CallPath::from_segments(["ll_dict_getitem"])),
            "ll_dict_getitem graph is convertible and registered"
        );
        assert!(
            cc.has_function_graph(&CallPath::from_segments(["ll_dict_setitem"])),
            "ll_dict_setitem graph is convertible and registered"
        );
    }

    /// `getcalldescr` (`call.py`) walks `copy_string_to_raw`,
    /// `ll_dict_getitem`, and `ll_dict_setitem`. Dict setitem writes
    /// (`_ll_dict_setitem_lookup_done`: `setfield resize_counter`,
    /// `setinteriorfield` key/value/f_valid) become write descriptors;
    /// interior writes also record the containing array
    /// (`effectinfo.py` extraef).
    #[test]
    fn get_raw_address_buf_from_string_walks_copy_and_dict_graphs() {
        use majit_ir::descr::ExtraEffect;
        let cc = mint();
        for name in [
            "copy_string_to_raw",
            "ll_dict_getitem",
            "ll_dict_setitem",
            "_ll_dict_setitem_lookup_done",
        ] {
            assert!(
                cc.has_function_graph(&CallPath::from_segments([name])),
                "missing converted graph {name}"
            );
        }

        let copy = cc
            .function_graphs()
            .get(&CallPath::from_segments(["copy_string_to_raw"]))
            .expect("copy_string_to_raw");
        assert!(
            copy.iter_block_ops().any(|(_, op)| matches!(
                &op.kind,
                OpKind::LoweredBlackholeOp { opname, .. } if opname == "raw_memcopy"
            )),
            "copy_string_to_raw raw_memcopy"
        );
        assert!(
            copy.iter_block_ops()
                .any(|(_, op)| matches!(op.kind, OpKind::Live)),
            "copy_string_to_raw keepalive"
        );

        let mut interior_writes = std::collections::BTreeSet::new();
        for (_, graph) in cc.function_graphs().iter() {
            for (_, op) in graph.iter_block_ops() {
                if let OpKind::InteriorFieldWrite { field, .. } = &op.kind {
                    interior_writes.insert(field.name.clone());
                }
            }
        }
        assert!(
            interior_writes.contains("key")
                && interior_writes.contains("value")
                && interior_writes.contains("f_valid"),
            "dict setitem interior writes: {interior_writes:?}"
        );

        let descr = calldescr(&cc, BUF_LEAF);
        assert_eq!(descr.extra_info.extraeffect, ExtraEffect::CanRaise);
        assert!(descr.extra_info.can_collect);
        assert!(
            descr
                .extra_info
                ._write_descrs_arrays
                .as_ref()
                .is_some_and(|ds| !ds.is_empty()),
            "setitem interior writes are array writes: {:?}",
            descr
                .extra_info
                ._write_descrs_arrays
                .as_ref()
                .map(|ds| ds.len())
        );
        assert!(
            descr
                .extra_info
                .write_descrs_arrays
                .as_ref()
                .is_some_and(|bs| bs.iter().any(|&b| b != 0)),
            "write_descrs_arrays bitstring"
        );
        assert!(
            descr
                .extra_info
                .write_descrs_fields
                .as_ref()
                .is_some_and(|bs| bs.iter().any(|&b| b != 0)),
            "setitem setfield resize_counter"
        );
    }
}
