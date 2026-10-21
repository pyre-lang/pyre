//! Residual registration of `rffi.get_raw_address_of_string`.
//!
//! `@jit.dont_look_inside` (`rffi.py get_raw_address_of_string`) keeps the
//! helper out of `look_inside_graph`. Spine B would panic on `gc_can_move`
//! / `gc_move_out_of_nursery`; the call is a residual of the native
//! `HELPER_FNADDRS` row (`lltypesystem::rffi::get_raw_address_of_string`).
//! Shape matches [`super::alloc_and_set::register`]: bind fnaddr →
//! `register_function_graph` → `add_candidate_graph_if_look_inside`.
//!
//! `_get_raw_address_buf_from_string` is the same residual: its body
//! (`objectmodel.compute_unique_id`, `str2charp`, `raw_copies`) is not
//! expressed as ops. The translator graph `direct_call`s a
//! `functionptr_with_external_name` (`llexternal` sibling); this module
//! binds the native fnaddr.

use crate::codewriter::call::CallControl;
use crate::flowspace::model::Variable;
use crate::model::{CallTarget, ConcreteType, FunctionGraph, OpKind, SpaceOperation, ValueType};
use crate::parse::CallPath;
use crate::runtime_names::shims;

/// Rtyper helper name (`lowlevel_helper_function_with_builder`).
pub const LL_GET_RAW_ADDRESS_OF_STRING: &str = "ll_get_raw_address_of_string";
/// Native `HELPER_FNADDRS` leaf after crate-strip.
const NATIVE_LEAF: &str = "get_raw_address_of_string";
/// Native buf helper leaf.
const BUF_LEAF: &str = "_get_raw_address_buf_from_string";

fn native_fnaddr_key(leaf: &str) -> CallPath {
    CallPath::from_segments(["lltypesystem", "rffi", leaf])
}

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
    register_residual_graph(cc, NATIVE_LEAF);
    register_residual_graph(cc, LL_GET_RAW_ADDRESS_OF_STRING);
    register_residual_graph(cc, shims::GET_RAW_ADDRESS_OF_STRING);
    register_residual_graph(cc, BUF_LEAF);
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
    let mut graph = FunctionGraph::new(name);
    let start = graph.startblock;
    let s = graph.alloc_value_var_with_type(ConcreteType::GcRef);
    graph.push_inputarg_var(start, s.clone());
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "s".to_string(),
            ty: ValueType::Str,
            class_root: None,
        },
        s,
    );
    let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
    graph.push_op_with_result_var(start, OpKind::ConstInt(0), result.clone());
    graph.set_return(start, Some(result));
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::call::CallKind;

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
    fn get_raw_address_buf_from_string_is_residual_with_real_fnaddr() {
        let cc = mint();
        assert_residual(&cc, BUF_LEAF, 0x2000);
        assert_eq!(
            cc.lookup_function_fnaddr(&native_fnaddr_key(BUF_LEAF)),
            Some(0x2000)
        );
    }
}
