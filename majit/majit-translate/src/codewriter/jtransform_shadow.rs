//! Inert opname-reachability shadow over the post-rtype flowspace graph.
//!
//! The production codewriter still consumes the rich-`OpKind`
//! `crate::model::FunctionGraph` (`jtransform.rs`'s `Transformer::transform`
//! dispatches on `OpKind::FieldRead`/`OpKind::Call`/…).  The rtyper, however,
//! already lowers the *flowspace* graph to upstream-shaped low-level
//! `SpaceOperation`s (`getfield`/`setfield`/`getarrayitem`/`direct_call`/
//! `int_add`/…) in place (`rtyper.rs` `specialize_block` rewrites
//! `block.operations`), and `specialize_legacy_graph_with_registry_…` holds
//! that graph as a local (`graph`, the `FlowspaceAdapterOutput.graph` from
//! `function_graph_to_flowspace`) through to its return.  Converging the
//! codewriter onto an opname-dispatching jtransform (matching `jtransform.py`'s
//! `rewrite_op_<opname>` model) is the endpoint; this module is the first,
//! observational step toward it.
//!
//! [`report_if_enabled`] walks the rtyped flowspace graph by
//! `SpaceOperation.opname` and reports, per graph, the opname histogram and
//! the subset of opnames that fall outside the admissible input set jtransform
//! accepts.  It never mutates the graph and never feeds the production jitcode
//! path; it is a no-op unless `MAJIT_JTRANSFORM_SHADOW` is set in the
//! environment, so the default build is byte-for-byte unchanged.  The
//! histogram of unhandled opnames collected across the corpus is the
//! per-handler migration backlog for the opname-dispatch rewrite.
//!
//! Maintenance note: [`REWRITE_OP_OPNAMES`] and [`BHIMPL_DEFAULT_OPNAMES`]
//! statically replicate upstream's reflective `_with_prefix('rewrite_op_')`
//! and `BlackholeInterpreter.__dict__` enumerations (`jtransform.py`).
//! They are a hand-captured snapshot, not a live mirror — an upstream rename
//! or a newly added `rewrite_op_*` / `bhimpl_*` will not surface here until
//! these arrays are re-synced by hand, so a stale entry would misreport an
//! admissible opname as "unknown" (or vice versa).
//!
//! Placement note: the fused path reports only after `drive_subject(...,
//! do_rtype=true)` finishes specialization. The production two-phase path
//! retains each Phase-A `GraphRef`, specializes it in place during Phase B,
//! then reports only graphs absent from `rtype_skipped`. Reporting during
//! Phase A would be wrong: `simple_call` / `getattr` are still high-level
//! operations there and are not an opname-jtransform backlog. Graphs that
//! fall back to the legacy walker never reach the report. The resulting
//! histogram is the actual per-handler migration backlog for successfully
//! rtyped graphs.

use std::collections::{BTreeMap, HashSet};

use crate::flowspace::model::FunctionGraph;

/// Every opname for which `Transformer` defines a `rewrite_op_<opname>`
/// attribute, i.e. the keys of `_with_prefix('rewrite_op_')`
/// (`jtransform.py`).  `_with_prefix` reflects over `dir(Transformer)`,
/// so the set is every `rewrite_op_*` name the class carries, from four
/// forms — all reproduced here (172 names):
///   - explicit `def rewrite_op_<name>` methods;
///   - assignment aliases (`rewrite_op_int_add = _rewrite_symmetric`,
///     `rewrite_op_int_abs = _do_builtin_call`, …);
///   - the `exec`-generated long-long / unsigned-long-long handlers
///     (`jtransform.py`): the `llong_*` / `ullong_*` ops and the
///     `cast_*_to_(u)longlong` / `(cast|truncate)_longlong_*` family;
///   - the `exec`-generated renames (`jtransform.py`): `bool_not`,
///     `keepalive`, `char_*`, `unichar_*`, `uint_*`, `adr_add`, ….
const REWRITE_OP_OPNAMES: &[&str] = &[
    "adr_add",
    "bool_not",
    "cast_bool_to_float",
    "cast_bool_to_int",
    "cast_bool_to_uint",
    "cast_char_to_int",
    "cast_float_to_longlong",
    "cast_float_to_uint",
    "cast_float_to_ulonglong",
    "cast_int_to_char",
    "cast_int_to_longlong",
    "cast_int_to_uint",
    "cast_int_to_ulonglong",
    "cast_int_to_unichar",
    "cast_longlong_to_float",
    "cast_opaque_ptr",
    "cast_pointer",
    "cast_primitive",
    "cast_ptr_to_adr",
    "cast_ptr_to_int",
    "cast_uint_to_float",
    "cast_uint_to_int",
    "cast_uint_to_longlong",
    "cast_uint_to_ulonglong",
    "cast_ulonglong_to_float",
    "cast_unichar_to_int",
    "char_eq",
    "char_ge",
    "char_gt",
    "char_le",
    "char_lt",
    "char_ne",
    "convert_float_bytes_to_longlong",
    "convert_longlong_bytes_to_float",
    "debug_assert",
    "debug_assert_not_none",
    "direct_call",
    "direct_ptradd",
    "float_add",
    "float_ge",
    "float_gt",
    "float_is_true",
    "float_le",
    "float_lt",
    "float_mul",
    "force_cast",
    "free",
    "gc_add_memory_pressure",
    "gc_id",
    "gc_identityhash",
    "gc_load_indexed",
    "gc_pin",
    "gc_store_indexed",
    "gc_unpin",
    "getarrayitem",
    "getarraysize",
    "getarraysubstruct",
    "getfield",
    "getinteriorarraysize",
    "getinteriorfield",
    "getsubstruct",
    "hint",
    "indirect_call",
    "int_abs",
    "int_add",
    "int_add_nonneg_ovf",
    "int_add_ovf",
    "int_and",
    "int_eq",
    "int_floordiv",
    "int_ge",
    "int_gt",
    "int_is_true",
    "int_le",
    "int_lt",
    "int_mod",
    "int_mul",
    "int_mul_ovf",
    "int_ne",
    "int_or",
    "int_sub_ovf",
    "int_xor",
    "jit_conditional_call",
    "jit_conditional_call_value",
    "jit_enter_portal_frame",
    "jit_force_quasi_immutable",
    "jit_force_virtual",
    "jit_force_virtualizable",
    "jit_is_virtual",
    "jit_leave_portal_frame",
    "jit_marker",
    "jit_record_exact_class",
    "jit_record_exact_value",
    "jit_record_known_result",
    "keepalive",
    "likely",
    "ll_get_timestamp_unit",
    "ll_read_timestamp",
    "llong_abs",
    "llong_add",
    "llong_and",
    "llong_eq",
    "llong_floordiv",
    "llong_ge",
    "llong_gt",
    "llong_invert",
    "llong_is_true",
    "llong_le",
    "llong_lshift",
    "llong_lt",
    "llong_mod",
    "llong_mul",
    "llong_ne",
    "llong_neg",
    "llong_or",
    "llong_rshift",
    "llong_sub",
    "llong_xor",
    "malloc",
    "malloc_varsize",
    "ptr_eq",
    "ptr_iszero",
    "ptr_ne",
    "ptr_nonzero",
    "raw_load",
    "raw_malloc_usage",
    "raw_store",
    "revdb_do_next_call",
    "same_as",
    "setarrayitem",
    "setfield",
    "setinteriorfield",
    "threadlocalref_get",
    "truncate_longlong_to_int",
    "uint_add",
    "uint_and",
    "uint_eq",
    "uint_ge",
    "uint_gt",
    "uint_invert",
    "uint_is_true",
    "uint_le",
    "uint_lshift",
    "uint_lt",
    "uint_mul",
    "uint_ne",
    "uint_or",
    "uint_sub",
    "uint_xor",
    "ullong_add",
    "ullong_and",
    "ullong_eq",
    "ullong_floordiv",
    "ullong_ge",
    "ullong_gt",
    "ullong_invert",
    "ullong_is_true",
    "ullong_le",
    "ullong_lshift",
    "ullong_lt",
    "ullong_mod",
    "ullong_mul",
    "ullong_ne",
    "ullong_or",
    "ullong_rshift",
    "ullong_sub",
    "ullong_xor",
    "unichar_eq",
    "unichar_ne",
    "unlikely",
    "weakref_create",
    "weakref_deref",
];

/// Every opname the blackhole interpreter carries a `bhimpl_<opname>`
/// handler for, i.e. the opnames `_add_default_ops` folds into
/// `_rewrite_ops` by scanning `BlackholeInterpreter.__dict__`
/// (`jtransform.py`).  Mirrors the *upstream* universe in
/// `rpython/jit/metainterp/blackhole.py` — every `def bhimpl_<name>` plus
/// the `bhimpl_<name> = bhimpl_<other>` aliases — with the `bhimpl_` prefix
/// stripped (`key[len('bhimpl_'):]`), exactly as upstream derives the
/// opname.  242 names.
///
/// This is the upstream contract, NOT pyre's partially-ported blackhole:
/// `insns::wellknown_bh_insns()` is only the byte-assigned subset pyre's
/// runtime emits today, so keying on it would misclassify upstream-valid
/// opnames (`new`, `guard_class`, `strgetitem`, `virtual_ref`, …) as
/// "unknown".  jtransform passes those through unmodified to the blackhole,
/// so they are admissible input even before pyre ports their handler;
/// surfacing them as unknown would overstate the per-handler backlog.  The
/// argcode-suffixed method names (`getfield_gc_i`, `record_known_result_i_ir_v`,
/// `residual_call_ir_i`, …) are the opnames verbatim, since `_add_default_ops`
/// keys on the full method name minus prefix.
const BHIMPL_DEFAULT_OPNAMES: &[&str] = &[
    "arraylen_gc",
    "arraylen_vable",
    "assert_not_none",
    "cast_float_to_int",
    "cast_float_to_singlefloat",
    "cast_int_to_float",
    "cast_int_to_ptr",
    "cast_ptr_to_int",
    "cast_singlefloat_to_float",
    "catch_exception",
    "check_neg_index",
    "check_resizable_neg_index",
    "conditional_call_ir_v",
    "conditional_call_value_ir_i",
    "conditional_call_value_ir_r",
    "convert_float_bytes_to_longlong",
    "convert_longlong_bytes_to_float",
    "copystrcontent",
    "copyunicodecontent",
    "current_trace_length",
    "debug_fatalerror",
    "float_abs",
    "float_add",
    "float_assert_green",
    "float_copy",
    "float_eq",
    "float_ge",
    "float_gt",
    "float_guard_value",
    "float_isconstant",
    "float_le",
    "float_lt",
    "float_mul",
    "float_ne",
    "float_neg",
    "float_pop",
    "float_push",
    "float_return",
    "float_sub",
    "float_truediv",
    "gc_load_indexed_f",
    "gc_load_indexed_i",
    "gc_store_indexed_f",
    "gc_store_indexed_i",
    "getarrayitem_gc_f",
    "getarrayitem_gc_f_pure",
    "getarrayitem_gc_i",
    "getarrayitem_gc_i_pure",
    "getarrayitem_gc_r",
    "getarrayitem_gc_r_pure",
    "getarrayitem_raw_f",
    "getarrayitem_raw_i",
    "getarrayitem_vable_f",
    "getarrayitem_vable_i",
    "getarrayitem_vable_r",
    "getfield_gc_f",
    "getfield_gc_f_greenfield",
    "getfield_gc_f_pure",
    "getfield_gc_i",
    "getfield_gc_i_greenfield",
    "getfield_gc_i_pure",
    "getfield_gc_r",
    "getfield_gc_r_greenfield",
    "getfield_gc_r_pure",
    "getfield_raw_f",
    "getfield_raw_i",
    "getfield_raw_r",
    "getfield_vable_f",
    "getfield_vable_i",
    "getfield_vable_r",
    "getinteriorfield_gc_f",
    "getinteriorfield_gc_i",
    "getinteriorfield_gc_r",
    "getlistitem_gc_f",
    "getlistitem_gc_i",
    "getlistitem_gc_r",
    "goto",
    "goto_if_exception_mismatch",
    "goto_if_not",
    "goto_if_not_float_eq",
    "goto_if_not_float_ge",
    "goto_if_not_float_gt",
    "goto_if_not_float_le",
    "goto_if_not_float_lt",
    "goto_if_not_float_ne",
    "goto_if_not_int_eq",
    "goto_if_not_int_ge",
    "goto_if_not_int_gt",
    "goto_if_not_int_is_true",
    "goto_if_not_int_is_zero",
    "goto_if_not_int_le",
    "goto_if_not_int_lt",
    "goto_if_not_int_ne",
    "goto_if_not_ptr_eq",
    "goto_if_not_ptr_iszero",
    "goto_if_not_ptr_ne",
    "goto_if_not_ptr_nonzero",
    "guard_class",
    "hint_force_virtualizable",
    "inline_call_ir_i",
    "inline_call_ir_r",
    "inline_call_ir_v",
    "inline_call_irf_f",
    "inline_call_irf_i",
    "inline_call_irf_r",
    "inline_call_irf_v",
    "inline_call_r_i",
    "inline_call_r_r",
    "inline_call_r_v",
    "instance_ptr_eq",
    "instance_ptr_ne",
    "int_add",
    "int_add_jump_if_ovf",
    "int_and",
    "int_assert_green",
    "int_between",
    "int_copy",
    "int_eq",
    "int_force_ge_zero",
    "int_ge",
    "int_gt",
    "int_guard_value",
    "int_invert",
    "int_is_true",
    "int_is_zero",
    "int_isconstant",
    "int_isvirtual",
    "int_le",
    "int_lshift",
    "int_lt",
    "int_mul",
    "int_mul_jump_if_ovf",
    "int_ne",
    "int_neg",
    "int_or",
    "int_pop",
    "int_push",
    "int_return",
    "int_rshift",
    "int_same_as",
    "int_signext",
    "int_sub",
    "int_sub_jump_if_ovf",
    "int_xor",
    "jit_debug",
    "jit_enter_portal_frame",
    "jit_force_quasi_immutable",
    "jit_leave_portal_frame",
    "jit_merge_point",
    "last_exc_value",
    "last_exception",
    "live",
    "loop_header",
    "new",
    "new_array",
    "new_array_clear",
    "new_with_vtable",
    "newlist",
    "newlist_clear",
    "newlist_hint",
    "newstr",
    "newunicode",
    "ptr_eq",
    "ptr_iszero",
    "ptr_ne",
    "ptr_nonzero",
    "raise",
    "raw_load_f",
    "raw_load_i",
    "raw_store_f",
    "raw_store_i",
    "record_exact_class",
    "record_exact_value_i",
    "record_exact_value_r",
    "record_known_result_i_ir_v",
    "record_known_result_r_ir_v",
    "record_quasiimmut_field",
    "recursive_call_f",
    "recursive_call_i",
    "recursive_call_r",
    "recursive_call_v",
    "ref_assert_green",
    "ref_copy",
    "ref_guard_value",
    "ref_isconstant",
    "ref_isvirtual",
    "ref_pop",
    "ref_push",
    "ref_return",
    "reraise",
    "residual_call_ir_i",
    "residual_call_ir_r",
    "residual_call_ir_v",
    "residual_call_irf_f",
    "residual_call_irf_i",
    "residual_call_irf_r",
    "residual_call_irf_v",
    "residual_call_r_i",
    "residual_call_r_r",
    "residual_call_r_v",
    "rvmprof_code",
    "setarrayitem_gc_f",
    "setarrayitem_gc_i",
    "setarrayitem_gc_r",
    "setarrayitem_raw_f",
    "setarrayitem_raw_i",
    "setarrayitem_vable_f",
    "setarrayitem_vable_i",
    "setarrayitem_vable_r",
    "setfield_gc_f",
    "setfield_gc_i",
    "setfield_gc_r",
    "setfield_raw_f",
    "setfield_raw_i",
    "setfield_vable_f",
    "setfield_vable_i",
    "setfield_vable_r",
    "setinteriorfield_gc_f",
    "setinteriorfield_gc_i",
    "setinteriorfield_gc_r",
    "setlistitem_gc_f",
    "setlistitem_gc_i",
    "setlistitem_gc_r",
    "str_guard_value",
    "strgetitem",
    "strhash",
    "strlen",
    "strsetitem",
    "switch",
    "uint_ge",
    "uint_gt",
    "uint_le",
    "uint_lt",
    "uint_mul_high",
    "uint_rshift",
    "unicodegetitem",
    "unicodehash",
    "unicodelen",
    "unicodesetitem",
    "unreachable",
    "virtual_ref",
    "virtual_ref_finish",
    "void_return",
];

/// The opname set jtransform accepts as input, mirroring upstream
/// `_rewrite_ops` (`jtransform.py`):
///   `_with_prefix('rewrite_op_')`  — the explicit rewrites ([`REWRITE_OP_OPNAMES`])
///   `∪ _add_default_ops`            — every blackhole `bhimpl_<opname>`, kept
///                                     unmodified ([`BHIMPL_DEFAULT_OPNAMES`])
///   `∪ {'-live-'}`.
///
/// The default-ops half is the *upstream* `BlackholeInterpreter.__dict__`
/// `bhimpl_*` universe ([`BHIMPL_DEFAULT_OPNAMES`]), not the byte-assigned
/// `insns::wellknown_bh_insns()` subset pyre's runtime emits today — see that
/// const's doc for why the subset would misclassify upstream-valid
/// passthrough opnames as "unknown".  `'-live-'` is added separately, exactly
/// as `_add_default_ops` does (`jtransform.py`).
fn admissible_opnames() -> HashSet<String> {
    REWRITE_OP_OPNAMES
        .iter()
        .chain(BHIMPL_DEFAULT_OPNAMES.iter())
        .map(|name| name.to_string())
        .chain(std::iter::once("-live-".to_string()))
        .collect()
}

/// Whether the env-gated shadow gauge is active (`MAJIT_JTRANSFORM_SHADOW`
/// set).  Centralizes the switch so a call site can skip even the graph
/// borrow when the gauge is off, keeping the default build's behaviour
/// unchanged.
pub fn is_enabled() -> bool {
    std::env::var_os("MAJIT_JTRANSFORM_SHADOW").is_some()
}

/// Walk the post-rtype flowspace graph by `SpaceOperation.opname` and emit a
/// per-graph diagnostic.  No-op unless `MAJIT_JTRANSFORM_SHADOW` is set; never
/// mutates the graph.
pub fn report_if_enabled(graph: &FunctionGraph) {
    if !is_enabled() {
        return;
    }
    let admissible = admissible_opnames();
    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut total = 0usize;
    for block in graph.iterblocks() {
        let block = block.borrow();
        for op in &block.operations {
            *histogram.entry(op.opname.clone()).or_default() += 1;
            total += 1;
        }
    }
    let unknown: BTreeMap<&String, &usize> = histogram
        .iter()
        .filter(|(opname, _)| !admissible.contains(opname.as_str()))
        .collect();
    eprintln!(
        "[jtransform-shadow] graph={:?} total={} distinct={} unknown={:?} hist={:?}",
        graph.name,
        total,
        histogram.len(),
        unknown,
        histogram,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admissible_set_covers_rtyper_emitted_opnames() {
        let admissible = admissible_opnames();
        // Structural low-level ops the rtyper emits via `genop` on the
        // flowspace graph — each must be in jtransform's input contract.
        for opname in [
            "getfield",
            "setfield",
            "getarrayitem",
            "setarrayitem",
            "direct_call",
            "indirect_call",
            "malloc",
            "same_as",
            "cast_pointer",
            "ptr_eq",
            "int_add",
            "int_sub",
        ] {
            assert!(
                admissible.contains(opname),
                "expected {opname:?} in jtransform admissible set"
            );
        }
    }

    #[test]
    fn admissible_set_covers_passthrough_bhimpl_opnames() {
        let admissible = admissible_opnames();
        // Opnames with no `rewrite_op_` but a `bhimpl_*` handler — jtransform
        // passes them through unmodified, so they are admissible input even
        // though pyre's runtime blackhole has not ported them yet.  The
        // byte-table subset would have flagged these as "unknown".
        for opname in [
            "new",
            "new_array",
            "guard_class",
            "strgetitem",
            "unicodegetitem",
            "virtual_ref",
            "int_is_zero",
            "record_known_result_i_ir_v",
        ] {
            assert!(
                admissible.contains(opname),
                "expected passthrough bhimpl opname {opname:?} in admissible set"
            );
        }
    }

    #[test]
    fn admissible_set_excludes_unknown_opname() {
        let admissible = admissible_opnames();
        assert!(!admissible.contains("definitely_not_a_real_opname_xyz"));
        // Pyre-only opnames (no upstream `bhimpl_*`) stay outside the
        // admissible set so the gauge surfaces them in its "unknown" backlog.
        assert!(!admissible.contains("abort"));
    }

    #[test]
    fn opname_slices_are_unique() {
        let cases: &[(&str, &[&str])] = &[
            ("rewrite", REWRITE_OP_OPNAMES),
            ("bhimpl_default", BHIMPL_DEFAULT_OPNAMES),
        ];
        for (name, names) in cases {
            let unique: std::collections::HashSet<&&str> = names.iter().collect();
            assert_eq!(unique.len(), names.len(), "case {name}");
        }
    }
}
