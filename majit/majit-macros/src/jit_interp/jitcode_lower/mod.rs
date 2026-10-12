mod api;
mod dispatch;
mod helpers;
mod jit_state_analysis;
mod liveness;
mod lower_control;
mod lower_stmt;
mod lower_vable;
mod lower_value;
mod lowerer;
mod regalloc;

pub use api::GeneratedJitCodeBody;
#[allow(unused_imports)]
pub(crate) use api::{
    CallerLocalLayout, assign_caller_local_layout, generate_inline_helper_jitcode_with_calls,
    inline_helper_arg_classes, inline_helper_param_counts, inline_helper_param_layout,
    inline_return_kind_class, try_generate_jitcode_body_parts_with_caller_bindings,
    try_generate_jitcode_pc_return_body_with_caller_bindings,
};
#[allow(unused_imports)]
pub use api::{try_generate_jitcode_body, try_generate_jitcode_body_with_config};
pub(crate) use dispatch::{is_can_enter_jit_macro, is_jit_merge_point_macro, lower_dispatch_body};
pub(super) use helpers::helper_policy_path;
pub(crate) use helpers::{call_is_null_ptr, classify_param_type};

// Re-export submodule items for sibling-submodule access via `use super::*`.
// These appear unused in mod.rs itself but are consumed by submodules and tests.
mod reexports {
    #![allow(unused_imports)]
    pub(super) use super::api::bind_pre_merge_point_stmts;
    pub(super) use super::dispatch::{
        collect_arm_caller_locals, collect_pat_bound_idents, dispatch_arm_inline_call_tokens,
        emit_promote_greens, env_array_descr_expr, find_dispatch_loop_body, green_schema,
        is_jit_merge_point_macro, lower_dispatch_chain, lower_pre_dispatch_stmts, pc_is_green,
        red_schema, resolve_greens, resolve_reds,
    };
    pub(super) use super::helpers::{
        assert_ref_result_one_word, binding_kind_for_inline_policy, binop_f_emit_tokens,
        binop_i_accepts_constant, binop_i_const_emit_tokens, binop_i_emit_tokens,
        binop_is_symmetric, block_has_loop_control, call_is_null_ptr, expr_has_loop_control,
        expr_is_literal_call, expr_is_null_ptr, expr_is_ptr_is_null_method, expr_is_unsigned_int,
        extend_int_args_with_array_identity, extract_block_tail_int, extract_bool_branch_values,
        extract_branch_int, extract_pat_switch_case_tokens, extract_pat_value_tokens,
        extract_stmts, inline_call_read_regs, inline_call_tokens, inline_call_tokens_void,
        inline_float_arg_tokens, inline_prebuild_path, inline_ref_arg_tokens, inline_shared_path,
        int_arg_regs, int_literal_value, is_lowercase_binding_pat, is_supported_float_type,
        is_supported_int_cast, is_supported_ref_type, is_word_width_int, jit_arg_kind_tokens,
        mirrored_compare_binop, opcode_for_assign_binop, opcode_for_assign_binop_f,
        opcode_for_binop, opcode_for_binop_f, opcode_for_compare_f, plain_array_identity_int_pairs,
        quote_inline_arg_pairs, stmt_has_loop_control, type_is_raw_pointer, type_is_unsigned_int,
        typed_call_arg_tokens, word_result_addr_for_kind, word_result_addr_tokens,
        word_void_addr_tokens,
    };
    pub(super) use super::liveness::{
        annotate_live_markers_with_liveness, compute_per_marker_liveness, get_liveness_info,
        liveness_prebuild_tokens, liveness_triple, liveness_triple_from_reads, maybe_dump_liveness,
        remove_repeated_live, rewrite_live_marker_statements_with_triples,
    };
    pub(super) use super::lowerer::Lowerer;
}
#[allow(unused_imports)]
use reexports::*;

use std::collections::{BTreeSet, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};

use super::call_policy_byte::{
    INT_DONT_LOOK_INSIDE, INT_DONT_LOOK_INSIDE_CANNOT_RAISE, INT_ELIDABLE,
    INT_ELIDABLE_CANNOT_RAISE, INT_ELIDABLE_OR_MEMERROR, INT_INLINE, INT_LOOP_INVARIANT,
    INT_MAY_FORCE, REF_DONT_LOOK_INSIDE, REF_DONT_LOOK_INSIDE_CANNOT_RAISE, REF_ELIDABLE,
    REF_ELIDABLE_CANNOT_RAISE, REF_ELIDABLE_OR_MEMERROR, REF_LOOP_INVARIANT, REF_MAY_FORCE,
    VOID_DONT_LOOK_INSIDE, VOID_DONT_LOOK_INSIDE_CANNOT_RAISE, VOID_LOOP_INVARIANT, VOID_MAY_FORCE,
};
use super::codegen_trace::{
    block_contains_match, find_dispatch_match, find_portal_loop, is_assert_not_none_call_path,
    is_promote_call_path, is_record_exact_class_call_path, portal_loop_body, stmt_contains_match,
};
use syn::{
    BinOp, Block, Expr, ExprAssign, ExprBinary, ExprCall, ExprCast, ExprIf, ExprLit, ExprMatch,
    ExprMethodCall, ExprParen, ExprPath, ExprReference, ExprUnary, FnArg, Ident, ItemFn, Lit,
    Local, Pat, Path, ReturnType, Stmt, Type, UnOp,
};

// Duplicated from majit-translate::hints — proc-macro crates cannot depend
// on heavy library crates, so we inline the small enum + classifier here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VirtualizableHintKind {
    AccessDirectly,
    FreshVirtualizable,
    ForceVirtualizable,
}

pub(super) fn classify_virtualizable_hint_segments<'a, I>(
    segments: I,
) -> Option<VirtualizableHintKind>
where
    I: IntoIterator<Item = &'a str>,
{
    match segments.into_iter().last().unwrap_or_default() {
        "hint_access_directly" => Some(VirtualizableHintKind::AccessDirectly),
        "hint_fresh_virtualizable" => Some(VirtualizableHintKind::FreshVirtualizable),
        "hint_force_virtualizable" => Some(VirtualizableHintKind::ForceVirtualizable),
        _ => None,
    }
}

// ── LowererConfig ────────────────────────────────────────────────────

/// One `pool_arrays` declaration, resolved against the state's `ref(_)` fields.
///
/// `JitInterpConfig.PoolArrayEntry` names the base by its state-field name;
/// this carries the base struct's `Path` alongside it, recovered from
/// `state_ref_scalars`, because the emitted `offset_of!` needs the type and the
/// declaration only names the field.
#[derive(Clone)]
pub(super) struct PoolArrayLowering {
    /// The `ref(T)` state field that holds the base pointer.
    pub(super) base: String,
    /// The marker function whose call indexes the array, canonicalized.
    pub(super) getter: Vec<String>,
    /// `T` of the `ref(T)` — the struct the array lives in.
    pub(super) struct_path: syn::Path,
    /// The array field inside `struct_path`.
    pub(super) items_field: Ident,
    /// A length word ahead of the items, when the consumer keeps one.
    pub(super) len_field: Option<Ident>,
    /// The element's pointee type, when declared.
    pub(super) element_type: Option<syn::Path>,
}

/// Configuration for state_fields-aware JitCode lowering.
///
/// Built from `JitInterpConfig` at proc-macro time and passed to the Lowerer
/// to recognize state-field reads/writes, virtualizable accesses, I/O shims,
/// and helper-call policies.
#[derive(Clone)]
pub struct LowererConfig {
    /// Canonical I/O func path → shim ident.
    pub(super) io_shims: Vec<(Vec<String>, Ident)>,
    /// Canonical helper func path → explicit or inferred call policy.
    pub(super) calls: Vec<(Vec<String>, CallPolicySpec)>,
    /// Whether top-level traced calls should auto-infer helper policy.
    pub(super) auto_calls: bool,
    /// Virtualizable variable name (normalized, e.g., "frame").
    /// RPython jtransform.py: `is_virtualizable_getset()` uses this to check
    /// if a field access target is the virtualizable variable.
    pub(super) vable_var: Option<String>,
    /// Ref-register assigned to the virtualizable input variable.
    ///
    /// RPython `MIFrame.setup_call(original_boxes)` distributes portal args
    /// by kind before opimpls consume `v_inst` / `v_base`.  The generated
    /// observer JitCode fragment receives the virtualizable as its first Ref
    /// input, so the line-by-line graph variable is `registers_r[0]`.
    pub(super) vable_input_ref_reg: Option<u16>,
    /// Field name → (field_index, field_type).
    /// RPython: `vinfo.static_field_to_extra_box[fieldname]` → index.
    pub(super) vable_fields: HashMap<String, (usize, ValueKind)>,
    /// Array name → (array_index, item_type).
    /// RPython: `vinfo.array_field_counter[fieldname]` → index.
    pub(super) vable_arrays: HashMap<String, (usize, ValueKind)>,
    /// Scalar declarations: field_name → legacy state-slot index. Fields also
    /// present in `vable_fields` use that object's field index instead.
    pub(super) state_scalars: HashMap<String, usize>,
    /// Plain `[int]` state arrays: field_name → global_array_index.
    /// Each is two identity slots: items pointer then length
    /// (`rlist.py` `rtype_getitem` `dum_nocheck`; the Vec is a raw
    /// `{ptr, len}` view).
    pub(super) state_arrays: HashMap<String, usize>,
    /// State field virtualizable arrays: field_name → (virt_array_index, item_type).
    /// These emit GETARRAYITEM_RAW_I/SETARRAYITEM_RAW instead of element-level tracking.
    pub(super) state_virt_arrays: HashMap<String, (usize, ValueKind)>,
    /// State field ref scalars: field_name → (ref_scalar_index, struct Path).
    /// The index is 0-based in its own space (separate from `state_scalars`);
    /// fields in `vable_fields` use the live object's field instead of a
    /// legacy state slot. The `ref(T)` struct Path `T` is retained so a field
    /// read/write through the ref (`state.<ref_scalar>.<member>`) can emit
    /// `getfield_gc_*`/`setfield_gc_*` with `offset_of!(T, member)` + the
    /// matching `struct_type_id(T)`.
    pub(super) state_ref_scalars: HashMap<String, (usize, syn::Path)>,
    /// State field float scalars: field_name → float_scalar_index.
    pub(super) state_float_scalars: HashMap<String, usize>,
    /// Green-variable expressions for `jit_merge_point` / `promote_greens`.
    ///
    /// Source: `JitInterpConfig.greens` (mod.rs) — the `greens = [...]` list
    /// from the `#[jit_interp]` attribute.  Consumed by A.3.2 (green register
    /// byte list emit) and A.3.5 (promote_greens pre-portal emission).
    pub greens: Vec<Expr>,
    /// Per-green explicit lltype subtype tag (`: str` / `: unicode` / etc.)
    /// from `JitInterpConfig.green_type_tags`.  Lockstep with `greens`;
    /// `None` for an untagged entry.  Consumed by `green_schema()` so the
    /// `JitDriverStaticData::green_args_spec` reflects the upstream
    /// `warmspot.py:663 _green_args_spec` STR/UNICODE distinction
    /// instead of collapsing to `GreenType::Ref`.
    pub green_type_tags: Vec<Option<crate::jit_interp::green_type_tag::GreenTypeTag>>,
    /// Function-parameter greens beyond the conventional `pc` / `program`
    /// pair, in declaration order.  These are real portal arguments, just as
    /// every variable in `JitDriver.greens` is an argument of RPython's
    /// `JIT_ENTER_FUNCTYPE`; body-local greens are deliberately absent.
    pub(super) portal_green_params: Vec<(String, ValueKind)>,
    /// Function-scope `let mut` greens the portal loop does not rebind.
    /// Inputs after `portal_green_params`, so an inlined arm's assignment
    /// moves into a header register (`join_merge`) and `guard_value`
    /// snapshots it. `warmspot.py ll_portal_runner` restores the same
    /// values into the interpreter locals.
    pub(super) loop_carried_greens: Vec<(String, ValueKind)>,
    /// Slice (audit Issue #6) — explicit red declarations.  Source:
    /// `JitInterpConfig.reds` (mod.rs).  Empty = use the default
    /// `[program, pc(+ optional vable)]` candidate list.
    pub(super) reds: Vec<Expr>,
    /// Canonical state-parameter type name. Used by `lower_method_call_value`
    /// to synthesize `<type>::<method>` path lookups for receiver `state`.
    /// Source: `JitInterpConfig.state_type` Ident.
    pub(super) state_type_name: String,
    /// Canonical env-parameter type name. Used by `lower_method_call_value`
    /// to synthesize `<type>::<method>` path lookups for receiver `program`
    /// (the env parameter — convention fixed at the dispatch portal-input
    /// installer below). Source: `JitInterpConfig.env_type` Ident.
    pub(super) env_type_name: String,
    /// Residual helpers that mutate a ref-scalar field, resolved per helper to
    /// `(helper path segments, struct Path, field Ident)`.  When
    /// `lower_config_call_stmt` emits a residual call whose path segments match,
    /// it attaches a write-set `EffectInfo` (`struct_field_write_effect_info`)
    /// naming the field so the optimizer invalidates the cached
    /// `getfield_gc_i`.  Source: `JitInterpConfig.residual_writes`, the struct
    /// `Path` recovered from `state_ref_scalars[ref_scalar]`.
    /// The trailing `bool` is the `[]` suffix: the helper writes the ELEMENTS
    /// of the array the field points at, rather than the field itself.
    pub(super) residual_writes: Vec<(Vec<String>, syn::Path, Ident, bool)>,
    /// A call `<getter>(state.<base>, <int>)` whose function path matches
    /// `getter` AND whose arg0 is the `base` lowers to `getarrayitem_gc_r`
    /// instead of a residual CALL_R.  Source: `JitInterpConfig.pool_arrays`.
    pub(super) pool_arrays: Vec<PoolArrayLowering>,
    /// Ref-kind struct field declarations.  Key = `"StructType::field"`,
    /// value = `(struct_path, field_ident, pointee_path)`.  When the lowerer
    /// encounters a field access and the `(struct, field)` pair matches, it
    /// emits `getfield_gc_r` / `setfield_gc_r` (ref-kind) instead of `_gc_i`.
    pub(super) ref_fields: HashMap<String, (syn::Path, Ident, syn::Path)>,
    /// `array_fields = { Struct::field => ElementType }`, keyed
    /// `"StructLastSegment::field"` → (struct_path, field_ident,
    /// element_path). The field holds an array BASE POINTER, so
    /// `base.field[i]` is a `getfield_gc_r` followed by
    /// `get/setarrayitem_gc_i` — never `arraylen_gc`, since the buffer
    /// carries no header length.
    pub(super) array_fields: HashMap<String, (syn::Path, Ident, syn::Path)>,
    /// `ElementType in Header` entries, same key as `array_fields`.
    /// The value is the header struct; element 0 is its `items` field,
    /// and the element op is `get/setarrayitem_gc_r`.
    pub(super) array_headers: HashMap<String, syn::Path>,
    /// Sub-word integer struct field declarations.  Key = `"StructType::field"`,
    /// value = `(rust_int_type, is_signed)`.  Source:
    /// `JitInterpConfig.int_fields`.  A field listed here registers its real
    /// width and signedness with the struct layout instead of the machine-word
    /// default, which is what makes `descr.is_integer_bounded()` true for it.
    pub(super) int_fields: HashMap<String, (Ident, bool)>,
    /// `int_fields` entries whose type is `f64`. Key = `"StructType::field"`.
    /// A read is `getfield_gc_f` and the layout flag is `FLAG_FLOAT`
    /// (`descr.py` `get_type_flag`).
    pub(super) float_fields: HashSet<String>,
    /// `"StructType::field"` keys some access site asked about while lowering.
    ///
    /// A declared key that never appears here matched no access, so it emitted
    /// nothing — no descr width, and no witness either. That is what makes it
    /// worth reporting rather than merely tidying: an entry that produces no
    /// tokens is also an entry whose claim about the field was never checked,
    /// and the two are indistinguishable from the declaration alone.
    ///
    /// Shared rather than cloned: the config is cloned per lowering entry
    /// point, and a key consulted by any of them counts for all.
    pub(super) consulted_field_keys: std::rc::Rc<std::cell::RefCell<HashSet<String>>>,
    /// Return struct type for ref-returning calls.  Key = canonical func
    /// path segments, value = struct type path.  When a `residual_ref` call
    /// result is bound, the lowerer sets `Binding.struct_type` from this map
    /// so subsequent `result.field` accesses resolve through `ref_fields`.
    pub(super) call_returns: HashMap<Vec<String>, syn::Path>,
    /// Struct types whose `New` allocation should carry the headerless bit.
    pub(super) headerless_structs: std::collections::HashSet<Vec<String>>,
    /// `Outer` canonical segments -> (base field name, base path), from
    /// `inlined_prefix`.  Empty unless a caller declares one, in which case
    /// [`LowererConfig::declaring_struct`] is the identity.
    pub(super) inlined_prefix: HashMap<Vec<String>, (String, syn::Path)>,
    /// Pure function → native IR integer binop aliases.  Key = canonical func
    /// path segments, value = IR opcode name (e.g. "IntAdd").  When
    /// `lower_native_int_binop_call` encounters a call whose path matches a
    /// key, it emits the named IR opcode directly — bypassing the call-policy
    /// machinery.  jtransform.py `_handle_int_special()` parity.
    pub(super) native_int_binops: Vec<(Vec<String>, String)>,
    /// Pure unary tag-small helpers.  Key = canonical func path segments.
    /// `lower_native_tag_small_call` emits `(x << 1) | 1` directly.
    pub(super) native_tag_small: Vec<Vec<String>>,
    /// Unary helpers that return their argument unchanged.  Key = canonical
    /// func path segments.  `lower_native_identity_call` hands the argument's
    /// binding to the consumer and emits nothing.
    pub(super) native_identity: Vec<Vec<String>>,
    /// Source: `JitInterpConfig.split_dispatch`.  When set, the dispatch lowerer
    /// routes pure forward-advancing green-pc arms through the per-arm
    /// sub-JitCode path with a pc-returning `inline_call_<types>_i` instead of
    /// force-inlining them into the dispatch JitCode.  Off → byte-identical.
    pub(super) split_dispatch: bool,
    /// Source: `JitInterpConfig.switch_dispatch`.  When set, the dispatch
    /// lowerer emits one RPython-style `switch/id` over opcode cases instead
    /// of a per-arm guard chain. Off → byte-identical.
    pub(super) switch_dispatch: bool,
}

impl LowererConfig {
    /// Per-bank portal argument counts before a possible virtualizable.
    /// `program` owns r0 and `pc` owns i0; function-parameter greens follow
    /// densely in their respective banks, matching `MIFrame.setup_call`.
    pub(super) fn portal_input_kind_counts(&self) -> (u16, u16, u16) {
        let mut ints = 1u16;
        let mut refs = 1u16;
        let mut floats = 0u16;
        for (_, kind) in self
            .portal_green_params
            .iter()
            .chain(self.loop_carried_greens.iter())
        {
            match kind {
                ValueKind::Int => ints += 1,
                ValueKind::Ref => refs += 1,
                ValueKind::Float => floats += 1,
            }
        }
        (ints, refs, floats)
    }

    /// First ref-bank register available for ref-scalar identity slots.
    /// `MIFrame::setup_call` packs the dispatch JitCode's ref args densely;
    /// the virtualizable, when present, follows the explicit portal greens.
    pub(super) fn ref_identity_base(&self) -> u16 {
        let (_, refs, _) = self.portal_input_kind_counts();
        refs + u16::from(self.vable_var.is_some())
    }

    /// First int-bank register available for scalar/array identity
    /// slots — the int-bank mirror of `ref_identity_base`. `pc` is i0;
    /// portal-parameter greens and loop-carried greens follow
    /// (`portal_input_kind_counts`). An identity slot aliasing one of
    /// those inputs would overwrite the green, so the blackhole's
    /// re-executed jit_merge_point would read a state scalar where it
    /// expects that green.
    pub(super) fn int_identity_base(&self) -> u16 {
        self.portal_input_kind_counts().0
    }

    /// First float-bank register available for float-scalar identity slots.
    /// The current dispatch shapes have no float green arguments, so this is
    /// zero; keep it computed from the schema helper instead of baking call
    /// sites to `0`.
    pub(super) fn float_identity_base(&self) -> u16 {
        self.portal_input_kind_counts().2
    }

    /// One past the last float-bank identity slot.
    pub(super) fn float_identity_end(&self) -> u16 {
        self.state_float_scalars
            .iter()
            .filter(|(name, _)| !self.is_state_vable_field(name))
            .map(|(_, index)| self.float_identity_base() + *index as u16 + 1)
            .max()
            .unwrap_or(0)
    }

    pub(super) fn is_state_vable_field(&self, name: &str) -> bool {
        self.vable_var.as_deref() == Some("state") && self.vable_fields.contains_key(name)
    }

    /// Int scalars that keep an independent identity slot (not redirected
    /// onto the virtualizable). Dense `0..n` indices, so `max(index)+1`
    /// is the slot count. Plain `[int]` arrays follow as ptr then len
    /// (`identity_slot_registers`).
    pub(super) fn plain_int_scalar_count(&self) -> u16 {
        self.state_scalars
            .iter()
            .filter(|(name, _)| !self.is_state_vable_field(name))
            .map(|(_, index)| *index as u16 + 1)
            .max()
            .unwrap_or(0)
    }

    /// Plain `[int]` arrays in `state_arrays` index order.
    pub(super) fn state_array_names_in_order(&self) -> Vec<&str> {
        let mut names: Vec<(&str, usize)> = self
            .state_arrays
            .iter()
            .map(|(name, idx)| (name.as_str(), *idx))
            .collect();
        names.sort_by_key(|(_, idx)| *idx);
        names.into_iter().map(|(name, _)| name).collect()
    }

    /// Two INT identity slots per plain `[int]`: items pointer then length.
    pub(super) fn plain_array_identity_slots(&self) -> u16 {
        (self.state_arrays.len() as u16).saturating_mul(2)
    }

    /// Exclusive end of the int-bank and ref-bank identity-slot ranges
    /// `[int_identity_base, int_end)` / `[ref_identity_base, ref_end)`.
    ///
    /// Mirrors the dispatch JitCode's identity reservation: int identity =
    /// the scalar slots plus the ONE virtualizable identity word (present
    /// whenever the state declares any `[.. ; virt]` array — the
    /// virtualizable is one red, `warmspot.py:538`, and array lengths are
    /// read off the live object, `virtualizable.py:150-153`);
    /// ref identity = the ref scalars. An arm sub-JitCode must reserve the
    /// SAME prefix so its register file spans the identity slots that the
    /// arm body's `load/store_state_field` ops address (`MIFrame.setup_call`
    /// plants them on the portal; the arm addresses the same range).
    /// Returns `0` for a bank with no identity slots so the caller's `.max()`
    /// floor is inert there.
    pub(super) fn split_identity_reg_ends(&self) -> (u16, u16) {
        // Scalars redirected onto the virtualizable (`vable_fields`) are
        // not independent identity slots. Counting them here would
        // reserve registers the split arm never uses and can overflow
        // the 256-register JitCode ceiling.
        let int_scalars = self.plain_int_scalar_count();
        // A plain `[int]` is ptr then len (`jtransform.py`
        // `rewrite_op_getarrayitem` raw arm: the Vec / slice element
        // buffer is a raw `{ptr, len}` view).
        // `warmspot.py` reds are a fixed name list, never one register
        // per cell.
        let array_slots = self.plain_array_identity_slots();
        let int_end = self
            .int_identity_base()
            .saturating_add(int_scalars)
            .saturating_add(array_slots)
            .min(256);
        let ref_scalars = self
            .state_ref_scalars
            .iter()
            .filter(|(name, _)| !self.is_state_vable_field(name))
            .map(|(_, (index, _))| *index as u16 + 1)
            .max()
            .unwrap_or(0);
        let ref_end = if ref_scalars == 0 {
            0
        } else {
            self.ref_identity_base() + ref_scalars
        };
        (int_end, ref_end)
    }

    /// Portal identity-slot registers that `load/store_state_field*` address.
    ///
    /// `jtransform.py handle_regular_call` emits `inline_call_*` then `-live-`.
    /// `MIFrame.get_list_of_active_boxes(in_a_call=True)` reads that marker
    /// on the caller frame, so the reds that live in the portal identity
    /// prefix must be force-alive there. A later `load_state_field` already
    /// keeps a slot live; a working-register reuse of the same value does
    /// not, and the slot would drop from the parent snapshot.
    ///
    /// A plain `[int]` is ptr then len after the int scalars.
    pub(super) fn identity_slot_registers(&self) -> Vec<Register> {
        let mut regs = Vec::new();
        let int_base = self.int_identity_base();
        let mut int_scalars: Vec<(usize, &str)> = self
            .state_scalars
            .iter()
            .filter(|(name, _)| !self.is_state_vable_field(name))
            .map(|(name, index)| (*index, name.as_str()))
            .collect();
        int_scalars.sort_by_key(|(index, _)| *index);
        for (index, _) in &int_scalars {
            regs.push(Register::int(int_base + *index as u16));
        }
        let mut offset = self.plain_int_scalar_count();
        for _ in self.state_array_names_in_order() {
            regs.push(Register::int(int_base + offset));
            offset = offset.saturating_add(1);
            regs.push(Register::int(int_base + offset));
            offset = offset.saturating_add(1);
        }
        let ref_base = self.ref_identity_base();
        let mut ref_scalars: Vec<(usize, &str)> = self
            .state_ref_scalars
            .iter()
            .filter(|(name, _)| !self.is_state_vable_field(name))
            .map(|(name, (index, _))| (*index, name.as_str()))
            .collect();
        ref_scalars.sort_by_key(|(index, _)| *index);
        for (index, _) in &ref_scalars {
            regs.push(Register::ref_(ref_base + *index as u16));
        }
        let float_base = self.float_identity_base();
        let mut float_scalars: Vec<(usize, &str)> = self
            .state_float_scalars
            .iter()
            .filter(|(name, _)| !self.is_state_vable_field(name))
            .map(|(name, index)| (*index, name.as_str()))
            .collect();
        float_scalars.sort_by_key(|(index, _)| *index);
        for (index, _) in &float_scalars {
            regs.push(Register::float(float_base + *index as u16));
        }
        regs
    }

    /// Per-kind counts of the identity-slot reds `resolve_reds` appends
    /// after the named portal reds (`identity_slot_registers`).
    pub(super) fn identity_scalar_red_kind_counts(&self) -> (usize, usize, usize) {
        let mut counts = (0usize, 0usize, 0usize);
        for reg in self.identity_slot_registers() {
            match reg.kind {
                BindingKind::Int => counts.0 += 1,
                BindingKind::Ref => counts.1 += 1,
                BindingKind::Float => counts.2 += 1,
            }
        }
        counts
    }
}

pub(super) const MAX_HELPER_CALL_ARITY: usize = 16;

pub(super) fn classify_virtualizable_hint_syn_path(path: &Path) -> Option<VirtualizableHintKind> {
    let segments = path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect::<Vec<_>>();
    classify_virtualizable_hint_segments(segments.iter().map(String::as_str))
}

pub(crate) struct InlineHelperJitCode {
    pub body: TokenStream,
    pub return_reg: u16,
    pub return_kind: InlineReturnKind,
    /// Helper-side per-marker liveness prebuild tokens. Threaded into the
    /// parent's `__prebuild_jitcode_liveness_*` so RPython
    /// `pyjitpl.py finish_setup`'s "all `-live-` entries land in
    /// `asm.all_liveness` before the snapshot" invariant is preserved
    /// when the helper is invoked at trace time. Without this thread, the
    /// helper's `JitCodeBuilder::finalize_liveness(asm)` at trace time
    /// would register triples the snapshot didn't see, growing
    /// `staticdata.liveness_info` past the install-time freeze and
    /// tripping the `__trace_*` snapshot-invariant assertion.
    pub liveness_prebuild: TokenStream,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InlineReturnKind {
    Int,
    Ref,
    Float,
    Void,
}

#[derive(Clone)]
pub(super) enum CallPolicySpec {
    Explicit(crate::jit_interp::CallPolicyKind),
    Infer,
}

#[derive(Clone, Copy)]
pub(super) enum InferenceFailureMode {
    ReturnNone,
    Panic,
}

#[derive(Clone, Copy)]
pub(super) enum ValueKind {
    Int,
    Ref,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CondCallEffectSlot {
    CanRaise,
    /// `EF_CANNOT_RAISE` — `call.py getcalldescr`'s non-elidable
    /// `else` branch.  Selected by a `residual_*_cannot_raise` policy
    /// when the producer statically knows the callee cannot raise.
    CannotRaise,
    ElidableCanRaise,
    ElidableCannotRaise,
    ElidableOrMemerror,
    LoopInvariant,
}

impl CondCallEffectSlot {
    pub(super) fn token(self) -> TokenStream {
        match self {
            Self::CanRaise => quote! { majit_metainterp::EffectInfoSlot::CanRaise },
            Self::CannotRaise => quote! { majit_metainterp::EffectInfoSlot::CannotRaise },
            Self::ElidableCanRaise => quote! { majit_metainterp::EffectInfoSlot::ElidableCanRaise },
            Self::ElidableCannotRaise => {
                quote! { majit_metainterp::EffectInfoSlot::ElidableCannotRaise }
            }
            Self::ElidableOrMemerror => {
                quote! { majit_metainterp::EffectInfoSlot::ElidableOrMemerror }
            }
            Self::LoopInvariant => quote! { majit_metainterp::EffectInfoSlot::LoopInvariant },
        }
    }

    pub(super) fn can_raise(self) -> bool {
        matches!(
            self,
            Self::CanRaise | Self::ElidableCanRaise | Self::ElidableOrMemerror
        )
    }

    pub(super) fn is_elidable(self) -> bool {
        matches!(
            self,
            Self::ElidableCanRaise | Self::ElidableCannotRaise | Self::ElidableOrMemerror
        )
    }

    /// Emit the `EffectInfoSlot` token for a statically-known wrapped
    /// `CallPolicyKind`.  Used by the `*Wrapped` lowering arms so the
    /// registered call-target descr carries the real effect classification
    /// (`call.py getcalldescr`) rather than a blanket `CanRaise`.
    /// Falls back to `CanRaise` for `MayForce` / `Inline*` kinds whose
    /// conditional-call slot is `None` in `call_policy_effect_slot`; the
    /// actual call surface dispatches them through dedicated
    /// `call_may_force_*` / inline-helper paths that ignore the registered
    /// slot.
    pub(super) fn for_wrapped_kind(kind: crate::jit_interp::CallPolicyKind) -> TokenStream {
        call_policy_effect_slot(kind)
            .map(|slot| slot.token())
            .unwrap_or_else(|| quote! { majit_metainterp::EffectInfoSlot::CanRaise })
    }

    /// Emit a runtime `match __policy { ... }` expression that resolves the
    /// `EffectInfoSlot` from the helper's `_jit_helper_policy` byte.  Used
    /// by the `Infer` lowering paths where the policy kind is only known at
    /// runtime (`call.py getcalldescr` analyzer chain executed on
    /// the live byte).
    pub(super) fn slot_from_policy_tokens() -> TokenStream {
        quote! {
            match __policy {
                #VOID_DONT_LOOK_INSIDE | #INT_DONT_LOOK_INSIDE | #REF_DONT_LOOK_INSIDE
                | #VOID_MAY_FORCE | #INT_MAY_FORCE | #REF_MAY_FORCE => {
                    majit_metainterp::EffectInfoSlot::CanRaise
                }
                #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE
                | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE
                | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                    majit_metainterp::EffectInfoSlot::CannotRaise
                }
                #INT_ELIDABLE | #REF_ELIDABLE => {
                    majit_metainterp::EffectInfoSlot::ElidableCanRaise
                }
                #VOID_LOOP_INVARIANT | #INT_LOOP_INVARIANT | #REF_LOOP_INVARIANT => {
                    majit_metainterp::EffectInfoSlot::LoopInvariant
                }
                #INT_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_CANNOT_RAISE => {
                    majit_metainterp::EffectInfoSlot::ElidableCannotRaise
                }
                #INT_ELIDABLE_OR_MEMERROR | #REF_ELIDABLE_OR_MEMERROR => {
                    majit_metainterp::EffectInfoSlot::ElidableOrMemerror
                }
                _ => majit_metainterp::EffectInfoSlot::CanRaise,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CallResultKind {
    Void,
    Int,
    Ref,
    Float,
}

impl ValueKind {
    pub(super) fn from_ident(ident: &Ident) -> Self {
        match ident.to_string().as_str() {
            "ref" => Self::Ref,
            "float" => Self::Float,
            _ => Self::Int,
        }
    }
}

pub(super) fn call_policy_effect_slot(
    kind: crate::jit_interp::CallPolicyKind,
) -> Option<CondCallEffectSlot> {
    use crate::jit_interp::CallPolicyKind as K;
    match kind {
        K::ResidualVoid
        | K::ResidualVoidWrapped
        | K::ResidualInt
        | K::ResidualIntWrapped
        | K::ResidualRef
        | K::NurseryAllocRef
        | K::AllocRef
        | K::ResidualRefWrapped
        | K::ResidualFloatWrapped => Some(CondCallEffectSlot::CanRaise),

        K::ResidualVoidCannotRaise
        | K::ResidualVoidCannotRaiseWrapped
        | K::ResidualIntCannotRaise
        | K::ResidualIntCannotRaiseWrapped
        | K::ResidualRefCannotRaiseWrapped
        | K::ResidualFloatCannotRaiseWrapped => Some(CondCallEffectSlot::CannotRaise),

        K::LoopInvariantVoid
        | K::LoopInvariantVoidWrapped
        | K::LoopInvariantInt
        | K::LoopInvariantIntWrapped
        | K::LoopInvariantRefWrapped
        | K::LoopInvariantFloatWrapped => Some(CondCallEffectSlot::LoopInvariant),

        K::ElidableInt
        | K::ElidableIntWrapped
        | K::ElidableRefWrapped
        | K::ElidableFloatWrapped => Some(CondCallEffectSlot::ElidableCanRaise),
        K::ElidableIntCannotRaise
        | K::ElidableIntCannotRaiseWrapped
        | K::ElidableRefCannotRaiseWrapped
        | K::ElidableFloatCannotRaiseWrapped => Some(CondCallEffectSlot::ElidableCannotRaise),
        K::ElidableIntOrMemerror
        | K::ElidableIntOrMemerrorWrapped
        | K::ElidableRefOrMemerrorWrapped
        | K::ElidableFloatOrMemerrorWrapped => Some(CondCallEffectSlot::ElidableOrMemerror),

        K::MayForceVoid
        | K::MayForceVoidWrapped
        | K::MayForceInt
        | K::MayForceIntWrapped
        | K::MayForceRefWrapped
        | K::MayForceFloatWrapped
        | K::InlineInt
        | K::InlineRef
        | K::InlineFloat
        | K::InlineVoid
        | K::InlinePipelineInt
        | K::InlinePipelineRef
        | K::InlinePipelineFloat
        | K::InlinePipelineVoid
        | K::ConcreteOnlyVoid => None,
    }
}

/// Whether an explicit residual/elidable lowering arm for `kind` must be
/// followed by a `-live-` marker.  `jtransform.py:467-471` appends `-live-`
/// after a residual `call_*` only when `calldescr_canraise`;
/// `jtransform.py:480-482` appends it after every `inline_call_*`.  The
/// inline arms emit their own trailing marker (`inline_call_tokens`'
/// `post_live`), so inline kinds return `false`; otherwise the calldescr's
/// can-raise classification decides.  MayForce has no
/// conditional-call slot but force / may raise, so they keep the marker.
pub(super) fn explicit_call_emits_post_live(kind: crate::jit_interp::CallPolicyKind) -> bool {
    if binding_kind_for_inline_policy(kind).is_some()
        || matches!(
            kind,
            crate::jit_interp::CallPolicyKind::InlineVoid
                | crate::jit_interp::CallPolicyKind::InlinePipelineVoid
        )
    {
        return false;
    }
    if matches!(kind, crate::jit_interp::CallPolicyKind::ConcreteOnlyVoid) {
        return false;
    }
    call_policy_effect_slot(kind)
        .map(CondCallEffectSlot::can_raise)
        .unwrap_or(true)
}

pub(super) fn call_policy_result_kind(
    kind: crate::jit_interp::CallPolicyKind,
) -> Option<CallResultKind> {
    use crate::jit_interp::CallPolicyKind as K;
    match kind {
        K::ResidualVoid
        | K::ResidualVoidWrapped
        | K::ResidualVoidCannotRaise
        | K::ResidualVoidCannotRaiseWrapped
        | K::MayForceVoid
        | K::MayForceVoidWrapped
        | K::LoopInvariantVoid
        | K::LoopInvariantVoidWrapped
        | K::InlineVoid
        | K::InlinePipelineVoid
        | K::ConcreteOnlyVoid => Some(CallResultKind::Void),

        K::ResidualInt
        | K::ResidualIntWrapped
        | K::ResidualIntCannotRaise
        | K::ResidualIntCannotRaiseWrapped
        | K::MayForceInt
        | K::MayForceIntWrapped
        | K::LoopInvariantInt
        | K::LoopInvariantIntWrapped
        | K::ElidableInt
        | K::ElidableIntWrapped
        | K::ElidableIntCannotRaise
        | K::ElidableIntCannotRaiseWrapped
        | K::ElidableIntOrMemerror
        | K::ElidableIntOrMemerrorWrapped
        | K::InlineInt
        | K::InlinePipelineInt => Some(CallResultKind::Int),

        K::ResidualRef
        | K::NurseryAllocRef
        | K::AllocRef
        | K::ResidualRefWrapped
        | K::ResidualRefCannotRaiseWrapped
        | K::MayForceRefWrapped
        | K::LoopInvariantRefWrapped
        | K::ElidableRefWrapped
        | K::ElidableRefCannotRaiseWrapped
        | K::ElidableRefOrMemerrorWrapped
        | K::InlineRef
        | K::InlinePipelineRef => Some(CallResultKind::Ref),

        K::ResidualFloatWrapped
        | K::ResidualFloatCannotRaiseWrapped
        | K::MayForceFloatWrapped
        | K::LoopInvariantFloatWrapped
        | K::ElidableFloatWrapped
        | K::ElidableFloatCannotRaiseWrapped
        | K::ElidableFloatOrMemerrorWrapped
        | K::InlineFloat
        | K::InlinePipelineFloat => Some(CallResultKind::Float),
    }
}

pub(super) fn call_policy_is_wrapped(kind: crate::jit_interp::CallPolicyKind) -> bool {
    use crate::jit_interp::CallPolicyKind as K;
    matches!(
        kind,
        K::ResidualVoidWrapped
            | K::ResidualVoidCannotRaiseWrapped
            | K::MayForceVoidWrapped
            | K::LoopInvariantVoidWrapped
            | K::ResidualIntWrapped
            | K::ResidualIntCannotRaiseWrapped
            | K::MayForceIntWrapped
            | K::LoopInvariantIntWrapped
            | K::ElidableIntWrapped
            | K::ElidableIntCannotRaiseWrapped
            | K::ElidableIntOrMemerrorWrapped
            | K::ResidualRef
            | K::ResidualRefWrapped
            | K::ResidualRefCannotRaiseWrapped
            | K::MayForceRefWrapped
            | K::LoopInvariantRefWrapped
            | K::ElidableRefWrapped
            | K::ElidableRefCannotRaiseWrapped
            | K::ElidableRefOrMemerrorWrapped
            | K::ResidualFloatWrapped
            | K::ResidualFloatCannotRaiseWrapped
            | K::MayForceFloatWrapped
            | K::LoopInvariantFloatWrapped
            | K::ElidableFloatWrapped
            | K::ElidableFloatCannotRaiseWrapped
            | K::ElidableFloatOrMemerrorWrapped
    )
}

pub(super) fn call_result_matches_binding(
    result_kind: CallResultKind,
    binding_kind: BindingKind,
) -> bool {
    matches!(
        (result_kind, binding_kind),
        (CallResultKind::Int, BindingKind::Int)
            | (CallResultKind::Ref, BindingKind::Ref)
            | (CallResultKind::Float, BindingKind::Float)
    )
}

/// Build the runtime guard that wraps `live_placeholder*` for an
/// inferred-policy callee.  See [`LiveMarkerCondition`] for the
/// TODO rationale and the convergence path that retires this wrapper.
pub(super) fn inferred_policy_live_condition(func: &Expr, can_raise_codes: &[u8]) -> TokenStream {
    let policy_path =
        helper_policy_path(func).expect("inferred helper policy requires a path expression");
    let patterns = can_raise_codes
        .iter()
        .copied()
        .map(|code| quote! { #code })
        .collect::<Vec<_>>();
    if patterns.is_empty() {
        return quote! { false };
    }
    quote! {{
        let (__policy, _, _, _, _, _) = #policy_path();
        matches!(__policy, #(#patterns)|*)
    }}
}

pub(super) fn inferred_conditional_call_policy_check(func_args_empty: bool) -> TokenStream {
    let loop_invariant_arm = if func_args_empty {
        quote! { #VOID_LOOP_INVARIANT => {} }
    } else {
        quote! {
            #VOID_LOOP_INVARIANT => panic!(
                "conditional_call!: arguments not supported for loop-invariant function",
            )
        }
    };
    quote! {
        match __policy {
            // Void-return, non-forcing calldescrs accepted by
            // jtransform.py:1677.  `VOID_DONT_LOOK_INSIDE_CANNOT_RAISE`
            // is the EF_CANNOT_RAISE void surface; jtransform's gate
            // accepts both EF_CAN_RAISE and EF_CANNOT_RAISE — the
            // latter just skips the trailing `-live-` per
            // `jtransform.py:1681 calldescr_canraise`.
            #VOID_DONT_LOOK_INSIDE | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE => {},
            #loop_invariant_arm,
            // Void-return but rejected by jtransform.py:1677's
            // `check_forces_virtual_or_virtualizable` gate or by the
            // release-gil structural surface.
            #VOID_MAY_FORCE => panic!(
                "conditional_call! cannot dispatch MayForce callees",
            ),
            // PyPy `call.py:getcalldescr` checks actual return type before
            // effect flags; `conditional_call!` is the void-result opcode.
            #INT_DONT_LOOK_INSIDE | #INT_ELIDABLE | #INT_MAY_FORCE
            | #INT_LOOP_INVARIANT | #INT_ELIDABLE_CANNOT_RAISE | #INT_ELIDABLE_OR_MEMERROR
            | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE
            | #REF_ELIDABLE | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR
            | #REF_LOOP_INVARIANT | #REF_DONT_LOOK_INSIDE | #REF_MAY_FORCE => panic!(
                "conditional_call! requires a void-return helper policy",
            ),
            _ => panic!(
                "conditional_call! could not infer a PyPy-compatible helper policy",
            ),
        }
    }
}

pub(super) fn inferred_conditional_call_value_policy_check(
    value_kind: BindingKind,
    func_args_empty: bool,
) -> TokenStream {
    match value_kind {
        BindingKind::Int => {
            let loop_invariant_arm = if func_args_empty {
                quote! { #INT_LOOP_INVARIANT => {} }
            } else {
                quote! {
                    #INT_LOOP_INVARIANT => panic!(
                        "conditional_call_elidable!: arguments not supported for loop-invariant function",
                    )
                }
            };
            quote! {
                match __policy {
                    // INT_DONT_LOOK_INSIDE_CANNOT_RAISE: int residual
                    // EF_CANNOT_RAISE accepted on the int value branch
                    // per `call.py:300`.
                    #INT_DONT_LOOK_INSIDE | #INT_ELIDABLE | #INT_ELIDABLE_CANNOT_RAISE
                    | #INT_ELIDABLE_OR_MEMERROR | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => {},
                    #loop_invariant_arm,
                    #INT_MAY_FORCE => panic!(
                        "conditional_call_elidable! cannot dispatch MayForce callees",
                    ),
                    // VOID_DONT_LOOK_INSIDE_CANNOT_RAISE (28) and
                    // REF_DONT_LOOK_INSIDE_CANNOT_RAISE (30) are wrong
                    // result kind for the int value branch.
                    #VOID_DONT_LOOK_INSIDE | #VOID_MAY_FORCE
                    | #VOID_LOOP_INVARIANT | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE
                    | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE | #REF_ELIDABLE
                    | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR
                    | #REF_LOOP_INVARIANT | #REF_DONT_LOOK_INSIDE | #REF_MAY_FORCE => panic!(
                        "conditional_call_elidable! value/result kind mismatch for inferred helper policy",
                    ),
                    _ => panic!(
                        "conditional_call_elidable! could not infer a PyPy-compatible helper policy",
                    ),
                }
            }
        }
        BindingKind::Ref => {
            let loop_invariant_arm = if func_args_empty {
                quote! { #REF_LOOP_INVARIANT => {} }
            } else {
                quote! {
                    #REF_LOOP_INVARIANT => panic!(
                        "conditional_call_elidable!: arguments not supported for loop-invariant function",
                    )
                }
            };
            quote! {
                match __policy {
                    // REF_DONT_LOOK_INSIDE_CANNOT_RAISE: ref residual
                    // EF_CANNOT_RAISE accepted on the ref value branch.
                    #REF_ELIDABLE | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR
                    | #REF_DONT_LOOK_INSIDE | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE => {},
                    #loop_invariant_arm,
                    #REF_MAY_FORCE => panic!(
                        "conditional_call_elidable! cannot dispatch MayForce callees",
                    ),
                    // VOID_DONT_LOOK_INSIDE_CANNOT_RAISE (28) and
                    // INT_DONT_LOOK_INSIDE_CANNOT_RAISE (29) are wrong
                    // result kind for the ref value branch.
                    #VOID_DONT_LOOK_INSIDE | #INT_DONT_LOOK_INSIDE | #INT_ELIDABLE
                    | #VOID_MAY_FORCE | #INT_MAY_FORCE
                    | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE
                    | #VOID_LOOP_INVARIANT | #INT_LOOP_INVARIANT | #INT_ELIDABLE_CANNOT_RAISE
                    | #INT_ELIDABLE_OR_MEMERROR => panic!(
                        "conditional_call_elidable! value/result kind mismatch for inferred helper policy",
                    ),
                    _ => panic!(
                        "conditional_call_elidable! could not infer a PyPy-compatible helper policy",
                    ),
                }
            }
        }
        BindingKind::Float => quote! {
            panic!("Conditional call does not support floats");
        },
    }
}

pub(super) fn inferred_record_known_result_policy_check(result_kind: BindingKind) -> TokenStream {
    match result_kind {
        BindingKind::Int => quote! {
            match __policy {
                #INT_ELIDABLE | #INT_ELIDABLE_CANNOT_RAISE | #INT_ELIDABLE_OR_MEMERROR => {},
                // INT_DONT_LOOK_INSIDE_CANNOT_RAISE (29): int residual
                // EF_CANNOT_RAISE — not elidable, rejected here.
                #INT_DONT_LOOK_INSIDE | #INT_MAY_FORCE | #INT_LOOP_INVARIANT
                | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => panic!(
                    "record_known_result! requires an elidable helper policy",
                ),
                #VOID_DONT_LOOK_INSIDE | #VOID_MAY_FORCE
                | #VOID_LOOP_INVARIANT | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE
                | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE | #REF_ELIDABLE
                | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR
                | #REF_LOOP_INVARIANT | #REF_DONT_LOOK_INSIDE | #REF_MAY_FORCE => panic!(
                    "record_known_result! result kind mismatch for inferred helper policy",
                ),
                _ => panic!(
                    "record_known_result! could not infer a PyPy-compatible helper policy",
                ),
            }
        },
        BindingKind::Ref => quote! {
            match __policy {
                #REF_ELIDABLE | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR => {},
                // REF_DONT_LOOK_INSIDE_CANNOT_RAISE (30): ref residual
                // EF_CANNOT_RAISE — not elidable, rejected here.
                #REF_LOOP_INVARIANT | #REF_DONT_LOOK_INSIDE | #REF_MAY_FORCE
                | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE => panic!(
                    "record_known_result! requires an elidable helper policy",
                ),
                #VOID_DONT_LOOK_INSIDE | #INT_DONT_LOOK_INSIDE | #INT_ELIDABLE
                | #VOID_MAY_FORCE | #INT_MAY_FORCE
                | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE
                | #VOID_LOOP_INVARIANT | #INT_LOOP_INVARIANT | #INT_ELIDABLE_CANNOT_RAISE
                | #INT_ELIDABLE_OR_MEMERROR => panic!(
                    "record_known_result! result kind mismatch for inferred helper policy",
                ),
                _ => panic!(
                    "record_known_result! could not infer a PyPy-compatible helper policy",
                ),
            }
        },
        BindingKind::Float => quote! {
            panic!("record_known_result does not support floats");
        },
    }
}

/// Build the `int_fields` lookup: key = `"StructLastSegment::field"`.
fn field_entry_key(struct_type: &syn::Path, field: &Ident) -> String {
    let struct_name = struct_type
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();
    format!("{}::{}", struct_name, field)
}

fn int_fields_map(
    int_fields: &[crate::jit_interp::IntFieldEntry],
) -> HashMap<String, (Ident, bool)> {
    int_fields
        .iter()
        .filter(|entry| !entry.is_float())
        .map(|entry| {
            let key = field_entry_key(&entry.struct_type, &entry.field);
            // Validated at parse time.
            (
                key,
                (entry.int_type.clone(), entry.is_signed().unwrap_or(true)),
            )
        })
        .collect()
}

fn float_fields_set(int_fields: &[crate::jit_interp::IntFieldEntry]) -> HashSet<String> {
    int_fields
        .iter()
        .filter(|entry| entry.is_float())
        .map(|entry| field_entry_key(&entry.struct_type, &entry.field))
        .collect()
}

/// Key `array_fields` entries the same way `ref_fields` are keyed —
/// `"StructLastSegment::field"` -> `(struct_path, field_ident, element_path)` —
/// so the lowerer resolves an array field off a binding's `struct_type`
/// exactly as it resolves a ref field.
fn array_field_key(entry: &crate::jit_interp::ArrayFieldEntry) -> String {
    let struct_name = entry
        .struct_type
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();
    format!("{}::{}", struct_name, entry.field)
}

fn build_array_fields_map(
    array_fields: &[crate::jit_interp::ArrayFieldEntry],
) -> HashMap<String, (syn::Path, Ident, syn::Path)> {
    array_fields
        .iter()
        .map(|entry| {
            (
                array_field_key(entry),
                (
                    entry.struct_type.clone(),
                    entry.field.clone(),
                    entry.element_type.clone(),
                ),
            )
        })
        .collect()
}

fn build_array_headers_map(
    array_fields: &[crate::jit_interp::ArrayFieldEntry],
) -> HashMap<String, syn::Path> {
    array_fields
        .iter()
        .filter_map(|entry| {
            entry
                .header
                .clone()
                .map(|header| (array_field_key(entry), header))
        })
        .collect()
}

impl LowererConfig {
    pub fn inline_helper(
        ref_fields: &[crate::jit_interp::RefFieldEntry],
        array_fields: &[crate::jit_interp::ArrayFieldEntry],
        int_fields: &[crate::jit_interp::IntFieldEntry],
        native_int_binops: &[(syn::Path, syn::Ident)],
        native_tag_small: &[syn::Path],
        native_identity: &[syn::Path],
        headerless_structs: &[syn::Path],
        inlined_prefix: &[crate::jit_interp::InlinedPrefixEntry],
    ) -> Self {
        let array_fields_map = build_array_fields_map(array_fields);
        let array_headers_map = build_array_headers_map(array_fields);
        let ref_fields_map: HashMap<String, (syn::Path, Ident, syn::Path)> = ref_fields
            .iter()
            .map(|entry| {
                let struct_name = entry
                    .struct_type
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                let key = format!("{}::{}", struct_name, entry.field);
                (
                    key,
                    (
                        entry.struct_type.clone(),
                        entry.field.clone(),
                        entry.pointee_type.clone(),
                    ),
                )
            })
            .collect();
        Self {
            io_shims: Vec::new(),
            calls: Vec::new(),
            auto_calls: false,
            vable_var: None,
            vable_input_ref_reg: None,
            vable_fields: HashMap::new(),
            vable_arrays: HashMap::new(),
            state_scalars: HashMap::new(),
            state_arrays: HashMap::new(),
            state_virt_arrays: HashMap::new(),
            state_ref_scalars: HashMap::new(),
            state_float_scalars: HashMap::new(),
            greens: Vec::new(),
            green_type_tags: Vec::new(),
            portal_green_params: Vec::new(),
            loop_carried_greens: Vec::new(),
            reds: Vec::new(),
            state_type_name: String::new(),
            env_type_name: String::new(),
            residual_writes: Vec::new(),
            pool_arrays: Vec::new(),
            ref_fields: ref_fields_map,
            array_fields: array_fields_map,
            array_headers: array_headers_map,
            int_fields: int_fields_map(int_fields),
            float_fields: float_fields_set(int_fields),
            consulted_field_keys: Default::default(),
            call_returns: HashMap::new(),
            headerless_structs: headerless_structs
                .iter()
                .map(canonical_path_segments)
                .collect(),
            inlined_prefix: inlined_prefix_map(inlined_prefix),
            native_int_binops: native_int_binops
                .iter()
                .map(|(path, op)| (canonical_path_segments(path), op.to_string()))
                .collect(),
            native_tag_small: native_tag_small
                .iter()
                .map(canonical_path_segments)
                .collect(),
            native_identity: native_identity
                .iter()
                .map(canonical_path_segments)
                .collect(),
            split_dispatch: false,
            switch_dispatch: false,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The argument order is the stable JIT IR/descriptor or generated-interpreter ABI shape; grouping it into a Rust-only options object would obscure opcode-field correspondence and macro call-site parity"
    )]
    pub(super) fn new(
        io_shims: &[(Path, Ident)],
        calls: &[crate::jit_interp::CallEntry],
        auto_calls: bool,
        vable_decl: Option<&crate::jit_interp::VirtualizableDecl>,
        state_fields_cfg: Option<&crate::jit_interp::StateFieldsConfig>,
        greens: &[Expr],
        green_type_tags: &[Option<crate::jit_interp::green_type_tag::GreenTypeTag>],
        portal_green_params: &[(String, ValueKind)],
        reds: &[Expr],
        state_type: &Ident,
        env_type: &Ident,
        residual_writes: &[crate::jit_interp::ResidualWriteEntry],
        pool_arrays: &[crate::jit_interp::PoolArrayEntry],
        ref_fields: &[crate::jit_interp::RefFieldEntry],
        array_fields: &[crate::jit_interp::ArrayFieldEntry],
        int_fields: &[crate::jit_interp::IntFieldEntry],
        call_returns: &[(Path, Path)],
        headerless_structs: &[Path],
        inlined_prefix: &[crate::jit_interp::InlinedPrefixEntry],
        native_int_binops: &[(Path, Ident)],
        native_tag_small: &[Path],
        native_identity: &[Path],
        split_dispatch: bool,
        switch_dispatch: bool,
    ) -> Self {
        let io_shims = io_shims
            .iter()
            .map(|(p, s)| (canonical_path_segments(p), s.clone()))
            .collect();
        let calls = calls
            .iter()
            .map(|entry| {
                let spec = match entry.policy {
                    Some(kind) => CallPolicySpec::Explicit(kind),
                    None => CallPolicySpec::Infer,
                };
                (canonical_path_segments(&entry.path), spec)
            })
            .collect();
        let (mut vable_var, mut vable_input_ref_reg, mut vable_fields, mut vable_arrays) =
            if let Some(decl) = vable_decl {
                let var = Some(decl.var_name.to_string());
                let fields = decl
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        (
                            f.name.to_string(),
                            (i, ValueKind::from_ident(&f.field_type)),
                        )
                    })
                    .collect();
                let arrays = decl
                    .arrays
                    .iter()
                    .enumerate()
                    .map(|(i, a)| (a.name.to_string(), (i, ValueKind::from_ident(&a.item_type))))
                    .collect();
                (var, Some(0), fields, arrays)
            } else {
                (None, None, HashMap::new(), HashMap::new())
            };
        let (
            state_scalars,
            state_arrays,
            state_virt_arrays,
            state_ref_scalars,
            state_float_scalars,
        ) = if let Some(sf) = state_fields_cfg {
            use crate::jit_interp::StateFieldKind;
            let mut scalars = HashMap::new();
            let mut arrays = HashMap::new();
            let mut virt_arrays = HashMap::new();
            let mut ref_scalars = HashMap::new();
            let mut float_scalars = HashMap::new();
            let mut scalar_idx = 0usize;
            let mut array_idx = 0usize;
            let mut virt_array_idx = 0usize;
            let mut ref_scalar_idx = 0usize;
            let mut float_scalar_idx = 0usize;
            for f in &sf.fields {
                match &f.kind {
                    StateFieldKind::Scalar { ir_type, .. } => {
                        if ir_type == "float" {
                            float_scalars.insert(f.name.to_string(), float_scalar_idx);
                            float_scalar_idx += 1;
                        } else {
                            scalars.insert(f.name.to_string(), scalar_idx);
                            scalar_idx += 1;
                        }
                    }
                    StateFieldKind::Array(_) => {
                        arrays.insert(f.name.to_string(), array_idx);
                        array_idx += 1;
                    }
                    StateFieldKind::VirtArray(item_type) => {
                        virt_arrays.insert(
                            f.name.to_string(),
                            (virt_array_idx, ValueKind::from_ident(item_type)),
                        );
                        virt_array_idx += 1;
                    }
                    // Opaque fields are not registered in any index map —
                    // the lowering layer must not see them as state slots.
                    StateFieldKind::Opaque(_) => {}
                    // ref(T) scalars get a separate 0-based index space;
                    // they lower to the ref register bank. Retain the
                    // struct Path `T` so a field access through the ref
                    // can resolve `offset_of!(T, member)` + struct type_id.
                    StateFieldKind::Ref(p) => {
                        ref_scalars.insert(f.name.to_string(), (ref_scalar_idx, p.clone()));
                        ref_scalar_idx += 1;
                    }
                    // `str` is a ref inputarg with no heap class. The path is
                    // only stored so the slot shares `state_ref_scalars`.
                    StateFieldKind::Str => {
                        let path: syn::Path = syn::parse_quote!(rpy_string);
                        ref_scalars.insert(f.name.to_string(), (ref_scalar_idx, path));
                        ref_scalar_idx += 1;
                    }
                }
            }
            (scalars, arrays, virt_arrays, ref_scalars, float_scalars)
        } else {
            (
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            )
        };
        // State-field `[int; virt]` arrays converge onto the standard
        // virtualizable path: the state binding is the vable identity var and
        // each virt array becomes a `vable_arrays` entry under its virt index.
        // The dispatch lowerer overrides the identity ref register to 1 (the
        // argbox order is `program`(ref reg 0), then the pushed vable identity
        // → ref reg 1; see `with_vable_input_ref_reg(1)` in codegen_trace).
        // `state_virt_arrays` stays populated; the raw-varray lowering branch
        // is retired in a later task.
        if vable_var.is_none() && !state_virt_arrays.is_empty() {
            vable_var = Some("state".to_string());
            vable_input_ref_reg = Some(1);
            for (name, &(idx, kind)) in &state_virt_arrays {
                vable_arrays.insert(name.clone(), (idx, kind));
            }
            // jtransform.py rewrite_op_getfield: all redirected fields use
            // the live virtualizable argument, including in inline callees.
            for f in &state_fields_cfg.unwrap().fields {
                let kind = match &f.kind {
                    crate::jit_interp::StateFieldKind::Scalar { ir_type, .. } => {
                        ValueKind::from_ident(ir_type)
                    }
                    crate::jit_interp::StateFieldKind::Ref(_) => ValueKind::Ref,
                    _ => continue,
                };
                let index = vable_fields.len();
                vable_fields.insert(f.name.to_string(), (index, kind));
            }
        }
        // Fail closed: a `residual_writes` ref_scalar or a `pool_arrays` name
        // that does not resolve to a declared `ref(_)` state field would
        // otherwise silently drop its write-set / pool-array lowering (the
        // `state_ref_scalars.get(...)` below yields `None` and `filter_map`
        // discards the entry), compiling a config that does not do what it
        // declares.  Reject the typo at expansion time instead.
        for entry in residual_writes {
            // `@ Struct` names the layout itself. A state ref-scalar is only
            // required when the struct has to be recovered from one.
            if entry.struct_type.is_some() {
                continue;
            }
            let name = entry.ref_scalar.to_string();
            assert!(
                state_ref_scalars.contains_key(&name),
                "jit_interp: residual_writes ref_scalar `{name}` is not a declared `ref(_)` state field",
            );
        }
        for entry in pool_arrays {
            let name = entry.base.to_string();
            assert!(
                state_ref_scalars.contains_key(&name),
                "jit_interp: pool_arrays entry `{name}` is not a declared `ref(_)` state field",
            );
        }
        // Resolve each `residual_writes` entry into per-helper
        // `(helper segments, struct Path, field Ident)`, recovering the struct
        // `Path` from `state_ref_scalars[ref_scalar]` unless the declaration
        // explicitly names a nominal layout alias.  The latter supports raw
        // pointers which type-pun compatible repr(C) prefixes but whose field
        // descr identities remain per concrete struct type.
        let residual_writes = residual_writes
            .iter()
            .flat_map(|entry| {
                let struct_path = entry.struct_type.clone().or_else(|| {
                    state_ref_scalars
                        .get(&entry.ref_scalar.to_string())
                        .map(|(_, p)| p.clone())
                });
                entry.helpers.iter().filter_map(move |helper| {
                    struct_path.clone().map(|p| {
                        (
                            canonical_path_segments(helper),
                            p,
                            entry.field.clone(),
                            entry.writes_elements,
                        )
                    })
                })
            })
            .collect();
        let array_fields_map = build_array_fields_map(array_fields);
        let array_headers_map = build_array_headers_map(array_fields);
        // Build the ref_fields lookup: key = "StructLastSegment::field",
        // value = (struct_path, field_ident, pointee_path).
        let ref_fields_map: HashMap<String, (syn::Path, Ident, syn::Path)> = ref_fields
            .iter()
            .map(|entry| {
                let struct_name = entry
                    .struct_type
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                let key = format!("{}::{}", struct_name, entry.field);
                (
                    key,
                    (
                        entry.struct_type.clone(),
                        entry.field.clone(),
                        entry.pointee_type.clone(),
                    ),
                )
            })
            .collect();
        // Resolved before the struct literal takes `state_ref_scalars`: the
        // emitted `offset_of!` needs the base struct's type, and the
        // declaration names only its state-field.
        let pool_arrays: Vec<PoolArrayLowering> = pool_arrays
            .iter()
            .map(|entry| {
                let base = entry.base.to_string();
                let (_, struct_path) = state_ref_scalars
                    .get(&base)
                    .expect("the loop above rejected a base that is not a ref(_) state field");
                PoolArrayLowering {
                    base,
                    getter: canonical_path_segments(&entry.getter),
                    struct_path: struct_path.clone(),
                    items_field: entry.items_field.clone(),
                    len_field: entry.len_field.clone(),
                    element_type: entry.element_type.clone(),
                }
            })
            .collect();
        Self {
            io_shims,
            calls,
            auto_calls,
            vable_var,
            vable_input_ref_reg,
            vable_fields,
            vable_arrays,
            state_scalars,
            state_arrays,
            state_virt_arrays,
            state_ref_scalars,
            state_float_scalars,
            greens: greens.to_vec(),
            green_type_tags: green_type_tags.to_vec(),
            portal_green_params: portal_green_params.to_vec(),
            loop_carried_greens: Vec::new(),
            reds: reds.to_vec(),
            state_type_name: state_type.to_string(),
            env_type_name: env_type.to_string(),
            residual_writes,
            ref_fields: ref_fields_map,
            array_fields: array_fields_map,
            array_headers: array_headers_map,
            int_fields: int_fields_map(int_fields),
            float_fields: float_fields_set(int_fields),
            consulted_field_keys: Default::default(),
            call_returns: call_returns
                .iter()
                .map(|(func, ret_type)| (canonical_path_segments(func), ret_type.clone()))
                .collect(),
            headerless_structs: headerless_structs
                .iter()
                .map(canonical_path_segments)
                .collect(),
            inlined_prefix: inlined_prefix_map(inlined_prefix),
            pool_arrays,
            native_int_binops: native_int_binops
                .iter()
                .map(|(path, op)| (canonical_path_segments(path), op.to_string()))
                .collect(),
            native_tag_small: native_tag_small
                .iter()
                .map(canonical_path_segments)
                .collect(),
            native_identity: native_identity
                .iter()
                .map(canonical_path_segments)
                .collect(),
            split_dispatch,
            switch_dispatch,
        }
    }

    pub fn with_vable_input_ref_reg(&self, reg: u16) -> Self {
        let mut cloned = self.clone();
        if cloned.vable_var.is_some() {
            cloned.vable_input_ref_reg = Some(reg);
        }
        cloned
    }

    pub(super) fn with_loop_carried_greens(&self, greens: Vec<(String, ValueKind)>) -> Self {
        let mut cloned = self.clone();
        cloned.loop_carried_greens = greens;
        cloned
    }

    pub(super) fn is_headerless_struct(&self, struct_path: &syn::Path) -> bool {
        self.headerless_structs
            .contains(&canonical_path_segments(struct_path))
    }

    /// The gc-kind of a struct type — a property of the TYPE, never of the
    /// operation reaching it.
    ///
    /// `lltype.py:418` keeps `_gckind` on the STRUCT itself, and
    /// `jtransform.py:880` reads it off the struct a field belongs to rather
    /// than off the operation touching the field. One type therefore has one
    /// answer, and every descriptor minted for it has to give the same one:
    /// [`struct_type_id_tokens`](super::lower_value::struct_type_id_tokens)
    /// folds this flag into the type id, so two answers mint two identities
    /// for a single field. An optimizer that keys its field cache by
    /// descriptor identity then cannot see a write through one identity when
    /// it reads through the other, and serves the value from before the write.
    ///
    /// A struct declared in `headerless_structs` is allocated out of the
    /// interpreter's own GC pool, so it is gc-managed. What it lacks is the
    /// type-id word ahead of the payload, and that absence is carried by the
    /// descriptor's `headerless` flag — the flag the type guard actually
    /// consults — not by a separate type id.
    ///
    /// Every other struct stays raw here. A type the interpreter only ever
    /// reaches through a pointer it did not allocate is a genuinely separate
    /// low-level type from anything it allocates itself, and two ids for the
    /// two of them is the right answer rather than an accident.
    pub(super) fn struct_gc_kind_is_managed(&self, struct_path: &syn::Path) -> bool {
        self.is_headerless_struct(struct_path)
    }

    /// The struct that DECLARES `field`, following inlined leading
    /// substructures outward-in.
    ///
    /// `rclass.py InstanceRepr.getfield` resolves a field against the
    /// repr that owns it: `if attr in self.fields` … else recurse into
    /// `self.rbase` with `force_cast=True`. The cast is why
    /// `jtransform.py:881` always reads the descr off the declaring struct,
    /// and therefore why one physical field has one descriptor no matter how
    /// many outer structs embed it.
    ///
    /// Callers rebind their struct path through this before building the field
    /// key, so the declaration lookup, the type id, and every compile-time
    /// witness (`size_of`, `offset_of!`) move together onto the declaring
    /// struct. That is the whole of the cast: `jtransform.py rewrite_op_cast_pointer
    /// rewrite_op_cast_pointer` lowers `cast_pointer` to `same_as`, so no
    /// operation is emitted and the base pointer is unchanged — which holds
    /// only because an inlined base is required to start at offset 0.
    ///
    /// Returns `struct_path` unchanged when the struct declares the field
    /// itself, when it has no inlined base, or when nothing declared one.
    pub(super) fn declaring_struct(&self, struct_path: &syn::Path, field: &str) -> syn::Path {
        crate::jit_interp::declaring_struct_of(
            &self.inlined_prefix,
            |path, field| self.declares_field(path, field),
            struct_path,
            field,
        )
    }

    /// The layout entries an inlined base contributes to the struct embedding
    /// it, plus the witness that the base really is the leading field.
    ///
    /// `heaptracker.py:68-69` recurses into a nested struct when building a
    /// struct's positional field list, so an outer struct's list begins with
    /// the base's own fields and its own fields are numbered after them. That
    /// numbering is load-bearing twice over: `get_fielddescr_index_in`
    /// (`heaptracker.py`) assigns `index_in_parent` from it, and the
    /// per-object field cache slots by that index. Without the base's entries
    /// an outer struct's first own field takes slot 0 — the slot the base's
    /// first field already occupies for the same object.
    ///
    /// Returns `(entries, witness)`. `entries` are `(offset, is_ref, name,
    /// size, signed, is_float)` tuples in the shape the layout registration
    /// takes;
    /// offsets are relative to the OUTER struct. Empty when the struct embeds
    /// no declared base, which is the case for every struct unless a caller
    /// declares one.
    pub(super) fn prefix_field_entries_tokens(
        &self,
        outer: &syn::Path,
    ) -> (Vec<TokenStream>, TokenStream) {
        let mut entries = Vec::new();
        let mut witnesses = TokenStream::new();
        let mut current = outer.clone();
        let mut base_offset = quote! { 0usize };
        // Bounded for the same reason the declaring-struct walk is: the map is
        // hand-written and can name a cycle the Rust types never see.
        let mut remaining = self.inlined_prefix.len();
        while let Some((base_field, base)) = crate::jit_interp::inlined_prefix_entry(
            &self.inlined_prefix,
            &canonical_path_segments(&current),
        ) {
            if remaining == 0 {
                break;
            }
            remaining -= 1;
            let base_field = syn::Ident::new(base_field, proc_macro2::Span::call_site());
            // `lltype.py _first_struct` admits an inlined substructure only at
            // `_names[0]`, and the offsets below assume it: a redirected access
            // keeps the outer pointer in its register and offsets by the
            // field's place within the base, which names the right word only
            // when the base starts at zero.
            let message = format!(
                "`{}` is declared as the inlined leading substructure of `{}`, so it must be \
                 that struct's first field at offset 0.",
                path_display(base),
                path_display(&current),
            );
            witnesses.extend(quote! {
                const _: () = assert!(
                    ::core::mem::offset_of!(#current, #base_field) == 0,
                    #message,
                );
            });
            let outer_to_base = quote! {
                (#base_offset + ::core::mem::offset_of!(#current, #base_field))
            };
            for (field, is_ref, size, signed, is_float) in self.declared_fields_of(base) {
                entries.push(quote! {
                    (
                        #outer_to_base + ::core::mem::offset_of!(#base, #field),
                        #is_ref,
                        stringify!(#field),
                        #size,
                        #signed,
                        #is_float,
                    )
                });
            }
            base_offset = outer_to_base;
            current = base.clone();
        }
        (entries, witnesses)
    }

    /// The fields `struct_path` declares, as `(field, is_ref, size, signed)`.
    ///
    /// All three vocabularies, because all three take a slot: an array field's
    /// access site registers its base pointer as a layout entry of its own, so
    /// a base that declares one and is omitted here would leave the embedding
    /// struct's own fields numbered a slot too low, which is the collision the
    /// prefix exists to prevent. `declares_field` reads the same three, and the
    /// two must agree — a field the walk redirects onto a base but does not
    /// number after it is worse than one it never redirected.
    ///
    /// Sorted by field name. The declarations live in hash maps, whose
    /// iteration order varies per process, and an unsorted walk would emit a
    /// different token order on every compile of the same source.
    fn declared_fields_of(
        &self,
        struct_path: &syn::Path,
    ) -> Vec<(syn::Ident, bool, TokenStream, TokenStream, bool)> {
        let Some(last) = struct_path.segments.last() else {
            return Vec::new();
        };
        let prefix = format!("{}::", last.ident);
        let mut fields: Vec<_> = self
            .ref_fields
            .keys()
            .map(|key| (key, true))
            .chain(self.int_fields.keys().map(|key| (key, false)))
            .chain(self.float_fields.iter().map(|key| (key, false)))
            .chain(self.array_fields.keys().map(|key| (key, true)))
            .filter_map(|(key, is_ref)| key.strip_prefix(&prefix).map(|field| (field, is_ref)))
            .collect();
        fields.sort_unstable();
        fields.dedup();
        fields
            .into_iter()
            .map(|(field, is_ref)| {
                let key = format!("{prefix}{field}");
                // The same widths the access sites register: a declared integer
                // reports its own type's width, an array field the pointer it
                // reaches its buffer through, and anything else the eight-byte
                // scalar an undeclared field defaults to.
                let is_float = self.float_fields.contains(&key);
                let (size, signed) = match self.int_fields.get(&key) {
                    Some((ty, signed)) => {
                        (quote! { ::core::mem::size_of::<#ty>() }, quote! { #signed })
                    }
                    None if is_float => {
                        (quote! { ::core::mem::size_of::<f64>() }, quote! { false })
                    }
                    None if self.array_fields.contains_key(&key) => {
                        (quote! { ::core::mem::size_of::<usize>() }, quote! { false })
                    }
                    None => (quote! { ::core::mem::size_of::<i64>() }, quote! { true }),
                };
                (
                    syn::Ident::new(field, proc_macro2::Span::call_site()),
                    is_ref,
                    size,
                    signed,
                    is_float,
                )
            })
            .collect()
    }

    /// Whether `struct_path` itself declares `field` in any of the three field
    /// vocabularies. A field in none of them is undeclared everywhere, so the
    /// walk stops at the struct it was spelled with and the default applies.
    fn declares_field(&self, struct_path: &syn::Path, field: &str) -> bool {
        let Some(struct_name) = struct_path.segments.last() else {
            return false;
        };
        let key = format!("{}::{}", struct_name.ident, field);
        self.ref_fields.contains_key(&key)
            || self.int_fields.contains_key(&key)
            || self.float_fields.contains(&key)
            || self.array_fields.contains_key(&key)
    }
}

use crate::jit_interp::inlined_prefix_index as inlined_prefix_map;

/// A path as written, for a diagnostic that has to name it.
fn path_display(path: &Path) -> String {
    canonical_path_segments(path).join("::")
}

pub(super) fn canonical_path_segments(path: &Path) -> Vec<String> {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect()
}

pub(super) fn canonical_member_name(member: &syn::Member) -> String {
    match member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(idx) => idx.index.to_string(),
    }
}

pub(super) fn canonical_expr_segments(expr: &Expr) -> Option<Vec<String>> {
    match expr {
        Expr::Path(path) => Some(canonical_path_segments(&path.path)),
        Expr::Field(field) => {
            let mut segments = canonical_expr_segments(&field.base)?;
            segments.push(canonical_member_name(&field.member));
            Some(segments)
        }
        Expr::Paren(paren) => canonical_expr_segments(&paren.expr),
        Expr::Reference(reference) => canonical_expr_segments(&reference.expr),
        _ => None,
    }
}

pub(super) fn unwrap_ref_expr(expr: &Expr) -> &Expr {
    match expr {
        Expr::Reference(ExprReference { expr, .. }) => expr,
        _ => expr,
    }
}

pub(super) fn expr_matches_local_name(expr: &Expr, expected: &str) -> bool {
    match expr {
        Expr::Path(path) => path
            .path
            .get_ident()
            .map(|ident| ident == expected)
            .unwrap_or(false),
        Expr::Reference(reference) => expr_matches_local_name(&reference.expr, expected),
        Expr::Paren(paren) => expr_matches_local_name(&paren.expr, expected),
        _ => false,
    }
}

pub(super) fn named_member(member: &syn::Member) -> Option<String> {
    match member {
        syn::Member::Named(ident) => Some(ident.to_string()),
        _ => None,
    }
}

// ── Lowerer ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum BindingKind {
    Int,
    Ref,
    Float,
}

#[derive(Clone)]
pub(super) struct Binding {
    pub(super) reg: u16,
    pub(super) kind: BindingKind,
    pub(super) depends_on_stack: bool,
    /// When this binding is a Ref pointing to a known struct, tracks the
    /// struct type so that subsequent field access (`binding.field`) can
    /// resolve `offset_of!` and `ref_fields` without the `state.<ref>`
    /// two-level pattern.
    pub(super) struct_type: Option<syn::Path>,
}

/// A condition after `jtransform.py`'s `optimize_goto_if_not` decision.
///
/// A comparison whose boolean result is consumed only by the branch is kept
/// as its two operands and becomes `goto_if_not_<comparison>` directly.
/// Other conditions retain their materialized int result. In particular a
/// leading `!` stays in the latter form: inverting a float comparison would
/// change its NaN semantics.
pub(super) enum LoweredCondition {
    Value {
        binding: Binding,
        negated: bool,
        /// `true` when this value is an integer `== 0` / `!= 0` compare
        /// rewritten as a truth test. Rust `if`/`while` conditions are
        /// otherwise `bool`, which flatten emits as `goto_if_not`.
        /// A Ref binding is the `_rewrite_equality` fold of `ptr_eq` /
        /// `ptr_ne` against null; `negated` then selects
        /// `goto_if_not_ptr_iszero` / `goto_if_not_ptr_nonzero`.
        int_is_true: bool,
    },
    Compare {
        lhs: Binding,
        rhs: Binding,
        branch: Ident,
    },
    /// `goto_if_not_int_<cmp>` whose second operand is a `Constant` that
    /// `assembler.py` `emit_const` encodes in the branch. `opcode` names the
    /// comparison (`IntLt`, …).
    CompareConst {
        lhs: Binding,
        value: TokenStream,
        opcode: Ident,
    },
}

impl LoweredCondition {
    pub(super) fn depends_on_stack(&self) -> bool {
        match self {
            Self::Value { binding, .. } => binding.depends_on_stack,
            Self::Compare { lhs, rhs, .. } => lhs.depends_on_stack || rhs.depends_on_stack,
            Self::CompareConst { lhs, .. } => lhs.depends_on_stack,
        }
    }
}

/// Mirror of RPython `rpython/jit/codewriter/flatten.py:Register(kind, index)`.
/// Each emitted register carries its bank with it; the liveness walker
/// (`liveness.py:33-79`) keeps a single `set()` of `Register` objects per
/// marker, and `assembler.py get_liveness_info(args, kind)` filters
/// by `reg.kind == kind` at encode time to split into the per-bank bitsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct Register {
    /// Total order is `(kind, index)` so `BTreeSet<Register>` iterates in
    /// kind-grouped order — convenient for encoders that emit per-bank
    /// bitsets.
    pub(super) kind: BindingKind,
    pub(super) index: u8,
}

impl Register {
    /// Construct a `Register` from a `(kind, u16-index)` pair, asserting that
    /// the index fits in the `assembler.py:225` bitset addressing range
    /// (0..=255). The lowerer's `Lowerer::next_reg` counter already obeys
    /// this bound; the assert traps regressions where a u16 reg leaked from
    /// outside that bound.
    #[allow(dead_code)]
    pub(super) fn new(kind: BindingKind, index: u16) -> Self {
        assert!(
            index <= u8::MAX as u16,
            "Register index {} exceeds u8 (assembler.py:225 bitset range)",
            index,
        );
        Self {
            kind,
            index: index as u8,
        }
    }

    /// Per-bank constructor shortcut — `Register::int(0)` mirrors the
    /// RPython sugar of `Register('int', 0)`.
    #[allow(dead_code)]
    pub(super) fn int(index: u16) -> Self {
        Self::new(BindingKind::Int, index)
    }

    #[allow(dead_code)]
    pub(super) fn ref_(index: u16) -> Self {
        Self::new(BindingKind::Ref, index)
    }

    #[allow(dead_code)]
    pub(super) fn float(index: u16) -> Self {
        Self::new(BindingKind::Float, index)
    }

    /// Convenience: build a typed `Register` from a `Binding`.
    #[allow(dead_code)]
    pub(super) fn from_binding(b: &Binding) -> Self {
        Self::new(b.kind, b.reg)
    }

    /// Build a `Vec<Register>` of `Int` from a slice of indices. Used by
    /// emit sites whose reads list is uniformly Int (binop, guard_value,
    /// etc.).
    #[allow(dead_code)]
    pub(super) fn ints(indices: &[u16]) -> Vec<Register> {
        indices.iter().copied().map(Self::int).collect()
    }

    #[allow(dead_code)]
    pub(super) fn refs(indices: &[u16]) -> Vec<Register> {
        indices.iter().copied().map(Self::ref_).collect()
    }

    #[allow(dead_code)]
    pub(super) fn floats(indices: &[u16]) -> Vec<Register> {
        indices.iter().copied().map(Self::float).collect()
    }
}

// ── Op metadata for backward liveness analysis ─────
//
// `op_metadata[i]` describes the i-th emitted op so a downstream backward
// walker can produce per-marker live sets matching RPython
// `liveness.py _compute_liveness_must_continue`. Currently only the
// `LiveMarker` sites are populated.
//
// `kind` and `control` are split because future op categories (binop,
// load_const, jump, ...) carry the same `Linear`/`UnconditionalJump`/etc
// shape as several others; control flow is the orthogonal axis the walker
// branches on.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OpKind {
    /// `BC_LIVE` marker emitted before a guard. RPython `flatten.py:259`
    /// `-live-`. Carries no def/use; the walker records the alive set at
    /// this point.
    LiveMarker,
    LoadConstI,
    LoadConstR,
    LoadConstF,
    MoveI,
    MoveR,
    MoveF,
    /// `as usize` / `as isize`: a rename on a 64-bit word, a narrowing on a
    /// 32-bit one (`JitCodeBuilder::cast_int_to_word`). Register allocation
    /// coalesces it like a `MoveI`; unlike a move it is kept when source and
    /// target share a color, because the 32-bit narrowing still applies.
    CastIntToWord,
    BinopI,
    BinopF,
    UnaryI,
    /// Unconditional `jump` to a target label.
    Jump,
    /// `goto_if_not_*` — conditional branch with fail exit on miss.
    GotoIfNot,
    /// `mark_label` — defines a label.
    MarkLabel,
    /// Any `call_*_typed` / `call_*_args` / `residual_call_*` /
    /// `conditional_call_*` family op. `reads` carries arg regs;
    /// `writes` carries the result reg if the call is value-form.
    Call,
    /// `inline_call_*` family — sub-jitcode invocation.
    InlineCall,
    /// `vable_*` family (getfield/setfield/getarrayitem/setarrayitem/
    /// arraylen/force).
    Vable,
    /// `new` / `new_with_vtable` — allocate a JIT struct.  `writes`
    /// carries the result ref reg; `reads` is empty.
    New,
    /// `setfield_gc_{i,r,f}` — store a field through a struct ref.
    /// `reads` carries `[struct_reg, value_reg]`; no writes.
    SetfieldGc,
    /// `raw_store_i` — store an int into raw native memory at a byte
    /// offset. `reads` carries `[base_reg, ea_reg, value_reg]` (all int);
    /// no writes. jtransform.py `rewrite_op_raw_store`.
    RawStore,
    /// `raw_load_i` — load an int from raw native memory at a byte offset.
    /// `reads` carries `[base_reg, ea_reg]` (int); `writes` carries the
    /// result reg. jtransform.py `rewrite_op_raw_load`.
    RawLoad,
    /// `load_state_*` / `store_state_*` family.
    StateField,
    /// `int_guard_value` / `float_guard_value` / `ref_guard_value`.
    GuardValue,
    /// `assert_not_none` — record that a ref operand is non-null
    /// (pyjitpl.py opimpl_assert_not_none).
    AssertNotNone,
    /// `record_exact_class` — record that a ref operand's class is a
    /// known constant (pyjitpl.py opimpl_record_exact_class).
    RecordExactClass,
    /// `record_known_result_*` — pure-call result hint, no real call.
    RecordKnownResult,
    /// `jit_merge_point` portal merge-point marker.
    /// interp_jit.py pypyjitdriver.jit_merge_point(...).
    JitMergePoint,
    /// `loop_header` loop-header marker before the dispatch body.
    /// jtransform.py handle_jit_marker__loop_header.
    LoopHeader,
    /// Builder-side auxiliary statement that emits no BC_* op. Examples:
    /// `let #label = __builder.new_label();` (label allocation), Rust
    /// `let` bindings injected into the generated trace body for
    /// register-side use, sub-jitcode-add helpers. Carries no def/use;
    /// the backward walker treats it as a no-op pass-through.
    Aux,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ControlFlowClass {
    /// Falls through to the next op.
    Linear,
    /// `-live-` marker. Behaves Linear but is the recording point for
    /// backward liveness analysis.
    LiveMarker,
    /// `goto_if_not_*` — emits a fail exit and falls through on no-fail.
    /// Walker treats this both as Linear (fall-through into next op) and
    /// as a branch into the named label (joins at label backward propagation).
    ConditionalGuard,
    /// `jump` — unconditional branch. Walker resets `alive` from the
    /// label's accumulated set instead of the prior fall-through.
    UnconditionalJump,
    /// `mark_label` — defines a label. Walker records the current `alive`
    /// against the label name so forward jumps backward-feed from here.
    LabelDef,
    /// `*_return` family — terminal op with no fall-through and no
    /// successor. Walker resets `alive` to empty; the op's own register
    /// reads are still added as uses so the source value stays live.
    /// blackhole.py bhimpl_int_return / ref_return / float_return /
    /// void_return.
    Terminal,
}

/// TODO: no upstream counterpart.
///
/// `rpython/jit/codewriter/liveness.py remove_repeated_live`'s `-live-` is always
/// unconditional; `jtransform.py:311-312` decides whether to emit one at
/// translation time from `calldescr_canraise(calldescr)`, which is
/// statically known once the calldescr is built.  pyre's macro expansion
/// sees only a runtime helper-policy byte (`__majit_call_policy_<name>()`)
/// for inferred-policy callees because cross-crate proc macros cannot read
/// another crate's proc-macro-generated function at expand time, so the
/// emit decision is deferred to a runtime guard wrapped around
/// `live_placeholder*`.  `remove_repeated_live` merges adjacent markers
/// only when the run contains at least one unconditional marker (which
/// guarantees emit at this position, so unioning the conditional
/// siblings' reads is safe); a run consisting entirely of conditional
/// markers stays unmerged so that each marker's BC_LIVE captures only
/// its own alive set when its condition holds — unioning them would
/// over-capture vs PyPy's per-site `liveness.py`
/// `liveset.update(live[1:])` (which only sees `-live-`s that actually
/// exist).  Convergence path: once the ann/rtyper EffectInfo
/// infrastructure (Tasks #146/#235) exposes the helper's analyzer
/// outcome at expand time, this conditional surface retires and
/// `LiveMarkerCondition` plus `live_marker_if` /
/// `inferred_policy_live_condition` can be removed.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(super) struct LiveMarkerCondition {
    /// Boolean expression evaluated both in the JitCode builder body and in
    /// the liveness prebuild body.
    pub(super) emit: TokenStream,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(super) struct OpMeta {
    pub(super) kind: OpKind,
    /// Source registers (uses). Each `Register` carries `kind` directly per
    /// `flatten.py:Register(kind, index)` so the liveness walker stays a
    /// single-bag set and the encoder (`assembler.py:225-232`) splits into
    /// per-bank bitsets on demand.
    pub(super) reads: Vec<Register>,
    /// Destination registers (defs).
    pub(super) writes: Vec<Register>,
    /// Branch target label, for control-flow ops.
    pub(super) target_label: Option<Ident>,
    /// `-live-` marker TLabel operands. RPython stores every TLabel in
    /// the instruction tuple; a marker can carry more than one.
    pub(super) live_target_labels: Vec<Ident>,
    /// Optional guard for the physical `BC_LIVE` emission.  Unconditional
    /// markers match normal RPython ssarepr.  Conditional markers are the
    /// strict-parity bridge for inferred helper policies whose can-raise
    /// answer is represented by a runtime policy byte.
    pub(super) live_condition: Option<LiveMarkerCondition>,
    pub(super) control: ControlFlowClass,
}

#[allow(dead_code)]
impl OpMeta {
    pub(super) fn live_marker() -> Self {
        Self::live_marker_with(Vec::new(), Vec::new())
    }

    /// Conditional `-live-` for inferred-policy callees.  See
    /// [`LiveMarkerCondition`] for the TODO rationale
    /// and convergence path.
    pub(super) fn live_marker_if(condition: TokenStream) -> Self {
        let mut marker = Self::live_marker();
        marker.live_condition = Some(LiveMarkerCondition { emit: condition });
        marker
    }

    /// `-live-` marker carrying explicit force-alive register args and/or
    /// target labels whose accumulated alive sets should fold in. Mirrors
    /// RPython `rpython/jit/codewriter/liveness.py:44-53`'s handling of
    /// `-live-` insns whose tuple tail includes Register / TLabel
    /// entries. The lowerer currently never emits such enriched markers
    /// itself, but parity-aware consumers (snapshot helpers that synth
    /// extra live regs around an inline call) can produce them through
    /// this constructor.
    #[allow(dead_code)]
    pub(super) fn live_marker_with(reads: Vec<Register>, live_target_labels: Vec<Ident>) -> Self {
        Self {
            kind: OpKind::LiveMarker,
            reads,
            writes: Vec::new(),
            target_label: None,
            live_target_labels,
            live_condition: None,
            control: ControlFlowClass::LiveMarker,
        }
    }

    /// Linear op with explicit reads/writes. The most common shape —
    /// load_const, move, binop, unary, call, vable, state-field,
    /// guard_value, record_known_result, inline_call.
    pub(super) fn linear(kind: OpKind, reads: Vec<Register>, writes: Vec<Register>) -> Self {
        Self {
            kind,
            reads,
            writes,
            target_label: None,
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::Linear,
        }
    }

    /// Unconditional jump to `target`.
    pub(super) fn jump(target: Ident) -> Self {
        Self {
            kind: OpKind::Jump,
            reads: Vec::new(),
            writes: Vec::new(),
            target_label: Some(target),
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::UnconditionalJump,
        }
    }

    /// Conditional guard branching to `target` on miss. `cond_reg` is
    /// the read register feeding the guard.
    pub(super) fn conditional_guard(cond_reg: Register, target: Ident) -> Self {
        Self {
            kind: OpKind::GotoIfNot,
            reads: vec![cond_reg],
            writes: Vec::new(),
            target_label: Some(target),
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::ConditionalGuard,
        }
    }

    /// Two-register conditional guard produced by `optimize_goto_if_not`.
    /// Both comparison operands are read uses; their register banks carry the
    /// `/iiL` versus `/ffL` distinction used by `assembler.py`.
    pub(super) fn conditional_guard_compare(
        a_reg: Register,
        b_reg: Register,
        target: Ident,
    ) -> Self {
        Self {
            kind: OpKind::GotoIfNot,
            reads: vec![a_reg, b_reg],
            writes: Vec::new(),
            target_label: Some(target),
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::ConditionalGuard,
        }
    }

    /// Fused overflow-checked binop `int_*_jump_if_ovf(dst, lhs, rhs, label)`
    /// (`blackhole.py:478-497`).  Reads `lhs`/`rhs`, defines `dst` on the
    /// no-overflow fall-through, and branches to `label` on overflow.  Modeled
    /// as a `ConditionalGuard` (fall-through = taken iteration, branch =
    /// overflow) so the preceding `-live-` snapshot folds the overflow label's
    /// alive set and keeps `lhs`/`rhs` live; `dst` is a def (killed by the
    /// walker's `ConditionalGuard` write handling) so the not-yet-written
    /// result register does not enter the guard snapshot.
    pub(super) fn int_binop_jump_if_ovf(
        lhs: Register,
        rhs: Register,
        dst: Register,
        target: Ident,
    ) -> Self {
        Self {
            kind: OpKind::GotoIfNot,
            reads: vec![lhs, rhs],
            writes: vec![dst],
            target_label: Some(target),
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::ConditionalGuard,
        }
    }

    /// Label definition site. Walker uses `target` to associate the
    /// current `alive` set with the label name.
    pub(super) fn label_def(target: Ident) -> Self {
        Self {
            kind: OpKind::MarkLabel,
            reads: Vec::new(),
            writes: Vec::new(),
            target_label: Some(target),
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::LabelDef,
        }
    }

    /// Builder-side aux op (label allocation, Rust `let` bindings,
    /// sub-jitcode add). Linear, no def/use.
    pub(super) fn aux() -> Self {
        Self {
            kind: OpKind::Aux,
            reads: Vec::new(),
            writes: Vec::new(),
            target_label: None,
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::Linear,
        }
    }

    /// Terminal op (`*_return` family). No fall-through, no branch target.
    /// `reads` carries the source register so the walker still keeps it
    /// alive upstream; the walker resets the alive set on encountering
    /// this op (no successor to inherit from).
    pub(super) fn terminal(reads: Vec<Register>) -> Self {
        Self {
            kind: OpKind::Aux,
            reads,
            writes: Vec::new(),
            target_label: None,
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::Terminal,
        }
    }
}

#[derive(Default)]
pub(super) struct LoweredSequence {
    pub(super) statements: Vec<TokenStream>,
    pub(super) op_metadata: Vec<OpMeta>,
}

impl LoweredSequence {
    pub(super) fn new(statements: Vec<TokenStream>, op_metadata: Vec<OpMeta>) -> Self {
        assert_eq!(
            statements.len(),
            op_metadata.len(),
            "RPython ssarepr.insns parity requires statement/op_metadata streams to stay paired"
        );
        Self {
            statements,
            op_metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `@ Struct` is the layout. The name before the dot is not a state field.
    #[test]
    fn residual_writes_at_struct_does_not_need_a_state_ref_scalar() {
        let state_type = syn::parse_quote!(State);
        let env_type = syn::parse_quote!(Env);
        let entry = crate::jit_interp::ResidualWriteEntry {
            ref_scalar: syn::parse_quote!(col),
            field: syn::parse_quote!(data),
            struct_type: Some(syn::parse_quote!(W_IntColumn)),
            writes_elements: true,
            helpers: vec![syn::parse_quote!(append_cell)],
        };
        let config = LowererConfig::new(
            &[],
            &[],
            false,
            None,
            None,
            &[],
            &[],
            &[],
            &[],
            &state_type,
            &env_type,
            &[entry],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            false,
            false,
        );
        assert_eq!(config.residual_writes.len(), 1);
        assert_eq!(config.residual_writes[0].0, vec!["append_cell".to_string()]);
        assert!(config.residual_writes[0].3);
    }

    #[test]
    fn pointer_cast_temporaries_are_colored_with_their_builder_operands() {
        for expr in [
            syn::parse_quote!(arg as isize),
            syn::parse_quote!(arg as usize),
        ] {
            let mut lowerer = Lowerer::new(None);
            lowerer
                .bindings
                .insert("arg".into(), binding(0, BindingKind::Int));
            lowerer.next_reg = 200;
            let result = lowerer.lower_value_expr(&expr).unwrap();
            let (_, result) = regalloc::compact_registers(
                &mut lowerer,
                regalloc::RegisterCounts {
                    ints: 1,
                    ..Default::default()
                },
                Some(Register::int(result.reg)),
            );
            assert!(result.unwrap().index < 8);
            let statements = &lowerer.statements;
            let body = quote!(#(#statements)*).to_string();
            for old in 200..205 {
                assert!(
                    !body.contains(&format!("{old}u16")),
                    "uncolored register in {body}"
                );
            }
        }
    }

    #[test]
    fn another_virtualizable_does_not_hide_state_field_reads_or_writes() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("frame".into());
        config
            .vable_fields
            .insert("value".into(), (0, ValueKind::Int));
        config.state_scalars.insert("value".into(), 0);
        let mut lowerer = Lowerer::new(Some(&config));
        assert!(
            lowerer
                .lower_value_expr(&syn::parse_quote!(state.value))
                .is_some()
        );
        assert!(
            lowerer
                .lower_stmt(&syn::parse_quote!(state.value = 7;))
                .is_some()
        );
    }

    #[test]
    fn a_ref_state_field_virtualizable_lowers_to_vable_ops() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("frame".into());
        config.vable_input_ref_reg = Some(1);
        config
            .vable_fields
            .insert("last_instr".into(), (0, ValueKind::Int));
        config
            .vable_fields
            .insert("valuestackdepth".into(), (1, ValueKind::Int));
        config
            .vable_arrays
            .insert("locals_stack_w".into(), (0, ValueKind::Ref));
        config
            .state_ref_scalars
            .insert("frame".into(), (0, syn::parse_quote!(Frame)));
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.install_vable_input_binding();
        lowerer
            .bindings
            .insert("depth".into(), binding(4, BindingKind::Int));
        lowerer
            .bindings
            .insert("w".into(), binding(5, BindingKind::Ref));

        assert!(
            lowerer
                .lower_value_expr(&syn::parse_quote!(state.frame.last_instr))
                .is_some()
        );
        assert!(
            lowerer
                .lower_stmt(&syn::parse_quote!(state.frame.valuestackdepth = depth;))
                .is_some()
        );
        assert!(
            lowerer
                .lower_value_expr(&syn::parse_quote!(state.frame.locals_stack_w[depth]))
                .is_some()
        );
        assert!(
            lowerer
                .lower_stmt(&syn::parse_quote!(state.frame.locals_stack_w[depth] = w;))
                .is_some()
        );
        let statements = &lowerer.statements;
        let body = quote::quote!(#(#statements)*).to_string();
        assert!(body.contains("vable_getfield_int_with_base"), "{body}");
        assert!(body.contains("vable_setfield_int_with_base"), "{body}");
        assert!(body.contains("vable_getarrayitem_ref_with_base"), "{body}");
        assert!(body.contains("vable_setarrayitem_ref_with_base"), "{body}");
        assert!(!body.contains("getfield_gc"), "{body}");
        assert!(!body.contains("setfield_gc"), "{body}");
        assert!(!body.contains("getarrayitem_gc"), "{body}");
        assert!(!body.contains("setarrayitem_gc"), "{body}");
    }

    #[test]
    fn identity_ranges_preserve_sparse_indices_and_exclude_state_vable_fields() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("state".into());
        config.state_scalars.insert("redirected".into(), 0);
        config.state_scalars.insert("kept".into(), 3);
        config
            .state_ref_scalars
            .insert("redirected".into(), (0, syn::parse_quote!(Object)));
        config
            .state_ref_scalars
            .insert("kept".into(), (2, syn::parse_quote!(Object)));
        config.state_float_scalars.insert("redirected".into(), 250);
        config.state_float_scalars.insert("kept".into(), 1);
        config
            .vable_fields
            .insert("redirected".into(), (0, ValueKind::Int));
        assert_eq!(
            config.split_identity_reg_ends(),
            (
                config.int_identity_base() + 4,
                config.ref_identity_base() + 3
            )
        );
        assert_eq!(
            config.float_identity_end(),
            config.float_identity_base() + 2
        );
        config
            .vable_fields
            .insert("kept".into(), (1, ValueKind::Int));
        assert_eq!(config.float_identity_end(), 0);
        config.vable_var = Some("frame".into());
        assert_eq!(
            config.split_identity_reg_ends(),
            (
                config.int_identity_base() + 4,
                config.ref_identity_base() + 3
            )
        );
        assert_eq!(
            config.float_identity_end(),
            config.float_identity_base() + 251
        );
    }

    #[test]
    fn identity_slot_registers_list_plain_reds_not_vable_fields() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("state".into());
        config.state_scalars.insert("redirected".into(), 0);
        config.state_scalars.insert("kept".into(), 3);
        config
            .state_ref_scalars
            .insert("kept_ref".into(), (1, syn::parse_quote!(Object)));
        config.state_float_scalars.insert("kept_f".into(), 0);
        config
            .vable_fields
            .insert("redirected".into(), (0, ValueKind::Int));
        let mut regs = config.identity_slot_registers();
        regs.sort();
        assert_eq!(
            regs,
            vec![
                Register::int(config.int_identity_base() + 3),
                Register::ref_(config.ref_identity_base() + 1),
                Register::float(config.float_identity_base()),
            ]
        );
    }

    #[test]
    fn identity_slot_registers_list_plain_int_array_as_ptr_then_len() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.state_arrays.insert("cells".into(), 0);
        let mut regs = config.identity_slot_registers();
        regs.sort();
        let base = config.int_identity_base();
        assert_eq!(
            regs,
            vec![Register::int(base), Register::int(base + 1)],
            "a plain [int] is items-pointer then length after the int scalars"
        );
    }

    #[test]
    fn liveness_records_alive_at_marker_after_use() {
        // [marker, linear reads=[5]]
        // backward: linear adds 5 → alive={5}; marker saves {5}.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[5]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(result, vec![BTreeSet::from([Register::int(5)])]);
    }

    #[test]
    fn liveness_def_kills_use_kept() {
        // [marker, linear1 reads=[3], linear2 writes=[3] reads=[5]]
        // backward: linear2 adds 5 (writes 3 first but alive empty);
        //           linear1 adds 3; marker saves {3, 5}.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[5]), Register::ints(&[3])),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([Register::int(3), Register::int(5)])],
        );
    }

    #[test]
    fn liveness_jump_overwrites_alive_with_label_set() {
        // [marker, jump L1, mark_label L2, reads=[7], mark_label L1, reads=[9]]
        // backward single pass:
        //   reads(9): alive={9}
        //   mark_label L1: label_alive[L1]={9}, alive stays {9}
        //   reads(7): alive={9,7}
        //   mark_label L2: label_alive[L2]={9,7}, alive stays {9,7}
        //   jump L1: alive = label_alive[L1] = {9}
        //   marker: save {9}
        let l1 = format_ident!("L1");
        let l2 = format_ident!("L2");
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::jump(l1.clone()),
            OpMeta::label_def(l2.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[7]), vec![]),
            OpMeta::label_def(l1.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[9]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(result, vec![BTreeSet::from([Register::int(9)])]);
    }

    #[test]
    fn liveness_back_edge_loop_reaches_fixed_point() {
        // [mark_label START, marker, reads=[2], jump START]
        // pass 1: jump finds label_alive[START]={}, then reads(2) → {2}, marker={2}, mark_label sets START={2}.
        // pass 2: jump finds {2}, reads(2) keeps {2}, marker {2}, label unchanged → done.
        let start = format_ident!("LOOP_START");
        let ops = vec![
            OpMeta::label_def(start.clone()),
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[2]), vec![]),
            OpMeta::jump(start.clone()),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(result, vec![BTreeSet::from([Register::int(2)])]);
    }

    #[test]
    fn liveness_conditional_guard_reads_cond_and_unions_branch_target() {
        // [marker, conditional_guard cond_reg=4 → ELSE, reads=[6], mark_label ELSE, reads=[8]]
        // backward:
        //   reads(8): alive={8}
        //   mark_label ELSE: label_alive[ELSE]={8}
        //   reads(6): alive={8, 6}  (fall-through past ELSE in backward order)
        //
        //   Wait, "fall-through past ELSE" backward direction means we already passed reads(8) and
        //   mark_label, now at reads(6). reads(6) sets alive={8,6}.
        //
        //   conditional_guard target=ELSE, reads=[4]: alive folds in label_alive[ELSE]={8}
        //   → alive={8,6} (already had 8) ∪ {8} = {8,6}; then add reads [4] → {8,6,4}.
        //   marker: save {8,6,4}.
        let else_label = format_ident!("ELSE");
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::conditional_guard(Register::int(4), else_label.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[6]), vec![]),
            OpMeta::label_def(else_label.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[8]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([
                Register::int(4),
                Register::int(6),
                Register::int(8),
            ])],
        );
    }

    #[test]
    fn liveness_marker_with_explicit_args_force_alive() {
        // [marker_with([7]), reads=[5]]
        // marker carries `7` as a force-alive register; backward walk
        // adds 5 (linear) then folds 7 into alive at marker → {5, 7}.
        let ops = vec![
            OpMeta::live_marker_with(Register::ints(&[7]), Vec::new()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[5]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([Register::int(5), Register::int(7)])],
        );
    }

    #[test]
    fn liveness_marker_with_target_label_unions_label_set() {
        // [marker_with([], target=L1), reads=[3], jump L1, label_def L1, reads=[9]]
        // pass 1: reads(9) alive={9}; label L1 saved {9}; jump L1 → alive={9};
        //         reads(3) alive={9,3}; marker fold L1 alive={9,3}∪{9}={9,3}; saved.
        // pass 2: stable.
        let l1 = format_ident!("L1");
        let ops = vec![
            OpMeta::live_marker_with(Vec::new(), vec![l1.clone()]),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
            OpMeta::jump(l1.clone()),
            OpMeta::label_def(l1.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[9]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([Register::int(3), Register::int(9)])],
        );
    }

    #[test]
    fn liveness_marker_with_multiple_target_labels_unions_all_label_sets() {
        let l1 = format_ident!("L1");
        let l2 = format_ident!("L2");
        let ops = vec![
            OpMeta::live_marker_with(Vec::new(), vec![l1.clone(), l2.clone()]),
            OpMeta::jump(l1.clone()),
            OpMeta::label_def(l2.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[7]), vec![]),
            OpMeta::label_def(l1.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[9]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([Register::int(7), Register::int(9)])],
        );
    }

    #[test]
    fn remove_repeated_live_is_no_op_when_runs_have_one_marker() {
        // [marker, reads=[1], marker, reads=[2]] — two markers but each is
        // separated by a non-marker op, so no run to collapse.
        let mut ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[1]), vec![]),
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[2]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len())
            .map(|i| quote! { /* op #i */ let _ = #i; })
            .collect();
        let before_len = ops.len();
        remove_repeated_live(&mut ops, &mut stmts);
        assert_eq!(ops.len(), before_len);
        assert_eq!(stmts.len(), before_len);
    }

    #[test]
    fn remove_repeated_live_moves_labels_before_a_lone_marker() {
        // RPython `liveness.py remove_repeated_live` collects labels after a
        // marker and emits them before that marker even when the run contains
        // only one `-live-`.  An edge entering at L must therefore execute the
        // marker instead of jumping past it.
        let l = format_ident!("L");
        let mut ops = vec![
            OpMeta::live_marker_with(Register::ints(&[1]), Vec::new()),
            OpMeta::label_def(l),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[2]), vec![]),
        ];
        let marker_stmt = quote! { let marker = 1; };
        let label_stmt = quote! { let label = 2; };
        let linear_stmt = quote! { let linear = 3; };
        let mut stmts = vec![marker_stmt.clone(), label_stmt.clone(), linear_stmt.clone()];

        remove_repeated_live(&mut ops, &mut stmts);

        assert!(matches!(ops[0].control, ControlFlowClass::LabelDef));
        assert!(matches!(ops[1].control, ControlFlowClass::LiveMarker));
        assert!(matches!(ops[2].control, ControlFlowClass::Linear));
        assert_eq!(stmts[0].to_string(), label_stmt.to_string());
        assert_eq!(stmts[1].to_string(), marker_stmt.to_string());
        assert_eq!(stmts[2].to_string(), linear_stmt.to_string());
    }

    #[test]
    fn remove_repeated_live_collapses_consecutive_markers() {
        // [marker(reads=[1]), marker(reads=[2]), label_def L, marker, reads=[3]]
        // RPython: collapse the run before reads=[3] into a single marker
        // carrying union({1, 2}) (label L stays as a separate op kept in
        // place between original positions, though its position relative
        // to the merged marker shifts to before per RPython).
        let l = format_ident!("L");
        let mut ops = vec![
            OpMeta::live_marker_with(Register::ints(&[1]), Vec::new()),
            OpMeta::live_marker_with(Register::ints(&[2]), Vec::new()),
            OpMeta::label_def(l.clone()),
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len()).map(|_| quote! { let _ = (); }).collect();
        remove_repeated_live(&mut ops, &mut stmts);
        // Resulting layout: [label_def L, merged_marker(reads={1,2}), reads=[3]].
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0].control, ControlFlowClass::LabelDef));
        assert!(matches!(ops[1].control, ControlFlowClass::LiveMarker));
        assert_eq!(ops[1].reads, vec![Register::int(1), Register::int(2)]);
        assert!(matches!(ops[2].control, ControlFlowClass::Linear));
    }

    #[test]
    fn remove_repeated_live_preserves_all_marker_target_labels() {
        let l1 = format_ident!("L1");
        let l2 = format_ident!("L2");
        let mut ops = vec![
            OpMeta::live_marker_with(Register::ints(&[1]), vec![l2.clone()]),
            OpMeta::live_marker_with(Register::ints(&[2]), vec![l1.clone()]),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len()).map(|_| quote! { let _ = (); }).collect();
        remove_repeated_live(&mut ops, &mut stmts);
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0].control, ControlFlowClass::LiveMarker));
        assert_eq!(ops[0].reads, vec![Register::int(1), Register::int(2)]);
        let labels: Vec<String> = ops[0]
            .live_target_labels
            .iter()
            .map(|label| label.to_string())
            .collect();
        assert_eq!(labels, vec!["L1".to_string(), "L2".to_string()]);
        assert!(matches!(ops[1].control, ControlFlowClass::Linear));
    }

    #[test]
    fn remove_repeated_live_keeps_conditional_only_runs_unmerged() {
        // A run consisting entirely of conditional markers stays
        // unmerged: unioning their reads would over-capture vs PyPy's
        // per-site `liveness.py` `liveset.update(live[1:])`
        // (which only ever sees `-live-`s that actually exist).  Each
        // marker's BC_LIVE fires (or not) on its own condition and
        // captures only its own alive set.
        let mut ops = vec![
            OpMeta::live_marker_if(quote! { policy_a() }),
            OpMeta::live_marker_if(quote! { policy_b() }),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len()).map(|_| quote! { let _ = (); }).collect();
        remove_repeated_live(&mut ops, &mut stmts);
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0].control, ControlFlowClass::LiveMarker));
        assert!(matches!(ops[1].control, ControlFlowClass::LiveMarker));
        assert!(ops[0].live_condition.is_some());
        assert!(ops[1].live_condition.is_some());
        assert!(matches!(ops[2].control, ControlFlowClass::Linear));
    }

    #[test]
    fn remove_repeated_live_drops_conditions_when_run_includes_unconditional_marker() {
        // An unconditional marker mixed in with conditional ones forces
        // the merged result to be unconditional — PyPy emits at this
        // position regardless, so the conditional siblings must follow
        // suit (their alive sets fold in).
        let mut ops = vec![
            OpMeta::live_marker_if(quote! { policy_a() }),
            OpMeta::live_marker_with(Register::ints(&[1]), vec![]),
            OpMeta::live_marker_if(quote! { policy_b() }),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len()).map(|_| quote! { let _ = (); }).collect();
        remove_repeated_live(&mut ops, &mut stmts);
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0].control, ControlFlowClass::LiveMarker));
        assert!(
            ops[0].live_condition.is_none(),
            "merged marker must be unconditional when run contains an unconditional marker"
        );
        assert_eq!(ops[0].reads, vec![Register::int(1)]);
        assert!(matches!(ops[1].control, ControlFlowClass::Linear));
    }

    #[test]
    fn rewrite_live_marker_replaces_placeholder_with_triple() {
        // [marker, reads=[1], writes=[2]] — backward walk records {1}
        // alive at marker, def[2] discards 2 from alive carry-over.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[1]), Register::ints(&[2])),
        ];
        let other_marker = quote! { let __probe = 1234; }.to_string();
        let mut stmts: Vec<TokenStream> = vec![
            quote! { let _ = __builder.live_placeholder(); },
            quote! { let __probe = 1234; },
        ];
        rewrite_live_marker_statements_with_triples(&ops, &mut stmts);
        let rendered = stmts[0].to_string();
        assert!(
            rendered.contains("live_placeholder_with_triple"),
            "post-rewrite stmt[0] still uses bare live_placeholder: {rendered}"
        );
        // The triple must reflect the walker output: live_i = [1], live_r = [], live_f = [].
        assert!(
            rendered.contains("& [1u8]"),
            "live_i array missing or wrong: {rendered}"
        );
        assert!(
            rendered.contains("& []"),
            "live_r/live_f empty arrays missing: {rendered}"
        );
        // Non-marker statement is untouched (token-stream equality, since
        // `quote!` strips comments and reformats whitespace).
        assert_eq!(stmts[1].to_string(), other_marker);
    }

    #[test]
    fn rewrite_live_marker_emits_typed_arrays_per_bank() {
        // [marker, reads_int=[3], reads_ref=[7]] — walker records both
        // banks; rewrite must emit per-bank typed arrays (live_i, live_r,
        // live_f) so `live_placeholder_with_triple` sees the right shape.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::int(3), Register::ref_(7)],
                vec![],
            ),
        ];
        let mut stmts: Vec<TokenStream> = vec![
            quote! { let _ = __builder.live_placeholder(); },
            quote! { /* op */ },
        ];
        rewrite_live_marker_statements_with_triples(&ops, &mut stmts);
        let rendered = stmts[0].to_string();
        assert!(rendered.contains("& [3u8]"), "live_i missing: {rendered}");
        assert!(rendered.contains("& [7u8]"), "live_r missing: {rendered}");
    }

    #[test]
    fn liveness_aux_is_pass_through() {
        // [marker, aux, reads=[1]] — aux carries no def/use; alive at marker = {1}.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::aux(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[1]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(result, vec![BTreeSet::from([Register::int(1)])]);
    }

    #[test]
    fn get_liveness_info_filters_by_kind() {
        // RPython `assembler.py get_liveness_info(args, kind)` parity:
        // a single set of typed Registers projected per bank yields the same
        // sorted u8 indices RPython would emit into the BC_LIVE bitset.
        let set: BTreeSet<Register> = [
            Register::int(3),
            Register::ref_(7),
            Register::int(5),
            Register::float(2),
            Register::ref_(1),
        ]
        .into_iter()
        .collect();
        assert_eq!(get_liveness_info(&set, BindingKind::Int), vec![3u8, 5u8]);
        assert_eq!(get_liveness_info(&set, BindingKind::Ref), vec![1u8, 7u8]);
        assert_eq!(get_liveness_info(&set, BindingKind::Float), vec![2u8]);
    }

    #[test]
    fn liveness_triple_keeps_per_bank_sort_order() {
        // BTreeSet<Register> orders by (kind, index); `liveness_triple` must
        // surface that ordering as three independent sorted Vec<u8> slices,
        // matching `assembler.py _encode_liveness` which encodes each
        // bank as a sorted bitset.
        let set: BTreeSet<Register> = [
            Register::float(9),
            Register::int(2),
            Register::ref_(4),
            Register::float(1),
            Register::int(0),
        ]
        .into_iter()
        .collect();
        let (live_i, live_r, live_f) = liveness_triple(&set);
        assert_eq!(live_i, vec![0u8, 2u8]);
        assert_eq!(live_r, vec![4u8]);
        assert_eq!(live_f, vec![1u8, 9u8]);
    }

    #[test]
    fn liveness_triple_empty_when_set_is_empty() {
        let set: BTreeSet<Register> = BTreeSet::new();
        assert_eq!(liveness_triple(&set), (Vec::new(), Vec::new(), Vec::new()));
    }

    fn binding(reg: u16, kind: BindingKind) -> Binding {
        Binding {
            reg,
            kind,
            depends_on_stack: false,
            struct_type: None,
        }
    }

    /// `_immutable_fields_ = ['items[*]']` (`rewrite_op_getarrayitem`) emits
    /// `getarrayitem_gc_r_pure`. The choice is the struct's
    /// `__MAJIT_IMMUTABLE_FIELDS`, not a CEL-specific opcode.
    #[test]
    fn immutable_array_item_emits_getarrayitem_gc_pure() {
        let func = parse_fn(
            r#"
            fn load(tup: *mut Tup, i: i64) -> *mut CelObject {
                tup.items[i]
            }
            "#,
        );
        let array_fields = [crate::jit_interp::ArrayFieldEntry {
            struct_type: syn::parse_quote!(Tup),
            field: syn::parse_quote!(items),
            element_type: syn::parse_quote!(CelRef),
            header: Some(syn::parse_quote!(Block)),
        }];
        let int_fields = [crate::jit_interp::IntFieldEntry {
            struct_type: syn::parse_quote!(Block),
            field: syn::parse_quote!(capacity),
            int_type: syn::parse_quote!(usize),
        }];
        let helper = generate_inline_helper_jitcode_with_calls(
            &func,
            &[],
            &[(syn::parse_quote!(tup), syn::parse_quote!(Tup))],
            &[],
            &array_fields,
            &int_fields,
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .expect("jit_inline lowering should succeed")
        .expect("helper should lower");
        let body = helper.body.to_string();
        assert!(
            body.contains("getarrayitem_gc_r_pure"),
            "immutable array item must be getarrayitem_gc_pure: {body}"
        );
        assert!(body.contains("__MAJIT_IMMUTABLE_FIELDS"), "{body}");
        assert!(body.contains("items[*]"), "{body}");
    }

    fn parse_fn(code: &str) -> ItemFn {
        syn::parse_str(code).expect("failed to parse function")
    }

    fn inline_policy_with_kind(
        path: &str,
        kind: crate::jit_interp::CallPolicyKind,
    ) -> crate::jit_interp::CallEntry {
        crate::jit_interp::CallEntry {
            path: syn::parse_str(path).expect("failed to parse path"),
            policy: Some(kind),
        }
    }

    fn inline_policy(path: &str) -> crate::jit_interp::CallEntry {
        inline_policy_with_kind(path, crate::jit_interp::CallPolicyKind::InlineInt)
    }

    fn lowerer_with_call_policy(
        path: &str,
        kind: crate::jit_interp::CallPolicyKind,
    ) -> Lowerer<'static> {
        let path: Path = syn::parse_str(path).expect("failed to parse path");
        Lowerer::new_with_call_policies(
            None,
            vec![(
                canonical_path_segments(&path),
                CallPolicySpec::Explicit(kind),
            )],
            InferenceFailureMode::ReturnNone,
        )
    }

    fn lowerer_with_inferred_call_policy(path: &str) -> Lowerer<'static> {
        let path: Path = syn::parse_str(path).expect("failed to parse path");
        Lowerer::new_with_call_policies(
            None,
            vec![(canonical_path_segments(&path), CallPolicySpec::Infer)],
            InferenceFailureMode::ReturnNone,
        )
    }

    fn parse_call(code: &str) -> ExprCall {
        syn::parse_str(code).expect("failed to parse call")
    }

    fn method_call_config(
        owner: &str,
        method: &str,
        kind: crate::jit_interp::CallPolicyKind,
    ) -> LowererConfig {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.state_type_name = "Machine".to_string();
        config.env_type_name = "Program".to_string();
        config.calls.push((
            vec![owner.to_string(), method.to_string()],
            CallPolicySpec::Explicit(kind),
        ));
        config
    }

    fn inline_call_tokens_combined(bindings: &[Binding], result_reg: u16) -> String {
        let (call_match, post_live) = inline_call_tokens(bindings, result_reg, &[]);
        format!("{} {}", call_match, post_live)
    }

    #[test]
    fn append_lowered_sequence_keeps_statements_and_metadata_aligned() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        let seq = LoweredSequence::new(
            vec![quote! { __builder.load_const_i_value(1, 42); }],
            vec![OpMeta::linear(
                OpKind::LoadConstI,
                Vec::new(),
                Register::ints(&[1]),
            )],
        );
        lowerer.append_lowered_sequence(seq);
        assert_eq!(lowerer.statements.len(), lowerer.op_metadata.len());
        assert!(matches!(lowerer.op_metadata[1].kind, OpKind::LoadConstI));
    }

    #[test]
    fn method_call_uses_the_shared_residual_ref_lowering() {
        let mut config = method_call_config(
            "Program",
            "lookup",
            crate::jit_interp::CallPolicyKind::ResidualRef,
        );
        config.call_returns.insert(
            vec!["Program".to_string(), "lookup".to_string()],
            syn::parse_quote!(Node),
        );
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer
            .bindings
            .insert("program".to_string(), binding(0, BindingKind::Ref));
        lowerer
            .bindings
            .insert("pc".to_string(), binding(0, BindingKind::Int));
        let call: Expr = syn::parse_quote! { program.lookup(pc) };

        let result = lowerer
            .lower_value_expr(&call)
            .expect("a method has the same residual-ref policy surface as a free call");

        assert!(matches!(result.kind, BindingKind::Ref));
        assert_eq!(result.struct_type, Some(syn::parse_quote!(Node)));
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("Program :: lookup"), "{body}");
        assert!(
            body.contains("residual_call_ref_canonical_via_target"),
            "{body}"
        );
        assert!(body.contains("live_placeholder"), "{body}");
    }

    #[test]
    fn method_call_uses_the_shared_wrapped_float_lowering() {
        let config = method_call_config(
            "Machine",
            "ratio",
            crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped,
        );
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer
            .bindings
            .insert("state".to_string(), binding(0, BindingKind::Ref));
        lowerer
            .bindings
            .insert("value".to_string(), binding(0, BindingKind::Int));
        let call: Expr = syn::parse_quote! { state.ratio(value) };

        let result = lowerer
            .lower_value_expr(&call)
            .expect("method syntax must not narrow the wrapped float policy surface");

        assert!(matches!(result.kind, BindingKind::Float));
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(
            body.contains("Machine :: __majit_call_policy_ratio"),
            "{body}"
        );
        assert!(
            body.contains("call_pure_float_canonical_via_target_cannot_raise"),
            "{body}"
        );
        assert!(!body.contains("live_placeholder"), "{body}");
    }

    #[test]
    fn method_statement_uses_the_shared_void_lowering() {
        let config = method_call_config(
            "Machine",
            "mutate",
            crate::jit_interp::CallPolicyKind::ResidualVoid,
        );
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer
            .bindings
            .insert("state".to_string(), binding(0, BindingKind::Ref));
        lowerer
            .bindings
            .insert("value".to_string(), binding(0, BindingKind::Int));
        let call: Expr = syn::parse_quote! { state.mutate(value) };

        lowerer
            .lower_config_call_stmt(&call)
            .expect("method statements share the free-call void policy surface");

        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("Machine :: mutate"), "{body}");
        assert!(
            body.contains("residual_call_void_canonical_via_target"),
            "{body}"
        );
        assert!(body.contains("live_placeholder"), "{body}");
        assert_eq!(lowerer.op_metadata[0].reads.len(), 2);
        assert!(matches!(lowerer.op_metadata[1].kind, OpKind::LiveMarker));
    }

    #[test]
    fn record_known_result_metadata_reads_known_result_and_writes_nothing() {
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ElidableInt);
        lowerer
            .bindings
            .insert("known".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("record_known_result!(known, helper, arg)").expect("parse macro expr");

        lowerer
            .lower_record_known_result(&expr)
            .expect("record_known_result should lower");

        let record = lowerer
            .op_metadata
            .iter()
            .find(|m| matches!(m.kind, OpKind::RecordKnownResult))
            .expect("RecordKnownResult metadata emitted");
        assert_eq!(record.reads, Register::ints(&[0, 1]));
        assert!(record.writes.is_empty());
        // `jtransform.py:311-312` trailing `-live-` after a can-raise
        // record_known_result call.
        let last = lowerer.op_metadata.last().expect("metadata emitted");
        assert!(matches!(last.kind, OpKind::LiveMarker));
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("add_word_abi_fn_ptr_with_slot"));
        assert!(tokens.contains("ElidableCanRaise"));
    }

    #[test]
    fn record_known_result_cannot_raise_elidable_omits_live_marker() {
        let mut lowerer = lowerer_with_call_policy(
            "helper",
            crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise,
        );
        lowerer
            .bindings
            .insert("known".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("record_known_result!(known, helper, arg)").expect("parse macro expr");

        lowerer
            .lower_record_known_result(&expr)
            .expect("record_known_result should lower");

        assert_eq!(lowerer.op_metadata.len(), 1);
        assert!(matches!(
            lowerer.op_metadata[0].kind,
            OpKind::RecordKnownResult
        ));
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("ElidableCannotRaise"));
        assert!(!tokens.contains("live_placeholder"));
    }

    #[test]
    fn record_known_result_inferred_policy_validates_and_conditions_live_marker() {
        let mut lowerer = lowerer_with_inferred_call_policy("helper");
        lowerer
            .bindings
            .insert("known".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("record_known_result!(known, helper, arg)").expect("parse macro expr");

        lowerer
            .lower_record_known_result(&expr)
            .expect("record_known_result should lower");

        assert_eq!(lowerer.op_metadata.len(), 2);
        assert!(matches!(
            lowerer.op_metadata[0].kind,
            OpKind::RecordKnownResult
        ));
        assert!(lowerer.op_metadata[1].live_condition.is_some());
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("match __policy"));
        assert!(tokens.contains("requires an elidable helper policy"));
    }

    #[test]
    #[should_panic(expected = "record_known_result! requires an elidable helper policy")]
    fn record_known_result_rejects_non_elidable_policy() {
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ResidualInt);
        lowerer
            .bindings
            .insert("known".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("record_known_result!(known, helper, arg)").expect("parse macro expr");

        let _ = lowerer.lower_record_known_result(&expr);
    }

    #[test]
    fn conditional_call_loopinvariant_omits_live_marker() {
        // `call.py getcalldescr` forbids non-void args for
        // loop-invariant direct_call, so the cond_call shape must
        // also have no func args when the slot is `LoopInvariant`.
        let mut lowerer = lowerer_with_call_policy(
            "helper",
            crate::jit_interp::CallPolicyKind::LoopInvariantVoid,
        );
        lowerer
            .bindings
            .insert("cond".to_string(), binding(0, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("conditional_call!(cond, helper)").expect("parse macro expr");

        lowerer
            .lower_conditional_call(&expr)
            .expect("conditional_call should lower");

        assert_eq!(lowerer.op_metadata.len(), 1);
        assert!(matches!(lowerer.op_metadata[0].kind, OpKind::Call));
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("LoopInvariant"));
        assert!(!tokens.contains("live_placeholder"));
    }

    #[test]
    #[should_panic(expected = "arguments not supported for loop-invariant function")]
    fn conditional_call_loopinvariant_rejects_func_args() {
        // `call.py getcalldescr` asserts `not NON_VOID_ARGS`
        // for loop-invariant direct_call.  The cond_call macro path
        // mirrors that assert at expansion time.
        let mut lowerer = lowerer_with_call_policy(
            "helper",
            crate::jit_interp::CallPolicyKind::LoopInvariantVoid,
        );
        lowerer
            .bindings
            .insert("cond".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("conditional_call!(cond, helper, arg)").expect("parse macro expr");

        let _ = lowerer.lower_conditional_call(&expr);
    }

    #[test]
    fn conditional_call_inferred_policy_keeps_runtime_loopinvariant_arg_check() {
        let mut lowerer = lowerer_with_inferred_call_policy("helper");
        lowerer
            .bindings
            .insert("cond".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("conditional_call!(cond, helper, arg)").expect("parse macro expr");

        lowerer
            .lower_conditional_call(&expr)
            .expect("conditional_call should lower");

        assert!(lowerer.op_metadata.last().unwrap().live_condition.is_some());
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("match __policy"));
        assert!(tokens.contains("arguments not supported for loop-invariant function"));
    }

    #[test]
    #[should_panic(expected = "conditional_call! cannot lower helper policy MayForceVoid")]
    fn conditional_call_rejects_may_force_policy() {
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::MayForceVoid);
        lowerer
            .bindings
            .insert("cond".to_string(), binding(0, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("conditional_call!(cond, helper)").expect("parse macro expr");

        let _ = lowerer.lower_conditional_call(&expr);
    }

    #[test]
    fn conditional_call_elidable_residual_policy_keeps_live_marker() {
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ResidualInt);
        lowerer
            .bindings
            .insert("value".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr = syn::parse_str("conditional_call_elidable!(value, helper, arg)")
            .expect("parse macro expr");

        let result = lowerer
            .lower_conditional_call_elidable(&expr)
            .expect("conditional_call_elidable should lower");

        assert_eq!(result.kind, BindingKind::Int);
        let last = lowerer.op_metadata.last().expect("metadata emitted");
        assert!(matches!(last.kind, OpKind::LiveMarker));
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("CanRaise"));
        assert!(tokens.contains("live_placeholder"));
    }

    #[test]
    fn promote_assign_aliases_lhs_to_promoted_arg_binding() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("y".to_string(), binding(7, BindingKind::Int));
        lowerer
            .bindings
            .insert("x".to_string(), binding(1, BindingKind::Int));
        let expr: Expr = syn::parse_str("x = promote(y)").expect("parse promote assignment");

        lowerer
            .lower_promote_stmt(&expr)
            .expect("promote assignment should lower");

        let x = lowerer.bindings.get("x").expect("x binding must exist");
        assert_eq!(
            x.reg, 7,
            "jtransform.py:613-615 returns None so result aliases arg0"
        );
        assert!(matches!(x.kind, BindingKind::Int));
        assert!(matches!(
            lowerer.op_metadata[0].control,
            ControlFlowClass::LiveMarker
        ));
        assert!(matches!(lowerer.op_metadata[1].kind, OpKind::GuardValue));
        assert_eq!(lowerer.op_metadata[1].reads, Register::ints(&[7]));
    }

    #[test]
    fn liveness_prebuild_emits_parent_markers_before_inline_helpers() {
        let helper_prebuild = quote! { helper_prebuild(__asm); };
        let tokens = liveness_prebuild_tokens(
            &[OpMeta::live_marker_with(Register::ints(&[3]), Vec::new())],
            &[helper_prebuild],
        )
        .to_string();
        let parent_pos = tokens
            .find("_register_liveness_offset")
            .expect("parent live marker should register a liveness offset");
        let helper_pos = tokens
            .find("helper_prebuild")
            .expect("nested helper prebuild should be present");
        assert!(
            parent_pos < helper_pos,
            "RPython assembles the caller before pending inline callees"
        );
    }

    #[test]
    fn inline_call_tokens_use_r_family_for_ref_only_args() {
        let tokens = inline_call_tokens_combined(&[binding(0, BindingKind::Ref)], 7);
        assert!(tokens.contains("inline_call_r_i"));
        assert!(tokens.contains("inline_call_r_r"));
        assert!(tokens.contains("inline_call_irf_f"));
        assert!(!tokens.contains("inline_call_ir_i"));
        assert!(!tokens.contains("inline_call_ir_r"));
        assert!(!tokens.contains("inline_call_irf_i"));
        assert!(!tokens.contains("inline_call_irf_r"));
    }

    #[test]
    fn inline_call_tokens_use_ir_family_when_any_int_arg_is_present() {
        let tokens = inline_call_tokens_combined(
            &[binding(0, BindingKind::Ref), binding(1, BindingKind::Int)],
            9,
        );
        assert!(tokens.contains("inline_call_ir_i"));
        assert!(tokens.contains("inline_call_ir_r"));
        assert!(tokens.contains("inline_call_irf_f"));
        assert!(!tokens.contains("inline_call_r_i"));
        assert!(!tokens.contains("inline_call_r_r"));
        assert!(!tokens.contains("inline_call_irf_i"));
        assert!(!tokens.contains("inline_call_irf_r"));
    }

    #[test]
    fn inline_call_tokens_use_irf_family_when_any_float_arg_is_present() {
        let tokens = inline_call_tokens_combined(
            &[binding(0, BindingKind::Int), binding(1, BindingKind::Float)],
            11,
        );
        assert!(tokens.contains("inline_call_irf_i"));
        assert!(tokens.contains("inline_call_irf_r"));
        assert!(tokens.contains("inline_call_irf_f"));
        assert!(!tokens.contains("inline_call_r_i"));
        assert!(!tokens.contains("inline_call_r_r"));
        assert!(!tokens.contains("inline_call_ir_i"));
        assert!(!tokens.contains("inline_call_ir_r"));
    }

    #[test]
    fn inline_call_tokens_emit_post_call_live_marker() {
        let (_, post_live) = inline_call_tokens(&[binding(0, BindingKind::Ref)], 7, &[]);
        let post = post_live.to_string();
        assert!(
            post.contains("live_placeholder"),
            "RPython jtransform.py emits inline_call followed by -live-"
        );
    }

    #[test]
    fn inline_helper_codegen_uses_canonical_r_surface() {
        let helper = generate_inline_helper_jitcode_with_calls(
            &parse_fn(
                r#"
                fn outer(arg: usize) -> usize {
                    callee(arg)
                }
                "#,
            ),
            &[inline_policy("callee")],
            // ref_params, ref_fields, array_fields, int_fields,
            // native_int_binops, native_tag_small, native_identity,
            // headerless_structs, inlined_prefix — the surface these tests
            // assert is the call encoding, which none of them participate in.
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .expect("jit_inline lowering should succeed")
        .expect("helper should lower");
        let body = helper.body.to_string();
        assert!(body.contains("inline_call_r_r"));
        assert!(!body.contains("inline_call_with_typed_args"));
    }

    /// A `native_identity` helper leaves no call behind: the argument's
    /// binding is what the consumer reads.  The call policies are empty, so a
    /// recognizer that failed to fire would reach the inference-failure panic
    /// before any assertion below runs.
    #[test]
    fn inline_helper_codegen_drops_a_native_identity_call() {
        let identity: syn::Path = syn::parse_str("unwrap_word").expect("path");
        let helper = generate_inline_helper_jitcode_with_calls(
            &parse_fn(
                r#"
                fn outer(arg: i64) -> i64 {
                    unwrap_word(arg)
                }
                "#,
            ),
            &[],
            // ref_params, ref_fields, array_fields, int_fields,
            // native_int_binops, native_tag_small.
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            std::slice::from_ref(&identity),
            // headerless_structs, inlined_prefix.
            &[],
            &[],
        )
        .expect("jit_inline lowering should succeed")
        .expect("helper should lower");
        let body = helper.body.to_string();
        assert!(
            !body.contains("call"),
            "a native_identity call must not reach the jitcode: {body}"
        );
    }

    #[test]
    fn inline_helper_codegen_uses_canonical_ir_surface() {
        let helper = generate_inline_helper_jitcode_with_calls(
            &parse_fn(
                r#"
                fn outer(lhs: usize, rhs: i64) -> i64 {
                    callee(lhs, rhs)
                }
                "#,
            ),
            &[inline_policy("callee")],
            // ref_params, ref_fields, array_fields, int_fields,
            // native_int_binops, native_tag_small, native_identity,
            // headerless_structs, inlined_prefix — the surface these tests
            // assert is the call encoding, which none of them participate in.
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .expect("jit_inline lowering should succeed")
        .expect("helper should lower");
        let body = helper.body.to_string();
        assert!(body.contains("inline_call_ir_i"));
        assert!(!body.contains("inline_call_with_typed_args"));
    }

    #[test]
    fn inline_helper_codegen_uses_canonical_irf_surface() {
        let helper = generate_inline_helper_jitcode_with_calls(
            &parse_fn(
                r#"
                fn outer(arg: f64) -> f64 {
                    callee(arg)
                }
                "#,
            ),
            &[inline_policy("callee")],
            // ref_params, ref_fields, array_fields, int_fields,
            // native_int_binops, native_tag_small, native_identity,
            // headerless_structs, inlined_prefix — the surface these tests
            // assert is the call encoding, which none of them participate in.
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .expect("jit_inline lowering should succeed")
        .expect("helper should lower");
        let body = helper.body.to_string();
        assert!(body.contains("inline_call_irf_f"));
        assert!(!body.contains("inline_call_with_typed_args"));
    }

    #[test]
    fn inline_helper_param_layout_uses_dense_per_kind_banks() {
        let func = parse_fn(
            r#"
            fn helper(ptr: usize, value: i64, scale: f64, other: usize, more: i64) -> i64 {
                value + more
            }
            "#,
        );
        let layout = inline_helper_param_layout(&func).expect("layout should build");
        assert_eq!(
            layout,
            vec![
                (InlineReturnKind::Ref, 0),
                (InlineReturnKind::Int, 0),
                (InlineReturnKind::Float, 0),
                (InlineReturnKind::Ref, 1),
                (InlineReturnKind::Int, 1),
            ]
        );
    }

    #[test]
    fn inline_helper_param_counts_match_dense_layout() {
        let func = parse_fn(
            r#"
            fn helper(ptr: usize, value: i64, scale: f64, other: usize, more: i64) -> i64 {
                value + more
            }
            "#,
        );
        let counts = inline_helper_param_counts(&func).expect("counts should build");
        assert_eq!(counts, (2, 2, 1));
    }

    #[test]
    fn explicit_inline_ref_policy_sets_ref_binding_kind() {
        let call = parse_call("callee()");
        let mut lowerer = Lowerer::new_with_call_policies(
            None,
            vec![(
                vec!["callee".to_string()],
                CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::InlineRef),
            )],
            InferenceFailureMode::Panic,
        );
        let binding = lowerer
            .lower_call_value(&call)
            .expect("inline ref call should lower");
        assert!(matches!(binding.kind, BindingKind::Ref));
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("add_sub_jitcode"));
        assert!(body.contains("__sub_return_kind"));
    }

    #[test]
    fn explicit_inline_float_policy_sets_float_binding_kind() {
        let call = parse_call("callee()");
        let mut lowerer = Lowerer::new_with_call_policies(
            None,
            vec![(
                vec!["callee".to_string()],
                CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::InlineFloat),
            )],
            InferenceFailureMode::Panic,
        );
        let binding = lowerer
            .lower_call_value(&call)
            .expect("inline float call should lower");
        assert!(matches!(binding.kind, BindingKind::Float));
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("add_sub_jitcode"));
        assert!(body.contains("__sub_return_kind"));
    }

    #[test]
    fn inline_pipeline_int_policy_resolves_callee_by_name() {
        let call = parse_call("stack_pop()");
        let mut lowerer = Lowerer::new_with_call_policies(
            None,
            vec![(
                vec!["stack_pop".to_string()],
                CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::InlinePipelineInt),
            )],
            InferenceFailureMode::Panic,
        );
        let binding = lowerer
            .lower_call_value(&call)
            .expect("inline-pipeline int call should lower");
        assert!(matches!(binding.kind, BindingKind::Int));
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        // The callee is resolved by name through the host convention symbol,
        // then attached as a sub-jitcode by Arc.
        assert!(body.contains("__majit_pipeline_jitcode"));
        assert!(body.contains("stack_pop"));
        assert!(body.contains("add_sub_jitcode_arc"));
        assert!(body.contains("__sub_return_kind"));
    }

    #[test]
    fn inline_pipeline_void_policy_resolves_callee_by_name() {
        let expr: syn::Expr = syn::parse_quote! { stack_swap() };
        let mut lowerer = Lowerer::new_with_call_policies(
            None,
            vec![(
                vec!["stack_swap".to_string()],
                CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::InlinePipelineVoid),
            )],
            InferenceFailureMode::Panic,
        );

        lowerer
            .lower_config_call_stmt(&expr)
            .expect("inline-pipeline void call should lower");

        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("__majit_pipeline_jitcode"));
        assert!(body.contains("stack_swap"));
        assert!(body.contains("add_sub_jitcode_arc"));
        let prebuild = &lowerer.inline_liveness_prebuild;
        let prebuild = quote! { #(#prebuild)* }.to_string();
        assert!(prebuild.contains("__majit_pipeline_liveness_prebuild"));
    }

    #[test]
    fn nested_branch_preserves_pipeline_liveness_prebuild() {
        let mut lowerer = Lowerer::new_with_call_policies(
            None,
            vec![(
                vec!["stack_pop".to_string()],
                CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::InlinePipelineInt),
            )],
            InferenceFailureMode::Panic,
        );
        let expr_if: syn::ExprIf = syn::parse_quote! {
            if 1 {
                let popped = stack_pop();
            }
        };

        lowerer
            .lower_if_stmt(&expr_if)
            .expect("branch containing a pipeline call should lower");

        let prebuild = &lowerer.inline_liveness_prebuild;
        let tokens = quote! { #(#prebuild)* }.to_string();
        assert!(tokens.contains("__majit_pipeline_liveness_prebuild"));
    }

    #[test]
    fn rejected_inline_arm_rolls_back_pipeline_liveness_prebuild() {
        let mut lowerer = Lowerer::new_with_call_policies(
            None,
            vec![(
                vec!["stack_pop".to_string()],
                CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::InlinePipelineInt),
            )],
            InferenceFailureMode::Panic,
        );
        let body: syn::Expr = syn::parse_quote! {{
            let popped = stack_pop();
            match 0 {
                (1, 2) => unsupported_body(),
                _ => 1,
            };
        }};

        assert_eq!(
            lowerer.try_inline_dispatch_arm(&body),
            dispatch::InlineArmOutcome::Rejected,
        );
        assert!(lowerer.inline_liveness_prebuild.is_empty());
    }

    /// A registered int-result target is dispatched by
    /// `bh_call_i_dispatch`, which reads the whole return register; a callee
    /// whose result is narrower than a word does not define it.  So the
    /// address that reaches `add_fn_ptr` must be the widening shim's, never
    /// the callee's own.
    #[test]
    fn int_result_call_registers_the_widening_shim() {
        let call = parse_call("helper(1)");
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ResidualInt);
        lowerer
            .lower_call_value(&call)
            .expect("residual int call should lower");
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("into_call_word"));
        assert!(!body.contains("add_fn_ptr (helper as * const ())"));
    }

    /// The shim stands in for the callee at the compiled call, so it has to
    /// have the signature that call emits: words in as well as a word out.
    /// Leaving a parameter at the callee's own type is invisible where a
    /// pointer is a word wide and a type error where it is not, so assert the
    /// spelling rather than the target it happens to run on.
    #[test]
    fn the_widening_shim_takes_its_arguments_as_words() {
        let call = parse_call("helper(1)");
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ResidualInt);
        lowerer
            .lower_call_value(&call)
            .expect("residual int call should lower");
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(
            body.contains("fn (i64) -> i64"),
            "the one-argument shim is spelled in words: {body}"
        );
        assert!(
            body.contains("from_call_word"),
            "and narrows each word to what the callee declares: {body}"
        );
    }

    /// Registering it as word-spelled is the other half: a backend whose word
    /// is wider than its pointer reflects over every target it was not told
    /// about, and the shim is exactly the target it can be told about.
    #[test]
    fn the_widening_shim_registers_as_word_spelled() {
        let call = parse_call("helper(1)");
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ResidualInt);
        lowerer
            .lower_call_value(&call)
            .expect("residual int call should lower");
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("add_word_abi_fn_ptr"), "{body}");
    }

    /// A ref result is a pointer and a void result is never read, so neither
    /// has anything to widen. The ref arm therefore registers the callee's own
    /// address; the void arm goes through the void registration, whose target
    /// is still that same address wherever a pointer already fills a word.
    #[test]
    fn non_int_result_calls_register_the_callee_address() {
        let call = parse_call("helper(1)");
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ResidualRef);
        lowerer
            .lower_call_value(&call)
            .expect("residual ref call should lower");
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("add_fn_ptr (helper as * const ())"));
        assert!(!body.contains("into_call_word"));

        let expr: syn::Expr = syn::parse_quote! { helper(1) };
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ResidualVoid);
        lowerer
            .lower_config_call_stmt(&expr)
            .expect("residual void call should lower");
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("add_word_abi_fn_ptr_void"), "{body}");
        assert!(body.contains("helper as * const ()"), "{body}");
        assert!(!body.contains("into_call_word"));
    }

    /// `record_known_result!` registers through `add_word_abi_fn_ptr_with_slot`,
    /// which takes the same shim for an int result.
    #[test]
    fn record_known_result_int_registers_the_widening_shim() {
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ElidableInt);
        lowerer
            .bindings
            .insert("known".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("record_known_result!(known, helper, arg)").expect("parse macro expr");
        lowerer
            .lower_record_known_result(&expr)
            .expect("record_known_result should lower");
        let body = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(body.contains("into_call_word"));
        assert!(!body.contains("add_fn_ptr_with_slot (helper as * const ()"));
    }
}
