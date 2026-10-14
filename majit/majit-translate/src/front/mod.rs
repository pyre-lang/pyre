//! Front-end scaffolding for semantic graph construction.
//!
//! ## Positioning
//!
//! This module bridges **Rust source (`syn::ItemFn`)** to the **`FunctionGraph`** type that the rest of the codewriter pipeline (`jtransform`, `flatten`, `regalloc`, `liveness`, `assembler`) consumes.
//!
//! RPython has no direct counterpart. In upstream, `rpython/jit/codewriter/codewriter.py` `CodeWriter.make_jitcodes()` (`:74-79`) takes only `verbose`: the graphs reach it through `self.callcontrol.enum_pending_graphs()`, already rtyped by `rpython/rtyper/` from RPython source. The codewriter never sees interpreter source files.
//!
//! pyre cannot inherit that assumption. Rust functions must become `FunctionGraph`s somewhere, and this module is where. Every file under `front/` is Rust-specific lowering that has no RPython structural match.
//!
//! ## Why this is the right layer
//!
//! - The boundary coincides with an upstream boundary: `FunctionGraph` is the line-by-line analogue of RPython `FlowGraph` / rtyper's post-translation graph form. Everything downstream (codewriter) consumes the same shape RPython consumes.
//! - Keeping the adapter inside `front/` means no new opnames, no new `OpKind`, no new jitcode keys leak into the parity layer.
//! - Upstream conventions (`SpaceOperation`, `Block.inputargs`, `Terminator`) are re-used without modification.
//!
//! ## Out of scope
//!
//! - New IR opkinds (`OpKind::TryOp` and similar) are forbidden here. `?` / `PyResult` must be lowered to exceptional successor edges of the existing `Terminator`, matching `rpython/translator/exceptiontransform.py` + `rpython/jit/codewriter/jtransform.py:rewrite_op_direct_call`.
//! - New JitCode key schemas (variant-keyed maps, opcode-to-fragment lookups) are forbidden. The canonical output is `{graph: JitCode}` per `rpython/jit/codewriter/call.py enum_pending_graphs` and `rpython/jit/codewriter/codewriter.py transform_func_to_jitcode`.
//!
//! ## Maintenance rule
//!
//! Every non-trivial addition to this module must include a comment citing the RPython file:line it replaces or bridges. If no such line exists, the addition is further pyre-specific deviation and must be justified explicitly in the commit message.
//!
//! ## Source of graphs: Charon-extracted MIR
//!
//! The semantic graphs are built from Charon-extracted ULLBC, not from
//! syn AST.  `front::mir::build_semantic_program_from_llbcs` consumes the
//! extracted artefacts and produces the `SemanticProgram` whose
//! `program.functions` carry the per-method graphs the rest of the
//! pipeline consumes.  `auto_discover_workspace_llbc_paths` in `lib.rs`
//! resolves `<workspace>/build/llbc/{pyre-object,pyre-interpreter,pyre-jit}.ullbc`
//! when `MAJIT_MIR_FRONTEND_LLBC` is unset; that canonical set is REQUIRED.
//! `build_semantic_program_via_active_frontend` panics when no LLBC source
//! resolves, so every build asserts that graphs come from MIR.
//!
//! Per-method graphs come straight from MIR.  `extract_trait_impls` /
//! `extract_inherent_impl_methods` index `program.functions` via
//! `front::semantic::MirGraphLookup` and substitute the MIR graph;
//! `SemanticFunction.self_ty_root` and `SemanticFunction.trait_root`,
//! populated on MIR-built impl methods (including trait-default bodies
//! detected by `trait_default_owner_for_fundecl`), drive that
//! registration.  `build_semantic_program_from_llbcs` dedups merged
//! entries by the full qualified path (`{module_path}::{name}`) so that
//! same-named impl methods across types are not collapsed.  The
//! per-callsite return type is taken from the MIR-built
//! `SemanticFunction` signature directly.  `tyref_to_value_type` maps
//! `Literal::{Int,UInt}` / atom `"Bool"` / atom `"Char"` so that
//! `usize` / `isize` arguments classify as integer rather than `Ref`,
//! satisfying `flatten.rs`'s `"switch exitswitch must be int"` for
//! graphs that switch on integer-typed arguments.
//!
//! ## dyn-Trait / impl-Trait classification
//!
//! Charon extracts every `dyn Trait` / `impl Trait` site in the
//! JIT-consumed crates (`pyre-object`, `pyre-interpreter`,
//! `pyre-module`) cleanly.  This driver targets the as-extracted ULLBC
//! and treats `Dynamic` calls as first-class opaque indirect calls
//! (type-flow devirtualization only if a hot path later demands it).
//! Dict dispatch is statically-resolved `Trait`-kind, not virtual: none
//! of the `Dynamic` fat-pointer calls in `pyre-interpreter.ullbc` are
//! `DictStrategy::*`.  A same-crate `&dyn` / `Box<dyn>` return is still
//! two words; `front::dyn_fat` splices that callee so the caller's
//! `method_*` read loads the vtable word.  The `Dynamic` call itself
//! stays an opaque indirect call.  No RPITIT / GAT / trait alias appears
//! in scope;
//! the only extraction gap is std `thread_local!` accessor stubs,
//! treated as opaque ops.

pub(crate) mod bigint_binop;
pub(crate) mod bool_then;
pub(crate) mod checked_arith;
pub(crate) mod checked_arith_uint;
pub(crate) mod clause_spec;
pub(crate) mod dyn_fat;
pub(crate) mod exc_from_raise;
pub(crate) mod from_raw_parts;
pub(crate) mod from_size_align;
#[cfg(any(test, feature = "mir-frontend"))]
pub(crate) mod graph_body;
pub mod host_layout;
pub(crate) mod iter_adapter;
pub(crate) mod iter_next;
pub mod llbc_hints;
pub mod mir;
pub(crate) mod option_closure_select;
pub(crate) mod option_ctor;
pub(crate) mod option_expect;
pub(crate) mod option_is_none;
pub(crate) mod option_map_or;
pub(crate) mod option_try;
pub(crate) mod option_unwrap;
pub(crate) mod option_unwrap_or;
pub(crate) mod range_contains;
pub(crate) mod range_iter;
pub(crate) mod rbigint_call;
pub(crate) mod result_as_ref;
pub(crate) mod result_exc;
pub(crate) mod result_map_err;
pub(crate) mod rfloat_call;
pub(crate) mod saturating_add;
pub(crate) mod saturating_mul;
pub(crate) mod saturating_sub;
pub mod semantic;
pub(crate) mod slice_first;
pub(crate) mod slice_get;
pub(crate) mod slice_index;
pub(crate) mod std_identity;
pub(crate) mod str_find;
pub mod typestr;

pub use semantic::{
    AstGraphOptions, FieldRow, SemanticFunction, SemanticProgram, StructFieldRegistry,
};
