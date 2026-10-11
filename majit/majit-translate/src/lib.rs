//! majit-translate: RPython translation pipeline.
//!
//! Upstream counterparts:
//! - `codewriter/` ← `rpython/jit/codewriter/`
//! - `flowspace/`, `annotator/`, `rtyper/`, `translator/` — line-by-line
//!   ports of `rpython/{flowspace,annotator,rtyper,translator}/`
//! - `translator/rtyper/legacy_{annotator,resolve,pipeline}.rs` —
//!   transitional pyre-only adapters with no upstream counterpart;
//!   slated for retirement once Skip categories close
//!   (see [`crate::translator::rtyper::cutover`])
//!
//! Everything under one crate because upstream has circular imports
//! (flowspace/operation.py ↔ annotator, annotator/annrpython.py ↔
//! rtyper/normalizecalls.py) that Cargo's DAG crate boundary cannot model.

#[cfg_attr(
    test,
    expect(
        clippy::mutable_key_type,
        clippy::type_complexity,
        reason = "translator tests intentionally exercise RPython identity-keyed Variable maps and literal annotation signatures; Variable hashes only its immutable ID, and wrapping the upstream shapes would hide parity"
    )
)]
pub mod annotator;
pub mod artifacts;
#[cfg_attr(
    test,
    expect(
        clippy::mutable_key_type,
        clippy::arc_with_non_send_sync,
        reason = "codewriter tests preserve RPython object identity: Variables hash only immutable IDs and JitCode Arc values are shared identity handles confined to the single-threaded translation run"
    )
)]
pub mod codewriter;
pub use majit_config as config;
// Decline census — no upstream counterpart.  Upstream gates that cannot
// lower a shape raise a named error (`jtransform.py _handle_list_call`);
// every gate here declines silently, so the refusals are counted instead.
pub mod decline;
#[cfg_attr(
    test,
    expect(
        clippy::mutable_key_type,
        reason = "flowspace Variable keys compare and hash exclusively by immutable identity ID; interior annotation state is deliberately excluded, matching RPython's object-identity dictionaries"
    )
)]
pub mod flowspace;
#[cfg_attr(
    test,
    expect(
        clippy::mutable_key_type,
        reason = "these tests use Variable as RPython identity-dictionary keys; equality and hashing depend only on the immutable Variable ID"
    )
)]
pub mod tool;
pub use codewriter::{
    assembler, call, flatten, format, insns, jitcode, jtransform, liveness, policy, support,
};
pub use tool::algo::regalloc;

#[cfg_attr(
    test,
    expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc is the established shared identity handle for generated JitCode; code generation is single-thread confined and never transfers its non-Send construction payload across threads"
    )
)]
mod codegen;
#[cfg_attr(
    test,
    expect(
        clippy::type_complexity,
        reason = "the graph-body census retains Charon's literal nested block/body map shape so extraction and source-translation parity remain directly auditable"
    )
)]
pub mod codeobject_layout;
pub mod fat_ptr_layout;
pub mod front;
pub mod vec_layout;
// TODO(pyre): pyre-interpreter handler JitCode registry
// (Rust source → FunctionGraph bridge with no RPython counterpart;
// upstream assumes rtyper-produced `translator.graphs` is already in
// memory at codewriter entry).
pub mod generated;
pub mod hints;
pub mod inline;
pub mod layout;
mod local_crates;
#[cfg_attr(
    test,
    expect(
        clippy::type_complexity,
        reason = "the model test spells the exact nested RPython graph shape under test; a one-use alias would conceal rather than simplify that contract"
    )
)]
pub mod memory;
pub mod model;
pub mod model_ssa;
pub mod module_path;
mod parse;
#[cfg_attr(
    test,
    expect(
        clippy::arc_with_non_send_sync,
        reason = "the pipeline returns Arc<JitCode> as a stable shared identity handle across generated artifacts; all mutation and use remain within the single-threaded translator"
    )
)]
pub mod pipeline;
mod runtime_names;
#[cfg(test)]
pub(crate) mod test_support;
mod unsimplify;
pub mod virtualizable_decl;
// `translator/` is the RPython-orthodox port home — see
// `translator/mod.rs` for the contract.  Currently hosts
// `translator/rtyper/{rclass.rs, rpbc.rs}`, the `rpython/rtyper/` 1:1
// port, alongside the transitional `legacy_{annotator,resolve,pipeline}.rs`
// adapters that the cutover (`translator/rtyper/cutover.rs`) consumes
// until the real-rtyper path types every production graph end-to-end.
#[cfg_attr(
    test,
    expect(
        clippy::mutable_key_type,
        clippy::arc_with_non_send_sync,
        clippy::type_complexity,
        reason = "rtyper parity tests intentionally retain RPython identity dictionaries, shared identity nodes, and literal nested signatures; keys hash immutable IDs and the translator is single-thread confined"
    )
)]
pub mod translator;

pub use call::{CallDescriptor, StructFieldLayout, StructLayout};
pub use codewriter::type_state::ConcreteType;
pub use flatten::{FlatOp, GraphFlattener, Label, RegKind, SSARepr, flatten_graph};
pub use front::{AstGraphOptions, SemanticFunction, SemanticProgram};
pub use jtransform::{
    CallEffectKind, CallEffectOverride, DeclaredCallEffects, DeclaredExtraEffect,
    GraphTransformConfig, GraphTransformResult, StructStorageDescriptor,
    VirtualizableFieldDescriptor, transform_graph,
};
pub use layout::{HeuristicLayoutProvider, LayoutProvider};
pub use model::{Block, BlockId, CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};
pub use parse::CallPath;
pub use pipeline::{
    CompiledJitDriver, JitDriverSpec, PipelineConfig, PipelineResult, ProgramPipelineResult,
};

use serde::{Deserialize, Serialize};

// Enables focused tracing of prepass nondeterminism.
pub(crate) fn determinism_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("MAJIT_DETERMINISM_TRACE").is_some_and(|value| value == "1")
    })
}

/// Configuration for the canonical graph/pipeline analyzer.
///
/// Consumers supply graph-rewrite metadata such as virtualizable
/// field/array mappings before the codewriter-style passes run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeConfig {
    pub pipeline: PipelineConfig,
}

/// Trait implementation info
#[derive(Debug, Serialize, Deserialize)]
pub struct TraitImplInfo {
    pub trait_name: String,
    pub for_type: String,
    #[serde(default)]
    pub self_ty_root: Option<String>,
    pub methods: Vec<MethodInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MethodInfo {
    pub name: String,
    /// Canonical semantic graph for this method when available.
    #[serde(skip, default)]
    pub graph: Option<model::LazyGraph>,
    /// RPython: op.result.concretetype — return type for array identity.
    #[serde(default)]
    pub return_type: Option<String>,
    /// RPython: function-level JIT hints (elidable, close_stack, etc.).
    #[serde(default)]
    pub hints: Vec<String>,
}

/// Feature-gated SemanticProgram builder.
///
/// When the `mir-frontend` feature is enabled, route the build
/// through [`front::mir::build_semantic_program_from_llbcs`] using
/// LLBC artefacts discovered via, in priority order:
///
/// 1. Paths supplied by an explicit-LLBC analyzer entry point. These are the
///    complete input set and disable every fallback below.
/// 2. `MAJIT_MIR_FRONTEND_LLBC` env-var (OS path-list: `;`-separated on
///    Windows, `:`-separated elsewhere). Explicit override for CI /
///    test fixtures targeting a specific LLBC set.
///
/// Panics when neither source resolves. Artifact selection belongs to the
/// consumer because only the consumer knows which crates form its program.
fn build_semantic_program_via_active_frontend(
    module_paths: &[&str],
    static_addrs: HostStaticAddrs<'_>,
    jitdriver_receiver_roots: &[String],
    explicit_llbc_paths: Option<&[&str]>,
    funcobj_declarations: &call::FuncObjDeclarations,
    lowering_skips: &front::mir::LoweringSkips,
    prof: &mut PhaseProfiler,
) -> (
    front::SemanticProgram,
    std::collections::HashSet<majit_ir::descr::StructId>,
) {
    #[cfg(feature = "mir-frontend")]
    {
        // Accept an OS path-list so a consumer can pass its LLBC set in one
        // environment variable.
        // `std::env::split_paths` uses the platform separator (`;` on
        // Windows, `:` elsewhere) so a Windows drive letter like `Z:`
        // is not mistaken for a separator.  The single-path form also
        // works.
        //
        let resolved_paths: Option<Vec<String>> = explicit_llbc_paths
            .filter(|paths| !paths.is_empty())
            .map(|paths| paths.iter().map(|path| (*path).to_string()).collect())
            .or_else(|| {
                std::env::var_os("MAJIT_MIR_FRONTEND_LLBC")
                    .map(|v| {
                        std::env::split_paths(&v)
                            .filter(|p| !p.as_os_str().is_empty())
                            .map(|p| p.to_string_lossy().into_owned())
                            .collect()
                    })
                    // A present-but-blank override (`MAJIT_MIR_FRONTEND_LLBC=`)
                    // collects to an empty vec; treat it as unset instead of
                    // feeding an empty LLBC set into the frontend.
                    .filter(|paths: &Vec<String>| !paths.is_empty())
            });
        if let Some(paths) = resolved_paths {
            // The harvest loop parses one artefact at a time and drops it.
            // The second loop reloads each file after
            // `register_transparent_scalar_kinds` so cross-crate
            // `repr(transparent)` scalar returns resolve, and hands it to
            // the `GraphBodyProvider`, which keeps it and the state its
            // decls were lowered with.
            let mut discovered = Vec::new();
            let mut foldable_cross = Vec::new();
            let mut foldable_impl_by_ord = Vec::new();
            let mut duplicate_leaf_facts = front::mir::DuplicateLeafFacts::default();
            let mut crate_names = Vec::new();
            let mut hints: std::collections::HashMap<String, Vec<String>> =
                std::collections::HashMap::new();
            let mut immutable_fields = std::collections::HashMap::new();
            let mut unsafe_fn_stubs = Vec::new();
            let mut foreign_opaque_method_externals = Vec::new();
            let mut eval_hook_graphs = Vec::new();
            let mut declared_gc = front::mir::DeclaredGcFacts::default();
            // Root-stack effects, harvested in link order: each artefact is
            // analysed with the effects of the crates it depends on already
            // published, so a call into one of them reads that body's answer.
            let mut root_stack_crates: Vec<String> = Vec::new();
            let mut root_stack_touching: Vec<String> = Vec::new();
            // Paths are in dependency order, so a crate's callees from
            // earlier artefacts are already classified when it is.
            let mut stack_sensitive: Vec<String> = Vec::new();
            let mut stack_depth_neutral: Vec<String> = Vec::new();
            let mut stack_leaves_above: Vec<String> = Vec::new();
            let mut stack_param_slots: Vec<(String, Vec<u8>)> = Vec::new();
            let mut stack_returns_index: Vec<String> = Vec::new();
            let mut stack_observes: Vec<String> = Vec::new();
            for p in &paths {
                let llbc = majit_charon_reader::Llbc::load(p)
                    .unwrap_or_else(|e| panic!("Step 4.4 cutover: load {p}: {e}"));
                prof.mark(&format!("    harvest {p}"));
                llbc.register_stack_sensitive_fns(stack_sensitive.iter().cloned());
                llbc.register_stack_depth_neutral_fns(stack_depth_neutral.iter().cloned());
                llbc.register_stack_leaves_above_fns(stack_leaves_above.iter().cloned());
                llbc.register_stack_param_slots_fns(stack_param_slots.iter().cloned());
                llbc.register_stack_returns_index_fns(stack_returns_index.iter().cloned());
                llbc.register_stack_observes_fns(stack_observes.iter().cloned());
                let (sens, leaves, params, ret_idx, observes) =
                    front::mir::discover_stack_fn_effects(&llbc);
                stack_observes.extend(observes);
                stack_sensitive.extend(sens);
                stack_leaves_above.extend(leaves);
                stack_param_slots.extend(params);
                stack_returns_index.extend(ret_idx);
                llbc.register_stack_sensitive_fns(stack_sensitive.iter().cloned());
                llbc.register_stack_leaves_above_fns(stack_leaves_above.iter().cloned());
                llbc.register_stack_param_slots_fns(stack_param_slots.iter().cloned());
                llbc.register_stack_returns_index_fns(stack_returns_index.iter().cloned());
                llbc.register_stack_observes_fns(stack_observes.iter().cloned());
                stack_depth_neutral.extend(front::mir::discover_depth_neutral_fns(&llbc));
                crate_names.push(llbc.crate_name().to_string());
                llbc.set_root_stack_effects(root_stack_crates.clone(), root_stack_touching.clone());
                root_stack_touching.extend(front::mir::harvest_root_stack_touching_paths(&llbc));
                root_stack_crates.push(llbc.crate_name().to_string());
                discovered.extend(front::mir::discover_transparent_scalar_kinds(&llbc));
                duplicate_leaf_facts.absorb(front::mir::DuplicateLeafFacts::discover(&llbc));
                declared_gc.absorb(front::mir::harvest_declared_gc_facts(&llbc));
                let mut impl_folds = Vec::new();
                for (path, op) in front::mir::discover_foldable_const_lits(&llbc) {
                    if path.contains('<') {
                        impl_folds.push((path, op));
                    } else {
                        foldable_cross.push((path, op));
                    }
                }
                foldable_impl_by_ord.push(impl_folds);
                for (k, v) in
                    front::llbc_hints::harvest_hints_from_llbcs(std::slice::from_ref(&llbc))
                {
                    hints.entry(k).or_default().extend(v);
                }
                for (k, v) in front::llbc_hints::harvest_immutable_fields_from_llbcs(
                    std::slice::from_ref(&llbc),
                ) {
                    immutable_fields.entry(k).or_insert(v);
                }
                for path in front::mir::discover_eval_hook_graphs(&llbc) {
                    if !eval_hook_graphs.contains(&path) {
                        eval_hook_graphs.push(path);
                    }
                }
            }
            discovered.sort_by(|a, b| a.0.cmp(&b.0));
            discovered.dedup();
            let foldable_cross = front::mir::register_foldable_const_lits(foldable_cross);
            let cross_tombstoned_leaves = duplicate_leaf_facts.tombstoned_leaves();
            crate::local_crates::register_local_crate_roots(crate_names);

            let mut graph_bodies = front::graph_body::GraphBodyProvider::new(
                static_addrs,
                jitdriver_receiver_roots,
                hints.clone(),
                funcobj_declarations.clone(),
                lowering_skips.clone(),
            );
            let mut merged = None;
            let mut seen_function_keys = std::collections::HashSet::new();
            let mut seen_struct_names = std::collections::HashSet::new();
            let mut seen_trait_names = std::collections::HashSet::new();
            let gc_struct_ids = declared_gc.gc_struct_ids();
            for (ord, p) in paths.iter().enumerate() {
                let llbc = majit_charon_reader::Llbc::load(p)
                    .unwrap_or_else(|e| panic!("Step 4.4 cutover: load {p}: {e}"));
                llbc.set_exception_carrier(static_addrs.error_carrier.carrier_path);
                if !eval_hook_graphs.is_empty() {
                    llbc.set_eval_hook_graphs(eval_hook_graphs.clone());
                }
                llbc.set_root_stack_effects(root_stack_crates.clone(), root_stack_touching.clone());
                llbc.register_transparent_scalar_kinds(discovered.iter().cloned());
                llbc.register_stack_sensitive_fns(stack_sensitive.iter().cloned());
                llbc.register_stack_depth_neutral_fns(stack_depth_neutral.iter().cloned());
                llbc.register_stack_leaves_above_fns(stack_leaves_above.iter().cloned());
                llbc.register_stack_param_slots_fns(stack_param_slots.iter().cloned());
                llbc.register_stack_returns_index_fns(stack_returns_index.iter().cloned());
                llbc.register_stack_observes_fns(stack_observes.iter().cloned());
                llbc.mark_stack_sensitive_fns_complete();
                front::mir::attach_foldable_const_lits(&llbc, &foldable_cross);
                front::mir::attach_foldable_const_lits(&llbc, &foldable_impl_by_ord[ord]);
                front::mir::register_ambiguous_impl_foldable_const_lits(&llbc);
                unsafe_fn_stubs.extend(front::mir::collect_unsafe_fn_stubs_from_llbc(
                    &llbc,
                    &gc_struct_ids,
                    static_addrs.error_carrier,
                ));
                unsafe_fn_stubs.extend(front::mir::collect_policy_opaque_fn_stubs_from_llbc(
                    &llbc,
                    &gc_struct_ids,
                    static_addrs.error_carrier,
                ));
                unsafe_fn_stubs
                    .extend(front::mir::collect_marked_class_ctor_stubs_from_llbc(&llbc));
                foreign_opaque_method_externals
                    .extend(front::mir::collect_foreign_opaque_method_externals(&llbc));
                let prog = graph_bodies.lower_prelinked_crate(
                    llbc,
                    module_paths,
                    &cross_tombstoned_leaves,
                    &gc_struct_ids,
                );
                prof.mark(&format!("    lower {p}"));
                front::mir::absorb_semantic_program(
                    &mut merged,
                    prog,
                    &mut seen_function_keys,
                    &mut seen_struct_names,
                    &mut seen_trait_names,
                    &front::mir::semantic_function_dedup_key,
                );
            }
            let mut program = merged.unwrap_or_else(|| front::SemanticProgram {
                functions: Vec::new(),
                harvested_hints: std::collections::HashMap::new(),
                known_struct_names: std::collections::HashSet::new(),
                known_trait_names: std::collections::HashSet::new(),
                struct_fields: front::semantic::StructFieldRegistry::default(),
                immutable_fields: std::collections::HashMap::new(),
                enum_variant_by_discriminant: std::collections::HashMap::new(),
                struct_origins: std::collections::HashMap::new(),
                struct_field_attrs: std::collections::HashMap::new(),
                exact_layouts: std::collections::HashMap::new(),
                struct_ids: std::collections::HashMap::new(),
                unsafe_fn_stubs: Vec::new(),
                foreign_opaque_method_externals: Vec::new(),
                atomic_load_decls: Vec::new(),
            });
            front::mir::harden_duplicate_leaf_metadata(
                &mut program.struct_fields,
                &mut program.struct_origins,
                &mut program.enum_variant_by_discriminant,
                Some(&program.struct_ids),
            );
            program.harvested_hints = hints;
            program.immutable_fields = immutable_fields;
            program.unsafe_fn_stubs = unsafe_fn_stubs;
            program.foreign_opaque_method_externals = foreign_opaque_method_externals;
            return (program, gc_struct_ids);
        }
    }
    let _ = (
        module_paths,
        static_addrs,
        jitdriver_receiver_roots,
        explicit_llbc_paths,
        funcobj_declarations,
        lowering_skips,
        &mut *prof,
    );
    // The MIR front-end is the only graph builder.  Reaching this
    // point means neither an explicit path set nor
    // `MAJIT_MIR_FRONTEND_LLBC` supplied an LLBC source. Surface the
    // misconfiguration immediately.
    panic!(
        "no LLBC source resolved.\n\
         Pass explicit LLBC paths to the analyzer, or set \
         `MAJIT_MIR_FRONTEND_LLBC` to an OS path-list \
         (`;`-separated on Windows, `:` elsewhere) explicitly."
    );
}

/// `make_virtualizable_infos` constructor closure type — mirrors the
/// upstream `VirtualizableInfo(self, VTYPEPTR)` call (warmspot.py).
/// `(jd_idx, vtypeptr_token) -> Option<handle>`.  Hosts that own a
/// runtime `VirtualizableInfo` impl supply the constructor here.
pub type VirtualizableInfoFactory<'a> =
    dyn Fn(usize, &str) -> Option<std::sync::Arc<dyn call::VirtualizableInfoHandle>> + 'a;

/// Optional binding table from macro-generated helper path
/// (`module_path!()::helper_name`) to the compiled trace-call address.
///
/// `#[jit_module]::__majit_helper_trace_fnaddrs()` produces this shape.
/// `analyze_pipeline_from_module_paths` strips the crate-name prefix and binds
/// both canonical aliases (`helpers::foo` and `crate::helpers::foo`) on
/// `CallControl` before `get_jitcode()` / `jtransform` query fnaddrs.
pub type FnAddrBindings<'a> = [(&'a str, i64)];

/// Structured binding table for impl-method helpers.  Each entry is
/// `(module_path_with_crate, impl_type_as_written, method_name, fnaddr)`.
/// The codewriter applies the
/// `CallControl::register_macro_impl_helper_trace_fnaddr` qualification
/// rule — bare types get the module prefix (minus crate
/// name) prepended, qualified types are kept verbatim — before storing
/// the canonical `[impl_type_joined, method]` 2-segment CallPath.
///
/// `#[jit_module]::__majit_helper_impl_trace_fnaddrs()` produces this
/// shape and `analyze_pipeline_from_module_paths` feeds it through
/// `CallControl::register_macro_impl_helper_trace_fnaddr`.
pub type ImplFnAddrBindings<'a> = [(&'a str, &'a str, &'a str, i64)];

/// Identity of the interpreter's fallible-return carrier — the `E` of the
/// `Result<T, E>` that [`front::result_exc`] lowers into exception edges.
///
/// That pass is otherwise carrier-agnostic graph surgery; these four fields
/// are the whole of what names one interpreter's error type.  The consumer
/// supplies its own the way `PipelineConfig::register_trait_families` takes
/// a consumer's trait names.  There is no default identity to fall back on:
/// see the [`Default`] impl.
#[derive(Debug, Clone, Copy)]
pub struct ErrorCarrierSpec<'a> {
    /// Qualified ADT path of `E`, compared after peeling
    /// [`Self::carrier_wrappers`].
    pub carrier_path: &'a str,
    /// Classdef of the raised object: the handle's `Deref::Target`.
    /// Empty: the handle [`Self::carrier_path`] is itself the class.
    /// Result matching still uses [`Self::carrier_path`].
    pub carrier_class: &'a str,
    /// ADT paths to peel off `E` before that compare, each read through the
    /// wrapper's own first type argument.  Applied outermost-first and at
    /// most once per entry, so `&["alloc::boxed::Box"]` takes
    /// `Box<InterpError>` to `InterpError`.  Empty for a bare carrier.
    pub carrier_wrappers: &'a [&'a str],
    /// Free function materialising a raised carrier into the runtime
    /// exception value (`BH_LAST_EXC_VALUE`'s domain), emitted by the
    /// codewriter at every raise of the carrier
    /// (`codewriter::error_carrier_edges`).  `None` when the carrier already
    /// *is* one such value — a single owned word needs no materialisation
    /// and the raise stores it unchanged.
    pub to_exc_object: Option<&'a [&'a str]>,
    /// The inverse, `(receiver_root, method)`, emitted where a handler
    /// catches the carrier.  `None` for the same reason and with the same
    /// effect as on [`Self::to_exc_object`].
    pub from_exc_object: Option<(&'a str, &'a str)>,
}

/// No carrier declared.
///
/// This layer ships none: naming one interpreter's error type here is the
/// thing the spec exists to undo, and it is also the boundary this subtree
/// keeps against a consumer's object model. A pipeline that has not named a
/// carrier has no `Result` to lower, so an empty `carrier_path` matches no
/// ADT and [`front::result_exc`] leaves every graph alone — the same outcome
/// as running with the pass switched off, rather than a silently different
/// interpreter's shape.
impl Default for ErrorCarrierSpec<'_> {
    fn default() -> Self {
        Self {
            carrier_path: "",
            carrier_class: "",
            carrier_wrappers: &[],
            to_exc_object: None,
            from_exc_object: None,
        }
    }
}

impl ErrorCarrierSpec<'_> {
    /// Classdef key of the raised object. [`Self::carrier_class`] when
    /// named, otherwise the handle [`Self::carrier_path`].
    pub fn classdef_path(&self) -> &str {
        if self.carrier_class.is_empty() {
            self.carrier_path
        } else {
            self.carrier_class
        }
    }
}

/// Owned mirror of [`ErrorCarrierSpec`], for a holder that outlives the
/// caller frame the spec borrows from: a demanded body has to lower with the
/// spec the whole-program pass used, and the codewriter's representation of
/// the carrier's exception edges reads it after the front has returned.
#[derive(Debug, Default, Clone)]
pub struct OwnedErrorCarrierSpec {
    pub carrier_path: String,
    pub carrier_class: String,
    pub carrier_wrappers: Vec<String>,
    pub to_exc_object: Option<Vec<String>>,
    pub from_exc_object: Option<(String, String)>,
}

impl OwnedErrorCarrierSpec {
    pub fn own(spec: ErrorCarrierSpec<'_>) -> Self {
        let own_path = |segments: &[&str]| -> Vec<String> {
            segments.iter().map(|s| (*s).to_string()).collect()
        };
        Self {
            carrier_path: spec.carrier_path.to_string(),
            carrier_class: spec.carrier_class.to_string(),
            carrier_wrappers: own_path(spec.carrier_wrappers),
            to_exc_object: spec.to_exc_object.map(own_path),
            from_exc_object: spec
                .from_exc_object
                .map(|(receiver, method)| (receiver.to_string(), method.to_string())),
        }
    }
}

/// The value bank a [`ScalarFieldStore`] writes.
///
/// Only the two scalar banks are spellable: this boundary exists for a field
/// the callee borrows as a Rust `&mut` primitive, and a reference is not one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBank {
    Int,
    Float,
}

/// A consumer function whose whole effect is one scalar field store —
/// `(target, value) -> ()`, writing `value` into one field of the object
/// `target` points at.
///
/// The lowering cannot reach such a store by descending into the callee: the
/// body borrows a Rust `&mut i64` / `&mut f64` into the field, and a pointer
/// to a primitive has no low-level repr here, while rebuilding the container
/// around it would allocate over a field that is inline. So the consumer
/// names the boundary and the call lowers to the `setfield_gc_<bank>` the
/// body performs.
///
/// Declared here rather than matched by name inside the lowering for the
/// reason [`ErrorCarrierSpec`] gives: naming one interpreter's functions in
/// this layer is what the spec exists to undo. This layer ships none.
#[derive(Debug, Clone, Copy)]
pub struct ScalarFieldStore<'a> {
    /// The function's qualified path, spelled as `item_meta.name_path()`
    /// does — the same string the callee-side gates compare against.
    pub function_path: &'a str,
    /// Crate-stripped owner root of the written field: the
    /// variant-qualified `module::Enum::Variant` for a variant payload, or
    /// `module::Struct` for a plain field. The same key the layout pass
    /// installs, so the emitted descr matches the one a `target.field = v`
    /// assignment in the consumer's own source would carry.
    pub owner_root: &'a str,
    /// The field's name within that owner, `__pos_<N>` for a positional one.
    pub field: &'a str,
    /// Which bank the stored scalar belongs to.
    pub bank: ScalarBank,
}

/// Host-supplied addresses of prebuilt object-space singletons that pyre
/// source carries through the flowgraph as opaque `LOAD_GLOBAL`
/// constants (the static `PyType` pointers and dict-strategy refs).
///
/// `majit-translate` is the `rpython/` translation layer and must not
/// import the `pyre-object` object space; the driver
/// (`pyre-jit-trace/build.rs`, via `pyre_interpreter::jit_static_*_addrs()`)
/// supplies these prebuilt-instance addresses across the translation
/// boundary, exactly as `rpython/jit` receives `Constant(GCREF)` from the
/// host instead of importing `pypy/objspace`.  Resolved in the same
/// build-script process that runs the translator, so the addresses match
/// a direct `&pyre_object::X` read at the codewriter call site.
///
/// `pytypes` are recorded as `ValueType::Int`, `refs` as
/// `ValueType::Ref(None)`, matching the front-end `KnownStaticsCatalogue`
/// classification.  Empty (the `Default`) for test / legacy entry points
/// that lower fixtures not referencing these singletons.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostStaticAddrs<'a> {
    pub pytypes: &'a [(&'a str, i64)],
    /// The same class singletons as [`Self::pytypes`], keyed by the Rust
    /// path of the TYPE that owns each one rather than by the path of the
    /// static holding it.
    ///
    /// `#[pyre_class]` emits one `PyType` address under two names: the
    /// static, and the `PyreClassPyTypeOf::PYTYPE` associated const.
    /// Charon renders the second `<module>::<Impl>::PYTYPE`, and that
    /// rendering is not injective — every trait impl in a module flattens
    /// onto it — so no key in `pytypes` can name one of them without also
    /// naming its siblings. The reader resolves that family through the
    /// impl's identity instead (`ItemMeta::trait_impl_id` -> the impl's
    /// `Self` type -> that ADT's path) and looks the answer up here, which
    /// is why this bucket exists rather than more `pytypes` rows.
    ///
    /// This key is injective: a Rust type has one path. That matters
    /// beyond this lookup, because it is also the linkage symbol
    /// `runtime_fnaddr_patch::patch_static_addr_constants` re-pairs across
    /// the build/run boundary — the one place a name is unavoidable, and
    /// where `rpython/translator/gensupp.py`'s `NameManager.uniquename`
    /// holds upstream to the same requirement by numbering collisions.
    pub pytypes_by_struct: &'a [(&'a str, i64)],
    pub refs: &'a [(&'a str, i64)],
    /// Immutable size `const`s baked as their compile-time value
    /// (`ValueType::Int` `ConstInt`), keyed by crate-stripped
    /// `module::NAME`.  Distinct from `refs`: the value, not the address.
    pub int_values: &'a [(&'a str, i64)],
    /// Which `Result<T, E>` the exception-link lowering owns.  Carried here
    /// because this struct already reaches both `front::result_exc`
    /// consumers — the callee gate in
    /// `lower_unstructured_with_static_addrs_and_attrs` and the caller
    /// capture on `Lowering` — so no consumer grows a parameter for it.
    pub error_carrier: ErrorCarrierSpec<'a>,
    /// Boundaries the consumer declares as one scalar field store each.
    /// Empty (the `Default`) declares none, which leaves every call to be
    /// lowered by descending into its callee as usual.
    pub scalar_field_stores: &'a [ScalarFieldStore<'a>],
}

/// Multi-file analysis with explicit per-source module paths.
///
/// `module_paths[i]` is the crate-stripped module path of the i-th
/// analyzed source file (e.g. `"intobject"` for
/// `pyre_object/src/intobject.rs`).  The graph surface itself comes
/// from the Charon-extracted LLBC set; the `module_paths` slice drives
/// per-file lexical resolution. Source text is not
/// consumed — callers pass paths only.
///
/// An empty `module_paths[i]` keeps simple-name registration only —
/// runtime convergence is then handled solely by the
/// `build_object_descr_group_with_def_path` dual-publish.
pub fn analyze_multiple_pipeline_with_modules(
    module_paths: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    static_addrs: HostStaticAddrs<'_>,
) -> pipeline::ProgramPipelineResult {
    analyze_pipeline_from_module_paths(
        module_paths,
        None,
        &[],
        config,
        layout_provider,
        vinfo_factory,
        fnaddr_bindings,
        &[],
        static_addrs,
        None,
    )
}

/// Multi-file analysis with an explicit, ordered LLBC input set.
///
/// Consumer build scripts should prefer this entry point over the legacy
/// `MAJIT_MIR_FRONTEND_LLBC` compatibility environment variable. The supplied
/// paths are the complete translation input and never fall back to the
/// environment variable.
///
/// `policy` is the target's `jitpolicy` (`driver.py task_pyjitpl_lltype`
/// hands `self.jitpolicy` to `apply_jit`); `None` is `JitPolicy()`.
pub fn analyze_multiple_pipeline_from_llbc_with_modules(
    llbc_paths: &[&str],
    module_paths: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    static_addrs: HostStaticAddrs<'_>,
    policy: Option<&mut dyn policy::JitPolicy>,
) -> pipeline::ProgramPipelineResult {
    assert!(
        !llbc_paths.is_empty(),
        "explicit LLBC analysis requires at least one input artifact"
    );
    analyze_pipeline_from_module_paths(
        module_paths,
        Some(llbc_paths),
        &[],
        config,
        layout_provider,
        vinfo_factory,
        fnaddr_bindings,
        &[],
        static_addrs,
        policy,
    )
}

/// A registration of the funcobj `graph` that stamps the source return type
/// (`funcptr._obj.TO.RESULT`) and then `hints` onto its copy of the graph.
fn lazy_graph_source(
    graph: &model::LazyGraph,
    return_type: &Option<String>,
    hints: &[String],
) -> call::GraphSource {
    call::GraphSource::Lazy {
        graph: graph.clone(),
        transform: call::GraphTransform {
            return_type: return_type.clone(),
            hints: hints.to_vec(),
        },
    }
}

/// Register a free-function graph under one alias path.  Panics if the
/// same alias is already mapped to a different `func.name` — this is
/// the parity guard against silent cross-crate name-tail collisions.
/// Same-name re-registration (e.g. a function reachable through more
/// than one well-known crate alias) is treated as idempotent.
/// `graph` is shared, not copied, across the ~`4 + 2 * local_crate_roots`
/// spellings [`free_function_alias_paths`] produces for one funcobj —
/// `bookkeeper.getdesc` binds every alias to the same graph object, and a
/// per-alias deep copy of `FunctionGraph.blocks` multiplied the whole
/// free-function graph set by the alias count.
fn register_function_graph_alias(
    graphs: &mut std::collections::HashMap<crate::parse::CallPath, call::GraphSource>,
    sources: &mut std::collections::HashMap<crate::parse::CallPath, String>,
    path: crate::parse::CallPath,
    source_name: &str,
    graph: &call::GraphSource,
) {
    if let Some(prev) = sources.get(&path) {
        assert!(
            prev == source_name,
            "function-graph alias collision at {}: previously registered by {prev:?}, now {source_name:?}; \
             cross-crate name-tail aliasing must not silently route to a different graph",
            path.canonical_key(),
        );
        return;
    }
    sources.insert(path.clone(), source_name.to_string());
    graphs.insert(path, graph.clone());
}

/// Register each `(name, source)` from
/// [`pipeline::PipelineConfig::builtin_graphs`].
///
/// `support.py` `builtin_func_for_spec` / `inline_calls_to` key the helper
/// graph on the single-segment impl name (`setup_extra_builtin`).
/// `register_function_graph_alias` keeps the first free function that owns
/// that leaf, and the hint re-registration overwrites the call-control
/// entry, so this runs after that pass. The seed lookup in
/// `CallControl::find_all_graphs_bfs` and the route-(a) `function_path`
/// are that one segment; nothing looks up `crate::name` for these helpers.
fn pin_builtin_graphs(
    graphs: &std::collections::HashMap<crate::parse::CallPath, call::GraphSource>,
    bindings: &[(String, crate::parse::CallPath)],
    call_control: &mut call::CallControl,
) {
    for (name, source) in bindings {
        let Some(graph) = graphs.get(source) else {
            continue;
        };
        call_control.register_function_graph(
            crate::parse::CallPath::from_segments([name.as_str()]),
            graph.clone(),
        );
    }
}

/// Compute the full alias spelling set for a free function lifted
/// from a Rust source.  Mirrors the graph-alias loop in
/// [`analyze_pipeline_from_module_paths`] so call-site lookups that key on
/// these spellings (function_graphs,
/// elidable/loopinvariant/cannot_collect/oopspec targets) all see
/// the same FunctionPath set.  Without this, a module-qualified call
/// site finds the graph alias but misses the elidable/loopinvariant
/// hint registered only under the bare name.
///
/// `name` is the `SemanticFunction.name` (already module-prefixed
/// when the function lives inside a `mod foo { fn bar() }` block);
/// `source_module` is the file's crate-stripped path (populated by
/// `front::mir` from the module portion of Charon's `name_path()`).
fn free_function_alias_paths(name: &str, source_module: &str) -> Vec<crate::parse::CallPath> {
    let segments: Vec<&str> = name.split("::").collect();
    let mut paths = Vec::new();
    paths.push(crate::parse::CallPath::from_segments(
        segments.iter().copied(),
    ));
    let mut crate_segs = vec!["crate"];
    crate_segs.extend(segments.iter().copied());
    paths.push(crate::parse::CallPath::from_segments(crate_segs));
    let crate_roots = crate::local_crates::local_crate_roots();
    for crate_alias in &crate_roots {
        let mut alias_segs = vec![crate_alias.as_str()];
        alias_segs.extend(segments.iter().copied());
        paths.push(crate::parse::CallPath::from_segments(alias_segs));
    }
    let name_segs_prefix: Vec<&str> = name.split("::").collect();
    let mod_segs_prefix: Vec<&str> = source_module.split("::").collect();
    // The starts_with check is meant to skip the module-qualified loop
    // when `name` already carries the module prefix (e.g. a nested
    // `mod foo { fn bar() }` whose `sf.name` is set to "foo::bar"
    // by `front::mir`).  Without
    // the length-strict guard, a function whose bare leaf happens to
    // equal its containing module's name (`pyre-interpreter/src/
    // stack_check.rs` `pub fn stack_check`) collides — its single-
    // segment name "stack_check" `starts_with` ["stack_check"] is
    // true, so the loop is skipped and the
    // `["module", "name"]` / `["crate", "module", "name"]` / aliased
    // spellings never get registered, leaving every call site that
    // writes `crate::stack_check::stack_check` looking for an
    // unregistered path.  Require `name` to be STRICTLY LONGER than
    // `module` to count as already-prefixed.
    let name_already_prefixed = name_segs_prefix.len() > mod_segs_prefix.len()
        && name_segs_prefix.starts_with(&mod_segs_prefix);
    if !source_module.is_empty() && !name_already_prefixed {
        let module_segs: Vec<&str> = source_module.split("::").collect();
        let mut module_qualified_segs: Vec<&str> = module_segs.clone();
        module_qualified_segs.extend(segments.iter().copied());
        paths.push(crate::parse::CallPath::from_segments(
            module_qualified_segs.iter().copied(),
        ));
        let mut crate_module_segs = vec!["crate"];
        crate_module_segs.extend(module_qualified_segs.iter().copied());
        paths.push(crate::parse::CallPath::from_segments(
            crate_module_segs.iter().copied(),
        ));
        for crate_alias in &crate_roots {
            let mut alias_segs = vec![crate_alias.as_str()];
            alias_segs.extend(module_qualified_segs.iter().copied());
            paths.push(crate::parse::CallPath::from_segments(
                alias_segs.iter().copied(),
            ));
        }
    }
    paths
}

/// `MAJIT_PROFILE_PIPELINE=1` phase profiler: elapsed wall time plus the
/// process high-water RSS at each mark, so a phase's contribution to the
/// translation prepass's peak footprint is visible next to its cost in time.
pub struct PhaseProfiler {
    enabled: bool,
    start: std::time::Instant,
    last: std::time::Instant,
}

impl PhaseProfiler {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            enabled: std::env::var_os("MAJIT_PROFILE_PIPELINE").is_some(),
            start: now,
            last: now,
        }
    }

    pub fn mark(&mut self, name: &str) {
        if !self.enabled {
            return;
        }
        let now = std::time::Instant::now();
        const GB: f64 = (1u64 << 30) as f64;
        eprintln!(
            "[MAJIT_PROFILE_PIPELINE] {:>9.3}s  {:>9.3}s  live {:>6.2}GB  peak {:>6.2}GB  {name}",
            (now - self.start).as_secs_f64(),
            (now - self.last).as_secs_f64(),
            live_rss_bytes() as f64 / GB,
            peak_rss_bytes() as f64 / GB,
        );
        self.last = now;
    }

    /// Emit a free-form line under the same gate as [`Self::mark`], for
    /// per-phase input sizes that explain a phase's cost. Takes a closure so
    /// counting the sizes costs nothing when the profiler is off.
    pub fn note(&self, message: impl FnOnce() -> String) {
        if self.enabled {
            eprintln!("[MAJIT_PROFILE_PIPELINE] {}", message());
        }
    }
}

impl Default for PhaseProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// `getrusage(RUSAGE_SELF).ru_maxrss` — the process high-water RSS. Monotonic,
/// so the difference between two marks is what the span between them added to
/// the peak. macOS reports bytes, Linux kilobytes.
#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    let raw = usage.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

/// Current resident set size. Unlike [`peak_rss_bytes`] this falls back when a
/// phase releases memory, so the pair separates "this phase allocated N" from
/// "this phase retains N". Shells out to `ps` because the in-process query is
/// per-platform (`proc_pidinfo` on macOS, `/proc/self/statm` on Linux) and the
/// profiler samples fewer than a dozen times per run.
#[cfg(unix)]
fn live_rss_bytes() -> u64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p"])
        .arg(std::process::id().to_string())
        .output();
    let Ok(output) = output else { return 0 };
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

#[cfg(not(unix))]
fn live_rss_bytes() -> u64 {
    0
}

/// Give every `struct_fields` spelling of a declared struct the same
/// `_immutable_fields_` entry list.
///
/// `rclass.py _parse_field_list` reads the declaration off the class
/// object, which has one identity. Pyre's registry keys layouts by name and
/// carries several names for one struct — the bare leaf (`PyType`) and the
/// crate-qualified path (`pyre_object::pyobject::PyType`) — while the harvester
/// registers only the leaf and the crate-stripped path. Since layouts are
/// stored per [`majit_ir::descr::StructId`], an undeclared spelling could
/// overwrite the declared one's layout and silently drop `is_immutable`, and
/// which spelling won depended on `HashMap` iteration order, so the
/// declaration took effect on some builds and not others.
///
/// Declared keys are visited in sorted order so that a struct reachable under
/// several declared spellings resolves to the same entry list on every run.
fn expand_immutable_fields_to_all_spellings(
    declared: &std::collections::HashMap<String, Vec<(String, model::ImmutableRank)>>,
    struct_fields: &front::StructFieldRegistry,
) -> std::collections::HashMap<String, Vec<(String, model::ImmutableRank)>> {
    let mut by_struct_id: std::collections::HashMap<
        majit_ir::descr::StructId,
        &Vec<(String, model::ImmutableRank)>,
    > = std::collections::HashMap::new();
    let mut declared_names: Vec<&String> = declared.keys().collect();
    declared_names.sort();
    for name in declared_names {
        if let Some(sid) = majit_ir::descr::struct_template_id_for_name(name) {
            by_struct_id.entry(sid).or_insert(&declared[name]);
        }
    }
    let mut out = declared.clone();
    for name in struct_fields.fields.keys() {
        if out.contains_key(name) {
            continue;
        }
        let Some(entries) = majit_ir::descr::struct_template_id_for_name(name)
            .and_then(|sid| by_struct_id.get(&sid))
        else {
            continue;
        };
        out.insert(name.clone(), (*entries).clone());
    }
    out
}

fn struct_layout_census_enabled() -> bool {
    std::env::var_os("MAJIT_STRUCT_LAYOUT_CENSUS")
        .is_some_and(|value| value == std::ffi::OsStr::new("1"))
}

/// Count distinct struct *identities* per bare leaf for unique trait-impl
/// devirtualization.
///
/// RPython's bookkeeper keys classes by the class object itself, so several
/// lookup spellings for one class never make it ambiguous.  The LLBC metadata
/// deliberately publishes full-crate, crate-relative, and bare aliases for
/// one Rust type; count their shared [`majit_ir::descr::StructId`] once.  A
/// qualified spelling whose identity is missing remains fail-closed by using
/// its own path-derived id.  The small per-leaf `Vec` mirrors identity lookup
/// without introducing an unordered side set.
///
/// The caller drops any unique-impl entry whose owner collapses to a leaf
/// this reports more than once: such an entry could seed the receiver with
/// the OTHER same-named class.  Multiple lookup aliases of one `StructId`
/// are one RPython class object and must not make that class ambiguous.
fn distinct_struct_identities_by_leaf(
    program: &front::SemanticProgram,
) -> std::collections::HashMap<String, Vec<majit_ir::descr::StructId>> {
    let mut by_leaf: std::collections::HashMap<String, Vec<majit_ir::descr::StructId>> =
        std::collections::HashMap::new();
    for key in program.struct_fields.fields.keys() {
        let Some((_, leaf)) = key.rsplit_once("::") else {
            continue;
        };
        let identity = program
            .struct_ids
            .get(key)
            .copied()
            .flatten()
            .unwrap_or_else(|| majit_ir::descr::StructId::from_canonical(key));
        let identities = by_leaf.entry(leaf.to_string()).or_default();
        if !identities.contains(&identity) {
            identities.push(identity);
        }
    }
    by_leaf
}

type TraitImplOwners = std::collections::HashMap<String, std::collections::BTreeSet<String>>;

type TraitMethodOverrides<'a> =
    std::collections::HashMap<(&'a str, &'a str), (&'a str, &'a front::semantic::SemanticFunction)>;

/// Struct identity of one impl-owner spelling.
///
/// `struct_ids` stores `None` for a bare leaf two modules share; that
/// spelling is not one class.  A qualified spelling with no registry
/// entry keeps its own path-derived id, so it cannot be merged into a
/// different class that happens to use the same leaf.
fn struct_id_for_trait_owner(
    program: &front::SemanticProgram,
    owner: &str,
    leaf_identities: &std::collections::HashMap<String, Vec<majit_ir::descr::StructId>>,
) -> Option<majit_ir::descr::StructId> {
    match program.struct_ids.get(owner) {
        Some(Some(id)) => return Some(*id),
        Some(None) => return None,
        None => {}
    }
    let leaf = owner.rsplit("::").next().unwrap_or(owner);
    let identities = leaf_identities.get(leaf);
    if !owner.contains("::") && identities.is_some_and(|ids| ids.len() > 1) {
        return None;
    }
    if let Some(id) = identities.and_then(|ids| (ids.len() == 1).then_some(ids[0])) {
        return Some(id);
    }
    Some(majit_ir::descr::StructId::from_canonical(owner))
}

/// One [`majit_ir::descr::StructId`] when every owner spelling names that
/// class, otherwise `None` (no impl, or two classes).
fn collapse_owner_identities(
    program: &front::SemanticProgram,
    owners: &[&str],
    leaf_identities: &std::collections::HashMap<String, Vec<majit_ir::descr::StructId>>,
) -> Option<majit_ir::descr::StructId> {
    if owners.is_empty() {
        return None;
    }
    let mut ids = Vec::new();
    for owner in owners {
        let id = struct_id_for_trait_owner(program, owner, leaf_identities)?;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    if ids.len() == 1 { Some(ids[0]) } else { None }
}

/// Record one concrete `impl Trait for Type` method on the override maps.
///
/// The key is `(trait leaf, method name)`.  Owner strings are not part of
/// the key: several lookup spellings of one class last-win the same
/// method, and [`unique_trait_default_override`] decides whether those
/// spellings are one [`majit_ir::descr::StructId`].
fn index_concrete_trait_method<'a>(
    func: &'a front::semantic::SemanticFunction,
    trait_concrete_impl_types: &mut std::collections::HashMap<&'a str, Vec<&'a str>>,
    trait_method_overrides: &mut TraitMethodOverrides<'a>,
) {
    let (Some(owner), Some(trait_leaf)) =
        (func.self_ty_root.as_deref(), func.trait_root.as_deref())
    else {
        return;
    };
    let types = trait_concrete_impl_types.entry(trait_leaf).or_default();
    if !types.contains(&owner) {
        types.push(owner);
    }
    trait_method_overrides.insert((trait_leaf, func.name.as_str()), (owner, func));
}

/// The unique concrete override of a trait default, when every owner
/// spelling of that trait is one struct identity and that identity
/// implements this method.
///
/// `classdesc.py lookup` walks the receiver's MRO and a concrete method
/// shadows the base-class (trait-default) body.  A generic
/// `<E: Trait>` call site lowers to `[<Trait>, <method>]`.  Lookup
/// aliases of one class are one RPython class object, so they must not
/// keep the default.  Two struct identities leave the receiver ambiguous
/// and keep the default (indirect-call family).  A method the unique
/// identity does not override also keeps the default.
fn unique_trait_default_override<'a>(
    program: &front::SemanticProgram,
    leaf_identities: &std::collections::HashMap<String, Vec<majit_ir::descr::StructId>>,
    is_default: bool,
    trait_leaf: &str,
    method_name: &str,
    trait_method_overrides: &'a TraitMethodOverrides<'a>,
    trait_concrete_impl_types: &std::collections::HashMap<&str, Vec<&str>>,
) -> Option<(&'a str, &'a front::semantic::SemanticFunction)> {
    if !is_default {
        return None;
    }
    let owners = trait_concrete_impl_types.get(trait_leaf)?;
    let impl_id = collapse_owner_identities(program, owners, leaf_identities)?;
    let (owner, func) = trait_method_overrides
        .get(&(trait_leaf, method_name))
        .copied()?;
    let owner_id = struct_id_for_trait_owner(program, owner, leaf_identities)?;
    if owner_id != impl_id {
        return None;
    }
    Some((owner, func))
}

/// Graph registered on `[<Trait>, <method>]` for one trait-default body.
///
/// The unique identity's override replaces the default.  `lookup_impl_method`
/// may miss when an inherent method shares the name; the override's own
/// graph is that method, so it is the fallback.
fn resolved_trait_default_graph(
    lookup: &front::semantic::MirGraphLookup<'_>,
    method_name: &str,
    default_graph: &model::LazyGraph,
    default_return: &Option<String>,
    devirt: Option<(&str, &front::semantic::SemanticFunction)>,
) -> call::GraphSource {
    let (source, ret) = match devirt {
        Some((impl_type, info)) => (
            lookup
                .lookup_impl_method(impl_type, method_name)
                .unwrap_or_else(|| info.lazy_graph()),
            &info.return_type,
        ),
        None => (default_graph, default_return),
    };
    lazy_graph_source(source, ret, &[])
}

/// Bind `[<Trait>, <method>]` for every trait default in `program`.
///
/// CallKind::Trait lowers to that direct path.  The pseudo-type path
/// `[<default methods of Trait>, <method>]` stays the base-class body
/// for impls that do not override the method.
fn bind_trait_default_direct_paths(
    program: &front::SemanticProgram,
    call_control: &mut call::CallControl,
    leaf_identities: &std::collections::HashMap<String, Vec<majit_ir::descr::StructId>>,
) {
    let lookup = front::semantic::MirGraphLookup::from_program(program);
    let mut trait_concrete_impl_types: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    let mut trait_method_overrides: TraitMethodOverrides<'_> = std::collections::HashMap::new();
    for func in &program.functions {
        index_concrete_trait_method(
            func,
            &mut trait_concrete_impl_types,
            &mut trait_method_overrides,
        );
    }
    for func in &program.functions {
        let (None, Some(trait_leaf)) = (func.self_ty_root.as_deref(), func.trait_root.as_deref())
        else {
            continue;
        };
        let devirt = unique_trait_default_override(
            program,
            leaf_identities,
            true,
            trait_leaf,
            func.name.as_str(),
            &trait_method_overrides,
            &trait_concrete_impl_types,
        );
        let direct = resolved_trait_default_graph(
            &lookup,
            func.name.as_str(),
            func.lazy_graph(),
            &func.return_type,
            devirt,
        );
        call_control.register_function_graph(
            crate::parse::CallPath::from_segments([trait_leaf, func.name.as_str()]),
            direct,
        );
    }
}

/// Resolve a trait's sole concrete owner to the struct root used by the
/// annotator, preserving class identity across the LLBC registry's aliases.
fn unique_trait_impl_roots(
    program: &front::SemanticProgram,
    trait_impl_owners: TraitImplOwners,
) -> std::collections::HashMap<String, String> {
    let struct_leaf_identities = distinct_struct_identities_by_leaf(program);
    trait_impl_owners
        .into_iter()
        .filter_map(|(trait_qualified, owners)| {
            // Collapse lookup aliases to one StructId before the uniqueness
            // test.  Two spellings of one class are one RPython class object.
            let owner_refs: Vec<&str> = owners.iter().map(String::as_str).collect();
            let owner_id =
                collapse_owner_identities(program, &owner_refs, &struct_leaf_identities)?;
            // RPython carries the class / lltype object, so every spelling of
            // one identity reaches one descriptor cache entry.  The Rust
            // pipeline still transports a string at this seam; choose the
            // fully crate-qualified registry spelling deterministically so a
            // method owner such as `option::Option` and its SizeDescr owner
            // `core::option::Option` cannot split one field object by name.
            let canonical_owner = program
                .struct_fields
                .fields
                .keys()
                .filter(|candidate| {
                    program.struct_ids.get(*candidate).copied().flatten() == Some(owner_id)
                })
                .max_by(|left, right| {
                    left.matches("::")
                        .count()
                        .cmp(&right.matches("::").count())
                        .then_with(|| left.len().cmp(&right.len()))
                        .then_with(|| right.cmp(left))
                })
                .cloned()?;
            Some((trait_qualified, canonical_owner))
        })
        .collect()
}

/// Generate only explicit helper roots for a macro-owned portal. Unlike the
/// full portal API this does not manufacture JIT-driver metadata. This is a
/// Rust consumer adapter; it uses the same codewriter graph transformation.
/// Host bindings and static/error-carrier metadata are supplied by the consumer
/// just as for portal analysis, including callees reached only from a helper.
pub fn analyze_helper_pipeline_with_modules(
    module_paths: &[&str],
    helper_roots: &[CallPath],
    config: &AnalyzeConfig,
    fnaddr_bindings: &FnAddrBindings<'_>,
    impl_fnaddr_bindings: &ImplFnAddrBindings<'_>,
    static_addrs: HostStaticAddrs<'_>,
) -> pipeline::ProgramPipelineResult {
    assert!(
        !helper_roots.is_empty(),
        "helper analysis requires explicit roots"
    );
    assert!(
        config.pipeline.jit_drivers.is_empty(),
        "helper analysis has no portal"
    );
    analyze_pipeline_from_module_paths(
        module_paths,
        None,
        helper_roots,
        config,
        None,
        &|_, _| None,
        fnaddr_bindings,
        impl_fnaddr_bindings,
        static_addrs,
        None,
    )
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
fn analyze_pipeline_from_module_paths(
    module_paths: &[&str],
    explicit_llbc_paths: Option<&[&str]>,
    helper_roots: &[CallPath],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    impl_fnaddr_bindings: &ImplFnAddrBindings<'_>,
    static_addrs: HostStaticAddrs<'_>,
    policy: Option<&mut dyn policy::JitPolicy>,
) -> pipeline::ProgramPipelineResult {
    // Dump the decline census when this run ends — on the normal return
    // AND on the unwind, since a pipeline that panics on an undigestible
    // shape is exactly the run whose silent refusals a reader needs.
    // Silent unless `MAJIT_DECLINE_LOG` / `PYRE_MIR_FRONTEND_DEBUG` is
    // set, so default output is unchanged.
    let _decline_census = decline::CensusScope::new("analyze_pipeline");
    let mut prof = PhaseProfiler::new();
    macro_rules! mark_phase {
        ($name:literal) => {
            prof.mark($name)
        };
    }
    mark_phase!("entry");
    // `FORCE_ATTRIBUTES_INTO_CLASSES` is seeded from the LLBC-sourced
    // `program.struct_field_attrs` further below, once `program` is
    // built.
    // RPython `translator/translator.py buildflowgraph` — FlowingError
    // propagates out and translation halts.  Pyre's top-level analyzer
    // requires a complete program; a FlowingError here means a user-
    // facing source file contains a construct we cannot yet lower, and
    // the correct response is to abort loudly so the coverage audit
    // surfaces the unsupported expression rather than silently dropping
    // a graph.
    // When the `mir-frontend` feature is enabled, route the production
    // SemanticProgram build through the MIR-driven
    // `front::mir::build_semantic_program_from_llbcs` path.  The LLBC
    // source is a Charon-extracted .ullbc snapshot (produced by
    // `scripts/extract-llbc.py`), supplied explicitly by the consumer or
    // located via `MAJIT_MIR_FRONTEND_LLBC`.
    mark_phase!("known_statics + struct_field_attrs populated");
    // The funcobjs the front end declares apart from `program.functions`,
    // registered by `CallControl` on lookup. A clause specialization is
    // another graph of its generic funcobj (`description.py
    // FunctionDesc.cachedgraph(key)`), reached only from the call sites
    // that name it: it is declared under that one path, with no alias
    // spelling, class member or indirect-call family row of its own.
    let funcobj_declarations = call::FuncObjDeclarations::default();
    let lowering_skips = front::mir::LoweringSkips::default();
    // `rlib/jit.py`'s `hint` entry reads `classdesc.get_param('_virtualizable_')`
    // off the class before it mints `access_directly`. Pyre has no `ClassDesc`
    // at that point, so the declaration arrives with the config — and it is
    // already there: every `VirtualizableFieldDescriptor` the consumer builds
    // carries the `owner_root` that declares it and the field it lists. Derive
    // the registry from that
    // rather than taking a second, out-of-band declaration, so the set the
    // front end reads is a property of THIS invocation's config instead of
    // whatever the thread was last told. A config that declares no owner
    // leaves the set empty, which fails closed at the minter's class test.
    crate::virtualizable_decl::register_virtualizable_declarations(
        crate::virtualizable_decl::declarations_from_config(&config.pipeline.transform),
    );
    let (mut program, declared_gc) = build_semantic_program_via_active_frontend(
        module_paths,
        static_addrs,
        &config.pipeline.transform.jitdriver_receiver_roots,
        explicit_llbc_paths,
        &funcobj_declarations,
        &lowering_skips,
        &mut prof,
    );
    // Publish the `(bare struct leaf → defining crate-relative module
    // path)` map into the process-global `STRUCT_ORIGIN_REGISTRY` so the
    // later `codewriter` `canonical_struct_name` `path_hash` sites
    // resolve bare struct tokens to their qualified canonical form (PyPy
    // `bookkeeper.getdesc(TYPE)` analog).  Derived from the LLBC
    // `iter_type_decls()` name paths in
    // `front::mir::derive_program_metadata`; any leaf absent from the map
    // still resolves through the runtime's simple-name dual-publish slot.
    majit_ir::descr::register_struct_origins(program.struct_origins.clone());
    // Seed the name → StructId resolver so the layout consumers that only
    // hold a string (`llmemory::FieldOffset`'s `st._name`, a nested
    // field's rendered type) can reach the identity-keyed layout maps.
    majit_ir::descr::register_struct_ids(program.struct_ids.clone());
    // Tier-3: seed `FORCE_ATTRIBUTES_INTO_CLASSES` (classdesc.py)
    // from the LLBC-sourced `program.struct_field_attrs` so
    // `ClassDesc::_init_classdef` pre-fills `ClassDef.attrs` before the
    // annotator's `attrs_populated` narrowing gate
    // (`flowspace_adapter.rs::derive_subject_inputcells`).  Replaces the
    // syn `pre_register_struct_fields_from_file` walk.
    // `struct_field_attrs` is the Charon `derive_program_metadata`
    // projection, keyed by the crate-stripped qualified item path
    // (`intobject::W_IntObject`).  Register each entry under that single
    // canonical key only.  The consumer reads by `cls.qualname()` (the
    // bare struct leaf) and canonicalises it through the
    // `STRUCT_ORIGIN_REGISTRY` (`canonical_struct_name`) before lookup,
    // so no bare-leaf alias is needed — dropping it keeps a leaf shared
    // by distinct modules (e.g. two `FrameBlock`s with different field
    // shapes) from clobbering each other.  Iterate in sorted key order
    // for build determinism.
    {
        let mut entries: Vec<_> = program.struct_field_attrs.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (qualified, fields) in entries {
            let layout = program
                .struct_fields
                .fields
                .get(qualified)
                .map(Vec::as_slice);
            crate::annotator::classdesc::register_struct_fields_with_layout(
                qualified, fields, layout,
            );
        }
    }
    mark_phase!("build_semantic_program_from_parsed_files");
    // A clause specialization is another graph of its generic funcobj
    // (`description.py FunctionDesc.cachedgraph(key)`), reached only from
    // the call sites that name it: it is declared under that one path, with
    // no alias spelling, class member or indirect-call family row of its
    // own.
    prof.note(|| {
        // Counts the bodies built so far; the note builds none.
        let built = || {
            program
                .functions
                .iter()
                .map(|f| f.lazy_graph())
                .filter(|graph| graph.is_built())
                .filter_map(|graph| graph.get())
        };
        format!(
            "  program: {} functions ({} built: {} blocks, {} ops), {} struct_fields, {} type layouts",
            program.functions.len(),
            built().count(),
            built().map(|g| g.blocks.len()).sum::<usize>(),
            built()
                .flat_map(|g| g.blocks.iter())
                .map(|b| b.operations.len())
                .sum::<usize>(),
            program.struct_fields.fields.len(),
            program.exact_layouts.len(),
        )
    });
    // The rtyper synthesises a resizable-list `GcStruct("list", ("length",
    // Signed), ("items", Ptr(GcArray(ITEM))), hints={'list': True})`
    // (`translator/rtyper/rlist.rs`).  Because it is synthesised by
    // the rtyper — not a Rust source type Charon extracts — it never appears
    // in `program.struct_fields`, so `fielddescrof`/`compute_struct_size`
    // return None/0 for owner "list" and any `new(descr)` keyed on it would
    // trip the `bh_size_spec_from_callcontrol().unwrap_or_else(panic!)` in
    // `codewriter/assembler.rs`.  Register the {length, items} shape here,
    // before both the `set_struct_fields` snapshot and the
    // `struct_layouts` loop below, so both see it.  Offsets accumulate
    // by field order via `get_type_flag`: `i64`→(Signed,8) and a leading-`&`
    // spelling→(Pointer,Ref,8), giving length@0, items@8, struct size 16 — the
    // natural word-sized GcStruct layout the backend's `build_ll_newlist`
    // malloc uses (length-first, items-second, both word-sized;
    // `rlist.rs`).  `.entry().or_insert_with` leaves a real "list"
    // entry untouched if one ever exists.  The `items` type is a bare pointer
    // spelling: the field only has to classify as `(Pointer, Ref, word)`, and
    // the real element type is inert here — it lives on the op's `item_ty`
    // (the `new_array_clear` / `newlist_clear` arraydescr), never in this
    // struct layout, since the resized list header is element-uniform.
    program
        .struct_fields
        .fields
        .entry("list".to_string())
        .or_insert_with(|| {
            vec![
                ("length".to_string(), "i64".to_string()),
                ("items".to_string(), "&()".to_string()),
            ]
        });
    let mut canonical_trait_impls = Vec::new();
    let mut canonical_inherent_methods = Vec::new();
    // `(trait_leaf, trait_qualified, method_name, owner, return_type,
    // hints)` for every concrete trait-impl method — input to the
    // single-impl devirtualization pass below.  `trait_qualified` is
    // the trait's full `name_path()`; the leaf keys the
    // direct-path-binding bookkeeping (matching `canonical_trait_impls`
    // spelling), the qualified path keys the unique-impl map.
    #[allow(clippy::type_complexity)]
    let mut concrete_trait_methods: Vec<(
        String,
        String,
        String,
        String,
        Option<String>,
        Vec<String>,
        crate::model::LazyGraph,
    )> = Vec::new();
    let mut canonical_function_graphs: std::collections::HashMap<
        crate::parse::CallPath,
        call::GraphSource,
    > = std::collections::HashMap::new();
    // `bookkeeper.py getdesc` / `newfuncdesc` keys on the host
    // function-object identity, so two unrelated `crate_a::helper` and
    // `crate_b::helper` resolve to distinct `FunctionDesc` instances.
    // The call registry is keyed on `CallPath` segment strings, and the
    // alias expansion below intentionally registers each free function under
    // every crate root in the loaded LLBC set. Without a
    // collision check the second registration silently overwrites the
    // first when two source crates happen to define a function with
    // the same tail segments, so the call resolver may then route to
    // the wrong graph.  `canonical_function_alias_source` records the
    // canonical `func.name` that won each alias slot so a later
    // mismatched registration panics with a diagnostic instead of
    // silently aliasing across crates.
    let mut canonical_function_alias_source: std::collections::HashMap<
        crate::parse::CallPath,
        String,
    > = std::collections::HashMap::new();

    // Source impl-method registration from the MIR-lowered
    // `program.functions`.  Each `SemanticFunction` carries
    // `self_ty_root` / `trait_root` populated to the exact registration
    // keys the downstream loops below consume, so those loops are
    // agnostic to where the `canonical_*` vectors come from.
    //
    // `program.functions` is a superset of every graph-carrying
    // method the source surface defines.  Trait methods with no
    // graph are intentional non-JIT targets — `#[cfg(test)]` impls,
    // `into_py` on primitive receivers Charon cannot key to an ADT,
    // residual `PyreBlackholeAllocator::*` allocator hooks, std-trait
    // impls (`Drop`/`From`/`PartialEq`/…), and `BoxEnv` default bodies
    // MIR does not lower.  They carry no graph (so the BFS never
    // reaches them) and their `return_types` are redundant with the
    // CFG-scan result kind (`graph_result_kind` = `getkind(FUNC.RESULT)`,
    // `codewriter.rs`) — the calldescr builder reads the declared
    // type only as an `i↔r` tiebreak and `debug_assert`s it against the
    // CFG kind; no impl method diverges, so omitting the side-table
    // entries is a no-op for call-descriptor typing.
    //
    // `mir_graph_lookup` is consulted by the registration loops below
    // (trait-default vs concrete-impl graph fetch).
    let mir_graph_lookup = front::semantic::MirGraphLookup::from_program(&program);
    // `classdesc.py lookup` MRO walk for generic trait dispatch:
    // a concrete override shadows the trait default.  `front::mir`
    // lowers `CallKind::Trait` (a call through a generic `<E: Trait>`
    // receiver) to the direct path `[<Trait>, <method>]`, and the
    // registration loop below binds that path for BFS discovery and
    // `get_jitcode` resolution.  Collect each trait's concrete impl
    // types and per-method overrides so the direct-path registration
    // can prefer the unique override: with exactly one concrete impl
    // in the analyzed program the annotator binds the receiver to
    // that instantiation, so the MRO finds the override before the
    // default body.  With several concrete impl types the receiver
    // stays ambiguous and the default-body registration is kept
    // (indirect-call family territory).
    let mut trait_concrete_impl_types: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    let mut trait_method_overrides: std::collections::HashMap<
        (&str, &str),
        (&str, &front::semantic::SemanticFunction),
    > = std::collections::HashMap::new();
    for func in &program.functions {
        match (&func.self_ty_root, &func.trait_root) {
            // Concrete trait-impl method: `impl Trait for Type { fn m }`.
            // Registration-wise it behaves exactly like an inherent
            // method (`[Owner, method]` graph + hints below); routing it
            // through `register_trait_method` instead would also seed
            // `method_to_impl_types`, flipping `resolve_method`'s
            // name-based lookup for every same-named method.  The trait
            // identity feeds two complementary direct-path consumers:
            // the `concrete_trait_methods` loop below binds the unique
            // impl when the trait method has NO default body, and the
            // `trait_method_overrides` lookup inside the default-body
            // registration prefers the unique override over the default
            // (`classdesc.py` — a concrete override shadows the
            // trait default in the MRO walk).
            (Some(owner), Some(trait_leaf)) => {
                concrete_trait_methods.push((
                    trait_leaf.clone(),
                    func.trait_qualified
                        .clone()
                        .unwrap_or_else(|| trait_leaf.clone()),
                    func.name.clone(),
                    owner.clone(),
                    func.return_type.clone(),
                    func.hints.clone(),
                    func.lazy_graph().clone(),
                ));
                index_concrete_trait_method(
                    func,
                    &mut trait_concrete_impl_types,
                    &mut trait_method_overrides,
                );
                canonical_inherent_methods.push(parse::InherentMethodInfo {
                    for_type: owner.clone(),
                    self_ty_root: Some(owner.clone()),
                    name: func.name.clone(),
                    graph: func.lazy_graph().clone(),
                    return_type: func.return_type.clone(),
                    hints: func.hints.clone(),
                });
            }
            // Trait default-body: `trait T { fn m { … } }`.  The pseudo
            // `for_type` matches the extractor's sentinel so the loop's
            // `is_default` branch and the `call.rs` resolve-method
            // filters keep distinguishing default from concrete impl.
            (None, Some(trait_leaf)) => {
                canonical_trait_impls.push(TraitImplInfo {
                    trait_name: trait_leaf.clone(),
                    for_type: format!("<default methods of {}>", trait_leaf),
                    self_ty_root: None,
                    methods: vec![MethodInfo {
                        name: func.name.clone(),
                        graph: Some(func.lazy_graph().clone()),
                        return_type: func.return_type.clone(),
                        hints: func.hints.clone(),
                    }],
                });
            }
            // Inherent method: `impl Type { fn m }`.
            (Some(owner), None) => {
                canonical_inherent_methods.push(parse::InherentMethodInfo {
                    for_type: owner.clone(),
                    self_ty_root: Some(owner.clone()),
                    name: func.name.clone(),
                    graph: func.lazy_graph().clone(),
                    return_type: func.return_type.clone(),
                    hints: func.hints.clone(),
                });
            }
            // Free function — registered by the dedicated loop below.
            (None, None) => {}
        }
    }
    prof.mark("  classify methods into side tables");
    // RPython: use the rtyped graphs (with concretetype info) for all analysis.
    // Use program.functions' graphs which were built with full struct_fields
    // context, NOT re-parsed graphs (which lose array_type_id etc.).
    // One graph object per free function, shared by every alias spelling
    // and by the hint registration below: upstream's aliases name the same
    // Python graph object.
    let mut free_function_graphs: Vec<Option<call::GraphSource>> =
        vec![None; program.functions.len()];
    for (index, func) in program.functions.iter().enumerate() {
        if func.self_ty_root.is_none() {
            // Stamp the source return type onto the graph so the JIT
            // codewriter signature validator reads `FUNC.RESULT`
            // directly off the callee graph (RPython
            // `funcptr._obj.TO.RESULT`).
            let graph = lazy_graph_source(func.lazy_graph(), &func.return_type, &func.hints);
            free_function_graphs[index] = Some(graph.clone());
            // Free function: register under every canonical alias
            // spelling computed by `free_function_alias_paths` — bare
            // segments, `crate::` prefix, three pyre-crate prefixes,
            // and the same set re-prefixed with `func.module_path`
            // (the file's crate-stripped module path).  The hint
            // registration loop below uses the same helper so call
            // sites that hit a module-qualified spelling find both the
            // graph and its elidable/loopinvariant/oopspec hints.
            for path in free_function_alias_paths(&func.name, &func.module_path) {
                register_function_graph_alias(
                    &mut canonical_function_graphs,
                    &mut canonical_function_alias_source,
                    path,
                    &func.name,
                    &graph,
                );
            }
        }
    }
    prof.mark("  free-fn graph aliases");

    // ── Build CallControl (RPython call.py) ──
    // Populate with all discovered function graphs and trait impl methods.
    let mut call_control = call::CallControl::new();
    call_control.use_funcobj_declarations(funcobj_declarations);
    // RPython: known struct types for get_type_flag(ARRAY.OF) → FLAG_STRUCT.
    call_control.set_known_struct_names(program.known_struct_names.clone());
    // RPython: struct field types for op.args[0].concretetype resolution.
    call_control.set_struct_fields(program.struct_fields.clone());
    call_control.set_error_carrier(OwnedErrorCarrierSpec::own(static_addrs.error_carrier));
    // Enum `discriminant → variant` tables for the `__discriminant`
    // getattr's discriminant→variant narrowing knowntypedata.
    call_control.set_enum_variant_by_discriminant(program.enum_variant_by_discriminant.clone());
    // RPython: symbolic.get_field_token / get_size — resolve struct layouts
    // through the LayoutProvider. If no provider is given, use the heuristic
    // (type-string-based approximation of #[repr(C)] layout).
    // `_immutable_fields_` is declared on the class (`rclass.py _parse_field_list`
    // reads it off the class object, which has exactly one identity), but
    // `struct_fields` carries several spellings of the same struct —
    // `PyType` and `pyre_object::pyobject::PyType` both resolve to one
    // `StructId`. The layout loop below stores a single layout per
    // `StructId`, so the spelling `HashMap` iteration happens to reach last
    // decides whether that layout carries `is_immutable`. Give every
    // spelling of a declared struct the declaration first.
    let immutable_fields =
        expand_immutable_fields_to_all_spellings(&program.immutable_fields, &program.struct_fields);
    let heuristic;
    let provider: &dyn layout::LayoutProvider = match layout_provider {
        Some(p) => p,
        None => {
            heuristic = layout::HeuristicLayoutProvider::from_struct_fields(
                &program.struct_fields.fields,
                &program.known_struct_names,
                &immutable_fields,
            );
            &heuristic
        }
    };
    // RPython: per-class `_immutable_fields_` declaration. Drives
    // `FieldDescr.is_pure` for the heuristic-fallback path inside
    // `all_interiorfielddescrs` (the layout-provider path already carries
    // `is_immutable` on `StructFieldLayout`).
    call_control.immutable_fields_by_struct = immutable_fields;
    // `descr.py ARRAY_INSIDE._immutable_field(None)` parity.
    // Summarise `field[*]` annotations into the array-type-keyed set so
    // `arraydescrof_concrete` can fold field-level immutability into the
    // shared per-ARRAY descr's `is_pure` flag.
    call_control.recompute_immutable_array_types();
    // Historical `unsafe_fn_stubs` carrier for every annotator-only residual
    // declaration: unsafe path aliases, `dont_look_inside` functions, and
    // marked allocation constructors.  Each adds the FunctionDesc-equivalent
    // key/signature the annotator needs, never a JitCode body — see
    // `CallControl::unsafe_fn_stubs`.  Sourced from Charon and populated on the
    // SemanticProgram in `build_semantic_program_via_active_frontend`.
    call_control.unsafe_fn_stubs = program.unsafe_fn_stubs.clone();
    call_control.atomic_load_decls = program.atomic_load_decls.clone();
    // `rpython.rtyper.extregistry.register_external` parity for the three
    // shadow-stack operations `eval::FrameAnchor` reaches.  `majit_gc` is not
    // one of the crates lowered by `scripts/extract-llbc.py` (see the
    // canmallocgc census below), so these safe free functions have neither a
    // graph nor an unsafe-declaration stub.  Leaving them undeclared poisons
    // `FrameAnchor::from_raw`'s CallRegistry entry, which in turn rejects
    // every frame getset gateway before the codewriter can remove its
    // `jit_force_virtualizable` marker.
    //
    // These entries use the existing annotator-only stub carrier: it supplies
    // the declared signature/result shell but never adds a codewriter graph.
    // The matching runtime addresses are published by
    // `pyre_interpreter::jit_trace_fnaddrs`, exactly like an RPython
    // `register_external` declaration is paired with its linker symbol.
    use crate::flowspace::argument::Signature;
    call_control.unsafe_fn_stubs.extend([
        (
            vec!["majit_gc".into(), "shadow_stack".into(), "push".into()],
            Signature::new(vec!["gcref".into()], None, None),
            // `shadow_stack::push` returns the native-word stack index
            // (`usize`).  Stub results use the front end's canonical semantic
            // tokens (`dont_look_inside_return_token` maps `Unsigned` to
            // `u64`), not literal Rust type spellings.  RPython's
            // shadow-stack top/depth is likewise a non-negative machine word;
            // using the signed `i64` token makes `FrameAnchor.depth` merge
            // `SomeInteger` with `r_uint` at its readers.
            Some("u64".into()),
        ),
        (
            vec!["majit_gc".into(), "shadow_stack".into(), "get".into()],
            Signature::new(vec!["depth".into()], None, None),
            Some("gcref".into()),
        ),
        (
            vec![
                "majit_gc".into(),
                "shadow_stack".into(),
                "try_pop_to".into(),
            ],
            Signature::new(vec!["depth".into()], None, None),
            None,
        ),
        // Fixed frame slots for a local live across several collecting calls.
        // `gc_save_root` / `gc_restore_root` write one index; `push_roots` /
        // `pop_roots` only bump the LIFO shadow stack around a single call.
        // `majit_gc` is not lowered, so the four operations arrive as calls
        // with no graph — the same `register_external` carrier as `push`.
        (
            vec![
                "majit_gc".into(),
                "shadow_stack".into(),
                "acquire_owner_root".into(),
            ],
            Signature::new(vec!["root".into()], None, None),
            Some("u64".into()),
        ),
        (
            vec![
                "majit_gc".into(),
                "shadow_stack".into(),
                "release_owner_root".into(),
            ],
            Signature::new(vec!["index".into()], None, None),
            None,
        ),
        (
            vec![
                "majit_gc".into(),
                "shadow_stack".into(),
                "get_owner_root".into(),
            ],
            Signature::new(vec!["index".into()], None, None),
            Some("gcref".into()),
        ),
        (
            vec![
                "majit_gc".into(),
                "shadow_stack".into(),
                "set_owner_root".into(),
            ],
            Signature::new(vec!["index".into(), "root".into()], None, None),
            None,
        ),
        // `framework.py` keeps the GC as a prebuilt constant, so a nursery
        // request always has a route. This query is the port's stand-in and
        // lives in unlowered `majit_gc`; declare the bool result here.
        (
            vec!["majit_gc".into(), "gc_allocator_installed".into()],
            Signature::new(vec![], None, None),
            Some("bool".into()),
        ),
    ]);
    // Parallel carrier for foreign opaque-ADT method externals
    // (`<BigInt as Add>::add`, …) — `populate_call_registry_from_call_graphs`
    // registers each as an opaque external so the residual `FunctionPath`
    // form resolves; see `cutover::register_foreign_opaque_method_externals`.
    call_control.foreign_opaque_method_externals = program.foreign_opaque_method_externals.clone();
    // Diagnostic-only spelling census. Do not sort the production loop or
    // alter its last-writer-wins behaviour: this pass measures whether two
    // registry spellings that resolve to one StructId produce different
    // layouts before choosing the canonical producer. The side table groups
    // equal layouts instead of comparing every spelling pair, keeping the
    // census linear in the number of spellings times the (small) number of
    // distinct layouts.
    let census_struct_layouts = struct_layout_census_enabled();
    let mut struct_layout_variants: std::collections::HashMap<
        majit_ir::descr::StructId,
        Vec<(StructLayout, Vec<String>)>,
    > = std::collections::HashMap::new();
    // Populate CallControl with layouts from the provider.  Where Charon
    // resolved an exact layout (`program.exact_layouts`), use the true Rust
    // offsets and total size — `#[repr(Rust)]` reorders/repacks fields, so
    // the heuristic's `#[repr(C)]` approximation can disagree with the real
    // allocation. For enum-variant keys the exact offsets are tag-inclusive,
    // so the variant payload resolves at its real position. Per-field type
    // classification, immutability rank, and size stay as the provider
    // computed them; only the byte offsets and the struct size are corrected.
    prof.mark("  call_control setup");
    for struct_name in program.struct_fields.fields.keys() {
        // Resolve the (possibly multi-spelled) field-registry key to its
        // identity token; an unresolved / ambiguous bare leaf has no
        // identity to key a layout on, so skip it.
        let Some(sid) = majit_ir::descr::struct_id_for_name(struct_name) else {
            continue;
        };
        // `exact_layouts` originates on Charon's defining TypeDecl. Concrete
        // generic identities are derived later from use-site type arguments,
        // so inherit the declaration's exact offsets when it has no concrete
        // entry (notably the explicit translated `Result` shell).
        let exact = program.exact_layouts.get(&sid).or_else(|| {
            majit_ir::descr::struct_template_id_for_name(struct_name)
                .and_then(|template| program.exact_layouts.get(&template))
        });
        // An opaque/shadow declaration can share the runtime identity of its
        // defining type (e.g. pyre-interpreter's zero-field GetSetProperty
        // marker versus pyre-object's real definition).  If Charon's exact
        // definition carries fields but this spelling has no rows, it is an
        // alias consumer, not a layout producer; letting it write would erase
        // the real field layout according to HashMap iteration order.
        let incomplete_shadow = exact.is_some_and(|layout| !layout.field_offsets.is_empty())
            && program
                .struct_fields
                .fields
                .get(struct_name)
                .is_some_and(Vec::is_empty);
        if incomplete_shadow {
            continue;
        }
        let layout = match exact {
            Some(exact) => {
                provider.get_struct_layout_exact(struct_name, &exact.field_offsets, exact.size)
            }
            None => provider.get_struct_layout(struct_name),
        };
        if let Some(mut layout) = layout {
            if let Some(host) = exact.and_then(|item| item.host.clone()) {
                layout.host = Some(host);
            }
            layout.gckind = if declared_gc.contains(&sid) {
                crate::translator::rtyper::lltypesystem::lltype::GcKind::Gc
            } else {
                crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw
            };
            if census_struct_layouts {
                let variants = struct_layout_variants.entry(sid).or_default();
                if let Some((_, spellings)) = variants.iter_mut().find(|(observed, _)| {
                    observed.size == layout.size && observed.fields == layout.fields
                }) {
                    spellings.push(struct_name.clone());
                } else {
                    variants.push((layout.clone(), vec![struct_name.clone()]));
                }
            }
            call_control.set_struct_layout(sid, layout);
        }
    }
    if census_struct_layouts {
        let struct_id_count = struct_layout_variants.len();
        let aliased_struct_ids = struct_layout_variants
            .values()
            .filter(|variants| variants.iter().map(|(_, names)| names.len()).sum::<usize>() > 1)
            .count();
        let mut conflicts: Vec<_> = struct_layout_variants
            .iter_mut()
            .filter(|(_, variants)| variants.len() > 1)
            .collect();
        conflicts.sort_by_key(|(sid, _)| **sid);
        for (sid, variants) in &mut conflicts {
            for (_, spellings) in variants.iter_mut() {
                spellings.sort();
            }
            variants.sort_by(|(_, left), (_, right)| left[0].cmp(&right[0]));
            let spelling_count = variants
                .iter()
                .map(|(_, spellings)| spellings.len())
                .sum::<usize>();
            eprintln!(
                "MAJIT_STRUCT_LAYOUT_CENSUS conflict sid={sid:?} \
                 spellings={spelling_count} layout_variants={}",
                variants.len(),
            );
            for (index, (layout, spellings)) in variants.iter().enumerate() {
                eprintln!(
                    "MAJIT_STRUCT_LAYOUT_CENSUS variant sid={sid:?} index={index} \
                     spelling_count={} spellings={spellings:?} layout={layout:?}",
                    spellings.len(),
                );
            }
        }
        eprintln!(
            "MAJIT_STRUCT_LAYOUT_CENSUS summary struct_ids={} \
             aliased_struct_ids={aliased_struct_ids} conflicting_struct_ids={}",
            struct_id_count,
            conflicts.len(),
        );
    }
    // Register graphs collected above (free functions only — trait
    // methods are handled separately via register_trait_method).
    prof.mark("  struct layouts");
    for (path, graph) in &canonical_function_graphs {
        // Each alias re-presents the shared graph so
        // `insert_function_graph_indexed` can fold that spelling's pending
        // external-funcobj effects onto the one stored graph; `GraphStore`
        // keys on the funcobj, so the copy is transient.
        call_control.register_function_graph(path.clone(), graph.clone());
    }
    prof.mark("  register_function_graph (free fns)");
    // RPython `CallControl.graphs_from` obtains the graph from the concrete
    // function pointer, distinguishing same-named methods from distinct trait
    // impls. The string-keyed Rust registry needs that identity explicitly:
    // `impl From<bool> for Dynamic` and `impl From<FnPtr> for Dynamic` both
    // otherwise collapse onto `[types, dynamic, Dynamic, from]`. Charon's
    // trait-impl id is already carried on the SemanticFunction; register the
    // exact path that `front::mir::blanket_into_devirt` emits.
    for func in &program.functions {
        let (Some(owner), Some(impl_id)) = (&func.self_ty_root, func.trait_impl_id) else {
            continue;
        };
        let path = crate::parse::CallPath::for_trait_impl_method(owner, impl_id, &func.name);
        let graph = lazy_graph_source(func.lazy_graph(), &func.return_type, &[]);
        if func.hints.is_empty() {
            call_control.register_function_graph(path, graph);
        } else {
            call_control.register_function_graph_with_hints(path, graph, func.hints.clone());
        }
    }
    prof.mark("  register concrete trait-impl identities");
    // `rpbc.py FunctionReprBase.call`: the `c_graphs` constant is the full
    // `row_of_graphs.values()` PBC family, including concrete overrides of a
    // required trait method.  Concrete impl graphs stay on the
    // receiver-driven inherent-method registration path below (so
    // name-based method lookup is unchanged), but their graph identities
    // must also populate the independent indirect-call family row.
    for (trait_leaf, _, method_name, owner, _, _, _) in &concrete_trait_methods {
        call_control.register_trait_family_member(method_name, trait_leaf, owner);
    }
    prof.mark("  register concrete indirect-call families");
    // Re-register free functions with their RPython-equivalent hints
    // (`elidable`, `loop_invariant`, `unroll_safe`, `jit_look_inside`)
    // so `JitPolicy::look_inside_graph` sees the same metadata RPython
    // reads off `func._jit_*_` / `_elidable_function_`.
    //
    // RPython parity: hints live on `graph.func` and survive alias
    // routing because the call path resolves to a single function
    // object identity (`policy.py:48` / `call.py:126`). This port keys hints
    // on `CallPath` segments, so every spelling the graph alias loop
    // registered above (`free_function_alias_paths`: bare + `crate::` +
    // each loaded LLBC crate root + the
    // `source_module`-prefixed variants thereof) must carry the same
    // hint set.  Missing the source-module-qualified aliases silently
    // disables `_jit_look_inside_` etc. for module-qualified callers,
    // which `CallControl::find_all_graphs` looks up by callee path.
    for (func, graph) in program.functions.iter().zip(&free_function_graphs) {
        if func.self_ty_root.is_some() || func.hints.is_empty() {
            continue;
        }
        let Some(graph) = graph else {
            continue;
        };
        for path in free_function_alias_paths(&func.name, &func.module_path) {
            call_control.register_function_graph_with_hints(
                path,
                graph.clone(),
                func.hints.clone(),
            );
        }
    }
    pin_builtin_graphs(
        &canonical_function_graphs,
        &config.pipeline.builtin_graphs,
        &mut call_control,
    );
    // The registration loop below prefers the graph from
    // `mir_graph_lookup` over the one already carried in
    // `canonical_trait_impls` / `canonical_inherent_methods`.  Both
    // come from `program.functions`, so the lookup is a defensive
    // re-fetch keyed on the registration path — cheap, and it keeps a
    // single source of truth for the trait-default vs concrete-impl
    // distinction.
    // `[<Trait>, <method>]` is the CallKind::Trait direct path.  Bind it
    // before hint stamping so the override graph, not the raising default,
    // is what those hints attach to.
    let leaf_identities = distinct_struct_identities_by_leaf(&program);
    bind_trait_default_direct_paths(&program, &mut call_control, &leaf_identities);
    for impl_info in &canonical_trait_impls {
        let impl_type = impl_info
            .self_ty_root
            .as_deref()
            .unwrap_or(&impl_info.for_type);
        // RPython parity: `trait_root=Some(trait_name)` for real trait impls,
        // `None` for inherent impls (impl SomeType { ... } without `for Trait`).
        // `parse.rs` always writes `trait_name`; a sentinel empty string
        // from the inherent branch (see parse.rs) needs special-casing.
        let trait_root = if impl_info.trait_name.is_empty() {
            None
        } else {
            Some(impl_info.trait_name.as_str())
        };
        let is_default = impl_info.for_type.starts_with("<default methods of ");
        for method in &impl_info.methods {
            // `classdesc.py lookup` MRO: on the generic-dispatch
            // direct path `[<Trait>, <method>]`, the unique concrete
            // override shadows the trait default body.  `None` for
            // concrete-impl entries, defaults without an override, and
            // traits with several concrete impl types (the receiver
            // stays ambiguous — indirect-call family territory keeps
            // the default).
            let devirt: Option<(&str, &front::semantic::SemanticFunction)> =
                unique_trait_default_override(
                    &program,
                    &leaf_identities,
                    is_default,
                    impl_info.trait_name.as_str(),
                    method.name.as_str(),
                    &trait_method_overrides,
                    &trait_concrete_impl_types,
                );
            // Hints for the direct path follow the graph registered
            // there (RPython binds hints to graph identity).
            let direct_hints: &Vec<String> = match devirt {
                Some((_, override_info)) => &override_info.hints,
                None => &method.hints,
            };
            // Read the MIR-built graph from `program.functions`.
            // `method.graph` (`Option`) remains as a residual fallback
            // for the handful of MIR-uncovered entries, though every
            // method registered above carries a graph so the fallback
            // is effectively unreached.
            let mir_graph: Option<&model::LazyGraph> = if is_default {
                mir_graph_lookup.lookup_trait_default(&impl_info.trait_name, &method.name)
            } else {
                mir_graph_lookup.lookup_impl_method(impl_type, &method.name)
            };
            let graph_source: Option<&model::LazyGraph> = mir_graph.or(method.graph.as_ref());
            if let Some(graph) = graph_source {
                // Stamp the source return type onto the graph itself so
                // the JIT codewriter signature validator reads
                // `FUNC.RESULT` directly off the callee graph
                // (RPython `funcptr._obj.TO.RESULT`).
                let graph = lazy_graph_source(graph, &method.return_type, &[]);
                call_control.register_trait_method(&method.name, trait_root, impl_type, graph);
                // The pseudo-type path `[<default methods of Trait>, <method>]`
                // is the base-class body.  `[<Trait>, <method>]` is bound by
                // `bind_trait_default_direct_paths` (`classdesc.py` `lookup`,
                // the CallKind::Trait path).
            }
            let path = crate::parse::CallPath::for_impl_method(impl_type, method.name.as_str());
            // Mirror RPython `func._elidable_function_` / `func._jit_*_`:
            // `register_trait_method` registers the graph without hints, so
            // the BFS would see hint-less SemanticFunctions for trait methods
            // and inline elidable methods that should remain residual.
            // `register_function_hints_for` stamps the hints onto the
            // already-registered graph (`graph.hints`) keyed on the same
            // `[impl_type, method_name]` path the BFS reads.
            if !method.hints.is_empty() {
                call_control.register_function_hints_for(path.clone(), method.hints.clone());
            }
            // Default-method bodies also register under `[trait_name,
            // method_name]` (see the `register_function_graph(direct_path,
            // direct_graph)` branch above); mirror the hint registration
            // so the BFS reaches the same `_reject_function("elidable")`
            // verdict regardless of which path it walks.  `direct_hints`
            // follows the graph registered there — the override's when
            // the direct path was devirtualized.
            if is_default && !direct_hints.is_empty() {
                let direct_path = crate::parse::CallPath::from_segments([
                    impl_info.trait_name.as_str(),
                    method.name.as_str(),
                ]);
                call_control.register_function_hints_for(direct_path, direct_hints.clone());
            }
            // RPython: hints bound to graph identity.
            for hint in &method.hints {
                if call_control.mark_aroundstate_hint(path.clone(), hint) {
                    continue;
                }
                match hint.as_str() {
                    "elidable" => call_control.mark_elidable(path.clone()),
                    "elidable_cannot_raise" => {
                        call_control.mark_cannot_raise_assertion(path.clone())
                    }
                    "cannot_raise" => call_control.mark_cannot_raise_assertion(path.clone()),
                    "elidable_or_memerror" => {
                        call_control.mark_memerror_only_assertion(path.clone())
                    }
                    "loopinvariant" => call_control.mark_loopinvariant(path.clone()),
                    "close_stack" => call_control.mark_close_stack(path.clone()),
                    "cannot_collect" => call_control.mark_cannot_collect(path.clone()),
                    "gc_effects" => call_control.mark_external_gc_effects(path.clone()),
                    _ => {}
                }
            }
            if is_default {
                let dp = crate::parse::CallPath::from_segments([
                    impl_info.trait_name.as_str(),
                    method.name.as_str(),
                ]);
                for hint in direct_hints {
                    match hint.as_str() {
                        "elidable" => call_control.mark_elidable(dp.clone()),
                        "elidable_cannot_raise" => {
                            call_control.mark_cannot_raise_assertion(dp.clone())
                        }
                        "cannot_raise" => call_control.mark_cannot_raise_assertion(dp.clone()),
                        "elidable_or_memerror" => {
                            call_control.mark_memerror_only_assertion(dp.clone())
                        }
                        "loopinvariant" => call_control.mark_loopinvariant(dp.clone()),
                        _ => {}
                    }
                }
            }
        }
    }
    prof.mark("  trait-impl registration");
    // Single-impl devirtualization for REQUIRED trait methods.  A call
    // site `<Trait>::<method>(receiver, …)` lowers to
    // `CallTarget::FunctionPath { segments: [<Trait>, <method>], fun_decl_id: None }`
    // (`front/mir.rs` `call_target_segments` `CallKind::Trait` arm).
    // Default bodies registered that direct path in the loop above; a
    // required method (declaration only, no default body) left it
    // unregistered, so the registry lift failed with "not registered
    // in CallRegistry" and poisoned every caller.  RPython
    // resolves the call on the receiver's class
    // (`classdesc.py lookup` MRO walk); when exactly one class
    // implements the method in the closed LLBC world, that walk has a
    // single possible answer — bind the direct path to it.  Two or
    // more impls (or a default body, already covered) stay off this
    // path so ambiguous dispatch keeps failing loud until
    // receiver-driven resolution lands.
    let mut default_trait_methods: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    for impl_info in &canonical_trait_impls {
        if impl_info.trait_name.is_empty()
            || !impl_info.for_type.starts_with("<default methods of ")
        {
            continue;
        }
        for method in &impl_info.methods {
            default_trait_methods.insert((impl_info.trait_name.clone(), method.name.clone()));
        }
    }
    let mut concrete_impl_counts: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();
    for (trait_leaf, _, method_name, _, _, _, _) in &concrete_trait_methods {
        *concrete_impl_counts
            .entry((trait_leaf.clone(), method_name.clone()))
            .or_insert(0) += 1;
    }
    // Trait → owner of its only concrete impl, for the dual-gate
    // bookkeeper's generic-receiver classdef seeding
    // (`derive_subject_inputcells` resolves a bound-trait `class_root`
    // through this map).  Unlike the direct-path registration below,
    // this is annotation-only metadata — it pulls no graph bodies into
    // callers.
    let mut trait_impl_owners: TraitImplOwners = std::collections::HashMap::new();
    for (_, trait_qualified, _, owner, _, _, _) in &concrete_trait_methods {
        // Keyed by the trait's qualified `name_path()` — two distinct
        // traits sharing a leaf name must not pool their impl owners
        // (`tyref_generic_trait_bound_root` resolves bound-trait
        // receivers through this map by the same qualified path).
        trait_impl_owners
            .entry(trait_qualified.clone())
            .or_default()
            .insert(owner.clone());
    }
    // Opt-in receiver-driven dispatch families (receiver-dispatch configuration): a consumer
    // names `>=2`-impl trait qualified paths
    // whose `dyn Trait` receivers should annotate to a base ClassDef
    // linking the impl subclasses, so a method getattr on the receiver
    // resolves the impl MethodDesc family.  Built from `trait_impl_owners`
    // (before it is consumed below) and forwarded through `CallControl`
    // to `dual_gate_registry`, which mints each family before the
    // class-method seeding.
    //
    // Per-family builder shared by the config and auto-population paths.
    // Concrete owners retain their qualified type identity: two unrelated
    // `Port` structs must become two distinct subclasses, just as RPython
    // keys classes by object identity rather than by the final name segment.
    let make_registration = |trait_qualified: &str,
                             owners: &std::collections::BTreeSet<String>|
     -> Option<call::TraitFamilyRegistration> {
        let impl_roots = owners.iter().cloned().collect();
        Some(call::TraitFamilyRegistration {
            base_root: trait_qualified.to_string(),
            impl_roots,
        })
    };
    // Config-named families first (empty in pyre production).  Preserves
    // the prior spelling-mismatch diagnostic on the `None` arm.
    let mut trait_family_registrations: Vec<call::TraitFamilyRegistration> = config
        .pipeline
        .register_trait_families
        .iter()
        .filter_map(
            |trait_qualified| match trait_impl_owners.get(trait_qualified) {
                Some(owners) => make_registration(trait_qualified, owners),
                None => {
                    // Config validation: a named family that matches no
                    // harvested trait is almost always a spelling mismatch
                    // (the key is the trait's full `name_path()`).  List the
                    // available multi-impl trait paths so the consumer can
                    // correct it.
                    let candidates: Vec<&String> = trait_impl_owners
                        .iter()
                        .filter(|(_, owners)| owners.len() >= 2)
                        .map(|(path, _)| path)
                        .collect();
                    eprintln!(
                        "register_trait_families: no harvested trait matches {trait_qualified:?}; \
                 known multi-impl trait paths: {candidates:?}"
                    );
                    None
                }
            },
        )
        .collect();
    // Auto-population (receiver-dispatch configuration): register EVERY `>=2`-impl trait so its
    // inline `dyn Trait` receiver (lowered to `CallTarget::Indirect`) narrows
    // to the family base ClassDef.  Union with the config list (dedup by
    // base_root), never replacing it, so an existing configuration stays
    // valid. Minting base/impl subclass classdefs and lowering the indirect
    // call are one decision. Implementations retain their qualified owner
    // paths, so same-leaf types from different modules remain distinct
    // subclasses of their respective trait families.
    {
        let already: std::collections::HashSet<&str> = trait_family_registrations
            .iter()
            .map(|r| r.base_root.as_str())
            .collect();
        let mut auto: Vec<call::TraitFamilyRegistration> = trait_impl_owners
            .iter()
            .filter(|(trait_qualified, owners)| {
                owners.len() >= 2 && !already.contains(trait_qualified.as_str())
            })
            .filter_map(|(trait_qualified, owners)| make_registration(trait_qualified, owners))
            .collect();
        // Deterministic mint order (BTreeMap iteration on `trait_impl_owners`
        // is already sorted, but sort defensively since a HashMap seeded it).
        auto.sort_by(|a, b| a.base_root.cmp(&b.base_root));
        trait_family_registrations.extend(auto);
    }
    call_control.set_trait_family_registrations(trait_family_registrations);
    let trait_unique_impls = unique_trait_impl_roots(&program, trait_impl_owners);
    call_control.set_trait_unique_impls(trait_unique_impls);
    for (trait_leaf, _, method_name, _owner, return_type, hints, graph) in &concrete_trait_methods {
        let key = (trait_leaf.clone(), method_name.clone());
        if concrete_impl_counts.get(&key) != Some(&1) || default_trait_methods.contains(&key) {
            continue;
        }
        // Use this trait-impl method's own graph rather than
        // `lookup_impl_method(owner, method_name)`: when the impl owner
        // also has a same-named *inherent* method (e.g. `PyFrame` has both
        // the inherent `peek_at` and `<PyFrame as SharedOpcodeHandler>::
        // peek_at`), the two collide on the `(owner, name)` key and the
        // lookup resolves to `Err(())` (ambiguous), silently dropping the
        // single-impl devirtualization.  The carried graph is unambiguous.
        let graph = lazy_graph_source(graph, return_type, &[]);
        let direct_path =
            crate::parse::CallPath::from_segments([trait_leaf.as_str(), method_name.as_str()]);
        call_control.register_function_graph(direct_path.clone(), graph);
        // Mirror the default-direct hint registration above so the
        // BFS reaches the same `_reject_function` verdict on this
        // path as on `[impl_type, method]`.
        if !hints.is_empty() {
            call_control.register_function_hints_for(direct_path.clone(), hints.clone());
        }
        for hint in hints {
            match hint.as_str() {
                "elidable" => call_control.mark_elidable(direct_path.clone()),
                "elidable_cannot_raise" => {
                    call_control.mark_cannot_raise_assertion(direct_path.clone())
                }
                "cannot_raise" => call_control.mark_cannot_raise_assertion(direct_path.clone()),
                "elidable_or_memerror" => {
                    call_control.mark_memerror_only_assertion(direct_path.clone())
                }
                "loopinvariant" => call_control.mark_loopinvariant(direct_path.clone()),
                _ => {}
            }
        }
    }
    // RPython: direct_call → funcobj.graph — inherent methods are resolved
    // by direct graph linkage, not name-based trait method lookup.
    // Register only in function_graphs under qualified path (Type::method).
    for method_info in &canonical_inherent_methods {
        let impl_type = method_info
            .self_ty_root
            .as_deref()
            .unwrap_or(&method_info.for_type);
        let path = crate::parse::CallPath::for_impl_method(impl_type, method_info.name.as_str());
        // Read the MIR-built graph; `method_info.graph` is the
        // residual fallback for the handful of MIR-uncovered entries,
        // effectively unreached because every inherent method
        // registered above carries a graph.
        let graph: &model::LazyGraph = mir_graph_lookup
            .lookup_impl_method(impl_type, &method_info.name)
            .unwrap_or(&method_info.graph);
        // Pair the graph with the method's hints so the BFS-driven
        // `look_inside_graph` synthesises a `SemanticFunction` whose
        // `_reject_function("elidable")` mirrors RPython's
        // `getattr(func, "_elidable_function_", False)`.  Without this
        // the BFS sees impl methods as hint-less and inlines elidable
        // methods (e.g. `PyFrame::nlocals`) that should remain residual.
        // Stamp the source return type onto the graph so the JIT
        // codewriter signature validator reads `FUNC.RESULT` directly off
        // the callee graph (`funcptr._obj.TO.RESULT`), matching the
        // free-function and trait-method registration paths above.
        let graph = lazy_graph_source(graph, &method_info.return_type, &[]);
        if method_info.hints.is_empty() {
            call_control.register_function_graph(path.clone(), graph);
        } else {
            call_control.register_function_graph_with_hints(
                path.clone(),
                graph,
                method_info.hints.clone(),
            );
        }
        // RPython: hints bound to graph identity.
        for hint in &method_info.hints {
            if call_control.mark_aroundstate_hint(path.clone(), hint) {
                continue;
            }
            match hint.as_str() {
                "elidable" => call_control.mark_elidable(path.clone()),
                "elidable_cannot_raise" => call_control.mark_cannot_raise_assertion(path.clone()),
                "cannot_raise" => call_control.mark_cannot_raise_assertion(path.clone()),
                "elidable_or_memerror" => call_control.mark_memerror_only_assertion(path.clone()),
                "loopinvariant" => call_control.mark_loopinvariant(path.clone()),
                "close_stack" => call_control.mark_close_stack(path.clone()),
                "cannot_collect" => call_control.mark_cannot_collect(path.clone()),
                "gc_effects" => call_control.mark_external_gc_effects(path.clone()),
                _ => {}
            }
        }
    }
    prof.mark("  concrete-trait + inherent method registration");
    for &(full_path, fnaddr) in fnaddr_bindings {
        call_control.register_macro_helper_trace_fnaddr(full_path, fnaddr);
    }
    for &(module_path_with_crate, impl_type_as_written, method, fnaddr) in impl_fnaddr_bindings {
        call_control.register_macro_impl_helper_trace_fnaddr(
            module_path_with_crate,
            impl_type_as_written,
            method,
            fnaddr,
        );
    }
    // RPython: GC transformer sets _gctransformer_hint_close_stack_,
    // _gctransformer_hint_cannot_collect_ on functions, and
    // random_effects_on_gcobjs on external function objects.
    // In majit, detect these from #[jit_close_stack], #[jit_cannot_collect],
    // #[jit_gc_effects] attributes on functions in the parsed source.
    // RPython: hints are bound to the function/graph object, not the name.
    // Register under ALL canonical paths so any call-site lookup finds them.
    for func in &program.functions {
        if func.hints.is_empty() {
            continue;
        }
        // Build all canonical paths for this function.  Free functions
        // must use the same alias spelling set as the graph-alias loop
        // above so a module-qualified call site finds both the graph
        // AND the hints (elidable/loopinvariant/cannot_collect/oopspec).
        let paths = if let Some(ref owner) = func.self_ty_root {
            // impl method: ["owner", "method"]
            let mut paths = vec![crate::parse::CallPath::from_segments([
                owner.as_str(),
                func.name.as_str(),
            ])];
            if let Some(impl_id) = func.trait_impl_id {
                paths.push(crate::parse::CallPath::for_trait_impl_method(
                    owner, impl_id, &func.name,
                ));
            }
            paths
        } else {
            // Same alias-spelling set as the graph-registration loop
            // above so every spelling the call-site might canonicalise
            // to also finds the `#[oopspec]` / `#[loop_invariant]` /
            // `#[cannot_collect]` / elidable hint.
            free_function_alias_paths(&func.name, &func.module_path)
        };
        for path in &paths {
            call_control.mark_decorator_hints(path, &func.hints);
        }
    }
    // A helper may never appear in `program.functions` (no body, or the
    // body did not lower) and still be the target of a residual call.
    // `#[oopspec]` / `cannot_raise` live on the funcobj either way:
    // `guess_call_kind` reads `func.oopspec` (`call.py`), and a missing
    // `cannot_raise` makes `getcalldescr` record GUARD_NO_EXCEPTION.
    for (path_str, hints) in &program.harvested_hints {
        let cannot_raise = hints
            .iter()
            .any(|h| h == "cannot_raise" || h == "elidable_cannot_raise");
        let oopspecs: Vec<&str> = hints
            .iter()
            .filter_map(|h| h.strip_prefix("oopspec:"))
            .collect();
        if !cannot_raise && oopspecs.is_empty() {
            continue;
        }
        let segs: Vec<&str> = path_str.split("::").collect();
        let Some((name, module_segs)) = segs.split_last() else {
            continue;
        };
        // `Type::method` is not a free function. Aliasing it as bare
        // `method` would share the GraphStore row with an unrelated
        // `module::method` (e.g. `RootStackGuard::get` vs
        // `baseobjspace::get`).
        let paths = if module_segs
            .last()
            .is_some_and(|s| s.starts_with(|c: char| c.is_uppercase()))
        {
            vec![crate::parse::CallPath::from_segments(segs.iter().copied())]
        } else {
            let module = module_segs.join("::");
            free_function_alias_paths(name, &module)
        };
        for p in &paths {
            for spec in &oopspecs {
                call_control.mark_oopspec(p.clone(), (*spec).to_string());
            }
        }
        if cannot_raise {
            for p in paths {
                call_control.mark_cannot_raise_assertion(p);
            }
        }
    }

    // RPython: setup_jitdriver(jitdriver_sd) — register every explicit
    // portal and its green/red layout. Portal binding and graph discovery
    // are independent of any interpreter dispatch representation.
    // Helper-only analysis (`helper_roots` non-empty) has no portal.
    if helper_roots.is_empty() {
        register_configured_jitdrivers(
            &mut call_control,
            &config.pipeline.jit_drivers,
            &config.pipeline.transform.jitdriver_receiver_roots,
        );
    }
    // call.py seeds inline_calls_to helper graphs so residual opcode
    // callees join the BFS; a helper-only pipeline still accepts them.
    for path in &config.pipeline.helper_graphs {
        call_control.register_helper_graph(path.clone());
    }
    // warmspot.py WarmRunnerDesc.make_virtualizable_infos —
    // assigns each registered driver's virtualizable metadata only after the
    // complete driver set exists.
    call_control
        .make_virtualizable_infos(|jd_idx, vtypeptr_token| vinfo_factory(jd_idx, vtypeptr_token));
    // Register oopspecs for jit.* builtin functions.
    // rlib/jit.py: these functions carry @oopspec("jit.*") decorators;
    // the codewriter converts calls to them into dedicated opcodes.
    for (func_name, spec) in &[
        // rlib/jit.py — @oopspec("jit.*") decorated functions
        ("isconstant", "jit.isconstant"),
        ("isvirtual", "jit.isvirtual"),
        ("current_trace_length", "jit.current_trace_length"),
        ("jit_debug", "jit.debug"),
        ("assert_green", "jit.assert_green"),
        // NOTE: conditional_call!/conditional_call_elidable!/record_known_result!
        // are handled by jitcode_lower (proc-macro level), not here.
        // They are macro_rules! that the codewriter AST parser does not expand.
    ] {
        // Register under common call-site path patterns.
        for path in [
            parse::CallPath::from_segments(["jit", func_name]),
            parse::CallPath::from_segments(["crate", "jit", func_name]),
            parse::CallPath::from_segments(["majit_metainterp", "jit", func_name]),
            parse::CallPath::from_segments(["majit_rlib", "jit", func_name]),
        ] {
            call_control.mark_oopspec(path, spec.to_string());
        }
    }
    // rlist.py — `@jit.oopspec("newlist_clear(count)")` on
    // `_ll_alloc_and_clear`.  The helper graph is minted by the rtyper
    // (`rtype_alloc_and_set`), so there is no source-level `#[oopspec]`
    // attribute for the walker above to harvest.  Attach the spec here,
    // keyed on the minted graph's NAME — the codewriter derives the
    // CallPath from the funcptr's graph name (`target_to_path`), so a
    // `direct_call` to the clear helper lowers to `new_array_clear`
    // instead of residualizing.  `alloc_and_set_sub_names` mints a
    // per-layout clear helper (`_ll_alloc_and_clear` resized,
    // `_ll_fixed_alloc_and_clear` fixed) so the `(name, args, ret)` helper
    // cache stays disjoint; both names must carry the spec.
    for clear_name in ["_ll_alloc_and_clear", "_ll_fixed_alloc_and_clear"] {
        call_control.mark_oopspec(
            parse::CallPath::from_segments([clear_name]),
            "newlist_clear(count)".to_string(),
        );
    }

    // `LLHelpers.ll_str2unicode` owns
    // `@jit.oopspec("str.str2unicode(str)")` in `rstr.py`.  Pyre's rtyper
    // synthesizes that helper graph (`build_ll_str2unicode_helper_graph`), so
    // it has no source-level Rust attribute for the hint harvester above to
    // see. Attach the upstream function-owned oopspec to the synthesized
    // graph name, just as the neighboring synthesized list and builder
    // helpers receive theirs.
    call_control.mark_oopspec(
        parse::CallPath::from_segments(["ll_str2unicode"]),
        "str.str2unicode(str)".to_string(),
    );

    // `ll_math.sqrt_nonneg` assigns
    // `sqrt_nonneg.oopspec = "math.sqrt_nonneg(x)"` after the function
    // definition. The interpreter owner is `sqrt_nonneg` in
    // `descroperation.rs`, which also carries `#[oopspec]` / `#[elidable]`
    // so the harvester registers every alias path. This bare-name mark is
    // the owner-side spelling `target_to_path` resolves for that callee;
    // `_handle_math_sqrt_call` then asks for `EF_ELIDABLE_CANNOT_RAISE`.
    call_control.mark_oopspec(
        parse::CallPath::from_segments(["sqrt_nonneg"]),
        "math.sqrt_nonneg(x)".to_string(),
    );

    // rbuilder.py `ll_shrink_final` calls `rgc.ll_shrink_array(buf, size)`,
    // whose `@jit.oopspec("rgc.ll_shrink_array")` is minted by the rtyper
    // (`rtype_builder_build`'s nested `ll_shrink_array` helper) with a stub
    // body, so there is no source-level `#[oopspec]` for the walker to
    // harvest.  Attach the spec here, keyed on the minted graph's NAME — the
    // codewriter's `_handle_rgc_call` then lowers the `direct_call` to a
    // `ShrinkArray` residual with a trailing `-live-` instead of tracing the
    // stub body.
    call_control.mark_oopspec(
        parse::CallPath::from_segments(["ll_shrink_array"]),
        "rgc.ll_shrink_array".to_string(),
    );

    // rgc.py `@jit.oopspec('list.ll_arraymove(array, source_start,
    // dest_start, length)')`. The rtyper mints this generic helper graph,
    // so its decorator cannot be harvested from a Rust source item above.
    call_control.mark_oopspec(
        parse::CallPath::from_segments(["ll_arraymove"]),
        "list.ll_arraymove(array, source_start, dest_start, length)".to_string(),
    );
    // rgc.py `@jit.oopspec('list.ll_arraycopy(source, dest, source_start,
    // dest_start, length)')`.  The rtyper and the rich-spine getslice
    // helpers both mint a `direct_call` of this name, so the decorator
    // cannot be harvested from a Rust source item above.
    call_control.mark_oopspec(
        parse::CallPath::from_segments(["ll_arraycopy"]),
        "list.ll_arraycopy(source, dest, source_start, dest_start, length)".to_string(),
    );

    // `collectanalyze.py analyze_simple_operation` answers True for the
    // allocation operations — `malloc` / `malloc_varsize` with `flavor='gc'`,
    // and any LLOp declared `canmallocgc=True`.  Upstream sees them because
    // they are operations in a graph it lowered.  `majit_gc` is not among the
    // crates `scripts/extract-llbc.py` lowers, so the same allocations arrive
    // here as calls to a callee with no graph, and `analyze_can_collect`'s
    // no-graph arm has nothing to read.  Nothing in a non-lowered crate's
    // source can carry a declaration across, so it is written here — the same
    // place upstream writes the one declaration it cannot derive either
    // (`random_effects_on_gcobjs=True` at the `llexternal` callsite,
    // `_rffi_stacklet.py`).
    //
    // `mark_canmallocgc`, not `mark_external_gc_effects`: random effects
    // additionally answer `RandomEffectsAnalyzer` True, which upstream never
    // does for an allocation (`effectinfo.py`) and which would make
    // every elidable caller of a bigint allocator a contradiction
    // (`assert not (elidable_function and random_effects_on_gcobjs)`,
    // `rffi.py`).
    //
    // Every entry below reaches a collection on its slow path: the
    // `*_collecting_*` allocators run a minor collection when the nursery
    // cannot satisfy the request, and the `collect_*` entries are requested
    // collections outright (`lloperation.py` marks both `gc__collect`
    // and `gc__collect_step` `canmallocgc=True`).  Their non-collecting twins
    // (`alloc_fast_nursery_typed`, `alloc_nursery_headerless_no_collect`, …)
    // are deliberately absent: they spill to old-gen instead of collecting.
    for gc_entry in [
        "alloc_nursery_collecting_typed",
        "alloc_nursery_collecting_typed_rooted",
        "alloc_fast_nursery_collecting_typed_rooted",
        "alloc_fast_nursery_collecting_typed_roots",
        "standalone_alloc_nursery_collecting_typed_rooted",
        "standalone_alloc_fast_nursery_collecting_typed_rooted",
        "standalone_alloc_fast_nursery_collecting_typed_roots",
        "collect_full",
        "collect_step",
        "collect_oldgen_nonmoving",
        "maybe_collect_for_external_malloc",
    ] {
        call_control.mark_canmallocgc(parse::CallPath::from_segments(["majit_gc", gc_entry]));
    }
    // warmspot.py `WarmRunnerDesc.__init__`: `if policy is None: policy =
    // JitPolicy()`.
    let mut default_policy;
    let policy: &mut dyn policy::JitPolicy = match policy {
        Some(policy) => policy,
        None => {
            default_policy = policy::DefaultJitPolicy::new();
            &mut default_policy
        }
    };
    // warmspot.py copies cpu.supports_* onto the policy before the BFS.
    // Both 64-bit CPUs support floats. longlong is the 32-bit CPU flag.
    // singlefloats stay off: f32 arithmetic has no cast_singlefloat_to_float
    // wrapper, so an accepted f32 graph would lower over the integer-bank
    // bit pattern.
    policy.state_mut().set_supports_floats(true);
    policy
        .state_mut()
        .set_supports_longlong(crate::codewriter::longlong::supports_longlong);
    policy.state_mut().set_supports_singlefloats(false);
    if helper_roots.is_empty() {
        call_control.find_all_graphs(policy);
    } else {
        call_control.find_helper_graphs(policy, helper_roots);
    }
    policy
        .state()
        .dump_unsafe_loops_to_udir()
        .expect("dump_unsafe_loops");
    prof.mark("  find_all_graphs");
    prof.note(|| {
        // Two different populations are each legitimately "the universe" the
        // closure could be read against, and they differ by ~4x:
        // `program.functions` is what was lowered into this program, while
        // `function_graphs` is the eagerly-lowered registry that
        // `candidate_graph_count`'s own doc names.  Print both absolutes with
        // the field each comes from — a bare quotient leaves the reader to
        // supply the denominator, and picking the other one yields a number
        // that is arithmetically correct and means something else entirely.
        format!(
            "  portal closure: {} candidate graphs; denominators: \
             {} lowered functions (program.functions), \
             {} registered graphs (function_graphs)",
            call_control.candidate_graph_count(),
            program.functions.len(),
            call_control.function_graphs().len(),
        )
    });

    // The canonical jitcode emitter below is the production analysis
    // path; `ProgramPipelineResult.functions` / `total_*` remain diagnostic fields.
    let mut pipeline = pipeline::ProgramPipelineResult {
        functions: Vec::new(),
        jit_drivers: Vec::new(),
        jitcodes: Vec::new(),
        symbolic_fnaddr_paths: Vec::new(),
        indirectcalltarget_indices: Vec::new(),
        jitcodes_by_path: indexmap::IndexMap::new(),
        insns: indexmap::IndexMap::new(),
        descrs: Vec::new(),
        all_liveness: Vec::new(),
        callinfo_rows: Vec::new(),
        ei_descr_mints: Vec::new(),
        total_blocks: 0,
        total_ops: 0,
        total_vable_rewrites: 0,
    };

    mark_phase!("call_control + canonical_trait_impls + register graphs");
    call_control.set_exc_pytype_rows(
        static_addrs
            .pytypes
            .iter()
            .map(|(key, addr)| ((*key).to_string(), *addr))
            .collect(),
    );
    let (jitcodes, indirectcalltarget_indices, insns, descrs, all_liveness, callinfo_rows) =
        make_jitcodes(
            &config.pipeline,
            &mut call_control,
            &mut prof,
            static_addrs.pytypes,
            static_addrs.pytypes_by_struct,
            static_addrs.refs,
        );
    mark_phase!("make_jitcodes");
    // warmspot.py `WarmRunnerDesc.finish` after `make_jitcodes`: unique
    // `vinfo.finish()` then `replace_force_virtualizable_with_call` over
    // remaining residual force ops. Looked-inside copies are already
    // deleted by `rewrite_op_jit_force_virtualizable`.
    call_control.finish();
    // Every body the pipeline builds is built by now.
    front::mir::report_lowering_skips(&lowering_skips);
    // callee census: how many callees the six `getcalldescr` analyzers answer as
    // upstream's declared-external arm without a declaration behind them.
    // Off by default — it is a whole extra walk of the registered universe,
    // and it reports a population, not a defect.
    if std::env::var_os("MAJIT_CALLEE_CENSUS").is_some_and(|v| v == "1") {
        eprint!("{}", call_control.unknown_callee_census());
        // the unused-override check's falsifier. `call_target_matches_loose` reports a
        // non-match by returning `false`, so an override entry that names a
        // spelling no call site carries is silently never consulted. Printed
        // beside the census because that is where the spellings it must
        // reproduce are read from.
        for line in codewriter::jtransform::call_effect_override_census(
            &config.pipeline.transform.call_effects,
        ) {
            eprintln!("[callee census] {line}");
        }
    }
    pipeline.jit_drivers = call_control
        .jitdrivers_sd()
        .iter()
        .map(|driver| pipeline::CompiledJitDriver {
            portal: driver.portal_graph.clone(),
            portal_runner: driver.portal_runner.clone(),
            main_jitcode_index: driver
                .mainjitcode
                .as_ref()
                .expect("configured JIT driver must have a main JitCode")
                .index(),
            greens: driver.greens.clone(),
            reds: driver.reds.clone(),
            // The graph-derived kinds jtransform recorded at the merge point,
            // not the declaration on `green_kinds` / `red_kinds`: those are
            // optional and a driver may leave them empty.
            green_args_spec: driver.green_args_spec.clone(),
            red_args_types: driver.red_args_types.clone(),
            virtualizables: driver.virtualizables.clone(),
        })
        .collect();
    pipeline.jitcodes = jitcodes;
    pipeline.symbolic_fnaddr_paths = call::symbolic_fnaddr_paths_snapshot();
    pipeline.indirectcalltarget_indices = indirectcalltarget_indices;
    // Mirror of `CallControl::jitcodes` (RPython `call.py self.jitcodes`)
    // captured before `call_control` is dropped. Needed because consumers
    // that look up a JitCode by graph identity cannot reconstruct the key
    // from the alloc-ordered `pipeline.jitcodes` vector alone.
    pipeline.jitcodes_by_path = call_control.jitcodes().clone();
    pipeline.insns = insns;
    pipeline.descrs = descrs;
    // Snapshotted in `make_jitcodes` from the assembler's shared `all_liveness`
    // table (the target of `pyjitpl.py:2264 self.liveness_info =
    // "".join(asm.all_liveness)`) so the runtime can resolve the `BC_LIVE`
    // offsets baked into `JitCode.code`.
    pipeline.all_liveness = all_liveness;
    pipeline.callinfo_rows = callinfo_rows;
    // Taken after `make_jitcodes`, when every `EffectInfo` has been built and
    // so every raw-set member has passed through its `get_*_descr` mint site.
    // The equivalent of what `descr.py setup_descrs` would have picked up
    // for free had the analyzer and the runtime shared one gccache.
    pipeline.ei_descr_mints = majit_ir::descr::ei_descr_mints_snapshot();

    pipeline
}

/// RPython `CodeWriter.setup_jitdriver` validation and registration.
fn register_configured_jitdrivers(
    call_control: &mut call::CallControl,
    specs: &[pipeline::JitDriverSpec],
    driver_roots: &[String],
) {
    assert!(
        !specs.is_empty(),
        "full JitCode analysis requires at least one explicit JIT driver"
    );
    for (index, spec) in specs.iter().enumerate() {
        assert!(
            !spec.portal.segments.is_empty(),
            "JIT driver {index} has an empty portal CallPath"
        );
        assert!(
            spec.red_types.is_empty() || spec.red_types.len() == spec.reds.len(),
            "JIT driver `{}` has {} red types for {} reds",
            spec.portal.canonical_key(),
            spec.red_types.len(),
            spec.reds.len(),
        );
        let portal_graph = call_control.function_graphs().get(&spec.portal);
        assert!(
            portal_graph.is_some(),
            "configured JIT driver portal `{}` does not resolve to an exact graph; \
             configure its qualified CallPath rather than a leaf-name alias",
            spec.portal.canonical_key(),
        );
        let portal_graph = portal_graph.unwrap();
        assert!(
            !specs[..index].iter().any(|previous| {
                call_control
                    .function_graphs()
                    .get(&previous.portal)
                    .is_some_and(|previous_graph| {
                        std::rc::Rc::ptr_eq(&previous_graph, &portal_graph)
                    })
            }),
            "duplicate JIT driver portal graph for `{}`; aliases of one graph \
             must share one JitDriverStaticData",
            spec.portal.canonical_key(),
        );
        // `warmspot.py find_portals` runs `autodetect_jit_markers_redvars`
        // before `split_graph_and_record_jitdriver`, so an autoreds portal
        // splits once numreds is known. `jtransform.py try_handle_jit_marker`
        // still rediscovers those reds when emitting the merge-point op.
        let portal_path = if !spec.split_portal {
            spec.portal.clone()
        } else {
            // Ends the `call_control` borrow the `register_function_graph`
            // below needs mutably; also the `warmspot.py:441` copy source.
            let original = portal_graph.clone();
            let mut portal = crate::model::copygraph(&original);
            let original_identity = original
                .source_identity
                .clone()
                .unwrap_or_else(|| original.name.clone());
            let portal_name = format!(
                "{}_portal",
                spec.portal
                    .last_segment()
                    .expect("non-empty portal path asserted above")
            );
            portal.name = portal_name.clone();
            portal.source_identity = Some(format!("{original_identity}#portal"));
            let mut portal_segments = spec.portal.segments.clone();
            *portal_segments
                .last_mut()
                .expect("non-empty portal path asserted above") = portal_name;
            let portal_path = crate::parse::CallPath::from_segments(portal_segments);
            assert!(
                !call_control.function_graphs().contains_key(&portal_path),
                "derived split portal path `{}` collides with an existing graph",
                portal_path.canonical_key(),
            );

            let (marker_block, marker_index) =
                crate::codewriter::support::find_jit_merge_point(&portal, driver_roots)
                    .expect("no jit_merge_point found in configured portal graph");
            // `support.py autodetect_jit_markers_redvars` before
            // `split_before_jit_merge_point`: declared reds stay as written,
            // autoreds learn the live set at the marker.
            let numreds = if spec.autoreds {
                let greens = match &portal
                    .block(marker_block)
                    .operations
                    .get(marker_index)
                    .expect("find_jit_merge_point returned a live marker")
                    .kind
                {
                    crate::model::OpKind::Call { args, .. } => args
                        .iter()
                        .skip(1)
                        .take(spec.greens.len())
                        .filter_map(|a| a.as_variable().cloned())
                        .collect::<Vec<_>>(),
                    _ => panic!("jit_merge_point must be a call"),
                };
                let reds = crate::codewriter::jtransform::autodetect_jit_markers_redvars(
                    &portal,
                    &greens,
                    driver_roots,
                );
                // support.py autodetect_jit_markers_redvars: `op.args.extend(reds_v)`
                // so `decode_hp_hint_args` can split greens/reds off the marker.
                match &mut portal.block_mut(marker_block).operations[marker_index].kind {
                    crate::model::OpKind::Call { args, .. } => {
                        args.extend(reds.iter().cloned().map(crate::model::LinkArg::from));
                    }
                    _ => panic!("jit_merge_point must be a call"),
                }
                reds.len()
            } else {
                spec.reds.len()
            };
            portal.startblock = crate::codewriter::support::split_before_jit_merge_point(
                &mut portal,
                marker_block,
                marker_index,
                spec.greens.len(),
                numreds,
                driver_roots,
            );
            let split_start = portal.startblock;
            crate::codewriter::support::seed_split_portal_input_ops(
                &mut portal,
                split_start,
                &spec.green_kinds,
                &spec.red_kinds,
                &spec.red_types,
            );
            // `warmspot.py WarmRunnerDesc.split_graph_and_record_jitdriver`:
            // keep the copied portal as a backend-inlining boundary and let
            // JIT policy inspect its loop.
            portal.func.dont_inline = true;
            portal.push_hint("unroll_safe");
            let split_start = portal.startblock;
            call_control.register_function_graph(portal_path.clone(), portal);
            let registered = call_control
                .function_graphs()
                .get(&portal_path)
                .expect("registered split portal path must resolve");
            // The marker heads the split block up to the operations
            // `split_block` re-emits before it: upstream inserts `same_as` of
            // a constant there and `remove_same_as` folds it into the marker's
            // operand, so the marker lands at index 0; here the receiver's
            // producer stays as a real operation because operands are
            // Variables, so only the block is asserted.
            assert_eq!(
                crate::codewriter::support::find_jit_merge_point(&registered, driver_roots)
                    .map(|(block, _)| block),
                Some(split_start),
                "registered portal path aliased away from the split graph body"
            );
            portal_path
        };
        call_control.setup_jitdriver(
            portal_path,
            spec.greens.clone(),
            spec.reds.clone(),
            spec.green_kinds.clone(),
            spec.red_kinds.clone(),
            spec.autoreds,
            spec.virtualizables.clone(),
            spec.red_types.clone(),
            spec.portal.clone(),
        );
        call_control.set_jitdriver_portal_runner(index, spec.portal_runner.clone());
        // `warmspot.py WarmRunnerDesc.rewrite_jit_merge_point` runs on
        // `_jit_merge_point_in`, the graph callers still name, after
        // `split_graph_and_record_jitdriver` has copied the loop into
        // `portal_graph`. The copy keeps the marker. `guess_call_kind`
        // classifies a direct call to `portal_runner_ptr` as recursive.
        if spec.split_portal
            && let Some(runner) = spec.portal_runner.clone()
            && let Some(original) = call_control.function_graphs_mut().get_mut(&spec.portal)
        {
            crate::codewriter::support::rewrite_jit_merge_point(
                original,
                &runner,
                spec.greens.len(),
                spec.reds.len(),
                driver_roots,
            );
        }
    }
}

/// Produce JitCodes for the graph closure reachable from configured portals.
///
/// RPython parity (`rpython/jit/codewriter/codewriter.py` `make_jitcodes`):
///
/// ```python
/// def make_jitcodes(self, verbose=False):
///     self.callcontrol.grab_initial_jitcodes()
///     all_jitcodes = []
///     for graph, jitcode in self.callcontrol.enum_pending_graphs():
///         self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
///         all_jitcodes.append(jitcode)
///     self.assembler.finished(self.callcontrol.callinfocollection)
///     return all_jitcodes
/// ```
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
fn make_jitcodes(
    pipeline_config: &pipeline::PipelineConfig,
    call_control: &mut call::CallControl,
    prof: &mut PhaseProfiler,
    pytypes: &[(&str, i64)],
    pytypes_by_struct: &[(&str, i64)],
    refs: &[(&str, i64)],
) -> (
    Vec<std::sync::Arc<jitcode::JitCode>>,
    Vec<usize>,
    indexmap::IndexMap<String, u8>,
    Vec<jitcode::BhDescr>,
    Vec<u8>,
    Vec<majit_ir::effectinfo::CallInfoRow>,
) {
    // RPython codewriter.py: make_jitcodes().
    //
    // `Arc<JitCode>` shells live in `CallControl::jitcodes`; the drain loop
    // commits each shell's body via `JitCode::set_body`. After all phases,
    // `collect_jitcodes_in_alloc_order` materialises the `all_jitcodes[]`
    // vector with `all_jitcodes[i].index == i` (RPython codewriter.py
    // invariant).
    call_control.set_struct_storage(&pipeline_config.transform.struct_storage);
    let mut codewriter = codewriter::CodeWriter::new();
    // `Assembler.constants_r` is filled while this codewriter runs.
    // The name rows live on that assembler, not a process-global map.
    codewriter.assembler.intern_type_static_addrs(pytypes);
    codewriter
        .assembler
        .intern_type_static_addrs(pytypes_by_struct);
    codewriter.assembler.intern_static_ref_addrs(refs);

    // `warmspot.py:262-264` `vrefinfo = VirtualRefInfo(self);
    //  self.codewriter.setup_vrefinfo(vrefinfo)` — installs the
    // virtualref descr-index carrier on the codewriter's callcontrol
    // BEFORE `make_jitcodes` runs (line 281 in warmspot).  Pyre uses
    // a const-backed `DefaultVirtualRefInfoHandle` so majit-translate
    // can perform the install without depending on majit-metainterp
    // (the concrete `VirtualRefInfo` lives there).  The runtime path
    // through `MetaInterpStaticData::finish_setup` reads the handle
    // back at `pyjitpl.py self.virtualref_info = codewriter.
    // callcontrol.virtualref_info` parity, so this site is the
    // codewrite-time anchor for that read.
    codewriter.setup_vrefinfo(
        call_control,
        std::sync::Arc::new(call::DefaultVirtualRefInfoHandle),
    );

    // RPython grab_initial_jitcodes + drain portals and their callees.
    // RPython call.py.
    call_control.grab_initial_jitcodes();
    prof.mark("  grab_initial_jitcodes");
    // Two-phase rtyper prepass: annotate-all → rtype-all over the portal closure
    // before the drain publishes any covered graph, so a shared callee (e.g.
    // `type_error`) is unioned across all its callers before being rtyped.
    codewriter.run_two_phase_prepass(call_control);
    prof.mark("  run_two_phase_prepass");
    // RPython lowers indirect calls once on the translator's graphs,
    // before codewriter and the effect analyzers read them. The prepass
    // above still needed `family_key`; this consumes it.
    call_control.lower_registered_indirect_calls();
    prof.mark("  lower_registered_indirect_calls");
    codewriter.drain_pending_graphs(call_control, &pipeline_config.transform);
    prof.mark("  drain_pending_graphs");

    // `BuiltinCode.func` is a SomePBC function-pointer field.  The ordinary
    // translated indirect-call op contributes this family through
    // `IndirectCallTargets`; pyre's interpreter call boundary hides that op
    // behind the runtime `call_fn` helper, so publish the annotator's same
    // finite wrapper family on the shared Assembler explicitly.  The handles
    // are the exact `CallControl.jitcodes` objects materialized by
    // `grab_initial_jitcodes`, preserving RPython object identity.
    let builtin_wrapper_targets: Vec<jitcode::JitCodeHandle> = call_control
        .builtin_wrapper_indirect_graphs()
        .iter()
        .filter_map(|path| call_control.jitcodes().get(path).cloned())
        .map(jitcode::JitCodeHandle::from)
        .collect();
    codewriter
        .assembler
        .indirectcalltargets
        .extend(builtin_wrapper_targets);

    // RPython codewriter.py:85: self.assembler.finished(callinfocollection).
    codewriter
        .assembler
        .finished(&call_control.callinfocollection);
    prof.mark("  assembler.finished");

    // The collection's calldescr is the one `getcalldescr` handed the
    // residual call. Pin each row to that descr's index in `Assembler.descrs`
    // before the snapshot, inserting a missing descr so the index exists.
    let callinfo_rows =
        export_callinfo_rows(&mut codewriter.assembler, &call_control.callinfocollection);

    // Materialise `all_jitcodes[]` from the completed jitcodes. Each
    // jitcode receives its dense index when appended, matching RPython
    // `make_jitcodes()`.
    let jitcodes = call_control.collect_jitcodes_in_alloc_order();
    let mut indirectcalltarget_indices: Vec<usize> = codewriter
        .assembler
        .indirectcalltargets
        .iter()
        .map(|target| target.index())
        .collect();
    indirectcalltarget_indices.sort_unstable();
    indirectcalltarget_indices.dedup();

    // RPython codewriter.py + assembler.py: `Assembler.insns` grows as
    // `write_insn` encounters new keys.  We snapshot the final table
    // here so the runtime can map `JitCode.code[i]` bytes back to
    // opnames — the key consumed by `BlackholeInterpBuilder::setup_insns`.
    let insns = codewriter.assembler.insns().clone();

    // RPython blackhole.py `self.setup_descrs(asm.descrs)` — the
    // shared descr table every 'd'/'j' argcode indexes into at runtime.
    // Snapshotted here so the build artifact carries it alongside
    // `insns`, mirroring RPython's single-store model.
    let descrs: Vec<jitcode::BhDescr> = codewriter.assembler.snapshot_descrs();

    if std::env::var_os("MAJIT_DESCR_POOL_CENSUS").is_some_and(|v| v == "1") {
        let dup = codewriter.assembler.descr_pool_duplication();
        // C1: the mint universe, beside the pool, in the same generation.
        // `all_descrs` stable while the pool moves is a SELECTION defect;
        // both moving is a POOL defect.  No artefact diff can tell them apart.
        let all_descrs = majit_ir::descr::gc_cache().lock().all_descrs_len();
        // C2: one cache slot answering for two different logical fields.
        let (field_hits, field_collisions) = majit_ir::descr::field_descr_cache_collisions();
        eprintln!(
            "[descriptor pool] total {} | all_descrs {} | effect-keyed {} | \
             concrete {} | wildcard {} | comparable {} | structurally distinct {} | \
             invariant violations {} | field-descr cache hits {} collisions {}",
            dup.total,
            all_descrs,
            dup.effect_keyed,
            dup.concrete,
            dup.wildcard,
            dup.comparable,
            dup.structurally_distinct,
            dup.invariant_violations,
            field_hits,
            field_collisions,
        );
        // The three counters above all measure the pool POPULATION, and all
        // three came back stable while `descrs.bin` still changed size.  The
        // bytes therefore moved inside entries, which no population counter
        // can see.  This walks exactly the slice that gets serialized and
        // sums each length-bearing channel separately, so the next A/B names
        // WHICH channel moved instead of only that the file did.
        let content = jitcode::descr_pool_content(&descrs);
        eprintln!(
            "[descriptor content] entries {} | strings {} ({} bytes) | \
             vec members {} | descr-set members {} | wildcard effects {} | \
             type_ids {} (sum {:#018x}) | kinds {:?}",
            content.entries,
            content.string_count,
            content.string_bytes,
            content.vec_members,
            content.descr_set_members,
            content.wildcard_effects,
            content.type_id_count,
            content.type_id_sum,
            content.kind_counts,
        );
        // The pool key carries every component of `BhDescr::Field`, so it is
        // injective and two entries for one logical field must differ
        // somewhere.  Printing the differing spellings turns the run-to-run
        // count difference into a single-run observation naming the component
        // that varied — the count is intermittent, so a cross-run diff can
        // easily sample two runs that agree and show nothing.
        // M1. Printed on the same line as the split totals because it is a
        // SUBSET of them and reading it alone invites the wrong denominator.
        // Zero says the `Option<usize>` key added no pool entry, so nothing
        // downstream that counts pool entries moved — the claim the widening
        // has to make good on, taken here rather than argued from the key's
        // shape. It must be READ OUT, not merely computed: a number nobody
        // prints is the same instrument as no number at all, which is the
        // defect this whole change exists to repair.
        eprintln!(
            "[field split] groups {} | entries {} | index-provenance splits {}",
            content.field_dupe_groups,
            content.field_dupe_entries,
            content.field_index_provenance_splits,
        );
        for line in &content.field_dupe_report {
            eprintln!("[field split] {line}");
        }
        // Two walks of the same population that disagree on its size make
        // every per-channel sum above unattributable.
        if content.entries != dup.total {
            eprintln!(
                "[descriptor content] content walk saw {} entries, pool \
                 census saw {} — the two walks disagree and every channel \
                 sum above is suspect",
                content.entries, dup.total
            );
        }
        // The census asserts nothing until it is shown to have seen both
        // shapes: a run with no `concrete` entry cannot have seen an
        // over-split whatever `structurally_distinct` says.  Reported rather
        // than asserted so a build still completes and says why.
        if !dup.counts_reconcile() {
            eprintln!(
                "[descriptor pool] counts do not reconcile — the shape \
                 classification lost entries; every field above is suspect"
            );
        } else if !dup.saw_both_shapes() {
            eprintln!(
                "[descriptor pool] census saw only one shape (concrete {}, \
                 wildcard {}) — `structurally distinct` is NOT interpretable",
                dup.concrete, dup.wildcard
            );
        }
    }

    // RPython `pyjitpl.py MetaInterpStaticData.finish_setup self.liveness_info = "".join(asm.all_liveness)`.
    // Snapshot the assembler's shared `all_liveness` byte stream so the runtime
    // can resolve the `BC_LIVE` offsets baked into `JitCode.code`.
    let all_liveness = codewriter.assembler.all_liveness().to_vec();

    (
        jitcodes,
        indirectcalltarget_indices,
        insns,
        descrs,
        all_liveness,
        callinfo_rows,
    )
}

/// `(oopspecindex, Assembler.descrs index, build-time func address)` for
/// every row `callinfocollection.add` recorded.
fn export_callinfo_rows(
    assembler: &mut codewriter::assembler::Assembler,
    cic: &majit_ir::CallInfoCollection,
) -> Vec<majit_ir::effectinfo::CallInfoRow> {
    cic.entries_in_oopspec_order()
        .into_iter()
        .map(|(oopspec, calldescr, func)| {
            let descr_index = callinfo_descr_index(assembler, oopspec, &calldescr);
            majit_ir::effectinfo::CallInfoRow {
                oopspecindex: oopspec,
                descr_index: u32::try_from(descr_index).expect("descr index exceeds u32"),
                func: func as i64,
            }
        })
        .collect()
}

/// `assembler.py` `_descr_dict`: the calldescr resolves to the index the
/// residual call's operand already holds, or is inserted if no call site
/// reached the assembler.
fn callinfo_descr_index(
    assembler: &mut codewriter::assembler::Assembler,
    oopspec: majit_ir::OopSpecIndex,
    calldescr: &majit_ir::DescrRef,
) -> usize {
    let Some(cd) = calldescr.as_call_descr() else {
        panic!("callinfocollection row {oopspec:?} is not a calldescr");
    };
    let result_type = cd.result_type();
    let result_signed = cd.is_result_signed();
    let result_size = cd.result_size();
    assembler.emit_ready_descr(jitcode::BhDescr::Call {
        calldescr: jitcode::BhCallDescr {
            arg_classes: cd.arg_classes(),
            result_type: cd.result_class(),
            result_signed,
            result_size,
            result_erased: jitcode::CallResultErasedKey::from_ir_layout(
                result_type,
                result_signed,
                result_size,
            ),
            void_word_abi: cd.result_class() == 'v' && result_size == 8,
            extra_info: cd.get_extra_info().clone(),
            translated_effect_info_id: None,
            call_stub: std::sync::OnceLock::new(),
        },
    })
}

/// Generate tracing code directly from the canonical pipeline result.
pub fn generate_trace_code_from_pipeline(result: &pipeline::ProgramPipelineResult) -> String {
    codegen::generate_from_pipeline(result)
}

/// Like [`generate_trace_code_from_pipeline`] but takes a
/// [`codegen::CodegenFlavor`]. The flavor is currently a no-op (the
/// pyre-specific helpers it used to gate now live in
/// `pyre-jit-trace/src/trace_helpers.rs`); retained for external callers.
pub fn generate_trace_code_from_pipeline_with_flavor(
    result: &pipeline::ProgramPipelineResult,
    flavor: codegen::CodegenFlavor,
) -> String {
    codegen::generate_from_pipeline_with_flavor(result, flavor)
}

pub use codegen::CodegenFlavor;

/// Produce a recognition report for canonical driver and JitCode output.
pub fn recognition_report(result: &pipeline::ProgramPipelineResult) -> codegen::RecognitionReport {
    codegen::recognition_report(result)
}
pub use codegen::{JitCodeRecognition, RecognitionReport};

/// Generate code from graph pipeline results.
/// `rlib` — Rust port of `rpython/rlib/` helpers pulled in on demand.
/// Currently only the pieces required by the annotator port are
/// present (rarithmetic subset for `compute_restype`).
pub mod rlib;

#[cfg(test)]
mod portal_driver_tests {
    use super::*;

    #[test]
    fn unique_impl_leaf_census_deduplicates_aliases_by_struct_identity() {
        let mut program = front::SemanticProgram::default();
        let pyframe = majit_ir::descr::StructId::from_canonical("pyframe::PyFrame");
        for alias in [
            "PyFrame",
            "pyframe::PyFrame",
            "pyre_interpreter::pyframe::PyFrame",
        ] {
            program
                .struct_fields
                .fields
                .insert(alias.to_string(), Vec::new());
            program.struct_ids.insert(alias.to_string(), Some(pyframe));
        }
        let identities = distinct_struct_identities_by_leaf(&program);
        assert_eq!(
            identities.get("PyFrame").map(Vec::len),
            Some(1),
            "lookup aliases of one RPython class identity are not ambiguous"
        );

        let mut owners = TraitImplOwners::new();
        owners
            .entry("pyre_interpreter::pyopcode::OpcodeStepExecutor".to_string())
            .or_default()
            .insert("pyframe::PyFrame".to_string());
        let unique = unique_trait_impl_roots(&program, owners);
        assert_eq!(
            unique.get("pyre_interpreter::pyopcode::OpcodeStepExecutor"),
            Some(&"pyre_interpreter::pyframe::PyFrame".to_string()),
            "one class identity uses its fully-qualified registry spelling"
        );

        let mut owners = TraitImplOwners::new();
        let bucket = owners
            .entry("pyre_interpreter::pyopcode::OpcodeStepExecutor".to_string())
            .or_default();
        bucket.insert("PyFrame".to_string());
        bucket.insert("pyframe::PyFrame".to_string());
        let unique = unique_trait_impl_roots(&program, owners);
        assert_eq!(
            unique.get("pyre_interpreter::pyopcode::OpcodeStepExecutor"),
            Some(&"pyre_interpreter::pyframe::PyFrame".to_string()),
            "two lookup spellings of one StructId stay one class"
        );

        let foreign = "other_runtime::PyFrame";
        program
            .struct_fields
            .fields
            .insert(foreign.to_string(), Vec::new());
        program.struct_ids.insert(
            foreign.to_string(),
            Some(majit_ir::descr::StructId::from_canonical(foreign)),
        );
        let identities = distinct_struct_identities_by_leaf(&program);
        assert_eq!(
            identities.get("PyFrame").map(Vec::len),
            Some(2),
            "genuinely distinct same-leaf classes remain ambiguous"
        );
    }

    #[test]
    fn unique_trait_default_override_is_method_agnostic() {
        let store = front::semantic::SemanticFunction::with_empty_graph(
            "store_attr_cached",
            "PyFrame::store_attr_cached",
            Some("PyFrame".into()),
            Some("OpcodeStepExecutor".into()),
            Some("pyre_interpreter::pyopcode::OpcodeStepExecutor".into()),
        );
        let to_bool = front::semantic::SemanticFunction::with_empty_graph(
            "to_bool",
            "PyFrame::to_bool",
            Some("PyFrame".into()),
            Some("OpcodeStepExecutor".into()),
            Some("pyre_interpreter::pyopcode::OpcodeStepExecutor".into()),
        );
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            ("OpcodeStepExecutor", "store_attr_cached"),
            ("PyFrame", &store),
        );
        overrides.insert(("OpcodeStepExecutor", "to_bool"), ("PyFrame", &to_bool));
        let mut types = std::collections::HashMap::new();
        types.insert("OpcodeStepExecutor", vec!["PyFrame"]);
        let program = front::SemanticProgram::default();
        let leaf_identities = std::collections::HashMap::new();

        let hit = unique_trait_default_override(
            &program,
            &leaf_identities,
            true,
            "OpcodeStepExecutor",
            "store_attr_cached",
            &overrides,
            &types,
        );
        assert_eq!(
            hit.map(|(owner, func)| (owner, func.graph().name.as_str())),
            Some(("PyFrame", "PyFrame::store_attr_cached"))
        );
        let hit = unique_trait_default_override(
            &program,
            &leaf_identities,
            true,
            "OpcodeStepExecutor",
            "to_bool",
            &overrides,
            &types,
        );
        assert_eq!(
            hit.map(|(owner, func)| (owner, func.graph().name.as_str())),
            Some(("PyFrame", "PyFrame::to_bool"))
        );
        assert!(
            unique_trait_default_override(
                &program,
                &leaf_identities,
                true,
                "OpcodeStepExecutor",
                "no_override",
                &overrides,
                &types,
            )
            .is_none(),
            "a method the unique impl does not override keeps the trait default"
        );
        assert!(
            unique_trait_default_override(
                &program,
                &leaf_identities,
                false,
                "OpcodeStepExecutor",
                "store_attr_cached",
                &overrides,
                &types,
            )
            .is_none(),
            "a concrete impl entry is not a default-body shadow"
        );

        types.insert("OpcodeStepExecutor", vec!["PyFrame", "OtherFrame"]);
        assert!(
            unique_trait_default_override(
                &program,
                &leaf_identities,
                true,
                "OpcodeStepExecutor",
                "store_attr_cached",
                &overrides,
                &types,
            )
            .is_none(),
            "several concrete impls leave the default on the direct path"
        );
    }

    fn sem_fn(
        name: &str,
        graph: &str,
        owner: Option<&str>,
        trait_root: Option<&str>,
        trait_qualified: Option<&str>,
    ) -> front::semantic::SemanticFunction {
        front::semantic::SemanticFunction::with_empty_graph(
            name,
            graph,
            owner.map(str::to_string),
            trait_root.map(str::to_string),
            trait_qualified.map(str::to_string),
        )
    }

    fn pyframe_id() -> majit_ir::descr::StructId {
        majit_ir::descr::StructId::from_canonical("pyre_interpreter::pyframe::PyFrame")
    }

    /// T1. Two lookup spellings of one class still bind the override.
    #[test]
    fn alias_spellings_of_one_class_bind_the_trait_default_override() {
        let id = pyframe_id();
        let mut program = front::SemanticProgram::default();
        for alias in ["pyframe::PyFrame", "pyre_interpreter::pyframe::PyFrame"] {
            program.struct_ids.insert(alias.to_string(), Some(id));
        }
        let first = sem_fn(
            "to_bool",
            "PyFrame::to_bool",
            Some("pyframe::PyFrame"),
            Some("OpcodeStepExecutor"),
            Some("pyre_interpreter::pyopcode::OpcodeStepExecutor"),
        );
        let second = sem_fn(
            "to_bool",
            "PyFrame::to_bool",
            Some("pyre_interpreter::pyframe::PyFrame"),
            Some("OpcodeStepExecutor"),
            Some("pyre_interpreter::pyopcode::OpcodeStepExecutor"),
        );
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            ("OpcodeStepExecutor", "to_bool"),
            (first.self_ty_root.as_deref().unwrap(), &first),
        );
        overrides.insert(
            ("OpcodeStepExecutor", "to_bool"),
            (second.self_ty_root.as_deref().unwrap(), &second),
        );
        let mut types = std::collections::HashMap::new();
        types.insert(
            "OpcodeStepExecutor",
            vec!["pyframe::PyFrame", "pyre_interpreter::pyframe::PyFrame"],
        );
        let leaf_identities = std::collections::HashMap::new();
        let hit = unique_trait_default_override(
            &program,
            &leaf_identities,
            true,
            "OpcodeStepExecutor",
            "to_bool",
            &overrides,
            &types,
        );
        let (owner, func) = hit.expect("alias spellings of one StructId bind the override");
        assert!(
            owner == "pyframe::PyFrame" || owner == "pyre_interpreter::pyframe::PyFrame",
            "owner {owner}"
        );
        assert_eq!(func.graph().name, "PyFrame::to_bool");
    }

    /// T2. Two struct identities keep the trait default.
    #[test]
    fn two_struct_identities_keep_the_trait_default() {
        let mut program = front::SemanticProgram::default();
        program.struct_ids.insert(
            "pyframe::PyFrame".into(),
            Some(majit_ir::descr::StructId::from_canonical(
                "pyframe::PyFrame",
            )),
        );
        program.struct_ids.insert(
            "other::OtherFrame".into(),
            Some(majit_ir::descr::StructId::from_canonical(
                "other::OtherFrame",
            )),
        );
        let override_fn = sem_fn(
            "to_bool",
            "PyFrame::to_bool",
            Some("pyframe::PyFrame"),
            Some("OpcodeStepExecutor"),
            Some("pyre_interpreter::pyopcode::OpcodeStepExecutor"),
        );
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            ("OpcodeStepExecutor", "to_bool"),
            ("pyframe::PyFrame", &override_fn),
        );
        let mut types = std::collections::HashMap::new();
        types.insert(
            "OpcodeStepExecutor",
            vec!["pyframe::PyFrame", "other::OtherFrame"],
        );
        let leaf_identities = std::collections::HashMap::new();
        assert!(
            unique_trait_default_override(
                &program,
                &leaf_identities,
                true,
                "OpcodeStepExecutor",
                "to_bool",
                &overrides,
                &types,
            )
            .is_none(),
            "two struct identities leave the default on the direct path"
        );
    }

    /// T3. The registration loop binds `[OpcodeStepExecutor, to_bool]` to
    /// the override and leaves an unimplemented method on the default graph.
    #[test]
    fn registration_binds_unique_override_and_keeps_unoverridden_default() {
        let id = pyframe_id();
        let mut program = front::SemanticProgram::default();
        for alias in ["pyframe::PyFrame", "pyre_interpreter::pyframe::PyFrame"] {
            program.struct_ids.insert(alias.to_string(), Some(id));
        }
        program.functions.push(sem_fn(
            "to_bool",
            "OpcodeStepExecutor::to_bool",
            None,
            Some("OpcodeStepExecutor"),
            None,
        ));
        program.functions.push(sem_fn(
            "to_bool",
            "PyFrame::to_bool",
            Some("pyframe::PyFrame"),
            Some("OpcodeStepExecutor"),
            Some("pyre_interpreter::pyopcode::OpcodeStepExecutor"),
        ));
        program.functions.push(sem_fn(
            "to_bool",
            "PyFrame::to_bool",
            Some("pyre_interpreter::pyframe::PyFrame"),
            Some("OpcodeStepExecutor"),
            Some("pyre_interpreter::pyopcode::OpcodeStepExecutor"),
        ));
        program.functions.push(sem_fn(
            "rotate3",
            "OpcodeStepExecutor::rotate3",
            None,
            Some("OpcodeStepExecutor"),
            None,
        ));
        let leaf_identities = distinct_struct_identities_by_leaf(&program);
        let mut call_control = call::CallControl::new();
        bind_trait_default_direct_paths(&program, &mut call_control, &leaf_identities);
        let to_bool = call_control
            .function_graphs()
            .get(&CallPath::from_segments(["OpcodeStepExecutor", "to_bool"]))
            .expect("direct path registered");
        assert_eq!(to_bool.name, "PyFrame::to_bool");
        let rotate3 = call_control
            .function_graphs()
            .get(&CallPath::from_segments(["OpcodeStepExecutor", "rotate3"]))
            .expect("unoverridden default stays registered");
        assert_eq!(rotate3.name, "OpcodeStepExecutor::rotate3");
    }

    /// T4. The override key is the trait leaf, matching the default's
    /// `trait_root`, not the qualified trait path.
    #[test]
    fn trait_default_override_key_is_the_leaf_not_the_qualified_path() {
        let id = pyframe_id();
        let mut program = front::SemanticProgram::default();
        program
            .struct_ids
            .insert("pyframe::PyFrame".into(), Some(id));
        let override_fn = sem_fn(
            "to_bool",
            "PyFrame::to_bool",
            Some("pyframe::PyFrame"),
            Some("OpcodeStepExecutor"),
            Some("pyre_interpreter::pyopcode::OpcodeStepExecutor"),
        );
        let mut types = std::collections::HashMap::new();
        let mut overrides = std::collections::HashMap::new();
        index_concrete_trait_method(&override_fn, &mut types, &mut overrides);
        assert!(overrides.contains_key(&("OpcodeStepExecutor", "to_bool")));
        assert!(
            !overrides.contains_key(&("pyre_interpreter::pyopcode::OpcodeStepExecutor", "to_bool")),
            "the qualified trait path is not the override key"
        );
        let leaf_identities = std::collections::HashMap::new();
        let hit = unique_trait_default_override(
            &program,
            &leaf_identities,
            true,
            "OpcodeStepExecutor",
            "to_bool",
            &overrides,
            &types,
        );
        assert_eq!(
            hit.map(|(owner, func)| (owner, func.graph().name.as_str())),
            Some(("pyframe::PyFrame", "PyFrame::to_bool"))
        );
        assert!(
            unique_trait_default_override(
                &program,
                &leaf_identities,
                true,
                "pyre_interpreter::pyopcode::OpcodeStepExecutor",
                "to_bool",
                &overrides,
                &types,
            )
            .is_none(),
            "looking the leaf-keyed override up under the qualified path misses"
        );
    }

    fn driver(portal: CallPath) -> pipeline::JitDriverSpec {
        pipeline::JitDriverSpec {
            portal,
            portal_runner: None,
            greens: Vec::new(),
            reds: Vec::new(),
            green_kinds: Vec::new(),
            red_kinds: Vec::new(),
            autoreds: false,
            virtualizables: Vec::new(),
            red_types: Vec::new(),
            split_portal: false,
        }
    }

    fn return_graph(name: &str) -> FunctionGraph {
        let mut graph = FunctionGraph::new(name);
        graph.set_return(graph.startblock, None);
        graph
    }

    #[test]
    #[should_panic(expected = "duplicate JIT driver portal graph")]
    fn rejects_two_aliases_of_one_portal_graph() {
        let mut call_control = call::CallControl::new();
        let first = CallPath::from_segments(["first_alias"]);
        let second = CallPath::from_segments(["second_alias"]);
        // GraphStore intentionally interns aliases of one source graph. The
        // driver layer must preserve that identity instead of allocating two
        // JitDriverStaticData records for it.
        call_control.register_function_graph(first.clone(), return_graph("portal"));
        call_control.register_function_graph(second.clone(), return_graph("portal"));

        register_configured_jitdrivers(
            &mut call_control,
            &[driver(first), driver(second)],
            &GraphTransformConfig::default().jitdriver_receiver_roots,
        );
    }

    #[test]
    #[should_panic(expected = "derived split portal path `fixture::eval_loop_jit_portal` collides")]
    fn rejects_split_portal_path_collision() {
        use crate::codewriter::type_state::ConcreteType;

        let mut call_control = call::CallControl::new();
        let portal = CallPath::from_segments(["fixture", "eval_loop_jit"]);
        let derived = CallPath::from_segments(["fixture", "eval_loop_jit_portal"]);
        let mut graph = FunctionGraph::new("eval_loop_jit");
        let receiver = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        graph.block_mut(graph.startblock).inputargs = vec![receiver.clone()];
        graph
            .block_mut(graph.startblock)
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::method("jit_merge_point", Some("PyPyJitDriver".into())),
                    args: crate::model::call_args(vec![receiver]),
                    result_ty: ValueType::Void,
                },
            });
        graph.set_return(graph.startblock, None);
        call_control.register_function_graph(portal.clone(), graph);
        call_control.register_function_graph(derived, return_graph("unrelated_existing_graph"));
        let mut spec = driver(portal);
        spec.split_portal = true;

        register_configured_jitdrivers(
            &mut call_control,
            &[spec],
            &GraphTransformConfig::default().jitdriver_receiver_roots,
        );
    }

    #[test]
    fn inherent_method_on_raw_storage_owner_resolves_as_portal() {
        // `call.py CallControl.setup_jitdriver` / `warmspot.py find_portals`
        // take the portal graph from the function object. An inherent method
        // is an ordinary function graph whatever the gckind of its `self`:
        // a raw `self` changes how *calls* to the method lower (`direct_call`
        // with an address argument), never whether the method's own graph
        // exists or the path it is registered under.
        let mut call_control = call::CallControl::new();
        let portal = CallPath::for_impl_method("mod::RawOwner", "dispatch");
        assert_eq!(
            portal,
            CallPath::from_segments(["mod", "RawOwner", "dispatch"])
        );
        call_control.register_function_graph(portal.clone(), return_graph("dispatch"));
        register_configured_jitdrivers(
            &mut call_control,
            &[driver(portal.clone())],
            &GraphTransformConfig::default().jitdriver_receiver_roots,
        );
        assert!(
            call_control.function_graphs().get(&portal).is_some(),
            "the configured portal must resolve the inherent method graph"
        );
        assert_eq!(call_control.jitdrivers_sd().len(), 1);
        assert_eq!(call_control.jitdrivers_sd()[0].portal_graph, portal);
    }

    #[test]
    fn make_jitcodes_compiles_a_registered_portal_graph() {
        let mut call_control = call::CallControl::new();
        let portal = CallPath::from_segments(["fixture", "portal"]);
        call_control.register_function_graph(portal.clone(), return_graph("portal"));
        let config = pipeline::PipelineConfig {
            transform: GraphTransformConfig::default(),
            jit_drivers: vec![driver(portal.clone())],
            register_trait_families: Vec::new(),
            helper_graphs: Vec::new(),
            builtin_graphs: Vec::new(),
        };
        register_configured_jitdrivers(
            &mut call_control,
            &config.jit_drivers,
            &config.transform.jitdriver_receiver_roots,
        );
        let mut policy = policy::DefaultJitPolicy::new();
        call_control.find_all_graphs(&mut policy);

        let (jitcodes, _, _, _, _, _) = make_jitcodes(
            &config,
            &mut call_control,
            &mut PhaseProfiler::new(),
            &[],
            &[],
            &[],
        );
        assert_eq!(jitcodes.len(), 1);
        assert_eq!(jitcodes[0].index(), 0);
        assert!(call_control.jitcodes().contains_key(&portal));
    }

    /// A portal whose body calls a `dyn` method: one family with a
    /// registered impl, one with none. The public `CodeWriter::make_jitcodes`
    /// must lower both before jtransform, the unregistered one to the
    /// unknown family.
    #[test]
    fn codewriter_make_jitcodes_lowers_indirect_calls() {
        let handler_impl = |name: &str| {
            let mut graph = FunctionGraph::new(name);
            graph
                .push_op_var(
                    graph.startblock,
                    OpKind::Input {
                        name: "self".to_string(),
                        ty: ValueType::Ref(None),
                        class_root: None,
                    },
                    true,
                )
                .unwrap();
            graph.set_return(graph.startblock, None);
            graph
        };
        let mut call_control = call::CallControl::new();
        call_control.register_trait_method("run", Some("Handler"), "A", handler_impl("A::run"));
        let portal = CallPath::from_segments(["fixture", "portal"]);
        let mut graph = FunctionGraph::new("portal");
        let receiver = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "handler".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        for (trait_root, method) in [("Handler", "run"), ("Unregistered", "go")] {
            graph.push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::indirect(trait_root, method),
                    args: crate::model::call_args(vec![receiver.clone()]),
                    result_ty: ValueType::Void,
                },
                true,
            );
        }
        graph.set_return(graph.startblock, None);
        call_control.register_function_graph(portal.clone(), graph);
        let config = pipeline::PipelineConfig {
            transform: GraphTransformConfig::default(),
            jit_drivers: vec![driver(portal.clone())],
            register_trait_families: Vec::new(),
            helper_graphs: Vec::new(),
            builtin_graphs: Vec::new(),
        };
        register_configured_jitdrivers(
            &mut call_control,
            &config.jit_drivers,
            &config.transform.jitdriver_receiver_roots,
        );
        let mut policy = policy::DefaultJitPolicy::new();
        call_control.find_all_graphs(&mut policy);

        let mut codewriter = codewriter::CodeWriter::new();
        let all = codewriter.make_jitcodes(&mut call_control, &config.transform);
        assert!(all.by_path.contains_key(&portal));

        // The in-place pass fills `Handler::run` with its one registered
        // impl and leaves `Unregistered::go` as a `CallTarget::Indirect`.
        let stored = call_control
            .function_graphs()
            .get(&portal)
            .expect("portal graph stays registered");
        let ops: Vec<_> = stored
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .collect();
        let filled: Vec<_> = ops
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::IndirectCall {
                    graphs, family_key, ..
                } => Some((graphs.clone(), family_key.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            filled,
            vec![(Some(vec![CallPath::from_segments(["A", "run"])]), None)],
            "Handler::run must keep A::run as its only candidate"
        );
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Indirect { trait_root, method_name },
                    ..
                } if trait_root == "Unregistered" && method_name == "go"
            )),
            "Unregistered::go must stay an unresolved indirect call"
        );
    }

    #[test]
    fn make_jitcodes_keeps_explicit_driver_reds_from_marker_callsite() {
        use crate::codewriter::flatten::FlatOp;
        use crate::codewriter::type_state::ConcreteType;

        let mut call_control = call::CallControl::new();
        let portal = CallPath::from_segments(["fixture", "eval_loop_jit"]);
        let mut graph = FunctionGraph::new("pyre_jit::eval::eval_loop_jit");
        let next_instr = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let is_being_profiled = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let pycode = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let frame = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let ec = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        graph.blocks[graph.startblock.0].inputargs = vec![
            next_instr.clone(),
            is_being_profiled.clone(),
            pycode.clone(),
            frame.clone(),
            ec.clone(),
        ];
        graph.blocks[graph.startblock.0]
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::method("jit_merge_point", Some("PyPyJitDriver".into())),
                    args: crate::model::call_args(vec![
                        frame.clone(),
                        next_instr,
                        is_being_profiled,
                        pycode,
                        frame.clone(),
                        ec.clone(),
                    ]),
                    result_ty: ValueType::Void,
                },
            });
        graph.set_return(graph.startblock, None);
        call_control.register_function_graph(portal.clone(), graph);

        let config = pipeline::PipelineConfig {
            transform: GraphTransformConfig::default(),
            jit_drivers: vec![pipeline::JitDriverSpec {
                portal: portal.clone(),
                portal_runner: Some(CallPath::from_segments(["fixture", "portal_runner"])),
                greens: vec![
                    "next_instr".into(),
                    "is_being_profiled".into(),
                    "pycode".into(),
                ],
                reds: vec!["frame".into(), "ec".into()],
                green_kinds: Vec::new(),
                red_kinds: Vec::new(),
                autoreds: false,
                virtualizables: vec!["frame".into()],
                red_types: vec!["PyFrame".into(), "ExecutionContext".into()],
                split_portal: true,
            }],
            register_trait_families: Vec::new(),
            helper_graphs: Vec::new(),
            builtin_graphs: Vec::new(),
        };
        register_configured_jitdrivers(
            &mut call_control,
            &config.jit_drivers,
            &config.transform.jitdriver_receiver_roots,
        );
        let split_portal = CallPath::from_segments(["fixture", "eval_loop_jit_portal"]);
        let original_graph = call_control
            .function_graphs()
            .get(&portal)
            .expect("original portal graph remains registered");
        let calls_runner = original_graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => segments.as_slice() == ["fixture", "portal_runner"],
                _ => false,
            })
        });
        assert!(
            calls_runner,
            "rewrite_jit_merge_point must leave a direct call to portal_runner"
        );
        assert!(
            !original_graph.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call { target, .. }
                            if crate::codewriter::jtransform::jit_marker_key_from_target(
                                target,
                                &config.transform.jitdriver_receiver_roots,
                            ) == Some(crate::codewriter::jtransform::JitMarkerKey::JitMergePoint)
                    )
                })
            }),
            "original portal must not keep jit_merge_point after the rewrite"
        );
        assert_eq!(call_control.jitdrivers_sd()[0].portal_graph, split_portal);
        assert_eq!(
            call_control.jitdrivers_sd()[0].portal_runner,
            Some(CallPath::from_segments(["fixture", "portal_runner"]))
        );
        let split_graph = call_control
            .function_graphs()
            .get(&split_portal)
            .expect("split portal graph must be registered under its derived path");
        assert_eq!(split_graph.name, "eval_loop_jit_portal");
        assert!(split_graph.func.dont_inline);
        assert!(split_graph.hints.iter().any(|hint| hint == "unroll_safe"));
        assert_eq!(
            split_graph.source_identity.as_deref(),
            Some("pyre_jit::eval::eval_loop_jit#portal")
        );
        let split_reds = split_graph.block(split_graph.startblock).inputargs[3..5].to_vec();
        let mut policy = policy::DefaultJitPolicy::new();
        call_control.find_all_graphs(&mut policy);

        let (jitcodes, _, _, _, _, _) = make_jitcodes(
            &config,
            &mut call_control,
            &mut PhaseProfiler::new(),
            &[],
            &[],
            &[],
        );
        assert_eq!(jitcodes[0].name, "eval_loop_jit_portal");
        let merge = jitcodes[0]
            .body()
            ._ssarepr
            .as_ref()
            .and_then(|ssarepr| ssarepr.as_any().downcast_ref::<SSARepr>())
            .expect("assembled portal keeps its SSA representation")
            .insns
            .iter()
            .find_map(|flat_op| match flat_op {
                FlatOp::Op(inner) => match &inner.kind {
                    OpKind::JitMergePoint { reds_r, .. } => Some(reds_r),
                    _ => None,
                },
                _ => None,
            })
            .expect("portal emits a jit_merge_point");
        assert_eq!(merge.len(), 2);
        assert_eq!(merge, &split_reds);
        assert!(
            merge
                .iter()
                .all(|var| FunctionGraph::concretetype_of(var) == ConcreteType::GcRef)
        );
        assert_ne!(merge[0].id(), merge[1].id());
        let jd0 = &call_control.jitdrivers_sd()[0];
        assert!(!jd0.autoreds);
        assert_eq!(jd0.numreds, Some(2));
    }

    #[test]
    fn splits_an_autoreds_portal_after_autodetect() {
        use crate::codewriter::type_state::ConcreteType;
        use crate::flowspace::model::Variable;
        use crate::model::Link;

        let mut call_control = call::CallControl::new();
        let portal = CallPath::from_segments(["fixture", "unpackiterable"]);
        let mut graph = FunctionGraph::new("unpackiterable");
        let receiver = Variable::named("driver");
        let green = Variable::named("greenkey");
        let w_iterator = Variable::named("w_iterator");
        let items = Variable::named("items");
        for variable in [&receiver, &green, &w_iterator, &items] {
            FunctionGraph::set_concretetype_of_inline(variable, ConcreteType::GcRef);
        }
        graph.block_mut(graph.startblock).inputargs =
            vec![green.clone(), w_iterator.clone(), items.clone()];
        // RPython's marker receiver is a JitDriver Constant. Rematerialize it
        // after the split the way `unsimplify.split_block` does for a
        // const-produced driver singleton.
        graph
            .block_mut(graph.startblock)
            .operations
            .push(SpaceOperation {
                result: Some(receiver.clone()),
                kind: OpKind::ConstRefNull,
            });
        graph
            .block_mut(graph.startblock)
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::method(
                        "jit_merge_point",
                        Some("UnpackIterableJitDriver".into()),
                    ),
                    args: crate::model::call_args(vec![receiver, green.clone()]),
                    result_ty: ValueType::Void,
                },
            });
        graph.block_mut(graph.startblock).exits = vec![Link::from_variables(
            &graph,
            vec![green, w_iterator, items],
            graph.startblock,
            None,
        )];
        call_control.register_function_graph(portal.clone(), graph);

        let mut spec = driver(portal);
        spec.greens = vec!["greenkey".into()];
        spec.autoreds = true;
        spec.split_portal = true;
        register_configured_jitdrivers(
            &mut call_control,
            &[spec],
            &GraphTransformConfig::default().jitdriver_receiver_roots,
        );
        let split = CallPath::from_segments(["fixture", "unpackiterable_portal"]);
        assert_eq!(call_control.jitdrivers_sd()[0].portal_graph, split);
        let split_graph = call_control
            .function_graphs()
            .get(&split)
            .expect("autoreds split registers the cut portal");
        assert!(split_graph.func.dont_inline);
        assert!(
            crate::codewriter::support::find_jit_merge_point(
                &split_graph,
                &GraphTransformConfig::default().jitdriver_receiver_roots,
            )
            .is_some()
        );
    }
}
