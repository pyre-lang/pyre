//! MIR-driven flowspace driver.
//!
//! This module consumes Charon's ULLBC (a basic-block CFG derived from
//! rustc MIR) and produces the same [`FunctionGraph`] shape the rest of
//! the codewriter pipeline consumes.
//!
//! It is structurally simpler than a recursive-walk driver because the
//! input is already in CFG form: a driver that reconstructs a CFG from a
//! recursive AST walk needs to reconstruct join points, lazily install
//! per-block locals, thread Variables between blocks, and track per-scope
//! bindings. None of that is needed here, because every join point is
//! already an explicit MIR basic block with explicit predecessor edges.
//!
//! ## Reference
//!
//! `rpython/flowspace/flowcontext.py build_flow`
//! ([`FlowContext.build_flow`], [`FlowContext.record_block`],
//! [`FlowContext.mergeblock`]).
//!
//! The RPython reference iterates Python *bytecode positions* and uses
//! `mergeblock` to discover join points lazily. MIR's CFG already has
//! explicit predecessor edges and explicit block boundaries, so the
//! mergeblock dance collapses to a no-op: every join point is already
//! a single MIR basic block with N predecessors.
//!
//! ## Scope — production coverage
//!
//! The driver lowers the entire 4-function corpus end-to-end (see
//! `tests/test_mir_frontend.rs`) and achieves ≥ 99.9% coverage on the
//! real `pyre-interpreter.ullbc` (5434 / 5435 functions) and
//! `pyre-object.ullbc` (1717 / 1718) snapshots, gated by the stress
//! test in `tests/test_mir_stress.rs`. Surfaces handled:
//!
//! ### Statements
//!   - `Assign(Local, Rvalue)` — primary lowering site.
//!   - `Assign(Projection(.., Field|Deref|Index), ...)` — emits a
//!     side-effectful `FieldWrite` / `ArrayWrite` / `__deref_write`.
//!   - `StorageLive` / `PlaceMention` — skipped. `StorageDead` of a
//!     window local emits `objectmodel.keepalive_until_here`
//!     (`rtype_keepalive_until_here`).
//!   - `Assert` — stripped (overflow asserts collapse into success
//!     edge; see the `TermKind::Assert` lowering note below).
//!
//! ### Rvalues
//!   - `Use(operand)` — same-Variable alias.
//!   - `BinaryOp` — `OpKind::BinOp` with a canonical snake_case label
//!     (`add`, `eq`, `and`, …) so the assembler reaches the wired
//!     `int_*` / `ptr_*` keys without inventing PascalCase shapes.
//!     Integer ops with no pointer form (`lt`/`le`/`gt`/`ge`,
//!     `mod`/`floordiv`/`div`) over a Ref operand first emit
//!     `simple_call(lltype.cast_ptr_to_int)` — there is no `ptr_lt`.
//!   - `UnaryOp` — `OpKind::UnaryOp` with a canonical label
//!     (`neg`, `invert`, `cast_int_to_float`, …) per `binop_label` /
//!     `unary_op_label`.
//!   - `Ref` / `RawPtr` — same-Variable alias (JIT does not model
//!     lifetimes). A call that passes that alias into a raw
//!     scalar-pointer parameter materializes an address first
//!     (`RawMalloc` / `RawStore`): `history.py` `getkind` of a raw
//!     `Ptr` is the address, not the pointee word. Aliases of one
//!     place share that address. A mutable parameter writes the word
//!     back into the place, including a field or element projection.
//!     The spill is freed at the call. A result that carries a pointer
//!     or a reference is not lowered: the result itself, a payload, a
//!     field, or an opaque body whose layout is not zero-sized. A generic
//!     argument counts when a field stores it. That address would outlive
//!     the free.
//!   - `Cast` — same-Variable alias.
//!   - `Discriminant(place)` — synthetic `FieldRead("__discriminant")`.
//!   - `Aggregate` — a named struct is its fields (`OptVirtualize.make_vstruct`):
//!     the constructor is a temporary marker so boxing fusion and the
//!     consumer rewrites still see the construct-on-stack spelling, then
//!     [`scalar_replace_named_struct_aggregates`] replaces field reads
//!     with the stored SSA values. A by-value copy, return, store, or phi
//!     use materialises one `New` + `setfield`s per use so later mutation
//!     of one copy is not visible through the others. Closure environments
//!     and field-less structs stay constructors: `rewrite_op_malloc` needs
//!     a non-empty `all_fielddescrs`. Transparent newtype wrappers stay a
//!     no-op alias. Enum variants, tuples, and arrays still emit
//!     `Call(SyntheticTransparentCtor)` for later rewrites.
//!   - `ShallowInitBox` — synthetic `Call(SyntheticTransparentCtor)`.
//!   - `Repeat` / `Len` / `NullaryOp` — synthetic `Call(__array_repeat
//!     / __len / __nullary_*)`. `new_array_clear` is produced later by
//!     `do_fixed_newlist_clear`, after the rtyper, so a `[i64; N]`
//!     repeat stays the `__array_repeat(fill, count)` call here.
//!
//! ### Terminators
//!   - `Return` → `returnblock`.
//!   - `UnwindResume` / `Abort` → `exceptblock`.
//!   - `Goto { target }` — direct edge.
//!   - `Switch { discr, targets }` — `ExitSwitch::Value` + per-arm
//!     `Link` with `ExitCase::Bool` / `ExitCase::Const`.
//!   - `Call` — Direct / Trait → `Call(FunctionPath)`; Dynamic vtable calls
//!     use the trait-family indirect pipeline, plain safe function pointers
//!     become `IndirectCall`, and the residue — `unsafe fn` pointers and
//!     `dyn` receivers the vtable arm could not resolve — uses synthetic
//!     `Call(__dyn_call)`.
//!   - `Drop` — pass-through `Goto` (JIT does not model destructor
//!     semantics).
//!   - `Assert` — strip and forward to the success target.
//!
//! ### Constants
//!   - `Scalar(Signed|Isize)` → `ConstInt`; `Scalar(Unsigned|Usize)` →
//!     `ConstUInt`.
//!   - `Bool` → `ConstBool`. `Float` → `ConstFloat`.
//!   - `Str` / `ByteStr` → synthetic `Call(__str_const)`.
//!   - `Char` → `ConstInt` of the code point then `builtin_unichr`
//!     (`__majit_unichr`), so a Rust `char` annotates as
//!     `SomeUnicodeCodePoint` while staying in the Int bank
//!     (`getkind` of UniChar).
//!   - `FnDef` → synthetic 0-arg `Call(FunctionPath)`.
//!   - `Opaque(reason)` / `VTableRef` — synthetic opaque-string Call.
//!   - `TraitConst` on a selected impl folds to that impl's `NamedConst`
//!     (`resolved_trait_const_op`). A `Clause` that names no impl stays
//!     on the opaque-string path in `decode_constant`.
//!
//! Anything not in the above set returns [`LowerError::Unsupported`]
//! with the precise shape that prompted the failure — the driver grows
//! by widening this surface, not by failing silently.

use majit_charon_reader::{
    Llbc,
    ullbc::{
        BasicBlock, CallClass, CallFunc, CallKind, CallPayload, FieldDecl, FunDecl, FunId,
        GlobalDecl, NameSeg, Operand, Place, PlaceKind, ProjectionElem, RegularCall, Rvalue,
        StmtKind, SwitchTargets, TermKind, TyRef, TypeDecl, TypeDeclKind, Unstructured,
        const_fn_def_regular_id,
    },
};

use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{
    Block, BlockId, CallTarget, ExitCase, ExitSwitch, FieldDescriptor, FrameState, FunctionGraph,
    InlineLeaf, Link, LinkArg, OpKind, SpaceOperation, ValueType, container_leaf_descriptor,
    nest_inline_leaf,
};
use serde_json::Value;

/// Opaque non-null value for a prebuilt JIT-driver `NamedConst`.
///
/// The driver instance is only the receiver constant of its marker methods;
/// rtyping erases that receiver to `Void`, and `jtransform` discards operand
/// zero while rewriting the marker.  It therefore has no runtime address to
/// preserve, but the front-end graph still needs one constant-producing
/// operation until that rewrite.  Non-null keeps it distinct from the null
/// reference representation on every backend.
const JITDRIVER_NAMEDCONST_SENTINEL_ADDR: i64 = 8;

/// Top-level entry — load `function_name` out of `llbc`, lower it,
/// return the constructed [`FunctionGraph`].
///
/// The lookup is the same `ends_with("::<name>")` rule the reader's
/// `local_fn` uses. Replace with a fully-qualified-path lookup once
/// the call graph plumbing makes it useful.
pub fn lower_function(llbc: &Llbc, function_name: &str) -> Result<FunctionGraph, LowerError> {
    lower_function_with_static_addrs(llbc, function_name, crate::HostStaticAddrs::default())
}

/// [`lower_function`] against a caller-supplied host table — the same pairing
/// [`lower_fun_decl`] and [`lower_fun_decl_with_static_addrs`] already have.
/// A caller that lowers a fallible-returning graph needs this one, because the
/// default names no error carrier.
pub fn lower_function_with_static_addrs(
    llbc: &Llbc,
    function_name: &str,
    static_addrs: crate::HostStaticAddrs<'_>,
) -> Result<FunctionGraph, LowerError> {
    let fd = llbc
        .local_fn(function_name)
        .ok_or_else(|| LowerError::FunctionNotFound(function_name.to_string()))?;
    let context = LowerContext::new(llbc);
    lower_fun_decl_with_static_addrs(&context, fd, static_addrs)
}

/// Merge functions and metadata from a slice of LLBCs into one
/// `SemanticProgram`.  When `pyre-jit-trace` parses pyre-object +
/// pyre-interpreter together, each crate's `.ullbc` is supplied so
/// cross-crate calls in the merged SemanticProgram resolve.  Per-LLBC
/// duplicates (a function defined in both, e.g. via dependency closure)
/// keep the first occurrence.
pub fn build_semantic_program_from_llbcs(
    llbcs: &[Llbc],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    build_semantic_program_from_llbcs_with_static_addrs(llbcs, crate::HostStaticAddrs::default())
}

pub fn build_semantic_program_from_llbcs_with_static_addrs(
    llbcs: &[Llbc],
    static_addrs: crate::HostStaticAddrs<'_>,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    let jitdriver_receiver_roots =
        crate::codewriter::jtransform::default_jitdriver_receiver_roots();
    build_semantic_program_from_llbcs_with_static_addrs_filtered(
        llbcs,
        static_addrs,
        &jitdriver_receiver_roots,
        None,
        None,
        true,
        no_tombstoned_leaves(),
    )
}

pub fn build_semantic_program_from_llbcs_with_static_addrs_and_module_paths<
    L: std::borrow::Borrow<Llbc>,
>(
    llbcs: &[L],
    static_addrs: crate::HostStaticAddrs<'_>,
    module_paths: &[&str],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    let jitdriver_receiver_roots =
        crate::codewriter::jtransform::default_jitdriver_receiver_roots();
    build_semantic_program_from_llbcs_with_static_addrs_module_paths_and_jitdriver_roots(
        llbcs,
        static_addrs,
        module_paths,
        &jitdriver_receiver_roots,
    )
}

pub(crate) fn build_semantic_program_from_llbcs_with_static_addrs_module_paths_and_jitdriver_roots<
    L: std::borrow::Borrow<Llbc>,
>(
    llbcs: &[L],
    static_addrs: crate::HostStaticAddrs<'_>,
    module_paths: &[&str],
    jitdriver_receiver_roots: &[String],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    let module_filter = normalize_module_filter(module_paths);
    build_semantic_program_from_llbcs_with_static_addrs_filtered(
        llbcs,
        static_addrs,
        jitdriver_receiver_roots,
        module_filter.as_ref(),
        None,
        true,
        no_tombstoned_leaves(),
    )
}

/// The single-artefact prelude of
/// [`build_semantic_program_from_llbcs_with_static_addrs_filtered`] for a
/// crate the production frontend loads on its own: the duplicate-leaf
/// tombstones to paint (this artefact's and `cross_tombstoned_leaves`),
/// its named-const folds merged into the invocation table, and its
/// eval-hook graph list. Returns the tombstones.
#[cfg(any(test, feature = "mir-frontend"))]
pub(crate) fn prelink_crate(
    llbc: &Llbc,
    cross_tombstoned_leaves: &std::collections::HashSet<String>,
) -> std::collections::HashSet<String> {
    let mut duplicate_leaf_facts = DuplicateLeafFacts::default();
    duplicate_leaf_facts.absorb(DuplicateLeafFacts::discover(llbc));
    let mut paint_tombstones = duplicate_leaf_facts.tombstoned_leaves();
    paint_tombstones.extend(cross_tombstoned_leaves.iter().cloned());
    merge_named_const_folds(llbc.crate_name(), harvest_named_const_folds(llbc));
    let published = llbc.eval_hook_graphs();
    let paths = if published.is_empty() {
        discover_eval_hook_graphs(llbc)
    } else {
        published
    };
    let mut eval_hook_graphs = Vec::new();
    for path in paths {
        if !eval_hook_graphs.contains(&path) {
            eval_hook_graphs.push(path);
        }
    }
    if !eval_hook_graphs.is_empty() && llbc.eval_hook_graphs() != eval_hook_graphs {
        llbc.set_eval_hook_graphs(eval_hook_graphs);
    }
    paint_tombstones
}

/// The `module_paths` entry point narrowed further to a set of leaf
/// function names.
///
/// A fixture that asserts on a handful of named graphs pays for lowering
/// every function the module filter admits, which for `pyre-interpreter`
/// is thousands of bodies. `function_names` keeps the same whole-program
/// metadata — the type and trait tables that decide how a body lowers are
/// still derived from the entire `llbc` — and only skips building the
/// bodies nothing will read. A name that matches in several modules
/// lowers in each; the allowlist is a leaf name, not a path.
///
/// Only the bodies are dropped, so a fixture that reaches for a graph it
/// forgot to list fails loudly on the missing lookup rather than
/// silently asserting over a smaller program.
pub fn build_semantic_program_from_llbcs_with_static_addrs_and_function_names<
    L: std::borrow::Borrow<Llbc>,
>(
    llbcs: &[L],
    static_addrs: crate::HostStaticAddrs<'_>,
    module_paths: &[&str],
    function_names: &[&str],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    let module_filter = normalize_module_filter(module_paths);
    let function_filter = normalize_function_filter(function_names);
    let jitdriver_receiver_roots =
        crate::codewriter::jtransform::default_jitdriver_receiver_roots();
    build_semantic_program_from_llbcs_with_static_addrs_filtered(
        llbcs,
        static_addrs,
        &jitdriver_receiver_roots,
        module_filter.as_ref(),
        function_filter.as_ref(),
        true,
        no_tombstoned_leaves(),
    )
}

pub(crate) fn semantic_function_dedup_key(f: &crate::front::semantic::SemanticFunction) -> String {
    let path = if f.module_path.is_empty() {
        f.name.clone()
    } else {
        format!("{}::{}", f.module_path, f.name)
    };
    let path = match f.self_ty_root.as_deref() {
        Some(owner) => format!("{path}@{owner}"),
        None => path,
    };
    // `CallControl.graphs_from` / `Bookkeeper.getdesc` keys on the
    // function object. A non-ADT `Self` (`cmp::impls::<Impl>::le` for
    // `char` vs `i64`) shares `{module_path}::{name}`; the impl's trait
    // ref and Self type is that identity across LLBC artifacts.
    match f.trait_impl_identity.as_deref() {
        Some(identity) => format!("{path}::<impl {identity}>"),
        None => path,
    }
}

/// Call-path segments for one concrete trait-impl method, keyed by the
/// impl's trait-ref + Self identity rather than the extraction-local
/// impl index (`Bookkeeper.getdesc`).
pub(crate) fn trait_impl_identity_path(module: &str, identity: &str, leaf: &str) -> Vec<String> {
    let mut segments: Vec<String> = if module.is_empty() {
        Vec::new()
    } else {
        module.split("::").map(str::to_string).collect()
    };
    segments.push(format!("<impl {identity}>"));
    segments.push(leaf.to_string());
    segments
}

/// Whether `segment` is the trait-impl identity segment
/// [`trait_impl_identity_path`] writes between the impl owner and the
/// method leaf (`"<impl {identity}>"`). Charon's FunDecl placeholder
/// `"<Impl>"` is a different spelling and does not match.
pub(crate) fn is_trait_impl_identity_segment(segment: &str) -> bool {
    segment.starts_with("<impl ") && segment.ends_with('>')
}

pub(crate) fn absorb_semantic_program(
    merged: &mut Option<crate::front::semantic::SemanticProgram>,
    prog: crate::front::semantic::SemanticProgram,
    seen_function_keys: &mut std::collections::HashSet<String>,
    seen_struct_names: &mut std::collections::HashSet<String>,
    seen_trait_names: &mut std::collections::HashSet<String>,
    dedup_key: &dyn Fn(&crate::front::semantic::SemanticFunction) -> String,
) {
    match merged {
        None => {
            for f in &prog.functions {
                seen_function_keys.insert(dedup_key(f));
            }
            for n in &prog.known_struct_names {
                seen_struct_names.insert(n.clone());
            }
            for n in &prog.known_trait_names {
                seen_trait_names.insert(n.clone());
            }
            *merged = Some(prog);
        }
        Some(acc) => {
            for f in prog.functions {
                if seen_function_keys.insert(dedup_key(&f)) {
                    acc.functions.push(f);
                }
            }
            for n in prog.known_struct_names {
                if seen_struct_names.insert(n.clone()) {
                    acc.known_struct_names.insert(n);
                }
            }
            for n in prog.known_trait_names {
                if seen_trait_names.insert(n.clone()) {
                    acc.known_trait_names.insert(n);
                }
            }
            for (key, fields) in prog.struct_fields.fields {
                acc.struct_fields.fields.entry(key).or_insert(fields);
            }
            acc.struct_fields
                .raw_word_owners
                .extend(prog.struct_fields.raw_word_owners);
            for root in prog.struct_fields.foreign_body_roots {
                acc.struct_fields.note_foreign_body_crate_root(&root);
            }
            for (crate_root, stripped) in prog.struct_fields.foreign_callables {
                acc.struct_fields
                    .note_foreign_body_callable(&crate_root, &stripped);
            }
            for (enum_key, by_discr) in prog.enum_variant_by_discriminant {
                acc.enum_variant_by_discriminant
                    .entry(enum_key)
                    .or_insert(by_discr);
            }
            for (leaf, module) in prog.struct_origins {
                acc.struct_origins.entry(leaf).or_insert(module);
            }
            for (key, rows) in prog.struct_field_attrs {
                acc.struct_field_attrs.entry(key).or_insert(rows);
            }
            for (key, layout) in prog.exact_layouts {
                acc.exact_layouts.insert(key, layout);
            }
            for (key, id) in prog.struct_ids {
                acc.struct_ids
                    .entry(key)
                    .and_modify(|slot| {
                        if *slot != id {
                            *slot = None;
                        }
                    })
                    .or_insert(id);
            }
            acc.raw_alloc_flavor_structs
                .extend(prog.raw_alloc_flavor_structs);
            acc.atomic_load_decls.extend(prog.atomic_load_decls);
        }
    }
}

fn build_semantic_program_from_llbcs_with_static_addrs_filtered<L: std::borrow::Borrow<Llbc>>(
    llbcs: &[L],
    static_addrs: crate::HostStaticAddrs<'_>,
    jitdriver_receiver_roots: &[String],
    module_filter: Option<&std::collections::HashSet<String>>,
    function_filter: Option<&std::collections::HashSet<String>>,
    link_scalars: bool,
    cross_tombstoned_leaves: &std::collections::HashSet<String>,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    let llbc_refs: Vec<&Llbc> = llbcs.iter().map(std::borrow::Borrow::borrow).collect();
    let llbcs: &[&Llbc] = &llbc_refs;
    if link_scalars {
        link_transparent_scalar_types(llbcs);
    }
    // One duplicate-leaf verdict for the whole input, computed before any
    // body is painted. A leaf unique inside each artefact (the two
    // `FrameBlock`s) is still withdrawn once the artefacts are merged, and
    // the per-file lowering has to paint that qualified root up front.
    let mut duplicate_leaf_facts = DuplicateLeafFacts::default();
    for llbc in llbcs {
        duplicate_leaf_facts.absorb(DuplicateLeafFacts::discover(llbc));
    }
    let mut paint_tombstones = duplicate_leaf_facts.tombstoned_leaves();
    paint_tombstones.extend(cross_tombstoned_leaves.iter().cloned());
    // Defining-crate NamedConst values have to be harvested while that
    // crate's artefact is loaded: a foreign `const` is `Opaque` in the
    // caller's LLBC.  The production frontend reloads one crate at a
    // time (`build_semantic_program_from_prelinked_llbc`), so a
    // single-artefact call MERGES into the invocation table and later
    // crates see earlier ones — production load order puts
    // `pyre-object` before `pyre-interpreter`.  A multi-artefact call
    // has every tree live and replaces the table for the duration.
    let _named_const_folds = if llbcs.len() > 1 {
        let mut folds = std::collections::HashMap::new();
        for llbc in llbcs {
            folds.extend(harvest_named_const_folds(llbc));
        }
        Some(push_named_const_folds(folds))
    } else {
        for llbc in llbcs {
            merge_named_const_folds(llbc.crate_name(), harvest_named_const_folds(llbc));
        }
        None
    };
    // Union of each artefact's published list, or the family discovered
    // in that artefact when nothing has been published yet. The first
    // nonempty list is not the whole set: a later crate can publish or
    // discover paths the earlier one never saw.
    let mut eval_hook_graphs = Vec::new();
    for llbc in llbcs {
        let published = llbc.eval_hook_graphs();
        let paths = if published.is_empty() {
            // A leaf-name filter keeps this fixture from lowering the
            // eval-hook family; discovering it walks every body in the
            // crate. The unfiltered production path still harvests.
            if function_filter.is_some() {
                Vec::new()
            } else {
                discover_eval_hook_graphs(llbc)
            }
        } else {
            published
        };
        for path in paths {
            if !eval_hook_graphs.contains(&path) {
                eval_hook_graphs.push(path);
            }
        }
    }
    if !eval_hook_graphs.is_empty() {
        for llbc in llbcs {
            if llbc.eval_hook_graphs() != eval_hook_graphs {
                llbc.set_eval_hook_graphs(eval_hook_graphs.clone());
            }
        }
    }
    let mut merged: Option<crate::front::semantic::SemanticProgram> = None;
    // Dedup key combines `self_ty_root` (the impl owner, when known),
    // `trait_impl_id` (Charon's function-object identity for a trait
    // impl), `module_path`, and `name`.  Without `self_ty_root`, two
    // distinct impl methods would collide on a shared
    // `{module_path}::{name}`: both `impl FrameDebugData { fn new(...) }`
    // and `impl PyFrame { fn new(...) }` land under
    // `module_path = "pyframe::<Impl>"` (the Impl NameSeg renders as
    // `<Impl>`), so without the owner in the key the second would be
    // silently dropped.  Without `trait_impl_id`, every
    // `cmp::impls::<Impl>::le` width collapses the same way.  Falls
    // back to the bare `{module_path}::{name}` shape (or just `name`)
    // for entries that have neither.
    let mut seen_function_keys = std::collections::HashSet::new();
    let mut seen_struct_names = std::collections::HashSet::new();
    let mut seen_trait_names = std::collections::HashSet::new();
    let dedup_key = semantic_function_dedup_key;
    for llbc in llbcs {
        let prog = build_semantic_program_from_llbc_with_static_addrs_filtered(
            llbc,
            static_addrs,
            jitdriver_receiver_roots,
            module_filter,
            function_filter,
            &paint_tombstones,
        )?;
        absorb_semantic_program(
            &mut merged,
            prog,
            &mut seen_function_keys,
            &mut seen_struct_names,
            &mut seen_trait_names,
            &dedup_key,
        );
    }
    // The per-file builder hardened each program individually, but the
    // `or_insert` merges above can re-introduce a bare-leaf alias that
    // is unique within one crate yet collides across crates (e.g. the
    // pyre-interpreter and pyre-jit `FrameBlock`s).  Re-derive the
    // verdict from the merged qualified keys.  (The exact-layout channel
    // is keyed by `StructId` and so is collision-free by construction —
    // only the still-string-keyed metadata channels need hardening.)
    if let Some(acc) = &mut merged {
        harden_duplicate_leaf_metadata(
            &mut acc.struct_fields,
            &mut acc.struct_origins,
            &mut acc.enum_variant_by_discriminant,
            Some(&acc.struct_ids),
        );
    }
    Ok(
        merged.unwrap_or_else(|| crate::front::semantic::SemanticProgram {
            functions: Vec::new(),
            harvested_hints: std::collections::HashMap::new(),
            known_struct_names: std::collections::HashSet::new(),
            known_trait_names: std::collections::HashSet::new(),
            struct_fields: crate::front::semantic::StructFieldRegistry::default(),
            immutable_fields: std::collections::HashMap::new(),
            raw_alloc_flavor_structs: std::collections::HashSet::new(),
            enum_variant_by_discriminant: std::collections::HashMap::new(),
            struct_origins: std::collections::HashMap::new(),
            struct_field_attrs: std::collections::HashMap::new(),
            exact_layouts: std::collections::HashMap::new(),
            struct_ids: std::collections::HashMap::new(),
            unsafe_fn_stubs: Vec::new(),
            foreign_opaque_method_externals: Vec::new(),
            atomic_load_decls: Vec::new(),
        }),
    )
}

pub(crate) fn normalize_module_filter(
    module_paths: &[&str],
) -> Option<std::collections::HashSet<String>> {
    let modules: std::collections::HashSet<String> = module_paths
        .iter()
        .copied()
        .filter(|module_path| !module_path.is_empty())
        .map(str::to_string)
        .collect();
    (!modules.is_empty()).then_some(modules)
}

fn normalize_function_filter(function_names: &[&str]) -> Option<std::collections::HashSet<String>> {
    let names: std::collections::HashSet<String> = function_names
        .iter()
        .copied()
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect();
    (!names.is_empty()).then_some(names)
}

/// Build a [`SemanticProgram`] by lowering every local function
/// declaration in `llbc`.  This is the production pipeline's
/// program-build entry point (`lib.rs`).
///
/// **Whole-program metadata** (`known_struct_names`,
/// `known_trait_names`, `struct_fields`) is populated from
/// `type_decls` / `trait_decls`; struct field-type strings are resolved
/// by [`tyref_to_ast_string`] from Charon's type IR.  `immutable_fields`
/// is left empty here and filled by
/// [`crate::front::llbc_hints::harvest_immutable_fields_from_llbcs`],
/// which reads the `_immutable_fields_<Struct>` marker consts
/// `#[majit_macros::jit_immutable_fields]` leaves in `global_decls` —
/// the attribute itself is consumed at expansion time and never
/// reaches Charon's `attr_info`.
///
/// Functions Charon could not extract (opaque body / `null` entry) or
/// global-initializer bodies are skipped silently — they are not JIT
/// call targets.  A function whose MIR shape the driver cannot yet lower
/// produces a [`LowerError`] that is captured per-function: whether it is
/// a recognised, tracked gap (an uninitialised-local read that survives
/// even the reverse-postorder re-lower) or any other unrecognised failure,
/// body is omitted and callers can take the residual path. This is
/// PRE-EXISTING-ADAPTATION, not exceptiontransform.py parity:
/// `ExceptionTransformer.transform_completely` transforms every graph
/// without an exception-to-residual catch. A residual is executable only
/// with a registered target and a compatible ABI; dropping a graph alone
/// does not establish safety. #346 must close the lowering gaps and remove
/// this per-function fallback. The coverage reporting below currently
/// reports the gap and proceeds rather than enforcing that terminal gate.
fn is_known_lowering_gap(msg: &str) -> bool {
    // The forward-reference shape: a body reads a MIR local on a path the
    // driver has not yet bound (`read of MIR local N before any Assign`).
    // `lower_fun_decl` first hits this under MIR-index order, then retries
    // the body in reverse-postorder — which orders the defining block
    // before the reading block and resolves every such read in the
    // current snapshot.  This predicate triggers that RPO retry, and
    // (defensively) if even RPO cannot bind the read it classifies the
    // residual failure as a tracked degradation (the function becomes a
    // residual call) rather than a build-failing regression.
    //
    // Loop-carried definitions (a local defined on a back-edge and read
    // at the loop header) are NOT a residual case: every block's live-in
    // locals already receive a block inputarg in `Lowering::new`, and
    // `edge_args` threads each predecessor's binding — including the
    // back-edge — into it, so the header read resolves to the inputarg
    // regardless of processing order.  The only requirement is that
    // `compute_mir_liveness` mark every local the lowering reads; see
    // `mark_projection_index_offset_use` for the `place[idx]` and
    // `Offset.bytes` cases.
    if msg.contains("uninitialised local") {
        return true;
    }
    // A Result-of-PyError wrapper whose body is a pure tail-forward of a
    // callee whose `?`-site the caller rule did not record
    // (`let step = executor.method()?; Ok(step)` where `method`'s return is
    // not `Result<T, PyError>`).  The forward collapses to a direct
    // returnblock link with no `Ok`/`Err` shell, so the callee rule finds
    // nothing to rewrite and the caller rule never saw a scoped call —
    // `result_exc::lower_result_exc_returns` reports "no rewritable
    // returns".  The exception-link lowering cannot model this shape, so
    // the wrapper degrades to a residual call (the trivial forward runs
    // unchanged at the interpreter level); the inner method still JITs
    // through its own scoped callees.
    msg.contains("no rewritable returns")
}

/// A discovered reference-payload generic-enum instantiation: the enum's
/// Charon def id and qualified name, the concrete type arguments in
/// generic-parameter order (`["Tuple", "PyError"]`), and the `<…>` suffix
/// the per-instantiation classdef carries (`<Tuple,PyError>`).
#[derive(Clone, PartialEq, Eq, Hash)]
struct RefEnumInst {
    def_id: u64,
    name_path: String,
    args: Vec<String>,
    suffix: String,
}

/// Physical layout identity for a concrete ADT use. The defining path supplies
/// the template identity; all rendered type arguments supply the
/// monomorphization identity. This is deliberately broader than
/// `adt_head_instantiation_suffix`, whose reference-payload predicate controls
/// annotator class splitting rather than Rust memory layout.
fn concrete_adt_struct_id(
    template: majit_ir::descr::StructId,
    adt: Option<&serde_json::Map<String, serde_json::Value>>,
    llbc: &Llbc,
) -> majit_ir::descr::StructId {
    let Some(adt) = adt else {
        return template;
    };
    let args = render_adt_type_args(adt, llbc, 0);
    if args.is_empty() {
        template
    } else {
        template.instantiate(&format!("<{}>", args.join(",")))
    }
}

/// The [`RefEnumInst`] for the ADT descriptor `adt` (`{"id": …,
/// "generics": …}`) when it names a split-eligible reference-payload
/// generic enum, else `None`.  Shared by both instantiation scans below so
/// a constructor head and a type-node receiver seed the identical set and
/// spelling.
fn ref_enum_instantiation_of_adt(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> Option<RefEnumInst> {
    let suffix = adt_head_instantiation_suffix(adt, llbc)?;
    let def_id = type_decl_ref_adt_id(adt)?;
    let name_path = llbc.type_by_id(def_id)?.item_meta.name_path();
    let node = {
        let mut m = serde_json::Map::new();
        m.insert("Adt".into(), serde_json::Value::Object(adt.clone()));
        serde_json::Value::Object(m)
    };
    // A pointer-niche Option is the null word (`tyref_is_niche_option`,
    // including Option of a thin Arc/Rc/Box and OwnerRootGuard). Intern of
    // a tagged variant class would mint `__pos_0` on a ClassDef the
    // FieldRead identity fold never reads; CAST then getattr a missing attr.
    if lltype::tyref_is_niche_option(&TyRef::Other(node), llbc) {
        return None;
    }
    // Payload rows, not the `<…>` class suffix. The suffix stays the
    // rendered argument (`PyError`); a handle's row is `*mut` of its
    // `Deref::Target` class so the payload attr and the cast agree. A
    // pointer-niche Option payload is `*mut` of the pointee: CAST
    // strips `&`, and intern of `Option<&T>` as `Option<T>` would
    // getattr a tagged class over a pointer word. Type-arg nodes from
    // call generics are hash-consed (`Value` / `Deduplicated`); intern
    // ROW peels those wrappers so the niche test sees the Option ADT.
    let args = render_adt_payload_type_args(
        adt,
        llbc,
        0,
        gc_struct_ids,
        &std::collections::HashSet::<String>::new(),
    );
    Some(RefEnumInst {
        def_id,
        name_path,
        args,
        suffix,
    })
}

/// Collect every reference-payload generic-enum instantiation in the
/// program as `(name_path, "<…>")` pairs (`Result`, `<Tuple>`).  Two
/// sources, both filtered through [`adt_head_instantiation_suffix`] so the
/// discovered set agrees with what the constructor / field-read / receiver
/// sites project (a non-enum `Vec<T>` / `Box<T>` yields `None`; only the
/// intended `Option`/`Result`-family enums seed).
///
/// A module or leaf-name filter walks only the declarations that filter
/// admits: those are the bodies whose constructors and locals can mention
/// an instantiation this program will lower. No filter still walks every
/// local function, including global initialisers.
///   - CONSTRUCTOR heads — a `Result::Ok` / `Option::Some` payload
///     `setattr("__pos_0")` carries its concrete generics inline on the
///     `Rvalue::Aggregate(AggregateKind::Adt, …)` head.
///   - TYPE-NODE receivers — a function that only READS an instantiation
///     (a `__discriminant` match on a `Result<Tuple>` parameter, never a
///     constructor) carries it solely in a local's declared type.  Seeding
///     these pre-mints the variant subclasses before
///     `assign_inheritance_ids`; otherwise the discriminant narrowing mints
///     them lazily afterwards, unnumbered (per-graph Skip→legacy walker).
///   - CONCRETE CALLEE SIGNATURES — a `RegularCall` `Fun { id }` whose
///     generic arguments are already concrete
///     (`clause_spec::concrete_type_args`) specialises the callee
///     signature (`substituted_signature`). The caller's locals need not
///     name the enum: `Option::map`'s first input is `Option<T>`, and the
///     substituted `T` intern the instantiation the spec copy will read.
fn collect_ref_enum_instantiations(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    module_filter: Option<&std::collections::HashSet<String>>,
    function_filter: Option<&std::collections::HashSet<String>>,
) -> Vec<RefEnumInst> {
    let mut found: std::collections::HashSet<RefEnumInst> = std::collections::HashSet::new();
    let mut queue: Vec<(u64, Vec<Value>, Vec<Value>, Vec<Value>)> = Vec::new();
    let mut seen: std::collections::HashSet<(u64, String)> = std::collections::HashSet::new();
    let filtered = module_filter.is_some() || function_filter.is_some();
    // TypeDecl field types (`Vec<Option<T>>` on a struct) are not in any
    // Unstructured body. Walk them when the harvest is whole-program so a
    // payload that only lives in a field still intern's its variant rows.
    if !filtered {
        for td in llbc.iter_type_decls() {
            intern_enums_from_type_decl(llbc, gc_struct_ids, td, &mut found);
        }
    }
    for fd in llbc.iter_local_fns() {
        if filtered {
            if fd.is_global_initializer().is_some() {
                continue;
            }
            let (module_path, name) = decl_module_path_and_name(fd);
            if !should_lower_module(module_filter, &module_path)
                || !should_lower_function(function_filter, &name)
            {
                continue;
            }
        }
        intern_enums_from_signature(llbc, gc_struct_ids, fd, &mut found);
        let Some(u) = fd.unstructured() else {
            continue;
        };
        intern_enums_from_unstructured(llbc, gc_struct_ids, &u, &mut found);
        intern_enqueue_generic_calls(llbc, &u, &mut queue, &mut seen);
    }
    // Spec copies are not in original LLBC (`specialize.py`
    // `FunctionDesc.cachedgraph`). Walk each concrete generic callee
    // through its substituted signature and body so `Option<T>` locals
    // inside `map` / nested RegularCalls intern before numbering.
    while let Some((id, types, const_generics, trait_refs)) = queue.pop() {
        let Some(callee) = llbc.fn_by_id(id) else {
            continue;
        };
        let sig = crate::front::clause_spec::substituted_signature(
            &callee.signature,
            llbc,
            &types,
            &const_generics,
        );
        for ty in sig.inputs.iter().chain(std::iter::once(&sig.output)) {
            intern_enum_from_ty(llbc, gc_struct_ids, ty, &mut found);
        }
        if let Some(u) = crate::front::clause_spec::substituted_unstructured(
            callee,
            llbc,
            &trait_refs,
            &types,
            &const_generics,
        ) {
            intern_enums_from_unstructured(llbc, gc_struct_ids, &u, &mut found);
            intern_enqueue_generic_calls(llbc, &u, &mut queue, &mut seen);
        }
    }
    found.into_iter().collect()
}

fn intern_enum_from_ty(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    ty: &TyRef,
    found: &mut std::collections::HashSet<RefEnumInst>,
) {
    intern_enum_from_ty_at(llbc, gc_struct_ids, ty, found, 0);
}

fn intern_enum_from_ty_at(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    ty: &TyRef,
    found: &mut std::collections::HashSet<RefEnumInst>,
    depth: usize,
) {
    if depth > 12 {
        return;
    }
    let Some(adt) = tyref_adt_map(ty, llbc) else {
        return;
    };
    if let Some(pair) = ref_enum_instantiation_of_adt(adt, llbc, gc_struct_ids) {
        found.insert(pair);
    }
    // Nested payloads (`Vec<Option<T>>`, `HashMap<K, Option<T>>`) are
    // not the head ADT. Walk the type arguments so a tagged Option that
    // only appears inside another constructor still intern's.
    let Some(types) = type_decl_ref_generics(adt, llbc)
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
    else {
        return;
    };
    for t in types {
        intern_enum_from_ty_at(
            llbc,
            gc_struct_ids,
            &TyRef::Other(t.clone()),
            found,
            depth + 1,
        );
    }
}

fn intern_enums_from_signature(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    fd: &FunDecl,
    found: &mut std::collections::HashSet<RefEnumInst>,
) {
    for ty in fd
        .signature
        .inputs
        .iter()
        .chain(std::iter::once(&fd.signature.output))
    {
        intern_enum_from_ty(llbc, gc_struct_ids, ty, found);
    }
}

fn intern_enums_from_type_decl(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    td: &TypeDecl,
    found: &mut std::collections::HashSet<RefEnumInst>,
) {
    match &td.kind {
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => {
            for f in fields {
                intern_enum_from_ty(llbc, gc_struct_ids, &f.ty, found);
            }
        }
        TypeDeclKind::Enum(variants) => {
            for v in variants {
                for f in &v.fields {
                    intern_enum_from_ty(llbc, gc_struct_ids, &f.ty, found);
                }
            }
        }
        TypeDeclKind::Alias(body) => {
            intern_enum_from_ty(llbc, gc_struct_ids, &TyRef::Other(body.clone()), found);
        }
        TypeDeclKind::Opaque | TypeDeclKind::Unknown => {}
    }
}

fn intern_enums_from_unstructured(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    u: &Unstructured,
    found: &mut std::collections::HashSet<RefEnumInst>,
) {
    // Constructor heads that exist only on an `on_unwind` cleanup chain
    // are not part of the lowered body. The local-type scan below still
    // sees an instantiation a normal block only reads.
    let forward_blocks = forward_reachable_mask(llbc, u);
    for (bb_idx, bb) in u.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) {
            intern_enum_from_ty(llbc, gc_struct_ids, &call.dest.ty, found);
            for arg in &call.args {
                match arg {
                    Operand::Copy(place) | Operand::Move(place) => {
                        intern_enum_from_ty(llbc, gc_struct_ids, &place.ty, found);
                    }
                    Operand::Const(_) => {}
                }
            }
        }
        for st in &bb.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = st.stmt_kind_ref() else {
                continue;
            };
            intern_enum_from_ty(llbc, gc_struct_ids, &place.ty, found);
            match rvalue {
                Rvalue::Aggregate(kind, _) => {
                    if let Some(head) = kind
                        .get("Adt")
                        .and_then(serde_json::Value::as_array)
                        .and_then(|adt| adt.first())
                        .and_then(serde_json::Value::as_object)
                        && let Some(pair) = ref_enum_instantiation_of_adt(head, llbc, gc_struct_ids)
                    {
                        found.insert(pair);
                    }
                }
                Rvalue::Discriminant(src) => {
                    intern_enum_from_ty(llbc, gc_struct_ids, &src.ty, found);
                }
                _ => {}
            }
        }
    }
    for local in &u.locals.locals {
        intern_enum_from_ty(llbc, gc_struct_ids, &local.ty, found);
    }
}

fn intern_enqueue_generic_calls(
    llbc: &Llbc,
    u: &Unstructured,
    queue: &mut Vec<(u64, Vec<Value>, Vec<Value>, Vec<Value>)>,
    seen: &mut std::collections::HashSet<(u64, String)>,
) {
    let forward_blocks = forward_reachable_mask(llbc, u);
    for (bb_idx, bb) in u.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        intern_enqueue_regular_call(llbc, reg, queue, seen);
    }
}

fn intern_enqueue_regular_call(
    llbc: &Llbc,
    reg: &RegularCall,
    queue: &mut Vec<(u64, Vec<Value>, Vec<Value>, Vec<Value>)>,
    seen: &mut std::collections::HashSet<(u64, String)>,
) {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return;
    };
    let Some(callee) = llbc.fn_by_id(*id) else {
        return;
    };
    if !crate::front::clause_spec::decl_is_generic(callee) {
        return;
    }
    // A later type arg may still name a TypeVar (closure `F` trait-ref
    // JSON). `concrete_type_args` would drop the whole call; intern still
    // needs `Option<T>` from `types[0]` (`specialize.py`
    // `FunctionDesc.cachedgraph`). A leftover TypeVar suffix is skipped
    // by `intern_enum_from_ty`.
    let (types, const_generics) = crate::front::clause_spec::generic_args_lists(&reg.generics);
    if types.is_empty() && const_generics.is_empty() {
        return;
    }
    let key = (*id, serde_json::to_string(&types).unwrap_or_default());
    if !seen.insert(key) {
        return;
    }
    let trait_refs = intern_call_trait_refs(&reg.generics);
    queue.push((*id, types, const_generics, trait_refs));
}

fn intern_call_trait_refs(generics: &Value) -> Vec<Value> {
    generics
        .get("trait_refs")
        .or_else(|| {
            generics
                .get("skip_binder")
                .and_then(|binder| binder.get("trait_refs"))
        })
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Substitute the instantiation's concrete type args into a variant
/// field's type: a bound type variable resolves to `args[idx]`; any other
/// shape (a concrete field, or a `??…` placeholder for an unresolved node)
/// renders as-is via [`tyref_to_ast_string`].  Only a bare type variable
/// is substituted — a `Box<T>` / `Vec<T>` field keeps its `??TypeVar`
/// inner rendering, which the caller treats as unresolved and skips.
fn substitute_field_type(
    ty: &TyRef,
    args: &[String],
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> String {
    if let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc))
        && let Some(idx) = typevar_bound_index(node)
    {
        return args
            .get(idx as usize)
            .cloned()
            .unwrap_or_else(|| "??typevar_oob".to_string());
    }
    // A concrete payload that is itself the handle (a monomorphized
    // `Result::Err` field) has to name the same class the cast writes.
    // A type variable already carries that spelling in `args`.
    if let Some(spelling) = transparent_handle_field_spelling(
        ty,
        llbc,
        no_tombstoned_leaves(),
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) {
        return spelling;
    }
    tyref_to_ast_string(ty, llbc)
}

/// Whether a rendered payload type is an unboxed scalar rather than a
/// one-GC-word heap reference.  The textual descr path resolves a scalar's
/// bank/width from the row type string via `type_flag_from_str`
/// (`assembler.rs`) / `get_type_flag` (`call.rs`), so a scalar row is
/// layout-safe *as a variant's sole field* (byte offset 0).  In a
/// multi-field variant, mixed scalar widths could reorder under
/// `repr(Rust)` away from the textual offset accumulator, so a primitive
/// field there stays fail-closed (offset ambiguity).
fn is_primitive_payload_type(s: &str) -> bool {
    matches!(
        s.trim(),
        "i8" | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "f32"
            | "f64"
            | "bool"
            | "char"
            | "()"
            | ""
    )
}

/// Whether a rendered payload remains an inline Rust array/slice for which
/// the translated field representation has not been established here.  A
/// tuple is deliberately excluded: RPython gives every non-empty tuple a
/// `Ptr(GcStruct(...))` representation, and pyre lowers tuple construction to
/// that same GC-managed object shape.
fn is_inline_array_or_slice_type(s: &str) -> bool {
    s.trim().starts_with('[')
}

/// Register per-instantiation SUFFIXED variant payload rows for the
/// discovered reference-payload enum instantiations (#312 C4).
///
/// The unsuffixed template rows (`Result::Ok`, [`derive_program_metadata`])
/// carry the bare type variable, which `tyref_to_ast_string` renders as
/// `??TypeVar` → `project_struct_field_type` → `Impossible`.  So a
/// narrowing-only receiver (`Result<Tuple>` matched + read, never
/// constructed in-graph) finds no concrete payload: its suffixed variant
/// classdef `__pos_0` stays `Impossible` (annotator bottom) and the graph
/// routes Skip → legacy walker.  A constructor instead flows the concrete
/// payload onto the classdef via `setattr`, so only the read-only case
/// gaps.
///
/// Substituting the instantiation's concrete type args into each variant
/// field's type-var position recovers the concrete row, keyed under the
/// same suffixed spellings (`{enum}{suffix}::{variant}`) that
/// `getuniqueclassdef_for_enum_variant` projects through.  A tuple payload is
/// a GC reference in the translated program, even though Rust stores it
/// inline: RPython's `TupleRepr` uses
/// `Ptr(GcStruct('tupleN', ...))` (`rpython/rtyper/rtuple.py`), and
/// pyre's synthetic aggregate constructor follows that representation.
/// Consequently its textual field descr must use the ordinary one-word
/// `Ref` fallback, not reject the row based on the source Rust width.  A
/// variant registers when every payload field resolves and a primitive
/// scalar in a multi-field source variant cannot make the descriptor offset
/// ambiguous.  An unresolved `??` placeholder, an array/slice aggregate whose
/// translated field representation is not established here, or a primitive
/// field in a multi-field variant (mixed scalar widths could reorder under
/// `repr(Rust)`) leaves the variant unregistered (fail-closed Skip).  The
/// qualified spelling always publishes; the bare-leaf / crate-
/// stripped spellings publish only while the leaf survived
/// `harden_duplicate_leaf_metadata`, so a duplicate enum leaf fails closed
/// instead of projecting whichever colliding decl was inserted first.  The
/// rows only ever fill an `Impossible`/force-shell attr, so a constructed
/// instantiation already carrying its concrete payload is left untouched
/// (monotonic).
fn register_ref_enum_instantiation_rows(
    llbc: &Llbc,
    insts: &[RefEnumInst],
    enum_variant_by_discriminant: &std::collections::HashMap<
        String,
        std::collections::HashMap<i64, String>,
    >,
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
    struct_ids: &mut std::collections::HashMap<String, Option<majit_ir::descr::StructId>>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) {
    for inst in insts {
        let Some(td) = llbc.type_by_id(inst.def_id) else {
            continue;
        };
        let TypeDeclKind::Enum(variants) = &td.kind else {
            continue;
        };
        let name = td.item_meta.name_path();
        let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
        let canon_base = strip_crate_prefix(&name);
        // The bare leaf survives in `enum_variant_by_discriminant` only when
        // `harden_duplicate_leaf_metadata` did not withdraw it on cross-decl
        // collision; gate the ambiguous bare spellings on that, same as the
        // discriminant-map mirror.
        let leaf_survived = enum_variant_by_discriminant.contains_key(&leaf);
        let mut bases: Vec<&str> = vec![name.as_str()];
        // FieldRead / getuniqueclassdef owner_root is crate-stripped
        // (`class_identity_spelling` / `strip_crate_prefix`):
        // `option::Option<CodePoint>::Some`, not `core::option::Option<…>`.
        // A withdrawn leaf still needs this key; the qualified intern
        // spelling is not the lookup lowering emits. The bare leaf stays
        // gated: it collides across modules sharing an enum name.
        if !canon_base.is_empty() && canon_base != name {
            bases.push(canon_base.as_str());
        }
        if leaf_survived {
            bases.push(leaf.as_str());
        }
        for v in variants {
            if v.fields.is_empty() {
                continue;
            }
            // A sole payload field sits at byte offset 0, so a resolved
            // primitive scalar is layout-safe there; a primitive in a
            // multi-field variant could reorder under `repr(Rust)`.
            let single_field = v.fields.len() == 1;
            let mut rows: Vec<crate::front::semantic::FieldRow> =
                Vec::with_capacity(v.fields.len());
            let mut registrable = true;
            for (i, f) in v.fields.iter().enumerate() {
                let fname = field_decl_name(i, f);
                let concrete = intern_payload_registry_spelling(
                    llbc,
                    &substitute_field_type(
                        &f.ty,
                        &inst.args,
                        llbc,
                        gc_struct_ids,
                        &std::collections::HashSet::<String>::new(),
                    ),
                    struct_fields,
                );
                let trimmed = concrete.trim();
                // A sole pair-slice payload is its pointer and length words,
                // under the same rule as a sole scalar.
                if spelling_pair_field_kind(trimmed).is_some() {
                    if !single_field {
                        registrable = false;
                        break;
                    }
                    let len = pair_len_field_name(&fname);
                    rows.push(field_decl_row(i, f, "usize"));
                    rows.push(crate::front::semantic::FieldRow::named(len, "usize"));
                    continue;
                }
                // A sole unresolved associated-type projection is one GC
                // word (`InstanceRepr`), the same intern Opaque payloads
                // take. Nested unresolved stays fail-closed.
                if lltype::type_arg_is_whole_unresolved_projection(trimmed) {
                    if !single_field {
                        registrable = false;
                        break;
                    }
                    intern_projection_payload_spelling(trimmed, struct_fields);
                    rows.push(field_decl_row(i, f, concrete));
                    continue;
                }
                // Register a field only when it is layout-safe for the
                // translated suffixed owner's textual descr: a GC-managed
                // value (including RPython tuples, represented by one
                // pointer), or (sole field, offset 0) a real unboxed scalar.
                // Fail closed on an unresolved `??` placeholder, an
                // array/slice aggregate, the unit / empty render, or a scalar
                // outside a single-field variant.
                let layout_safe = !concrete.contains("??")
                    && !is_inline_array_or_slice_type(trimmed)
                    && trimmed != "()"
                    && !trimmed.is_empty()
                    && (!is_primitive_payload_type(&concrete) || single_field);
                if !layout_safe {
                    registrable = false;
                    break;
                }
                rows.push(field_row_from_decl(i, f, concrete, llbc));
            }
            if !registrable {
                continue;
            }
            for row in &rows {
                intern_opaque_payload_spelling(llbc, &row.ty, struct_fields);
            }
            // Mirror the unsuffixed dual-publish so the suffixed variant path
            // the narrowing builds — `canonical_struct_name("{base}{suffix}")
            // ::{variant}` — hits regardless of which spelling the receiver
            // carries.  The qualified `name` is unambiguous and always
            // publishes; the bare-leaf / crate-stripped spellings collide
            // across modules sharing an enum leaf, so they publish only while
            // the bare leaf survived `harden_duplicate_leaf_metadata` (mirrors
            // the discriminant-map guard) — a withdrawn leaf fails closed
            // rather than projecting whichever colliding decl was inserted
            // first.
            let sid = majit_ir::descr::StructId::from_canonical_spelling(&format!(
                "{canon_base}{}::{}",
                inst.suffix, v.name
            ));
            for base in &bases {
                let key = format!("{base}{}::{}", inst.suffix, v.name);
                struct_fields
                    .fields
                    .entry(key.clone())
                    .or_insert_with(|| rows.clone());
                record_struct_id(struct_ids, key, sid);
            }
        }
        // Suffixed enum BASE (`{base}{suffix}`): copy the unsuffixed
        // `__discriminant` row so `getuniqueclassdef` of a tagged
        // instantiation (`Option<char>`) has a classdef (`rclass.py`
        // `getuniqueclassdef`). `lookup_fields` strips `<…>`; the
        // bookkeeper exact-key (`reg.fields.get`) consumes this spelling.
        // Copy the `__discriminant` row even when the unsuffixed base
        // carries extra rows. Requiring the exact `[__discriminant]`
        // sentinel dropped a tagged primitive instantiation
        // (`Option<char>`) whose template later gained a payload alias.
        let disc_rows = bases.iter().find_map(|base| {
            struct_fields.fields.get(*base).and_then(|rows| {
                rows.iter()
                    .find(|row| row.name == "__discriminant")
                    .cloned()
                    .map(|row| vec![row])
            })
        });
        if let Some(disc_rows) = disc_rows {
            for base in &bases {
                struct_fields
                    .fields
                    .entry(format!("{base}{}", inst.suffix))
                    .or_insert_with(|| disc_rows.clone());
            }
        }
        for arg in &inst.args {
            intern_opaque_payload_spelling(
                llbc,
                &intern_payload_registry_spelling(llbc, arg, struct_fields),
                struct_fields,
            );
        }
    }
}

/// Empty registry row for a whole-arg unresolved associated-type
/// projection. `intern_opaque_payload_spelling` looks up an Opaque
/// TypeDecl and skips `??`; this spelling has none. One GC word
/// (`InstanceRepr`), same as that Opaque arm.
fn intern_projection_payload_spelling(
    spelling: &str,
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
) {
    let spelling = spelling.trim();
    if !lltype::type_arg_is_whole_unresolved_projection(spelling) {
        return;
    }
    struct_fields
        .fields
        .entry(spelling.to_string())
        .or_insert_with(Vec::new);
}

/// Intern payload field types render as a TypeDecl leaf
/// (`charon_adt_to_ast_string`). The intern pass runs per LLBC, before
/// the cross-crate `harden_duplicate_leaf_metadata` that withdraws a
/// duplicate last-segment. A leaf still present in this crate's
/// `struct_fields` can vanish from the merged registry; `project_struct_field_type`
/// of that spelling is then Impossible and getattr of the interned
/// variant's `__pos_0` blocks. Map a TypeDecl leaf onto the unique
/// declaration whose crate-stripped path or full `name_path` is already
/// registered, even while the leaf alias still exists. Two TypeDecls
/// sharing the leaf keep the original spelling (fail-closed). The intern
/// `<…>` class suffix is unchanged: CAST owner_root still uses the leaf
/// render.
fn intern_payload_registry_spelling(
    llbc: &Llbc,
    spelling: &str,
    struct_fields: &crate::front::semantic::StructFieldRegistry,
) -> String {
    let spelling = spelling.trim();
    if spelling.is_empty()
        || spelling.contains('<')
        || spelling.contains("::")
        || spelling.starts_with('*')
        || spelling.starts_with('&')
        || spelling.starts_with('[')
        || spelling.starts_with('(')
    {
        return spelling.to_string();
    }
    let mut resolved: Option<String> = None;
    for td in llbc.iter_type_decls() {
        let name = td.item_meta.name_path();
        let leaf = name.rsplit("::").next().unwrap_or(name.as_str());
        if spelling != leaf {
            continue;
        }
        // A fieldless enum is its discriminant integer, not a ClassDef.
        // Unique-leaf remap to the declared tag width so getattr of
        // `Result::Err.__pos_0` is Integer (`tyref_to_value_type`).
        if let Some(int_ty) = fieldless_enum_payload_int_ty(td, llbc) {
            match resolved.as_ref() {
                Some(prev) if prev != &int_ty => return spelling.to_string(),
                Some(_) => {}
                None => resolved = Some(int_ty),
            }
            continue;
        }
        let canon = strip_crate_prefix(&name);
        let key = if !canon.is_empty() && struct_fields.fields.contains_key(&canon) {
            canon
        } else if struct_fields.fields.contains_key(&name) {
            name
        } else {
            continue;
        };
        match resolved.as_ref() {
            Some(prev) if prev != &key => return spelling.to_string(),
            Some(_) => {}
            None => resolved = Some(key),
        }
    }
    resolved.unwrap_or_else(|| spelling.to_string())
}

/// Publish an empty struct row for a `TypeDeclKind::Opaque` payload the
/// interned variant names. `derive_program_metadata` skips Opaque decls,
/// so `project_struct_field_type` of that leaf was Impossible and
/// Blocked getattr of `Some.__pos_0` (`rclass.py` `s_getattr`). An empty
/// registry entry is the same registered-struct arm a Struct TypeDecl
/// already takes; the payload stays one GC word (`InstanceRepr`).
fn intern_opaque_payload_spelling(
    llbc: &Llbc,
    spelling: &str,
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
) {
    let spelling = spelling.trim();
    if spelling.is_empty()
        || spelling.contains("??")
        || is_primitive_payload_type(spelling)
        || spelling.contains('<')
        || spelling.starts_with('*')
        || spelling.starts_with('&')
        || spelling.starts_with('[')
        || spelling.starts_with('(')
    {
        return;
    }
    for td in llbc.iter_type_decls() {
        if !matches!(td.kind, TypeDeclKind::Opaque) {
            continue;
        }
        let name = td.item_meta.name_path();
        let leaf = name.rsplit("::").next().unwrap_or(name.as_str());
        let canon = strip_crate_prefix(&name);
        if spelling != name && spelling != leaf && spelling != canon {
            continue;
        }
        for key in [name.as_str(), canon.as_str(), leaf] {
            if key.is_empty() {
                continue;
            }
            struct_fields
                .fields
                .entry(key.to_string())
                .or_insert_with(Vec::new);
        }
        return;
    }
}

/// One concrete `rordereddict_entries::Entry` / `GcEntries` use found in a
/// signature or a local. `found` holds the rendered type arguments.
struct EntryInstantiationScan<'a> {
    llbc: &'a Llbc,
    entry_id: u64,
    gc_entries_id: Option<u64>,
    seen_dedup: std::collections::HashSet<u64>,
    seen_suffix: std::collections::HashSet<String>,
    found: Vec<Vec<String>>,
}

impl EntryInstantiationScan<'_> {
    fn consider(&mut self, root: Option<&serde_json::Value>) {
        let Some(root) = root else {
            return;
        };
        let mut stack = vec![(root, 0usize)];
        while let Some((node, depth)) = stack.pop() {
            if depth > 24 {
                continue;
            }
            let Some(obj) = node.as_object() else {
                continue;
            };
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                if !self.seen_dedup.insert(id) {
                    continue;
                }
                if let Some(body) = self.llbc.dedup_body(id) {
                    stack.push((body, depth + 1));
                }
                continue;
            }
            if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
                && arr.len() == 2
            {
                stack.push((&arr[1], depth + 1));
                continue;
            }
            if let Some(adt) = obj.get("Adt").and_then(serde_json::Value::as_object) {
                if let Some(id) = type_decl_ref_adt_id(adt)
                    && (id == self.entry_id || self.gc_entries_id == Some(id))
                {
                    let args = render_adt_type_args(adt, self.llbc, depth);
                    if !args.is_empty()
                        && args
                            .iter()
                            .all(|arg| !arg.is_empty() && !arg.contains("??"))
                    {
                        let suffix = args.join(",");
                        if self.seen_suffix.insert(suffix) {
                            self.found.push(args);
                        }
                    }
                }
                if let Some(types) = type_decl_ref_generics(adt, self.llbc)
                    .and_then(serde_json::Value::as_object)
                    .and_then(|generics| generics.get("types"))
                    .and_then(serde_json::Value::as_array)
                {
                    for ty in types {
                        stack.push((ty, depth + 1));
                    }
                }
                continue;
            }
            if let Some(inner) = obj
                .get("RawPtr")
                .and_then(serde_json::Value::as_array)
                .and_then(|arr| arr.first())
                .or_else(|| {
                    obj.get("Ref")
                        .and_then(serde_json::Value::as_array)
                        .and_then(|arr| arr.get(1))
                })
                .or_else(|| {
                    obj.get("Array")
                        .and_then(serde_json::Value::as_array)
                        .and_then(|arr| arr.first())
                })
                .or_else(|| {
                    obj.get("Slice")
                        .and_then(serde_json::Value::as_array)
                        .and_then(|arr| arr.first())
                })
            {
                stack.push((inner, depth + 1));
            }
        }
    }
}

/// Publish concrete `Entry<i64, *mut PyObject>` field rows.
///
/// Charon leaves `rordereddict_entries::Entry` generic, so
/// `derive_program_metadata` records `key` as a type variable.
/// `_handle_dict_lookup_call` extra descrs need that concrete `i64` key
/// (`arraydescrof(DICTENTRYARRAY)`). A fully concrete use of `Entry`, or
/// the same arguments on `GcEntries`, supplies the rows. Only the
/// qualified spelling is inserted. `#[repr(C)]` places `key`, `f_valid`,
/// `value`, and `f_hash` in declaration order.
fn register_ordereddict_i64_entry_rows(
    llbc: &Llbc,
    known_struct_names: &mut std::collections::HashSet<String>,
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) {
    let mut entry: Option<(u64, String)> = None;
    let mut gc_entries_id = None;
    for td in llbc.iter_type_decls() {
        let path = td.item_meta.name_path();
        if entry.is_none()
            && path.ends_with("rordereddict_entries::Entry")
            && matches!(td.kind, TypeDeclKind::Struct(_))
        {
            entry = Some((td.def_id, path));
        } else if gc_entries_id.is_none() && path.ends_with("rordereddict_entries::GcEntries") {
            gc_entries_id = Some(td.def_id);
        }
    }
    let Some((entry_id, entry_path)) = entry else {
        return;
    };
    let mut scan = EntryInstantiationScan {
        llbc,
        entry_id,
        gc_entries_id,
        seen_dedup: std::collections::HashSet::new(),
        seen_suffix: std::collections::HashSet::new(),
        found: Vec::new(),
    };
    for fd in llbc.iter_local_fns() {
        for ty in fd
            .signature
            .inputs
            .iter()
            .chain(std::iter::once(&fd.signature.output))
        {
            scan.consider(tyref_node(ty, llbc));
        }
        if let Some(locals) = fd.unstructured_locals() {
            for local in &locals.locals {
                scan.consider(tyref_node(&local.ty, llbc));
            }
        }
    }
    let Some(td) = llbc.type_by_id(entry_id) else {
        return;
    };
    let TypeDeclKind::Struct(fields) = &td.kind else {
        return;
    };
    for args in scan.found {
        let mut rows = Vec::with_capacity(fields.len());
        let mut key_is_i64 = false;
        let mut valid = false;
        let mut value = false;
        let mut hash = false;
        let mut concrete = true;
        for (index, field) in fields.iter().enumerate() {
            let fname = field_decl_name(index, field);
            let concrete_ty = substitute_field_type(
                &field.ty,
                &args,
                llbc,
                gc_struct_ids,
                &std::collections::HashSet::<String>::new(),
            );
            if concrete_ty.contains("??") {
                concrete = false;
                break;
            }
            match fname.as_str() {
                "key" => key_is_i64 = concrete_ty == "i64",
                "f_valid" => valid = true,
                "value" => value = true,
                "f_hash" => hash = true,
                _ => {}
            }
            rows.push(field_row_from_decl(index, field, concrete_ty, llbc));
        }
        if !concrete || !key_is_i64 || !valid || !value || !hash {
            continue;
        }
        let name = format!("{entry_path}<{}>", args.join(","));
        struct_fields.fields.entry(name.clone()).or_insert(rows);
        known_struct_names.insert(name);
    }
}

pub fn build_semantic_program_from_llbc(
    llbc: &Llbc,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    build_semantic_program_from_llbc_with_static_addrs(llbc, crate::HostStaticAddrs::default())
}

pub fn build_semantic_program_from_llbc_with_static_addrs(
    llbc: &Llbc,
    static_addrs: crate::HostStaticAddrs<'_>,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    // The linking pass belongs to a whole translation input, so the
    // multi-artefact entry point runs it once over all of them and the
    // per-artefact builder below does not repeat it. A caller handing over a
    // single artefact has named its whole input here, so link it here.
    link_transparent_scalar_types(std::slice::from_ref(llbc));
    let jitdriver_receiver_roots =
        crate::codewriter::jtransform::default_jitdriver_receiver_roots();
    build_semantic_program_from_llbc_with_static_addrs_filtered(
        llbc,
        static_addrs,
        &jitdriver_receiver_roots,
        None,
        None,
        no_tombstoned_leaves(),
    )
}

/// `{struct_name → [(field, rank)]}` from `_immutable_fields_<Struct>`
/// marker consts (`harvest_immutable_fields_from_llbcs`).
pub(crate) type ImmutableFieldsTable =
    std::collections::HashMap<String, Vec<(String, crate::model::ImmutableRank)>>;

/// One artefact's lowering state: the declaration tables every body of the
/// artefact lowers against, the hint sets harvested from it, its root-stack
/// analysis, its clause-specialization queue and the declarations that did
/// not lower. A [`CrateLowering`] borrows it together with the artefact.
pub(crate) struct CrateLoweringState {
    known_trait_names: std::collections::HashSet<String>,
    struct_field_attrs: std::collections::HashMap<String, Vec<(String, ValueType)>>,
    /// Copied by [`Self::finish`]; a body lowered after that still reads
    /// it, including Opaque scalar-word projections that resolve a
    /// declared name from the defining crate's harvest.
    struct_fields: crate::front::semantic::StructFieldRegistry,
    /// Taken by [`Self::finish`]; no body lowering reads them.
    exports: std::cell::RefCell<CrateExports>,
    tombstoned_leaves: std::collections::HashSet<String>,
    /// Gc structs and enums visible to every body of this artefact.
    gc_struct_ids: std::collections::HashSet<majit_ir::descr::StructId>,
    /// Harvested `_alloc_flavor_ = raw` structs. `get_type_flag` banks a
    /// pointer to one as an unsigned int word (`descr.py get_type_flag`).
    raw_alloc_flavor_structs: std::collections::HashSet<String>,
    /// `_immutable_fields_<Struct>` harvested once for this artefact
    /// (`harvest_immutable_fields_from_llbcs`). Overlay dests and the
    /// raw-storage classifier look the struct up by the same keys the
    /// GC-schema path uses (leaf and crate-stripped path).
    immutable_fields: ImmutableFieldsTable,
    dont_look_inside: std::collections::HashSet<String>,
    elidable_residual: std::collections::HashSet<String>,
    not_rpython: std::collections::HashSet<String>,
    /// The `func._jit_*_` / `_elidable_function_` attributes of every
    /// funcobj in the translation input, keyed by `{module_path}::{name}`
    /// (`llbc_hints::harvest_hints_from_llbcs` across the whole set). The
    /// header stamps a function's bag onto it as it is built, so the
    /// attributes exist whether or not its body is.
    func_hints: std::collections::HashMap<String, Vec<String>>,
    /// One root-stack analysis per artefact, so a callee shared by many
    /// brackets is analysed once (`GraphAnalyzer._analyzed_calls`).
    root_stack: RootStackState,
    spec: std::cell::RefCell<crate::front::clause_spec::SpecQueue>,
    skipped: LoweringSkips,
    atomic_load_decls: std::cell::RefCell<
        Vec<crate::translator::rtyper::lltypesystem::module::ll_extaccessor::DeclinedFunDecl>,
    >,
}

/// The declaration tables [`CrateLoweringState::finish`] hands to the
/// program.
#[derive(Default)]
struct CrateExports {
    known_struct_names: std::collections::HashSet<String>,
    enum_variant_by_discriminant:
        std::collections::HashMap<String, std::collections::HashMap<i64, String>>,
    struct_origins: std::collections::HashMap<String, String>,
    exact_layouts:
        std::collections::HashMap<majit_ir::descr::StructId, crate::front::semantic::ExactLayout>,
    struct_ids: std::collections::HashMap<String, Option<majit_ir::descr::StructId>>,
}

/// One artefact's lowering context: the artefact, the host addresses and
/// jitdriver roots its bodies lower against, and its [`CrateLoweringState`].
///
/// `build_semantic_program_from_llbc_with_static_addrs_filtered` lowers each
/// declaration through [`CrateLowering::lower_decl`] and each queued
/// specialization through [`CrateLowering::lower_spec`].
pub(crate) struct CrateLowering<'l> {
    llbc: &'l Llbc,
    static_addrs: crate::HostStaticAddrs<'l>,
    jitdriver_receiver_roots: &'l [String],
    state: &'l CrateLoweringState,
    root_stack: RootStackAnalyzer<'l>,
}

impl CrateLoweringState {
    pub(crate) fn new(
        llbc: &Llbc,
        cross_tombstoned_leaves: &std::collections::HashSet<String>,
        func_hints: std::collections::HashMap<String, Vec<String>>,
        skipped: LoweringSkips,
        gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
        program_raw_alloc_flavor_structs: &std::collections::HashSet<String>,
        module_filter: Option<&std::collections::HashSet<String>>,
        function_filter: Option<&std::collections::HashSet<String>>,
    ) -> Self {
        shadow_stack_erase::ensure_stack_sensitive_fns(llbc);
        let immutable_fields = crate::front::llbc_hints::harvest_immutable_fields_from_llbcs(
            std::slice::from_ref(llbc),
        );
        let mut raw_alloc_flavor_structs =
            crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(
                llbc,
            ));
        // Flavor is a program-wide class property; a dependency's raw classes count here too.
        raw_alloc_flavor_structs.extend(program_raw_alloc_flavor_structs.iter().cloned());
        // ── Pass 1: walk type_decls + trait_decls ─────────────────────
        let (
            mut known_struct_names,
            known_trait_names,
            mut struct_fields,
            mut enum_variant_by_discriminant,
            mut struct_origins,
            mut struct_field_attrs,
            mut exact_layouts,
            mut struct_ids,
        ) = derive_program_metadata(
            llbc,
            gc_struct_ids,
            &immutable_fields,
            &raw_alloc_flavor_structs,
        );
        promote_cross_crate_stripped_keys(
            llbc,
            cross_tombstoned_leaves,
            &mut known_struct_names,
            &mut struct_fields,
            &mut struct_field_attrs,
            &mut exact_layouts,
            &mut struct_ids,
        );
        // Only leaves this pass withdrew. A crate-root decl
        // (`charon_corpus::ClassObject`) also stores an empty origin module,
        // and that empty string is not a withdrawal.
        let mut tombstoned_leaves = harden_duplicate_leaf_metadata(
            &mut struct_fields,
            &mut struct_origins,
            &mut enum_variant_by_discriminant,
            Some(&struct_ids),
        );

        // Per-instantiation enum-variant pre-registration source (#100): a
        // reference-payload generic enum constructor (`Result<Tuple>::Ok`)
        // projects a per-instantiation variant class so its payload does not
        // union across instantiations.  Publish a discriminant-table entry
        // per such instantiation, keyed by the LEAF-suffixed `{leaf}{suffix}`
        // — the exact spelling the constructor carries as its owner tail
        // (`resolve_aggregate_adt`) — so the prologue pre-mint and the
        // constructor pass the identical string to `canonical_struct_name`
        // and resolve ONE variant classdef regardless of what
        // `STRUCT_ORIGIN_REGISTRY` maps the base to.  The cloned tag→variant
        // map is instantiation-invariant.  `pre_register_enum_variant_classes`
        // accepts the `<`-bearing key (its filter admits `::`-qualified OR
        // per-instantiation roots) and numbers the variant subclasses before
        // `assign_inheritance_ids`, so the split classes drain rather than
        // landing unnumbered (per-graph Skip).
        let ref_enum_insts =
            collect_ref_enum_instantiations(llbc, gc_struct_ids, module_filter, function_filter);
        for inst in &ref_enum_insts {
            let leaf = inst
                .name_path
                .rsplit("::")
                .next()
                .unwrap_or(&inst.name_path);
            // Mirror the discriminant map to the `{leaf}{suffix}` spelling only
            // while the bare leaf survived `harden_duplicate_leaf_metadata`.  A
            // suffixed key carries no `::`, so it escapes the `::`-keyed
            // dup-leaf pass; re-arming it for a leaf the hardening withdrew on
            // cross-decl ambiguity would resurrect the silent-winner alias.
            // Fail-closed — a withheld alias misses to the qualified key or
            // routes the suffixed receiver to a per-graph Skip.
            if !enum_variant_by_discriminant.contains_key(leaf) {
                continue;
            }
            if let Some(bare) = enum_variant_by_discriminant.get(&inst.name_path).cloned() {
                enum_variant_by_discriminant
                    .entry(format!("{leaf}{}", inst.suffix))
                    .or_insert(bare);
            }
        }
        // Per-instantiation SUFFIXED variant payload rows (#312 C4): give a
        // narrowing-only reference-payload receiver its concrete payload type
        // so the suffixed variant classdef does not stay `Impossible` → Skip.
        register_ref_enum_instantiation_rows(
            llbc,
            &ref_enum_insts,
            &enum_variant_by_discriminant,
            &mut struct_fields,
            &mut struct_ids,
            gc_struct_ids,
        );
        // Concrete `Entry<i64, *mut PyObject>` rows. The generic decl's
        // `key` is a type variable; extra descrs for `OS_DICT_LOOKUP`
        // need the monomorph `get_ll_dict` would have built.
        register_ordereddict_i64_entry_rows(
            llbc,
            &mut known_struct_names,
            &mut struct_fields,
            gc_struct_ids,
        );

        // Pass 2 paints each ADT as its bare leaf. A leaf `harden` withdrew
        // (`eval::Code` beside `module::struct::Code`) is not a class: the
        // paint has to name the declaration actually being lowered, or the
        // value seeds `SomeInstance(classdef=None)` and a later
        // `__discriminant` read raises `MissingRTypeAttribute`. The set is
        // the leaves the pass withdrew, not every empty origin: a crate-root
        // decl stores an empty module without being a duplicate. Leaves that
        // collide only across input LLBCs are absent from this file; the
        // caller computed that verdict before lowering.
        tombstoned_leaves.extend(cross_tombstoned_leaves.iter().cloned());

        // ── Pass 2: lower every function body and build SemanticFunctions ─
        // `@jit.dont_look_inside` (`rlib/jit.py`) callees declare a
        // FUNC.RESULT that the legacy walker reads off the Call op's
        // `result_ty` (`legacy_resolve.rs infer_concrete_from_op`), but the
        // real path stubs the opaque body and otherwise drops the residual
        // call result to void.  Harvest the marker set once (same
        // `_jit_look_inside_` source as `func_hints`) so the per-fn push
        // can stamp the matching `return_type` token, keyed by the
        // identical `{module_path}::{name}` path.
        let harvested =
            crate::front::llbc_hints::harvest_hints_from_llbcs(std::slice::from_ref(llbc));
        let dont_look_inside: std::collections::HashSet<String> = harvested
            .iter()
            .filter(|(_, hints)| hints.iter().any(|h| h == "dont_look_inside"))
            .map(|(path, _)| path.clone())
            .collect();
        // `#[majit_macros::elidable]` callees (`llbc_hints.rs` maps the
        // `_elidable_function_` marker → `"elidable"`): the codewriter's
        // elidable effect already lowers the callsite to CALL_PURE and never
        // looks inside the body.  Harvest the elidable set from the same
        // vector so `stamp_return_token` can stamp the matching FUNC.RESULT
        // token. `JitPolicy._reject_function()` makes this unconditional.
        let elidable_residual: std::collections::HashSet<String> = harvested
            .iter()
            .filter(|(_, hints)| hints.iter().any(|h| h == "elidable"))
            .map(|(path, _)| path.clone())
            .collect();
        // `objectmodel.py @not_rpython` sets `func._not_rpython_`; the
        // flowspace refuses that function before executing its body
        // (`flowspace/objspace.py assert_rpythonic`).  Apply the same gate
        // before MIR lowering, which also prevents host-only carrier types (for
        // example rbigint.fromlong's i128 test surface) from becoming spurious
        // translated graph-coverage failures.
        let not_rpython: std::collections::HashSet<String> = harvested
            .iter()
            .filter(|(_, hints)| hints.iter().any(|h| h == "not_rpython"))
            .map(|(path, _)| path.clone())
            .collect();
        // The eager builder passes an empty map: it has no cross-crate
        // union. Leaving that empty drops every `_jit_*_` token, so a
        // loopy spec copy never receives `unroll_safe` and
        // `look_inside_graph` declines it. The local harvest is that
        // crate's whole bag. A caller that already unioned every crate
        // passes a non-empty map and keeps it.
        let func_hints = if func_hints.is_empty() {
            harvested
        } else {
            func_hints
        };
        for fd in llbc.iter_local_fns() {
            note_body_crate_root_into(fd, &mut struct_fields);
        }
        Self {
            known_trait_names,
            struct_field_attrs,
            struct_fields,
            exports: std::cell::RefCell::new(CrateExports {
                known_struct_names,
                enum_variant_by_discriminant,
                struct_origins,
                exact_layouts,
                struct_ids,
            }),
            tombstoned_leaves,
            gc_struct_ids: gc_struct_ids.clone(),
            raw_alloc_flavor_structs,
            immutable_fields,
            dont_look_inside,
            elidable_residual,
            not_rpython,
            func_hints,
            root_stack: RootStackState::new(llbc),
            spec: std::cell::RefCell::new(crate::front::clause_spec::SpecQueue::new()),
            skipped,
            atomic_load_decls: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Union `other`'s harvested rows into this crate's registry. Later
    /// crates see earlier ones so an Opaque projection can name a field
    /// the defining crate registered. `or_insert` keeps the first row.
    #[cfg(any(test, feature = "mir-frontend"))]
    pub(crate) fn absorb_struct_fields(
        &mut self,
        other: &crate::front::semantic::StructFieldRegistry,
    ) {
        for (key, fields) in other.fields.iter() {
            self.struct_fields
                .fields
                .entry(key.clone())
                .or_insert_with(|| fields.clone());
        }
        for root in &other.foreign_body_roots {
            self.struct_fields.note_foreign_body_crate_root(root);
        }
        for (crate_root, stripped) in &other.foreign_callables {
            self.struct_fields
                .note_foreign_body_callable(crate_root, stripped);
        }
    }

    #[cfg(any(test, feature = "mir-frontend"))]
    pub(crate) fn struct_fields(&self) -> &crate::front::semantic::StructFieldRegistry {
        &self.struct_fields
    }

    /// Union `other`'s harvested raw-flavor structs so a later crate's
    /// `get_type_flag` sees earlier crates' `_alloc_flavor_ = raw` markers.
    #[cfg(any(test, feature = "mir-frontend"))]
    pub(crate) fn absorb_raw_alloc_flavor_structs(
        &mut self,
        other: &std::collections::HashSet<String>,
    ) {
        self.raw_alloc_flavor_structs.extend(other.iter().cloned());
    }

    #[cfg(any(test, feature = "mir-frontend"))]
    pub(crate) fn raw_alloc_flavor_structs(&self) -> &std::collections::HashSet<String> {
        &self.raw_alloc_flavor_structs
    }
}

impl<'l> CrateLowering<'l> {
    pub(crate) fn new(
        llbc: &'l Llbc,
        static_addrs: crate::HostStaticAddrs<'l>,
        jitdriver_receiver_roots: &'l [String],
        state: &'l CrateLoweringState,
    ) -> Self {
        Self {
            llbc,
            static_addrs,
            jitdriver_receiver_roots,
            state,
            root_stack: RootStackAnalyzer::new(llbc, &state.root_stack),
        }
    }

    /// Lower every declaration the filters admit, then every clause
    /// specialization those bodies queued.
    pub(crate) fn lower_all(
        &self,
        module_filter: Option<&std::collections::HashSet<String>>,
        function_filter: Option<&std::collections::HashSet<String>>,
    ) -> Vec<crate::front::semantic::SemanticFunction> {
        let mut functions: Vec<_> = self
            .llbc
            .iter_local_fns()
            .filter_map(|fd| self.lower_decl(fd, module_filter, function_filter))
            .collect();
        functions.extend(self.lower_specs());
        functions
    }

    /// Lower every clause specialization queued so far.
    ///
    /// `specialize.py` `cachedgraph` keys one graph per instantiation. A
    /// built body enqueues each concrete call; lowering a copy enqueues the
    /// callees whose clauses that copy just bound.
    pub(crate) fn lower_specs(&self) -> Vec<crate::front::semantic::SemanticFunction> {
        let mut functions = Vec::new();
        while let Some(req) = self.pop_spec() {
            functions.extend(self.lower_spec(req));
        }
        functions
    }

    /// Lower one declaration of this artefact. `None` when a gate refuses it
    /// or its body does not lower; the latter is recorded in `skipped`.
    pub(crate) fn lower_decl(
        &self,
        fd: &'l FunDecl,
        module_filter: Option<&std::collections::HashSet<String>>,
        function_filter: Option<&std::collections::HashSet<String>>,
    ) -> Option<crate::front::semantic::SemanticFunction> {
        if !self.admit_decl(fd, module_filter, function_filter) {
            return None;
        }
        if self.declare_llexternal(fd) {
            return None;
        }
        match self.build_decl(fd) {
            Ok(function) => Some(function),
            Err(error) => {
                self.record_decl_failure(fd, error);
                None
            }
        }
    }

    /// Whether `fd` declares a funcobj of this program: the membership
    /// gates [`Self::lower_decl`] applies before building anything.
    pub(crate) fn admit_decl(
        &self,
        fd: &FunDecl,
        module_filter: Option<&std::collections::HashSet<String>>,
        function_filter: Option<&std::collections::HashSet<String>>,
    ) -> bool {
        // Charon emits static / const initialiser bodies (e.g. the
        // body that builds `static NONE_SINGLETON`) as ordinary
        // `FunDecl` entries whose `src` is `GlobalInitializer` of the
        // backing `GlobalDecl` id.  These bodies are not call targets
        // at the JIT level — skip them so they never surface as
        // call-registry entries the rest of the pipeline does not
        // model.  (Their unwind paths lower via `set_raise`,
        // `model.rs`; the flowspace adapter converts only the
        // reachable block closure, so an unreachable unwind block's
        // orphan etype/evalue slots no longer reject the graph — this
        // skip is about call-target modelling, not adapter safety.)
        if fd.is_global_initializer().is_some() {
            return false;
        }
        let (module_path, name) = decl_module_path_and_name(fd);
        if !should_lower_module(module_filter, &module_path) {
            return false;
        }
        if !should_lower_function(function_filter, &name) {
            return false;
        }
        // `rlib/jit.py` `ConditionalCallEntry` is an ExtRegistry object, not
        // a graph (`HOST_ENV` `jit_conditional_call`). Harvesting the
        // residual `conditional_call*` FunDecl makes every instantiation
        // share one `FunctionGraph` whose `mergeinputargs` slot=1 unions
        // distinct fn-ptr `Ptr` TOs. `isconstant` is also HOST_ENV and
        // stays admitted. Recognition is the callable's real path
        // (`majit_rlib::jit::<leaf>` / local-crate-stripped `jit::<leaf>`).
        if fun_decl_is_jit_conditional_call(fd) {
            return false;
        }
        // A foreign generic template is not a graph of this artefact
        // (`specialize.py` `FunctionDesc.cachedgraph`). Spec copies are
        // declared from the caller's concrete arguments. A local generic
        // template stays: this artefact's own `cachedgraph` (`execute_opcode_step`
        // with a clause) is the dispatcher the codewriter looks up.
        if crate::front::clause_spec::decl_is_generic(fd) && !fd.item_meta.is_local {
            return false;
        }
        let fn_path = decl_fn_path(&module_path, &name);
        !self.state.not_rpython.contains(&fn_path)
    }

    /// Whether `fd` is declared as an `rffi.llexternal` word reader
    /// ([`WORD_LOAD_LLEXTERNALS`]): a function object with no graph, whose
    /// signature is recorded for the call registry's external.
    ///
    /// [`WORD_LOAD_LLEXTERNALS`]: crate::translator::rtyper::lltypesystem::module::ll_extaccessor::WORD_LOAD_LLEXTERNALS
    pub(crate) fn declare_llexternal(&self, fd: &FunDecl) -> bool {
        use crate::translator::rtyper::lltypesystem::module::ll_extaccessor::WORD_LOAD_LLEXTERNALS;
        let name_path = fd.item_meta.name_path();
        let declared = WORD_LOAD_LLEXTERNALS
            .iter()
            .any(|row| row.segments.iter().copied().eq(name_path.split("::")));
        if declared {
            self.state
                .atomic_load_decls
                .borrow_mut()
                .push(declined_atomic_load_fun_decl(
                    self.llbc,
                    fd,
                    "rffi.llexternal declaration".to_string(),
                ));
        }
        declared
    }

    /// Record why the body of `fd` produced no graph.
    pub(crate) fn record_decl_failure(&self, fd: &FunDecl, error: DeclBuildError) {
        match error {
            DeclBuildError::NoBody => {
                // A declaration with no unstructured body never becomes a
                // `SemanticFunction`, so it never reaches `function_graphs` and
                // every callsite resolves it as an unregistered path.  The
                // lowering errors below are already surfaced by `skipped`; this
                // arm was the one membership drop that left no trace at all.
                let (module_path, name) = decl_module_path_and_name(fd);
                crate::decline::record_named(
                    crate::decline::gate::SEMANTIC_FN_LOOP,
                    "declaration-has-no-unstructured-body",
                    &decl_fn_path(&module_path, &name),
                );
            }
            DeclBuildError::Lower { name, error } => {
                self.state
                    .skipped
                    .borrow_mut()
                    .push((name, error.to_string()));
            }
        }
    }

    /// Build the function `fd` declares: its header from the declaration,
    /// then its body. The membership gates are [`Self::lower_decl`]'s.
    pub(crate) fn build_decl(
        &self,
        fd: &'l FunDecl,
    ) -> Result<crate::front::semantic::SemanticFunction, DeclBuildError> {
        let mut header = self.decl_header(fd);
        let graph = self.build_decl_body(fd)?;
        // After `FUNC.RESULT` is chosen. Adding the path to the harvested
        // `dont_look_inside` set would restamp that token.
        header.residualize_unresolved_trait_const(&graph);
        Ok(header.into_function(graph))
    }

    /// The declaration facts of the funcobj `fd` declares.
    pub(crate) fn decl_header(&self, fd: &FunDecl) -> SemanticFunctionHeader {
        let CrateLoweringState {
            known_trait_names,
            dont_look_inside,
            elidable_residual,
            func_hints,
            ..
        } = self.state;
        let (module_path, name) = decl_module_path_and_name(fd);
        let fn_path = decl_fn_path(&module_path, &name);
        // A monomorphized copy registers under its instance leaf, as a
        // clause-specialized copy does below; `fn_path` stays the template's
        // for the policy and hint lookups every instance shares.
        let name = instance_leaf(self.llbc, fd).unwrap_or(name);
        // `return_type` stays `None` for ordinary fns: the Charon
        // dedup-table resolution cannot yet map a
        // `TyRef::Deduplicated{id}` to its primitive name, and the
        // codewriter's call-signature validator skips the check when the
        // declared type is None, which is the right behaviour while the
        // resolution gap is open — TyRef labels (`ty#170`) would
        // otherwise be classified as `Type::Ref` and trip a spurious
        // mismatch panic against a real `Type::Int` callee result.  The
        // header stamps the `dont_look_inside` exception: an opaque
        // callee's FUNC.RESULT token is the only signal the rtyper stub
        // has to shell its residual-call result to the same register
        // kind the legacy walker derives, so it is filled in even though
        // the general resolution gap remains open.
        // Surface the impl-method owner on the SemanticFunction so
        // `lib.rs`'s `analyze_pipeline_from_module_paths` and the
        // `extract_inherent_impl_methods` / `extract_trait_impls`
        // consumers see the same `self_ty_root` the MIR driver records.
        // Without this, every impl method built by the MIR driver looks
        // like a free function to the canonical registration loop and
        // the impl-key return-type / hint registrations get dropped.
        SemanticFunctionHeader::new(
            self.llbc,
            fd,
            name,
            module_path,
            &fd.signature,
            known_trait_names,
            dont_look_inside,
            elidable_residual,
            func_hints,
            self.static_addrs.error_carrier,
            &self.state.gc_struct_ids,
            &self.state.raw_alloc_flavor_structs,
            &fn_path,
        )
    }

    /// What the funcobj `fd` declares, built from the declaration alone —
    /// `pygraph.py PyGraph.__init__` reads the code object, not the flowed
    /// body: the startblock the body lowering starts from and the
    /// `FUNC.RESULT` it stamps, under the header's `stamp`. `None` without
    /// an `Unstructured` body.
    #[cfg(any(test, feature = "mir-frontend"))]
    pub(crate) fn decl_header_graph(
        &self,
        fd: &FunDecl,
        stamp: &GraphStamp,
    ) -> Option<crate::model::FunctionGraph> {
        let locals = fd.unstructured_locals()?;
        let mut graph = crate::model::FunctionGraph::new(graph_name_of(self.llbc, fd));
        let cells = gc_mut_ref_param_locals(
            &locals,
            self.llbc,
            &self.state.tombstoned_leaves,
            &self.state.gc_struct_ids,
            fun_decl_is_dont_look_inside(fd, &self.state.dont_look_inside),
            &self.state.raw_alloc_flavor_structs,
        );
        pygraph_initial_block(
            &mut graph,
            &locals,
            self.llbc,
            fd.generics.as_ref(),
            &self.state.tombstoned_leaves,
            &cells,
            &self.state.gc_struct_ids,
            fundecl_keeps_zst_self(
                self.llbc,
                fd,
                Some(&locals),
                &self.state.tombstoned_leaves,
                &self.state.gc_struct_ids,
                &self.state.raw_alloc_flavor_structs,
            ),
            &self.state.raw_alloc_flavor_structs,
        );
        if result_exc_ok_is_unit(
            &fd.signature.output,
            self.llbc,
            self.static_addrs.error_carrier,
        ) {
            graph.return_type = Some("()".to_string());
        }
        Some(stamp.apply(graph))
    }

    /// Lower the body of `fd`, unstamped: the funcobj's header stamps it
    /// ([`SemanticFunctionHeader::graph_stamp`]).
    pub(crate) fn build_decl_body(
        &self,
        fd: &FunDecl,
    ) -> Result<crate::model::FunctionGraph, DeclBuildError> {
        let CrateLoweringState {
            struct_field_attrs,
            struct_fields,
            tombstoned_leaves,
            gc_struct_ids,
            immutable_fields,
            raw_alloc_flavor_structs,
            dont_look_inside,
            spec,
            ..
        } = self.state;
        let Self {
            llbc,
            static_addrs,
            jitdriver_receiver_roots,
            ref root_stack,
            ..
        } = *self;
        // `FunDecl::body` is raw JSON and `unstructured()` re-parses it on
        // every call, so hold the one projection this build needs: it is
        // both the "has a lowerable body" gate and the input the lowering
        // below reads.
        let Some(mut body) = fd.unstructured() else {
            return Err(DeclBuildError::NoBody);
        };
        elaborate_explicit_root_closes(llbc, &mut body, &|reg| regular_call_name_path(reg, llbc));
        elaborate_vec_drops(llbc, &mut body);
        let body = shadow_stack_erase::erase_or_keep(fd, body, llbc);
        // A single function whose body the driver does not yet handle
        // should not abort the whole-program build.  Capture
        // per-function errors into a side bucket and continue; they are
        // surfaced via `MAJIT_MIR_FRONTEND_DEBUG=1` for triage, but
        // production keeps going with a degraded SemanticProgram —
        // failing-loud on the single broken function rather than
        // erroring out at program-build time.
        let accum = AccumulatorFacts::build(llbc, &body);
        let builder_mode = accum.has_builder;
        let graph = match lower_unstructured_with_static_addrs_and_attrs(
            llbc,
            fd,
            &body,
            static_addrs,
            jitdriver_receiver_roots,
            struct_field_attrs,
            struct_fields,
            dont_look_inside,
            tombstoned_leaves,
            gc_struct_ids,
            immutable_fields,
            raw_alloc_flavor_structs,
            builder_mode,
            &accum,
            root_stack,
            Some(spec),
            false,
            None,
        ) {
            Ok(g) => g,
            Err(error) => {
                return Err(DeclBuildError::Lower {
                    name: instance_leaf(llbc, fd)
                        .unwrap_or_else(|| decl_module_path_and_name(fd).1),
                    error,
                });
            }
        };
        Ok(graph)
    }

    /// Pop the next queued clause specialization.
    pub(crate) fn pop_spec(&self) -> Option<crate::front::clause_spec::SpecRequest> {
        self.state.spec.borrow_mut().pop()
    }

    /// Lower one clause specialization. `None` when its body does not
    /// substitute or lower; either is recorded in `skipped`.
    fn lower_spec(
        &self,
        req: crate::front::clause_spec::SpecRequest,
    ) -> Option<crate::front::semantic::SemanticFunction> {
        let mut spec = self.declare_spec(req)?;
        let graph = self.build_spec_body(&spec.body)?;
        spec.header.residualize_unresolved_trait_const(&graph);
        let graph = spec.header.graph_stamp().apply(graph);
        Some(spec.into_semantic(crate::model::LazyGraph::built(graph)))
    }

    /// Declare one clause specialization: the graph
    /// `FunctionDesc.cachedgraph(key)` names before it builds it, with the
    /// substituted body it is built from. `None`, recorded in `skipped`,
    /// when the body does not substitute.
    pub(crate) fn declare_spec(
        &self,
        req: crate::front::clause_spec::SpecRequest,
    ) -> Option<DeclaredSpec> {
        let CrateLoweringState {
            known_trait_names,
            dont_look_inside,
            elidable_residual,
            func_hints,
            skipped,
            ..
        } = self.state;
        let llbc = self.llbc;
        let fd = llbc.fn_by_id(req.fn_id)?;
        let Some(mut body) = crate::front::clause_spec::substituted_unstructured(
            fd,
            llbc,
            &req.trait_refs,
            &req.types,
            &req.const_generics,
        ) else {
            skipped
                .borrow_mut()
                .push((req.leaf, "no substituted unstructured body".into()));
            return None;
        };
        elaborate_explicit_root_closes(llbc, &mut body, &|reg| regular_call_name_path(reg, llbc));
        elaborate_vec_drops(llbc, &mut body);
        let signature = crate::front::clause_spec::substituted_signature(
            &fd.signature,
            llbc,
            &req.types,
            &req.const_generics,
        );
        let stripped = strip_crate_prefix(&fd.item_meta.name_path());
        let (module_path, bare_leaf) = match stripped.rsplit_once("::") {
            Some((module, leaf)) => (module.to_string(), leaf.to_string()),
            None => (String::new(), stripped),
        };
        let policy_fn_path = if module_path.is_empty() {
            bare_leaf
        } else {
            format!("{module_path}::{bare_leaf}")
        };
        let header = SemanticFunctionHeader::new(
            llbc,
            fd,
            req.leaf,
            module_path,
            &signature,
            known_trait_names,
            dont_look_inside,
            elidable_residual,
            func_hints,
            self.static_addrs.error_carrier,
            &self.state.gc_struct_ids,
            &self.state.raw_alloc_flavor_structs,
            &policy_fn_path,
        );
        let segments = spec_segments(llbc, fd, &header.name);
        Some(DeclaredSpec {
            body: std::rc::Rc::new(SpecBody {
                name: header.name.clone(),
                segments,
                body,
                fn_id: fd.def_id,
            }),
            header,
            dont_look_inside: dont_look_inside.contains(&policy_fn_path),
        })
    }

    /// The header graph of the declared clause specialization `spec`: its
    /// startblock over the substituted locals and its header fields.
    #[cfg(any(test, feature = "mir-frontend"))]
    pub(crate) fn spec_header_graph(&self, spec: &DeclaredSpec) -> crate::model::FunctionGraph {
        let fd = self
            .llbc
            .fn_by_id(spec.body.fn_id)
            .expect("a declared specialization names its FunDecl");
        let mut graph = crate::model::FunctionGraph::new(spec.body.segments.join("::"));
        let cells = gc_mut_ref_param_locals(
            &spec.body.body.locals,
            self.llbc,
            &self.state.tombstoned_leaves,
            &self.state.gc_struct_ids,
            fun_decl_is_dont_look_inside(fd, &self.state.dont_look_inside),
            &self.state.raw_alloc_flavor_structs,
        );
        pygraph_initial_block(
            &mut graph,
            &spec.body.body.locals,
            self.llbc,
            fd.generics.as_ref(),
            &self.state.tombstoned_leaves,
            &cells,
            &self.state.gc_struct_ids,
            fundecl_keeps_zst_self(
                self.llbc,
                fd,
                None,
                &self.state.tombstoned_leaves,
                &self.state.gc_struct_ids,
                &self.state.raw_alloc_flavor_structs,
            ),
            &self.state.raw_alloc_flavor_structs,
        );
        // Local 0 is the return place. A clause-spec body is the FunDecl
        // with type arguments substituted, so that local is the instantiated
        // output. The template signature still names the generic Result.
        let output = spec
            .body
            .body
            .locals
            .locals
            .first()
            .map(|local| &local.ty)
            .unwrap_or(&fd.signature.output);
        if result_exc_ok_is_unit(output, self.llbc, self.static_addrs.error_carrier) {
            graph.return_type = Some("()".to_string());
        }
        spec.header.graph_stamp().apply(graph)
    }

    /// Lower the substituted body of a declared clause specialization.
    /// `None`, recorded in `skipped`, when it does not lower.
    pub(crate) fn build_spec_body(&self, spec: &SpecBody) -> Option<crate::model::FunctionGraph> {
        let CrateLoweringState {
            struct_field_attrs,
            struct_fields,
            tombstoned_leaves,
            gc_struct_ids,
            immutable_fields,
            raw_alloc_flavor_structs,
            dont_look_inside,
            spec: spec_queue,
            skipped,
            ..
        } = self.state;
        let Self {
            llbc,
            static_addrs,
            jitdriver_receiver_roots,
            ref root_stack,
            ..
        } = *self;
        let fd = llbc
            .fn_by_id(spec.fn_id)
            .expect("a declared specialization names its FunDecl");
        let body = &spec.body;
        let accum = AccumulatorFacts::build(llbc, body);
        let builder_mode = accum.has_builder;
        let mut graph = match lower_unstructured_with_static_addrs_and_attrs(
            llbc,
            fd,
            body,
            static_addrs,
            jitdriver_receiver_roots,
            struct_field_attrs,
            struct_fields,
            dont_look_inside,
            tombstoned_leaves,
            gc_struct_ids,
            immutable_fields,
            raw_alloc_flavor_structs,
            builder_mode,
            &accum,
            root_stack,
            Some(spec_queue),
            true,
            Some(spec.segments.join("::")),
        ) {
            Ok(g) => g,
            Err(e) => {
                skipped
                    .borrow_mut()
                    .push((spec.name.clone(), e.to_string()));
                return None;
            }
        };
        graph.name = spec.segments.join("::");
        Some(graph)
    }
}

/// A clause specialization as declared: the header of its generic
/// funcobj under the specialized signature, the path its call sites name,
/// and the substituted body its graph is built from.
pub(crate) struct DeclaredSpec {
    pub(crate) header: SemanticFunctionHeader,
    pub(crate) body: std::rc::Rc<SpecBody>,
    dont_look_inside: bool,
}

/// What a clause specialization's graph is built from.
pub(crate) struct SpecBody {
    name: String,
    segments: Vec<String>,
    body: majit_charon_reader::ullbc::Unstructured,
    fn_id: u64,
}

impl DeclaredSpec {
    /// Assemble the `SemanticFunction` around the specialization's graph
    /// handle. `FunctionDesc.cachedgraph` returns the specialized graph of
    /// the same function object, so the copy keeps `_jit_look_inside_`:
    /// `look_inside_graph` reads that hint off the callee.
    pub(crate) fn into_semantic(
        self,
        graph: crate::model::LazyGraph,
    ) -> crate::front::semantic::SemanticFunction {
        let mut lowered = self.header.into_semantic(graph);
        if self.dont_look_inside && !lowered.hints.iter().any(|hint| hint == "dont_look_inside") {
            lowered.hints.push("dont_look_inside".to_string());
        }
        lowered
    }
}

impl DeclaredSpec {
    /// The funcobj under the path the specialization's call sites name,
    /// with the stamps its registration carries.
    #[cfg(any(test, feature = "mir-frontend"))]
    pub(crate) fn into_declared(
        self,
        graph: crate::model::LazyGraph,
    ) -> crate::codewriter::call::DeclaredFuncObj {
        let path = crate::parse::CallPath::from_segments(self.body.segments.iter().cloned());
        let lowered = self.into_semantic(graph);
        crate::codewriter::call::DeclaredFuncObj {
            path,
            graph: lowered.lazy_graph().clone(),
            transform: crate::codewriter::call::GraphTransform {
                return_type: lowered.return_type,
                return_class_root: lowered.return_class_root,
                return_can_be_none: lowered.return_can_be_none,
                hints: lowered.hints,
            },
        }
    }
}

impl CrateLoweringState {
    /// The program.
    ///
    /// The program takes the tables only it reads. `known_trait_names`,
    /// `struct_field_attrs`, and `struct_fields` are copied: a body
    /// lowered after this still reads them.
    pub(crate) fn finish(
        &self,
        functions: Vec<crate::front::semantic::SemanticFunction>,
    ) -> crate::front::semantic::SemanticProgram {
        let known_trait_names = self.known_trait_names.clone();
        let struct_field_attrs = self.struct_field_attrs.clone();
        let struct_fields = self.struct_fields.clone();
        let CrateExports {
            known_struct_names,
            enum_variant_by_discriminant,
            struct_origins,
            exact_layouts,
            struct_ids,
        } = self.exports.take();
        let atomic_load_decls = self.atomic_load_decls.take();
        crate::front::semantic::SemanticProgram {
            functions,
            harvested_hints: std::collections::HashMap::new(),
            known_struct_names,
            known_trait_names,
            struct_fields,
            // Immutable-field tracking depends on `#[majit_macros::immutable]`
            // attribute serialization that Charon does not currently surface
            // (the `attributes` array carries DocComment / Outer but not our
            // proc-macro hints).
            immutable_fields: std::collections::HashMap::new(),
            raw_alloc_flavor_structs: self.raw_alloc_flavor_structs.clone(),
            enum_variant_by_discriminant,
            struct_origins,
            struct_field_attrs,
            exact_layouts,
            struct_ids,
            // Populated post-build in `build_semantic_program_via_active_frontend`
            // (it iterates the full LLBC set).
            unsafe_fn_stubs: Vec::new(),
            foreign_opaque_method_externals: Vec::new(),
            atomic_load_decls,
        }
    }
}

fn build_semantic_program_from_llbc_with_static_addrs_filtered(
    llbc: &Llbc,
    static_addrs: crate::HostStaticAddrs<'_>,
    jitdriver_receiver_roots: &[String],
    module_filter: Option<&std::collections::HashSet<String>>,
    function_filter: Option<&std::collections::HashSet<String>>,
    cross_tombstoned_leaves: &std::collections::HashSet<String>,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    let skips = LoweringSkips::default();
    let gc_struct_ids = harvest_declared_gc_facts(llbc).gc_struct_ids();
    let state = CrateLoweringState::new(
        llbc,
        cross_tombstoned_leaves,
        std::collections::HashMap::new(),
        skips.clone(),
        &gc_struct_ids,
        &std::collections::HashSet::new(),
        module_filter,
        function_filter,
    );
    let functions = CrateLowering::new(llbc, static_addrs, jitdriver_receiver_roots, &state)
        .lower_all(module_filter, function_filter);
    let program = state.finish(functions);
    report_lowering_skips(&skips);
    Ok(program)
}

/// The declarations whose body did not lower, across every crate, in the
/// order they were built. A body is built when something first asks for it,
/// so the log is complete only once the pipeline has built every body it
/// will; [`report_lowering_skips`] reads it then.
pub(crate) type LoweringSkips = std::rc::Rc<std::cell::RefCell<Vec<(String, String)>>>;

/// Report the logged lowering skips.
pub(crate) fn report_lowering_skips(skips: &LoweringSkips) {
    let skipped = skips.borrow();
    // Coverage gate. Every `skipped` entry is a function whose MIR shape
    // the driver could not lower — already after the reverse-postorder
    // retry in `lower_fun_decl`. The tracked gaps are exactly the shapes
    // `is_known_lowering_gap` recognises; its arms are the only statement
    // of that set that cannot go stale. One of them, an "uninitialised local
    // read" that even RPO could not bind, needs a genuine loop-carried def.
    // PRE-EXISTING-ADAPTATION: unlike flowcontext.py
    // FlowContext.record_block, this whole-program boundary does not
    // propagate unsupported lowering. Both tracked and untracked failures
    // omit a body; surviving callers need a valid residual target/ABI.
    // The `regressions` bucket includes EVERY non-tracked skip, not only
    // result-exception-lowering declines. #346 retires this fallback after
    // ordinary lowering handles the reachable closure; check.py remains
    // necessary but is not a proof that every omitted body is safe.
    if !skipped.is_empty() {
        let (tracked, regressions): (Vec<_>, Vec<_>) = skipped
            .iter()
            .partition(|(_, msg)| is_known_lowering_gap(msg));
        if std::env::var("MAJIT_MIR_FRONTEND_DEBUG").is_ok() && !tracked.is_empty() {
            eprintln!(
                "[mir-frontend] {} function(s) skipped via a shape \
                 `is_known_lowering_gap` recognises; the per-function \
                 message below names which one, and all degrade to a \
                 residual call:",
                tracked.len()
            );
            for (name, msg) in tracked.iter().take(20) {
                eprintln!("  {name}: {msg}");
            }
        }
        if !regressions.is_empty() {
            let mut detail = String::new();
            for (name, msg) in &regressions {
                detail.push_str(&format!("\n  - {name}: {msg}"));
            }
            // Report untracked body omissions too. No corresponding
            // exception-to-residual catch exists in upstream
            // ExceptionTransformer.transform_completely.
            eprintln!(
                "[mir-coverage] {} function(s) with an unrecognised MIR shape \
                 omitted; callers require a registered ABI-compatible residual; \
                 shape-coverage gap:{detail}",
                regressions.len()
            );
        }
    }
}

/// Why [`CrateLowering::build_decl`] produced no function.
pub(crate) enum DeclBuildError {
    /// The declaration carries no `Unstructured` body.
    NoBody,
    /// The body did not lower.
    Lower { name: String, error: LowerError },
}

/// A declaration's crate-stripped module path and bare leaf name. Each
/// SemanticFunction is keyed by the leaf plus a separate `module_path` so
/// `lib.rs`'s `register_function_graph_alias` walks
/// `{bare, crate::*, pyre_*::*}` correctly and the portal lookup in
/// `register_configured_jitdrivers` (`["eval_loop_jit"]`) resolves.
fn fun_decl_name_path_is_host_env(fd: &FunDecl) -> bool {
    let path = fd.item_meta.name_path();
    let segments: Vec<String> = path
        .split("::")
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect();
    if crate::flowspace::model::host_env_callable(&segments).is_some() {
        return true;
    }
    let dotted: Vec<String> = path
        .replace("::", ".")
        .split('.')
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect();
    crate::flowspace::model::host_env_callable(&dotted).is_some()
}

fn decl_module_path_and_name(fd: &FunDecl) -> (String, String) {
    let stripped = strip_crate_prefix(&fd.item_meta.name_path());
    match stripped.rsplit_once("::") {
        Some((module, leaf)) => (module.to_string(), leaf.to_string()),
        None => (String::new(), stripped),
    }
}

/// `rlib/jit.py` `ConditionalCallEntry` named by the FunDecl's real
/// path (`majit_rlib::jit::<leaf>`), or the local-crate-stripped
/// `jit::<leaf>` when `majit_rlib` is a seeded local crate.
fn fun_decl_is_jit_conditional_call(fd: &FunDecl) -> bool {
    let segs: Vec<String> = fd
        .item_meta
        .name_path()
        .split("::")
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    crate::flowspace::model::is_jit_conditional_call_path(&segs)
        || crate::local_crates::strip_alias_crate_prefix(&segs)
            .as_deref()
            .is_some_and(crate::flowspace::model::is_jit_conditional_call_path)
}

/// Harvest the crate root of a FunDecl whose body is present so
/// `free_function_alias_paths` / `canonical_dedup_key` treat a foreign
/// Transparent FunDecl the same way they treat a local one
/// (`bookkeeper.py` `Bookkeeper.getdesc` → `description.py`
/// `FunctionDesc.getuniquegraph` / `cachedgraph` → `translator.py`
/// `TranslationContext.buildflowgraph`). An Opaque FunDecl is
/// `extfunc.py` / `lloperation.py` and stays residual. Tables live on
/// the translation's `StructFieldRegistry`.
fn note_body_crate_root_into(
    fd: &FunDecl,
    fields: &mut crate::front::semantic::StructFieldRegistry,
) {
    if !fd.has_unstructured_body() {
        return;
    }
    let Some(NameSeg::Ident { ident: (root, _) }) = fd.item_meta.name.first() else {
        return;
    };
    fields.note_foreign_body_crate_root(root);
    let stripped = strip_crate_prefix(&fd.item_meta.name_path());
    fields.note_foreign_body_callable(root, &stripped);
}

fn decl_fn_path(module_path: &str, name: &str) -> String {
    if module_path.is_empty() {
        name.to_string()
    } else {
        format!("{module_path}::{name}")
    }
}

/// Declaration facts of one function or clause specialization: every
/// `SemanticFunction` field but the body, plus the `graph.func` fields
/// [`SemanticFunctionHeader::into_function`] stamps onto the lowered graph.
/// Built from the declaration alone, before the body lowers.
pub(crate) struct SemanticFunctionHeader {
    name: String,
    return_type: Option<String>,
    residual_return_type: Option<String>,
    return_class_root: Option<String>,
    return_is_str: bool,
    return_container_root: Option<String>,
    return_can_be_none: bool,
    self_ty_root: Option<String>,
    trait_impl_id: Option<u64>,
    trait_impl_identity: Option<String>,
    fun_decl_id: u64,
    module_path: String,
    hints: Vec<String>,
    trait_root: Option<String>,
    trait_qualified: Option<String>,
    returns_objectptr: bool,
    source_identity: String,
    module: Option<String>,
}

impl SemanticFunctionHeader {
    /// `name` is the bare leaf or the specialized leaf; `signature` is the
    /// declaration signature or the substituted copy. `policy_fn_path` is
    /// `module_path::<bare leaf>`, the key the hint sets use; `fn_path`
    /// stays the spec name. Both lowering loops build their header here.
    fn new(
        llbc: &Llbc,
        fd: &FunDecl,
        name: String,
        module_path: String,
        signature: &majit_charon_reader::ullbc::Signature,
        known_trait_names: &std::collections::HashSet<String>,
        dont_look_inside: &std::collections::HashSet<String>,
        elidable_residual: &std::collections::HashSet<String>,
        func_hints: &std::collections::HashMap<String, Vec<String>>,
        error_carrier: crate::ErrorCarrierSpec<'_>,
        gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
        raw_alloc_flavor_structs: &std::collections::HashSet<String>,
        policy_fn_path: &str,
    ) -> Self {
        let self_ty_root = impl_method_owner_for_fundecl(llbc, fd).map(|(owner, _)| owner);
        let trait_impl_id = trait_impl_id_for_fundecl(fd);
        let trait_impl_identity =
            trait_impl_id.and_then(|id| crate::front::clause_spec::render_trait_impl(llbc, id));
        let fn_path = if module_path.is_empty() {
            name.clone()
        } else {
            format!("{module_path}::{name}")
        };
        let source_identity = match (&self_ty_root, &trait_impl_identity) {
            (Some(owner), Some(identity)) => {
                format!("{module_path}::{owner}::<impl {identity}>::{name}")
            }
            (Some(owner), None) => format!("{module_path}::{owner}::{name}"),
            // `CallControl.graphs_from` / `Bookkeeper.getdesc` keys on the
            // function object. A non-ADT `Self` still has a trait-impl
            // identity (trait ref + Self type); keep it in the graph key
            // so two `cmp::impls::<Impl>::le` bodies do not share one
            // `FunctionDesc`.
            (None, Some(identity)) => format!("{fn_path}::<impl {identity}>"),
            (None, None) => fn_path.clone(),
        };
        // Trait-impl methods carry the trait leaf so registration calls
        // `register_trait_method`. Inherent impls leave `trait_root` empty.
        // Trait-default bodies match the parent ident against `known_trait_names`.
        let trait_qualified = trait_impl_trait_path_for_fundecl(llbc, fd);
        let trait_root = trait_qualified
            .as_ref()
            .and_then(|p| p.rsplit("::").next())
            .map(str::to_string)
            .or_else(|| trait_default_owner_for_fundecl(fd, known_trait_names));
        let returns_objectptr = output_type_is_objectptr(&signature.output, llbc);
        // `dont_look_inside` / `elidable` callees and every trait-method
        // member of an indirect-call row stamp FUNC.RESULT.
        // A `repr(transparent)` scalar wrapper is that word.
        // Aggregate `"ref"` results stay unstamped.
        // A declaration of the `BuiltinCodeFn` pointer type is a member of
        // that indirect-call family, so it stamps the pointer type's RESULT
        // for the same reason a trait-method member does.
        let stamp_return_token = dont_look_inside.contains(policy_fn_path)
            || elidable_residual.contains(policy_fn_path)
            || trait_root.is_some()
            || fun_decl_is_builtin_code_fn(fd, llbc);
        // A spec copy's own path is not in the harvested sets. Copy the
        // declaration's residual markers onto this graph so registration and
        // `look_inside_graph` keep the same status. `_jit_unroll_safe_` and
        // the other `_jit_*_` tokens are keyed by `policy_fn_path`, not by
        // the `__spec_` leaf. A loopy copy without `unroll_safe` is declined
        // and the call residualizes at a symbolic fnaddr.
        let mut hints = Vec::new();
        if policy_fn_path != fn_path {
            if dont_look_inside.contains(policy_fn_path) {
                hints.push("dont_look_inside".to_string());
            }
            if elidable_residual.contains(policy_fn_path) {
                hints.push("elidable".to_string());
            }
        }
        let signature_token = dont_look_inside_return_token(
            &signature.output,
            llbc,
            error_carrier,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
        let residual_return_type = signature_token.clone();
        let return_is_str = dont_look_inside_return_is_str(&signature.output, llbc, error_carrier);
        let return_container_root = dont_look_inside_return_container_root(
            &signature.output,
            llbc,
            error_carrier,
            gc_struct_ids,
        );
        // `dont_look_inside_return_token` spells a raw address word as
        // `i64` (`history.py` `getkind` of `Ptr`): a primitive scalar
        // pointer, or a declared-Raw ADT pointer
        // (`raw_ptr_node_is_declared_raw` / `declared_raw_adt_node`).
        // `scalar_result_token` matches that spelling. Ordinary
        // looked-inside graphs keep FUNC.RESULT unstamped so
        // `func_result_kind` follows the CFG (`flatten.py` `make_return`).
        // Residual and trait-method callees take `stamp_return_token` and
        // keep the token, so a stub matches the call result.
        let return_type = if stamp_return_token {
            signature_token
        } else if signature_token.as_deref().is_some_and(scalar_result_token)
            && !output_is_raw_address_word(
                &signature.output,
                llbc,
                error_carrier,
                gc_struct_ids,
                raw_alloc_flavor_structs,
            )
        {
            signature_token
        } else {
            None
        };
        let return_class_root = dont_look_inside_return_class_root(
            &signature.output,
            llbc,
            error_carrier,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
        let return_can_be_none = dont_look_inside_return_can_be_none(
            &signature.output,
            llbc,
            error_carrier,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
        // The funcobj's own `_jit_*_` attributes: `graph.func` carries them
        // to `look_inside_graph` and the BFS, so the first registration
        // already sees `unroll_safe`. A spec copy misses `fn_path` in the
        // harvested map; the declaration's tokens are under `policy_fn_path`.
        if dont_look_inside.contains(policy_fn_path)
            && !hints.iter().any(|hint| hint == "dont_look_inside")
        {
            hints.push("dont_look_inside".to_string());
        }
        let own_hints = func_hints
            .get(&fn_path)
            .or_else(|| func_hints.get(policy_fn_path));
        if let Some(own) = own_hints {
            for hint in own {
                if !hints.contains(hint) {
                    hints.push(hint.clone());
                }
            }
        }
        SemanticFunctionHeader {
            name,
            return_type,
            residual_return_type,
            return_class_root,
            return_is_str,
            return_container_root,
            return_can_be_none,
            self_ty_root,
            trait_impl_id,
            trait_impl_identity,
            fun_decl_id: fd.def_id,
            module_path,
            hints,
            trait_root,
            trait_qualified,
            returns_objectptr,
            source_identity,
            module: fundecl_module(fd),
        }
    }

    /// A `Clause` has no single `TraitConst` value.
    ///
    /// `resolved_trait_const_op` folds a selected impl's const to the
    /// literal `history.getkind` banks as `int`. A clause stays the
    /// `__trait_const` string, and `emit_constant` types that call `ref`.
    /// `CallControl.getcalldescr` raises when the word is passed for a
    /// `usize` parameter. Stamp `dont_look_inside` (`rlib/jit.py`
    /// `dont_look_inside` sets `_jit_look_inside_ = False`) so
    /// `policy.py` `look_inside_graph` keeps the template out of the
    /// jitcode closure. A copy that folded the const has no sentinel and
    /// stays look-inside. An explicit `jit_look_inside` wins.
    ///
    /// The eager `build_decl` path stamps the header, and `GraphStamp::apply`
    /// copies it onto the body. The on-demand path snapshots the header
    /// before the body exists (`GraphBodyProvider`), so that body is stamped
    /// by `stamp_unresolved_trait_const_residual` after it is lowered.
    fn residualize_unresolved_trait_const(&mut self, graph: &crate::model::FunctionGraph) {
        if unresolved_trait_const_needs_residual(&self.hints, graph) {
            self.hints.push("dont_look_inside".to_string());
        }
    }

    /// Stamp the header onto the lowered body and assemble the
    /// `SemanticFunction`.
    fn into_function(
        self,
        graph: crate::model::FunctionGraph,
    ) -> crate::front::semantic::SemanticFunction {
        let graph = self.graph_stamp().apply(graph);
        self.into_semantic(crate::model::LazyGraph::built(graph))
    }

    /// What the header stamps onto the funcobj's lowered body.
    pub(crate) fn graph_stamp(&self) -> GraphStamp {
        GraphStamp {
            owner_root: self.self_ty_root.clone(),
            source_identity: self.source_identity.clone(),
            fun_decl_id: self.fun_decl_id,
            module: self.module.clone(),
            hints: self.hints.clone(),
            return_type: self.return_type.clone(),
            residual_return_type: self.residual_return_type.clone(),
            return_class_root: self.return_class_root.clone(),
            return_is_str: self.return_is_str,
            return_container_root: self.return_container_root.clone(),
            return_can_be_none: self.return_can_be_none,
        }
    }

    /// Assemble the `SemanticFunction` around the funcobj's graph handle.
    pub(crate) fn into_semantic(
        self,
        graph: crate::model::LazyGraph,
    ) -> crate::front::semantic::SemanticFunction {
        crate::front::semantic::SemanticFunction {
            name: self.name,
            graph,
            return_type: self.return_type,
            return_class_root: self.return_class_root,
            return_can_be_none: self.return_can_be_none,
            self_ty_root: self.self_ty_root,
            trait_impl_id: self.trait_impl_id,
            trait_impl_identity: self.trait_impl_identity,
            fun_decl_id: Some(self.fun_decl_id),
            module_path: self.module_path,
            hints: self.hints,
            trait_root: self.trait_root,
            trait_qualified: self.trait_qualified,
            returns_objectptr: self.returns_objectptr,
        }
    }
}

/// `policy.py` `look_inside_graph` reads `_jit_look_inside_` first, so an
/// explicit hint already decides and a later `dont_look_inside` must not
/// override it.
fn look_inside_hint_is_set(hint: &str) -> bool {
    hint == "dont_look_inside" || hint.starts_with("jit_look_inside")
}

/// The lowered body still passes the unresolved `TraitConst` sentinel and
/// no `_jit_look_inside_` hint has already decided.
fn unresolved_trait_const_needs_residual(hints: &[String], graph: &FunctionGraph) -> bool {
    graph_calls_unresolved_trait_const(graph)
        && graph.func.jit_look_inside.is_none()
        && !hints.iter().any(|hint| look_inside_hint_is_set(hint))
}

/// Stamp `residualize_unresolved_trait_const` onto a body whose header
/// was snapshotted before the body existed. `policy.py`
/// `look_inside_graph` reads the hint off the built graph.
#[cfg(any(test, feature = "mir-frontend"))]
pub(crate) fn stamp_unresolved_trait_const_residual(graph: &mut FunctionGraph) {
    if unresolved_trait_const_needs_residual(&graph.hints, graph) {
        graph.push_hint("dont_look_inside");
    }
}

/// The body still passes the unresolved `TraitConst` sentinel
/// (`emit_constant`'s `Call(["__str_const", "__trait_const"])`).
fn graph_calls_unresolved_trait_const(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } = &op.kind
            else {
                return false;
            };
            segments.len() == 2 && segments[0] == "__str_const" && segments[1] == "__trait_const"
        })
    })
}

/// The header facts a funcobj's lowered body carries: `graph.func` and the
/// identity [`crate::model::FunctionGraph::graph_key`] reads.
pub(crate) struct GraphStamp {
    owner_root: Option<String>,
    source_identity: String,
    fun_decl_id: u64,
    module: Option<String>,
    hints: Vec<String>,
    return_type: Option<String>,
    residual_return_type: Option<String>,
    return_class_root: Option<String>,
    return_is_str: bool,
    return_container_root: Option<String>,
    return_can_be_none: bool,
}

impl GraphStamp {
    pub(crate) fn apply(&self, graph: crate::model::FunctionGraph) -> crate::model::FunctionGraph {
        let graph = match &self.owner_root {
            Some(owner) => graph.with_owner_root(owner.clone()),
            None => graph,
        };
        let mut graph = graph
            .with_source_identity(self.source_identity.clone())
            .with_fun_decl_id(self.fun_decl_id);
        graph.func.module = self.module.clone();
        if !self.hints.is_empty() {
            crate::front::llbc_hints::merge_hints_into_graph(&mut graph, &self.hints);
        }
        if let Some(rt) = &self.return_type {
            graph.return_type = Some(rt.clone());
        }
        graph.residual_return_type = self.residual_return_type.clone();
        graph.return_class_root = self.return_class_root.clone();
        graph.return_is_str = self.return_is_str;
        graph.return_container_root = self.return_container_root.clone();
        graph.return_can_be_none = self.return_can_be_none;
        graph
    }
}

fn should_lower_module(
    module_filter: Option<&std::collections::HashSet<String>>,
    module_path: &str,
) -> bool {
    let Some(module_filter) = module_filter else {
        return true;
    };
    if module_filter.iter().any(|root| {
        module_path == root
            || module_path
                .strip_prefix(root)
                .is_some_and(|rest| rest.starts_with("::"))
    }) {
        return true;
    }
    // A fixture's `module_paths` names the interpreter/JIT roots it wants
    // to analyze.  Keep shared helper crates/modules available because
    // opcode graphs routinely call into `pyre_object`, `error`,
    // `baseobjspace`, etc.; only skip unrequested built-in extension modules
    // under `pyre_interpreter::module::*`, whose large dispatch/register
    // bodies are not part of the JIT portal closure.
    !(module_path == "module" || module_path.starts_with("module::"))
}

/// Leaf-name allowlist companion to [`should_lower_module`].
///
/// `None` lowers every name the module gate admitted — the production
/// pipeline's shape, where the entry-point closure rather than a name list
/// decides what matters.
fn should_lower_function(
    function_filter: Option<&std::collections::HashSet<String>>,
    name: &str,
) -> bool {
    function_filter.is_none_or(|names| names.contains(name))
}

/// The `__pos_N` rows and attribute shells of the positional aggregate
/// `shape` (`Tuple<A,B>` / `Array<T;N>`), read off its spelling: one field
/// per item, as `rtuple.py TupleRepr` lays out `TUPLE_TYPE`. `None` for a
/// name that spells no shape, and for the empty tuple.
///
/// RPython makes one distinct `GcStruct('tupleN', item0, item1, ...)` per
/// [`SomeTuple`] representation when the rtyper first asks for its repr
/// (`rtuple.py`). Charon has no `TypeDecl` row for Rust's built-in
/// tuple/array aggregate, and `front::mir` spells the complete
/// `Tuple<T,...>` / `Array<T;N>` shape on the synthetic constructor and its
/// `__pos_N` fields, so every lookup that names a shape derives it from the
/// spelling. Each full shape has its own [`StructId`]
/// (`positional_shape_id`): item type and arity are part of its low-level
/// allocation identity.
pub(crate) fn positional_shape_metadata(
    shape: &str,
) -> Option<(
    Vec<crate::front::semantic::FieldRow>,
    Vec<(String, ValueType)>,
)> {
    if !majit_ir::descr::is_shaped_tuple_name(shape)
        && !majit_ir::descr::is_shaped_array_name(shape)
    {
        return None;
    }
    let items = if let Some(inner) = shape
        .strip_prefix("Tuple<")
        .and_then(|rest| rest.strip_suffix('>'))
    {
        split_top_level_type_args(inner)
    } else {
        let (item, len) = shaped_array_parts(shape)?;
        vec![item; len]
    };
    if items.is_empty() && majit_ir::descr::is_shaped_tuple_name(shape) {
        return None;
    }
    {
        // A pair-slice item is its pointer word under `__pos_N` and its
        // length word under `pair_len_field_name`.
        let pair_items: Vec<bool> = items
            .iter()
            .map(|ty| {
                majit_ir::descr::is_shaped_tuple_name(shape)
                    && (spelling_pair_field_kind(ty).is_some()
                        || majit_ir::descr::is_shared_u8_slice_spelling(ty))
            })
            .collect();
        let attrs: Vec<(String, ValueType)> = items
            .iter()
            .enumerate()
            .flat_map(|(index, ty)| {
                let name = format!("__pos_{index}");
                if pair_items[index] {
                    let len = pair_len_field_name(&name);
                    return vec![(name, ValueType::Unsigned), (len, ValueType::Unsigned)];
                }
                vec![(name, tuple_field_value_type(ty))]
            })
            .collect();
        // `TupleRepr` stores an instance-valued item as one GC pointer.  The
        // type renderer strips Rust references for nominal class lookup, so a
        // shaped tuple can arrive as `Tuple<Union,Union>` even when its two
        // SSA items are refs.  Do not then reinterpret those rows as two
        // embedded 16-byte structs: the FORCE attribute already records the
        // translated item repr as `Ref`, and the low-level tuple field must
        // use that same pointer repr.  Scalar/float/string rows keep their
        // existing specialized spelling. An inline `[T; N]` item is that
        // array's bytes (`positional_field_type`), not a pointer to it.
        let rows: Vec<crate::front::semantic::FieldRow> = items
            .iter()
            .enumerate()
            .flat_map(|(index, ty)| {
                let name = format!("__pos_{index}");
                if pair_items[index] {
                    let len = pair_len_field_name(&name);
                    return vec![
                        crate::front::semantic::FieldRow::positional(name, "usize"),
                        crate::front::semantic::FieldRow::named(len, "usize"),
                    ];
                }
                vec![crate::front::semantic::FieldRow::positional(
                    name,
                    positional_field_type(ty),
                )]
            })
            .collect();
        Some((rows, attrs))
    }
}

/// [`positional_shape_metadata`]'s rows, made once per shape for the
/// process: a pure function of the spelling, so the memo cannot change
/// what any reader sees.
pub(crate) fn positional_shape_rows(
    shape: &str,
) -> Option<&'static Vec<crate::front::semantic::FieldRow>> {
    type Rows = &'static Vec<crate::front::semantic::FieldRow>;
    static ROWS: std::sync::LazyLock<
        parking_lot::Mutex<std::collections::HashMap<String, Option<Rows>>>,
    > = std::sync::LazyLock::new(Default::default);
    if !majit_ir::descr::is_shaped_tuple_name(shape)
        && !majit_ir::descr::is_shaped_array_name(shape)
    {
        return None;
    }
    *ROWS.lock().entry(shape.to_string()).or_insert_with(|| {
        positional_shape_metadata(shape).map(|(rows, _)| &*Box::leak(Box::new(rows)))
    })
}

/// One-field GC cell `MutRef<T>` (`value: T`). The spelling is the layout,
/// the same way [`positional_shape_rows`] is: the cell is not a Charon struct,
/// so nothing registers it ahead of the first `new`.
pub(crate) fn mut_ref_shape_rows(
    name: &str,
) -> Option<&'static Vec<crate::front::semantic::FieldRow>> {
    type Rows = &'static Vec<crate::front::semantic::FieldRow>;
    static ROWS: std::sync::LazyLock<
        parking_lot::Mutex<std::collections::HashMap<String, Option<Rows>>>,
    > = std::sync::LazyLock::new(Default::default);
    if !is_mut_ref_cell_name(name) || !name.ends_with('>') {
        return None;
    }
    let inner = &name["MutRef<".len()..name.len() - 1];
    if inner.is_empty() {
        return None;
    }
    let inner = inner.to_string();
    *ROWS.lock().entry(name.to_string()).or_insert_with(|| {
        Some(&*Box::leak(Box::new(vec![
            crate::front::semantic::FieldRow::named("value", inner),
        ])))
    })
}

/// Byte size of the explicit `Result` / `Option` shell that carries
/// `field_offsets`.
///
/// The shell is a non-overlapping `[tag@0 | payload@8 | ...]` laid out by this
/// module rather than borrowed from Rust's enum layout, so its size follows
/// from the offsets recorded for it: the last field's offset plus its word.
/// A fixed 16 would truncate any shell that ever records more than one payload
/// word.  The floor keeps a tag-only shell at the full `[tag | payload]` width,
/// matching the `size.max(16)` the codewriter applies to the same shell.
/// Byte width of a Rust integer spelling, for the enum tag `TypeLayout`
/// records beside its offset (`discriminant_int_type`).  Anything unspelled
/// answers a machine word, which is the conservative direction: it widens the
/// tag's byte range and so decides more enums need the explicit sum shell.
fn int_type_byte_width(int_ty: &str) -> u64 {
    match int_ty {
        "i8" | "u8" => 1,
        "i16" | "u16" => 2,
        "i32" | "u32" => 4,
        _ => 8,
    }
}

fn sum_shell_size(field_offsets: &std::collections::HashMap<String, u64>) -> u64 {
    field_offsets
        .values()
        .copied()
        .max()
        .map_or(16, |last| (last + 8).max(16))
}

/// Split `A,B<C,D>,[E;2]` at top-level commas only.
pub(crate) fn split_top_level_type_args(input: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut prev = '\0';
    for (index, ch) in input.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            // The `>` of a `->` return arrow closes nothing; treating it as a
            // closer drops the depth a level early and splits a `Fn(A) -> B`
            // argument at the next top-level comma.
            '>' if prev == '-' => {}
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                let item = input[start..index].trim();
                if !item.is_empty() {
                    out.push(item);
                }
                start = index + ch.len_utf8();
            }
            _ => {}
        }
        prev = ch;
    }
    let tail = input[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

/// Split a shaped fixed-array owner (`Array<T;N>`) into its item spelling and
/// concrete length. The last semicolon is the outer array separator, so nested
/// array item spellings (`Array<[T;M];N>`) remain intact.
/// Split `Array<T;N>` or `[T; N]` into the item spelling and the length.
/// The last semicolon is the outer separator, so a nested item
/// (`Array<[T;M];N>`) stays intact. A slice `[T]` has no semicolon and
/// is not a shaped array.
pub(crate) fn shaped_array_parts(shape: &str) -> Option<(&str, usize)> {
    let shape = shape.trim();
    let inner = if let Some(rest) = shape.strip_prefix("Array<") {
        rest.strip_suffix('>')?
    } else {
        let rest = shape.strip_prefix('[')?;
        rest.strip_suffix(']')?
    };
    let (item, len) = inner.rsplit_once(';')?;
    let item = item.trim();
    if item.is_empty() {
        return None;
    }
    Some((item, len.trim().parse().ok()?))
}

/// Low-level field spelling of one positional `__pos_N` item.
///
/// `TupleRepr` stores an instance as one GC pointer, so a nominal item is
/// prefixed with `&`. An inline `[T; N]` is the array payload itself:
/// `get_type_flag` sizes `[u8; 4]` as four bytes, and prefixing `&` would
/// turn that item into a word-sized reference.
pub(crate) fn positional_field_type(ty: &str) -> String {
    let ty = ty.trim();
    let value_type = tuple_field_value_type(ty);
    let inline_array = ty.starts_with('[') && ty.ends_with(']');
    if matches!(value_type, ValueType::Ref(_))
        && !inline_array
        && !ty.starts_with('&')
        && !ty.starts_with("*mut ")
        && !ty.starts_with("*const ")
        && !ty.starts_with("Box<")
        && !ty.starts_with("Arc<")
        && !ty.starts_with("Rc<")
        && !ty.starts_with("Vec<")
        && !ty.starts_with("Option<")
        && !is_rust_heap_string_root(ty)
    {
        format!("&{ty}")
    } else {
        ty.to_string()
    }
}

pub(crate) fn tuple_field_value_type(type_name: &str) -> ValueType {
    let type_name = type_name.trim();
    // A one-word-item Vec is `Ptr(Struct(raw) "RustVec")`
    // (`rrustvec.rs rust_vec_lltype`); `getkind` of a raw pointer is
    // `'int'`. Same recognizer as `Bookkeeper::project_rust_vec`.
    if majit_ir::rvec::rust_vec_item_kind_for_spelling(type_name, crate::layout::target_word_size())
        .is_some()
    {
        return ValueType::Int;
    }
    // `history.py` `getkind` of a raw `Ptr` whose pointee is a non-byte
    // primitive is `'int'`. `Tuple<*mut usize, …>` intern's `__pos_0` from
    // this spelling (`rtuple.py` `TupleRepr`); a `Ref` seed makes
    // `raw_load` see `SomeInstance` (`ann_direct_ptradd` needs a pointer,
    // `rffi.py` `ptradd`). A raw `*u8` is `rffi.CCHARP` (int-banked).
    if type_name == "*mut u8" || type_name == "*const u8" {
        return ValueType::Unsigned;
    }
    if let Some(raw) = spelling_raw_scalar_address_value_type(type_name) {
        return raw;
    }
    match type_name {
        "()" => ValueType::Void,
        "f64" => ValueType::Float,
        // 128-bit fields are twice the machine word, so they carry their own
        // kind rather than collapsing into `Int` like the word-sized integers.
        "i128" => ValueType::Int128,
        "u128" => ValueType::UInt128,
        "f32" => ValueType::SingleFloat,
        "bool" | "char" | "i8" | "i16" | "i32" | "i64" | "isize" => ValueType::Int,
        // A `u*` element shells to an unsigned `SomeInteger`
        // (`valuetype_to_someshell(Unsigned)`), so an `r_uint` tuple element
        // reconciles with the FORCE_ATTRIBUTES seed instead of walling
        // `r_uint ∪ int`. Same register kind as `Int` (`getkind` = 'int').
        "u8" | "u16" | "u32" | "u64" | "usize" => ValueType::Unsigned,
        // A str-family element shells to `SomeString`
        // (`valuetype_to_someshell(Str)`) rather than the classdef-less
        // `Ref(None)` instance that walled `str ∪ Instance(classdef-less)`.
        // `Str` is register kind 'ref'/`GcRef` downstream, identical to `Ref`.
        // A Rust-heap `String` is a Vec-shaped header, not rstr `STR`.
        s if is_rust_heap_string_root(s) => ValueType::Ref(Some(s.into())),
        "str"
        | "Wtf8"
        | "Wtf8Buf"
        | "Utf8Str"
        | "UnicodeValueStorage"
        | "BytesBlock"
        | "LowlevelStringHeader" => ValueType::Str,
        _ => {
            if majit_ir::descr::is_shared_u8_slice_spelling(type_name)
                || majit_ir::descr::is_option_of_u8_window_spelling(type_name)
            {
                // Two-word fat window `{data: Address, len: Signed}`.
                // `getkind` of that raw struct is int; the descr size is
                // two words (`get_type_flag`).
                ValueType::Int
            } else {
                ValueType::Ref(None)
            }
        }
    }
}

/// `getkind` of a raw `Ptr` whose pointee is a non-byte primitive, keyed
/// off the rendered spelling [`tuple_field_value_type`] sees.
///
/// Same bank [`raw_scalar_address_value_type`] reads off a `TyRef`. A byte
/// pointer stays `None` (the caller keeps `Ref`): pyre erases a GC
/// reference to `*mut u8`. A pointer to a pointer or an ADT stays `None`.
fn spelling_raw_scalar_address_value_type(type_name: &str) -> Option<ValueType> {
    let type_name = type_name.trim();
    let pointee = type_name
        .strip_prefix("*mut ")
        .or_else(|| type_name.strip_prefix("*const "))?;
    if pointee.starts_with("*mut ") || pointee.starts_with("*const ") || pointee.starts_with('&') {
        return None;
    }
    match pointee {
        "u8" | "i8" => None,
        "bool" | "char" | "i16" | "i32" | "i64" | "isize" | "u16" | "u32" | "u64" | "usize"
        | "f32" | "f64" => Some(ValueType::Int),
        _ => None,
    }
}

/// Lowlevel field spelling the annotator publishes.
///
/// `rbigint.py` stores `_digits` as `GcArray(Signed)`. Rust stores that
/// array behind `*mut TypedItemsBlock`; the implementation pointer drops
/// the list annotation. `tupleobject.py` `FixedSizeListRepr` stores
/// `wrappeditems` as the immutable list itself (`GcArray(OBJECTPTR)`,
/// `ll_fixed_length` / `ll_fixed_items`). Rust spells the same storage
/// `*mut ItemsBlock`. Exact Charon layout still supplies the pointer-sized
/// offset; this spelling is the lowleveltype those operations key on.
/// Registry row for a Charon field: the positional origin is
/// `FieldDecl::name == None`, not the minted `__pos_N` spelling.
fn field_decl_row(
    index: usize,
    field: &FieldDecl,
    ty: impl Into<String>,
) -> crate::front::semantic::FieldRow {
    match &field.name {
        Some(name) => crate::front::semantic::FieldRow::named(name.clone(), ty),
        None => crate::front::semantic::FieldRow::positional(
            majit_charon_reader::ullbc::positional_field_name(index),
            ty,
        ),
    }
}

/// Registry row plus the Atomic* fact `tyref_atomic_arg` sees before
/// harvest spells the field as its inner scalar.
fn field_row_from_decl(
    index: usize,
    field: &FieldDecl,
    ty: impl Into<String>,
    llbc: &Llbc,
) -> crate::front::semantic::FieldRow {
    field_decl_row(index, field, ty)
        .with_atomic(lltype::tyref_atomic_arg(&field.ty, llbc).is_some())
}

fn field_decl_name(index: usize, field: &FieldDecl) -> String {
    field
        .name
        .clone()
        .unwrap_or_else(|| majit_charon_reader::ullbc::positional_field_name(index))
}

fn published_struct_field_layout(
    type_path: &str,
    field_name: &str,
    field_ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> String {
    let leaf = type_path.rsplit("::").next().unwrap_or(type_path);
    let is_rbigint = type_path == "rbigint::RBigInt" || type_path.ends_with("::rbigint::RBigInt");
    let is_tuple_object = leaf == "W_TupleObject"
        && (type_path == "tupleobject::W_TupleObject"
            || type_path.ends_with("::tupleobject::W_TupleObject"));
    if is_rbigint && field_name == "_digits" {
        "[i64]".to_string()
    } else if is_tuple_object && field_name == "wrappeditems" {
        // `rlist.py` `FixedSizeListRepr`: the tuple's `wrappeditems` is
        // `Ptr(GcArray(OBJECTPTR))`. `getitem` is `getarrayitem_gc`.
        "[*mut PyObject]".to_string()
    } else {
        annotation_struct_field_type(field_ty, llbc, gc_struct_ids, raw_alloc_flavor_structs)
    }
}

/// Type string `Bookkeeper::project_struct_field_type` consumes for a
/// declared field. A `Vec` of one-word items is a canonical header-pointer
/// spelling `project_rust_vec` accepts. An integer item keeps its own
/// spelling (`usize` / `u64` versus `isize` / `i64`) so that projector can
/// set `SomeInteger.unsigned`. [`tyref_to_field_layout_string`] writes
/// `isize`, which annotates as `SomeInteger` and then `len` of the field
/// raises `'int' has no length`.
fn annotation_struct_field_type(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> String {
    match tyref_rust_vec_item_kind(ty, llbc) {
        Some(majit_ir::rvec::VecItemKind::Int) => {
            let item =
                tyref_rust_vec_item_spelling(ty, llbc).unwrap_or_else(|| "usize".to_string());
            format!("*mut Vec<{item}>")
        }
        Some(majit_ir::rvec::VecItemKind::Ref) => "*mut Vec<*mut u8>".to_string(),
        Some(majit_ir::rvec::VecItemKind::Float) => "*mut Vec<f64>".to_string(),
        None => {
            let layout =
                tyref_to_field_layout_string(ty, llbc, gc_struct_ids, raw_alloc_flavor_structs);
            field_row_struct_identity(ty, llbc, &layout).unwrap_or(layout)
        }
    }
}

/// A fixed-list GcArray: `[ *mut PyObject ]` (published)
/// [`published_struct_field_layout`]) or `FixedObjectArray`. Both are
/// `ll_fixed_length` / `arraylen_gc` under [`OBJECT_REF_GCARRAY_TYPE_ID`].
/// An `ItemsBlock` header pointer is the resizable list's `ll_items` and
/// is not this type.
fn type_spelling_is_fixed_object_array(spelling: &str) -> bool {
    let normalized = peel_ref_prefix(spelling);
    if normalized.starts_with('[')
        && slice_array_type_id(normalized).as_deref() == Some(OBJECT_REF_GCARRAY_TYPE_ID)
    {
        return true;
    }
    let mut bare = normalized;
    loop {
        let next = bare
            .strip_prefix("*const ")
            .or_else(|| bare.strip_prefix("*mut "))
            .or_else(|| bare.strip_prefix("&mut "))
            .or_else(|| bare.strip_prefix('&'))
            .map(str::trim_start);
        match next {
            Some(rest) if rest != bare => bare = rest,
            _ => break,
        }
    }
    bare.rsplit("::").next().unwrap_or(bare) == "FixedObjectArray"
}

/// Derive whole-program type-metadata fields of `SemanticProgram` from
/// Charon's `type_decls` + `trait_decls` tables.
///
/// Returns `(known_struct_names, known_trait_names, struct_fields,
/// enum_variant_by_discriminant, struct_origins, struct_field_attrs,
/// exact_layouts)`.
/// Names are taken from `item_meta.name_path()`; struct field rows
/// resolve their type string via [`tyref_to_ast_string`] (Charon-resolved
/// types: references stripped, raw pointers kept, `Vec<T>` / `[T;N]`
/// generics, named structs by leaf).  `struct_origins` maps a bare
/// struct leaf to its defining module path with the crate prefix
/// stripped (so the value matches the runtime def-path convention).
/// `struct_field_attrs` maps the crate-stripped qualified struct name to
/// its declaration-ordered `(field, ValueType)` register classes.
/// Record a `name → StructId` mapping in the resolver table, collapsing
/// to `None` (ambiguous) if the same name already mapped to a different
/// identity — i.e. two distinct type definitions share that spelling
/// (only possible for a bare leaf across modules).
fn record_struct_id(
    table: &mut std::collections::HashMap<String, Option<majit_ir::descr::StructId>>,
    name: String,
    id: majit_ir::descr::StructId,
) {
    table
        .entry(name)
        .and_modify(|slot| {
            if *slot != Some(id) {
                *slot = None;
            }
        })
        .or_insert(Some(id));
}

/// Publish a `type T = Struct<…>` alias as its own layout identity.
///
/// Charon records the monomorph's rustc field offsets on the alias
/// TypeDecl, while the generic struct declaration keeps symbolic
/// `OffsetOf`. A field read of the alias must use those offsets.
fn register_type_alias_struct(
    td: &TypeDecl,
    llbc: &Llbc,
    target: &str,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
    exact_layouts: &mut std::collections::HashMap<
        majit_ir::descr::StructId,
        crate::front::semantic::ExactLayout,
    >,
    struct_ids: &mut std::collections::HashMap<String, Option<majit_ir::descr::StructId>>,
    struct_origins: &mut std::collections::HashMap<String, String>,
    struct_field_attrs: &mut std::collections::HashMap<String, Vec<(String, ValueType)>>,
    known_struct_names: &mut std::collections::HashSet<String>,
) {
    let Some(fields) = type_decl_struct_fields(td, llbc) else {
        return;
    };
    if fields.is_empty() {
        return;
    }
    let Some(layout) = td.layout_for_target(llbc, target) else {
        return;
    };
    let name = td.item_meta.name_path();
    let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
    let canonical_name = strip_crate_prefix(&name);
    let rows: Vec<crate::front::semantic::FieldRow> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let fname = field_decl_name(i, f);
            let field_ty = published_struct_field_layout(
                &name,
                &fname,
                &f.ty,
                llbc,
                gc_struct_ids,
                raw_alloc_flavor_structs,
            );
            field_row_from_decl(i, f, field_ty, llbc)
        })
        .collect();
    struct_fields.fields.insert(name.clone(), rows.clone());
    struct_fields
        .fields
        .insert(canonical_name.clone(), rows.clone());
    let sid = majit_ir::descr::StructId::from_canonical(&canonical_name);
    record_struct_id(struct_ids, name.clone(), sid);
    record_struct_id(struct_ids, canonical_name.clone(), sid);
    // A later Struct/Enum with the same leaf keeps that leaf. Poisoning the
    // resolver slot would drop both layouts; skip the alias leaf instead.
    let leaf_free = match struct_ids.get(&leaf) {
        None => true,
        Some(existing) => *existing == Some(sid),
    };
    if leaf_free {
        struct_fields.fields.insert(leaf.clone(), rows);
        record_struct_id(struct_ids, leaf.clone(), sid);
    }
    let mut field_offsets = std::collections::HashMap::new();
    for (i, f) in fields.iter().enumerate() {
        let fname = f
            .name
            .clone()
            .unwrap_or_else(|| majit_charon_reader::ullbc::positional_field_name(i));
        let Some(off) = layout.struct_field_offset(i) else {
            continue;
        };
        field_offsets.insert(fname, off);
    }
    if field_offsets.is_empty() {
        return;
    }
    exact_layouts.insert(
        sid,
        crate::front::semantic::ExactLayout {
            size: layout.size,
            align: layout.align,
            field_offsets,
            host: Some(crate::front::host_layout::host_layout_from_type(&layout)),
        },
    );
    let segs: Vec<&str> = name.split("::").collect();
    let module = if segs.len() >= 2 {
        segs[1..segs.len() - 1].join("::")
    } else {
        String::new()
    };
    struct_origins.entry(leaf.clone()).or_insert(module);
    let attr_rows: Vec<(String, ValueType)> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let fname = field_decl_name(i, f);
            let attr_ty = tyref_to_attr_value_type_for_struct_field(
                &f.ty,
                td,
                llbc,
                gc_struct_ids,
                raw_alloc_flavor_structs,
            );
            (fname, attr_ty)
        })
        .collect();
    struct_field_attrs.insert(canonical_name.clone(), attr_rows);
    known_struct_names.insert(name);
    known_struct_names.insert(canonical_name);
    if leaf_free {
        known_struct_names.insert(leaf);
    }
}

/// Record every spelling of an ADT whose fields are raw words.
/// `GcKind::Raw` also covers classed structs that are not GC headers
/// (a strategy singleton, a type object). Those stay instances.
fn note_raw_word_owner(
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
    td: &TypeDecl,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
) {
    let mut stack = Vec::new();
    if !adt_def_is_raw_storage(td.def_id, llbc, gc_struct_ids, immutable_fields, &mut stack) {
        return;
    }
    let name = td.item_meta.name_path();
    let leaf = name
        .rsplit("::")
        .next()
        .unwrap_or(name.as_str())
        .to_string();
    // Closure envs that only capture Copy words are raw storage, but they
    // are not C structs. `raw_struct_ptr_annotation` must not mint SomePtr
    // for them (`tyref_input_class_root`; RPython forbids closures).
    if majit_charon_reader::ullbc::is_closure_leaf(&leaf) {
        return;
    }
    struct_fields
        .raw_word_owners
        .insert(strip_crate_prefix(&name));
    struct_fields.raw_word_owners.insert(leaf);
    struct_fields.raw_word_owners.insert(name);
}

#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
fn derive_program_metadata(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> (
    std::collections::HashSet<String>,
    std::collections::HashSet<String>,
    crate::front::semantic::StructFieldRegistry,
    std::collections::HashMap<String, std::collections::HashMap<i64, String>>,
    std::collections::HashMap<String, String>,
    std::collections::HashMap<String, Vec<(String, ValueType)>>,
    std::collections::HashMap<majit_ir::descr::StructId, crate::front::semantic::ExactLayout>,
    std::collections::HashMap<String, Option<majit_ir::descr::StructId>>,
) {
    // Charon resolves layout per target; the LLBC carries a single
    // entry for the extraction target (build-script `TARGET`).  An
    // absent/non-matching target falls back to the sole entry inside
    // `layout_for_target`, so an empty value still resolves it.
    let target = std::env::var("TARGET").unwrap_or_default();
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = crate::front::semantic::StructFieldRegistry::default();
    let mut enum_variant_by_discriminant: std::collections::HashMap<
        String,
        std::collections::HashMap<i64, String>,
    > = std::collections::HashMap::new();
    let mut struct_origins: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut exact_layouts: std::collections::HashMap<
        majit_ir::descr::StructId,
        crate::front::semantic::ExactLayout,
    > = std::collections::HashMap::new();
    // name (any spelling) → canonical StructId; `None` marks a bare leaf
    // two distinct modules share.  Inserts go through `record_struct_id`
    // below so a cross-module bare-leaf clash collapses to `None`.
    let mut struct_ids: std::collections::HashMap<String, Option<majit_ir::descr::StructId>> =
        std::collections::HashMap::new();
    let mut struct_field_attrs: std::collections::HashMap<String, Vec<(String, ValueType)>> =
        std::collections::HashMap::new();

    for td in llbc.iter_type_decls() {
        if type_decl_is_builtin_adt(td) {
            continue;
        }
        // A copied import is not a second registration of the defining
        // crate's class. First-writer field rows and later exact layouts
        // would otherwise overwrite that class.
        if td.published_struct_copy {
            continue;
        }
        if matches!(td.kind, TypeDeclKind::Struct(_) | TypeDeclKind::Enum(_)) {
            note_raw_word_owner(
                &mut struct_fields,
                td,
                llbc,
                gc_struct_ids,
                immutable_fields,
            );
        }
        let name = td.item_meta.name_path();
        match &td.kind {
            TypeDeclKind::Struct(fields) => {
                // Register the qualified path *and* the bare leaf name
                // so downstream lookups (`canonical_call_target`'s
                // bare-leaf fallback) resolve either spelling.
                let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
                // A transparent GC handle's sole payload is a pointer to
                // `Deref::Target` (`PyError.0` is `*mut PyErrorObject`).
                // The declared field type is `PyObjectRef` (`*mut PyObject`);
                // projecting that class has no common annotator base with
                // the handle, which subclasses the object class.
                let owner_ty = type_decl_adt_tyref(td.def_id);
                let handle_payload = transparent_handle_field_spelling(
                    &owner_ty,
                    llbc,
                    no_tombstoned_leaves(),
                    gc_struct_ids,
                    raw_alloc_flavor_structs,
                );
                let handle_field_index = handle_payload
                    .as_ref()
                    .and_then(|_| transparent_nonzst_field(td, llbc).map(|(i, _)| i));
                let handle_attr = handle_payload.as_ref().and_then(|_| {
                    transparent_deref_target_class(
                        &owner_ty,
                        llbc,
                        no_tombstoned_leaves(),
                        gc_struct_ids,
                        raw_alloc_flavor_structs,
                    )
                    .map(|root| ValueType::Ref(Some(root)))
                });
                // A by-value transparent newtype is the inner struct
                // (`ModuleDictStorage` / `RDict`; `celldict.py` `erase({})`).
                // Flatten those leaves onto this class so getattr of
                // `num_live_items` resolves (`rordereddict.rtype_len`).
                let inner_fields = handle_payload
                    .is_none()
                    .then(|| transparent_by_value_inner_fields(td, llbc))
                    .flatten();
                let row_fields: &[FieldDecl] = inner_fields
                    .map(|(_, inner)| inner)
                    .unwrap_or(fields.as_slice());
                let layout_td = inner_fields.map(|(inner, _)| inner).unwrap_or(td);
                // A pair-slice field is its pointer word under the field's
                // own name and its length word under `pair_len_field_name`.
                let rows: Vec<crate::front::semantic::FieldRow> = row_fields
                    .iter()
                    .enumerate()
                    .flat_map(|(i, f)| {
                        let fname = field_decl_name(i, f);
                        if tyref_pair_field_kind(&f.ty, llbc).is_some() {
                            let len = pair_len_field_name(&fname);
                            return vec![
                                field_decl_row(i, f, "usize"),
                                crate::front::semantic::FieldRow::named(len, "usize"),
                            ];
                        }
                        let layout = match (handle_field_index == Some(i), handle_payload.as_ref())
                        {
                            (true, Some(spelling)) => spelling.clone(),
                            _ => published_struct_field_layout(
                                &name,
                                &fname,
                                &f.ty,
                                llbc,
                                gc_struct_ids,
                                raw_alloc_flavor_structs,
                            ),
                        };
                        let field_ty =
                            crate::virtualizable_decl::overlay_virtualizable_llfield_layout(
                                &strip_crate_prefix(&name),
                                &fname,
                                layout,
                            );
                        vec![field_row_from_decl(i, f, field_ty, llbc)]
                    })
                    .collect();
                let canonical_name = strip_crate_prefix(&name);
                struct_fields.fields.insert(name.clone(), rows.clone());
                struct_fields
                    .fields
                    .insert(canonical_name.clone(), rows.clone());
                struct_fields.fields.insert(leaf.clone(), rows);
                // Object identity for this struct type, minted from the
                // crate-stripped qualified path — the spelling the descr
                // layer keys `LLType::Struct` on and the runtime macro
                // publishes.  Every name spelling (full-crate, stripped,
                // bare leaf) resolves to it through the resolver table.
                let sid = majit_ir::descr::StructId::from_canonical(&canonical_name);
                record_struct_id(&mut struct_ids, name.clone(), sid);
                record_struct_id(&mut struct_ids, canonical_name.clone(), sid);
                record_struct_id(&mut struct_ids, leaf.clone(), sid);
                // Exact field byte offsets from Charon's resolved layout
                // (the true Rust layout, not the heuristic).  Field index
                // `i` matches the declaration order Charon offsets by.
                // Absent layout (opaque type) leaves the type unrecorded;
                // the heuristic provider covers it.  Keyed by the identity
                // token — one entry per type definition.
                if let Some(layout) = layout_td.layout_for_target(llbc, &target) {
                    let mut field_offsets = std::collections::HashMap::new();
                    for (i, f) in row_fields.iter().enumerate() {
                        let fname = f.name.clone().unwrap_or_else(|| {
                            majit_charon_reader::ullbc::positional_field_name(i)
                        });
                        let Some(off) = layout.struct_field_offset(i) else {
                            continue;
                        };
                        if tyref_pair_field_kind(&f.ty, llbc).is_some() {
                            let words = crate::vec_layout::slice_probe();
                            field_offsets
                                .insert(pair_len_field_name(&fname), off + words.len_offset as u64);
                            field_offsets.insert(fname, off + words.ptr_offset as u64);
                        } else {
                            field_offsets.insert(fname, off);
                        }
                    }
                    let exact = crate::front::semantic::ExactLayout {
                        size: layout.size,
                        align: layout.align,
                        field_offsets,
                        host: Some(crate::front::host_layout::host_layout_from_type(&layout)),
                    };
                    exact_layouts.insert(sid, exact);
                }
                // `bytecode::CodeObject` is `repr(Rust)` and Charon leaves
                // its offsets unresolved. A real layout, even an empty
                // one from a later extraction, wins; the probe fills only
                // a missing entry. Host `offset_of` is not applied when
                // the target word disagrees with this process.
                let unresolved = exact_layouts
                    .get(&sid)
                    .is_none_or(|exact| exact.field_offsets.is_empty());
                if unresolved
                    && let Some(exact) = crate::codeobject_layout::exact_layout_for(&canonical_name)
                {
                    exact_layouts.insert(sid, exact);
                }
                // `bare leaf → crate-relative module`: drop the crate
                // prefix (first segment) and the leaf (last segment) so
                // the value matches the runtime def-path
                // (`intobject::W_IntObject` ← `pyre_object::intobject::
                // W_IntObject`).  First-write-wins on duplicate leaves
                // defined in distinct modules; the loser's qualified
                // `name` key still resolves through the dual-publish.
                let segs: Vec<&str> = name.split("::").collect();
                let module = if segs.len() >= 2 {
                    segs[1..segs.len() - 1].join("::")
                } else {
                    String::new()
                };
                struct_origins.entry(leaf.clone()).or_insert(module);
                // Register-class rows for `FORCE_ATTRIBUTES_INTO_CLASSES`,
                // keyed by the crate-stripped defining path. This is the
                // closest Rust-side stand-in for RPython's class-object key:
                // same-leaf structs in distinct modules stay distinct, and
                // the spelling matches the def-path convention used by
                // `STRUCT_ORIGIN_REGISTRY`.
                let attr_rows: Vec<(String, ValueType)> = row_fields
                    .iter()
                    .enumerate()
                    .flat_map(|(i, f)| {
                        let fname = f.name.clone().unwrap_or_else(|| {
                            majit_charon_reader::ullbc::positional_field_name(i)
                        });
                        if tyref_pair_field_kind(&f.ty, llbc).is_some() {
                            let len = pair_len_field_name(&fname);
                            return vec![(fname, ValueType::Unsigned), (len, ValueType::Unsigned)];
                        }
                        let attr_ty = if handle_field_index == Some(i) {
                            handle_attr.clone().unwrap_or_else(|| {
                                tyref_to_attr_value_type_for_struct_field(
                                    &f.ty,
                                    td,
                                    llbc,
                                    gc_struct_ids,
                                    raw_alloc_flavor_structs,
                                )
                            })
                        } else {
                            tyref_to_attr_value_type_for_struct_field(
                                &f.ty,
                                layout_td,
                                llbc,
                                gc_struct_ids,
                                raw_alloc_flavor_structs,
                            )
                        };
                        vec![(
                            fname.clone(),
                            crate::virtualizable_decl::overlay_virtualizable_llfield_value_type(
                                &canonical_name,
                                &fname,
                                attr_ty,
                            ),
                        )]
                    })
                    .collect();
                struct_field_attrs.insert(canonical_name.clone(), attr_rows);
                known_struct_names.insert(name);
                known_struct_names.insert(canonical_name);
                known_struct_names.insert(leaf);
            }
            TypeDeclKind::Enum(variants) => {
                // Enums register under their type name *and* under each
                // variant path (`Strategy::Empty`, `Strategy::IntKeyed`,
                // …) so a synthetic Aggregate(SyntheticTransparentCtor)
                // can be matched downstream.
                let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
                // The crate-stripped `module::Enum` spelling — the
                // canonical key `canonical_struct_name(enum_leaf)` resolves
                // to (`STRUCT_ORIGIN_REGISTRY` prepends the same
                // crate-stripped module path).  The annotator interns the
                // base and its variants under this spelling
                // (`intern_enum_variant_host`), so registering rows /
                // attrs here lets `project_struct_rows` and the FORCE
                // table find them on the one class object the narrowing
                // and the constructor share.
                let canon_base = strip_crate_prefix(&name);
                // Register the bare-leaf → crate-relative module origin, the
                // same way the struct arm does above.  `intern_enum_variant_host`
                // relies on `canonical_struct_name(leaf)` resolving to the
                // `module::Enum` spelling so the constructor side (owner tail =
                // bare leaf) and the discriminant-narrowing side (the base
                // classdef's canonical name) normalise to one cache key; without
                // the origin the bare leaf passes through unchanged and the two
                // sides mint sibling classdefs for one Rust enum.  First-write-
                // wins on duplicate leaves; `harden_duplicate_leaf_metadata`
                // empties the entry when distinct modules collide.
                let module = canon_base
                    .rsplit_once("::")
                    .map(|(m, _)| m.to_string())
                    .unwrap_or_default();
                struct_origins.entry(leaf.clone()).or_insert(module);
                known_struct_names.insert(name.clone());
                known_struct_names.insert(leaf.clone());
                // Register the enum base class in `struct_fields` with
                // only the synthetic `__discriminant` tag — the sum-type
                // base carries the discriminant, each variant subclass
                // carries its OWN payload fields (`rclass.py:499-518`,
                // registered below under `{enum}::{variant}` keys and
                // projected onto the variant classdef by
                // `getuniqueclassdef_for_enum_variant`).
                // `Rvalue::Discriminant` lowers to
                // `FieldRead("__discriminant")` on the base; a payload
                // projection resolves on the narrowed `SomeInstance(variant)`
                // receiver (`enum_variant_narrowing_knowntypedata`) and
                // reads at the variant-qualified runtime offset
                // (`resolve_adt_field` owner_root = `{enum_leaf}::{variant}`).
                // `__discriminant` width.  For a FIELDLESS enum the tag IS
                // the whole value, so model it at the enum's true byte size
                // — the heuristic layout then sizes an inline field of this
                // enum exactly (`W_ListObject.strategy` = 1 byte for a
                // `repr(u8)` `ListStrategy`), so a discriminant read of the
                // field reads only the tag, not 7 trailing padding bytes.  A
                // data-carrying enum's `layout.size` includes the variant
                // payload, which is not the tag width, so keep the wide
                // `i64` model there (the tag still sits in the low bytes; a
                // switch key compares equal under the wide read).
                // The fieldless tag is modeled at an UNSIGNED width (u8/u16/u32).
                // This is correct only while every variant discriminant is
                // non-negative: a signed-repr / negative-discriminant enum would
                // zero-extend the tag on read (e.g. `-1` → `255`) while Charon's
                // variant key stays the negative `discriminant_i64()`, so the
                // JIT switch/guard would take the wrong arm. All fieldless enums
                // lowered here today (ListStrategy / ArrayKind / DictViewKind /
                // StrategyKind / ExcKind …) use the default non-negative
                // discriminants, so unsigned is exact. A negative-discriminant
                // enum would need a signed width here AND a sign-extending field
                // read; revisit if one is introduced.
                // A FIELDLESS (C-like) enum is modeled by-value as its
                // discriminant integer (`tyref_to_value_type` colors it
                // `Int`; `Rvalue::Discriminant` / `discriminant_value` alias
                // that int; the constructor folds to `ConstInt`), so it has
                // no `SomeInstance` base and needs no `__discriminant`-only
                // base ClassDef.  Skip the base + variant-subclass
                // registration entirely for it (only the payload-enum sum
                // type needs the base tag carrier + per-variant subclasses).
                // The inline-field read (`self.strategy`) sizes off the
                // enum's OWN Charon layout (`inline_fieldless_enum_field_tag0`
                // / the container field descr), not this registration, and
                // `enum_variant_by_discriminant` (kept below) supplies the
                // switch-arm discr→name mapping — both independent of the
                // base ClassDef.
                let fieldless = type_decl_is_fieldless_enum(td, llbc);
                let enum_layout = td.layout_for_target(llbc, &target);
                let base_sid = majit_ir::descr::StructId::from_canonical(&canon_base);
                // The tag is an int-bank value, but its physical width is
                // not necessarily a machine word. Charon records the exact
                // `Branch.int_ty`; using `i64` for a u8/u16/u32 tag reads
                // adjacent payload bytes and can select an unreachable switch
                // arm.  `i64` remains the model only where NO tag width was
                // recorded at all.  A tag that was recorded and declined is a
                // width no field type spells (`I128`/`U128`), and modelling it
                // as `i64` would name half of it: register no base at all
                // there, so a discriminant read declines instead of resolving
                // to a descriptor that reads eight of sixteen bytes.
                let discr_ty = enum_layout
                    .as_ref()
                    .and_then(|layout| layout.discriminant_int_type());
                let tag_recorded_but_unspellable = discr_ty.is_none()
                    && enum_layout
                        .as_ref()
                        .is_some_and(|layout| layout.has_branch_discriminant());
                if !fieldless && !tag_recorded_but_unspellable {
                    let discr_ty = discr_ty.unwrap_or("i64").to_string();
                    let rows: Vec<(String, String)> =
                        vec![("__discriminant".to_string(), discr_ty)];
                    struct_fields.fields.insert(name.clone(), rows.clone());
                    struct_fields.fields.insert(leaf.clone(), rows.clone());
                    struct_fields.fields.insert(canon_base.clone(), rows);
                    // Object identity for the enum base (the discriminant
                    // carrier), minted from the crate-stripped `module::Enum`.
                    record_struct_id(&mut struct_ids, name.clone(), base_sid);
                    record_struct_id(&mut struct_ids, leaf.clone(), base_sid);
                    record_struct_id(&mut struct_ids, canon_base.clone(), base_sid);
                }
                // Per-variant field rows + exact offsets under
                // `{enum}::{variant}` keys — the RPython sum-type subclass
                // layout where each variant carries its OWN fields, the base
                // only the discriminant (`rclass.py:499-518`).  These feed the
                // variant subclasses (their `getuniqueclassdef_for_enum_variant`
                // attr projection) and the per-variant runtime offsets.
                // Dual-published under the qualified and bare-leaf spellings
                // because `resolve_adt_field` emits `owner_root` =
                // `{enum_leaf}::{variant}`.  No cross-variant dedup: each
                // variant owns its field namespace.
                // `Result<T, E>` and `Option<T>` are not materialised as
                // Rust's native enums in translated code.  The inverse
                // exception transform removes ordinary `?` paths; a
                // hand-written `match` that remains, and every
                // runtime-discriminant pair `emit_tagged_pair_aggregate`
                // synthesises (`checked_neg`, `usize::try_from`,
                // `i64::try_from(&RBigInt)`), are represented by the explicit
                // rtyper shell emitted below:
                // `{ __discriminant: Signed, __pos_0: payload }`.  Rust's
                // niche layout commonly places both the implicit tag and the
                // payload at offset 0, but applying that host layout to this
                // explicit shell makes the payload store overwrite the tag
                // and every following switch take its unreachable default.
                // RPython's low-level representation is the explicit fields
                // the graph declares, so give that synthetic struct its own
                // non-overlapping layout instead of borrowing Rust's enum
                // layout.
                //
                // `Option` needs it for the same reason and did not have it:
                // `Option<i64>` registered `__discriminant` and
                // `Some.__pos_0` both at offset 0 of an 8-byte parent, so a
                // descent into any body holding a `checked_neg` read `-v` as
                // the tag and aborted on the match's unreachable arm.  A
                // pointer-niche `Option` is diverted to a bare nullable
                // pointer before any constructor is emitted
                // (`tyref_is_niche_option_ptr`) and never allocates, and the
                // base tag stays at offset 0 either way, so the shell only
                // moves the payload of an Option this front end actually
                // builds.
                //
                // Those two names were the cases this shell was first written
                // for, but the condition they share is physical rather than
                // nominal: whenever the host layout seats a variant's payload
                // field on the tag's own bytes, the constructor's payload store
                // overwrites the tag it just wrote.  Asking the layout catches
                // every such enum instead of the two that were noticed.
                //
                // `pyre_interpreter::pyopcode::StepResult` — what every opcode
                // handler returns — is one of them.  Charon gave `CloseLoop`'s
                // payload field and `Return.__pos_0` offset 0, the tag's own
                // offset, so `close_loop`'s jitcode stamped discriminant 2 and
                // then overwrote it with the payload; `execute_jump_backward`
                // read the null back as tag 0 and reported `Continue`, and the
                // portal took its `Continue` arm on every back edge.  A walk of
                // the portal could therefore never reach `loop_header`, and no
                // portal trace could close.
                // The shell is needed exactly when the base registered a
                // `__discriminant` row that the variant subclass inherits at
                // byte 0 (`rclass.py:499-518`): the payload must then start
                // after it.  `tag_recorded_but_unspellable` registers no base
                // row at all, so a payload at 0 there aliases nothing.
                let base_has_discriminant = !fieldless && !tag_recorded_but_unspellable;
                // An unrecorded tag width reads as a machine word: the wider
                // guess only moves more enums onto the explicit shell, which
                // is the representation that cannot alias in the first place.
                let tag_offset = enum_layout
                    .as_ref()
                    .and_then(|l| l.discriminant_offset())
                    .unwrap_or(0);
                let tag_size = enum_layout
                    .as_ref()
                    .and_then(|l| l.discriminant_int_type())
                    .map_or(8, int_type_byte_width);
                let payload_seats_on_tag = match enum_layout.as_ref() {
                    // Charon recorded no layout for this decl — the case that
                    // hid this bug.  The variant offsets then fall back to
                    // default packing from byte 0, which is the inherited
                    // discriminant's byte, on every payload enum in the set.
                    None => true,
                    Some(l) => variants.iter().enumerate().any(|(vidx, v)| {
                        (0..v.fields.len()).any(|i| {
                            l.field_offset(vidx, i)
                                .is_some_and(|off| off >= tag_offset && off < tag_offset + tag_size)
                        })
                    }),
                };
                let explicit_sum_shell = name == "core::result::Result"
                    || name == "core::option::Option"
                    || (base_has_discriminant && payload_seats_on_tag);
                // Register the enum BASE in `exact_layouts`: a single
                // `__discriminant` field at the tag's real byte position
                // (`discriminator.Branch.offset` via `discriminant_offset`).
                // `Rvalue::Discriminant` lowers to a `FieldRead("__discriminant")`
                // keyed to the base identity, so a niche/non-zero tag resolves
                // at its true offset instead of the heuristic 0.  A tag at
                // offset 0 (the common case) registers 0, matching the
                // heuristic exactly; a single-variant type has no `Branch`
                // tag (`discriminant_offset` → `None`) and also registers 0.
                // Fieldless enums skip this (int-valued, no base ClassDef).
                let host = enum_layout
                    .as_ref()
                    .map(crate::front::host_layout::host_layout_from_type);
                if !fieldless {
                    if explicit_sum_shell {
                        let mut base_offsets = std::collections::HashMap::new();
                        base_offsets.insert("__discriminant".to_string(), 0);
                        exact_layouts.insert(
                            base_sid,
                            crate::front::semantic::ExactLayout {
                                size: Some(sum_shell_size(&base_offsets)),
                                align: Some(8),
                                field_offsets: base_offsets,
                                host: host.clone(),
                            },
                        );
                    } else if let Some(l) = enum_layout.as_ref() {
                        let mut base_offsets = std::collections::HashMap::new();
                        base_offsets.insert(
                            "__discriminant".to_string(),
                            l.discriminant_offset().unwrap_or(0),
                        );
                        let base_exact = crate::front::semantic::ExactLayout {
                            size: l.size,
                            align: l.align,
                            field_offsets: base_offsets,
                            host: host.clone(),
                        };
                        exact_layouts.insert(base_sid, base_exact);
                    }
                }
                // Per-variant subclasses carry each variant's OWN payload
                // fields; a fieldless enum has none, so it needs no variant
                // subclass registration.  Only the payload-enum sum type
                // reaches here.
                if !fieldless {
                    for (vidx, v) in variants.iter().enumerate() {
                        let variant_qual = format!("{name}::{}", v.name);
                        let variant_leaf = format!("{leaf}::{}", v.name);
                        let variant_canon = format!("{canon_base}::{}", v.name);
                        let mut vrows: Vec<crate::front::semantic::FieldRow> =
                            Vec::with_capacity(v.fields.len());
                        let mut vattrs: Vec<(String, ValueType)> =
                            Vec::with_capacity(v.fields.len());
                        let mut voffsets: std::collections::HashMap<String, u64> =
                            std::collections::HashMap::new();
                        // `rclass.InstanceRepr._setup_repr` declares a subclass as
                        // `MkStruct(name, ('super', rbase.object_type), *llfields)`, so
                        // `heaptracker.all_fielddescrs` walks the embedded base FIRST and
                        // an inherited field keeps the same index in the subclass as in
                        // the base.  A shelled variant inherits exactly one field — the
                        // base's synthetic `__discriminant` at byte 0 — so register it as
                        // this subclass's own first row.  Without it the payload row takes
                        // slot 0, the same slot the tag's own descr claims, and a virtual's
                        // payload store replaces the tag store it is keyed against.
                        // A payload-less variant is skipped: a lone `__discriminant` row is
                        // the sentinel `StructFieldRegistry::is_enum_base` reads to tell a
                        // base from a subclass.
                        if explicit_sum_shell && !v.fields.is_empty() {
                            vrows.push(crate::front::semantic::FieldRow::named(
                                "__discriminant",
                                discr_ty.unwrap_or("i64"),
                            ));
                            voffsets.insert("__discriminant".to_string(), 0);
                        }
                        // The shell's next free payload word.
                        let mut shell_word = 0u64;
                        for (i, f) in v.fields.iter().enumerate() {
                            let fname = field_decl_name(i, f);
                            // A pair-slice payload is its pointer and length
                            // words, the length under `pair_len_field_name`.
                            if tyref_pair_field_kind(&f.ty, llbc).is_some() {
                                let len = pair_len_field_name(&fname);
                                let (ptr_off, len_off) = if explicit_sum_shell {
                                    shell_word += 2;
                                    (
                                        Some(8 + (shell_word - 2) * 8),
                                        Some(8 + (shell_word - 1) * 8),
                                    )
                                } else {
                                    let words = crate::vec_layout::slice_probe();
                                    let off =
                                        enum_layout.as_ref().and_then(|l| l.field_offset(vidx, i));
                                    (
                                        off.map(|off| off + words.ptr_offset as u64),
                                        off.map(|off| off + words.len_offset as u64),
                                    )
                                };
                                if let Some(off) = ptr_off {
                                    voffsets.insert(fname.clone(), off);
                                }
                                if let Some(off) = len_off {
                                    voffsets.insert(len.clone(), off);
                                }
                                vattrs.push((fname.clone(), ValueType::Unsigned));
                                vattrs.push((len.clone(), ValueType::Unsigned));
                                vrows.push(field_decl_row(i, f, "usize"));
                                vrows.push(crate::front::semantic::FieldRow::named(len, "usize"));
                                continue;
                            }
                            // A bytecode-arg marker reads as a `u32` at runtime
                            // and annotates as an integer — keep the row string
                            // and the FORCE attr type in agreement so the
                            // projection refines the same shell.
                            let (row_ty, attr_ty) = if tyref_is_bytecode_arg_marker(&f.ty, llbc) {
                                ("u32".to_string(), ValueType::Int)
                            } else if tyref_is_type_parameter(&f.ty, llbc) {
                                (
                                    tyref_to_field_layout_string(
                                        &f.ty,
                                        llbc,
                                        gc_struct_ids,
                                        raw_alloc_flavor_structs,
                                    ),
                                    ValueType::Unknown,
                                )
                            } else {
                                (
                                    annotation_struct_field_type(
                                        &f.ty,
                                        llbc,
                                        gc_struct_ids,
                                        raw_alloc_flavor_structs,
                                    ),
                                    tyref_to_attr_value_type_with(
                                        &f.ty,
                                        llbc,
                                        no_tombstoned_leaves(),
                                        gc_struct_ids,
                                        raw_alloc_flavor_structs,
                                    ),
                                )
                            };
                            let field_offset = if explicit_sum_shell {
                                shell_word += 1;
                                Some(8 + (shell_word - 1) * 8)
                            } else {
                                enum_layout.as_ref().and_then(|l| l.field_offset(vidx, i))
                            };
                            if let Some(off) = field_offset {
                                voffsets.insert(fname.clone(), off);
                            }
                            vattrs.push((fname.clone(), attr_ty));
                            vrows.push(field_row_from_decl(i, f, row_ty, llbc));
                        }
                        struct_fields
                            .fields
                            .insert(variant_qual.clone(), vrows.clone());
                        struct_fields
                            .fields
                            .insert(variant_leaf.clone(), vrows.clone());
                        struct_fields.fields.insert(variant_canon.clone(), vrows);
                        // Object identity for the variant subclass, minted from
                        // the crate-stripped `module::Enum::Variant`.
                        let vsid = majit_ir::descr::StructId::from_canonical(&variant_canon);
                        record_struct_id(&mut struct_ids, variant_qual.clone(), vsid);
                        record_struct_id(&mut struct_ids, variant_leaf.clone(), vsid);
                        record_struct_id(&mut struct_ids, variant_canon.clone(), vsid);
                        if explicit_sum_shell {
                            let exact = crate::front::semantic::ExactLayout {
                                size: Some(sum_shell_size(&voffsets)),
                                align: Some(8),
                                field_offsets: voffsets,
                                host: host.clone(),
                            };
                            exact_layouts.insert(vsid, exact);
                        } else if let Some(l) = enum_layout.as_ref() {
                            let exact = crate::front::semantic::ExactLayout {
                                size: l.size,
                                align: l.align,
                                field_offsets: voffsets,
                                host: host.clone(),
                            };
                            exact_layouts.insert(vsid, exact);
                        }
                        // Variant payload attrs for `FORCE_ATTRIBUTES_INTO_CLASSES`,
                        // keyed by the canonical `module::Enum::Variant` — the
                        // key `_init_classdef` derives for the variant classdef
                        // whether the narrowing or the constructor minted it, so
                        // the payload attrs are forced onto the one variant
                        // class.  Mirrors the struct arm above.
                        struct_field_attrs.insert(variant_canon, vattrs);
                    }
                }
                // discriminant → variant name, published under both the
                // qualified path and the bare leaf so the opcode-dispatch
                // extractor can resolve by either spelling.
                let mut by_discr: std::collections::HashMap<i64, String> =
                    std::collections::HashMap::new();
                for v in variants {
                    let variant_path = format!("{name}::{}", v.name);
                    known_struct_names.insert(variant_path.clone());
                    if let Some(d) = v.discriminant_i64() {
                        by_discr.insert(d, v.name.clone());
                    }
                }
                if !by_discr.is_empty() {
                    enum_variant_by_discriminant.insert(name.clone(), by_discr.clone());
                    enum_variant_by_discriminant.insert(leaf, by_discr);
                }
            }
            // A union's fields all start at offset 0 and only one is live,
            // so there is no field-offset row to project; the declarations
            // that reach here are foreign (`windows-sys`' `IN_ADDR_0`).
            TypeDeclKind::Alias(_) => {
                // Charon keeps `type T = Struct<…>` as its own TypeDecl with
                // the monomorph's rustc layout. Generic struct offsets are
                // symbolic `OffsetOf`. Publish the alias under its own StructId.
                register_type_alias_struct(
                    td,
                    llbc,
                    &target,
                    gc_struct_ids,
                    raw_alloc_flavor_structs,
                    &mut struct_fields,
                    &mut exact_layouts,
                    &mut struct_ids,
                    &mut struct_origins,
                    &mut struct_field_attrs,
                    &mut known_struct_names,
                );
            }
            TypeDeclKind::Union(_) | TypeDeclKind::Opaque | TypeDeclKind::Unknown => {}
        }
    }

    // RustPython compiler-core declares this exact source type as
    // `pub struct Constants<C: Constant>(Box<[C]>)`.  It is a dependency
    // `Opaque` declaration in `pyre-interpreter.ullbc`, so Charon cannot
    // provide the field row even though MIR contains the statically resolved
    // `Deref`/`Index` impl calls.  Supply the literal upstream row for the one
    // concrete interpreter instantiation (`C = ConstantData`) so the typed
    // `Constants.__pos_0` FieldRead annotates as a list.  State remains on the
    // wrapper object itself; this is metadata for its real field, not a side
    // table or replacement container.
    let compiler_constants_path = "rustpython_compiler_core::bytecode::Constants";
    if llbc
        .iter_type_decls()
        .any(|td| td.item_meta.name_path() == compiler_constants_path)
    {
        let rows = vec![crate::front::semantic::FieldRow::positional(
            "__pos_0",
            "Box<[ConstantData]>",
        )];
        for key in [compiler_constants_path, "bytecode::Constants"] {
            struct_fields.fields.insert(key.to_string(), rows.clone());
            known_struct_names.insert(key.to_string());
        }
        struct_fields
            .fields
            .entry("Constants".to_string())
            .or_insert_with(|| rows.clone());
        known_struct_names.insert("Constants".to_string());
        let sid = majit_ir::descr::StructId::from_canonical("bytecode::Constants");
        for key in [compiler_constants_path, "bytecode::Constants", "Constants"] {
            record_struct_id(&mut struct_ids, key.to_string(), sid);
        }
        struct_origins
            .entry("Constants".to_string())
            .or_insert_with(|| "bytecode".to_string());
        struct_field_attrs.insert(
            "bytecode::Constants".to_string(),
            vec![("__pos_0".to_string(), ValueType::Ref(None))],
        );
    }

    // `num_complex::Complex<T>` is dependency-opaque in the interpreter
    // snapshot, but compiler-core's `ConstantData::Complex` payload stores
    // the concrete `Complex<f64>` and `load_const_value` reads its public
    // `re`/`im` fields.  Restore the literal upstream `#[repr(C)] struct
    // Complex<T> { pub re: T, pub im: T }` rows for that one instantiation.
    // As with `Constants` above, this describes fields on the real value; it
    // introduces no parallel storage.  Gate it on the exact dependency type
    // declaration so an unrelated same-named local type cannot acquire this
    // shape.
    let num_complex_path = "num_complex::Complex";
    let num_complex_canon = strip_crate_prefix(num_complex_path);
    if llbc
        .iter_type_decls()
        .any(|td| td.item_meta.name_path() == num_complex_path)
    {
        let rows = vec![
            ("re".to_string(), "f64".to_string()),
            ("im".to_string(), "f64".to_string()),
        ];
        for key in [num_complex_path, "Complex"] {
            struct_fields.fields.insert(key.to_string(), rows.clone());
            known_struct_names.insert(key.to_string());
        }
        let sid = majit_ir::descr::StructId::from_canonical(&num_complex_canon);
        for key in [num_complex_path, "Complex"] {
            record_struct_id(&mut struct_ids, key.to_string(), sid);
        }
        struct_origins.insert("Complex".to_string(), "num_complex".to_string());
        struct_field_attrs.insert(
            num_complex_canon,
            vec![
                ("re".to_string(), ValueType::Float),
                ("im".to_string(), ValueType::Float),
            ],
        );
    }

    // `alloc::vec::Vec<T>` is Opaque in the interpreter snapshot. Field
    // reads of `buf` / `len` name the Vec value; `fielddescrof` then adds
    // the component offset (`vec_part`). Record both names at byte 0 so
    // the layout lookup hits before `getattr` panics, matching the
    // previous Unresolved mint of offset 0 plus `vec_part`.
    let alloc_vec_path = "alloc::vec::Vec";
    if llbc
        .iter_type_decls()
        .any(|td| td.item_meta.name_path() == alloc_vec_path)
    {
        let rows = vec![
            crate::front::semantic::FieldRow::named("buf", "*mut u8"),
            crate::front::semantic::FieldRow::named("len", "usize"),
        ];
        for key in [alloc_vec_path, "Vec"] {
            struct_fields.fields.insert(key.to_string(), rows.clone());
            known_struct_names.insert(key.to_string());
        }
        let sid = majit_ir::descr::StructId::from_canonical(alloc_vec_path);
        for key in [alloc_vec_path, "Vec"] {
            record_struct_id(&mut struct_ids, key.to_string(), sid);
        }
        let word = crate::layout::target_word_size() as u64;
        let mut field_offsets = std::collections::HashMap::new();
        field_offsets.insert("buf".to_string(), 0);
        field_offsets.insert("len".to_string(), 0);
        exact_layouts.insert(
            sid,
            crate::front::semantic::ExactLayout {
                size: Some(3 * word),
                align: Some(word),
                field_offsets,
                host: None,
            },
        );
    }

    // `core::alloc::layout::Layout` is Opaque. It is the two-word
    // `{size, align}` aggregate (`from_size_align.rs`); Charon has no
    // field rows. `__pos_0` is size at 0, `__pos_1` is align at one word.
    let layout_path = "core::alloc::layout::Layout";
    if llbc
        .iter_type_decls()
        .any(|td| td.item_meta.name_path() == layout_path)
    {
        let rows = vec![
            crate::front::semantic::FieldRow::positional("__pos_0", "usize"),
            crate::front::semantic::FieldRow::positional("__pos_1", "usize"),
        ];
        for key in [layout_path, "alloc::layout::Layout", "Layout"] {
            struct_fields.fields.insert(key.to_string(), rows.clone());
            known_struct_names.insert(key.to_string());
        }
        let sid = majit_ir::descr::StructId::from_canonical("alloc::layout::Layout");
        for key in [layout_path, "alloc::layout::Layout", "Layout"] {
            record_struct_id(&mut struct_ids, key.to_string(), sid);
        }
        let word = crate::layout::target_word_size() as u64;
        let mut field_offsets = std::collections::HashMap::new();
        field_offsets.insert("__pos_0".to_string(), 0);
        field_offsets.insert("__pos_1".to_string(), word);
        exact_layouts.insert(
            sid,
            crate::front::semantic::ExactLayout {
                size: Some(2 * word),
                align: Some(word),
                field_offsets,
                host: None,
            },
        );
    }

    for td in llbc.iter_trait_decls() {
        let name = td.item_meta.name_path();
        let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
        known_trait_names.insert(name);
        known_trait_names.insert(leaf);
    }

    (
        known_struct_names,
        known_trait_names,
        struct_fields,
        enum_variant_by_discriminant,
        struct_origins,
        struct_field_attrs,
        exact_layouts,
        struct_ids,
    )
}

/// Seeds and field-0 edges used to derive `Struct._gckind`.
///
/// A type is `Gc` when an impl of majit-gc's `GcType` names it as `Self`,
/// when it is field 0 (by value) of a `Gc` type, or when its own field 0
/// is a `Gc` type. The last two clauses are a fixpoint:
/// `Struct._note_inlined_into` allows a `GcStruct` only as the first
/// field of another `GcStruct`. The trait is identified by the
/// declaration's crate and name (`majit_gc` + `GcType`), not by a list
/// of payload type names. The facts are folded into
/// [`crate::call::StructLayout::gckind`]; they are not a lookup table.
#[derive(Default)]
pub(crate) struct DeclaredGcFacts {
    seeds: Vec<majit_ir::descr::StructId>,
    field0: Vec<(majit_ir::descr::StructId, majit_ir::descr::StructId)>,
}

impl DeclaredGcFacts {
    pub(crate) fn absorb(&mut self, other: Self) {
        self.seeds.extend(other.seeds);
        self.field0.extend(other.field0);
    }

    /// `StructId`s whose declared kind is `Gc`.
    ///
    /// (a) a majit-gc `GcType` impl, (b) field 0 of a `Gc` type, (c) a
    /// type whose field 0 is `Gc`. (b) and (c) run to a fixpoint over
    /// the field-0 edges. Every other analysed struct or enum is `Raw`.
    pub(crate) fn gc_struct_ids(&self) -> std::collections::HashSet<majit_ir::descr::StructId> {
        let mut children: std::collections::HashMap<
            majit_ir::descr::StructId,
            Vec<majit_ir::descr::StructId>,
        > = std::collections::HashMap::new();
        let mut parents: std::collections::HashMap<
            majit_ir::descr::StructId,
            Vec<majit_ir::descr::StructId>,
        > = std::collections::HashMap::new();
        for (owner, field0) in &self.field0 {
            children.entry(*owner).or_default().push(*field0);
            parents.entry(*field0).or_default().push(*owner);
        }
        let mut gc = std::collections::HashSet::new();
        let mut work = self.seeds.clone();
        while let Some(id) = work.pop() {
            if !gc.insert(id) {
                continue;
            }
            if let Some(inner) = children.get(&id) {
                work.extend(inner.iter().copied());
            }
            if let Some(owners) = parents.get(&id) {
                work.extend(owners.iter().copied());
            }
        }
        gc
    }
}

/// Whether `sid` is a declared GC struct: the exact identity, or the generic
/// template `name` instantiates.
///
/// Harvest seeds the template `StructId` of a generic `GcType` impl.
/// `struct_id_for_name` instantiates names that carry `<...>`, so a layout
/// key misses the seed unless this fallback runs. `history.py` `getkind` of
/// `Ptr(TO)` follows `TO._gckind`.
pub(crate) fn struct_id_is_declared_gc(
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    sid: majit_ir::descr::StructId,
    name: &str,
) -> bool {
    gc_struct_ids.contains(&sid)
        || majit_ir::descr::struct_template_id_for_name(name)
            .is_some_and(|template| gc_struct_ids.contains(&template))
}

fn struct_id_for_type_decl(td: &TypeDecl) -> majit_ir::descr::StructId {
    majit_ir::descr::StructId::from_canonical(&strip_crate_prefix(&td.item_meta.name_path()))
}

/// `true` when `path` is majit-gc's `GcType` declaration.
///
/// The crate is the first `name_path` segment and the name is the last,
/// so `majit_gc::header::GcType` and a root re-export's
/// `majit_gc::GcType` both match. `other_crate::GcType` does not.
fn trait_path_is_majit_gc_gctype(path: &str) -> bool {
    let crate_name = path.split("::").next();
    let name = path.rsplit("::").next();
    crate_name == Some("majit_gc") && name == Some("GcType")
}

/// By-value field 0 of a struct, when that field is itself a struct.
///
/// `lltype.py` `Struct._first_struct` returns the first field only when
/// `isinstance(FIRSTTYPE, Struct)` and `self._gckind == FIRSTTYPE._gckind`.
/// That is the only parent relation (`castable`, `_upcast`). An Enum is
/// never a parent. A reference or raw pointer is not the inlined parent
/// (`Struct.__init__` inlines a container only as the first field). `Ref`
/// and `RawPtr` stay in place here; only an `Adt` node yields an id.
fn struct_field0_type_id(llbc: &Llbc, td: &TypeDecl) -> Option<majit_ir::descr::StructId> {
    if td.published_struct_copy {
        return None;
    }
    let TypeDeclKind::Struct(fields) = &td.kind else {
        return None;
    };
    let field = fields.first()?;
    let node = strip_ty_indirections(tyref_node(&field.ty, llbc)?, llbc)?;
    if node.get("Ref").is_some() || node.get("RawPtr").is_some() {
        return None;
    }
    let inner = llbc.type_by_id(adt_node_def_id(node)?)?;
    if inner.published_struct_copy {
        return None;
    }
    match &inner.kind {
        TypeDeclKind::Struct(_) => Some(struct_id_for_type_decl(inner)),
        _ => None,
    }
}

pub(crate) fn harvest_declared_gc_facts(llbc: &Llbc) -> DeclaredGcFacts {
    let mut facts = DeclaredGcFacts::default();
    for row in llbc.trait_impls_raw() {
        let Some(impl_trait) = row.get("impl_trait") else {
            continue;
        };
        let Some(trait_id) = impl_trait.get("id").and_then(serde_json::Value::as_u64) else {
            continue;
        };
        let Some(decl) = llbc.trait_by_id(trait_id) else {
            continue;
        };
        if !trait_path_is_majit_gc_gctype(&decl.item_meta.name_path()) {
            continue;
        }
        let Some(self_ty) = impl_trait
            .get("generics")
            .and_then(|generics| generics.get("types"))
            .and_then(serde_json::Value::as_array)
            .and_then(|types| types.first())
        else {
            continue;
        };
        let Some(def_id) = resolve_tyexpr_to_adt_def_id_free(llbc, self_ty) else {
            continue;
        };
        let Some(td) = llbc.type_by_id(def_id) else {
            continue;
        };
        facts.seeds.push(struct_id_for_type_decl(td));
    }
    for td in llbc.iter_type_decls() {
        if type_decl_is_builtin_adt(td) {
            continue;
        }
        let Some(field0) = struct_field0_type_id(llbc, td) else {
            continue;
        };
        facts.field0.push((struct_id_for_type_decl(td), field0));
    }
    facts
}

/// Declared `GcKind` of every struct and enum in `llbcs`, keyed by the
/// crate-stripped type path. `Gc` follows [`DeclaredGcFacts::gc_struct_ids`]
/// (the field-0 fixpoint); every other analysed struct or enum is `Raw`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn declared_gckind_by_name(
    llbcs: &[Llbc],
) -> std::collections::HashMap<String, crate::translator::rtyper::lltypesystem::lltype::GcKind> {
    use crate::translator::rtyper::lltypesystem::lltype::GcKind;
    let mut facts = DeclaredGcFacts::default();
    for llbc in llbcs {
        facts.absorb(harvest_declared_gc_facts(llbc));
    }
    let gc = facts.gc_struct_ids();
    let mut out = std::collections::HashMap::new();
    for llbc in llbcs {
        for td in llbc.iter_type_decls() {
            if type_decl_is_builtin_adt(td) {
                continue;
            }
            if td.published_struct_copy {
                continue;
            }
            if !matches!(td.kind, TypeDeclKind::Struct(_) | TypeDeclKind::Enum(_)) {
                continue;
            }
            let canon = strip_crate_prefix(&td.item_meta.name_path());
            let kind = if struct_id_is_declared_gc(&gc, struct_id_for_type_decl(td), &canon) {
                GcKind::Gc
            } else {
                GcKind::Raw
            };
            out.insert(canon, kind);
        }
    }
    out
}

/// Per-LLBC fragment of the duplicate-leaf verdict. The streaming frontend
/// discovers one of these per artefact (the same pre-link shape as
/// [`discover_foldable_const_lits`]), merges them, and only then lowers.
#[derive(Default)]
pub(crate) struct DuplicateLeafFacts {
    struct_fields: crate::front::semantic::StructFieldRegistry,
    enum_variant_by_discriminant:
        std::collections::HashMap<String, std::collections::HashMap<i64, String>>,
    struct_origins: std::collections::HashMap<String, String>,
    struct_ids: std::collections::HashMap<String, Option<majit_ir::descr::StructId>>,
    /// Crate-stripped path → defining crate, for a declaration whose full
    /// path starts with this artefact's crate name. `a::Foo` and `b::Foo`
    /// both land on `Foo`, and `StructId::from_canonical` mints one id
    /// from that spelling, so the identity table sees one declaration.
    defined_at: std::collections::HashMap<String, String>,
    /// Leaves withdrawn because one stripped path was defined in two crates.
    cross_crate_leaves: std::collections::HashSet<String>,
    /// Those stripped paths themselves (`same::Code` from `a::same::Code`
    /// and `b::same::Code`). A leaf tombstone still paints the
    /// crate-stripped spelling; a path in this set keeps the crate.
    cross_crate_stripped: std::collections::HashSet<String>,
}

impl DuplicateLeafFacts {
    pub(crate) fn discover(llbc: &Llbc) -> Self {
        let (_, _, struct_fields, enum_variant_by_discriminant, struct_origins, _, _, struct_ids) =
            derive_program_metadata(
                llbc,
                &harvest_declared_gc_facts(llbc).gc_struct_ids(),
                no_immutable_fields(),
                &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(
                    std::slice::from_ref(llbc),
                ),
            );
        let crate_name = llbc.crate_name().to_string();
        let prefix = format!("{crate_name}::");
        let mut defined_at = std::collections::HashMap::new();
        for key in struct_ids.keys() {
            if let Some(rest) = key.strip_prefix(&prefix) {
                defined_at.insert(rest.to_string(), crate_name.clone());
            }
        }
        Self {
            struct_fields,
            enum_variant_by_discriminant,
            struct_origins,
            struct_ids,
            defined_at,
            cross_crate_leaves: std::collections::HashSet::new(),
            cross_crate_stripped: std::collections::HashSet::new(),
        }
    }

    pub(crate) fn absorb(&mut self, other: Self) {
        for (key, fields) in other.struct_fields.fields {
            self.struct_fields.fields.entry(key).or_insert(fields);
        }
        self.struct_fields
            .raw_word_owners
            .extend(other.struct_fields.raw_word_owners);
        for (enum_key, by_discr) in other.enum_variant_by_discriminant {
            self.enum_variant_by_discriminant
                .entry(enum_key)
                .or_insert(by_discr);
        }
        for (leaf, module) in other.struct_origins {
            self.struct_origins.entry(leaf).or_insert(module);
        }
        for (key, id) in other.struct_ids {
            self.struct_ids
                .entry(key)
                .and_modify(|slot| {
                    if *slot != id {
                        *slot = None;
                    }
                })
                .or_insert(id);
        }
        // Same stripped path, two defining crates: the identity table
        // minted both from that path, so harden sees one declaration.
        // A copy of the same full path inside another artefact does not
        // match that artefact's crate prefix, so it is not a second
        // definition.
        self.cross_crate_leaves.extend(other.cross_crate_leaves);
        self.cross_crate_stripped.extend(other.cross_crate_stripped);
        for (stripped, crate_name) in other.defined_at {
            if let Some(prev) = self.defined_at.get(&stripped)
                && prev != &crate_name
            {
                let leaf = stripped.rsplit("::").next().unwrap_or(stripped.as_str());
                self.cross_crate_leaves.insert(leaf.to_string());
                self.cross_crate_stripped.insert(stripped.clone());
            }
            self.defined_at.entry(stripped).or_insert(crate_name);
        }
    }

    /// Leaves whose origin [`harden_duplicate_leaf_metadata`] withdraws on
    /// this merged fragment. An empty origin module is not enough: a
    /// crate-root declaration stores one without being a duplicate.
    pub(crate) fn tombstoned_leaves(mut self) -> std::collections::HashSet<String> {
        let mut leaves = harden_duplicate_leaf_metadata(
            &mut self.struct_fields,
            &mut self.struct_origins,
            &mut self.enum_variant_by_discriminant,
            Some(&self.struct_ids),
        );
        leaves.extend(self.cross_crate_leaves);
        // Colliding stripped paths travel with the leaf set so every paint
        // site can tell `same::Code` (keep the crate) from a single-crate
        // tombstone (keep the stripped spelling). A one-segment path is
        // already the leaf.
        leaves.extend(
            self.cross_crate_stripped
                .into_iter()
                .filter(|path| path.contains("::")),
        );
        leaves
    }
}

/// Crate-stripped path, or the full declaration path when that stripped
/// spelling is a cross-crate collision carried in `tombstoned`.
fn decl_path_for_tombstone(
    name_path: &str,
    tombstoned: &std::collections::HashSet<String>,
) -> String {
    let stripped = strip_crate_prefix(name_path);
    if stripped.contains("::") && tombstoned.contains(&stripped) {
        name_path.to_string()
    } else {
        stripped
    }
}

/// Move rows whose crate-stripped key collides across crates onto the
/// full declaration path. The stripped key is dropped so a later
/// `or_insert` merge cannot keep one crate's layout for both.
fn promote_cross_crate_stripped_keys(
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    known_struct_names: &mut std::collections::HashSet<String>,
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
    struct_field_attrs: &mut std::collections::HashMap<String, Vec<(String, ValueType)>>,
    exact_layouts: &mut std::collections::HashMap<
        majit_ir::descr::StructId,
        crate::front::semantic::ExactLayout,
    >,
    struct_ids: &mut std::collections::HashMap<String, Option<majit_ir::descr::StructId>>,
) {
    if tombstoned.is_empty() {
        return;
    }
    for td in llbc.iter_type_decls() {
        let name = td.item_meta.name_path();
        let stripped = strip_crate_prefix(&name);
        if !stripped.contains("::") || !tombstoned.contains(&stripped) || stripped == name {
            continue;
        }
        let old_sid = majit_ir::descr::StructId::from_canonical(&stripped);
        let new_sid = majit_ir::descr::StructId::from_canonical(&name);
        if let Some(layout) = exact_layouts.remove(&old_sid) {
            exact_layouts.insert(new_sid, layout);
        }
        if let Some(rows) = struct_field_attrs.remove(&stripped) {
            struct_field_attrs.entry(name.clone()).or_insert(rows);
        }
        struct_fields.remove_field(&stripped);
        known_struct_names.remove(&stripped);
        struct_ids.insert(name.clone(), Some(new_sid));
        struct_ids.remove(&stripped);
        let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
        if struct_ids.get(&leaf) == Some(&Some(old_sid)) {
            struct_ids.insert(leaf.clone(), Some(new_sid));
        }
        let TypeDeclKind::Enum(variants) = &td.kind else {
            continue;
        };
        for v in variants {
            let old_key = format!("{stripped}::{}", v.name);
            let new_key = format!("{name}::{}", v.name);
            let old_vsid = majit_ir::descr::StructId::from_canonical(&old_key);
            let new_vsid = majit_ir::descr::StructId::from_canonical(&new_key);
            if let Some(layout) = exact_layouts.remove(&old_vsid) {
                exact_layouts.insert(new_vsid, layout);
            }
            if let Some(rows) = struct_field_attrs.remove(&old_key) {
                struct_field_attrs.insert(new_key.clone(), rows);
            }
            if let Some(rows) = struct_fields.remove_field(&old_key) {
                struct_fields.fields.insert(new_key.clone(), rows);
            }
            struct_ids.insert(new_key, Some(new_vsid));
            struct_ids.remove(&old_key);
            let variant_leaf = format!("{leaf}::{}", v.name);
            if struct_ids.get(&variant_leaf) == Some(&Some(old_vsid)) {
                struct_ids.insert(variant_leaf, Some(new_vsid));
            }
        }
    }
}

/// Owner spelling for a leaf [`harden_duplicate_leaf_metadata`] withdrew.
/// The crate-stripped path is kept, except when stripping yields the leaf
/// or that stripped path itself collides across crates — both paint the
/// full declaration path.
fn tombstoned_owner_spelling(
    name_path: &str,
    leaf: &str,
    tombstoned: &std::collections::HashSet<String>,
) -> String {
    let qualified = strip_crate_prefix(name_path);
    if qualified != leaf && tombstoned.contains(&qualified) {
        name_path.to_string()
    } else if qualified != leaf {
        qualified
    } else {
        name_path.to_string()
    }
}

/// Class identity of a type declaration: the crate-stripped
/// `module::Type` (`bookkeeper.py Bookkeeper.getdesc` keys one class
/// object per declaration; the bare leaf is display-only). A crate-root
/// declaration strips to its leaf. A withdrawn duplicate leaf uses
/// [`tombstoned_owner_spelling`].
///
/// `name_path` is always a full `item_meta.name_path()`. Never call this
/// on an already crate-stripped string (`option::Option` would collapse
/// to `Option`).
pub(crate) fn class_identity_spelling(
    name_path: &str,
    tombstoned: &std::collections::HashSet<String>,
) -> String {
    let leaf = name_path.rsplit("::").next().unwrap_or(name_path);
    if tombstoned.contains(leaf) {
        tombstoned_owner_spelling(name_path, leaf, tombstoned)
    } else {
        strip_crate_prefix(name_path)
    }
}

/// Instantiated enum class identity: [`class_identity_spelling`] plus a
/// `<…>` suffix. Harvested `struct_fields` (`derive_program_metadata`
/// `canon_base`) and recast dest (`resolve_aggregate_adt` variant owner)
/// share this spelling (`bookkeeper.py` `getuniqueclassdef(cls)`).
pub(crate) fn enum_ctor_class_identity(
    name_path: &str,
    suffix: Option<&str>,
    tombstoned: &std::collections::HashSet<String>,
) -> String {
    let identity = class_identity_spelling(name_path, tombstoned);
    match suffix {
        Some(s) if !s.is_empty() => majit_ir::descr::with_instantiation_suffix(&identity, s),
        _ => identity,
    }
}

/// Tombstones for one LLBC, the set [`LowerContext::new`] paints with.
pub(crate) fn tombstoned_leaves_of(llbc: &Llbc) -> std::collections::HashSet<String> {
    DuplicateLeafFacts::discover(llbc).tombstoned_leaves()
}

/// Withdraw the bare-leaf convenience aliases for struct leaves shared
/// by two or more distinct type declarations.
///
/// The dual-publish in [`derive_program_metadata`] makes the bare-leaf
/// channels silent-winner maps on a duplicate leaf — and the two
/// winners disagree (`struct_fields.fields` insert = last-decl-wins,
/// `struct_origins` `or_insert` = first-decl-wins), so a bare lookup
/// could denote one struct while `canonical_struct_name` names another.
/// RPython cannot express this state at all: every classdef and
/// `FORCE_ATTRIBUTES_INTO_CLASSES` key is a live class OBJECT
/// (bookkeeper.py:361, classdesc.py), so a name is never an  allow-line-citation
/// identity.  The string-carrier stand-in therefore keeps leaf
/// resolution only while it is injective:
///
/// - `struct_fields.fields`: the bare alias is removed when the
///   colliding declarations disagree on field shape (equal-shape
///   duplicates keep the alias — any winner answers field-type lookups
///   identically).  Lookups then miss and fall to the qualified key or
///   fail conservatively (`SomeValue::Impossible` / residual call),
///   mirroring `MirGraphLookup::insert_or_mark_ambiguous`.
/// - `struct_origins`: the entry is emptied when the colliding
///   declarations live in different crate-stripped modules, which
///   `canonical_struct_name` (descr.rs) already treats as
///   unresolvable — the bare spelling passes through unchanged instead
///   of canonicalising to whichever module registered first.
/// - variant-leaf aliases: an enum variant dual-publishes a bare
///   `Enum::Variant` 2-segment convenience alias alongside its qualified
///   spellings.  The bare struct-leaf pass keys on the LAST segment
///   (`Variant`), which mixes distinct enums' same-named variants and
///   drops a `Variant` key that was never published, so a cross-module
///   `Enum::Variant` collision survives.  A dedicated pass groups by the
///   `Enum::Variant` tail and withdraws that alias off the same
///   field-shape-divergence signal.
///
/// The exact-layout channel is no longer hardened here: it is keyed by
/// `StructId` object identity, so two distinct definitions sharing a leaf
/// are distinct entries by construction and there is no bare alias to
/// withdraw.
///
/// Derived from the current qualified (`::`-containing) keys after collapsing
/// their full/canonical spellings through the existing `StructId` object
/// identity.  The pass is therefore idempotent and safe to re-run after the
/// cross-LLBC merge re-introduces a per-crate-unique alias.
pub(crate) fn harden_duplicate_leaf_metadata(
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
    struct_origins: &mut std::collections::HashMap<String, String>,
    enum_variant_by_discriminant: &mut std::collections::HashMap<
        String,
        std::collections::HashMap<i64, String>,
    >,
    struct_ids: Option<&std::collections::HashMap<String, Option<majit_ir::descr::StructId>>>,
) -> std::collections::HashSet<String> {
    // RPython groups aliases by the live class object, never by spelling.
    // `StructId` is pyre's existing object-identity carrier: discard a
    // crate-included/canonical duplicate before comparing declaration rows,
    // and discard a resolver-tombstoned alias (`None`) because it denotes no
    // unique declaration.  Tests that exercise this helper in isolation pass
    // no identity table and retain the original spelling-based behavior.
    fn unique_declarations<'a>(
        keys: &[&'a str],
        struct_ids: Option<&std::collections::HashMap<String, Option<majit_ir::descr::StructId>>>,
    ) -> Vec<&'a str> {
        let mut seen = std::collections::HashSet::new();
        let mut unique = Vec::with_capacity(keys.len());
        for key in keys {
            match struct_ids.and_then(|ids| ids.get(*key)) {
                Some(Some(id)) if seen.insert(*id) => unique.push(*key),
                Some(Some(_)) | Some(None) => {}
                None => unique.push(*key),
            }
        }
        unique
    }
    // The same parent path is tested once per key. `is_enum_base` is pure
    // for the duration of this loop: rows are not mutated until the
    // withdrawals below.
    fn cached_enum_base(
        reg: &crate::front::semantic::StructFieldRegistry,
        owner: &str,
        memo: &mut std::collections::HashMap<String, bool>,
    ) -> bool {
        if let Some(&known) = memo.get(owner) {
            return known;
        }
        let known = reg.is_enum_base(owner);
        memo.insert(owner.to_string(), known);
        known
    }
    let mut enum_base_memo: std::collections::HashMap<String, bool> =
        std::collections::HashMap::new();
    let mut by_leaf: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
    for key in struct_fields.fields.keys() {
        if let Some((head, leaf)) = key.rsplit_once("::") {
            // A `{Enum}::{Variant}` key whose variant leaf ALSO names an
            // enum base is not a peer type-leaf of that base: its bare
            // alias is withdrawn by the variant-leaf pass below (keyed on
            // the `Enum::Variant` tail), so admitting it into this
            // type-leaf bucket lets the variant (`typedef::BytesSubArg::Buffer`,
            // rows `[__pos_0]`) withdraw the same-named enum type's bare
            // alias (`buffer::Buffer`, rows `[__discriminant]`) on a
            // spurious row divergence.  Restricting the skip to a leaf that
            // is itself an enum base withdraws only the genuine
            // type-vs-variant name collision, leaving every other
            // variant-leaf bucket (whose withdrawal the resume numbering
            // depends on) intact.
            //
            // The same holds for a variant leaf that names a struct: a
            // variant publishes only `{Enum}::{Variant}` spellings, so a
            // bare `leaf` key was published by the type declaration alone
            // and denotes that one class.  `SpaceCacheInstance::SysState`
            // (rows `[__pos_0]`) otherwise withdraws and tombstones
            // `module::sys::state::SysState`'s bare alias, and the
            // `ObjSpace.sys_state: SysState` row projects to no class.
            if cached_enum_base(struct_fields, head, &mut enum_base_memo)
                && (cached_enum_base(struct_fields, leaf, &mut enum_base_memo)
                    || struct_fields.fields.contains_key(leaf))
            {
                continue;
            }
            by_leaf.entry(leaf).or_default().push(key);
        }
    }
    let mut drop_field_aliases: Vec<String> = Vec::new();
    let mut tombstone_origins: Vec<String> = Vec::new();
    for (leaf, aliases) in &by_leaf {
        let quals = unique_declarations(aliases, struct_ids);
        if quals.len() < 2 {
            continue;
        }
        let first_rows = &struct_fields.fields[quals[0]];
        if quals[1..]
            .iter()
            .any(|q| &struct_fields.fields[*q] != first_rows)
        {
            drop_field_aliases.push((*leaf).to_string());
        }
        // Two surviving identity-backed keys are necessarily two distinct
        // declarations.  The fallback module comparison preserves the
        // isolated helper tests and metadata without a resolver table.
        let distinct_identity = struct_ids.is_some() || {
            let first_module = strip_crate_prefix(quals[0])
                .rsplit_once("::")
                .map(|(m, _)| m.to_string())
                .unwrap_or_default();
            quals[1..].iter().any(|q| {
                strip_crate_prefix(q)
                    .rsplit_once("::")
                    .map(|(m, _)| m)
                    .unwrap_or_default()
                    != first_module
            })
        };
        if distinct_identity {
            tombstone_origins.push((*leaf).to_string());
        }
    }
    drop(by_leaf);
    // Withdrawals change the key set. `remove_field` drops the suffix
    // index so a later insert that restores `fields.len()` cannot reuse it.
    for leaf in drop_field_aliases {
        struct_fields.remove_field(&leaf);
    }
    let mut tombstoned_leaves = std::collections::HashSet::new();
    for leaf in tombstone_origins {
        if let Some(module) = struct_origins.get_mut(&leaf) {
            module.clear();
            // Record the withdrawal itself. Clearing the string cannot be
            // told apart afterwards from a crate-root decl, whose module
            // was already empty.
            tombstoned_leaves.insert(leaf);
        }
    }
    // Variant-leaf aliases.  The enum-variant rows (`derive_program_metadata`)
    // dual-publish each variant under qualified spellings
    // (`module::Enum::Variant`, `crate::…::Enum::Variant`) AND a bare
    // `Enum::Variant` 2-segment convenience alias.  The struct-leaf pass above
    // groups by the LAST segment (`Variant`), which mixes distinct enums'
    // same-named variants and tries to drop a bare `Variant` key that was never
    // published — so a cross-module `Enum::Variant` collision (last-decl-wins on
    // the bare alias) survives.  Group by the `Enum::Variant` tail so only
    // genuine same-enum-leaf duplicates meet, then withdraw the bare alias off
    // the same field-shape-divergence signal.  `resolve_adt_field` emits
    // `owner_root = {enum_leaf}::{variant}`, so a withdrawn alias makes an
    // ambiguous lookup miss to the qualified key or fail conservatively.
    let mut variant_by_alias: std::collections::HashMap<String, Vec<&str>> =
        std::collections::HashMap::new();
    for key in struct_fields.fields.keys() {
        if let Some((head, var)) = key.rsplit_once("::")
            && let Some((_, enm)) = head.rsplit_once("::")
        {
            variant_by_alias
                .entry(format!("{enm}::{var}"))
                .or_default()
                .push(key);
        }
    }
    let mut drop_variant_aliases: Vec<String> = Vec::new();
    for (alias, aliases) in &variant_by_alias {
        let quals = unique_declarations(aliases, struct_ids);
        if quals.len() < 2 {
            continue;
        }
        let first_rows = &struct_fields.fields[quals[0]];
        if quals[1..]
            .iter()
            .any(|q| &struct_fields.fields[*q] != first_rows)
        {
            drop_variant_aliases.push(alias.clone());
        }
    }
    drop(variant_by_alias);
    for alias in drop_variant_aliases {
        struct_fields.remove_field(&alias);
    }
    // `enum_variant_by_discriminant` dual-publishes the same bare-leaf
    // alias (qualified path + leaf), so a cross-decl leaf collision
    // leaves a silent-winner discriminant map there too.  Same rule as
    // `struct_fields.fields`: the bare alias survives only while every
    // qualified decl answers discriminant lookups identically.
    let mut enum_by_leaf: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    for key in enum_variant_by_discriminant.keys() {
        if let Some((_, leaf)) = key.rsplit_once("::") {
            enum_by_leaf.entry(leaf).or_default().push(key);
        }
    }
    let mut drop_enum_aliases: Vec<String> = Vec::new();
    for (leaf, aliases) in &enum_by_leaf {
        let quals = unique_declarations(aliases, struct_ids);
        if quals.len() < 2 {
            continue;
        }
        let first_map = &enum_variant_by_discriminant[quals[0]];
        if quals[1..]
            .iter()
            .any(|q| &enum_variant_by_discriminant[*q] != first_map)
        {
            drop_enum_aliases.push((*leaf).to_string());
        }
    }
    drop(enum_by_leaf);
    for leaf in drop_enum_aliases {
        enum_variant_by_discriminant.remove(&leaf);
        // Every enum base row is the `__discriminant`-only sentinel
        // (`derive_program_metadata`'s enum arm), so the struct-shape
        // check above can never tell two distinct enums sharing a leaf
        // apart — their bare base alias survives and
        // `struct_root_names` pre-mints ONE merged base class for
        // both.  Withdraw it here off the same discriminant-divergence
        // signal, gated on the sentinel so a same-named real struct
        // (whose bare alias the shape check already adjudicated) is left
        // untouched.
        if struct_fields
            .fields
            .get(&leaf)
            .is_some_and(|rows| rows.len() == 1 && rows[0].name == "__discriminant")
        {
            struct_fields.remove_field(&leaf);
        }
    }
    tombstoned_leaves
}

/// Lower a single Charon [`FunDecl`] to a [`FunctionGraph`].
pub fn lower_fun_decl(
    context: &LowerContext<'_>,
    fd: &FunDecl,
) -> Result<FunctionGraph, LowerError> {
    lower_fun_decl_with_static_addrs(context, fd, crate::HostStaticAddrs::default())
}

/// Whether the framestate-threaded lowering runs for acyclic bodies.
/// Default-on; `MAJIT_MIR_FRAMESTATE=0` / `=false` is the rollback escape
/// hatch to the monotonic lowering.
fn framestate_enabled() -> bool {
    !matches!(
        std::env::var("MAJIT_MIR_FRAMESTATE").as_deref(),
        Ok("0") | Ok("false")
    )
}

/// Per-program owner of derived lowering metadata (`translator.py TranslationContext`).
/// Callers share this context across declarations from the borrowed program.
pub struct LowerContext<'a> {
    llbc: &'a Llbc,
    struct_field_attrs: std::collections::HashMap<String, Vec<(String, ValueType)>>,
    /// Harvested field rows, without `StructFieldRegistry`'s query-cache
    /// interior mutability (`RefCell` / `Cell`). Tests share this context
    /// through `OnceLock`, so the stored tables stay `Sync`; lowering
    /// rebuilds a registry from these rows for `field_name_at`.
    struct_fields: std::collections::HashMap<String, Vec<(String, String)>>,
    dont_look_inside: std::collections::HashSet<String>,
    /// Leaves [`harden_duplicate_leaf_metadata`] emptied for this LLBC.
    /// Single-function entry points paint with this set, the same one the
    /// whole-program loop computes from `struct_origins`.
    tombstoned_leaves: std::collections::HashSet<String>,
    /// Structs and enums declared Gc across this one artefact.
    /// Absence is Raw (`DeclaredGcFacts::gc_struct_ids`).
    gc_struct_ids: std::collections::HashSet<majit_ir::descr::StructId>,
    /// Harvested `_alloc_flavor_ = raw` structs. `get_type_flag` banks a
    /// pointer to one as an unsigned int word (`descr.py get_type_flag`).
    raw_alloc_flavor_structs: std::collections::HashSet<String>,
    /// `_immutable_fields_<Struct>` for this artefact, harvested once.
    immutable_fields: ImmutableFieldsTable,
}

impl<'a> LowerContext<'a> {
    /// Derive the program's lowering metadata once for this context.
    pub fn new(llbc: &'a Llbc) -> Self {
        Self::new_with_dependency_raw_flavors(llbc, &std::collections::HashSet::new())
    }

    /// `new` plus the flavor-raw classes of the crates `llbc` links against (`lib.rs` passes the program-wide set to whole-crate lowering).
    pub fn new_with_dependency_raw_flavors(
        llbc: &'a Llbc,
        dependency_raw_alloc_flavor_structs: &std::collections::HashSet<String>,
    ) -> Self {
        shadow_stack_erase::ensure_stack_sensitive_fns(llbc);
        // Single-function lowering does not run
        // [`link_transparent_scalar_types`]. Publish the same external
        // scalar the link pass adds, so a `CodeFlags` borrow is the word.
        llbc.register_transparent_scalar_kinds(external_transparent_scalar_kinds());
        let gc_struct_ids = harvest_declared_gc_facts(llbc).gc_struct_ids();
        let immutable_fields = crate::front::llbc_hints::harvest_immutable_fields_from_llbcs(
            std::slice::from_ref(llbc),
        );
        let mut raw_alloc_flavor_structs =
            crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(
                llbc,
            ));
        // Flavor is a program-wide class property; a dependency's raw classes count here too.
        raw_alloc_flavor_structs.extend(dependency_raw_alloc_flavor_structs.iter().cloned());
        let (_, _, struct_fields, _, _, struct_field_attrs, _, _) = derive_program_metadata(
            llbc,
            &gc_struct_ids,
            &immutable_fields,
            &raw_alloc_flavor_structs,
        );
        Self {
            llbc,
            struct_field_attrs,
            struct_fields: struct_fields.fields.into(),
            dont_look_inside: dont_look_inside_set_of(llbc),
            tombstoned_leaves: tombstoned_leaves_of(llbc),
            gc_struct_ids,
            raw_alloc_flavor_structs,
            immutable_fields,
        }
    }

    /// Harvested `{owner → field rows}` for this artefact. A test that
    /// assembles a graph lowered from the same LLBC installs this on
    /// `CallControl` so `fielddescrof` can `getattr` the names the
    /// lowering emitted (`descr.py get_field_descr`).
    pub fn struct_field_registry(&self) -> crate::front::semantic::StructFieldRegistry {
        struct_field_registry_from_rows(&self.struct_fields)
    }
}

/// Query-cached registry over the frozen rows [`LowerContext`] stores.
/// Same `field_name_at` / `lookup_fields` convention as the whole-program
/// harvest; `__pos_N` remains the missing-row fallback.
fn struct_field_registry_from_rows(
    rows: &std::collections::HashMap<String, Vec<(String, String)>>,
) -> crate::front::semantic::StructFieldRegistry {
    crate::front::semantic::StructFieldRegistry {
        fields: crate::front::semantic::FieldRows::from(rows.clone()),
        ..Default::default()
    }
}

pub fn lower_fun_decl_with_static_addrs(
    context: &LowerContext<'_>,
    fd: &FunDecl,
    static_addrs: crate::HostStaticAddrs<'_>,
) -> Result<FunctionGraph, LowerError> {
    lower_fun_decl_with_static_addrs_and_root_stack(
        context,
        fd,
        static_addrs,
        &RootStackState::new(context.llbc),
    )
}

/// Lower `fd` using the caller's per-program [`RootStackState`].
pub fn lower_fun_decl_with_root_stack(
    context: &LowerContext<'_>,
    fd: &FunDecl,
    root_stack_state: &RootStackState,
) -> Result<FunctionGraph, LowerError> {
    lower_fun_decl_with_static_addrs_and_root_stack(
        context,
        fd,
        crate::HostStaticAddrs::default(),
        root_stack_state,
    )
}

fn lower_fun_decl_with_static_addrs_and_root_stack(
    context: &LowerContext<'_>,
    fd: &FunDecl,
    static_addrs: crate::HostStaticAddrs<'_>,
    root_stack_state: &RootStackState,
) -> Result<FunctionGraph, LowerError> {
    let llbc = context.llbc;
    let struct_fields = struct_field_registry_from_rows(&context.struct_fields);
    crate::local_crates::with_local_crate_root(llbc.crate_name(), || {
        let jitdriver_receiver_roots =
            crate::codewriter::jtransform::default_jitdriver_receiver_roots();
        lower_fun_decl_with_static_addrs_attrs_and_jitdriver_roots(
            llbc,
            fd,
            static_addrs,
            &jitdriver_receiver_roots,
            &context.struct_field_attrs,
            &struct_fields,
            &context.dont_look_inside,
            &context.tombstoned_leaves,
            &context.gc_struct_ids,
            &context.immutable_fields,
            &context.raw_alloc_flavor_structs,
            root_stack_state,
        )
    })
}

/// The `#[dont_look_inside]` marker set for this LLBC, keyed
/// `strip_crate_prefix(name_path())` — the same derivation the
/// whole-program loop harvests inline for the return-token stamp.
/// Recovered per call on the standalone lowering entries so their
/// call-target builder declines the `CallTarget::Method` hint for the
/// same residual callees the whole-program build does.
pub(crate) fn dont_look_inside_set_of(llbc: &Llbc) -> std::collections::HashSet<String> {
    crate::front::llbc_hints::harvest_hints_from_llbcs(std::slice::from_ref(llbc))
        .into_iter()
        .filter(|(_, hints)| hints.iter().any(|h| h == "dont_look_inside"))
        .map(|(path, _)| path)
        .collect()
}

/// Whether the markers in `hints` keep a callee's body out of the JitCode
/// closure, without consulting its graph.
///
/// This is the function-level half of `JitPolicy.look_inside_graph`
/// (`rpython/jit/codewriter/policy.py`): `_jit_look_inside_` overrides
/// everything, and otherwise `_reject_function` rejects an
/// `_elidable_function_` callee unconditionally — "explicitly elidable
/// functions are always opaque".  `call.py find_all_graphs` then never makes
/// the rejected graph a candidate, so its call sites stay residual.
///
/// The two spellings differ only in how the rejection was written down, so
/// everything that exists to serve a body-less residual boundary — above all
/// the declaration-sourced stub in
/// [`collect_policy_opaque_fn_stubs_from_llbc`] — reads them as one set.
/// The other half of `look_inside_graph` (loops, unsupported variable types)
/// needs a graph and therefore cannot be decided here; it can only reject
/// further, never admit a body this predicate already rejected.
///
/// Every elidable spelling the macros accept (`elidable`,
/// `elidable_cannot_raise`, `elidable_or_memerror`) emits
/// `_elidable_function_` alongside its own marker, so the single `"elidable"`
/// token covers all three.
#[cfg(any(test, feature = "mir-frontend"))]
pub(crate) fn hints_reject_body(hints: &[String]) -> bool {
    if hints.iter().any(|h| h == "jit_look_inside") {
        return false;
    }
    hints
        .iter()
        .any(|h| h == "dont_look_inside" || h == "elidable")
}

/// The set of paths in this LLBC whose bodies [`hints_reject_body`] keeps out
/// of the JitCode closure, keyed `strip_crate_prefix(name_path())` — the same
/// derivation [`dont_look_inside_set_of`] uses for its narrower question.
#[cfg(any(test, feature = "mir-frontend"))]
fn policy_opaque_fn_set_of(llbc: &Llbc) -> std::collections::HashSet<String> {
    crate::front::llbc_hints::harvest_hints_from_llbcs(std::slice::from_ref(llbc))
        .into_iter()
        .filter(|(_, hints)| hints_reject_body(hints))
        .map(|(path, _)| path)
        .collect()
}

fn lower_fun_decl_with_static_addrs_attrs_and_jitdriver_roots(
    llbc: &Llbc,
    fd: &FunDecl,
    static_addrs: crate::HostStaticAddrs<'_>,
    jitdriver_receiver_roots: &[String],
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    struct_fields: &crate::front::semantic::StructFieldRegistry,
    dont_look_inside: &std::collections::HashSet<String>,
    tombstoned_leaves: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
    root_stack_state: &RootStackState,
) -> Result<FunctionGraph, LowerError> {
    let mut u = fd.unstructured().ok_or_else(|| {
        LowerError::Unsupported(format!(
            "{}: no Unstructured body (extracted with --ullbc?)",
            fd.item_meta.name_path()
        ))
    })?;
    elaborate_explicit_root_closes(llbc, &mut u, &|reg| regular_call_name_path(reg, llbc));
    elaborate_vec_drops(llbc, &mut u);
    let u = shadow_stack_erase::erase_or_keep(fd, u, llbc);
    let accum = AccumulatorFacts::build(llbc, &u);
    let builder_mode = accum.has_builder;
    lower_unstructured_with_static_addrs_and_attrs(
        llbc,
        fd,
        &u,
        static_addrs,
        jitdriver_receiver_roots,
        struct_field_attrs,
        struct_fields,
        dont_look_inside,
        tombstoned_leaves,
        gc_struct_ids,
        immutable_fields,
        raw_alloc_flavor_structs,
        builder_mode,
        &accum,
        &RootStackAnalyzer::new(llbc, root_stack_state),
        None,
        false,
        None,
    )
}

/// Lower a same-crate callee whose return is a fat `&dyn` / `Box<dyn>`,
/// so the caller can splice the body instead of taking the one-word
/// jitcode return. `None` leaves the call residual: the body is missing,
/// `dont_look_inside`, or this `def_id` is already being spliced (a
/// cycle). The body is spliced at any size. A cutoff would keep the
/// one-word return, and the caller's `method_*` read would load the slot
/// from the data pointer.
fn lower_fat_dyn_callee(
    llbc: &Llbc,
    def_id: u64,
    static_addrs: crate::HostStaticAddrs<'_>,
    jitdriver_receiver_roots: &[String],
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    struct_fields: &crate::front::semantic::StructFieldRegistry,
    dont_look_inside: &std::collections::HashSet<String>,
    tombstoned_leaves: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<FunctionGraph> {
    if crate::front::dyn_fat::splice_in_progress(def_id) {
        return None;
    }
    let decl = llbc.fn_by_id(def_id)?;
    if fun_decl_is_dont_look_inside(decl, dont_look_inside) || !decl.has_unstructured_body() {
        return None;
    }
    let nblocks = decl.unstructured()?.body.len();
    if nblocks == 0 {
        return None;
    }
    lower_fun_decl_with_static_addrs_attrs_and_jitdriver_roots(
        llbc,
        decl,
        static_addrs,
        jitdriver_receiver_roots,
        struct_field_attrs,
        struct_fields,
        dont_look_inside,
        tombstoned_leaves,
        gc_struct_ids,
        immutable_fields,
        raw_alloc_flavor_structs,
        &RootStackState::new(llbc),
    )
    .ok()
}

/// Splice recorded fat-dyn calls, then point `{vtable}` `method_*` reads
/// at the metadata word. The cache is per body: Charon `def_id`s restart
/// on the next LLBC, and a `None` while that callee is already on the
/// splice stack must not stick.
fn splice_recorded_fat_dyn_returns(
    lo: &mut Lowering<'_>,
    llbc: &Llbc,
    fd: &FunDecl,
    static_addrs: crate::HostStaticAddrs<'_>,
    jitdriver_receiver_roots: &[String],
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    struct_fields: &crate::front::semantic::StructFieldRegistry,
    dont_look_inside: &std::collections::HashSet<String>,
    tombstoned_leaves: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) {
    let dyn_results = lo.dyn_return_calls.clone();
    let cache =
        std::cell::RefCell::new(std::collections::HashMap::<u64, Option<FunctionGraph>>::new());
    let mut lower_callee = |def_id: u64| -> Option<FunctionGraph> {
        if let Some(cached) = cache.borrow().get(&def_id).cloned() {
            return cached;
        }
        if crate::front::dyn_fat::splice_in_progress(def_id) {
            return None;
        }
        let lowered = lower_fat_dyn_callee(
            llbc,
            def_id,
            static_addrs,
            jitdriver_receiver_roots,
            struct_field_attrs,
            struct_fields,
            dont_look_inside,
            tombstoned_leaves,
            gc_struct_ids,
            immutable_fields,
            raw_alloc_flavor_structs,
        );
        cache.borrow_mut().insert(def_id, lowered.clone());
        lowered
    };
    crate::front::dyn_fat::splice_fat_dyn_returns(
        &mut lo.graph,
        llbc,
        fd.def_id,
        &dyn_results,
        &mut lower_callee,
    );
}

#[path = "shadow_stack_erase.rs"]
mod shadow_stack_erase;
pub use shadow_stack_erase::census as shadow_stack_erase_census;
pub use shadow_stack_erase::{
    discover_depth_neutral_fns, discover_stack_fn_effects, discover_stack_leaves_above_fns,
    discover_stack_param_slots_fns, discover_stack_returns_index_fns, discover_stack_sensitive_fns,
    ensure_stack_sensitive_fns,
};

#[path = "owner_root_guard.rs"]
mod owner_root_guard;

mod address;
mod consteval;
mod desugar;
mod flowcontext;
mod framestate;
mod gcroot;
mod lltype;
mod malloc;
mod operation;
mod removeassert;
mod simplify;
mod strformat;
mod views;

use self::address::*;
use self::consteval::*;
use self::desugar::*;
pub(crate) use self::flowcontext::KEEPALIVE_UNTIL_HERE_PATH;
use self::flowcontext::*;
use self::framestate::*;
use self::gcroot::*;
use self::lltype::*;
use self::malloc::*;
use self::operation::*;
use self::removeassert::*;
use self::simplify::*;
use self::strformat::*;
use self::views::*;

pub(crate) use self::consteval::{
    attach_foldable_const_lits, discover_foldable_const_lits, discover_transparent_scalar_kinds,
    harvest_named_const_folds, push_named_const_folds, register_ambiguous_impl_foldable_const_lits,
    register_foldable_const_lits,
};
pub(crate) use self::desugar::callee::{
    discover_eval_hook_graphs, release_declared_vable_array_address,
};
pub(crate) use self::desugar::rewire::{
    AggregateCtor, AggregateShape, emit_aggregate_value, is_core_clone_impls_clone_path,
    is_core_default_path,
};
pub use self::gcroot::{
    PublishedStructBody, PublishedStructField, RootStackState, apply_published_structs,
    erased_root_bracket_guards, fn_returns_owned_scope, harvest_published_structs,
    harvest_root_stack_touching_paths, harvest_scope_owning_identities, mark_foreign_scope_owners,
    scope_owner_identity,
};
pub(crate) use self::gcroot::{
    ROOT_SCOPE_CLOSE, RootStackAnalyzer, harvest_root_stack_touching_paths_with,
    harvest_scope_owning_identities_with, is_root_scope_drop_glue_call, is_rust_vec_free_call,
    is_shadow_stack_bracket_close, is_slice_buffer_free_call,
};
pub(crate) use self::lltype::{
    ThreadLocalRefLlop, adt_head_instantiation_suffix, charon_array_len_to_string,
    charon_literal_to_ast_string, charon_type_value_to_ast_string,
    collect_foreign_opaque_method_externals, collect_policy_opaque_fn_stubs_from_llbc,
    fixed_array_index_identity, fnptr_indirect_enabled, host_rthread_threadlocalref_llop,
    is_flattened_storage_leaf, is_rstr_str_layout_declaration, is_str_family_class_root,
    is_str_length_field_class_root, is_str_oopspec_class_root,
    items_block_accessor_returns_its_block, nonzero_option_zero, output_type_is_objectptr,
    push_unichr, same_nominal_adt, strip_crate_prefix, type_decl_ref_adt_id, type_decl_ref_builtin,
    type_decl_ref_generics, tyref_is_fat_dyn, unichr_call,
};
pub use self::lltype::{collect_unsafe_fn_stubs_from_llbc, slice_array_type_id};
pub(crate) use self::malloc::forward_identity;
pub(crate) use self::operation::{
    cast_int_to_ptr_call, cast_ptr_to_int_call, lltype_for_direct_ptradd_pointer, value_type_bank,
};
pub(crate) use self::removeassert::bind_array_view_len_in_block;
pub(crate) use self::strformat::interpolant_is_rust_heap_string;
pub(crate) use self::views::{fat_part_concretetype, fat_part_value_type, pair_len_field_name};

/// Graphs that pass a pointer to `OwnerRootGuard` (or to an aggregate that
/// contains one) into a callee that stays residual. Each row is
/// `(graph path, callee path)`.
pub fn residual_owner_root_guard_escapes(llbc: &Llbc) -> Vec<(String, String)> {
    owner_root_guard::residual_escape_census(llbc)
}

/// Definitions of each MIR local: an `Assign` into the bare local, or a
/// `Call` whose destination is that local.  A projection write is not a
/// definition of the local.  [`owned_builder_dest_locals`] and
/// [`is_fresh_str_builder`] both count through here so the candidate list
/// and the proof cannot drift.
fn local_def_counts(body: &Unstructured, llbc: &Llbc, n: usize) -> Vec<usize> {
    let mut def_count = vec![0usize; n];
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &bb.statements {
            if let Ok(StmtKind::Assign(place, _)) = st.stmt_kind_ref()
                && let &PlaceKind::Local(i) = &place.kind
                && (i as usize) < n
            {
                def_count[i as usize] += 1;
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && let &PlaceKind::Local(i) = &call.dest.kind
            && (i as usize) < n
        {
            def_count[i as usize] += 1;
        }
    }
    def_count
}

/// Whether local `c` is an owned functional-concat accumulator: not the
/// return slot, not a parameter, declared as a by-value `Wtf8Buf` or
/// `String`, and defined exactly once.
fn owned_single_def_builder(body: &Unstructured, llbc: &Llbc, c: usize, def_count: usize) -> bool {
    let n = body.locals.locals.len();
    let arg_count = body.locals.arg_count as usize;
    (arg_count + 1..n).contains(&c)
        && def_count == 1
        && tyref_is_owned_str_builder(&body.locals.locals[c].ty, llbc)
}

/// MIR locals that can be a functional-concat string accumulator.
///
/// Each one is a by-value `Wtf8Buf` or `String` in
/// `(arg_count+1..n_locals)` — the return slot and every parameter are
/// excluded — with exactly one definition.  That definition may be a
/// builder constructor, any other call's result, or a move out of a `?`
/// `Continue` payload.  Rust ownership of the single definition means the
/// local is not aliased, so rebinding it as `buf = ll_strconcat(buf, piece)`
/// is sound the same way it is for a constructor-defined accumulator.
/// Resolved on demand from the body, no per-local side table.
fn owned_builder_dest_locals(body: &Unstructured, llbc: &Llbc) -> Vec<usize> {
    let n = body.locals.locals.len();
    let def_count = local_def_counts(body, llbc, n);
    (0..n)
        .filter(|&c| owned_single_def_builder(body, llbc, c, def_count[c]))
        .collect()
}

/// Whether `u` (the body of `fd`) contains at least one builder-mode string
/// accumulator ([`is_builder_mode_accumulator`]) — the pre-check that selects
/// the canonical builder-form lowering. `builder_mode[c]` is that predicate
/// for local `c`, already computed for this body.
fn graph_has_builder_accumulator(llbc: &Llbc, u: &Unstructured, builder_mode: &[bool]) -> bool {
    owned_builder_dest_locals(u, llbc)
        .iter()
        .any(|c| builder_mode.get(*c).copied().unwrap_or(false))
}

/// Lower `fd` from an already-projected `Unstructured` body.
///
/// `FunDecl::body` is retained as raw JSON and `FunDecl::unstructured`
/// re-parses it on every call, so a caller that has already projected the
/// body — the whole-program loop, which needs the projection to decide
/// whether the decl has one at all — passes it in here rather than paying
/// the parse a second time.
/// The name of `fd`'s graph: its template path, or for a monomorphized copy
/// the name a clause-specialized copy of the same instance gets.
///
/// `FunctionDesc.cachedgraph` (`rpython/annotator/description.py`) names a
/// specialized graph `"%s__%s" % (self.name, valid_identifier(nameof(key)))`.
/// [`crate::front::clause_spec::spec_leaf`] is that spelling for a Charon
/// instance key, and a Charon-monomorphized copy is the same instance, so it
/// takes the same name and the passes that see through `__spec_` see
/// through it too.  The copy's identity is its `FunDeclId`.
pub(crate) fn graph_name_of(llbc: &Llbc, fd: &FunDecl) -> String {
    match instance_leaf(llbc, fd) {
        Some(leaf) => spec_segments(llbc, fd, &leaf).join("::"),
        None => fd.item_meta.name_path(),
    }
}

/// The leaf a Charon-monomorphized copy registers and is called under:
/// [`crate::front::clause_spec::spec_leaf`] of its template leaf and its
/// instance arguments.  `None` for an item Charon did not instantiate, and
/// for a bodyless instance of a foreign or std declaration: that is one
/// external whatever it is instantiated at, so it keeps the template path, as
/// an opaque callee does in `enqueue_spec`.  A bodyless instance of a local
/// crate's item is the copy that crate's own LLBC carries a body for, so it
/// takes the same leaf and a call across the crate boundary resolves to that
/// copy.
fn instance_leaf(llbc: &Llbc, fd: &FunDecl) -> Option<String> {
    let args = fd.item_meta.instantiation()?;
    // A HOST_ENV residual (`rlib/jit.py ConditionalCallEntry`) is one
    // ExtRegistry object. Spec-suffixing a Charon instance makes
    // Layer-1 miss the declined template and Layer-3b miss the builtin
    // leaf, so `FunctionDesc.pycall` walks the unique template graph.
    if fun_decl_name_path_is_host_env(fd) {
        return None;
    }
    if !crate::front::clause_spec::decl_has_unstructured_body(fd) {
        let sibling = fd.item_meta.name.first().is_some_and(|seg| {
            matches!(seg, NameSeg::Ident { ident: (root, _) }
                if crate::local_crates::is_local_crate_root(root))
        });
        if !sibling {
            return None;
        }
    }
    let name = fd.item_meta.name_path();
    let leaf = name.rsplit("::").next().unwrap_or(&name);
    Some(crate::front::clause_spec::spec_leaf(
        leaf, fd.def_id, args, llbc,
    ))
}

/// A `Result<(), PyError>` scoped callee returns void after the
/// exception-link lowering; its returnblock is widened so the call
/// descriptor's `FUNC.RESULT` is `v`, not the `Ref`-typed unit shell.
/// `output` is the body's declared return type: the FunDecl signature
/// for a template, local 0 after substitution for a clause spec.
fn result_exc_ok_is_unit(
    output: &TyRef,
    llbc: &Llbc,
    error_carrier: crate::ErrorCarrierSpec<'_>,
) -> bool {
    crate::front::result_exc::tyref_is_result_of_carrier(output, llbc, error_carrier)
        && crate::front::result_exc::tyref_result_ok_is_unit(output, llbc)
}

fn lower_unstructured_with_static_addrs_and_attrs(
    llbc: &Llbc,
    fd: &FunDecl,
    u: &Unstructured,
    static_addrs: crate::HostStaticAddrs<'_>,
    jitdriver_receiver_roots: &[String],
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    struct_fields: &crate::front::semantic::StructFieldRegistry,
    dont_look_inside: &std::collections::HashSet<String>,
    tombstoned_leaves: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
    // When set, each strategy's `Lowering` is switched to builder form
    // ([`Lowering::enable_builder_mode`]) so a string accumulator
    // emits the `StringBuilder` markers instead of `ll_strconcat`.  The
    // caller sets this directly from [`graph_has_builder_accumulator`], so
    // qualifying functions have one canonical marker-emitting graph.
    builder_mode: bool,
    accum: &AccumulatorFacts,
    root_stack: &RootStackAnalyzer<'_>,
    spec: Option<&std::cell::RefCell<crate::front::clause_spec::SpecQueue>>,
    spec_body: bool,
    graph_name: Option<String>,
) -> Result<FunctionGraph, LowerError> {
    // A clause specialization's readable name (`RDict::new__spec_…`) is
    // known to the caller. The body is still the template `FunDecl`, so
    // `graph_name_of` would miss that suffix. The graph keeps the
    // readable name while the body lowers.
    let name = graph_name.unwrap_or_else(|| graph_name_of(llbc, fd));
    // `dont_look_inside` is keyed by `strip_crate_prefix(name_path())` on
    // the declaration. A Charon instance name does not spell that key.
    let self_dont_look_inside = fun_decl_is_dont_look_inside(fd, dont_look_inside);
    let keep_zst_self = fundecl_keeps_zst_self(
        llbc,
        fd,
        if spec_body { None } else { Some(&u.locals) },
        tombstoned_leaves,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    );
    // The Result-of-PyError exception-link lowering's callee rule
    // applies when this body is a scoped callee (see
    // `front::result_exc`); the caller rule applies to the diamond
    // sites the body lowering captured.  Both run before
    // `simplify_lowered_graph` so the freed shell ops feed the same
    // dead-op sweep the Abort → RaiseImplicit fold uses.
    //
    // Local 0 is the return place. A clause-spec body is the FunDecl
    // with type arguments substituted, so that local is the instantiated
    // output. The template signature still names the generic Result.
    let output = if spec_body {
        u.locals
            .locals
            .first()
            .map(|local| &local.ty)
            .unwrap_or(&fd.signature.output)
    } else {
        &fd.signature.output
    };
    let result_exc_callee = crate::front::result_exc::tyref_is_result_of_carrier(
        output,
        llbc,
        static_addrs.error_carrier,
    );
    let result_exc_ok_is_unit =
        result_exc_callee && crate::front::result_exc::tyref_result_ok_is_unit(output, llbc);
    let finish = |lo: &mut Lowering<'_>| -> Result<(), LowerError> {
        lo.emit_pending_vec_frees();
        lo.graph.return_class_root = dont_look_inside_return_class_root(
            output,
            llbc,
            static_addrs.error_carrier,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
        lo.graph.return_can_be_none = dont_look_inside_return_can_be_none(
            output,
            llbc,
            static_addrs.error_carrier,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
        lo.graph.return_is_str =
            dont_look_inside_return_is_str(output, llbc, static_addrs.error_carrier);
        lo.graph.return_container_root = dont_look_inside_return_container_root(
            output,
            llbc,
            static_addrs.error_carrier,
            gc_struct_ids,
        );
        let expand_inline_struct = |owner: &str, field: &str| {
            inline_struct_field_leaves(
                llbc,
                tombstoned_leaves,
                gc_struct_ids,
                owner,
                field,
                raw_alloc_flavor_structs,
            )
        };
        // MIR framestate argument threading must finish before adding native
        // enum arms: its successor table names MIR blocks, not these new
        // flow blocks. No temporary tagged-pair root reaches annotation.
        for site in &lo.tagged_pair_aggregate_sites {
            if lo.graph.block(site.block).dead {
                continue;
            }
            crate::front::option_variant::validate_dynamic_option_exit(&lo.graph, site.block)
                .map_err(LowerError::Unsupported)?;
            let operations = &lo.graph.block(site.block).operations;
            if operations.len() != site.operation_index + 3
                || operations[site.operation_index].result.as_ref() != Some(&site.result)
            {
                return Err(LowerError::Unsupported(
                    "tagged-pair tail changed before variant construction".to_string(),
                ));
            }
            let variants = if site.payload_owner == format!("{}::Some", site.owner) {
                ["None", "Some"]
            } else if site.payload_owner == format!("{}::Ok", site.owner) {
                ["Ok", "Err"]
            } else {
                return Err(LowerError::Unsupported(format!(
                    "tagged-pair payload has no concrete success variant: {}",
                    site.payload_owner
                )));
            };
            lo.graph
                .block_mut(site.block)
                .operations
                .truncate(site.operation_index);
            crate::front::option_variant::emit_sum_variant_dynamic(
                &mut lo.graph,
                site.block,
                site.result.clone(),
                &site.owner,
                site.disc.clone(),
                variants,
                Some((
                    &site.payload_owner,
                    site.payload.clone(),
                    site.payload_ty.clone(),
                )),
            );
        }
        if let Some(site) = lo
            .result_map_err_sites
            .iter()
            .find(|site| !site.closure_env_is_trivially_dropless)
        {
            return Err(LowerError::Unsupported(format!(
                "{}: Result::map_err closure {} captures values, but the Ok-arm destructor glue is not lowered",
                lo.graph.name, site.call_once_owner
            )));
        }
        // Carrier map_err rewrite waits until after result_exc hung
        // LastException on the dest-is-carrier call. Extra-simplify still
        // runs first: it may rename the recorded result variable.

        // `map(it, f).collect()` denotes a loop. Rewrite it at the
        // construction site — where `f` is a concrete closure ADT —
        // into `Vec::new` + `next` + `call_mut` + `push` BEFORE the extra
        // simplify and `result_exc`. A Result-of-carrier call in the same
        // body (`c_call_trace`) turns that extra simplify on; it collapses
        // empty blocks and `result_exc` then hangs `LastException` on the
        // collect block, so the collect-site gate fails and
        // `ops::range::Range::map` stays an unregistered FunctionPath
        // (phaseA never records the graph as a prepass subject). Fail-safe:
        // a site whose closure env is not a concrete ADT, or whose inner
        // iterator is not a list `iter` / exclusive Range, is left as the
        // residual `collect`. The synthesized `next` is recorded after
        // this extra-simplify gate so a map-collect-only graph does not
        // grow a simplify it did not previously run; Result-of-carrier
        // graphs still simplify because their own sites already trip it.
        let map_collect_synthesized = if !lo.map_collect_sites.is_empty() {
            crate::front::iter_adapter::rewire_map_collect_sites(
                &mut lo.graph,
                &lo.map_collect_sites,
            )
        } else {
            Vec::new()
        };
        if !lo.result_exc_call_results.is_empty()
            || !lo.option_ok_or_else_try_sites.is_empty()
            || !lo.carrier_combinator_sites.is_empty()
            || !lo.result_map_err_sites.is_empty()
            || result_exc_callee
            || !lo.next_call_results.is_empty()
            || !lo.checked_arith_call_results.is_empty()
            || !lo.result_try_sites.is_empty()
            || !lo.disc_combinator_sites.is_empty()
            || !lo.slice_index_rangefrom_sites.is_empty()
            || !lo.slice_index_range_sites.is_empty()
            || !lo.slice_index_rangeto_sites.is_empty()
        {
            // The exception-link transforms run on a simplified graph,
            // as exceptiontransform.py does (graphs reach it after
            // `simplify_graph`, simplify.py): the discriminant
            // switch's `default → Abort` else-unreachable arm must be
            // pruned by `remove_assertion_errors` before the diamond
            // matcher sees the switch, leaving the plain 0/1 pair.
            // Untouched graphs skip this and keep their single
            // end-of-lowering simplify, byte-identical.
            let guard_words = lo.guard_word_vars.clone();
            simplify_lowered_graph_expanding(
                &mut lo.graph,
                struct_field_attrs,
                false,
                &guard_words,
                &expand_inline_struct,
            )?;
        }
        lo.next_call_results.extend(map_collect_synthesized);
        let mut tail_forwarded_returns = 0usize;
        if !lo.slice_index_rangefrom_sites.is_empty() {
            crate::front::slice_index::rewire_slice_index_rangefrom_sites(
                &mut lo.graph,
                &lo.slice_index_rangefrom_sites,
            );
        }
        if !lo.slice_index_range_sites.is_empty() {
            crate::front::slice_index::rewire_slice_index_range_sites(
                &mut lo.graph,
                &lo.slice_index_range_sites,
            );
        }
        if !lo.slice_index_rangeto_sites.is_empty() {
            crate::front::slice_index::rewire_slice_index_rangeto_sites(
                &mut lo.graph,
                &lo.slice_index_rangeto_sites,
            );
        }
        // `map`/`and_then` on a carrier Result is an if/else on the receiver
        // (`DiscCombinatorSite::carrier_recv`); build it before the carrier
        // becomes exception edges.
        let (carrier_disc_sites, disc_combinator_sites): (Vec<_>, Vec<_>) = lo
            .disc_combinator_sites
            .iter()
            .cloned()
            .partition(|site| site.carrier_recv);
        if !carrier_disc_sites.is_empty() {
            rewire_disc_combinator_sites(
                &mut lo.graph,
                &carrier_disc_sites,
                static_addrs.error_carrier,
            );
        }
        // Carrier `unwrap_or` / `expect` / `unwrap` fuse onto the call's
        // `LastException` join (`try/except OperationError`, or the implicit
        // `AssertionError` panic for `core` `Result::expect`) before
        // [`catch_and_rewrap`] can rebuild a Result shell around the same call.
        let result_unwrap_or_rewritten = if lo.carrier_combinator_sites.is_empty() {
            0
        } else {
            crate::front::result_exc::rewire_carrier_combinator_sites(
                &mut lo.graph,
                &lo.carrier_combinator_sites,
            )
        };
        if !lo.result_exc_call_results.is_empty() {
            let carrier = static_addrs.error_carrier;
            let foreign_sites =
                crate::front::result_exc::foreign_from_residual_sites(&lo.graph, carrier);
            let mut conversions = Vec::with_capacity(foreign_sites.len());
            for site in &foreign_sites {
                conversions.push(
                    from_impl_conversion(lo.llbc, carrier.carrier_path, &site.error_ty).map_err(
                        |err| LowerError::Unsupported(format!("{}: {err}", lo.graph.name)),
                    )?,
                );
            }
            crate::front::result_exc::apply_foreign_from_residuals(
                &mut lo.graph,
                &foreign_sites,
                &conversions,
            )
            .map_err(LowerError::Unsupported)?;
            let outcome = crate::front::result_exc::rewire_result_exc_call_sites(
                &mut lo.graph,
                &lo.result_exc_call_results,
                result_exc_callee,
                static_addrs.error_carrier,
            )
            .map_err(LowerError::Unsupported)?;
            tail_forwarded_returns = outcome.tail_forwards;
        }
        // After result_exc: the dest-is-carrier call block now has
        // LastException. Err calls the mapper and raises; a declined
        // site stays the extracted helper graph.
        let result_map_err_rewritten = crate::front::result_map_err::rewire_result_map_err_sites(
            &mut lo.graph,
            &lo.result_map_err_sites,
        );
        let option_ok_or_else_try_rewritten = if lo.option_ok_or_else_try_sites.is_empty() {
            0
        } else {
            crate::front::result_exc::rewire_option_ok_or_else_try_sites(
                &mut lo.graph,
                &lo.option_ok_or_else_try_sites,
                result_exc_callee,
            )
        };

        if result_exc_callee {
            crate::front::result_exc::lower_result_exc_returns(
                &mut lo.graph,
                tail_forwarded_returns,
                static_addrs.error_carrier,
            )
            .map_err(LowerError::Unsupported)?;
            if result_exc_ok_is_unit {
                // Stamp `FUNC.RESULT = void`.  The exception-link lowering
                // already returns the unit `()` (the callee no longer
                // builds a `Result` shell), but `front::mir` types every
                // aggregate — including `()` — as `Ref`, so the CFG return
                // kind is still `r`.  The codewriter reconciles this by
                // collapsing the returnblock to a genuine void return
                // post-annotation (the `declared==v && cfg==r` gate),
                // mirroring `exceptiontransform.py` running after rtyping;
                // doing the structural collapse here, before the
                // whole-program annotation fixpoint, destabilises it.
                lo.graph.return_type = Some("()".to_string());
            }
        }
        // The `?`-diamond rewrite (`rewire_result_exc_call_sites`) detaches
        // the pre-rewrite branch / discriminant / break blocks: the call
        // block now exits straight to the continue arm and `exceptblock`,
        // so those blocks lose their only predecessor.  RPython graph
        // consumers only iterate blocks reachable from `startblock`
        // (`flowspace/model.py iterblocks`), so drop the now-unreachable
        // blocks here — before `prune_dead_phis`, which would otherwise
        // treat a no-predecessor block as an extra root
        // (`transform_dead_op_vars`'s start set), and before the
        // `codewriter` consumers that scan `graph.blocks` directly.
        // The `next`-diamond rewrite (`front::iter_next`) runs on the same
        // simplified graph: the Option discriminant switch's default→Abort
        // arm must be pruned first, identically to the `?` diamond.  It is
        // fail-safe — a non-for-loop `Option` match is left as the residual
        // call — so it runs over every recorded site and reports how many
        // it actually rewrote.  Only an actual rewrite detaches blocks (the
        // discriminant switch), so the unreachable-block sweep is gated on
        // that count, leaving a declined graph byte-identical.
        // Divert exclusive int-`Range` for-loops (`for _ in a..b`) to the
        // orthodox `range()` builtin + list-iter BEFORE the `next`-diamond
        // runs, so its `originates_from_iter_op` gate sees the
        // `["core","slice","iter"]` op and folds the loop.  Runs on the same
        // simplified graph; fail-safe — an unpaired / multi-consumer range
        // aggregate stays the ordinary ADT ctor (census Skip).
        // `map(it, f).collect()` already ran before the extra simplify /
        // `result_exc` above; `next_call_results` holds any synthesized
        // `next` from that rewrite.
        if !lo.range_iter_new_sites.is_empty() && !lo.next_call_results.is_empty() {
            // The range rewrite only locates the `next()` producer; the
            // element kind recorded beside it belongs to the diamond fold.
            let next_vars: Vec<Variable> = lo
                .next_call_results
                .iter()
                .map(|(var, _)| var.clone())
                .collect();
            crate::front::range_iter::rewire_range_iter_sites(
                &mut lo.graph,
                &lo.range_iter_new_sites,
                &next_vars,
            );
        }
        let next_rewritten = if lo.next_call_results.is_empty() {
            0
        } else {
            crate::front::iter_next::rewire_next_call_sites(&mut lo.graph, &lo.next_call_results)
        };
        // The checked-arith rewrite (`front::checked_arith`) runs on the
        // same simplified graph as the `next`-diamond rewrite: the Option
        // discriminant switch's default→Abort arm must be pruned first,
        // identically.  It is fail-safe — a non-overflow-fallback `Option`
        // match is left as the residual call — so it runs over every
        // recorded site and reports how many it actually rewrote.  Only an
        // actual rewrite detaches the discriminant switch's block, so the
        // unreachable-block sweep is gated on that count.
        let checked_arith_rewritten = if lo.checked_arith_call_results.is_empty() {
            crate::front::checked_arith::CheckedArithRewriteCount::default()
        } else {
            crate::front::checked_arith::rewire_checked_arith_call_sites(
                &mut lo.graph,
                &lo.checked_arith_call_results,
                &lo.checked_arith_ok_or_else_sites,
            )
        };
        // The unsigned checked-arith rewrite (`front::checked_arith_uint`) runs
        // AFTER `checked_arith`: a signed match-shape site the first pass
        // already rewrote is now a `*_ovf` BinOp producer (skipped by the "the
        // producer must still be the `Call`" gate), so the two passes never
        // fight over a site.  It replaces each residual in place with native
        // `uint_mul_high` / `uint_lt` overflow tests and a Some/None diamond.
        // The existing reachability-sweep gate covers the new arm edges.
        let checked_arith_uint_rewritten = if lo.checked_arith_uint_sites.is_empty() {
            0
        } else {
            crate::front::checked_arith_uint::rewire_checked_arith_uint_sites(
                &mut lo.graph,
                &lo.checked_arith_uint_sites,
            )
        };
        // `rewire_checked_arith_ok_or_else` rebuilds `Result::Ok(sum)` /
        // `Result::Err(closure())` shells on the two edges it fuses, and it
        // runs after the callee rule above.  A scoped callee whose return WAS
        // that `checked_*(..).ok_or_else(..)` pair therefore had nothing for
        // the callee rule to rewrite when it ran, and carries the fold's fresh
        // shells into its returnblock afterwards — while
        // `rewire_result_exc_call_sites` has already rewired this callee's
        // call sites to receive the unwrapped value.  Lower what the fold
        // produced, so the two halves of the one decision agree again.
        if result_exc_callee && checked_arith_rewritten.result_shells > 0 {
            // This is a new validation pass over the shells the checked-arith
            // rewrites just created.  Earlier tail-forwards and rewritten
            // returns cannot certify these new returns: if none of them has a
            // supported shape, the callee must decline instead of exposing a
            // mixture of unwrapped values and Result shells to its callers.
            crate::front::result_exc::lower_result_exc_returns(
                &mut lo.graph,
                0,
                static_addrs.error_carrier,
            )
            .map_err(LowerError::Unsupported)?;
            crate::front::result_exc::fuse_kind_ctor_raise(&mut lo.graph);
        }
        // `Layout::from_size_align(size, align)` itself (`front::from_size_align`)
        // — Opaque core, no extracted body — becomes int arithmetic plus a
        // virtualized `Result<Layout, LayoutError>`, the producer-side twin of
        // `checked_arith_uint`.  Downstream `Result::ok` / `expect` consume
        // that Result through the existing combinator passes.  Independent of
        // the checked-arith passes (it consumes their `Option<usize>` result
        // as its `size` arg only after they have already produced it).
        let from_size_align_call_rewritten = if lo.from_size_align_call_sites.is_empty() {
            0
        } else {
            crate::front::from_size_align::rewire_from_size_align_call_sites(
                &mut lo.graph,
                &lo.from_size_align_call_sites,
            )
        };
        // The `Layout::from_size_align(..).ok()` rewrite
        // (`front::from_size_align`) collapses the `from_size_align` + `ok`
        // residual pair into a native `uint_lt` bound test + a virtualized
        // nested `Option<Layout>` aggregate.  Independent of the checked-arith
        // passes (it consumes their `Option<usize>` result as its `size` arg
        // only after they have already produced it). Its Some/None arms use
        // the same reachability-sweep gate as the other graph rewrites.
        let from_size_align_rewritten = if lo.from_size_align_sites.is_empty() {
            0
        } else {
            crate::front::from_size_align::rewire_from_size_align_sites(
                &mut lo.graph,
                &lo.from_size_align_sites,
            )
        };
        // The by-value `Layout::from_size_align(..).expect(..)` sibling
        // (`front::from_size_align`) collapses the `from_size_align` + `expect`
        // residual pair into the same native `uint_lt` bound test + a by-value
        // virtualized `Layout`, branching to an implicit raise on overflow.  It
        // splits the `from_size_align` block into a fits/overflow diamond, so it
        // needs the reachability sweep gate below.
        let from_size_align_expect_rewritten = if lo.from_size_align_expect_sites.is_empty() {
            0
        } else {
            crate::front::from_size_align::rewire_from_size_align_expect_sites(
                &mut lo.graph,
                &lo.from_size_align_expect_sites,
            )
        };
        if !lo.layout_accessor_sites.is_empty() {
            crate::front::from_size_align::rewire_layout_accessor_sites(
                &mut lo.graph,
                &lo.layout_accessor_sites,
            );
        }
        // Non-carrier `Result` `?` is the ControlFlow diamond with Ok/Err
        // polarity and an `Err` return.  Option `?` is a regular call of
        // the extracted `core::option::<Impl>::branch` body.  Exception-carrier
        // Results are already rewritten above.
        let result_try_stats = if lo.result_try_sites.is_empty() {
            ResultTryStats::default()
        } else {
            let return_owners = lo.resolve_result_return_owners(output);
            rewire_result_try_call_sites(&mut lo.graph, &lo.result_try_sites, return_owners)
        };
        // A `?` whose diamond did not match still has `Try::branch`.
        // Equal `Result` / `ControlFlow` layouts were already emitted as
        // `same_as`. Anything left is the match: read the discriminant
        // and build `Continue` / `Break`.
        lower_result_branch_to_control_flow(&mut lo.graph);
        // The `<[T]>::first(slice)` length-checked `Option<&T>` diamond rewrite
        // (`front::slice_first`) splits the residual `first` call block into a
        // guarded `Some`/`None` diamond (`if len(slice) > 0`). It runs on the
        // post-lowering graph (its block A is closed with a single goto by
        // `lower_call`) and is fail-safe: a structural mismatch leaves the
        // residual call (rtyper Skip). Gate the reachability sweep on an
        // actual rewrite.
        let slice_first_rewritten = if lo.slice_first_sites.is_empty() {
            0
        } else {
            crate::front::slice_first::rewire_slice_first_call_sites(
                &mut lo.graph,
                &lo.slice_first_sites,
            )
        };
        // The `<[T]>::get(slice, i)` bounds-checked `Option<&T>` diamond
        // rewrite (`front::slice_get`) splits the residual `get` call block
        // into a guarded `Some`/`None` diamond (`if i < len(slice)`), same
        // post-lowering shape and fail-safe contract as `slice_first`, of
        // which it is the general case; gate the reachability sweep on an
        // actual rewrite.
        let slice_get_rewritten = if lo.slice_get_sites.is_empty() {
            0
        } else {
            crate::front::slice_get::rewire_slice_get_call_sites(&mut lo.graph, &lo.slice_get_sites)
        };
        // The `<[u8]>::is_ascii(slice)` rewrite (`front::slice_ascii`)
        // replaces the residual with `first_non_ascii_char(s) < 0`
        // (`rutf8.check_ascii`).  The scan loop lives in the helper graph,
        // not the caller; gate the reachability sweep on an actual rewrite.
        let slice_is_ascii_rewritten = if lo.slice_is_ascii_sites.is_empty() {
            0
        } else {
            crate::front::slice_ascii::rewire_slice_is_ascii_call_sites(
                &mut lo.graph,
                &lo.slice_is_ascii_sites,
            )
        };
        // Header-view `slice::from_raw_parts{,_mut}` (`front::from_raw_parts`)
        // aliases the residual fat slice to the length-prefixed GC string
        // header.  In-place `same_as`; no new blocks.
        let _from_raw_parts_rewritten =
            crate::front::from_raw_parts::rewire_from_raw_parts_sites(&mut lo.graph);
        // Word-sized `saturating_mul` clamp (`front::saturating_mul`) splits
        // the residual call into `lo = a * b; hi = uint_mul_high(a, b); if
        // ne(hi, 0) { MAX } else { lo }`.  Same fail-safe as add.
        let _saturating_mul_rewritten = if lo.saturating_mul_sites.is_empty() {
            0
        } else {
            crate::front::saturating_mul::rewire_saturating_mul_call_sites(
                &mut lo.graph,
                &lo.saturating_mul_sites,
            )
        };
        // `core::intrinsics::saturating_{add,sub}` clamp
        // (`front::saturating_intrinsic`) splits the residual Call into
        // the unsigned floor / signed `*_ovf` diamond. Same post-lowering
        // contract as `saturating_mul`: an in-lowering
        // split would replace the Call's goto to the MIR `target` with
        // edges to fresh non-MIR arms, so `lower_framestate` would
        // `stub_dead_block` the continuation (`flowcontext.py`
        // `make_next_block` unions only the just-closed exits).
        let saturating_add_rewritten = if lo.saturating_add_sites.is_empty() {
            0
        } else {
            crate::front::saturating_intrinsic::rewire_saturating_add_call_sites(
                &mut lo.graph,
                &lo.saturating_add_sites,
            )
        };
        let saturating_sub_rewritten = if lo.saturating_sub_sites.is_empty() {
            0
        } else {
            crate::front::saturating_intrinsic::rewire_saturating_sub_call_sites(
                &mut lo.graph,
                &lo.saturating_sub_sites,
            )
        };
        // Option/Result unwrap and plain Result expect are regular calls of
        // the extracted core bodies
        // (`match self { Some(v)/Ok(v) => v, None/Err => <diverging Never call> }`).
        // The Never callee is `RaiseImplicit` (`TermKind::Abort`);
        // `remove_assertion_errors` then prunes that arm. Carrier
        // `Result<T, PyError>` unwrap/expect stay on `result_exc`.
        let option_ok_or_else_disc = disc_combinator_sites
            .iter()
            .any(|site| site.kind == DiscCombinator::OptionOkOrElse);
        if !disc_combinator_sites.is_empty() {
            rewire_disc_combinator_sites(
                &mut lo.graph,
                &disc_combinator_sites,
                static_addrs.error_carrier,
            );
        }
        // `OptionOkOrElse` rebuilds `Result::Ok` / `Result::Err` after the
        // callee rule above.  A scoped graph whose return was that
        // `ok_or_else` pair therefore had nothing for the callee rule to
        // rewrite when it ran.  Lower the fold's fresh shells, the same
        // follow-up `rewire_checked_arith_ok_or_else` uses.
        //
        // `tail_forwarded_returns` is non-zero because this is a follow-up:
        // the callee is already transformed.  Zero new shells means the
        // combinator's Result was consumed in-graph (`?` / `and_then`), not
        // that the callee has no returns.
        if result_exc_callee && option_ok_or_else_disc {
            crate::front::result_exc::lower_result_exc_returns(
                &mut lo.graph,
                1,
                static_addrs.error_carrier,
            )
            .map_err(LowerError::Unsupported)?;
            crate::front::result_exc::fuse_kind_ctor_raise(&mut lo.graph);
        }
        // A scoped scalar callee can `return` an `Option<Result<T, PyError>>::Some`
        // payload it did not build. [`lower_result_exc_returns`] rewrites `Ok`/`Err`
        // ctors and tail-forwarded calls, so that aggregate stays `Ref` and
        // `graph_result_kind` reports `r` against the scalar `FUNC.RESULT`
        // (`dont_look_inside_return_token`). Checked-arith's second callee rule
        // can still rebuild shells, so this runs after both.
        // `exceptiontransform.py` `transform_completely` carries `T` on the
        // normal edge and raises on the error edge.
        let scalar_shell_owners = if result_exc_callee {
            lo.resolve_result_owners(output)
        } else {
            None
        };
        if let Some((result_owner, ok_owner, err_owner, ok_ty, err_ty, _)) = scalar_shell_owners {
            crate::front::result_exc::unwrap_returned_scalar_result_shells(
                &mut lo.graph,
                &result_owner,
                &ok_owner,
                &err_owner,
                &ok_ty,
                &err_ty,
            )
            .map_err(LowerError::Unsupported)?;
        }
        if !lo.result_exc_call_results.is_empty()
            || result_exc_callee
            || option_ok_or_else_try_rewritten > 0
            || result_unwrap_or_rewritten > 0
            || result_map_err_rewritten > 0
            || next_rewritten > 0
            || checked_arith_rewritten.total > 0
            || checked_arith_uint_rewritten > 0
            || from_size_align_call_rewritten > 0
            || from_size_align_rewritten > 0
            || from_size_align_expect_rewritten > 0
            || result_try_stats.rewritten > 0
            || !lo.result_try_sites.is_empty()
            || slice_first_rewritten > 0
            || slice_get_rewritten > 0
            || slice_is_ascii_rewritten > 0
            || saturating_add_rewritten > 0
            || saturating_sub_rewritten > 0
            || !lo.disc_combinator_sites.is_empty()
        {
            crate::model::clear_unreachable_blocks(&mut lo.graph);
        }
        let guard_words = lo.guard_word_vars.clone();
        simplify_lowered_graph_expanding(
            &mut lo.graph,
            struct_field_attrs,
            true,
            &guard_words,
            &expand_inline_struct,
        )?;
        // `removeassert.py` melt of abort-bound `format!` / `assert_eq!`
        // message construction.  Runs before the live `format!` collapsers
        // so a placeholder panic does not expand into `str` + `ll_strconcat`
        // only to be discarded.  `format!("line {lineno}")` that feeds an
        // exception string is not abort-bound and stays for those collapsers.
        let abort_messages_melted = melt_abort_message_ops(&mut lo.graph);
        // `format!`-chain expansion (descriptor census): rewrite the recognized
        // `Argument::new_display`/`Arguments::new`/`alloc::fmt::format`
        // chain into native `str` + `ll_strconcat` ops so the graph-less
        // fmt externs stop blocking the rtyper.  All emitted ops are ones
        // the legacy walker and codewriter already handle, so it runs
        // unconditionally.
        let mut fmt_collapsed = collapse_fmt_chains(&mut lo.graph);
        // The bytearray repr's sole non-default format is `\\x{c:02x}`.
        // PyPy spells it as two nibble lookups in a constant hex table; lower
        // the proven-u8 Rust fmt shell to that same source-level shape.
        fmt_collapsed += collapse_lower_hex_byte_fmt_chains(&mut lo.graph);
        // Constant `format!("literal")` — a zero-placeholder format charon has
        // already folded to the fully-rendered constant string, so the format
        // op's argument is a bare `__str_const` with no pieces/args chain.
        // Replace the op with that constant string in place.
        fmt_collapsed += collapse_const_fmt(&mut lo.graph);
        // Multi-argument `format!("{a}…{b}", …)` chains build an N-field
        // argument tuple and N `Argument::new_display` ctors across several
        // blocks — a shape the single-argument collapser above does not
        // recognize.  This phase rewrites each `new_display` to `str` in
        // place and folds the rendered values with the literal pieces at
        // the `Arguments::new` block, so multi-arg Display chains lower to
        // native `str` + `ll_strconcat` like the single-arg case.
        fmt_collapsed += collapse_fmt_chains_multi(&mut lo.graph);
        // Fieldless-enum `Debug` (`{:?}`) `format!` chains the string
        // collapsers above decline: a `Debug` render of a fieldless enum has
        // no native rstr counterpart, but each variant's derived `Debug` is
        // the bare variant identifier — a compile-time constant — so the whole
        // rendered message is a per-variant constant.  Lower the chain to a
        // switch on the enum discriminant emitting the constant message per
        // arm (the `discriminant → name-table` shape the enum's `Debug` is),
        // retiring the graph-less `fmt::rt::Argument::new_debug` extern.
        fmt_collapsed += collapse_debug_enum_fmt_chains(&mut lo.graph, lo.llbc);
        // Re-threading a rendered value onto a forwarding link and deleting
        // the chain's intermediate ops can leave a block with no remaining
        // predecessor (the pre-collapse branch arm) whose `Link.args` still
        // reference a now-deleted chain var; under framestate threading such
        // an orphan also pins the deleted var into a merge phi.  Drop the
        // orphaned blocks and the dead phis so the adapter does not later
        // reject the graph on an undefined `Link.args` operand — the same
        // post-collapse cleanup the panic-message phase below performs.
        if fmt_collapsed > 0 {
            // `rewrite_debug_enum_fmt_site` installs a new exitswitch use.
            // Thread it onto the block local before prune drops an unused
            // family member (`ssa.py` `SSA_to_SSI` `record_used_var`).
            crate::model::thread_undefined_op_operands(&mut lo.graph);
            crate::model::clear_unreachable_blocks(&mut lo.graph);
            crate::model::prune_dead_phis(&mut lo.graph);
        }
        // `panic!` / `assert!` message-block chains end in an implicit
        // `AssertionError` raise but route through graph-less `fmt`
        // message externs the rtyper can't type; collapse them to the
        // bare raise so `remove_assertion_errors` prunes the branch as it
        // does for a direct implicit raise.  Gated on an actual collapse
        // so untouched graphs keep their single end-of-lowering simplify.
        if abort_messages_melted > 0 || collapse_panic_message_chains(&mut lo.graph) > 0 {
            crate::model::clear_unreachable_blocks(&mut lo.graph);
            let guard_words = lo.guard_word_vars.clone();
            simplify_lowered_graph_expanding(
                &mut lo.graph,
                struct_field_attrs,
                true,
                &guard_words,
                &expand_inline_struct,
            )?;
        }
        splice_recorded_fat_dyn_returns(
            lo,
            llbc,
            fd,
            static_addrs,
            jitdriver_receiver_roots,
            struct_field_attrs,
            struct_fields,
            dont_look_inside,
            tombstoned_leaves,
            gc_struct_ids,
            immutable_fields,
            raw_alloc_flavor_structs,
        );
        // Field reads emitted while blocks were still being closed name a
        // base defined only in a predecessor. Thread once every edge exists.
        // Doing so earlier leaves a later edge short one link arg
        // (`simplify.py` `len(link.args) == len(link.target.inputargs)`).
        lo.graph.thread_undefined_field_bases();
        // `flowcontext.py` `fixeggblocks`, once the body is a finished graph.
        lo.graph.fixeggblocks();
        Ok(())
    };
    // Framestate-threaded lowering (the GAP-B path that threads locals
    // as block inputargs / phis — the orthodox flowspace shape).
    // Default-on; `MAJIT_MIR_FRAMESTATE=0` rolls back to the monotonic
    // lowering.  Acyclic bodies thread via the two-pass RPO walk; cyclic
    // bodies additionally pre-seed each loop header's entry framestate
    // with the live-in phis `new` pre-bound, so the back-edge threads
    // into them in pass 2 instead of skipping the body to the monotonic
    // single-slot scheme.  On any framestate failure the body falls back
    // to the monotonic path — so the threaded path is never worse than
    // the monotonic one — unless `MAJIT_MIR_FRAMESTATE_STRICT` is set,
    // which propagates the error for debugging.
    if framestate_enabled() {
        let mut lo = Lowering::new(
            llbc,
            name.clone(),
            u,
            static_addrs,
            jitdriver_receiver_roots,
            fd.generics.as_ref(),
            dont_look_inside,
            tombstoned_leaves,
            gc_struct_ids,
            immutable_fields,
            raw_alloc_flavor_structs,
            accum,
            root_stack,
            self_dont_look_inside,
            keep_zst_self,
        )?;
        lo.struct_fields = Some(struct_fields);
        lo.raw_alloc_flavor_structs = raw_alloc_flavor_structs.clone();
        if builder_mode {
            lo.enable_builder_mode();
        }
        if let Some(spec) = spec {
            lo.set_spec(spec);
            if spec_body {
                lo.set_spec_body();
            }
        }
        // Back-edge targets (loop headers); empty for an acyclic body, in
        // which case `lower_framestate` reduces exactly to the two-pass
        // RPO walk.  Treat the threaded lowering and its shared
        // post-lowering stage (`finish`) as one attempt.  `finish` runs
        // uniformly, same as the linear / RPO paths below — it folds
        // constant exitswitches, drops `Abort` arms, runs
        // `simplify_lowered_graph` and installs the exception-link ABI
        // (raise-through vs Result return); gating it on result-exc
        // activity would let the framestate path keep dead arms the other
        // strategies prune.  A post-pass (e.g. the result-exc diamond
        // rewrite) can reject a body the threading itself accepted, and
        // the threading can fail on a CFG shape it does not yet model, so
        // fold both into one fallback: on any error fall through to the
        // monotonic path below, which is known to lower the body — the
        // threaded path is never worse than the monotonic one.
        let loop_headers = lo.mir_model_loop_headers();
        let attempt = lo
            .lower_framestate(&loop_headers)
            .and_then(|()| finish(&mut lo));
        match attempt {
            Ok(()) => return Ok(lo.graph),
            Err(e) => {
                if std::env::var_os("MAJIT_MIR_FRAMESTATE_DEBUG").is_some() {
                    eprintln!("[FRAMESTATE fallback] {:?}: {e:?}", name);
                }
                if std::env::var_os("MAJIT_MIR_FRAMESTATE_STRICT").is_some() {
                    return Err(e);
                }
            }
        }
    }
    let mut lo = Lowering::new(
        llbc,
        name.clone(),
        u,
        static_addrs,
        jitdriver_receiver_roots,
        fd.generics.as_ref(),
        dont_look_inside,
        tombstoned_leaves,
        gc_struct_ids,
        immutable_fields,
        raw_alloc_flavor_structs,
        accum,
        root_stack,
        self_dont_look_inside,
        keep_zst_self,
    )?;
    lo.struct_fields = Some(struct_fields);
    lo.raw_alloc_flavor_structs = raw_alloc_flavor_structs.clone();
    if builder_mode {
        lo.enable_builder_mode();
    }
    if let Some(spec) = spec {
        lo.set_spec(spec);
        if spec_body {
            lo.set_spec_body();
        }
    }
    match lo.lower(BlockOrder::Linear) {
        Ok(()) => {
            finish(&mut lo)?;
            Ok(lo.graph)
        }
        // A forward-referenced definition — typically a `TermKind::Call`
        // dest at a higher MIR index than the block that reads it — reads
        // as an uninitialised local under MIR-index order.  Re-lower the
        // whole body in reverse-postorder, which visits the defining
        // block first.  This is scoped to exactly the bodies that fail
        // linearly (every other body keeps its linear-order bindings
        // untouched).  Loop-carried locals do not need RPO at all: they
        // ride the per-block live-in inputargs minted in `new` and
        // threaded by `edge_args` across the back-edge (phi / block
        // inputargs), which is order-independent.  RPO only resolves the
        // acyclic forward-reference case above.
        Err(LowerError::Unsupported(msg)) if is_known_lowering_gap(&msg) => {
            let mut lo = Lowering::new(
                llbc,
                name,
                u,
                static_addrs,
                jitdriver_receiver_roots,
                fd.generics.as_ref(),
                dont_look_inside,
                tombstoned_leaves,
                gc_struct_ids,
                immutable_fields,
                raw_alloc_flavor_structs,
                accum,
                root_stack,
                self_dont_look_inside,
                keep_zst_self,
            )?;
            lo.struct_fields = Some(struct_fields);
            lo.raw_alloc_flavor_structs = raw_alloc_flavor_structs.clone();
            if builder_mode {
                lo.enable_builder_mode();
            }
            if let Some(spec) = spec {
                lo.set_spec(spec);
                if spec_body {
                    lo.set_spec_body();
                }
            }
            lo.lower(BlockOrder::ReversePostorder)?;
            finish(&mut lo)?;
            Ok(lo.graph)
        }
        Err(e) => Err(e),
    }
}

/// Order in which [`Lowering::lower`] walks the MIR basic blocks.
#[derive(Clone, Copy)]
enum BlockOrder {
    /// Plain MIR index order (`0..len`) — the default.
    Linear,
    /// Reverse-postorder of the CFG — the fallback used only for bodies
    /// whose linear lowering hits a forward-referenced (uninitialised)
    /// local.
    ReversePostorder,
}

/// Errors the driver fails with. The driver fails loud — `Unsupported`
/// surfaces a precise variant + the MIR shape that prompted it so
/// each widening can be a small targeted change.
#[derive(Debug)]
pub enum LowerError {
    FunctionNotFound(String),
    /// A MIR construct the current driver does not yet handle.
    Unsupported(String),
    /// A failure to project raw JSON into the typed ULLBC subset.
    Schema(String),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::FunctionNotFound(n) => write!(f, "function not found: {n}"),
            LowerError::Unsupported(s) => write!(f, "unsupported MIR: {s}"),
            LowerError::Schema(s) => write!(f, "schema decode: {s}"),
        }
    }
}

impl std::error::Error for LowerError {}

// Lowering state

pub use majit_jitcode::codewriter::jtransform::OBJECT_REF_GCARRAY_TYPE_ID;
/// PyPy `BytesListStrategy` / `AsciiListStrategy` expose `SomeString`
/// elements, while `rmodel.externalvsinternal(..., gcref=True)` stores those
/// GC pointers in `GcArray(GCREF)`.  The logical list identity must therefore
/// remain `[str]`; the rtyper's ordinary ListRepr recast supplies the
/// STRPTR↔GCREF conversion at each item boundary.
const STRING_GCREF_GCARRAY_TYPE_ID: &str = "[str]";

/// MIR block → the flow-graph block recorded for it.
///
/// `FlowContext.build_flow` records a block only when `pendingblocks`
/// reaches it. Edges [`successor_blocks`] does not emit — every
/// `on_unwind`, and a `SwitchInt` panic default whose arms are outside
/// the niche shape — never enqueue a block, so those MIR blocks have no
/// [`BlockId`] and no per-block row. A niche-shaped `SwitchInt` keeps
/// that default so [`Lowering::lower_niche_option_switch`] can emit it
/// as the missing 0/1 arm.
/// `Index` yields the graph block; [`MirBlockMap::row`] is the dense
/// index into the per-block tables.
#[derive(Clone)]
struct MirBlockMap {
    ids: Vec<Option<BlockId>>,
    row_of: Vec<u32>,
    nrows: usize,
}

impl MirBlockMap {
    fn from_lowered(graph: &mut FunctionGraph, lowered: &[bool]) -> Self {
        let n = lowered.len();
        let mut ids = vec![None; n];
        let mut row_of = vec![u32::MAX; n];
        let mut nrows = 0usize;
        if n > 0 {
            debug_assert!(lowered[0], "bb0 is the entry of every body");
            ids[0] = Some(graph.startblock);
            row_of[0] = 0;
            nrows = 1;
            for mir_bb in 1..n {
                if !lowered[mir_bb] {
                    continue;
                }
                ids[mir_bb] = Some(graph.create_block());
                row_of[mir_bb] = nrows as u32;
                nrows += 1;
            }
        }
        Self { ids, row_of, nrows }
    }

    fn nrows(&self) -> usize {
        self.nrows
    }

    fn is_lowered(&self, mir_bb: usize) -> bool {
        self.ids.get(mir_bb).is_some_and(Option::is_some)
    }

    fn row(&self, mir_bb: usize) -> usize {
        let row = self.row_of.get(mir_bb).copied().unwrap_or(u32::MAX);
        assert!(
            row != u32::MAX,
            "MIR bb{mir_bb} is outside the lowered closure"
        );
        row as usize
    }

    /// Move the emit cursor for this MIR block to `id`. The dense row and
    /// the entry stay fixed: `store_enum_variant` / `copy_enum_switch` split
    /// the block and only move where the rest of the block is emitted.
    fn set(&mut self, mir_bb: usize, id: BlockId) {
        let slot = self
            .ids
            .get_mut(mir_bb)
            .unwrap_or_else(|| panic!("MIR bb{mir_bb} is outside the lowered closure"));
        assert!(
            slot.is_some(),
            "MIR bb{mir_bb} is outside the lowered closure"
        );
        *slot = Some(id);
    }

    fn iter_lowered(&self) -> impl Iterator<Item = (usize, BlockId)> + '_ {
        self.ids
            .iter()
            .enumerate()
            .filter_map(|(mir, id)| id.map(|id| (mir, id)))
    }
}

impl std::ops::Index<usize> for MirBlockMap {
    type Output = BlockId;

    fn index(&self, mir_bb: usize) -> &BlockId {
        self.ids[mir_bb]
            .as_ref()
            .unwrap_or_else(|| panic!("MIR bb{mir_bb} is outside the lowered closure"))
    }
}

fn terminator_is_panic_abort(llbc: &Llbc, body: &Unstructured, bb: u64) -> bool {
    matches!(
        body.body
            .get(bb as usize)
            .and_then(|block| block.term_ref(llbc).ok()),
        Some(
            TermKind::Abort(_)
                | TermKind::Panic { .. }
                | TermKind::UnwindResume
                | TermKind::UnwindTerminate
                | TermKind::UndefinedBehavior
        )
    )
}

/// Successors [`Lowering::lower_terminator`] materialises, plus — when
/// `include_panic_default` — a `SwitchInt` default that targets a panic
/// abort. A niche-shaped `SwitchInt` ([`switch_int_niche_emits_default`])
/// keeps that default even when `include_panic_default` is false, because
/// [`Lowering::lower_niche_option_switch`] may emit it as the missing 0/1
/// arm. `on_unwind` is never a successor: a Rust panic propagates out of
/// the frame, the way an RPython operation with no handler raises to its
/// caller.
///
/// A normal successor may be a cleanup block:
/// `kept_block_normal_successor_is_cleanup` is why `strip_cleanup_blocks`
/// left those bodies unchanged.
fn successor_blocks(
    llbc: &Llbc,
    body: &Unstructured,
    mir_bb: usize,
    include_panic_default: bool,
) -> Vec<usize> {
    let n = body.body.len();
    let Ok(term) = body.body[mir_bb].term_ref(llbc) else {
        return Vec::new();
    };
    let raw: Vec<u64> = match term {
        TermKind::Goto { target } => vec![*target],
        TermKind::Call { target, .. }
        | TermKind::Assert { target, .. }
        | TermKind::Drop { target, .. } => vec![*target],
        TermKind::Switch { targets, .. } => match targets {
            SwitchTargets::If(a, b) => vec![*a, *b],
            SwitchTargets::SwitchInt(_, arms, default) => {
                let mut targets: Vec<u64> = arms.iter().map(|(_, bb)| *bb).collect();
                let panic = terminator_is_panic_abort(llbc, body, *default);
                if include_panic_default || !panic || switch_int_niche_emits_default(llbc, arms) {
                    targets.push(*default);
                }
                targets
            }
        },
        TermKind::Return
        | TermKind::UnwindResume
        | TermKind::UnwindTerminate
        | TermKind::Abort(_)
        | TermKind::Panic { .. }
        | TermKind::UndefinedBehavior
        | TermKind::Unknown => Vec::new(),
    };
    let mut out = Vec::new();
    for target in raw {
        let target = target as usize;
        if target >= n || out.contains(&target) {
            continue;
        }
        out.push(target);
    }
    out
}

/// True when the panic default of a `SwitchInt` is kept as a successor
/// because [`Lowering::lower_niche_option_switch`] may emit it as the
/// missing 0/1 arm of a niche `Option` `If`.
///
/// Shape-only over-approximation: a single `Ref` arm, or exactly one of
/// the 0/1 integer arms and nothing else. Niche routing is decided during
/// lowering (`niche_disc_vars`), and the mask is computed before that set
/// is filled. The generic `SwitchInt` path still filters the panic default
/// link, so an over-included panic block just becomes a lowered block with
/// no incoming link.
fn switch_int_niche_emits_default(llbc: &Llbc, arms: &[(serde_json::Value, u64)]) -> bool {
    if arms.len() == 1 && arms[0].0.get("Ref").is_some() {
        return true;
    }
    niche_option_switch_missing_01_arm(llbc, arms)
}

fn niche_option_switch_missing_01_arm(llbc: &Llbc, arms: &[(serde_json::Value, u64)]) -> bool {
    let mut has0 = false;
    let mut has1 = false;
    for (scalar, _) in arms {
        match scalar_to_const_value(scalar, llbc) {
            Some(ConstValue::Int(0) | ConstValue::UInt(0)) => has0 = true,
            Some(ConstValue::Int(1) | ConstValue::UInt(1)) => has1 = true,
            _ => return false,
        }
    }
    has0 != has1
}

fn reachable_mask(llbc: &Llbc, body: &Unstructured, include_panic_default: bool) -> Vec<bool> {
    let n = body.body.len();
    let mut reached = vec![false; n];
    if n == 0 {
        return reached;
    }
    reached[0] = true;
    let mut stack = vec![0usize];
    while let Some(bb) = stack.pop() {
        for succ in successor_blocks(llbc, body, bb, include_panic_default) {
            if !reached[succ] {
                reached[succ] = true;
                stack.push(succ);
            }
        }
    }
    reached
}

/// Blocks reachable from bb0 along every normal edge, including a
/// `SwitchInt` default the lowering does not emit. Whole-body scans use
/// this set: a panic-abort block can still assign a local, and an
/// `on_unwind` cleanup block cannot.
pub(crate) fn forward_reachable_mask(llbc: &Llbc, body: &Unstructured) -> Vec<bool> {
    reachable_mask(llbc, body, true)
}

/// Blocks reachable from bb0 along the edges the lowering emits.
fn lowered_reachable_mask(llbc: &Llbc, body: &Unstructured) -> Vec<bool> {
    reachable_mask(llbc, body, false)
}

/// Resolve a `Variable` used as a call argument back to the operation
/// that produced it, following block-input `Link`s across blocks.
///
/// The `fmt`-chain recognizer (#277) fires at `alloc::fmt::format(v)`
/// where `v` is the `Arguments` value — but `v` reaches that block as an
/// `inputarg` threaded from the predecessor's `Arguments::new` result via
/// the outgoing `Link`, not as a direct op result in the current block.
/// Existing `try_lower_*` folds only read a call's direct args; the fmt
/// recognizer needs this cross-block back-trace.
///
/// Returns `(block, op_index)` of the producing `SpaceOperation`, or
/// `None` when `var` traces to a `Const`, a function input (no producing
/// op), a phi merge (a block with more than one incoming `Link`, so no
/// single producer), or a producer not yet emitted into `graph`.
pub(crate) fn resolve_to_producer_op(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> Option<(BlockId, usize)> {
    let mut current = var.clone();
    let mut visited: Vec<u64> = Vec::new();
    loop {
        if visited.contains(&current.id()) {
            return None; // cycle guard
        }
        visited.push(current.id());

        // (1) Produced directly by an op in some block?
        for block in &graph.blocks {
            for (idx, op) in block.operations.iter().enumerate() {
                if op.result.as_ref().is_some_and(|r| r.id() == current.id()) {
                    return Some((block.id, idx));
                }
            }
        }

        // (2) Otherwise it is a block inputarg — follow the single
        // incoming Link's matching positional arg back one block.
        let (owner, pos) = graph.blocks.iter().find_map(|block| {
            block
                .inputargs
                .iter()
                .position(|a| a.id() == current.id())
                .map(|pos| (block.id, pos))
        })?;
        let mut incoming = graph
            .blocks
            .iter()
            .flat_map(|b| b.exits.iter())
            .filter(|link| link.target == owner);
        let first = incoming.next()?;
        if incoming.next().is_some() {
            return None; // phi merge: no single producer
        }
        match first.args.get(pos)? {
            crate::model::LinkArg::Value(v) => current = v.clone(),
            crate::model::LinkArg::Const(_) => return None,
        }
    }
}

/// Whether `op` is this graph's `ll_vec_newemptylist` / `ll_vec_newlist_hint`
/// / `ll_vec_alloc_and_set` — the helpers that produce the
/// `raw_malloc_varsize_char` header `ll_vec_free` releases.
fn op_is_in_graph_vec_alloc(op: &SpaceOperation) -> bool {
    op_vec_alloc_kind(op).is_some()
}

fn op_vec_alloc_kind(op: &SpaceOperation) -> Option<majit_ir::rvec::VecItemKind> {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = &op.kind
    else {
        return None;
    };
    match majit_ir::rvec::vec_helper_for_path(&segments.join("::")) {
        Some((
            majit_ir::rvec::VecOp::NewEmpty
            | majit_ir::rvec::VecOp::NewHint
            | majit_ir::rvec::VecOp::AllocAndSet,
            kind,
        )) => Some(kind),
        _ => None,
    }
}

/// Whether `header` is a `Vec` this graph allocated through
/// `ll_vec_newemptylist` / `ll_vec_newlist_hint` / `ll_vec_alloc_and_set`.
///
/// `rewrite_op_free` (`jtransform.py`) emits `raw_free` wherever the graph
/// frees; the pointer is threaded through Links like any other variable,
/// so a phi merge is not a different owner. Every incoming SSA path must
/// reach one of those producers. A call result or function input is not
/// this graph's allocation.
pub(crate) fn rust_vec_header_allocated_in_graph(graph: &FunctionGraph, header: &Variable) -> bool {
    rust_vec_header_allocated_in_graph_rec(graph, header, &mut Vec::new(), &mut Vec::new())
        == Some(true)
}

/// `Some(true)` allocated here, `Some(false)` not, `None` a cycle in the
/// current walk (a loop-carried phi). Other incoming arms decide a cycle.
fn rust_vec_header_allocated_in_graph_rec(
    graph: &FunctionGraph,
    header: &Variable,
    visiting: &mut Vec<u64>,
    cache: &mut Vec<(u64, bool)>,
) -> Option<bool> {
    let id = header.id();
    if let Some((_, ok)) = cache.iter().find(|(k, _)| *k == id) {
        return Some(*ok);
    }
    if visiting.contains(&id) {
        return None;
    }
    visiting.push(id);

    let computed = 'compute: {
        for block in &graph.blocks {
            for op in &block.operations {
                if op.result.as_ref().is_some_and(|r| r.id() == id) {
                    break 'compute Some(op_is_in_graph_vec_alloc(op));
                }
            }
        }
        let Some((owner, pos)) = graph.blocks.iter().find_map(|block| {
            block
                .inputargs
                .iter()
                .position(|a| a.id() == id)
                .map(|pos| (block.id, pos))
        }) else {
            break 'compute Some(false);
        };
        let incoming: Vec<&Link> = graph
            .blocks
            .iter()
            .flat_map(|b| b.exits.iter())
            .filter(|link| link.target == owner)
            .collect();
        if incoming.is_empty() {
            break 'compute Some(false);
        }
        let mut saw_alloc = false;
        let mut saw_other = false;
        for link in incoming {
            match link.args.get(pos) {
                Some(LinkArg::Value(v)) => {
                    match rust_vec_header_allocated_in_graph_rec(graph, v, visiting, cache) {
                        Some(true) => saw_alloc = true,
                        Some(false) => saw_other = true,
                        None => {}
                    }
                }
                _ => saw_other = true,
            }
        }
        if saw_other {
            Some(false)
        } else if saw_alloc {
            Some(true)
        } else {
            // Every incoming arm is a cycle (`None`). Caching `Some(false)`
            // here makes the header that also carries `ll_vec_alloc_and_set`
            // look unowned, so the drop never emits `free`. A loop-carried
            // copy is that same raw pointer (`Block.inputargs`):
            // `rtype_free_non_gc_object` emits `free` for the malloc, and
            // `rewrite_op_free` turns flavor `raw` into `raw_free`. The
            // cycle stays `None` (and is not cached) so the alloc arm decides.
            None
        }
    };

    visiting.pop();
    if let Some(ok) = computed {
        cache.push((id, ok));
    }
    computed
}

/// Emit `ll_vec_free_*` of `header` when this graph allocated it.
///
/// Returns whether a free was emitted. A call-result or function-input
/// header records `header-not-allocated-in-graph` and is left alone.
pub(crate) fn emit_rust_vec_drop_if_allocated(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    header: &Variable,
    kind: majit_ir::rvec::VecItemKind,
) -> bool {
    if !rust_vec_header_allocated_in_graph(graph, header) {
        crate::decline::record_named(
            crate::decline::gate::RUST_VEC_DROP,
            "header-not-allocated-in-graph",
            &graph.name,
        );
        return false;
    }
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: rust_vec_helper_segments(majit_ir::rvec::VecOp::Free, kind),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![header.clone()]),
            result_ty: ValueType::Void,
        },
    });
    true
}

/// Whether `dom` dominates `target`: every path from the graph entry to
/// `target` passes through `dom`.  Computed as "with `dom` removed, is
/// `target` unreachable from `startblock`" (the standard reachability
/// formulation; a block dominates itself).  Used to prove a value defined
/// in `dom` is available at every op in `target`.
fn block_dominates(graph: &FunctionGraph, dom: BlockId, target: BlockId) -> bool {
    if dom == target {
        return true;
    }
    let mut seen: Vec<BlockId> = vec![graph.startblock];
    let mut stack = vec![graph.startblock];
    while let Some(b) = stack.pop() {
        if b == dom {
            continue; // cut: do not traverse through the candidate dominator
        }
        if b == target {
            return false; // reached target without passing through dom
        }
        if let Some(block) = graph.blocks.iter().find(|x| x.id == b) {
            for l in &block.exits {
                if !seen.contains(&l.target) {
                    seen.push(l.target);
                    stack.push(l.target);
                }
            }
        }
    }
    // If cutting `dom` makes `target` unreachable, `dom` dominates `target`.
    // Guard against an unreachable `target` (vacuously "dominated"): only
    // treat as dominated when `target` is genuinely reachable in the full
    // graph.
    block_reachable(graph, target)
}

/// Whether `target` is reachable from `startblock` following exits.
fn block_reachable(graph: &FunctionGraph, target: BlockId) -> bool {
    let mut seen: Vec<BlockId> = vec![graph.startblock];
    let mut stack = vec![graph.startblock];
    while let Some(b) = stack.pop() {
        if b == target {
            return true;
        }
        if let Some(block) = graph.blocks.iter().find(|x| x.id == b) {
            for l in &block.exits {
                if !seen.contains(&l.target) {
                    seen.push(l.target);
                    stack.push(l.target);
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests;
