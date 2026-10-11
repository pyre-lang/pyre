//! Stable-Rust parser for Charon `.llbc` / `.ullbc` JSON artefacts.
//!
//! This crate is the input layer of the MIR-driven flowspace driver.
//! It exposes:
//!
//!   - [`schema`] — `serde::Deserialize` structs covering the subset of
//!     Charon's IR we actually consume. Schema fields we do not yet
//!     consume are kept as opaque [`serde_json::Value`] so that newer
//!     Charon versions stay round-trippable; the typed schema is widened
//!     incrementally as each piece is needed.
//!   - [`Llbc`] — a thin wrapper around [`schema::LlbcFile`] with
//!     lookup helpers (`local_fn`, `iter_local_fns`, etc.).
//!   - [`SchemaError`] — fail-loud error type. The crate never silently
//!     drops bodies; an unrecognised variant returns a hard error.
//!
//! The crate compiles on **stable Rust**. The pinned-nightly toolchain
//! required to produce `.llbc` lives inside Charon itself
//! (`scripts/install-charon.py`); nothing in this crate touches it.

#![forbid(unsafe_code)]

pub mod schema;
pub mod ullbc;

pub use schema::LlbcFile;
pub use ullbc::{
    BasicBlock, FieldDecl, FunDecl, GlobalDecl, Locals, Statement, StmtKind, TagEncoding,
    TagLayout, TermKind, TraitDecl, TypeDecl, TypeDeclKind, Unstructured, VariantDecl,
};

use serde::Deserialize;
use std::path::Path;

/// Loaded `.llbc` / `.ullbc` artefact + lookup helpers.
#[derive(Debug)]
pub struct Llbc {
    pub file: LlbcFile,
    /// `dedup_id → ADT def_id` index built from inline
    /// `Value: [id, body]` occurrences whose body decodes as
    /// `{"Adt": {"id": <def_id>}}`.  Sorted by `dedup_id` for
    /// binary search.  Populated once at parse time.
    ///
    /// Consumed by `front::mir::Lowering` to resolve a Charon `Impl`
    /// segment's `skip_binder: {"Deduplicated": <id>}` reference to
    /// the receiver type's small `def_id` so `CallTarget::Method` can
    /// carry the leaf type name.  Without this, an inherent-impl
    /// method called through `CallTarget::FunctionPath` leaves the
    /// callee body's `self` arg typed as `SomeInstance(classdef=None)`
    /// and any `.field` projection on it panics in the annotator
    /// (`annotator/unaryop.rs:3587`).
    dedup_adt: Vec<(u64, u64)>,
    /// `dedup_id → body` index built from every inline
    /// `Value: [id, body]` occurrence in the raw LLBC JSON.
    /// Sorted by `dedup_id` for binary search.  Populated once at
    /// parse time.
    ///
    /// Consumed by `front::mir::Lowering::tyref_to_value_type` so a
    /// `TyRef::Deduplicated{id}` reference can be projected to its
    /// underlying `ValueType` (primitive `Literal` bodies → `Int` /
    /// `Bool` / `Float`, `Adt` / `Ref` / `RawPtr` → `Ref`).  Without
    /// this index, FunDecl return types serialized as `Deduplicated`
    /// (≈8190 of 8940 typed return signatures in `pyre-interpreter.ullbc`)
    /// fall back to `Ref` and downstream callers cannot distinguish
    /// `i64`-returning helpers from pointer-returning ones, defeating
    /// `fn_return_types`-based type checks.
    dedup_body: Vec<(u64, DedupBody)>,
    /// Trait-ref bodies. Their hash-cons ids restart at zero, independently
    /// of type ids, so they cannot share [`Self::dedup_body`].
    dedup_trait: Vec<(u64, DedupBody)>,
    /// Constant expressions `[literal, ty]`. Their ids are not type ids.
    dedup_const: Vec<(u64, DedupBody)>,
    /// Layout scalars `{"Constant": {"Value": [id, [literal, ty]]}}`.
    /// Their ids restart independently of MIR constant expressions.
    dedup_layout: Vec<(u64, DedupBody)>,
    /// Hash-consed span id → inline [`ullbc::SpanData`]. Built in the same
    /// scan as [`Self::dedup_body`]. A missing id is not a span.
    span_bodies: Vec<(u64, ullbc::SpanData)>,
    /// Qualified transparent-type path → scalar register shape learned from
    /// another LLBC in the same linked translation input. Dependency LLBCs
    /// retain layout attributes but may expose the type body as `Opaque`; the
    /// defining crate supplies the missing one-field scalar shape.
    transparent_scalar_kinds: parking_lot::RwLock<Vec<(String, TransparentScalarKind)>>,
    /// Folded initializer of one global, keyed by that global's `def_id`
    /// inside this artefact. The encoded literal is owned by the global
    /// the lowering already loads with `global_by_id`; it is not a path
    /// string and not a thread-local.
    foldable_const_lits: parking_lot::RwLock<Vec<(u64, String)>>,
    /// Crate-stripped call paths of the prebuilt eval hook.
    /// `register_eval_override`'s function argument plus the function
    /// `plain_eval_fn_addr` casts. Empty until the translator publishes
    /// the set harvested across the linked artefacts. Not a path-keyed
    /// map: membership is a short ordered list.
    eval_hook_graphs: parking_lot::RwLock<Vec<String>>,
    /// Root-stack effects of the linked artefacts analysed before this one:
    /// the crates whose every body was analysed, and the sorted paths of
    /// the bodies among them that can leave the shadow stack changed.
    /// Empty until the translator publishes them.
    root_stack_effects: parking_lot::RwLock<(Vec<String>, Vec<String>)>,
    /// Qualified path of the interpreter's exception carrier, published by
    /// the translator. A pointer to that ADT is the exception value `raise`
    /// stores, not a raw address. Empty until published.
    exception_carrier: parking_lot::RwLock<String>,
    /// Dedup id of `register_eval_override`'s first parameter, resolved
    /// once from every `FunDecl` this artefact carries (local and
    /// external). `None` once the scan has finished without a match.
    eval_fn_type_id: std::sync::OnceLock<Option<u64>>,
    /// Function paths, from this artefact or any it links against, whose
    /// body can leave shadow-stack slots published past its return or read
    /// slots below the depth it was entered at.  A caller that calls one
    /// cannot have its own root bracket scalar-replaced.  Sorted.
    stack_sensitive_fns: parking_lot::RwLock<Vec<String>>,
    /// Set once the set above is complete for this artefact.
    stack_sensitive_ready: std::sync::atomic::AtomicBool,
    /// Function paths whose own body opens and closes a `RootScope`.
    /// They rewind whatever they published, so a caller may treat the
    /// call as depth-neutral even when the path is also in
    /// `stack_sensitive_fns`.  Sorted.  Harvested in link order like
    /// the sensitive set, so a later crate sees earlier crates' answers.
    stack_depth_neutral_fns: parking_lot::RwLock<Vec<String>>,
    /// Function paths whose body can return with slots still published
    /// above its entry depth, or at a depth the walk could not bound.
    /// A subset of `stack_sensitive_fns`.  Sorted.  Harvested in link
    /// order like the sensitive set.
    stack_leaves_above_fns: parking_lot::RwLock<Vec<String>>,
    /// Function paths whose body can read or write a slot below its
    /// entry depth, or can end at a depth the walk could not bound.
    /// A subset of `stack_sensitive_fns`; a path may also be in
    /// `stack_leaves_above_fns`.  Sorted.  Harvested in link order like
    /// the sensitive set.
    stack_observes_fns: parking_lot::RwLock<Vec<String>>,
    /// Functions that read or write a slot whose index is one of their
    /// own parameters.  Sorted by path.  The `Vec<u8>` is 0-based
    /// positions in the callee's argument list.
    stack_param_slots_fns: parking_lot::RwLock<Vec<(String, Vec<u8>)>>,
    /// Function paths whose every reachable `Return` yields an own
    /// shadow-stack index (a lower bound relative to that body's entry
    /// depth).  Sorted.  Harvested in link order like the sensitive set.
    stack_returns_index_fns: parking_lot::RwLock<Vec<String>>,
    /// Trait-decl id → associated-type bindings of its unique impl.
    /// `trait_impls` is immutable after parse, so the map is built once.
    /// See [`TraitAssocIndex`].
    trait_assoc_index: std::sync::OnceLock<TraitAssocIndex>,
    /// ADT def_ids named as the `Self` type of a `Drop` impl, sorted.
    /// `trait_impls` is immutable after parse, so the set is built once.
    /// See [`Llbc::has_explicit_drop_impl`].
    drop_impl_owners: std::sync::OnceLock<Vec<u64>>,
}

/// Register-bank shape of a `#[repr(transparent)]` scalar wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransparentScalarKind {
    Signed,
    Unsigned,
    Bool,
    Float,
}

/// Charon-reader lookup index over `trait_impls`. It has no RPython/PyPy
/// owner. Iteration order is never observed: queries are by trait-decl id
/// and only a unique impl answers. The first `TraitType` row per assoc
/// wins, matching the linear scan this index replaced.
///
/// `trait decl id → Some(unique impl bindings)` or `None` when a second
/// impl of that trait was seen. Missing keys have no impl.
type TraitAssocIndex = std::collections::HashMap<u64, Option<UniqueTraitImpl>>;

#[derive(Debug)]
struct UniqueTraitImpl {
    impl_index: usize,
    by_assoc: std::collections::HashMap<serde_json::Value, AssocBind>,
}

#[derive(Debug)]
enum AssocBind {
    /// `types[entry].skip_binder.value` is present.
    Found(usize),
    /// The first `TraitType` row for this assoc has no value. A later
    /// row does not replace it.
    Absent,
}

fn tyref_dedup_id(ty: &crate::ullbc::TyRef) -> Option<u64> {
    match ty {
        crate::ullbc::TyRef::Dedup { id } => Some(*id),
        crate::ullbc::TyRef::Inline { value: (id, _) } => Some(*id),
        crate::ullbc::TyRef::Other(_) => None,
    }
}

fn build_trait_assoc_index(rows: &[serde_json::Value]) -> TraitAssocIndex {
    let mut index: TraitAssocIndex = std::collections::HashMap::new();
    for (impl_index, row) in rows.iter().enumerate() {
        let Some(trait_id) = row
            .get("impl_trait")
            .and_then(|impl_trait| impl_trait.get("id"))
            .and_then(serde_json::Value::as_u64)
        else {
            continue;
        };
        if let Some(slot) = index.get_mut(&trait_id) {
            *slot = None;
            continue;
        }
        let mut by_assoc = std::collections::HashMap::new();
        if let Some(entries) = row.get("types").and_then(serde_json::Value::as_array) {
            for (entry_index, entry) in entries.iter().enumerate() {
                let Some(kind) = entry
                    .get("kind")
                    .and_then(|kind| kind.get("TraitType"))
                    .and_then(serde_json::Value::as_array)
                else {
                    continue;
                };
                if kind.len() != 2 || by_assoc.contains_key(&kind[1]) {
                    continue;
                }
                let bind = if entry
                    .get("skip_binder")
                    .and_then(|binder| binder.get("value"))
                    .is_some()
                {
                    AssocBind::Found(entry_index)
                } else {
                    AssocBind::Absent
                };
                by_assoc.insert(kind[1].clone(), bind);
            }
        }
        index.insert(
            trait_id,
            Some(UniqueTraitImpl {
                impl_index,
                by_assoc,
            }),
        );
    }
    index
}

/// One hash-consed type body, kept as its raw JSON text and exploded to a
/// [`serde_json::Value`] only on first access.
///
/// The corpus holds ~208 k of these across the three pyre artefacts —
/// 66 MB of JSON that costs 1.60 GB as `Value` trees (21.4× measured),
/// while the consumers (`front::mir`, `front::result_exc`,
/// `front::option_try`) reach only the ids their lowered functions
/// mention.  Parsing on demand keeps the untouched majority at raw size.
#[derive(Debug)]
struct DedupBody {
    raw: Box<serde_json::value::RawValue>,
    parsed: std::sync::OnceLock<serde_json::Value>,
}

impl DedupBody {
    fn get(&self) -> Option<&serde_json::Value> {
        // A body that fails to re-parse cannot be projected by any
        // consumer, and every one of them already treats a missing id as
        // "not resolvable" — so a parse failure degrades to the same
        // `None` rather than aborting the load.
        if self.parsed.get().is_none() {
            let value = serde_json::from_str::<serde_json::Value>(self.raw.get()).ok()?;
            let _ = self.parsed.set(value);
        }
        self.parsed.get()
    }
}

impl Llbc {
    /// Load and parse a `.llbc` / `.ullbc` JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let bytes = std::fs::read(path.as_ref()).map_err(SchemaError::Io)?;
        Self::from_slice(&bytes)
    }

    /// Parse a `.llbc` / `.ullbc` artefact from an in-memory byte slice.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, SchemaError> {
        // Two single-purpose passes over `bytes`, never a full
        // `serde_json::Value` of the whole document (which costs
        // ~26× the input bytes as an exploded node tree):
        //   1. `collect_dedup_bodies` scans the raw bytes for inline
        //      `"Value":[id, body]` type occurrences, keeping only
        //      each small `body`'s raw text and discarding the rest.
        //   2. `from_slice` streams the bytes straight into the typed
        //      `LlbcFile` without the intermediate Value.
        // Peak settles at the larger of {bytes + dedup bodies} and
        // {bytes + LlbcFile}.
        let mut dedup_adt: Vec<(u64, u64)> = Vec::new();
        let mut dedup_body: Vec<(u64, DedupBody)> = Vec::new();
        let mut dedup_trait: Vec<(u64, DedupBody)> = Vec::new();
        let mut dedup_const: Vec<(u64, DedupBody)> = Vec::new();
        let mut dedup_layout: Vec<(u64, DedupBody)> = Vec::new();
        let mut span_bodies: Vec<(u64, ullbc::SpanData)> = Vec::new();
        collect_dedup_bodies(
            bytes,
            &mut dedup_adt,
            &mut dedup_body,
            &mut dedup_trait,
            &mut dedup_const,
            &mut dedup_layout,
            &mut span_bodies,
        );
        dedup_adt.sort_by_key(|&(id, _)| id);
        dedup_adt.dedup_by_key(|p| p.0);
        dedup_body.sort_by_key(|p| p.0);
        dedup_body.dedup_by_key(|p| p.0);
        dedup_trait.sort_by_key(|p| p.0);
        dedup_trait.dedup_by_key(|p| p.0);
        dedup_const.sort_by_key(|p| p.0);
        dedup_const.dedup_by_key(|p| p.0);
        dedup_layout.sort_by_key(|p| p.0);
        dedup_layout.dedup_by_key(|p| p.0);
        span_bodies.sort_by_key(|p| p.0);
        span_bodies.dedup_by_key(|p| p.0);
        let mut file: LlbcFile = serde_json::from_slice(bytes).map_err(SchemaError::Parse)?;
        ullbc::attach_promoted_inits(&mut file);
        Ok(Self {
            file,
            dedup_adt,
            dedup_body,
            dedup_trait,
            dedup_const,
            dedup_layout,
            span_bodies,
            transparent_scalar_kinds: parking_lot::RwLock::new(Vec::new()),
            foldable_const_lits: parking_lot::RwLock::new(Vec::new()),
            eval_hook_graphs: parking_lot::RwLock::new(Vec::new()),
            root_stack_effects: parking_lot::RwLock::new((Vec::new(), Vec::new())),
            exception_carrier: parking_lot::RwLock::new(String::new()),
            eval_fn_type_id: std::sync::OnceLock::new(),
            stack_sensitive_fns: parking_lot::RwLock::new(Vec::new()),
            stack_sensitive_ready: std::sync::atomic::AtomicBool::new(false),
            stack_depth_neutral_fns: parking_lot::RwLock::new(Vec::new()),
            stack_leaves_above_fns: parking_lot::RwLock::new(Vec::new()),
            stack_observes_fns: parking_lot::RwLock::new(Vec::new()),
            stack_param_slots_fns: parking_lot::RwLock::new(Vec::new()),
            stack_returns_index_fns: parking_lot::RwLock::new(Vec::new()),
            trait_assoc_index: std::sync::OnceLock::new(),
            drop_impl_owners: std::sync::OnceLock::new(),
        })
    }

    /// Merge transparent scalar shapes discovered across the linked LLBC set.
    /// Entries are kept sorted for deterministic, allocation-free lookup.
    pub fn register_transparent_scalar_kinds(
        &self,
        entries: impl IntoIterator<Item = (String, TransparentScalarKind)>,
    ) {
        let mut kinds = self.transparent_scalar_kinds.write();
        for (path, kind) in entries {
            match kinds.binary_search_by(|(known, _)| known.cmp(&path)) {
                Ok(index) => assert_eq!(
                    kinds[index].1, kind,
                    "transparent scalar type {path} has inconsistent linked definitions"
                ),
                Err(index) => kinds.insert(index, (path, kind)),
            }
        }
    }

    /// Store one folded initializer on the global `def_id` names.
    ///
    /// Two writes of the same id must carry the same literal. The id is
    /// this artefact's decl id, so a sibling global that renders the same
    /// `name_path` keeps its own slot.
    pub fn register_foldable_const_lit(&self, def_id: u64, encoded: String) {
        let mut lits = self.foldable_const_lits.write();
        match lits.binary_search_by_key(&def_id, |row| row.0) {
            Ok(index) => assert_eq!(
                lits[index].1, encoded,
                "foldable const def_id {def_id} has inconsistent linked definitions"
            ),
            Err(index) => lits.insert(index, (def_id, encoded)),
        }
    }

    /// Publish the prebuilt eval-hook family harvested from the linked
    /// artefacts. Later readers see this list; an empty publish clears it.
    pub fn set_eval_hook_graphs(&self, paths: Vec<String>) {
        *self.eval_hook_graphs.write() = paths;
    }

    /// Crate-stripped paths of `register_eval_override`'s target and of
    /// `plain_eval_fn_addr`'s function, when the translator has published
    /// them. Empty before that publish.
    pub fn eval_hook_graphs(&self) -> Vec<String> {
        self.eval_hook_graphs.read().clone()
    }

    /// Publish the root-stack effects harvested from other artefacts of the
    /// same translation input: `crates` names every crate whose bodies were
    /// analysed, `touching` the paths of those that can change the stack.
    /// Publish the exception carrier's qualified ADT path. Empty clears it.
    pub fn set_exception_carrier(&self, path: &str) {
        *self.exception_carrier.write() = path.to_string();
    }

    /// The published exception carrier path, or empty when none is named.
    pub fn exception_carrier(&self) -> String {
        self.exception_carrier.read().clone()
    }

    pub fn set_root_stack_effects(&self, crates: Vec<String>, mut touching: Vec<String>) {
        touching.sort();
        touching.dedup();
        *self.root_stack_effects.write() = (crates, touching);
    }

    /// Whether the body at `path` in crate `krate` can change the root
    /// stack, as published by [`Self::set_root_stack_effects`].  `None` when
    /// that crate was not analysed.
    pub fn root_stack_effect(&self, krate: &str, path: &str) -> Option<bool> {
        let effects = self.root_stack_effects.read();
        if !effects.0.iter().any(|c| c == krate) {
            return None;
        }
        Some(effects.1.binary_search_by(|p| p.as_str().cmp(path)).is_ok())
    }

    /// Whether any other artefact's root-stack effects were published here.
    pub fn has_root_stack_effects(&self) -> bool {
        !self.root_stack_effects.read().0.is_empty()
    }

    /// Dedup id of `register_eval_override`'s parameter type.
    ///
    /// Scans every `FunDecl` the artefact carries, including an external
    /// declaration whose body is `Opaque`. The id is computed once.
    pub fn eval_fn_type_id(&self) -> Option<u64> {
        *self.eval_fn_type_id.get_or_init(|| {
            self.file
                .translated
                .fun_decls
                .iter()
                .flatten()
                .find(|fd| {
                    fd.item_meta
                        .name_path()
                        .ends_with("::call::register_eval_override")
                })
                .and_then(|fd| fd.signature.inputs.first())
                .and_then(tyref_dedup_id)
        })
    }

    /// Record function paths whose shadow-stack effect a caller cannot see
    /// past.  See the `stack_sensitive_fns` field.
    pub fn register_stack_sensitive_fns(&self, paths: impl IntoIterator<Item = String>) {
        let mut known = self.stack_sensitive_fns.write();
        known.extend(paths);
        known.sort();
        known.dedup();
    }

    /// Declare the registered set complete: every function this artefact
    /// defines has been classified.
    pub fn mark_stack_sensitive_fns_complete(&self) {
        self.stack_sensitive_ready
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Whether [`mark_stack_sensitive_fns_complete`](Self::mark_stack_sensitive_fns_complete)
    /// ran.  Until it has, no callee's stack effect is known.
    pub fn stack_sensitive_fns_complete(&self) -> bool {
        self.stack_sensitive_ready
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Whether `path` was registered through
    /// [`register_stack_sensitive_fns`](Self::register_stack_sensitive_fns).
    pub fn is_stack_sensitive_fn(&self, path: &str) -> bool {
        self.stack_sensitive_fns
            .read()
            .binary_search_by(|known| known.as_str().cmp(path))
            .is_ok()
    }

    /// Record function paths whose body opens and closes a `RootScope`.
    pub fn register_stack_depth_neutral_fns(&self, paths: impl IntoIterator<Item = String>) {
        let mut known = self.stack_depth_neutral_fns.write();
        known.extend(paths);
        known.sort();
        known.dedup();
    }

    /// Whether `path` was registered through
    /// [`register_stack_depth_neutral_fns`](Self::register_stack_depth_neutral_fns).
    pub fn is_stack_depth_neutral_fn(&self, path: &str) -> bool {
        self.stack_depth_neutral_fns
            .read()
            .binary_search_by(|known| known.as_str().cmp(path))
            .is_ok()
    }

    /// Record function paths that leave slots published above entry.
    pub fn register_stack_leaves_above_fns(&self, paths: impl IntoIterator<Item = String>) {
        let mut known = self.stack_leaves_above_fns.write();
        known.extend(paths);
        known.sort();
        known.dedup();
    }

    /// Whether `path` was registered through
    /// [`register_stack_leaves_above_fns`](Self::register_stack_leaves_above_fns).
    pub fn is_stack_leaves_above_fn(&self, path: &str) -> bool {
        self.stack_leaves_above_fns
            .read()
            .binary_search_by(|known| known.as_str().cmp(path))
            .is_ok()
    }

    /// Record function paths that can touch a slot below their entry.
    pub fn register_stack_observes_fns(&self, paths: impl IntoIterator<Item = String>) {
        let mut known = self.stack_observes_fns.write();
        known.extend(paths);
        known.sort();
        known.dedup();
    }

    /// Whether `path` was registered through
    /// [`register_stack_observes_fns`](Self::register_stack_observes_fns).
    pub fn is_stack_observes_fn(&self, path: &str) -> bool {
        self.stack_observes_fns
            .read()
            .binary_search_by(|known| known.as_str().cmp(path))
            .is_ok()
    }

    /// Record functions that index the shadow stack through a parameter.
    pub fn register_stack_param_slots_fns(
        &self,
        rows: impl IntoIterator<Item = (String, Vec<u8>)>,
    ) {
        let mut known = self.stack_param_slots_fns.write();
        known.extend(rows);
        known.sort_by(|a, b| a.0.cmp(&b.0));
        let mut merged: Vec<(String, Vec<u8>)> = Vec::new();
        for (path, slots) in known.drain(..) {
            if let Some(last) = merged.last_mut()
                && last.0 == path
            {
                last.1.extend(slots);
                last.1.sort_unstable();
                last.1.dedup();
                continue;
            }
            let mut slots = slots;
            slots.sort_unstable();
            slots.dedup();
            merged.push((path, slots));
        }
        *known = merged;
    }

    /// Parameter positions `path` uses as shadow-stack indices, if any.
    pub fn stack_param_slots(&self, path: &str) -> Option<Vec<u8>> {
        let known = self.stack_param_slots_fns.read();
        let i = known
            .binary_search_by(|(known, _)| known.as_str().cmp(path))
            .ok()?;
        Some(known[i].1.clone())
    }

    /// Record function paths whose every reachable return is an own
    /// shadow-stack index relative to entry.
    pub fn register_stack_returns_index_fns(&self, paths: impl IntoIterator<Item = String>) {
        let mut known = self.stack_returns_index_fns.write();
        known.extend(paths);
        known.sort();
        known.dedup();
    }

    /// Whether `path` was registered through
    /// [`register_stack_returns_index_fns`](Self::register_stack_returns_index_fns).
    pub fn is_stack_returns_index_fn(&self, path: &str) -> bool {
        self.stack_returns_index_fns
            .read()
            .binary_search_by(|known| known.as_str().cmp(path))
            .is_ok()
    }

    /// The folded initializer stored on `def_id`, if this artefact has one.
    pub fn foldable_const_lit(&self, def_id: u64) -> Option<String> {
        let lits = self.foldable_const_lits.read();
        let index = lits.binary_search_by_key(&def_id, |row| row.0).ok()?;
        Some(lits[index].1.clone())
    }

    /// Look up the linked scalar shape of an opaque transparent declaration.
    pub fn transparent_scalar_kind(&self, path: &str) -> Option<TransparentScalarKind> {
        let kinds = self.transparent_scalar_kinds.read();
        let index = kinds
            .binary_search_by(|(known, _)| known.as_str().cmp(path))
            .ok()?;
        Some(kinds[index].1)
    }

    /// Resolve a Charon `Deduplicated: <id>` type reference to the
    /// underlying ADT `def_id` (suitable for [`Self::type_by_id`]).
    /// Returns `None` for non-ADT types (primitives, references,
    /// tuples) and for ids whose inline form never appeared in the
    /// LLBC.  See the [`Self::dedup_adt`] field doc for context.
    pub fn dedup_to_adt_def_id(&self, id: u64) -> Option<u64> {
        self.dedup_adt
            .binary_search_by_key(&id, |&(d, _)| d)
            .ok()
            .map(|i| self.dedup_adt[i].1)
    }

    /// Resolve a Charon `Deduplicated: <id>` reference to its
    /// underlying inline body (a `serde_json::Value` of the same
    /// shape Charon emits inline for a `Value: [id, body]`).
    /// Returns `None` for ids whose inline form never appeared in
    /// this LLBC.  See the [`Self::dedup_body`] field doc for
    /// context.
    pub fn dedup_body(&self, id: u64) -> Option<&serde_json::Value> {
        let i = self.dedup_body.binary_search_by_key(&id, |p| p.0).ok()?;
        self.dedup_body[i].1.get()
    }

    /// A trait-ref body stored under its own hash-cons id space.
    pub fn dedup_trait_body(&self, id: u64) -> Option<&serde_json::Value> {
        let i = self.dedup_trait.binary_search_by_key(&id, |p| p.0).ok()?;
        self.dedup_trait[i].1.get()
    }

    pub fn dedup_const_body(&self, id: u64) -> Option<&serde_json::Value> {
        let i = self.dedup_const.binary_search_by_key(&id, |p| p.0).ok()?;
        self.dedup_const[i].1.get()
    }

    /// The `ConstantExprKind` of a `ConstantExpr`. The expression is
    /// `[kind, ty]`, spelled inline, as `{"Value": [id, [kind, ty]]}` at its
    /// first occurrence, or as `{"Deduplicated": id}` afterwards.
    pub fn const_expr_kind(&self, expr: &serde_json::Value) -> Option<serde_json::Value> {
        self.const_expr_body(expr)?.first().cloned()
    }

    /// The `ty` of a `ConstantExpr`, the second half of its `[kind, ty]`.
    pub fn const_expr_ty<'a>(
        &'a self,
        expr: &'a serde_json::Value,
    ) -> Option<&'a serde_json::Value> {
        self.const_expr_body(expr)?.get(1)
    }

    fn const_expr_body<'a>(
        &'a self,
        expr: &'a serde_json::Value,
    ) -> Option<&'a Vec<serde_json::Value>> {
        let body = if let Some(pair) = expr.get("Value").and_then(serde_json::Value::as_array) {
            pair.get(1)?
        } else if let Some(id) = expr.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            self.dedup_const_body(id)?
        } else {
            expr
        };
        body.as_array()
    }

    /// The literal of a `ConstantExpr` in `Literal` form: an `Integer` kind
    /// becomes `{"Scalar": ...}`; `Bool` / `Char` / `Float` / `Str` /
    /// `ByteStr` are already literals. `None` for any other kind.
    pub fn const_expr_literal(&self, expr: &serde_json::Value) -> Option<serde_json::Value> {
        let kind = self.const_expr_kind(expr)?;
        if let Some(int) = kind.get("Integer") {
            return Some(serde_json::json!({ "Scalar": int }));
        }
        ["Bool", "Char", "Float", "Str", "ByteStr"]
            .iter()
            .any(|lit| kind.get(*lit).is_some())
            .then_some(kind)
    }

    /// A layout scalar `{"Constant": ...}`. Its ids are not MIR const ids.
    pub fn layout_scalar_body(&self, id: u64) -> Option<&serde_json::Value> {
        let i = self.dedup_layout.binary_search_by_key(&id, |p| p.0).ok()?;
        self.dedup_layout[i].1.get()
    }

    /// Resolve `span`. [`ullbc::SpanRef::Deduplicated`] reads the span table
    /// built at load; an id that never appeared inline is `None`.
    pub fn span_data<'a>(&'a self, span: &'a ullbc::SpanRef) -> Option<&'a ullbc::SpanData> {
        match span {
            ullbc::SpanRef::Inline(data) => Some(data),
            ullbc::SpanRef::Deduplicated(id) => {
                let i = self.span_bodies.binary_search_by_key(id, |p| p.0).ok()?;
                Some(&self.span_bodies[i].1)
            }
        }
    }

    /// Look up a local-crate function whose name ends with `::<name>`.
    pub fn local_fn(&self, name: &str) -> Option<&FunDecl> {
        let suffix = format!("::{name}");
        for f in self.iter_local_fns() {
            let path = f.item_meta.name_path();
            if path == name || path.ends_with(&suffix) {
                return Some(f);
            }
        }
        None
    }

    /// Look up a `FunDecl` by its Charon `def_id`. The `fun_decls`
    /// array is indexed by `def_id` (verified against extracted
    /// corpora), so this is an O(1) bounds-checked lookup.
    pub fn fn_by_id(&self, def_id: u64) -> Option<&FunDecl> {
        self.file
            .translated
            .fun_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Look up a `GlobalDecl` by its Charon `def_id`. Same indexing
    /// invariant as [`fn_by_id`].
    pub fn global_by_id(&self, def_id: u64) -> Option<&GlobalDecl> {
        self.file
            .translated
            .global_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Look up a `TypeDecl` by its Charon `def_id`. Same indexing
    /// invariant as [`fn_by_id`].
    pub fn type_by_id(&self, def_id: u64) -> Option<&TypeDecl> {
        self.file
            .translated
            .type_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// The source path a [`crate::ullbc::SpanData`]'s `file_id` names.
    ///
    /// Charon writes the table in id order, so the id is tried as an
    /// index first; the scan behind it is what keeps a table that ever
    /// stops being dense from silently returning a neighbouring file's
    /// path.
    pub fn file_path(&self, file_id: u64) -> Option<&str> {
        let files = &self.file.translated.files;
        let row = match files.get(file_id as usize) {
            Some(row) if row.id == file_id => row,
            _ => files.iter().find(|row| row.id == file_id)?,
        };
        // A row that spells its name as a bare string is answered before
        // the object form is tried: `as_object` on a string is `None`, so
        // asking for the object first ends the lookup and no fallback
        // behind that question can run.
        if let Some(path) = row.name.as_str() {
            return Some(path);
        }
        // `FileName` is otherwise a single-variant object; the variant names
        // the provenance and the payload is the path either way.
        row.name.as_object()?.values().next()?.as_str()
    }

    /// Look up a `TraitDecl` by its Charon `def_id`.
    pub fn trait_by_id(&self, def_id: u64) -> Option<&TraitDecl> {
        self.file
            .translated
            .trait_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// The name of the `method_id`-th method of trait `trait_id`
    /// (`TranslatedCrate::assoc_item_name`), falling back to the trait
    /// declaration's own `methods` row for an artefact without the
    /// `assoc_item_names` table.
    pub fn trait_method_name(&self, trait_id: u64, method_id: u64) -> Option<&str> {
        let table = &self.file.translated.assoc_item_names;
        if let Some(Some(names)) = table.get(trait_id as usize) {
            return names.methods.get(method_id as usize).map(String::as_str);
        }
        self.trait_by_id(trait_id)?
            .methods
            .get(method_id as usize)?
            .pointer("/skip_binder/name")?
            .as_str()
    }

    /// Iterate over every present `TypeDecl`.
    pub fn iter_type_decls(&self) -> impl Iterator<Item = &TypeDecl> {
        self.file
            .translated
            .type_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Iterate over every present `TraitDecl`.
    pub fn iter_trait_decls(&self) -> impl Iterator<Item = &TraitDecl> {
        self.file
            .translated
            .trait_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// The raw `trait_impls` table (schema-opaque; entries may be
    /// `null`).  Consumed by the front-end's trait-associated-type
    /// resolution: an `impl Trait for T` entry binds each associated
    /// type (`kind: {"TraitType": [trait_id, idx]}`) to a concrete
    /// type in its `types[].skip_binder.value`.
    pub fn trait_impls_raw(&self) -> &[serde_json::Value] {
        &self.file.translated.trait_impls
    }

    /// The type value the unique impl of `trait_decl_id` binds `assoc` to.
    ///
    /// `None` when that trait has zero impls or more than one, when no
    /// `types[]` entry selects `assoc`, or when the first selecting entry
    /// has no `skip_binder.value`. A later duplicate does not override the
    /// first, matching one forward scan of [`Self::trait_impls_raw`].
    pub fn unique_trait_assoc_value(
        &self,
        trait_decl_id: u64,
        assoc: &serde_json::Value,
    ) -> Option<&serde_json::Value> {
        let (impl_index, entry_index) = {
            let index = self
                .trait_assoc_index
                .get_or_init(|| build_trait_assoc_index(&self.file.translated.trait_impls));
            let bindings = index.get(&trait_decl_id)?.as_ref()?;
            match bindings.by_assoc.get(assoc)? {
                AssocBind::Absent => return None,
                AssocBind::Found(entry_index) => (bindings.impl_index, *entry_index),
            }
        };
        self.file
            .translated
            .trait_impls
            .get(impl_index)?
            .get("types")?
            .as_array()?
            .get(entry_index)?
            .get("skip_binder")?
            .get("value")
    }

    /// Whether a `trait_impls` row implements `Drop` for the ADT
    /// `adt_def_id`: its first generic type names that ADT, and its trait
    /// is `core::ops::drop::Drop` or has no declaration here to prove it
    /// is not. `adt_of` reads the ADT a type expression names; the owner
    /// set is built with the first caller's `adt_of`.
    pub fn has_explicit_drop_impl(
        &self,
        adt_def_id: u64,
        adt_of: impl Fn(&serde_json::Value) -> Option<u64>,
    ) -> bool {
        self.drop_impl_owners
            .get_or_init(|| {
                let mut owners: Vec<u64> = self
                    .file
                    .translated
                    .trait_impls
                    .iter()
                    .filter_map(|row| {
                        let impl_trait = row.get("impl_trait")?;
                        let owner = impl_trait
                            .get("generics")
                            .and_then(|generics| generics.get("types"))
                            .and_then(serde_json::Value::as_array)
                            .and_then(|types| types.first())
                            .and_then(|owner| adt_of(owner))?;
                        impl_trait
                            .get("id")
                            .and_then(serde_json::Value::as_u64)
                            .and_then(|trait_id| self.trait_by_id(trait_id))
                            .is_none_or(|decl| {
                                decl.item_meta.name_path() == "core::ops::drop::Drop"
                            })
                            .then_some(owner)
                    })
                    .collect();
                owners.sort_unstable();
                owners.dedup();
                owners
            })
            .binary_search(&adt_def_id)
            .is_ok()
    }

    /// The `trait_impls` row whose `def_id` is `id` — the impl block
    /// [`crate::ullbc::ItemMeta::trait_impl_id`] names.
    ///
    /// The id is tried as an index first, as the sibling `*_by_id`
    /// accessors do, and the row is only accepted when it agrees about
    /// its own `def_id`; a table that ever stops being dense then falls
    /// to the scan instead of returning a neighbouring impl. Returning
    /// the wrong impl here would bind one type's prebuilt address to
    /// another's, so the disagreement is checked rather than assumed
    /// away.
    pub fn trait_impl_by_id(&self, def_id: u64) -> Option<&serde_json::Value> {
        let says_id = |row: &serde_json::Value| {
            row.get("def_id").and_then(serde_json::Value::as_u64) == Some(def_id)
        };
        let rows = &self.file.translated.trait_impls;
        match rows.get(def_id as usize) {
            Some(row) if says_id(row) => Some(row),
            _ => rows.iter().find(|row| says_id(row)),
        }
    }

    /// Iterate over every present `FunDecl` (skipping opaque `null` entries).
    ///
    /// Includes external declarations (`is_local == false`, body `Opaque`).
    /// The name is historical; [`Self::iter_fun_decls`] is the same walk.
    pub fn iter_local_fns(&self) -> impl Iterator<Item = &FunDecl> {
        self.iter_fun_decls()
    }

    /// Every `FunDecl` this artefact carries, local or external.
    pub fn iter_fun_decls(&self) -> impl Iterator<Item = &FunDecl> {
        self.file
            .translated
            .fun_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Iterate over every present `GlobalDecl` (skipping opaque `null`
    /// entries).  Used by the hint harvester to read the macro-emitted
    /// `_elidable_function_<NAME>` / `_jit_*_<NAME>` marker consts.
    pub fn iter_global_decls(&self) -> impl Iterator<Item = &GlobalDecl> {
        self.file
            .translated
            .global_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Crate name (the `crate_name` field from `.llbc.translated`).
    pub fn crate_name(&self) -> &str {
        &self.file.translated.crate_name
    }

    /// The one pointer width, in bytes, shared by every extraction target in
    /// this artefact.  RPython's translator carries this on its target system
    /// configuration; Charon's ordered `target_information` rows are the
    /// corresponding source of truth here.  Missing or conflicting rows stay
    /// unresolved so width-sensitive lowerings fail closed.
    pub fn target_pointer_size(&self) -> Option<u8> {
        let mut rows = self.file.translated.target_information.iter();
        let width = rows.next()?.value.target_pointer_size;
        rows.all(|row| row.value.target_pointer_size == width)
            .then_some(width)
    }
}

/// Scan the raw LLBC bytes for every inline
/// `"Value":[id, body]` occurrence, recording the first
/// `body` seen per `id` into `bodies` (the generic dedup-id → body
/// index) and, when the body decodes as a nominal `{"Adt": {"id":
/// <def_id>, "builtin": null}}`, also into `adt` (the dedup-id → ADT def_id
/// index for fast Adt resolution).  Used during [`Llbc::from_slice`].
///
/// Operating on the raw bytes — rather than a fully materialised
/// `serde_json::Value` of the whole document — keeps peak memory at the
/// few thousand small type bodies actually deduplicated, instead of the
/// ~26× blow-up of an exploded Value tree. The byte scan finds nested
/// occurrences automatically (each is its own literal in the text), and
/// a `seen` set keeps only the first body per `id` — every occurrence of
/// a hash-consed `id` carries an identical body, so the choice is
/// immaterial and the post-sort `dedup_by_key` result is unchanged.
fn collect_dedup_bodies(
    bytes: &[u8],
    adt: &mut Vec<(u64, u64)>,
    bodies: &mut Vec<(u64, DedupBody)>,
    traits: &mut Vec<(u64, DedupBody)>,
    consts: &mut Vec<(u64, DedupBody)>,
    layouts: &mut Vec<(u64, DedupBody)>,
    spans: &mut Vec<(u64, ullbc::SpanData)>,
) {
    // The artefact is UTF-8 JSON; on the off chance it is not, there are
    // no Value entries to find and the typed parse will fail
    // loudly downstream.
    let Ok(text) = std::str::from_utf8(bytes) else {
        return;
    };
    // `SerDedup::Value` is the wrapper for every hash-consed value, so the
    // key is shared by types, trait refs, constants, and spans. Each kind
    // numbers its ids from zero, so the tables stay separate.
    const KEY: &str = "\"Value\":";
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut seen_trait: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut seen_const: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut seen_layout: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut seen_span: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for (off, _) in text.match_indices(KEY) {
        let val_start = off + KEY.len();
        // The value after the key is the `[id, body]` array; deserialize
        // exactly that one value (the deserializer stops at the array's
        // close, ignoring the trailing document).
        let mut de = serde_json::Deserializer::from_slice(&bytes[val_start..]);
        let Ok((id, raw)) = <(u64, Box<serde_json::value::RawValue>)>::deserialize(&mut de) else {
            continue;
        };
        match classify_value_body(&raw) {
            ValueBody::Ty => {
                if seen.insert(id) {
                    if let Some(def_id) = adt_def_id_from_ty_body(&raw) {
                        adt.push((id, def_id));
                    }
                    bodies.push((
                        id,
                        DedupBody {
                            raw,
                            parsed: std::sync::OnceLock::new(),
                        },
                    ));
                }
            }
            ValueBody::Trait => {
                if seen_trait.insert(id) {
                    traits.push((
                        id,
                        DedupBody {
                            raw,
                            parsed: std::sync::OnceLock::new(),
                        },
                    ));
                }
            }
            ValueBody::Const => {
                if seen_const.insert(id) {
                    consts.push((
                        id,
                        DedupBody {
                            raw,
                            parsed: std::sync::OnceLock::new(),
                        },
                    ));
                }
            }
            ValueBody::Layout => {
                if seen_layout.insert(id) {
                    layouts.push((
                        id,
                        DedupBody {
                            raw,
                            parsed: std::sync::OnceLock::new(),
                        },
                    ));
                }
            }
            ValueBody::Span => {
                if seen_span.insert(id)
                    && let Some(data) = ullbc::span_data_from_body(&raw)
                {
                    spans.push((id, data));
                }
            }
            ValueBody::Other => {}
        }
    }
}

enum ValueBody {
    Ty,
    Trait,
    Const,
    /// `{"Constant": ...}` layout scalar. Ids restart apart from const exprs.
    Layout,
    Span,
    Other,
}

/// One probe of the body's first key. An array body is a constant
/// expression (`[literal, ty]`, including `Str` / `ByteStr` / `FnDef` /
/// `Global` and the other const kinds). A layout scalar is the object
/// `{"Constant": {"Value": [id, [literal, ty]]}}` and uses its own id
/// space. An object body is otherwise a type, a trait ref (`"kind"`),
/// or a span (`"data"`). A string body is a payload-free type kind,
/// `"Never"`.
fn classify_value_body(raw: &serde_json::value::RawValue) -> ValueBody {
    const TY_KINDS: &[&str] = &[
        "Scalar",
        "Array",
        "Slice",
        "Adt",
        "Ref",
        "RawPtr",
        "FnDef",
        "FnPtr",
        "DynTrait",
        "Pattern",
        "Never",
        "TypeVar",
        "TraitType",
        "PtrMetadata",
    ];
    let text = raw.get().trim_start();
    let Some(first) = text.as_bytes().first().copied() else {
        return ValueBody::Other;
    };
    if first == b'[' {
        return ValueBody::Const;
    }
    if first == b'"' {
        return match serde_json::from_str::<&str>(text) {
            Ok(kind) if TY_KINDS.contains(&kind) => ValueBody::Ty,
            _ => ValueBody::Other,
        };
    }
    if first != b'{' {
        return ValueBody::Other;
    }
    match first_json_key(text) {
        Some("data") => ValueBody::Span,
        Some("kind") => ValueBody::Trait,
        Some("Constant") => ValueBody::Layout,
        Some(key) if TY_KINDS.contains(&key) => ValueBody::Ty,
        _ => ValueBody::Other,
    }
}

fn first_json_key(text: &str) -> Option<&str> {
    let rest = text.trim_start().strip_prefix('{')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// Project a type-expression body to its underlying ADT `def_id`,
/// when the body has shape `{"Adt": {"id": <def_id>, "builtin": …}}`.
/// Returns `None` for non-ADT bodies (`Scalar`, `Ref`, …).
///
/// Reads the raw text through a narrow typed projection rather than a
/// `Value` tree: this runs once per hash-consed id at load time, and
/// materialising all of them is what the [`DedupBody`] laziness avoids.
fn adt_def_id_from_ty_body(raw: &serde_json::value::RawValue) -> Option<u64> {
    #[derive(Deserialize)]
    struct Body {
        #[serde(rename = "Adt")]
        adt: Adt,
    }
    /// `builtin` is non-null for the tuple / `str` / `Box` decls. Tuple
    /// and `str` have no nominal owner; `Box` is the nominal
    /// `alloc::boxed::Box`.
    #[derive(Deserialize)]
    struct Adt {
        id: u64,
        #[serde(default)]
        builtin: Option<String>,
    }
    serde_json::from_str::<Body>(raw.get())
        .ok()
        .filter(|b| matches!(b.adt.builtin.as_deref(), None | Some("Box")))
        .map(|b| b.adt.id)
}

/// Errors produced when loading / parsing a `.llbc` artefact.
#[derive(Debug)]
pub enum SchemaError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    Decode(String),
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaError::Io(e) => write!(f, "io: {e}"),
            SchemaError::Parse(e) => write!(f, "parse: {e}"),
            SchemaError::Decode(s) => write!(f, "decode: {s}"),
        }
    }
}

impl std::error::Error for SchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SchemaError::Io(e) => Some(e),
            SchemaError::Parse(e) => Some(e),
            SchemaError::Decode(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llbc(files: &str) -> Llbc {
        let doc = format!(
            r#"{{"charon_version":"t","has_errors":false,
                "translated":{{"crate_name":"c","fun_decls":[],"files":{files}}}}}"#
        );
        Llbc::from_slice(doc.as_bytes()).expect("fixture parses")
    }

    #[test]
    fn a_file_id_names_the_source_path_its_span_was_written_against() {
        let l = llbc(
            r#"[{"id":0,"name":{"Local":"a/lib.rs"},"contents":"fn a() {}"},
                         {"id":1,"name":{"Local":"a/mod.rs"},"contents":"fn b() {}"}]"#,
        );
        assert_eq!(l.file_path(0), Some("a/lib.rs"));
        assert_eq!(l.file_path(1), Some("a/mod.rs"));
        assert_eq!(l.file_path(2), None);
    }

    #[test]
    fn a_table_that_is_not_dense_is_searched_rather_than_indexed() {
        // Position 0 holds id 7, so indexing by id would answer with a
        // neighbour's path -- the one wrong answer worth a scan.
        let l = llbc(
            r#"[{"id":7,"name":{"Local":"seven.rs"}},
                         {"id":3,"name":{"Local":"three.rs"}}]"#,
        );
        assert_eq!(l.file_path(7), Some("seven.rs"));
        assert_eq!(l.file_path(3), Some("three.rs"));
        assert_eq!(l.file_path(0), None);
    }

    #[test]
    fn an_unknown_filename_variant_still_yields_its_path() {
        // Charon may rename the provenance variant; the payload is the
        // path whatever the variant is called.
        let l = llbc(r#"[{"id":0,"name":{"Virtual":"v.rs"}}]"#);
        assert_eq!(l.file_path(0), Some("v.rs"));
    }

    #[test]
    fn a_name_written_as_a_bare_string_is_read_as_the_path() {
        // Not the object form, so the object lookup answers `None`; the
        // string has to be tried first or this row reads as unnamed.
        let l = llbc(r#"[{"id":0,"name":"bare.rs"}]"#);
        assert_eq!(l.file_path(0), Some("bare.rs"));
    }

    #[test]
    fn a_layout_constant_reads_its_id_in_the_const_table() {
        // Layout expressions and constant expressions number their ids
        // apart. Layout expression 0 is `Constant(const 1)` = 0, const 0 is
        // 3; a size of `Constant(const 0)` is 3, not layout 0's value.
        let usize_ty = r#"{"Scalar":{"Integer":{"Unsigned":"Usize"}}}"#;
        let lit = |n: u32| format!(r#"[{{"Integer":{{"Unsigned":["Usize","{n}"]}}}},{usize_ty}]"#);
        let doc = format!(
            r#"{{"charon_version":"t","has_errors":false,
            "translated":{{"crate_name":"c","fun_decls":[],"type_decls":[{{
                "def_id":0,
                "item_meta":{{"name":[{{"Ident":["S",0]}}],
                    "span":{{"data":{{"file_id":0,"beg":{{"line":1,"col":0}},"end":{{"line":1,"col":1}}}}}},
                    "source_text":null,
                    "attr_info":{{"attributes":[],"inline":null,"rename":null,"public":true}},
                    "is_local":true}},
                "kind":{{"Struct":[]}},
                "layout":[{{"key":"t","value":{{
                    "size":{{"chosen":{{"Value":[7,{{"Constant":{{"Deduplicated":0}}}}]}}}},
                    "align":{{"chosen":{{"Deduplicated":0}}}},
                    "variant_layouts":[]}}}}]
            }}]}},
            "pad":[{{"Value":[0,{{"Constant":{{"Value":[1,{one}]}}}}]}},
                   {{"Value":[0,{three}]}}]}}"#,
            one = lit(0),
            three = lit(3),
        );
        let l = Llbc::from_slice(doc.as_bytes()).expect("fixture parses");
        let layout = l
            .type_by_id(0)
            .expect("S")
            .layout_for_target(&l, "t")
            .expect("layout");
        assert_eq!(layout.size, Some(3));
        assert_eq!(layout.align, Some(0));
    }

    #[test]
    fn a_deduplicated_field_offset_resolves_like_size() {
        let usize_ty = r#"{"Scalar":{"Integer":{"Unsigned":"Usize"}}}"#;
        let lit = |n: u32| format!(r#"[{{"Integer":{{"Unsigned":["Usize","{n}"]}}}},{usize_ty}]"#);
        let doc = format!(
            r#"{{"charon_version":"t","has_errors":false,
            "translated":{{"crate_name":"c","fun_decls":[],"type_decls":[{{
                "def_id":0,
                "item_meta":{{"name":[{{"Ident":["S",0]}}],
                    "span":{{"data":{{"file_id":0,"beg":{{"line":1,"col":0}},"end":{{"line":1,"col":1}}}}}},
                    "source_text":null,
                    "attr_info":{{"attributes":[],"inline":null,"rename":null,"public":true}},
                    "is_local":true}},
                "kind":{{"Struct":[]}},
                "layout":[{{"key":"t","value":{{
                    "size":8,
                    "variant_layouts":[{{"field_offsets":[
                        {{"chosen":{{"Deduplicated":4}}}},
                        8
                    ]}}]}}}}]
            }}]}},
            "pad":[{{"Value":[4,{{"Constant":{{"Value":[1,{sixteen}]}}}}]}}]}}"#,
            sixteen = lit(16),
        );
        let l = Llbc::from_slice(doc.as_bytes()).expect("fixture parses");
        let layout = l
            .type_by_id(0)
            .expect("S")
            .layout_for_target(&l, "t")
            .expect("layout");
        assert_eq!(layout.struct_field_offset(0), Some(16));
        assert_eq!(layout.struct_field_offset(1), Some(8));
    }

    #[test]
    fn a_null_chosen_field_offset_has_no_layout() {
        // A generic enum whose payload seat is `chosen: null` and whose
        // discriminator was not recorded. The whole layout stays absent.
        let doc = r#"{"charon_version":"t","has_errors":false,
            "translated":{"crate_name":"c","fun_decls":[],"type_decls":[{
                "def_id":0,
                "item_meta":{"name":[{"Ident":["Option",0]}],
                    "span":{"data":{"file_id":0,"beg":{"line":1,"col":0},"end":{"line":1,"col":1}}},
                    "source_text":null,
                    "attr_info":{"attributes":[],"inline":null,"rename":null,"public":true},
                    "is_local":false},
                "kind":{"Enum":[]},
                "layout":[{"key":"t","value":{
                    "size":{"chosen":null},
                    "discriminator":null,
                    "variant_layouts":[
                        {"field_offsets":[]},
                        {"field_offsets":[{"guarantee":{"GuaranteedAlignment":{"Deduplicated":720}},"chosen":null}]}
                    ]
                }}]
            }]}}"#;
        let l = Llbc::from_slice(doc.as_bytes()).expect("fixture parses");
        assert!(
            l.type_by_id(0)
                .expect("decl")
                .layout_for_target(&l, "t")
                .is_none()
        );
    }

    #[test]
    fn a_payload_free_type_kind_is_a_string_body() {
        // `!` hash-conses as `{"Value": [id, "Never"]}`; a monomorphized
        // `ControlFlow<Result<!, E>, T>` names it among its instance
        // arguments, and an unresolved id renders that owner wrong.
        let doc = r#"{"charon_version":"t","has_errors":false,
            "translated":{"crate_name":"c","fun_decls":[]},
            "pad":[{"Value":[3,"Never"]},{"Value":[4,"NotATypeKind"]}]}"#;
        let l = Llbc::from_slice(doc.as_bytes()).expect("fixture parses");
        assert_eq!(l.dedup_body(3), Some(&serde_json::json!("Never")));
        assert_eq!(l.dedup_body(4), None);
    }

    #[test]
    fn dedup_switch_arms_decode_bool_if_and_int_scalar() {
        let span = serde_json::json!({"Untagged": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}, "generated_from_span": null}});
        let meta = |name: &str| {
            serde_json::json!({
                "name": [{"Ident": [name, 0]}],
                "span": span,
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
            })
        };
        let ret = serde_json::json!({"statements": [], "terminator": {"kind": "Return"}});
        let fun = |id: u64, name: &str, term: serde_json::Value, extra: Vec<serde_json::Value>| {
            let mut body =
                vec![serde_json::json!({"statements": [], "terminator": {"kind": term}})];
            body.extend(extra);
            serde_json::json!({
                "def_id": id,
                "item_meta": meta(name),
                "signature": {"is_unsafe": false, "inputs": [], "output": {"Deduplicated": 0}},
                "body": {"Unstructured": {"span": span, "locals": {"arg_count": 0, "locals": []}, "body": body}}
            })
        };
        let scrut = serde_json::json!({"Value": {"Copy": {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}}}});
        let bool_term = serde_json::json!({"Switch": {"data": {
            "scrutinee": scrut,
            "branches": [[{"Deduplicated": 7}, 0], [{"Deduplicated": 8}, 1]],
            "fallback": 1
        }, "branches": [1, 2]}});
        let int_term = serde_json::json!({"Switch": {"data": {
            "scrutinee": scrut,
            "branches": [[{"Deduplicated": 9}, 0], [{"Deduplicated": 10}, 1]],
            "fallback": 2
        }, "branches": [1, 2, 3]}});
        let doc = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {"crate_name": "c", "fun_decls": [
                fun(0, "bool_switch", bool_term, vec![ret.clone(), ret.clone()]),
                fun(1, "int_switch", int_term, vec![ret.clone(), ret.clone(), ret])
            ]},
            "pad": [
                {"Value": [7, [{"Bool": true}, {"Deduplicated": 0}]]},
                {"Value": [8, [{"Bool": false}, {"Deduplicated": 0}]]},
                {"Value": [9, [{"Integer": {"Signed": ["Isize", "0"]}}, {"Deduplicated": 0}]]},
                {"Value": [10, [{"Integer": {"Signed": ["Isize", "1"]}}, {"Deduplicated": 0}]]},
                {"Value": [4, {"data": {"file_id": 0, "beg": {"line": 4, "col": 1}, "end": {"line": 4, "col": 2}}, "generated_from_span": null}]}
            ]
        });
        let bytes = serde_json::to_vec(&doc).unwrap();
        let llbc = Llbc::from_slice(&bytes).expect("fixture");
        let bool_fn = llbc.fn_by_id(0).unwrap().unstructured().unwrap();
        match bool_fn.body[0].term(&llbc).unwrap() {
            ullbc::TermKind::Switch { targets, .. } => match targets {
                ullbc::SwitchTargets::If(1, 2) => {}
                other => panic!("bool arm was not If: {other:?}"),
            },
            other => panic!("bool switch: {other:?}"),
        }
        let int_fn = llbc.fn_by_id(1).unwrap().unstructured().unwrap();
        match int_fn.body[0].term(&llbc).unwrap() {
            ullbc::TermKind::Switch { targets, .. } => match targets {
                ullbc::SwitchTargets::SwitchInt(_, arms, 3) => {
                    assert_eq!(arms.len(), 2);
                    assert!(arms[0].0.get("Scalar").is_some());
                }
                other => panic!("int arm was not SwitchInt: {other:?}"),
            },
            other => panic!("int switch: {other:?}"),
        }
        let missing = ullbc::SpanRef::Deduplicated(99);
        assert!(llbc.span_data(&missing).is_none());
        let present = ullbc::SpanRef::Deduplicated(4);
        assert_eq!(llbc.span_data(&present).unwrap().beg.line, 4);
    }

    #[test]
    fn an_artefact_without_a_files_table_answers_nothing_rather_than_failing() {
        let doc = r#"{"charon_version":"t","has_errors":false,
                      "translated":{"crate_name":"c","fun_decls":[]}}"#;
        let l = Llbc::from_slice(doc.as_bytes()).expect("loads without a files table");
        assert_eq!(l.file_path(0), None);
    }
}
