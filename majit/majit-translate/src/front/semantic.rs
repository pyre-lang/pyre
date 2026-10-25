//! Front-end shared types — the data shapes `front::mir` produces, and
//! that the rest of the pipeline (`analyze_pipeline_from_module_paths`,
//! `codewriter::*`, `parse::*`) consumes.
//!
//! These types do not depend on any graph builder, so they live in
//! their own module rather than inside `front::mir`.
//!
//! Nothing in this module performs lowering.  Graphs are built by
//! `front::mir`, the Charon ULLBC driver.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::model::{FunctionGraph, ImmutableRank, LazyGraph, UnknownKind};

/// Options carried through the semantic-program build.  A distinct unit
/// type so the build entry point can accept an explicit options
/// parameter while preserving the upstream `build_flow_graph` call
/// shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstGraphOptions;

/// Signal that lowering was halted due to an unsupported construct.
///
/// RPython `rpython/flowspace/flowcontext.py` raises `FlowingError`
/// when the abstract interpreter hits a bytecode it cannot model; that
/// error propagates all the way out of `build_flow_graph`, aborting the
/// current graph rather than silently continuing with a synthetic value.
///
/// Pyre's `Option<Variable>` return conflates "expression legitimately
/// produced no value" (e.g. `return` / `break`) with "lowering halted"
/// — making the latter an explicit `Err` variant restores the RPython
/// invariant that unsupported constructs stop the walk at once.  The
/// `Unknown` op is still emitted at the failure site so downstream
/// passes see evidence of the drop; the `Err` just guarantees no
/// synthesised SSA value follows it.
#[derive(Debug, Clone)]
pub enum FlowingError {
    Unsupported { kind: UnknownKind },
}

/// Alias for `FlowingError`, used by callers that spell the abort
/// type as `LoweringAbort`.
pub type LoweringAbort = FlowingError;

#[derive(Debug, Clone)]
pub struct SemanticFunction {
    pub name: String,
    pub(in crate::front) graph: crate::model::LazyGraph,
    /// RPython: `op.result.concretetype` — full return type string.
    /// Used for array identity resolution on Call result values.
    pub return_type: Option<String>,
    /// Class key of a `ref` residual whose `_signature_` analog is
    /// `SomeInstance(classdef)`. Set for a by-value ADT and for a pointer
    /// to a registered GC struct.
    pub return_class_root: Option<String>,
    /// The residual instance may be null (`SomeInstance.can_be_None`).
    pub return_can_be_none: bool,
    /// Owner type for impl methods (e.g. "MyStruct" for `impl MyStruct { fn foo() }`).
    /// Used to construct the full CallPath for return_type registration.
    pub self_ty_root: Option<String>,
    /// Charon's concrete trait-impl identity when this function comes from an
    /// `Impl { Trait: id }` name segment. Intra-LLBC readers
    /// (`blanket_into_devirt`) keep the extraction-local id; cross-artifact
    /// keys use [`Self::trait_impl_identity`].
    pub trait_impl_id: Option<u64>,
    /// Trait-impl identity spelled from the impl's trait ref and Self type
    /// (`item_meta` / `impl_trait` + Self). `Bookkeeper.getdesc` keys by the
    /// callable; an extraction-local impl index is not that identity.
    pub trait_impl_identity: Option<String>,
    /// Charon `FunDecl.def_id` of the declaration this function was
    /// lowered from. Threaded onto `FunctionGraph` so `CallControl`
    /// resolves a call site by decl identity rather than a name suffix.
    pub fun_decl_id: Option<u64>,
    /// Module path of the defining file, crate-stripped (e.g.
    /// `"pyframe"` for `pyre-interpreter/src/pyframe.rs`), populated by
    /// `front::mir` from the module portion of Charon's `name_path()`.
    /// Empty when the producer did not supply a module path — top-level
    /// items remain at simple-name registration.
    ///
    /// Used by `lib.rs` registration so a free function's call sites that
    /// were qualified by `canonical_call_target:7494-7502` (single-segment
    /// bare call inside a non-empty module) can resolve through the
    /// `[module_path, name]` path, in addition to the bare-name and
    /// `crate::` alias paths.  Without the extra path the
    /// `#[majit_macros::elidable*]` / oopspec / loop-invariant hints
    /// registered against the bare name are silently dropped at every
    /// in-module call site.
    pub module_path: String,
    /// RPython: function-level hints set by GC transformer / decorators.
    /// "close_stack" → _gctransformer_hint_close_stack_
    /// "cannot_collect" → _gctransformer_hint_cannot_collect_
    /// "gc_effects" → random_effects_on_gcobjs
    /// "elidable" → _elidable_function_
    /// "loopinvariant" → _jit_loop_invariant_
    pub hints: Vec<String>,
    /// Trait name when this function is an `impl Trait for Type {…}`
    /// method, the trait's name when this is a trait default-body
    /// method, otherwise `None` (free function or inherent impl).
    ///
    /// Lets the registration loop in `lib.rs` walk
    /// `program.functions` directly and distinguish trait-impl methods
    /// (which need `register_trait_method`) from inherent methods
    /// (which need `register_function_graph`).
    pub trait_root: Option<String>,
    /// Fully-qualified `name_path()` of the trait when this function
    /// is an `impl Trait for Type {…}` method, otherwise `None`.
    /// Distinguishes traits whose leaf names collide — the unique-impl
    /// map (`trait_unique_impls`) keys on this, not `trait_root`.
    /// Trait default bodies leave it `None`: Charon names them with
    /// only the trait leaf segment, and they never feed the
    /// unique-impl map.
    pub trait_qualified: Option<String>,
    /// `true` when the function returns `*mut PyObject` (a `PyObjectRef`),
    /// detected structurally by `front::mir::output_type_is_objectptr`.
    /// The MIR driver leaves every `return_type` `None`, so a
    /// `dont_look_inside` callee returning an object pointer would
    /// residualize as a `None`→`Void` stub (a miscompile — the caller
    /// needs the pointer).  `SemanticFunctionHeader::new` reads this flag and,
    /// for a `dont_look_inside` callee, stamps the object-pointer
    /// `return_type` marker so the residual prefill projects a `Ref`
    /// result.
    pub returns_objectptr: bool,
}

impl SemanticFunction {
    /// The function's flow graph (`description.py FunctionDesc.getuniquegraph`).
    pub fn graph(&self) -> &FunctionGraph {
        self.graph.get().expect("a SemanticFunction has a graph")
    }

    pub fn graph_mut(&mut self) -> &mut FunctionGraph {
        self.graph
            .get_mut()
            .expect("a SemanticFunction has a graph")
    }

    /// The funcobj's graph handle, shared with every holder and built on
    /// first demand (`FunctionDesc.cachedgraph`).
    pub fn lazy_graph(&self) -> &crate::model::LazyGraph {
        &self.graph
    }

    /// Empty-body funcobj for tests outside `front`.
    #[cfg(test)]
    pub(crate) fn with_empty_graph(
        name: impl Into<String>,
        graph_name: impl Into<String>,
        self_ty_root: Option<String>,
        trait_root: Option<String>,
        trait_qualified: Option<String>,
    ) -> Self {
        Self {
            name: name.into(),
            graph: crate::model::LazyGraph::built(crate::model::FunctionGraph::new(
                graph_name.into(),
            )),
            return_type: None,
            return_class_root: None,
            return_can_be_none: false,
            self_ty_root,
            trait_impl_id: None,
            trait_impl_identity: None,
            fun_decl_id: None,
            module_path: String::new(),
            hints: Vec::new(),
            trait_root,
            trait_qualified,
            returns_objectptr: false,
        }
    }
}

/// One harvested struct-field registry row: the published name, the field
/// type string, and whether Charon `FieldDecl::name` was `None`.
///
/// The front mints [`majit_charon_reader::ullbc::positional_field_name`]
/// when that name is missing. The flag is the origin, not the spelling:
/// a declared field called `__pos_0` stays `is_positional == false`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FieldRow {
    pub name: String,
    pub ty: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_positional: bool,
}

impl FieldRow {
    pub fn named(name: impl Into<String>, ty: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
            is_positional: false,
        }
    }

    pub fn positional(name: impl Into<String>, ty: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
            is_positional: true,
        }
    }
}

impl From<(String, String)> for FieldRow {
    fn from((name, ty): (String, String)) -> Self {
        Self::named(name, ty)
    }
}

impl PartialEq<(String, String)> for FieldRow {
    fn eq(&self, other: &(String, String)) -> bool {
        self.name == other.0 && self.ty == other.1
    }
}

impl<'de> Deserialize<'de> for FieldRow {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum De {
            Pair(String, String),
            Full {
                name: String,
                ty: String,
                #[serde(default)]
                is_positional: bool,
            },
        }
        match De::deserialize(deserializer)? {
            De::Pair(name, ty) => Ok(Self::named(name, ty)),
            De::Full {
                name,
                ty,
                is_positional,
            } => Ok(Self {
                name,
                ty,
                is_positional,
            }),
        }
    }
}

/// Drop a leading local or harvested foreign-body crate root so intern
/// spellings (`ops::range::RangeInclusive`) match harvested keys
/// (`core::ops::range::RangeInclusive`). `bookkeeper.py`
/// `getuniqueclassdef` keys by the class object, not the import prefix.
pub(crate) fn crate_relative_struct_path<'a>(path: &'a str, foreign_roots: &[String]) -> &'a str {
    match path.split_once("::") {
        Some((root, rest))
            if crate::local_crates::is_local_crate_root(root)
                || foreign_roots.iter().any(|r| r == root) =>
        {
            rest
        }
        _ => path,
    }
}

/// First balanced `<…>` group, including brackets. A trailing enum
/// variant (`Result<T>::Ok`) is excluded.
fn generic_args_span(name: &str) -> Option<&str> {
    let start = name.find('<')?;
    let mut depth = 0usize;
    for (offset, byte) in name.as_bytes()[start..].iter().copied().enumerate() {
        match byte {
            b'<' => depth += 1,
            b'>' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&name[start..=start + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_comma_args(inner: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    let mut depth: i32 = 0;
    let mut start: usize = 0;
    for (i, b) in inner.bytes().enumerate() {
        match b {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                let segment = inner[start..i].trim();
                if !segment.is_empty() {
                    out.push(segment);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

/// Type arguments of the first `<…>` group (`CodeObject<ConstantData>` →
/// `["ConstantData"]`, `RangeInclusive<u8>` → `["u8"]`).
pub(crate) fn instantiation_type_args(name: &str) -> Vec<&str> {
    let Some(span) = generic_args_span(name) else {
        return Vec::new();
    };
    split_comma_args(&span[1..span.len() - 1])
}

/// True when `arg` is a scalar that changes a struct field's int/float
/// layout (`RangeInclusive<u8>` vs `<i64>`). Named ADTs (`ConstantData`)
/// are one class with the omitted-default spelling
/// (`bookkeeper.py` `getuniqueclassdef`).
pub(crate) fn type_arg_is_scalar_layout(arg: &str) -> bool {
    matches!(
        arg.trim().rsplit("::").next().unwrap_or(arg.trim()),
        "bool"
            | "char"
            | "i8"
            | "i16"
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
    )
}

/// True when any type argument of `name` is a scalar-layout primitive.
pub(crate) fn instantiation_has_scalar_layout_arg(name: &str) -> bool {
    instantiation_type_args(name)
        .iter()
        .any(|arg| type_arg_is_scalar_layout(arg))
}

/// RPython: struct field type info for `heaptracker.all_interiorfielddescrs`.
/// Maps struct_name → vec of field rows.
/// `field_element_type` is the array element type when the field is an
/// array container (e.g. `Vec<Point>` → `"Point"`), or the full type
/// string for non-array fields.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructFieldRegistry {
    /// struct_name → [FieldRow]
    pub fields: FieldRows,
    /// Suffix buckets for the key set of `fields`. Built on the first query
    /// that needs them and reused while the key-set fingerprint still
    /// matches. Replacing the rows of an existing key does not change the
    /// key set; those queries read the rows from `fields` after the bucket
    /// narrows. Inserting or removing a key changes the fingerprint, so
    /// the next query rebuilds — including a remove followed by an insert
    /// that restores `len`.
    /// Crate-visible so a struct literal can fill it with `..Default::default()`;
    /// queries treat a missing or stale fingerprint as absent and rebuild.
    #[serde(skip)]
    pub(crate) field_path_index: std::cell::RefCell<Option<FieldPathIndex>>,
    /// Owners whose fields are only scalar words and pointers to those
    /// (`adt_def_is_raw_storage`). Empty on a hand-built fixture, which
    /// keeps the `GcKind::Raw` gate. A harvested program lists every
    /// spelling of those owners; a classed struct that is merely not a
    /// GC header is absent, so a pointer to it stays an instance.
    #[serde(default, skip_serializing_if = "std::collections::HashSet::is_empty")]
    pub(crate) raw_word_owners: std::collections::HashSet<String>,
    /// Crate roots of FunDecls that lowered a body and are not local
    /// alias roots. Intern and field lookup strip these prefixes
    /// (`bookkeeper.py` `getuniqueclassdef`). Owned by this harvested
    /// registry, not TLS.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) foreign_body_roots: Vec<String>,
    /// Harvested foreign FunDecls that lowered a body:
    /// `(crate_root, crate-stripped name_path)`. Lookup strips the crate
    /// prefix only for these callables (`bookkeeper.py` `Bookkeeper.getdesc`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) foreign_callables: Vec<(String, String)>,
}

/// `struct_name → [FieldRow]`, with an O(1) key-set fingerprint. `insert` /
/// `remove` / `entry` update the xor of the keys' hashes; a lookup compares
/// that word and `len` and does not rescan the map. Reads deref to the inner
/// map. Serializes as the inner map; deserializing goes through `From` so
/// the fingerprint is recomputed. A `HashMap` of `(name, ty)` pairs becomes
/// named rows (`is_positional == false`).
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "HashMap<String, Vec<FieldRow>>")]
pub struct FieldRows {
    map: HashMap<String, Vec<FieldRow>>,
    /// Xor of [`field_key_fp`] over the current keys. `0` when empty.
    key_fp: std::cell::Cell<u64>,
}

impl Default for FieldRows {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            key_fp: std::cell::Cell::new(0),
        }
    }
}

impl From<HashMap<String, Vec<FieldRow>>> for FieldRows {
    fn from(map: HashMap<String, Vec<FieldRow>>) -> Self {
        let key_fp = map.keys().fold(0u64, |acc, key| acc ^ field_key_fp(key));
        Self {
            map,
            key_fp: std::cell::Cell::new(key_fp),
        }
    }
}

impl From<HashMap<String, Vec<(String, String)>>> for FieldRows {
    fn from(map: HashMap<String, Vec<(String, String)>>) -> Self {
        let map: HashMap<String, Vec<FieldRow>> = map
            .into_iter()
            .map(|(key, rows)| (key, rows.into_iter().map(FieldRow::from).collect()))
            .collect();
        Self::from(map)
    }
}

impl From<FieldRows> for HashMap<String, Vec<(String, String)>> {
    fn from(rows: FieldRows) -> Self {
        rows.map
            .into_iter()
            .map(|(key, rows)| {
                (
                    key,
                    rows.into_iter().map(|row| (row.name, row.ty)).collect(),
                )
            })
            .collect()
    }
}

impl Serialize for FieldRows {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.map.serialize(serializer)
    }
}

impl std::ops::Deref for FieldRows {
    type Target = HashMap<String, Vec<FieldRow>>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl IntoIterator for FieldRows {
    type Item = (String, Vec<FieldRow>);
    type IntoIter = std::collections::hash_map::IntoIter<String, Vec<FieldRow>>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<'a> IntoIterator for &'a FieldRows {
    type Item = (&'a String, &'a Vec<FieldRow>);
    type IntoIter = std::collections::hash_map::Iter<'a, String, Vec<FieldRow>>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

fn field_key_fp(key: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

impl FieldRows {
    /// Xor of the current keys' hashes. Compared by the path index in O(1).
    pub(crate) fn key_fp(&self) -> u64 {
        self.key_fp.get()
    }

    /// The rows registered under `key`, or, for a positional aggregate
    /// spelling (`Tuple<A,B>` / `Array<T;N>`), the rows its spelling
    /// derives: `TupleRepr` builds `TUPLE_TYPE` from the items whenever the
    /// rtyper asks (`rtuple.py`), so a shape needs no prior registration.
    pub fn get(&self, key: &str) -> Option<&Vec<FieldRow>> {
        self.map
            .get(key)
            .or_else(|| crate::front::mir::positional_shape_rows(key))
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn insert(
        &mut self,
        key: String,
        value: impl IntoIterator<Item = impl Into<FieldRow>>,
    ) -> Option<Vec<FieldRow>> {
        let rows: Vec<FieldRow> = value.into_iter().map(Into::into).collect();
        let piece = field_key_fp(&key);
        let replaced = self.map.insert(key, rows);
        if replaced.is_none() {
            self.key_fp.set(self.key_fp.get() ^ piece);
        }
        replaced
    }

    pub fn remove(&mut self, key: &str) -> Option<Vec<FieldRow>> {
        let removed = self.map.remove(key)?;
        self.key_fp.set(self.key_fp.get() ^ field_key_fp(key));
        Some(removed)
    }

    pub fn entry(&mut self, key: String) -> FieldRowsEntry<'_> {
        FieldRowsEntry {
            inner: self.map.entry(key),
            key_fp: &self.key_fp,
        }
    }
}

/// [`HashMap::entry`] for [`FieldRows`]. `or_insert` folds a new key into
/// the fingerprint; an occupied key leaves it unchanged.
pub struct FieldRowsEntry<'a> {
    inner: std::collections::hash_map::Entry<'a, String, Vec<FieldRow>>,
    key_fp: &'a std::cell::Cell<u64>,
}

impl<'a> FieldRowsEntry<'a> {
    pub fn or_insert(self, default: Vec<FieldRow>) -> &'a mut Vec<FieldRow> {
        self.or_insert_with(|| default)
    }

    pub fn or_insert_with<F>(self, default: F) -> &'a mut Vec<FieldRow>
    where
        F: FnOnce() -> Vec<FieldRow>,
    {
        match self.inner {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                self.key_fp
                    .set(self.key_fp.get() ^ field_key_fp(entry.key()));
                entry.insert(default())
            }
        }
    }
}

impl StructFieldRegistry {
    /// Look up a field's type.  For array-typed fields like
    /// `Vec<Point>`, this returns the full type string `"Vec<Point>"`.
    /// Callers use `array_element_type_from_str` to extract `"Point"`.
    ///
    /// Resolution order: (1) exact registered key, then (2) canonical
    /// lexical resolution through `STRUCT_ORIGIN_REGISTRY` (PyPy
    /// `bookkeeper.getdesc(value)` analog) on the receiver leaf —
    /// registration dual-publishes bare + canonical so both spellings
    /// land at the same field list.  (3) crate-prefix-tolerant
    /// suffix-match shim absorbs `pyre_object::functional::W_X` vs
    /// `functional::W_X` divergence — orthogonal to lexical scope
    /// resolution, kept for test entries (`parse::parse_source`)
    /// that bypass `analyze_pipeline_from_module_paths`'s
    /// `register_struct_origins`.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.lookup_fields(owner)?
            .iter()
            .find(|row| row.name == field_name)
            .map(|row| row.ty.as_str())
    }

    /// True when the harvested row for `owner.field_name` came from a
    /// `FieldDecl` whose `name` was `None`. False for a declared name,
    /// including a declared `__pos_N`, and when the owner has no such row.
    pub fn field_is_positional(&self, owner: &str, field_name: &str) -> bool {
        self.lookup_fields(owner)
            .into_iter()
            .flatten()
            .find(|row| row.name == field_name)
            .is_some_and(|row| row.is_positional)
    }

    /// Declared field name at `index` for `owner`, using this registry's
    /// own key convention ([`Self::lookup_fields`]). `None` when `owner`
    /// has no row, the index is past the harvested list, or that slot's
    /// name is empty.
    pub fn field_name_at(&self, owner: &str, index: usize) -> Option<&str> {
        let row = self.lookup_fields(owner)?.get(index)?;
        (!row.name.is_empty()).then_some(row.name.as_str())
    }

    /// True when `owner` is registered as an enum base class — its sole
    /// row is the synthetic `__discriminant` tag (`rclass.py:499-518`: the
    /// sum-type base carries only the discriminant, each variant subclass
    /// carries its own payload fields under `{enum}::{variant}` keys).  A
    /// struct or non-enum owner has its own field rows and returns false.
    ///
    /// The discriminator is the row shape, not a stored type-kind flag,
    /// because the registry erases `TypeDeclKind` once rows are projected.
    /// That is sound: `__discriminant` is a reserved key emitted only by
    /// the enum-base row builder in `front/mir.rs`; a `TypeDeclKind::Struct`
    /// projects its real Rust field names, none of which is `__discriminant`
    /// (`from_type_strings` skips synthetic `__pad`/typeptr rows and never
    /// mints that name).  So a real single-field struct cannot be
    /// misclassified as an enum base.
    /// Field rows for `owner`, after the same generic-argument strip and
    /// suffix resolution as [`Self::lookup_fields`]. `None` when the owner
    /// is not registered.
    pub(crate) fn field_rows(&self, owner: &str) -> Option<&[FieldRow]> {
        self.lookup_fields(owner)
    }

    pub fn is_enum_base(&self, owner: &str) -> bool {
        self.lookup_fields(owner)
            .is_some_and(|rows| matches!(rows, [row] if row.name == "__discriminant"))
    }

    /// The registry key of the discriminant-only enum base `owner` names.
    ///
    /// `lookup_fields` accepts a crate-prefixed or generic spelling, but the
    /// class cache keys the string it is given.  Returning the registered
    /// key (`pyopcode::StepResult`) rather than the ctor's longer spelling
    /// (`pyre_interpreter::pyopcode::StepResult`) makes both intern to one
    /// `ClassDef`.
    pub fn enum_base_registry_key(&self, owner: &str) -> Option<String> {
        let stripped = majit_ir::descr::strip_generic_args(owner);
        let owner = stripped.as_ref();
        let is_disc = |rows: &[FieldRow]| matches!(rows, [row] if row.name == "__discriminant");
        if self.fields.get(owner).is_some_and(|rows| is_disc(rows)) {
            return Some(owner.to_string());
        }
        let receiver_leaf = owner.rsplit("::").next().unwrap_or(owner);
        let canonical = majit_ir::descr::canonical_struct_name(receiver_leaf);
        if canonical != receiver_leaf
            && self
                .fields
                .get(&canonical)
                .is_some_and(|rows| is_disc(rows))
        {
            return Some(canonical);
        }
        let key = self.unique_suffix_owner_key(owner)?.to_string();
        self.fields
            .get(&key)
            .filter(|rows| is_disc(rows))
            .map(|_| key)
    }

    /// A payload variant under this enum base. A fieldless enum has none,
    /// so its value stays an integer; a payload enum's address is a pointer
    /// (`rclass.py` variant subclass under the discriminant-only base).
    pub fn enum_base_has_payload(&self, owner: &str) -> bool {
        if !self.is_enum_base(owner) {
            return false;
        }
        let canonical_owner = majit_ir::descr::canonical_struct_name(owner);
        let parent_leaf = canonical_owner
            .rsplit("::")
            .next()
            .unwrap_or(canonical_owner.as_str());
        self.ensure_field_path_index();
        let index = self.field_path_index.borrow();
        let index = index.as_ref().expect("path index built");
        index.by_parent_last.get(parent_leaf).is_some_and(|bucket| {
            bucket.iter().any(|key| {
                key.rsplit_once("::").is_some_and(|(parent, _variant)| {
                    majit_ir::descr::canonical_struct_name(parent) == canonical_owner
                        && self
                            .fields
                            .get(key)
                            .is_some_and(|rows| rows.iter().any(|row| row.name != "__discriminant"))
                })
            })
        })
    }

    /// Harvested type of `field_name` on `owner` itself, or on one of
    /// its enum variants (`rclass.py` variant subclass under the
    /// discriminant-only base).
    ///
    /// Rust keeps fields and methods in separate namespaces, while the
    /// RPython class model used by the annotator has one attribute namespace.
    /// A method on an enum base therefore also collides with a same-named
    /// payload field on a variant: seeding that method on the base would make
    /// the variant constructor inherit a function as the field's default.
    /// Variant keys group by the parent path's last segment, the same
    /// leaf `canonical_struct_name` keeps. The row check still reads
    /// `fields`, so a replaced field list is visible.
    pub fn owner_or_variant_field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        if let Some(ty) = self.field_type(owner, field_name) {
            return Some(ty);
        }
        if !self.is_enum_base(owner) {
            return None;
        }
        let canonical_owner = majit_ir::descr::canonical_struct_name(owner);
        let parent_leaf = canonical_owner
            .rsplit("::")
            .next()
            .unwrap_or(canonical_owner.as_str());
        self.ensure_field_path_index();
        let index = self.field_path_index.borrow();
        let index = index.as_ref().expect("path index built");
        let bucket = index.by_parent_last.get(parent_leaf)?;
        for key in bucket {
            let Some((parent, _variant)) = key.rsplit_once("::") else {
                continue;
            };
            if majit_ir::descr::canonical_struct_name(parent) != canonical_owner {
                continue;
            }
            if let Some(ty) = self.fields.get(key).and_then(|rows| {
                rows.iter()
                    .find(|row| row.name == field_name)
                    .map(|row| row.ty.as_str())
            }) {
                return Some(ty);
            }
        }
        None
    }

    /// Whether `owner` itself, or one of its enum variants, declares
    /// `field_name`.
    pub fn owner_or_variant_has_field(&self, owner: &str, field_name: &str) -> bool {
        self.owner_or_variant_field_type(owner, field_name)
            .is_some()
    }

    /// Drop the suffix buckets. [`FieldRows`] also records a key-set
    /// fingerprint, so a later lookup rebuilds after any key insert or
    /// remove; this drops the buckets immediately.
    pub(crate) fn invalidate_field_path_index(&self) {
        self.field_path_index.borrow_mut().take();
    }

    /// Remove one registered key and drop the suffix buckets.
    pub(crate) fn remove_field(&mut self, key: &str) -> Option<Vec<FieldRow>> {
        let removed = self.fields.remove(key);
        if removed.is_some() {
            self.invalidate_field_path_index();
        }
        removed
    }

    fn lookup_fields(&self, owner: &str) -> Option<&[FieldRow]> {
        // A per-instantiation enum spelling (`Result<Tuple>`,
        // `Result<Tuple>::Ok`) shares the bare template's rows: the
        // reference-payload split exists only to separate annotator attr
        // unification across instantiations, not to give each one its own
        // row layout.  Drop the `<…>` argument span (keeping any trailing
        // `::variant`) so the lookup resolves under the bare key the
        // template registered.
        let stripped = majit_ir::descr::strip_generic_args(owner);
        let owner = stripped.as_ref();
        if let Some(fields) = self.fields.get(owner) {
            return Some(fields.as_slice());
        }
        // Canonical lexical resolution: bare receiver leaves resolve
        // through `STRUCT_ORIGIN_REGISTRY` (PyPy `bookkeeper.getdesc`
        // analog).  Registration dual-publishes bare + canonical, so a
        // miss on the exact owner falls through canonical-leaf lookup
        // before the suffix-match shim below.
        let receiver_leaf = owner.rsplit("::").next().unwrap_or(owner);
        let canonical = majit_ir::descr::canonical_struct_name(receiver_leaf);
        if canonical != receiver_leaf
            && let Some(fields) = self.fields.get(&canonical)
        {
            return Some(fields.as_slice());
        }
        let key = self.unique_suffix_owner_key(owner)?;
        self.fields.get(key).map(Vec::as_slice)
    }

    /// Record a FunDecl crate root when that FunDecl has a body and the
    /// crate is not already a local alias root.
    pub(crate) fn note_foreign_body_crate_root(&mut self, root: &str) {
        if root == "crate" || crate::local_crates::is_local_crate_root(root) {
            return;
        }
        if !self.foreign_body_roots.iter().any(|r| r == root) {
            self.foreign_body_roots.push(root.to_string());
        }
    }

    /// Record a foreign FunDecl that lowered a body, so lookup may strip
    /// `crate_root` from this callable's path only (`Bookkeeper.getdesc`).
    pub(crate) fn note_foreign_body_callable(&mut self, crate_root: &str, stripped_path: &str) {
        if crate_root == "crate"
            || crate::local_crates::is_local_crate_root(crate_root)
            || stripped_path.is_empty()
        {
            return;
        }
        if !self
            .foreign_callables
            .iter()
            .any(|(root, path)| root == crate_root && path == stripped_path)
        {
            self.foreign_callables
                .push((crate_root.to_string(), stripped_path.to_string()));
        }
    }

    /// True when a harvested TypeDecl path for `owner` starts with a
    /// foreign-body crate root recorded on this registry. A generic struct
    /// in that crate is its own class per instantiation
    /// (`bookkeeper.py` `getuniqueclassdef`); local-crate structs stay
    /// unsuffixed so closures are not seeded from template rows.
    ///
    /// Intern spellings are often crate-stripped (`ops::range::RangeInclusive`
    /// after `core::` is dropped). Those no longer start with the harvested
    /// crate root; match them against registry keys that still do.
    pub(crate) fn owner_in_foreign_body_crate(&self, owner: &str) -> bool {
        let stripped = majit_ir::descr::strip_generic_args(owner);
        let crate_root = stripped.split("::").next().unwrap_or(stripped.as_ref());
        if self.foreign_body_roots.iter().any(|r| r == crate_root) {
            return true;
        }
        let leaf = stripped.rsplit("::").next().unwrap_or(stripped.as_ref());
        if leaf == crate_root {
            self.ensure_field_path_index();
            let index = self.field_path_index.borrow();
            let Some(index) = index.as_ref() else {
                return false;
            };
            let Some(bucket) = index.by_last.get(leaf) else {
                return false;
            };
            return bucket.iter().any(|key| {
                let crate_root = key.split("::").next().unwrap_or(key.as_str());
                self.foreign_body_roots.iter().any(|r| r == crate_root)
            });
        }
        let tail = crate_relative_struct_path(stripped.as_ref(), &self.foreign_body_roots);
        self.ensure_field_path_index();
        let index = self.field_path_index.borrow();
        let Some(index) = index.as_ref() else {
            return false;
        };
        let Some(bucket) = index.by_stripped_last.get(leaf) else {
            return false;
        };
        bucket.iter().any(|k| {
            let ks = majit_ir::descr::strip_generic_args(k);
            let root = ks.split("::").next().unwrap_or(ks.as_ref());
            self.foreign_body_roots.iter().any(|r| r == root)
                && crate_relative_struct_path(ks.as_ref(), &self.foreign_body_roots) == tail
        })
    }

    fn unique_suffix_owner_key<'a>(&'a self, owner: &str) -> Option<&'a str> {
        // Two path-suffix-related keys share their last `::` segment, so
        // the bucket is the full candidate set. More than one match is
        // still `None`.
        let leaf = owner.rsplit("::").next().unwrap_or(owner);
        self.ensure_field_path_index();
        let matched = {
            let index = self.field_path_index.borrow();
            let index = index.as_ref().expect("path index built");
            let mut found: Option<String> = None;
            let mut ambiguous = false;
            if let Some(bucket) = index.by_last.get(leaf) {
                for key in bucket {
                    let matches = is_path_suffix(owner, key) || is_path_suffix(key, owner);
                    if !matches {
                        continue;
                    }
                    if found.is_some() {
                        ambiguous = true;
                        break;
                    }
                    found = Some(key.clone());
                }
            }
            if ambiguous { None } else { found }
        }?;
        self.fields
            .get_key_value(&matched)
            .map(|(key, _)| key.as_str())
    }

    fn ensure_field_path_index(&self) {
        let len = self.fields.len();
        let key_fp = self.fields.key_fp();
        if self
            .field_path_index
            .borrow()
            .as_ref()
            .is_some_and(|index| index.len == len && index.key_fp == key_fp)
        {
            return;
        }
        let built = FieldPathIndex::build(&self.fields, key_fp);
        *self.field_path_index.borrow_mut() = Some(built);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FieldPathIndex {
    len: usize,
    /// [`FieldRows::key_fp`] at the time the buckets were built.
    key_fp: u64,
    /// Last `::` segment → keys ending with that segment.
    by_last: rustc_hash::FxHashMap<String, Vec<String>>,
    /// Last segment of the parent path → keys with that parent leaf.
    by_parent_last: rustc_hash::FxHashMap<String, Vec<String>>,
    /// Last `::` segment of [`majit_ir::descr::strip_generic_args`] of the key → keys.
    by_stripped_last: rustc_hash::FxHashMap<String, Vec<String>>,
}

impl FieldPathIndex {
    fn build(fields: &HashMap<String, Vec<FieldRow>>, key_fp: u64) -> Self {
        let mut by_last: rustc_hash::FxHashMap<String, Vec<String>> =
            rustc_hash::FxHashMap::default();
        let mut by_parent_last: rustc_hash::FxHashMap<String, Vec<String>> =
            rustc_hash::FxHashMap::default();
        let mut by_stripped_last: rustc_hash::FxHashMap<String, Vec<String>> =
            rustc_hash::FxHashMap::default();
        for key in fields.keys() {
            let leaf = key.rsplit("::").next().unwrap_or(key.as_str());
            by_last
                .entry(leaf.to_string())
                .or_default()
                .push(key.clone());
            if let Some((parent, _variant)) = key.rsplit_once("::") {
                let parent_leaf = parent.rsplit("::").next().unwrap_or(parent);
                by_parent_last
                    .entry(parent_leaf.to_string())
                    .or_default()
                    .push(key.clone());
            }
            let stripped = majit_ir::descr::strip_generic_args(key);
            let stripped_leaf = stripped.rsplit("::").next().unwrap_or(stripped.as_ref());
            by_stripped_last
                .entry(stripped_leaf.to_string())
                .or_default()
                .push(key.clone());
        }
        Self {
            len: fields.len(),
            key_fp,
            by_last,
            by_parent_last,
            by_stripped_last,
        }
    }
}

fn is_path_suffix(longer: &str, shorter: &str) -> bool {
    if longer.len() <= shorter.len() || !longer.ends_with(shorter) {
        return false;
    }
    let prefix_len = longer.len() - shorter.len();
    longer[..prefix_len].ends_with("::")
}

/// Exact memory layout of a type, resolved from Charon's per-target
/// `variant_layouts.field_offsets` in the LLBC (not the `#[repr(C)]`-
/// approximating heuristic).  Keyed in [`SemanticProgram::exact_layouts`]
/// by owner root: a struct leaf / qualified name, or — for an enum
/// variant — `{leaf}::{variant}`.  `field_offsets` carries the field's
/// byte offset within the type; the heuristic provider is used only for
/// roots without an entry here.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExactLayout {
    /// Total byte size when Charon resolved it.
    pub size: Option<u64>,
    /// Byte alignment when Charon resolved it.
    pub align: Option<u64>,
    /// `field_name → byte offset within the type`.
    pub field_offsets: HashMap<String, u64>,
    /// Host Rust layout. Absent on a heuristic-only record. Skipped by
    /// serde: it is rebuilt from Charon whenever metadata is derived.
    #[serde(skip)]
    pub host: Option<crate::front::host_layout::HostLayout>,
}

#[derive(Debug, Clone, Default)]
pub struct SemanticProgram {
    pub functions: Vec<SemanticFunction>,
    /// JIT-hint markers harvested from LLBC, keyed by crate-stripped
    /// function path. Applied even when no `SemanticFunction` was
    /// lowered (a `dont_look_inside` residual has no graph).
    pub harvested_hints: HashMap<String, Vec<String>>,
    /// RPython: known struct types for `get_type_flag(ARRAY.OF)` → FLAG_STRUCT.
    pub known_struct_names: std::collections::HashSet<String>,
    /// Known trait names used to canonicalize local `dyn Trait` family keys.
    pub known_trait_names: std::collections::HashSet<String>,
    /// RPython: struct field types for resolving `op.args[0].concretetype`
    /// on FieldRead-produced array bases.
    pub struct_fields: StructFieldRegistry,
    /// RPython: `_immutable_fields_ = [...]` declared on a class body.
    /// Maps struct name → `(field_name, rank)` pairs whose value never
    /// mutates after construction (or is quasi-immutable).  Both bare and
    /// qualified struct keys are inserted (mirroring `struct_fields`) so
    /// the same lookup logic works across module-prefix variants.  Rank
    /// encoding follows `rpython/rtyper/rclass.py _parse_field_list`.
    pub immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>>,
    /// Structs whose `_alloc_flavor_<Struct>` marker is `"raw"`.
    /// `rclass.py getgcflavor` / `getinstancerepr` build a raw
    /// `InstanceRepr` for those classdefs. Each key is the class intern
    /// key of the one struct the marker resolves to
    /// (`llbc_hints::harvest_alloc_flavor_raw_from_llbcs`).
    pub raw_alloc_flavor_structs: std::collections::HashSet<String>,
    /// Enum discriminant → variant name, keyed by enum type (both the
    /// qualified path and the bare leaf, mirroring `struct_fields`).
    /// The opcode-dispatch MIR extractor reads
    /// `enum_variant_by_discriminant["Instruction"]` to turn a switch
    /// case value (`ExitCase::Const(Int(K))`, the variant discriminant —
    /// which is *not* the variant index) back into the variant name.
    pub enum_variant_by_discriminant: HashMap<String, HashMap<i64, String>>,
    /// `bare_struct_name → defining crate-relative module path`,
    /// harvested from the LLBC `iter_type_decls()` name paths.
    /// Feeds `majit_ir::descr::STRUCT_ORIGIN_REGISTRY` so
    /// `canonical_struct_name` resolves a bare leaf to the qualified
    /// `module::Bare` key the runtime's
    /// `build_object_descr_group_with_def_path` dual-publishes (the
    /// crate prefix is stripped to match that def-path convention).
    pub struct_origins: HashMap<String, String>,
    /// `crate-relative qualified struct name → declaration-ordered
    /// `(field, ValueType)` register classes`, harvested from the LLBC
    /// `iter_type_decls()` struct field types.  Feeds
    /// `annotator::classdesc::register_struct_fields` →
    /// `FORCE_ATTRIBUTES_INTO_CLASSES` so `ClassDef::_init_classdef`
    /// pre-fills `ClassDef.attrs` before the annotator's
    /// `attrs_populated` narrowing gate.  Key drops the crate prefix to
    /// match the qualname `_init_classdef` reads; primitive fields carry
    /// `Int`/`Unsigned`/`Bool`/`Float`, every other shape `Ref(None)`.
    pub struct_field_attrs: HashMap<String, Vec<(String, crate::model::ValueType)>>,
    /// Exact per-type memory layout (byte offsets, size, align) harvested
    /// from Charon's LLBC `type_decl.layout`, keyed by the type's
    /// [`StructId`](majit_ir::descr::StructId) object identity (one entry
    /// per type definition, enum variants included).  Feeds the exact
    /// `LayoutProvider`, replacing the heuristic for types that have an
    /// entry; the heuristic remains the fallback.  Keying on the identity
    /// token rather than a name string makes two distinct definitions
    /// that share a leaf name structurally distinct, removing the
    /// last-writer-wins collision the bare-leaf string key had.
    pub exact_layouts: HashMap<majit_ir::descr::StructId, ExactLayout>,
    /// Resolves a struct / enum-variant name in any spelling (full-crate,
    /// crate-stripped, bare leaf, variant analogs) back to its canonical
    /// [`StructId`](majit_ir::descr::StructId), or `None` for a bare leaf
    /// two distinct modules share.  Registered into
    /// `majit_ir::descr::STRUCT_ID_BY_NAME` so the layout consumers that
    /// only hold a string (`llmemory::FieldOffset`'s `st._name`, a nested
    /// field's rendered type) can reach the identity-keyed layout maps.
    pub struct_ids: HashMap<String, Option<majit_ir::descr::StructId>>,
    /// Annotator-only residual declarations `(path-segments, Signature,
    /// return-lltype)`.  The historical field name covers three sources:
    /// unsafe path aliases, `dont_look_inside` functions whose bodies JitPolicy
    /// excludes, and marked allocation constructors.  All retain an RPython
    /// `FunctionDesc`-equivalent signature without becoming JitCode bodies.
    /// Feeds `CallControl.unsafe_fn_stubs` →
    /// `cutover::register_unsafe_fn_stubs`.
    pub unsafe_fn_stubs: Vec<(
        Vec<String>,
        crate::flowspace::argument::Signature,
        Option<String>,
    )>,
    /// `(path-segments, Signature, result ValueType)` for every method on
    /// a foreign **opaque** ADT owner (`malachite_bigint::bigint::BigInt`,
    /// …) whose result is faithfully modelable, harvested from the LLBC by
    /// `front::mir::collect_foreign_opaque_method_externals`.  Feeds
    /// `CallControl.foreign_opaque_method_externals` →
    /// `cutover::register_foreign_opaque_method_externals` so the
    /// `CallTarget::FunctionPath` form (which `impl_method_owner` falls
    /// back to for an opaque owner) resolves instead of panicking
    /// `SomeInstance.getattr` on the classdef-less receiver — the
    /// `register_external` / `@jit.dont_look_inside` analog.
    pub foreign_opaque_method_externals: Vec<(
        Vec<String>,
        crate::flowspace::argument::Signature,
        crate::model::ValueType,
    )>,
    /// Functions whose body the MIR loop declined for an ordered atomic
    /// load. The lowering records the declaration; callers do not walk
    /// the body again. `populate_call_registry_from_call_graphs` reads it.
    pub atomic_load_decls:
        Vec<crate::translator::rtyper::lltypesystem::module::ll_extaccessor::DeclinedFunDecl>,
}

/// Graph lookup table built from a `SemanticProgram` so registration and
/// codewriter graph discovery can fetch the MIR-built graph for a given
/// (impl_type or trait_root, method) pair by name.
///
/// Callers spell `self_ty_root` two ways: a qualified owner
/// ("pyframe::PyFrame"), or — for a top-level `impl Drop for PyFrame`
/// reached through `for_type` — the bare leaf "PyFrame".  The MIR
/// driver always stores the module-qualified spelling.  To bridge the
/// asymmetry without forcing callers to re-qualify, the lookup indexes
/// every impl method TWICE: once by qualified owner, once by the bare
/// leaf (rsplit on "::").  Bare-leaf collisions across distinct types
/// (e.g. `Drop::drop` on both `PyFrame` and `Other`) are tracked as
/// ambiguous and return None — the caller must then qualify.
pub struct MirGraphLookup<'a> {
    /// Impl methods (inherent + trait-impl): keyed by (self_ty_root, name).
    /// `Ok(&graph)` is a unique hit; `Err(())` marks the slot ambiguous
    /// (two or more graphs share the (owner-spelling, name) tuple).
    impl_methods: HashMap<(&'a str, &'a str), Result<&'a LazyGraph, ()>>,
    /// Trait-default bodies: keyed by (trait_root, name) with self_ty_root None.
    /// `Ok(&graph)` is a unique hit; `Err(())` marks the slot ambiguous
    /// (two distinct traits share a bare leaf + default-method name), so
    /// the caller falls back rather than registering an arbitrary body.
    trait_defaults: HashMap<(&'a str, &'a str), Result<&'a LazyGraph, ()>>,
    /// Free functions (no impl owner, no trait root): keyed by bare name.
    /// `Ok(&graph)` is a unique hit; `Err(())` marks the slot ambiguous
    /// (two or more free functions share a bare name across modules).
    /// Lets ordinary free-function registration and graph discovery resolve
    /// a unique MIR-built graph by its unqualified name.
    free_functions: HashMap<&'a str, Result<&'a LazyGraph, ()>>,
}

impl<'a> MirGraphLookup<'a> {
    /// Build the lookup by walking `program.functions` once.  The
    /// borrows are tied to `program`'s lifetime, so the caller must
    /// keep `program` alive for the duration of the lookup's use.
    pub fn from_program(program: &'a SemanticProgram) -> Self {
        let mut impl_methods: HashMap<(&'a str, &'a str), Result<&'a LazyGraph, ()>> =
            HashMap::new();
        let mut trait_defaults: HashMap<(&'a str, &'a str), Result<&'a LazyGraph, ()>> =
            HashMap::new();
        let mut free_functions: HashMap<&'a str, Result<&'a LazyGraph, ()>> = HashMap::new();
        for f in &program.functions {
            if let Some(owner) = f.self_ty_root.as_deref() {
                Self::insert_or_mark_ambiguous(
                    &mut impl_methods,
                    owner,
                    f.name.as_str(),
                    f.lazy_graph(),
                );
                // Also index by the bare leaf for callers that pass an
                // unqualified owner (e.g. top-level `impl Drop for
                // PyFrame` reached through `for_type`).  Bare leaf is the
                // last "::"-separated segment; identical to qualified
                // when self_ty_root has no module prefix.
                let leaf = owner.rsplit("::").next().unwrap_or(owner);
                if leaf != owner {
                    Self::insert_or_mark_ambiguous(
                        &mut impl_methods,
                        leaf,
                        f.name.as_str(),
                        f.lazy_graph(),
                    );
                }
            } else if let Some(tr) = f.trait_root.as_deref() {
                // Mark bare-leaf trait-name collisions ambiguous, mirroring
                // the impl_methods / free_functions tables, so two distinct
                // traits with a same-named default method do not last-win.
                Self::insert_or_mark_ambiguous(
                    &mut trait_defaults,
                    tr,
                    f.name.as_str(),
                    f.lazy_graph(),
                );
            } else {
                // Free function: index by bare name so the
                // opcode-dispatch extractor can resolve
                // `execute_opcode_step` and each `execute_<op>` handler.
                Self::insert_free_or_mark_ambiguous(
                    &mut free_functions,
                    f.name.as_str(),
                    f.lazy_graph(),
                );
            }
        }
        Self {
            impl_methods,
            trait_defaults,
            free_functions,
        }
    }

    fn insert_or_mark_ambiguous(
        map: &mut HashMap<(&'a str, &'a str), Result<&'a LazyGraph, ()>>,
        owner: &'a str,
        name: &'a str,
        graph: &'a LazyGraph,
    ) {
        use std::collections::hash_map::Entry;
        match map.entry((owner, name)) {
            Entry::Vacant(v) => {
                v.insert(Ok(graph));
            }
            Entry::Occupied(mut o) => {
                let existing = *o.get();
                if let Ok(g0) = existing {
                    // Same funcobj handle is fine (same entry
                    // visited via dual-key insert); only mark ambiguous
                    // when the pointer differs.
                    if !g0.ptr_eq(graph) {
                        let _ = o.insert(Err(()));
                    }
                }
                // already Err(()): stays ambiguous.
            }
        }
    }

    fn insert_free_or_mark_ambiguous(
        map: &mut HashMap<&'a str, Result<&'a LazyGraph, ()>>,
        name: &'a str,
        graph: &'a LazyGraph,
    ) {
        use std::collections::hash_map::Entry;
        match map.entry(name) {
            Entry::Vacant(v) => {
                v.insert(Ok(graph));
            }
            Entry::Occupied(mut o) => {
                if let Ok(g0) = *o.get()
                    && !g0.ptr_eq(graph)
                {
                    let _ = o.insert(Err(()));
                }
                // already Err(()): stays ambiguous.
            }
        }
    }

    /// Returns the MIR graph for a free function (no impl owner, no
    /// trait root) by bare name.  Returns None when the name does not
    /// resolve to a unique graph (no entry, or two modules share the
    /// bare name).
    pub fn lookup_free(&self, name: &str) -> Option<&'a LazyGraph> {
        self.free_functions.get(name).copied()?.ok()
    }

    /// Returns the MIR graph for an inherent or trait-impl method.
    /// Returns None when the (owner, name) tuple does not resolve to
    /// a unique graph (either no entry or ambiguous bare-leaf).
    pub fn lookup_impl_method(&self, impl_type: &str, name: &str) -> Option<&'a LazyGraph> {
        self.impl_methods.get(&(impl_type, name)).copied()?.ok()
    }

    /// Returns the MIR graph for a trait-default body.  Returns None
    /// when the (trait_root, name) tuple does not resolve to a unique
    /// graph (no entry or ambiguous bare-leaf trait name).
    pub fn lookup_trait_default(&self, trait_root: &str, name: &str) -> Option<&'a LazyGraph> {
        self.trait_defaults.get(&(trait_root, name)).copied()?.ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FunctionGraph;

    /// `remove` then `insert` of a different key restores `len`. The suffix
    /// lookup has to observe the new row.
    #[test]
    fn named_adt_instantiation_does_not_split_and_scalar_does() {
        assert_eq!(
            instantiation_type_args("bytecode::CodeObject<ConstantData>"),
            ["ConstantData"]
        );
        assert_eq!(
            instantiation_type_args("ops::range::RangeInclusive<u8>"),
            ["u8"]
        );
        assert!(!instantiation_has_scalar_layout_arg(
            "bytecode::CodeObject<ConstantData>"
        ));
        assert!(instantiation_has_scalar_layout_arg(
            "ops::range::RangeInclusive<u8>"
        ));
        assert!(type_arg_is_scalar_layout("i64"));
        assert!(!type_arg_is_scalar_layout("ConstantData"));
    }

    #[test]
    fn field_path_index_rebuilds_after_same_len_key_replacement() {
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "a::Foo".to_string(),
            vec![("x".to_string(), "i64".to_string())],
        );
        assert_eq!(reg.field_type("Foo", "x"), Some("i64"));
        reg.fields.remove("a::Foo");
        reg.fields.insert(
            "b::Foo".to_string(),
            vec![("x".to_string(), "u8".to_string())],
        );
        assert_eq!(reg.field_type("Foo", "x"), Some("u8"));
    }

    /// Dual-publish keys are the full path, the crate-stripped path, and
    /// the leaf; `lookup_fields` resolves all three. A missing owner has
    /// no row, so lowering keeps `__pos_N`.
    #[test]
    fn field_name_at_uses_registry_key_convention() {
        let mut reg = StructFieldRegistry::default();
        let rows = vec![("kind".to_string(), "i32".to_string())];
        reg.fields
            .insert("pyre_interpreter::error::PyError".to_string(), rows.clone());
        reg.fields
            .insert("error::PyError".to_string(), rows.clone());
        reg.fields.insert("PyError".to_string(), rows);
        assert_eq!(
            reg.field_name_at("pyre_interpreter::error::PyError", 0),
            Some("kind")
        );
        assert_eq!(reg.field_name_at("error::PyError", 0), Some("kind"));
        assert_eq!(reg.field_name_at("PyError", 0), Some("kind"));
        let mut stripped_only = StructFieldRegistry::default();
        stripped_only.fields.insert(
            "error::PyError".to_string(),
            vec![("kind".to_string(), "i32".to_string())],
        );
        assert_eq!(
            stripped_only.field_name_at("pyre_interpreter::error::PyError", 0),
            Some("kind")
        );
        assert_eq!(reg.field_name_at("UnknownOwner", 0), None);
        assert_eq!(reg.field_name_at("PyError", 1), None);
    }

    fn free_fn(name: &str) -> SemanticFunction {
        SemanticFunction {
            name: name.into(),
            graph: crate::model::LazyGraph::built(FunctionGraph::new(name)),
            return_type: None,
            return_class_root: None,
            return_can_be_none: false,
            self_ty_root: None,
            trait_impl_id: None,
            trait_impl_identity: None,
            fun_decl_id: None,
            module_path: String::new(),
            hints: Vec::new(),
            trait_root: None,
            trait_qualified: None,
            returns_objectptr: false,
        }
    }

    fn impl_method(owner: &str, name: &str) -> SemanticFunction {
        SemanticFunction {
            self_ty_root: Some(owner.into()),
            ..free_fn(name)
        }
    }

    fn program(functions: Vec<SemanticFunction>) -> SemanticProgram {
        SemanticProgram {
            functions,
            ..Default::default()
        }
    }

    #[test]
    fn lookup_free_resolves_unique_free_function() {
        let prog = program(vec![
            free_fn("execute_opcode_step"),
            free_fn("execute_pop_top"),
            impl_method("PyFrame", "push"),
        ]);
        let lookup = MirGraphLookup::from_program(&prog);
        assert!(lookup.lookup_free("execute_opcode_step").is_some());
        assert!(lookup.lookup_free("execute_pop_top").is_some());
        // An impl method is not a free function.
        assert!(lookup.lookup_free("push").is_none());
        // An unknown name resolves to nothing.
        assert!(lookup.lookup_free("execute_nope").is_none());
    }

    #[test]
    fn lookup_free_returns_none_on_ambiguous_bare_name() {
        // Two free functions sharing a bare name (e.g. the same helper
        // name in two modules) must not bind either graph.
        let prog = program(vec![free_fn("helper"), free_fn("helper")]);
        let lookup = MirGraphLookup::from_program(&prog);
        assert!(lookup.lookup_free("helper").is_none());
    }

    /// finding 2a: the enum variant-ctor discrimination probes the
    /// qualified enum-base spelling before the bare leaf, so a ctor of an
    /// enum whose leaf collides across modules — where the bare alias was
    /// withdrawn and `lookup_fields`'s suffix shim is ambiguous — still
    /// classifies as a variant ctor instead of misrouting to the
    /// struct-ctor branch.
    #[test]
    fn enum_variant_ctor_discriminates_via_qualified_spelling_after_leaf_collision() {
        // Two distinct enums share the leaf `E`.  `harden_duplicate_leaf_
        // metadata` withdrew the bare `E` alias on the collision; both
        // qualified spellings — the full `name_path`s — survive.
        let mut reg = StructFieldRegistry::default();
        let disc = vec![("__discriminant".to_string(), "i64".to_string())];
        reg.fields.insert("crate::m1::E".to_string(), disc.clone());
        reg.fields.insert("crate::m2::E".to_string(), disc);

        // The bare `E` misses: no exact key, and the suffix shim is
        // ambiguous across the two qualified spellings.
        assert!(
            !reg.is_enum_base("E"),
            "ambiguous bare leaf does not resolve to an enum base"
        );
        // Each qualified spelling resolves exactly.
        assert!(reg.is_enum_base("crate::m1::E"));
        assert!(reg.is_enum_base("crate::m2::E"));

        // The discrimination the variant ctor performs for `m1::E::V`:
        // probe the qualified `owner_path.join("::")` first, then the bare
        // tail.  The qualified probe routes correctly where the bare-only
        // probe (prior behaviour) would misroute to the struct-ctor branch.
        let owner_path = ["crate".to_string(), "m1".to_string(), "E".to_string()];
        let owner_tail = owner_path.last();
        let owner_qual = owner_path.join("::");
        assert!(
            reg.is_enum_base(&owner_qual) || owner_tail.is_some_and(|t| reg.is_enum_base(t)),
            "qualified probe routes the ctor to the variant path"
        );
        assert!(
            !owner_tail.is_some_and(|t| reg.is_enum_base(t)),
            "bare-tail-only probe misses under collision"
        );
    }

    #[test]
    fn enum_base_method_collides_with_variant_payload_field() {
        let mut reg = StructFieldRegistry::default();
        let base = vec![("__discriminant".to_string(), "i64".to_string())];
        let variant = vec![("w_obj".to_string(), "PyObjectRef".to_string())];
        reg.fields
            .insert("buffer::Buffer".to_string(), base.clone());
        reg.fields.insert("Buffer".to_string(), base);
        reg.fields
            .insert("buffer::Buffer::Array".to_string(), variant.clone());
        reg.fields.insert("Buffer::Array".to_string(), variant);

        assert!(reg.owner_or_variant_has_field("buffer::Buffer", "w_obj"));
        assert!(reg.owner_or_variant_has_field("Buffer", "w_obj"));
        assert_eq!(
            reg.owner_or_variant_field_type("buffer::Buffer", "w_obj"),
            Some("PyObjectRef")
        );
        assert_eq!(
            reg.owner_or_variant_field_type("Buffer", "w_obj"),
            Some("PyObjectRef")
        );
        assert!(!reg.owner_or_variant_has_field("buffer::Buffer", "readonly"));
        assert!(
            reg.owner_or_variant_field_type("buffer::Buffer", "readonly")
                .is_none()
        );
        assert!(reg.enum_base_has_payload("buffer::Buffer"));
        assert!(reg.enum_base_has_payload("Buffer"));
        reg.fields.insert(
            "plain::Color".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        assert!(!reg.enum_base_has_payload("plain::Color"));
    }

    #[test]
    fn enum_base_variant_field_type_is_harvested_mapnode_terminator() {
        let mut reg = StructFieldRegistry::default();
        let disc = vec![("__discriminant".to_string(), "i64".to_string())];
        let terminator = vec![("terminator".to_string(), "*const MapNode".to_string())];
        reg.fields
            .insert("mapdict::MapNode".to_string(), disc.clone());
        reg.fields.insert("MapNode".to_string(), disc);
        for key in [
            "mapdict::MapNode::Terminator",
            "MapNode::Terminator",
            "mapdict::MapNode::Plain",
            "MapNode::Plain",
        ] {
            reg.fields.insert(key.to_string(), terminator.clone());
        }
        assert_eq!(
            reg.owner_or_variant_field_type("mapdict::MapNode", "terminator"),
            Some("*const MapNode")
        );
        assert_eq!(
            reg.owner_or_variant_field_type("MapNode", "terminator"),
            Some("*const MapNode")
        );
        assert!(
            reg.owner_or_variant_field_type("MapNode", "no_such")
                .is_none()
        );
    }

    #[test]
    fn suffix_index_follows_remove_then_insert_that_restores_len() {
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "a::Foo".to_string(),
            vec![("x".to_string(), "i64".to_string())],
        );
        assert_eq!(reg.field_type("Foo", "x"), Some("i64"));
        reg.remove_field("a::Foo");
        reg.fields.insert(
            "b::Foo".to_string(),
            vec![("x".to_string(), "u8".to_string())],
        );
        assert_eq!(reg.field_type("Foo", "x"), Some("u8"));
        assert_eq!(reg.field_type("Foo", "missing"), None);
    }

    #[test]
    fn field_registry_serialized_shape_is_only_the_field_map() {
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "a::Foo".to_string(),
            vec![("x".to_string(), "i64".to_string())],
        );
        assert_eq!(reg.field_type("Foo", "x"), Some("i64"));
        let json = serde_json::to_value(&reg).expect("registry serializes");
        let keys: Vec<_> = json.as_object().expect("object").keys().cloned().collect();
        assert_eq!(keys, vec!["fields".to_string()]);
        let restored: StructFieldRegistry =
            serde_json::from_value(json).expect("registry deserializes");
        assert_eq!(restored.field_type("Foo", "x"), Some("i64"));
        assert!(restored.owner_or_variant_has_field("a::Foo", "x"));
    }

    #[test]
    fn absorb_semantic_program_merges_foreign_registry_tables() {
        let mut first = SemanticProgram::default();
        first.struct_fields.note_foreign_body_crate_root("core");
        first
            .struct_fields
            .note_foreign_body_callable("core", "num::<Impl>::div_ceil");
        first.struct_fields.fields.insert(
            "core::A".to_string(),
            vec![("x".to_string(), "i64".to_string())],
        );

        let mut second = SemanticProgram::default();
        second.struct_fields.note_foreign_body_crate_root("alloc");
        second
            .struct_fields
            .note_foreign_body_callable("alloc", "vec::<Impl>::push");
        second.struct_fields.fields.insert(
            "alloc::B".to_string(),
            vec![("y".to_string(), "u8".to_string())],
        );

        let mut merged = None;
        let mut seen_function_keys = std::collections::HashSet::new();
        let mut seen_struct_names = std::collections::HashSet::new();
        let mut seen_trait_names = std::collections::HashSet::new();
        let dedup = |f: &SemanticFunction| f.name.clone();
        crate::front::mir::absorb_semantic_program(
            &mut merged,
            first,
            &mut seen_function_keys,
            &mut seen_struct_names,
            &mut seen_trait_names,
            &dedup,
        );
        crate::front::mir::absorb_semantic_program(
            &mut merged,
            second,
            &mut seen_function_keys,
            &mut seen_struct_names,
            &mut seen_trait_names,
            &dedup,
        );
        let acc = merged.expect("merged program");
        assert!(
            acc.struct_fields
                .foreign_body_roots
                .iter()
                .any(|root| root == "core"),
            "first program's foreign_body_roots survive: {:?}",
            acc.struct_fields.foreign_body_roots
        );
        assert!(
            acc.struct_fields
                .foreign_body_roots
                .iter()
                .any(|root| root == "alloc"),
            "second program's foreign_body_roots merge in: {:?}",
            acc.struct_fields.foreign_body_roots
        );
        assert!(
            acc.struct_fields
                .foreign_callables
                .iter()
                .any(|(root, path)| { root == "core" && path == "num::<Impl>::div_ceil" }),
            "first program's foreign_callables survive: {:?}",
            acc.struct_fields.foreign_callables
        );
        assert!(
            acc.struct_fields
                .foreign_callables
                .iter()
                .any(|(root, path)| { root == "alloc" && path == "vec::<Impl>::push" }),
            "second program's foreign_callables merge in: {:?}",
            acc.struct_fields.foreign_callables
        );
        assert!(acc.struct_fields.fields.contains_key("core::A"));
        assert!(acc.struct_fields.fields.contains_key("alloc::B"));
    }
}
