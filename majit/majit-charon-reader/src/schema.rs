//! Top-level Charon `.llbc` schema.
//!
//! Only the fields the lowering driver actually reads are typed;
//! the rest stay as `serde_json::Value` so that newer Charon versions
//! load without code changes.

use serde::Deserialize;
use serde_json::Value;
use std::fmt;

#[derive(Debug, Deserialize)]
pub struct LlbcFile {
    pub charon_version: String,
    pub has_errors: bool,
    pub translated: Translated,
}

#[derive(Debug)]
pub struct Translated {
    pub crate_name: String,
    /// Extraction target metadata. Charon serializes this as an ordered
    /// sequence because one artefact can in principle name several targets;
    /// preserving that shape lets consumers require one unambiguous pointer
    /// width instead of silently choosing a map winner.
    pub target_information: Vec<TargetInformationEntry>,
    pub fun_decls: Vec<Option<crate::ullbc::FunDecl>>,
    /// Static / const items the MIR references via `Place::Global` and
    /// `Operand::Const(Global { ... })`. Indexed by `def_id` (the same
    /// invariant `fun_decls` upholds; verified against extracted
    /// corpora).
    pub global_decls: Vec<Option<crate::ullbc::GlobalDecl>>,
    /// User-defined types (`struct` / `enum` / alias / opaque).
    /// Indexed by `def_id`. Consumed to populate
    /// `SemanticProgram.{known_struct_names, struct_fields,
    /// immutable_fields}`.
    pub type_decls: Vec<Option<crate::ullbc::TypeDecl>>,
    /// Trait declarations. Indexed by `def_id`. Consumed for
    /// `SemanticProgram.known_trait_names`.
    pub trait_decls: Vec<Option<crate::ullbc::TraitDecl>>,
    /// Names of every trait's associated items, indexed by trait
    /// `def_id` and then by the item's id within the trait. Charon keeps
    /// this table for a trait whose declaration carries no items: a
    /// `--monomorphize` trait decl has an empty `methods`, so a
    /// `CallKind::Trait` payload's method index is named only here.
    pub assoc_item_names: Vec<Option<AssocItemNames>>,
    /// `impl Trait for T` table, indexed by trait-impl id. Kept as raw
    /// `Value` entries and projected on demand (see
    /// [`crate::Llbc::trait_impls_raw`]).  Read by the front-end's
    /// trait-associated-type resolution. Each row keeps only the paths
    /// the translator reads: `def_id`; `impl_trait.id`,
    /// `impl_trait.generics.types`, `impl_trait.generics.const_generics`;
    /// `generics.types`, `generics.const_generics`, `generics.trait_refs`;
    /// `implied_trait_refs`; `consts`; `types[].kind`,
    /// `types[].skip_binder.value`. Method ids live on the parallel
    /// [`Translated::trait_impl_methods`] table. A new reader of another
    /// path must add it to `TraitImplRow`.
    ///
    /// Every other top-level surface Charon emits (`ordered_decls`,
    /// `options`, `item_names`, `short_names`, …) is intentionally not
    /// modelled: serde skips unknown fields without allocating, which
    /// both keeps the loader resilient to Charon's release-to-release
    /// renames *and* avoids materialising the whole document as a
    /// `serde(flatten)` catch-all (the latter forces serde to buffer
    /// the entire `translated` object into an in-memory `Content` tree).
    pub trait_impls: Vec<Value>,
    /// Per-row methods of [`Self::trait_impls`], same indices, holes
    /// `None`. A missing `methods` key is `None`; JSON `null` slots stay
    /// `None` inside the inner vec.
    pub(crate) trait_impl_methods: Vec<Option<Vec<Option<ImplMethod>>>>,
    /// Source files, indexed by the `file_id` every
    /// [`crate::ullbc::SpanData`] carries.  Without it a span names a
    /// line in a file nothing can name.
    ///
    /// Each entry also carries the file's entire `contents`; that field
    /// is deliberately unmodelled, so serde walks past it without
    /// allocating.  The array was already being walked as an unknown
    /// field, so what this adds is the `id` and `name` of each entry and
    /// nothing else.
    pub files: Vec<SourceFile>,
}

#[derive(Deserialize)]
struct TranslatedDe {
    crate_name: String,
    #[serde(default)]
    target_information: Vec<TargetInformationEntry>,
    fun_decls: Vec<Option<crate::ullbc::FunDecl>>,
    #[serde(default)]
    global_decls: Vec<Option<crate::ullbc::GlobalDecl>>,
    #[serde(default)]
    type_decls: Vec<Option<crate::ullbc::TypeDecl>>,
    #[serde(default)]
    trait_decls: Vec<Option<crate::ullbc::TraitDecl>>,
    #[serde(default)]
    assoc_item_names: Vec<Option<AssocItemNames>>,
    #[serde(default, deserialize_with = "trait_impl_table")]
    trait_impls: TraitImplTable,
    #[serde(default)]
    files: Vec<SourceFile>,
}

#[derive(Default)]
struct TraitImplTable {
    rows: Vec<Value>,
    methods: Vec<Option<Vec<Option<ImplMethod>>>>,
}

impl<'de> Deserialize<'de> for Translated {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let de = TranslatedDe::deserialize(deserializer)?;
        Ok(Self {
            crate_name: de.crate_name,
            target_information: de.target_information,
            fun_decls: de.fun_decls,
            global_decls: de.global_decls,
            type_decls: de.type_decls,
            trait_decls: de.trait_decls,
            assoc_item_names: de.assoc_item_names,
            trait_impls: de.trait_impls.rows,
            trait_impl_methods: de.trait_impls.methods,
            files: de.files,
        })
    }
}

/// One row of [`Translated::assoc_item_names`]: Charon's
/// `AssocItemNames`, each list indexed by the item's id in its trait.
#[derive(Debug, Default, Deserialize)]
pub struct AssocItemNames {
    #[serde(default)]
    pub types: Vec<String>,
    #[serde(default)]
    pub methods: Vec<String>,
    #[serde(default)]
    pub consts: Vec<String>,
}

/// One row of [`Translated::files`].
#[derive(Debug, Deserialize)]
pub struct SourceFile {
    pub id: u64,
    /// Charon's `FileName`, a single-variant object (`{"Local": path}`,
    /// `{"Virtual": path}`, …).  Held raw and projected by
    /// [`crate::Llbc::file_path`] so a variant this crate has never seen
    /// loads rather than failing the whole artefact.
    pub name: Value,
}

#[derive(Debug, Deserialize)]
pub struct TargetInformationEntry {
    pub key: String,
    pub value: TargetInformation,
}

#[derive(Debug, Deserialize)]
pub struct TargetInformation {
    pub target_pointer_size: u8,
    pub is_little_endian: bool,
}

pub(crate) fn present<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(d).map(Some)
}

pub(crate) enum Nullable<T> {
    Null,
    Some(T),
}

pub(crate) fn present_nullable<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Option<Nullable<T>>, D::Error> {
    Ok(Some(match Option::<T>::deserialize(d)? {
        None => Nullable::Null,
        Some(t) => Nullable::Some(t),
    }))
}

impl<T> Nullable<T> {
    pub(crate) fn into_value(self, f: impl FnOnce(T) -> Value) -> Value {
        match self {
            Nullable::Null => Value::Null,
            Nullable::Some(t) => f(t),
        }
    }
}

pub(crate) fn insert_present(
    map: &mut serde_json::Map<String, Value>,
    key: &'static str,
    value: Option<Value>,
) {
    if let Some(value) = value {
        map.insert(key.to_string(), value);
    }
}

pub(crate) fn option_array_into_value<T>(entries: Vec<Option<T>>, f: impl Fn(T) -> Value) -> Value {
    Value::Array(
        entries
            .into_iter()
            .map(|entry| match entry {
                None => Value::Null,
                Some(entry) => f(entry),
            })
            .collect(),
    )
}

/// A `trait_impls` row reduced to the paths the translator reads.
#[derive(Deserialize)]
struct TraitImplRow {
    #[serde(default, deserialize_with = "present")]
    def_id: Option<Value>,
    #[serde(default, deserialize_with = "present_nullable")]
    impl_trait: Option<Nullable<ImplTraitRef>>,
    #[serde(default, deserialize_with = "present_nullable")]
    generics: Option<Nullable<RowGenerics>>,
    #[serde(default, deserialize_with = "present")]
    implied_trait_refs: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    consts: Option<Value>,
    /// Indexed by item id; a hole is JSON `null`.
    #[serde(default, deserialize_with = "present_nullable")]
    types: Option<Nullable<Vec<Option<AssocTypeEntry>>>>,
    /// Indexed by item id; a hole is JSON `null`.
    #[serde(default, deserialize_with = "present_nullable")]
    methods: Option<Nullable<Vec<Option<MethodEntry>>>>,
}

impl TraitImplRow {
    fn into_value_and_methods(self) -> (Value, Option<Vec<Option<ImplMethod>>>) {
        let methods = match self.methods {
            None => None,
            Some(Nullable::Null) => Some(Vec::new()),
            Some(Nullable::Some(entries)) => Some(
                entries
                    .into_iter()
                    .map(|entry| entry.map(MethodEntry::into_impl_method))
                    .collect(),
            ),
        };
        let mut map = serde_json::Map::new();
        insert_present(&mut map, "def_id", self.def_id);
        insert_present(
            &mut map,
            "impl_trait",
            self.impl_trait
                .map(|n| n.into_value(ImplTraitRef::into_value)),
        );
        insert_present(
            &mut map,
            "generics",
            self.generics.map(|n| n.into_value(RowGenerics::into_value)),
        );
        insert_present(&mut map, "implied_trait_refs", self.implied_trait_refs);
        insert_present(&mut map, "consts", self.consts);
        insert_present(
            &mut map,
            "types",
            self.types.map(|n| {
                n.into_value(|es| option_array_into_value(es, AssocTypeEntry::into_value))
            }),
        );
        (Value::Object(map), methods)
    }
}

#[derive(Deserialize)]
struct ImplTraitRef {
    #[serde(default, deserialize_with = "present")]
    id: Option<Value>,
    #[serde(default, deserialize_with = "present_nullable")]
    generics: Option<Nullable<TraitRefGenerics>>,
}

impl ImplTraitRef {
    fn into_value(self) -> Value {
        let mut map = serde_json::Map::new();
        insert_present(&mut map, "id", self.id);
        insert_present(
            &mut map,
            "generics",
            self.generics
                .map(|n| n.into_value(TraitRefGenerics::into_value)),
        );
        Value::Object(map)
    }
}

#[derive(Deserialize)]
struct TraitRefGenerics {
    #[serde(default, deserialize_with = "present")]
    types: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    const_generics: Option<Value>,
}

impl TraitRefGenerics {
    fn into_value(self) -> Value {
        let mut map = serde_json::Map::new();
        insert_present(&mut map, "types", self.types);
        insert_present(&mut map, "const_generics", self.const_generics);
        Value::Object(map)
    }
}

#[derive(Deserialize)]
struct RowGenerics {
    #[serde(default, deserialize_with = "present")]
    types: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    const_generics: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    trait_refs: Option<Value>,
}

impl RowGenerics {
    fn into_value(self) -> Value {
        let mut map = serde_json::Map::new();
        insert_present(&mut map, "types", self.types);
        insert_present(&mut map, "const_generics", self.const_generics);
        insert_present(&mut map, "trait_refs", self.trait_refs);
        Value::Object(map)
    }
}

#[derive(Deserialize)]
struct AssocTypeEntry {
    #[serde(default, deserialize_with = "present")]
    kind: Option<Value>,
    #[serde(default, deserialize_with = "present_nullable")]
    skip_binder: Option<Nullable<AssocBinder>>,
}

impl AssocTypeEntry {
    fn into_value(self) -> Value {
        let mut map = serde_json::Map::new();
        insert_present(&mut map, "kind", self.kind);
        insert_present(
            &mut map,
            "skip_binder",
            self.skip_binder
                .map(|n| n.into_value(AssocBinder::into_value)),
        );
        Value::Object(map)
    }
}

#[derive(Deserialize)]
struct AssocBinder {
    #[serde(default, deserialize_with = "present")]
    value: Option<Value>,
}

impl AssocBinder {
    fn into_value(self) -> Value {
        let mut map = serde_json::Map::new();
        insert_present(&mut map, "value", self.value);
        Value::Object(map)
    }
}

/// One impl method: `kind.TraitMethod` ids and `skip_binder.id`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplMethod {
    pub trait_id: Option<u64>,
    pub method_idx: Option<u64>,
    pub fn_id: Option<u64>,
}

#[derive(Deserialize)]
struct MethodEntry {
    #[serde(default, deserialize_with = "trait_method_ids")]
    kind: Option<(Option<u64>, Option<u64>)>,
    #[serde(default, deserialize_with = "present_nullable")]
    skip_binder: Option<Nullable<MethodBinder>>,
}

impl MethodEntry {
    fn into_impl_method(self) -> ImplMethod {
        let (trait_id, method_idx) = self.kind.unwrap_or((None, None));
        let fn_id = match self.skip_binder {
            Some(Nullable::Some(binder)) => binder.id.and_then(|id| id.as_u64()),
            _ => None,
        };
        ImplMethod {
            trait_id,
            method_idx,
            fn_id,
        }
    }
}

#[derive(Deserialize)]
struct MethodBinder {
    #[serde(default, deserialize_with = "present")]
    id: Option<Value>,
}

fn trait_method_ids<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<(Option<u64>, Option<u64>)>, D::Error> {
    struct KindVisitor;
    impl<'de> serde::de::Visitor<'de> for KindVisitor {
        type Value = Option<(Option<u64>, Option<u64>)>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a method kind")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut ids = None;
            while let Some(key) = map.next_key::<String>()? {
                if key == "TraitMethod" {
                    #[derive(Deserialize)]
                    #[serde(untagged)]
                    enum TraitMethodPayload {
                        Arr(Vec<Value>),
                        Other(serde::de::IgnoredAny),
                    }
                    if let TraitMethodPayload::Arr(arr) = map.next_value()? {
                        ids = Some((
                            arr.first().and_then(Value::as_u64),
                            arr.get(1).and_then(Value::as_u64),
                        ));
                    }
                } else {
                    let _ = map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
            Ok(ids)
        }

        fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> Result<Self::Value, A::Error> {
            while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
            Ok(None)
        }

        fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self::Value, E> {
            Ok(None)
        }
    }
    d.deserialize_any(KindVisitor)
}

fn trait_impl_table<'de, D: serde::Deserializer<'de>>(d: D) -> Result<TraitImplTable, D::Error> {
    let rows = Vec::<Option<TraitImplRow>>::deserialize(d)?;
    let mut table = TraitImplTable {
        rows: Vec::with_capacity(rows.len()),
        methods: Vec::with_capacity(rows.len()),
    };
    for row in rows {
        match row {
            None => {
                table.rows.push(Value::Null);
                table.methods.push(None);
            }
            Some(row) => {
                let (value, methods) = row.into_value_and_methods();
                table.rows.push(value);
                table.methods.push(methods);
            }
        }
    }
    Ok(table)
}
