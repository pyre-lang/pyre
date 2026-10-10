//! Charon ULLBC schema (basic-block CFG form).
//!
//! Mirrors the subset of Charon 0.1.196's ULLBC the lowering driver
//! consumes. The layout is reverse-engineered from
//! `majit/charon-corpus/corpus.ullbc`; see
//! `majit/charon-corpus/README.md` for the schema findings.
//!
//! ## Schema-drift policy
//!
//! - Fields we **read** are typed (`Ty`, `StmtKind`, `TermKind`, …).
//! - Fields we do not yet read stay as [`serde_json::Value`]. This is
//!   forward-compatible: a new Charon release that adds fields here
//!   loads without code changes.
//! - Enum variants we know about are listed by name. Unknown variants
//!   are surfaced via `#[serde(other)]` arms named `Unknown` so the
//!   reader fails-loud at the lowering site instead of silently
//!   discarding work.

use serde::Deserialize;
use serde_json::Value;
use serde_json::value::RawValue;
use std::sync::{Arc, OnceLock};

// FunDecl + meta

#[derive(Debug, Deserialize)]
pub struct FunDecl {
    pub def_id: u64,
    pub item_meta: ItemMeta,
    pub signature: Signature,
    /// Generic-parameter context of the declaration: `types` lists the
    /// type params (`{"index": N, "name": "H"|"Self"}`) and
    /// `trait_clauses` their bounds
    /// (`trait_.skip_binder.{id, generics.types[0]}` = bound trait id +
    /// subject TypeVar).  Kept as raw `Value`; only
    /// `front::mir::tyref_generic_trait_bound_root` projects it, to map
    /// a `&T`-where-`T: Trait` parameter to its bound trait's name leaf.
    #[serde(default)]
    pub generics: Option<Value>,
    /// Where the function comes from (`ItemSource`): `"Normal"`,
    /// `{"TraitImpl": ..}`, `{"GlobalInitializer": {"id": <GlobalDecl id>,
    /// ..}}`, …  Kept as raw `Value`; only
    /// [`FunDecl::is_global_initializer`] projects it.
    #[serde(default)]
    pub src: Option<Value>,
    /// `body` is `null` for opaque references and one of
    /// `{"Unstructured": {...}}`, `{"Structured": {...}}`, or
    /// `{"Error": {...}}` otherwise. Kept as the raw JSON text (not an
    /// exploded `Value` tree) so the whole-corpus retained footprint
    /// stays near the on-disk size; project to `Unstructured` on demand
    /// via [`FunDecl::unstructured`]. A schema change in the unused
    /// variants still does not break load.
    pub body: Option<Box<RawValue>>,
    /// [`FunDecl::has_unstructured_body`], decided on first query.
    #[serde(skip)]
    has_unstructured_body: OnceLock<bool>,
    /// [`FunDecl::first_arg_local_name`], decided on first query.
    #[serde(skip)]
    first_arg_local_name: OnceLock<Option<String>>,
    /// Inlinable promoted-constant initializers for this artefact, indexed
    /// by global decl id. Charon 0.1.281 emits each promoted constant as
    /// its own item; [`FunDecl::unstructured`] splices the initializer at
    /// every read, the shape 0.1.273 emitted inline. `None` when this
    /// declaration was not loaded through [`crate::Llbc::from_slice`].
    #[serde(skip)]
    promoted_inits: Option<Arc<Vec<Option<PromotedInit>>>>,
}

impl FunDecl {
    /// The `GlobalDecl` id when the function is a compiler-synthesised
    /// static / const initialiser body (e.g. the body that constructs
    /// `static NONE_SINGLETON`'s value), read from
    /// `src: {"GlobalInitializer": {"id": ..}}`.  Production lowering
    /// treats these as values rather than call targets — they have no
    /// call sites in user code, and their unwind paths use orphan
    /// exception slots that the flowspace adapter cannot lift.  `None`
    /// for ordinary function bodies.
    pub fn is_global_initializer(&self) -> Option<u64> {
        self.src
            .as_ref()?
            .get("GlobalInitializer")?
            .get("id")?
            .as_u64()
    }

    /// `true` for the function Charon synthesises for a tuple-struct or
    /// tuple-variant constructor used as a value (`src: "AdtConstructor"`).
    /// Its output type is the ADT and its name ends in the variant (or
    /// struct) name.
    pub fn is_adt_constructor(&self) -> bool {
        self.src.as_ref().and_then(Value::as_str) == Some("AdtConstructor")
    }

    /// Return the `Unstructured` (basic-block CFG) body if present.
    ///
    /// Charon 0.1.281 emits promoted constants as their own global items.
    /// When this declaration was loaded with the artefact table, each read
    /// of an inlinable promoted constant is replaced by the initializer's
    /// statements, the shape 0.1.273 emitted at the use site.
    pub fn unstructured(&self) -> Option<Unstructured> {
        #[derive(Deserialize)]
        struct Proj {
            #[serde(rename = "Unstructured")]
            unstructured: Unstructured,
        }
        let body = self.body.as_ref()?;
        let mut u = serde_json::from_str::<Proj>(body.get())
            .ok()
            .map(|p| p.unstructured)?;
        if let Some(table) = &self.promoted_inits
            && table.iter().any(Option::is_some)
            && body.get().contains("\"Global\"")
        {
            splice_promoted_reads(&mut u, table);
        }
        Some(u)
    }

    /// The `locals` table of the `Unstructured` body, without building its
    /// basic blocks: the parameters are what a declaration exposes before
    /// its body is lowered.
    pub fn unstructured_locals(&self) -> Option<Locals> {
        #[derive(Deserialize)]
        struct Body {
            locals: Locals,
        }
        #[derive(Deserialize)]
        struct Proj {
            #[serde(rename = "Unstructured")]
            unstructured: Body,
        }
        let body = self.body.as_ref()?;
        serde_json::from_str::<Proj>(body.get())
            .ok()
            .map(|p| p.unstructured.locals)
    }

    /// Whether the body is the `Unstructured` variant, the one
    /// [`Self::unstructured`] projects.
    ///
    /// Call lowering asks this of a callee at every call site; the answer is
    /// a property of the declaration, so it is decided once, from the
    /// variant tag alone: the basic blocks are skipped, not built.
    pub fn has_unstructured_body(&self) -> bool {
        *self.has_unstructured_body.get_or_init(|| {
            #[derive(Deserialize)]
            struct Proj {
                #[serde(rename = "Unstructured")]
                _unstructured: serde::de::IgnoredAny,
            }
            self.body
                .as_ref()
                .is_some_and(|body| serde_json::from_str::<Proj>(body.get()).is_ok())
        })
    }

    /// Source name of the first argument local (local index 1; index 0 is the
    /// return slot).  `None` when the body is absent, the function has no
    /// arguments, or Charon dropped the name.
    ///
    /// Projects only the `locals` table — the basic-block CFG is skipped — so a
    /// caller can distinguish a `self` receiver from an associated
    /// constructor's `Self`-typed parameter without paying for a full body
    /// parse.  Rust forbids naming a parameter `self`, so a genuine receiver's
    /// local is the `self` keyword (named `"self"`, or unnamed for a
    /// monomorphised trait method), whereas `W_Range::allocate(payload: Self)`
    /// keeps its source name (`"payload"`).
    pub fn first_arg_local_name(&self) -> Option<String> {
        self.first_arg_local_name
            .get_or_init(|| self.project_first_arg_local_name())
            .clone()
    }

    fn project_first_arg_local_name(&self) -> Option<String> {
        #[derive(Deserialize)]
        struct Proj {
            #[serde(rename = "Unstructured")]
            u: BodyProj,
        }
        #[derive(Deserialize)]
        struct BodyProj {
            locals: LocalsProj,
        }
        #[derive(Deserialize)]
        struct LocalsProj {
            arg_count: u64,
            locals: Vec<LocalNameProj>,
        }
        #[derive(Deserialize)]
        struct LocalNameProj {
            #[serde(default)]
            name: Option<String>,
        }
        let body = self.body.as_ref()?;
        let proj = serde_json::from_str::<Proj>(body.get()).ok()?;
        if proj.u.locals.arg_count < 1 {
            return None;
        }
        proj.u.locals.locals.get(1).and_then(|l| l.name.clone())
    }

    /// Returns `Some(msg)` if Charon recorded a translation error
    /// (e.g. `"charon does not support thread local references"`).
    pub fn error_message(&self) -> Option<String> {
        #[derive(Deserialize)]
        struct Proj {
            #[serde(rename = "Error")]
            error: ErrBody,
        }
        #[derive(Deserialize)]
        struct ErrBody {
            msg: String,
        }
        let body = self.body.as_ref()?;
        serde_json::from_str::<Proj>(body.get())
            .ok()
            .map(|p| p.error.msg)
    }
}

/// Static or const item referenced via [`PlaceKind::Global`].
///
/// Body / type / initialiser remain opaque (`Value`); the only field
/// the lowering driver consumes is [`ItemMeta::name_path`] for
/// constructing a stable `CallTarget::FunctionPath`-style identifier.
#[derive(Debug, Deserialize)]
pub struct GlobalDecl {
    pub def_id: u64,
    pub item_meta: ItemMeta,
    #[serde(flatten)]
    pub rest: std::collections::BTreeMap<String, Value>,
}

impl GlobalDecl {
    /// `"NamedConst"` / `"AnonConst"` / `"ThreadLocal"`, or `None` when
    /// `global_kind` is an object (`{"Static": {..}}`).
    pub fn global_kind_str(&self) -> Option<&str> {
        self.rest.get("global_kind").and_then(Value::as_str)
    }

    /// A translation-time value: a named `const`, or rustc's promoted
    /// anonymous const (`AnonConst`, Charon `PathElem::Builtin(PromotedConst)`).
    /// The promoted form copies the named const and returns a shared
    /// reference to it; both kinds fold as the same host value.
    pub fn is_const_value(&self) -> bool {
        matches!(self.global_kind_str(), Some("NamedConst" | "AnonConst"))
    }
}

/// User-defined type (`struct` / `enum` / `type` alias / opaque
/// forward-decl) the program references. The `kind` field is consumed
/// to populate `SemanticProgram.{known_struct_names,
/// struct_fields, known_trait_names}` from the LLBC alone.
#[derive(Debug, Deserialize)]
pub struct TypeDecl {
    pub def_id: u64,
    pub item_meta: ItemMeta,
    pub kind: TypeDeclKind,
    /// Per-target memory layout Charon resolved for the concrete type:
    /// a list of `{key: <target-triple>, value: TypeLayout}` entries
    /// (one per extraction target — a single-target extraction carries
    /// one). Field byte offsets live in
    /// `value.variant_layouts[variant_idx].field_offsets` (a struct is a
    /// single variant); the discriminant tag position in
    /// `value.discriminator`. Kept as raw JSON text and projected on
    /// demand via [`TypeDecl::layout_for_target`] so an unmodelled or
    /// exotic layout shape can never break the whole-LLBC load (the
    /// schema-drift policy above). `None` when Charon emitted no layout.
    #[serde(default)]
    pub layout: Option<Box<RawValue>>,
    /// Origin Charon recorded for this declaration. A compiler-generated
    /// closure environment is the object `{"Closure": …}`; an ordinary ADT
    /// is the string `"Normal"`. Kept raw so an
    /// unmodelled origin cannot fail the load.
    #[serde(default)]
    pub src: Option<Value>,
}

/// One `{key, value}` entry of [`TypeDecl::layout`].
#[derive(Debug, Deserialize)]
pub struct TargetLayout {
    pub key: String,
    pub value: TypeLayout,
}

/// Resolved memory layout of a single concrete type for one target.
#[derive(Debug, Deserialize)]
pub struct TypeLayout {
    /// Byte size. Charon spells this as a bare integer or as
    /// `{"chosen": <u64 | const expr>}`; a `Deduplicated` chosen value is
    /// filled in by [`TypeDecl::layout_for_target`].
    #[serde(default, deserialize_with = "de_layout_opt_u64")]
    pub size: Option<u64>,
    #[serde(default, deserialize_with = "de_layout_opt_u64")]
    pub align: Option<u64>,
    /// One entry per variant (a struct has a single entry). Carries the
    /// per-variant field byte offsets.
    #[serde(default)]
    pub variant_layouts: Vec<VariantLayout>,
    /// Tag/niche decoder (`{"Branch": {"offset": ..., ...}}` for a
    /// niche/tag enum). Kept raw; read the tag byte position via
    /// [`TypeLayout::discriminant_offset`].
    #[serde(default)]
    pub discriminator: Option<Value>,
    #[serde(default)]
    pub repr: Option<TypeRepr>,
}

/// Representation properties that affect the low-level value shape.
#[derive(Debug, Deserialize)]
pub struct TypeRepr {
    #[serde(default)]
    pub transparent: bool,
}

/// Per-variant field offsets within [`TypeLayout`].
#[derive(Debug, Deserialize)]
pub struct VariantLayout {
    /// Byte offset of each field, indexed by the variant's field
    /// declaration order (matches [`VariantDecl::fields`] / a struct's
    /// [`FieldDecl`] order). Each entry is a bare integer or
    /// `{"chosen": <u64>}`.
    #[serde(default, deserialize_with = "de_layout_offsets")]
    pub field_offsets: Vec<u64>,
    /// The writes that store this variant's tag, as
    /// `[[offset, {"Unsigned": ["U8", "<value>"]}], ...]`. Kept raw; read
    /// via [`VariantLayout::tag_i64`].
    #[serde(default)]
    pub tagger: Option<Value>,
}

impl VariantLayout {
    /// The tag value this variant stores, when the tagger is a single
    /// scalar write.
    pub fn tag_i64(&self) -> Option<i64> {
        let [write] = self.tagger.as_ref()?.as_array()?.as_slice() else {
            return None;
        };
        let scalar = write.as_array()?.get(1)?;
        let pair = scalar.get("Unsigned").or_else(|| scalar.get("Signed"))?;
        pair.get(1)?.as_str()?.parse::<i64>().ok()
    }
}

impl TypeDecl {
    /// Project the per-target layout, selecting the entry whose target
    /// triple equals `target`, or the sole entry when exactly one is
    /// present (single-target extraction). Returns `None` when layout is
    /// absent, unparseable, or no entry matches — callers fall back to
    /// the heuristic layout provider.
    ///
    /// `size` / `align` whose `chosen` value is a `Deduplicated` constant
    /// are resolved through `llbc`'s const table.
    pub fn layout_for_target(&self, llbc: &crate::Llbc, target: &str) -> Option<TypeLayout> {
        let raw = self.layout.as_ref()?;
        let entries: Vec<TargetLayout> = serde_json::from_str(raw.get()).ok()?;
        let mut layout = select_target_layout(entries, target)?;
        if layout.size.is_none() {
            layout.size = layout_measure(llbc, raw.get(), target, "size");
        }
        if layout.align.is_none() {
            layout.align = layout_measure(llbc, raw.get(), target, "align");
        }
        if layout
            .variant_layouts
            .iter()
            .any(|variant| variant.field_offsets.contains(&UNRESOLVED_OFFSET))
        {
            if let Some(rows) = resolved_field_offset_rows(llbc, raw.get(), target) {
                if rows.len() == layout.variant_layouts.len() {
                    for (variant, row) in layout.variant_layouts.iter_mut().zip(rows) {
                        variant.field_offsets = row;
                    }
                }
            }
        }
        if layout
            .variant_layouts
            .iter()
            .any(|variant| variant.field_offsets.contains(&UNRESOLVED_OFFSET))
        {
            return None;
        }
        Some(layout)
    }

    /// Byte size and alignment, even when a field offset is still a
    /// deduplicated expression. `size_of` / `align_of` only need these two
    /// words; [`Self::layout_for_target`] refuses the whole layout when an
    /// offset did not resolve.
    pub fn size_align_for_target(
        &self,
        llbc: &crate::Llbc,
        target: &str,
    ) -> Option<(Option<u64>, Option<u64>)> {
        let raw = self.layout.as_ref()?;
        let entries: Vec<TargetLayout> = serde_json::from_str(raw.get()).ok()?;
        let mut layout = select_target_layout(entries, target)?;
        if layout.size.is_none() {
            layout.size = layout_measure(llbc, raw.get(), target, "size");
        }
        if layout.align.is_none() {
            layout.align = layout_measure(llbc, raw.get(), target, "align");
        }
        (layout.size.is_some() || layout.align.is_some()).then_some((layout.size, layout.align))
    }

    /// Whether every emitted target layout records `#[repr(transparent)]`.
    pub fn is_repr_transparent(&self) -> bool {
        let Some(raw) = self.layout.as_ref() else {
            return false;
        };
        let Ok(entries) = serde_json::from_str::<Vec<TargetLayout>>(raw.get()) else {
            return false;
        };
        !entries.is_empty()
            && entries.iter().all(|entry| {
                entry
                    .value
                    .repr
                    .as_ref()
                    .is_some_and(|repr| repr.transparent)
            })
    }
}

/// Pick the [`TargetLayout`] matching `target`, or the sole entry when
/// exactly one is present. Returns `None` for an empty list or a
/// multi-target list with no match.
fn select_target_layout(mut entries: Vec<TargetLayout>, target: &str) -> Option<TypeLayout> {
    if let Some(pos) = entries.iter().position(|e| e.key == target) {
        return Some(entries.swap_remove(pos).value);
    }
    if entries.len() == 1 {
        return Some(entries.swap_remove(0).value);
    }
    None
}

/// A layout integer written inline: a bare `u64`, `{"chosen": <u64>}`, or a
/// constant expression whose literal is `{"Integer": {"Unsigned"|"Signed":
/// [width, decimal]}}`. A `Deduplicated` chosen value stays unresolved so
/// [`layout_measure`] can read it from the const table.
fn layout_u64_literal(v: &Value) -> Option<u64> {
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    if let Some(chosen) = v.get("chosen") {
        if chosen.get("Deduplicated").is_some() {
            return None;
        }
        return layout_u64_literal(chosen);
    }
    if let Some(body) = v
        .get("Value")
        .and_then(Value::as_array)
        .and_then(|a| a.get(1))
    {
        return layout_u64_literal(body);
    }
    if let Some(constant) = v.get("Constant") {
        return layout_u64_literal(constant);
    }
    if let Some(int) = v.get("Integer") {
        let pair = int.get("Unsigned").or_else(|| int.get("Signed"))?;
        return pair.get(1)?.as_str()?.parse().ok();
    }
    v.as_array()
        .and_then(|a| a.first())
        .and_then(layout_u64_literal)
}

fn de_layout_opt_u64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    let value = Option::<Value>::deserialize(d)?;
    Ok(value.as_ref().and_then(layout_u64_literal))
}

/// Placeholder left by [`de_layout_offsets`] when an offset is a
/// `Deduplicated` layout expression. [`TypeDecl::layout_for_target`]
/// replaces it through the layout-scalar table. A real field offset is
/// never this wide.
const UNRESOLVED_OFFSET: u64 = u64::MAX;

fn de_layout_offsets<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<u64>, D::Error> {
    let values = Vec::<Value>::deserialize(d)?;
    Ok(values
        .iter()
        .map(|v| layout_u64_literal(v).unwrap_or(UNRESOLVED_OFFSET))
        .collect())
}

/// Per-variant field offsets, each entry resolved the same way
/// [`layout_measure`] resolves `size` / `align`.
fn resolved_field_offset_rows(
    llbc: &crate::Llbc,
    raw: &str,
    target: &str,
) -> Option<Vec<Vec<u64>>> {
    let entries: Vec<Value> = serde_json::from_str(raw).ok()?;
    let entry = entries
        .iter()
        .find(|e| e.get("key").and_then(Value::as_str) == Some(target))
        .or_else(|| (entries.len() == 1).then(|| &entries[0]))?;
    let variants = entry.get("value")?.get("variant_layouts")?.as_array()?;
    let mut rows = Vec::with_capacity(variants.len());
    for variant in variants {
        let Some(offsets) = variant.get("field_offsets").and_then(Value::as_array) else {
            rows.push(Vec::new());
            continue;
        };
        let mut row = Vec::with_capacity(offsets.len());
        for offset in offsets {
            row.push(layout_u64_resolved(llbc, offset, 0)?);
        }
        rows.push(row);
    }
    Some(rows)
}

fn layout_measure(llbc: &crate::Llbc, raw: &str, target: &str, field: &str) -> Option<u64> {
    let entries: Vec<Value> = serde_json::from_str(raw).ok()?;
    let entry = entries
        .iter()
        .find(|e| e.get("key").and_then(Value::as_str) == Some(target))
        .or_else(|| (entries.len() == 1).then(|| &entries[0]))?;
    layout_u64_resolved(llbc, entry.get("value")?.get(field)?, 0)
}

/// A layout expression (`size` / `align`). Its `Deduplicated` ids index the
/// layout-expression table; the payload of a `Constant` is a constant
/// expression, whose ids index the const table instead.
fn layout_u64_resolved(llbc: &crate::Llbc, v: &Value, depth: u8) -> Option<u64> {
    if depth > 24 {
        return None;
    }
    if let Some(n) = layout_u64_literal(v) {
        return Some(n);
    }
    if let Some(id) = v.get("Deduplicated").and_then(Value::as_u64) {
        return llbc
            .layout_scalar_body(id)
            .and_then(|body| layout_u64_resolved(llbc, body, depth + 1));
    }
    if let Some(chosen) = v.get("chosen") {
        return layout_u64_resolved(llbc, chosen, depth + 1);
    }
    if let Some(body) = v
        .get("Value")
        .and_then(Value::as_array)
        .and_then(|a| a.get(1))
    {
        return layout_u64_resolved(llbc, body, depth + 1);
    }
    if let Some(constant) = v.get("Constant") {
        return layout_const_u64(llbc, constant, depth + 1);
    }
    None
}

/// A constant expression `[kind, ty]`, inline, as `{"Value": [id, [kind,
/// ty]]}`, or as `{"Deduplicated": id}` in the const table.
fn layout_const_u64(llbc: &crate::Llbc, v: &Value, depth: u8) -> Option<u64> {
    if depth > 24 {
        return None;
    }
    if let Some(n) = layout_u64_literal(v) {
        return Some(n);
    }
    if let Some(id) = v.get("Deduplicated").and_then(Value::as_u64) {
        return llbc
            .dedup_const_body(id)
            .and_then(|body| layout_const_u64(llbc, body, depth + 1));
    }
    if let Some(body) = v
        .get("Value")
        .and_then(Value::as_array)
        .and_then(|a| a.get(1))
    {
        return layout_const_u64(llbc, body, depth + 1);
    }
    v.as_array()
        .and_then(|a| a.first())
        .and_then(|kind| layout_const_u64(llbc, kind, depth + 1))
}

impl TypeLayout {
    /// Byte offset of field `field_idx` within variant `variant_idx`.
    pub fn field_offset(&self, variant_idx: usize, field_idx: usize) -> Option<u64> {
        self.variant_layouts
            .get(variant_idx)?
            .field_offsets
            .get(field_idx)
            .copied()
    }

    /// Byte offset of field `field_idx` of a struct (the single variant).
    pub fn struct_field_offset(&self, field_idx: usize) -> Option<u64> {
        self.field_offset(0, field_idx)
    }

    /// The tag of every variant of a tagged enum none of whose variants
    /// has a field, in variant order; `None` for any other layout (a
    /// struct or single-variant type has no `Branch` discriminator).
    ///
    /// A dependency's enum reaches another crate's artefact as an `Opaque`
    /// declaration that keeps only this layout, and the tags are the only
    /// place that artefact spells the variants' values.
    pub fn fieldless_enum_tags(&self) -> Option<Vec<i64>> {
        self.discriminant_offset()?;
        if self.variant_layouts.is_empty() {
            return None;
        }
        self.variant_layouts
            .iter()
            .map(|variant| {
                if variant.field_offsets.is_empty() {
                    variant.tag_i64()
                } else {
                    None
                }
            })
            .collect()
    }

    /// Byte position of the discriminant tag (`discriminator.Branch.offset`).
    /// `None` for a single-variant type or a non-`Branch` discriminator.
    pub fn discriminant_offset(&self) -> Option<u64> {
        let offset = self.discriminator.as_ref()?.get("Branch")?.get("offset")?;
        offset
            .get("chosen")
            .and_then(Value::as_u64)
            .or_else(|| offset.as_u64())
    }

    /// Rust primitive spelling of a branching enum's physical tag.
    ///
    /// Charon records this beside the tag offset as
    /// `discriminator.Branch.int_ty`.  Consumers must use both: reading a
    /// 32-bit tag as an `i64` also reads the first four payload bytes and can
    /// turn a valid variant into a switch miss.
    ///
    /// `I128`/`U128` decline. The consumers spell a tag as a field type
    /// string, and no width above a machine word has an int-bank
    /// representation there: `get_type_flag` would route `i128` through its
    /// unknown-type fallback and describe a sixteen-byte integer as an
    /// eight-byte GC reference. Declining keeps the caller on its existing
    /// no-discriminator path instead.
    pub fn discriminant_int_type(&self) -> Option<&'static str> {
        let int_ty = self.discriminator.as_ref()?.get("Branch")?.get("int_ty")?;
        let (signed, width) = if let Some(width) = int_ty.get("Signed") {
            (true, width.as_str()?)
        } else {
            (false, int_ty.get("Unsigned")?.as_str()?)
        };
        match (signed, width) {
            (true, "I8") => Some("i8"),
            (true, "I16") => Some("i16"),
            (true, "I32") => Some("i32"),
            (true, "I64") => Some("i64"),
            (false, "U8") => Some("u8"),
            (false, "U16") => Some("u16"),
            (false, "U32") => Some("u32"),
            (false, "U64") => Some("u64"),
            _ => None,
        }
    }

    /// `true` iff a `Branch` discriminator is recorded, whatever its width.
    ///
    /// The question [`Self::discriminant_int_type`] cannot answer on its own:
    /// it returns `None` both for a layout that records no discriminator at
    /// all — a single-variant enum, or a `Known` discriminator — and for one
    /// whose width has no field-type spelling. A consumer that falls back to a
    /// default tag model owes those two cases different answers.
    pub fn has_branch_discriminant(&self) -> bool {
        self.discriminator
            .as_ref()
            .and_then(|discriminator| discriminator.get("Branch"))
            .is_some()
    }

    /// Host tag of a `Branch` discriminator: byte offset, integer width,
    /// and either a direct tag per variant or a niche encoding.
    ///
    /// `None` when the discriminator is absent, its offset does not
    /// resolve, or the child ranges are not a direct map or one niche.
    pub fn tag(&self) -> Option<TagLayout> {
        let branch = self.discriminator.as_ref()?.get("Branch")?;
        let offset = self.discriminant_offset()?;
        let (signed, bits) = tag_int_ty(branch.get("int_ty")?)?;
        let children = branch.get("children").and_then(Value::as_array)?;
        let encoding = tag_encoding(children, branch.get("fallback"), bits)?;
        Some(TagLayout {
            offset,
            signed,
            bits,
            encoding,
        })
    }
}

/// Physical tag of one concrete enum layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagLayout {
    pub offset: u64,
    /// `true` when Charon recorded `Signed`.
    pub signed: bool,
    /// Bit width of the tag integer. `Isize` / `Usize` are the pointer
    /// width of the extracted target (64 on the layouts this reader sees).
    pub bits: u32,
    pub encoding: TagEncoding,
}

/// How a raw tag word names a variant. `Direct` stores one tag per
/// variant index. `Niche` is rustc's niche-filling rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TagEncoding {
    Direct {
        tags: Vec<u128>,
    },
    Niche {
        untagged_variant: usize,
        niche_variants: std::ops::RangeInclusive<usize>,
        niche_start: u128,
    },
}

fn tag_int_ty(int_ty: &Value) -> Option<(bool, u32)> {
    let (signed, width) = if let Some(width) = int_ty.get("Signed").and_then(Value::as_str) {
        (true, width)
    } else {
        (false, int_ty.get("Unsigned")?.as_str()?)
    };
    let bits = match width {
        "I8" | "U8" => 8,
        "I16" | "U16" => 16,
        "I32" | "U32" => 32,
        "I64" | "U64" | "Isize" | "Usize" => 64,
        "I128" | "U128" => 128,
        _ => return None,
    };
    Some((signed, bits))
}

fn scalar_bits(value: &Value) -> Option<u128> {
    if let Some(n) = value.as_u64() {
        return Some(u128::from(n));
    }
    let signed = value.get("Signed");
    let pair = signed.or_else(|| value.get("Unsigned"))?.as_array()?;
    let text = pair.get(1)?.as_str()?;
    if signed.is_some() {
        text.parse::<i128>().ok().map(|n| n as u128)
    } else {
        text.parse::<u128>().ok()
    }
}

enum TagDest {
    Known(usize),
    Invalid,
}

fn tag_dest(value: &Value) -> Option<TagDest> {
    if value.as_str() == Some("Invalid") {
        return Some(TagDest::Invalid);
    }
    if let Some(id) = value.get("Known").and_then(Value::as_u64) {
        return Some(TagDest::Known(id as usize));
    }
    None
}

fn tag_width_mask(bits: u32, raw: u128) -> u128 {
    if bits >= 128 {
        raw
    } else {
        raw & ((1u128 << bits) - 1)
    }
}

/// `TagEncoding::Niche` (`rustc_abi`): a variant `i` other than
/// `untagged_variant` stores `niche_start + (i - niche_variants.start)`
/// in the tag width. `niche_variants` is the span of those variants and
/// may contain `untagged_variant` as a dead value. A raw word outside
/// the span is `untagged_variant`.
fn tag_encoding(children: &[Value], fallback: Option<&Value>, bits: u32) -> Option<TagEncoding> {
    let mut known: Vec<(u128, u128, usize)> = Vec::new();
    for child in children {
        let pair = child.as_array()?;
        let range = pair.first()?;
        let start = scalar_bits(range.get("start")?)?;
        let end = scalar_bits(range.get("end")?)?;
        match tag_dest(pair.get(1)?)? {
            TagDest::Known(variant) => known.push((start, end, variant)),
            TagDest::Invalid => {}
        }
    }
    let fallback = fallback.and_then(tag_dest);
    if let Some(TagDest::Known(untagged)) = fallback {
        let mut niche: Vec<(u128, usize)> = Vec::new();
        for (start, end, variant) in known {
            if variant == untagged {
                continue;
            }
            if end != start {
                return None;
            }
            niche.push((tag_width_mask(bits, start), variant));
        }
        if niche.is_empty() {
            return None;
        }
        let lo = niche.iter().map(|(_, variant)| *variant).min()?;
        let hi = niche.iter().map(|(_, variant)| *variant).max()?;
        let niche_start = niche.iter().find(|(_, variant)| *variant == lo)?.0;
        let mut seen = vec![false; hi - lo + 1];
        for (start, variant) in &niche {
            let slot = variant - lo;
            if seen[slot] {
                return None;
            }
            let expect = tag_width_mask(bits, niche_start.wrapping_add(slot as u128));
            if *start != expect {
                return None;
            }
            seen[slot] = true;
        }
        return Some(TagEncoding::Niche {
            untagged_variant: untagged,
            niche_variants: lo..=hi,
            niche_start,
        });
    }
    if known.is_empty() || known.iter().any(|(start, end, _)| start != end) {
        return None;
    }
    let max_variant = known.iter().map(|(_, _, variant)| *variant).max()?;
    let mut tags = vec![0u128; max_variant + 1];
    let mut seen = vec![false; max_variant + 1];
    for (start, _, variant) in known {
        if seen[variant] {
            return None;
        }
        tags[variant] = tag_width_mask(bits, start);
        seen[variant] = true;
    }
    if seen.iter().any(|present| !present) {
        return None;
    }
    Some(TagEncoding::Direct { tags })
}

#[derive(Debug, Deserialize)]
pub enum TypeDeclKind {
    /// Struct body — vector of field declarations.
    Struct(Vec<FieldDecl>),
    /// Enum body — vector of variant declarations. Each variant
    /// carries its own field list (zero-arg for unit variants, named
    /// for `Foo { a: ... }`, positional for `Bar(T)`).
    Enum(Vec<VariantDecl>),
    /// Union body — a field list shaped like `Struct`'s, but every field
    /// starts at offset 0 and only one is live at a time. Foreign C unions
    /// reach the table through the dependency closure (`windows-sys`'
    /// `IN_ADDR_0`), so the variant must parse even though no field-offset
    /// row can be projected from it.
    Union(Vec<FieldDecl>),
    /// Type alias (`type T = ...`). The aliased type lives in
    /// `rest["aliased_ty"]`; not currently consumed.
    Alias(Value),
    /// Forward declaration / opaque type (Charon couldn't see body).
    Opaque,
    #[serde(other)]
    Unknown,
}

/// Fallback name the front assigns when [`FieldDecl::name`] is `None`.
/// Charon spells a positional field `"_N"` with `is_positional: true`;
/// the reader drops that spelling so a positional field and a named `_0`
/// stay distinct, and the front numbers the slot as `{prefix}{index}`.
pub const POSITIONAL_FIELD_PREFIX: &str = "__pos_";

/// `{POSITIONAL_FIELD_PREFIX}{index}` — the front's name for positional
/// field `index`.
pub fn positional_field_name(index: usize) -> String {
    format!("{POSITIONAL_FIELD_PREFIX}{index}")
}

/// Whether `name` is a positional-field fallback ([`positional_field_name`]).
pub fn is_positional_field_name(name: &str) -> bool {
    name.strip_prefix(POSITIONAL_FIELD_PREFIX)
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

/// `name` is `None` for a positional field (tuple struct / tuple variant
/// payload). Charon spells such a field `"_N"` with `is_positional: true`;
/// the name is dropped here so a positional field and a named field that
/// happens to be called `_0` stay distinct. The front then assigns
/// [`positional_field_name`].
#[derive(Debug, Deserialize)]
#[serde(from = "RawFieldDecl")]
pub struct FieldDecl {
    pub name: Option<String>,
    pub ty: TyRef,
    pub attr_info: Option<AttrInfo>,
}

#[derive(Deserialize)]
struct RawFieldDecl {
    name: Option<String>,
    #[serde(default)]
    is_positional: bool,
    ty: TyRef,
    #[serde(default)]
    attr_info: Option<AttrInfo>,
}

impl From<RawFieldDecl> for FieldDecl {
    fn from(raw: RawFieldDecl) -> Self {
        FieldDecl {
            name: if raw.is_positional { None } else { raw.name },
            ty: raw.ty,
            attr_info: raw.attr_info,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct VariantDecl {
    pub name: String,
    #[serde(default)]
    pub fields: Vec<FieldDecl>,
    /// Charon-assigned discriminant, kept raw because its scalar width
    /// varies by enum (`{"Unsigned":["U8","128"]}` for `Instruction`,
    /// `{"Signed":["Isize","0"]}` for others).
    /// Read via [`VariantDecl::discriminant_i64`]; staying [`Value`]
    /// keeps deserialization total under the schema-drift policy.
    #[serde(default)]
    pub discriminant: Option<Value>,
}

impl VariantDecl {
    /// Parse the discriminant to `i64`.
    ///
    /// Charon emits `{"Signed"|"Unsigned":[width, decimal_string]}`.
    /// A bare integer is the same value. Returns `None` for an absent or
    /// unparseable discriminant rather than failing — callers that need
    /// the value for an enum known to carry integer discriminants assert
    /// presence at the use site.
    pub fn discriminant_i64(&self) -> Option<i64> {
        let value = self.discriminant.as_ref()?;
        if let Some(n) = value.as_i64() {
            return Some(n);
        }
        let scalar = value.get("Scalar").unwrap_or(value);
        let pair = scalar.get("Unsigned").or_else(|| scalar.get("Signed"))?;
        pair.get(1)?.as_str()?.parse::<i64>().ok()
    }
}

/// Trait declaration — referenced when populating
/// `SemanticProgram.known_trait_names`. Body intentionally minimal:
/// only `item_meta.name_path()` is consumed.
#[derive(Debug, Deserialize)]
pub struct TraitDecl {
    pub def_id: u64,
    pub item_meta: ItemMeta,
    /// `methods[i].skip_binder.name` is the method a `CallKind::Trait`
    /// payload names by index. The payload no longer carries a fun-decl id.
    #[serde(default)]
    pub methods: Vec<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ItemMeta {
    pub name: Vec<NameSeg>,
    pub span: SpanRef,
    pub source_text: Option<String>,
    pub attr_info: AttrInfo,
    #[serde(default)]
    pub is_local: bool,
    /// [`ItemMeta::name_path`], rendered on first query.
    #[serde(skip)]
    name_path: OnceLock<String>,
}

impl ItemMeta {
    /// The `TraitImplId` of the trait impl block this item lives in, or
    /// `None` for an item that is not in one.
    ///
    /// [`Self::name_path`] renders that segment `"<Impl>"` and drops the
    /// id, so every trait impl in one module flattens onto a single
    /// spelling: `pyre_object::functional::<Impl>::DESCRIPTOR` is one path
    /// standing for ten distinct types' associated consts.  The id is what
    /// tells them apart, and it is the item's identity rather than a label
    /// for it — `rpython/annotator/bookkeeper.py` resolves a prebuilt
    /// through `Constant(pyobj)` (identity, per `rpython/tool/uid.py`
    /// `Hashable`) and never through a qualified name, and
    /// `rpython/rtyper/rclass.py` caches one prebuilt structure per object
    /// in an `identity_dict`.  A consumer that must distinguish siblings
    /// reads this; `name_path` stays the human-facing label.
    ///
    /// The last such segment wins: an item nested inside a trait impl
    /// belongs to the innermost one.
    pub fn trait_impl_id(&self) -> Option<u64> {
        self.name.iter().rev().find_map(|seg| match seg {
            NameSeg::Other(v) => v.get("Impl")?.get("Trait")?.as_u64(),
            NameSeg::Ident { .. } => None,
        })
    }

    /// `"crate::module::item"`-style flattened name. Trait-impl
    /// segments and other non-ident segments are rendered as
    /// `"<Variant>"` — a label, not an identity: the rendering is not
    /// injective, and [`Self::trait_impl_id`] is what recovers what it
    /// dropped.
    ///
    /// This is the template path: an `Instantiated` segment (a
    /// monomorphized copy's generic arguments) is left out, so every
    /// instance of one generic item spells the same path, the way every
    /// specialization of one `FunctionDesc` answers to `desc.name`
    /// (`rpython/annotator/description.py`).  [`Self::instantiation`]
    /// returns the arguments that tell the instances apart.
    pub fn name_path(&self) -> String {
        self.name_path_str().to_string()
    }

    /// [`Self::name_path`] without the copy.
    pub fn name_path_str(&self) -> &str {
        self.name_path.get_or_init(|| self.render_name_path())
    }

    fn render_name_path(&self) -> String {
        let mut out = String::new();
        for seg in self.template_name() {
            if !out.is_empty() {
                out.push_str("::");
            }
            match seg {
                NameSeg::Ident {
                    ident: (s, disambiguator),
                } => {
                    out.push_str(s);
                    // Multiple anonymous closures in one function all
                    // carry the bare segment `closure`; only the
                    // disambiguator index distinguishes them.  Keep it
                    // (as `closure#N`) so co-located closure envs mint
                    // distinct ClassDefs instead of collapsing their
                    // captured `__pos_N` fields onto one shared row.
                    if s == "closure" && *disambiguator > 0 {
                        out.push('#');
                        out.push_str(&disambiguator.to_string());
                    }
                }
                NameSeg::Other(v) => {
                    if let Some(label) = builtin_path_label(v) {
                        out.push_str(&label);
                        continue;
                    }
                    let label = v
                        .as_object()
                        .and_then(|m| m.keys().next().cloned())
                        .unwrap_or_else(|| "?".into());
                    out.push('<');
                    out.push_str(&label);
                    out.push('>');
                }
            }
        }
        out
    }

    /// The name segments of the template path: [`Self::name`] without the
    /// `Instantiated` segment.  A leaf read takes `.last()` of this, not of
    /// `name`, whose last segment is the instantiation on a monomorphized
    /// item.
    pub fn template_name(&self) -> impl DoubleEndedIterator<Item = &NameSeg> {
        self.name
            .iter()
            .filter(|seg| !matches!(seg, NameSeg::Other(v) if instantiated_args(v).is_some()))
    }

    /// The generic arguments of a monomorphized item: the `skip_binder` of
    /// its `Instantiated` name segment, spelled like any `GenericArgs`
    /// (`regions`, `types`, `const_generics`, `trait_refs`).  `None` for an
    /// item Charon did not instantiate.
    ///
    /// This is the specialization key `FunctionDesc.cachedgraph(key)`
    /// indexes by; [`Self::name_path`] is the desc's name.
    pub fn instantiation(&self) -> Option<&Value> {
        self.name.iter().rev().find_map(|seg| match seg {
            NameSeg::Other(v) => instantiated_args(v),
            NameSeg::Ident { .. } => None,
        })
    }
}

/// The `skip_binder` generic arguments of a `PathElem::Instantiated`
/// segment, spelled `{"Instantiated": {"params": .., "skip_binder": ..}}`.
fn instantiated_args(seg: &Value) -> Option<&Value> {
    seg.as_object()?.get("Instantiated")?.get("skip_binder")
}

/// Whether a struct-root leaf names a closure env — the bare `closure`
/// or a disambiguated `closure#N` (see [`ItemMeta::name_path`]).  Leaf
/// checks that special-case closures must accept both spellings.
pub fn is_closure_leaf(leaf: &str) -> bool {
    leaf == "closure" || leaf.starts_with("closure#")
}

/// Whether a path leaf is a Charon `PathElem::Builtin` const initializer
/// (`PromotedConst` / `AnonConst`), spelled `promoted` / `promoted#N` /
/// `anon_const` / `anon_const#N`.
///
/// Charon's default `--consts initializers` emits uses of these as
/// 0-arg calls to the synthesised FunDecl; they are values, not JIT
/// call targets.
pub fn is_const_initializer_leaf(leaf: &str) -> bool {
    leaf == "promoted"
        || leaf.starts_with("promoted#")
        || leaf == "anon_const"
        || leaf.starts_with("anon_const#")
}

/// Numbered Charon `PathElem::Builtin` leaf: `kind` at `n == 0`, `kind#n` else.
fn numbered_builtin_leaf(kind: &str, n: u64) -> String {
    if n == 0 {
        kind.to_string()
    } else {
        format!("{kind}#{n}")
    }
}

/// The segment label of a `PathElem::Builtin(kind, n)`, spelled
/// `{"Builtin": [kind, n]}`.
///
/// - `Closure` renders as `closure` / `closure#N`, the leaf the field
///   registry keys.
/// - `DropGlue` renders as `drop_in_place`, the method of the drop-glue
///   impl.
/// - `VTable` renders as `{vtable}`, the leaf of a trait's vtable struct.
/// - `PromotedConst` / `AnonConst` / `ClosureAsFn` / `VTableMethod` keep
///   distinct leaves so CallRegistry does not collapse them onto one
///   `<Builtin>` path.
///
/// Other builtins stay on the `<Builtin>` label.
pub fn builtin_path_label(seg: &Value) -> Option<String> {
    let arr = seg.as_object()?.get("Builtin")?.as_array()?;
    let n = arr.get(1).and_then(Value::as_u64).unwrap_or(0);
    match arr.first().and_then(Value::as_str)? {
        "Closure" => Some(numbered_builtin_leaf("closure", n)),
        "DropGlue" => Some("drop_in_place".to_string()),
        "VTable" => Some("{vtable}".to_string()),
        "PromotedConst" => Some(numbered_builtin_leaf("promoted", n)),
        "AnonConst" => Some(numbered_builtin_leaf("anon_const", n)),
        "ClosureAsFn" => Some(numbered_builtin_leaf("closure_as_fn", n)),
        "VTableMethod" => Some(numbered_builtin_leaf("{vtable_method}", n)),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum NameSeg {
    Ident {
        #[serde(rename = "Ident")]
        ident: (String, u64),
    },
    Other(Value),
}

#[derive(Debug, Deserialize)]
pub struct AttrInfo {
    pub attributes: Vec<Value>,
    /// `"Hint"` / `"Always"` / `"Never"` for explicit `#[inline*]` ;
    /// `null` for functions without any inline attribute.
    pub inline: Option<String>,
    pub rename: Option<String>,
    pub public: bool,
}

/// A span field. Inline bodies carry [`SpanData`]; `{"Deduplicated": id}`
/// names a row in [`crate::Llbc`]'s span table. A missing id stays unresolved.
#[derive(Debug, Clone)]
pub enum SpanRef {
    Inline(SpanData),
    Deduplicated(u64),
}

impl<'de> Deserialize<'de> for SpanRef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        span_ref_from_value(&value)
            .ok_or_else(|| serde::de::Error::custom(format!("span is not a span body: {value}")))
    }
}

fn span_ref_from_value(value: &Value) -> Option<SpanRef> {
    if let Some(data) = value.get("data") {
        return serde_json::from_value(data.clone())
            .ok()
            .map(SpanRef::Inline);
    }
    if let Some(inner) = value.get("Untagged") {
        return span_ref_from_value(inner);
    }
    if let Some(body) = value
        .get("Value")
        .and_then(Value::as_array)
        .and_then(|arr| arr.get(1))
    {
        return span_ref_from_value(body);
    }
    let id = value.get("Deduplicated").and_then(Value::as_u64)?;
    Some(SpanRef::Deduplicated(id))
}

/// `{"data": ...}` body of a hash-consed span. The id's other occurrences
/// are `{"Deduplicated": id}`.
pub(crate) fn span_data_from_body(raw: &RawValue) -> Option<SpanData> {
    #[derive(Deserialize)]
    struct Body {
        data: SpanData,
    }
    serde_json::from_str::<Body>(raw.get()).ok().map(|b| b.data)
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpanData {
    pub file_id: u64,
    pub beg: Loc,
    pub end: Loc,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Loc {
    pub line: u64,
    pub col: u64,
}

#[derive(Debug, Deserialize)]
pub struct Signature {
    pub is_unsafe: bool,
    pub inputs: Vec<TyRef>,
    pub output: TyRef,
}

// Types — kept thin. The lowering driver only needs to *label* types
// for diff output, not deeply reason about them, so the type table is
// not walked here.

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TyRef {
    /// Pointer into the global type-dedup table.
    Dedup {
        #[serde(rename = "Deduplicated")]
        id: u64,
    },
    /// Inline value with hash-cons id.
    Inline {
        #[serde(rename = "Value")]
        value: (u64, Value),
    },
    /// Anything else (e.g. literal-int short forms).
    Other(Value),
}

impl TyRef {
    /// Stable single-line label, e.g. `"ty#170"` or `"ty<Adt>"`.
    pub fn label(&self) -> String {
        match self {
            TyRef::Dedup { id } => format!("ty#{id}"),
            TyRef::Inline { value: (id, _) } => format!("ty#{id}*"),
            TyRef::Other(v) => v
                .as_object()
                .and_then(|m| m.keys().next().cloned())
                .map(|k| format!("ty<{k}>"))
                .unwrap_or_else(|| "ty<?>".into()),
        }
    }
}

// Bodies

#[derive(Debug, Deserialize)]
pub struct Unstructured {
    pub locals: Locals,
    pub body: Vec<BasicBlock>,
    pub span: SpanRef,
}

#[derive(Debug, Deserialize)]
pub struct Locals {
    pub arg_count: u64,
    pub locals: Vec<Local>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Local {
    pub index: u64,
    pub name: Option<String>,
    pub span: SpanRef,
    pub ty: TyRef,
}

#[derive(Debug, Deserialize)]
pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
    /// rustc unwind cleanup block (`is_cleanup` on the Charon block).
    #[serde(default)]
    pub is_cleanup: bool,
    /// First successful or failed projection of [`terminator`](Self::terminator).
    /// Later [`term`](Self::term) calls clone this value instead of parsing
    /// the raw JSON again.
    #[serde(skip)]
    term_cache: OnceLock<Result<TermKind, String>>,
}

impl BasicBlock {
    /// Project the terminator into the typed [`TermKind`] enum.
    /// Returns the raw JSON in the error if a variant is unknown so
    /// callers can decide whether to fail-loud or fall back.
    ///
    /// The projection is stored on the block. Every later call returns
    /// the same `Ok` value or the same `Err` string.
    pub fn term(&self, llbc: &crate::Llbc) -> Result<TermKind, String> {
        self.term_cached(llbc).clone()
    }

    /// Borrow the cached [`term`](Self::term) projection.
    pub fn term_ref(&self, llbc: &crate::Llbc) -> Result<&TermKind, &str> {
        match self.term_cached(llbc) {
            Ok(kind) => Ok(kind),
            Err(err) => Err(err.as_str()),
        }
    }

    /// Replace the raw terminator kind, forgetting the projection of the
    /// old one so the next [`term`](Self::term) reads the new kind.
    pub fn set_terminator_kind(&mut self, kind: Value) {
        self.terminator.kind =
            serde_json::value::to_raw_value(&kind).expect("a JSON value serializes");
        self.terminator.value_cache = OnceLock::from(kind);
        self.term_cache = OnceLock::new();
    }

    fn term_cached(&self, llbc: &crate::Llbc) -> &Result<TermKind, String> {
        self.term_cache
            .get_or_init(|| decode_term_kind(&self.terminator.kind, llbc))
    }
}

/// Cleanup blocks are reachable only through `on_unwind`, which the flow
/// graph does not carry, so they are left out. Every `on_unwind` edge
/// points at one terminal `UnwindResume` block.
pub fn strip_cleanup_blocks(mut body: Unstructured) -> Unstructured {
    if !body.body.iter().any(|bb| bb.is_cleanup) {
        return body;
    }
    // bb0 is the entry; a cleanup there would move another block to 0.
    if body.body.first().is_some_and(|bb| bb.is_cleanup)
        || kept_block_normal_successor_is_cleanup(&body.body)
    {
        return body;
    }

    let blocks = std::mem::take(&mut body.body);
    let mut old_to_new = vec![None; blocks.len()];
    let mut kept = Vec::with_capacity(blocks.len());
    let mut first_cleanup_span = None;
    for (i, bb) in blocks.into_iter().enumerate() {
        if bb.is_cleanup {
            if first_cleanup_span.is_none() {
                first_cleanup_span = bb.terminator.span.clone();
            }
            continue;
        }
        old_to_new[i] = Some(kept.len() as u64);
        kept.push(bb);
    }

    let resume = kept.len() as u64;
    for bb in &mut kept {
        let mut kind = bb.terminator.kind_value().clone();
        remap_term_kind_block_indices(&mut kind, &old_to_new, resume);
        bb.set_terminator_kind(kind);
    }
    kept.push(BasicBlock {
        statements: Vec::new(),
        terminator: Terminator {
            kind: serde_json::value::to_raw_value(&Value::String("UnwindResume".into()))
                .expect("a JSON value serializes"),
            span: first_cleanup_span,
            value_cache: OnceLock::new(),
        },
        is_cleanup: false,
        term_cache: OnceLock::new(),
    });
    body.body = kept;
    body
}

/// Inlinable promoted-constant initializer, keyed by global decl id.
///
/// Charon 0.1.281 emits each promoted constant as its own `AnonConst`
/// item whose last name segment is `PromotedConst`. The reader splices
/// this initializer at every read so the body matches the inline shape
/// 0.1.273 emitted.
#[derive(Debug)]
struct PromotedInit {
    /// Init locals with index ≥ 1, in table order.
    locals: Vec<Local>,
    /// Statement kinds other than those that name `_0`.
    statements: Vec<Value>,
    /// Rvalue assigned to `_0`.
    ret_rvalue: Value,
}

/// Build the per-artefact promoted-initializer table and store a clone on
/// every [`FunDecl`]. Declarations deserialized any other way keep `None`.
pub(crate) fn attach_promoted_inits(file: &mut crate::schema::LlbcFile) {
    let table = Arc::new(collect_promoted_inits(file));
    for fd in file.translated.fun_decls.iter_mut().flatten() {
        fd.promoted_inits = Some(Arc::clone(&table));
    }
}

fn collect_promoted_inits(file: &crate::schema::LlbcFile) -> Vec<Option<PromotedInit>> {
    let n = file.translated.global_decls.len();
    let mut table: Vec<Option<PromotedInit>> = (0..n).map(|_| None).collect();
    for gd in file.translated.global_decls.iter().flatten() {
        let Some(init) = promoted_init_from_global(file, gd) else {
            continue;
        };
        let idx = gd.def_id as usize;
        if idx < table.len() {
            table[idx] = Some(init);
        }
    }
    table
}

fn promoted_init_from_global(
    file: &crate::schema::LlbcFile,
    gd: &GlobalDecl,
) -> Option<PromotedInit> {
    if gd.rest.get("global_kind").and_then(Value::as_str) != Some("AnonConst") {
        return None;
    }
    if !gd
        .item_meta
        .name
        .last()
        .is_some_and(is_promoted_const_segment)
    {
        return None;
    }
    let init_id = global_init_fun_id(gd)?;
    let init_fd = file
        .translated
        .fun_decls
        .get(init_id as usize)
        .and_then(Option::as_ref)?;
    promoted_init_from_body(&init_fd.unstructured()?)
}

fn is_promoted_const_segment(seg: &NameSeg) -> bool {
    let NameSeg::Other(v) = seg else {
        return false;
    };
    v.get("Builtin")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_str)
        == Some("PromotedConst")
}

/// Init fun id stored in `gd.rest["value"]` as
/// `Value[1][0].Call[0].kind.Fun`.
fn global_init_fun_id(gd: &GlobalDecl) -> Option<u64> {
    let value = gd.rest.get("value")?;
    let body = value
        .get("Value")
        .and_then(Value::as_array)
        .and_then(|arr| arr.get(1))
        .unwrap_or(value);
    let lit = body.as_array()?.first()?;
    lit.pointer("/Call/0/kind/Fun").and_then(Value::as_u64)
}

fn is_return_terminator(kind: &Value) -> bool {
    match kind {
        Value::String(s) => s == "Return",
        Value::Object(map) => map.contains_key("Return"),
        _ => false,
    }
}

fn promoted_init_from_body(u: &Unstructured) -> Option<PromotedInit> {
    if u.body.len() != 1 {
        return None;
    }
    let bb = &u.body[0];
    if !is_return_terminator(bb.terminator.kind_value()) {
        return None;
    }
    let mut last_assign_to_0 = false;
    let mut n_assign_0 = 0;
    let mut ret_rvalue = None;
    let mut statements = Vec::new();
    for st in &bb.statements {
        let kind = st.kind_value();
        match st.stmt_kind().ok()? {
            StmtKind::StorageLive(0) | StmtKind::StorageDead(0) => {}
            StmtKind::StorageLive(_) | StmtKind::StorageDead(_) => {
                statements.push(kind.clone());
            }
            StmtKind::Assign(place, _) => {
                if matches!(place.kind, PlaceKind::Local(0)) {
                    last_assign_to_0 = true;
                    n_assign_0 += 1;
                    ret_rvalue = Some(kind.get("Assign")?.as_array()?.get(1)?.clone());
                } else {
                    last_assign_to_0 = false;
                    statements.push(kind.clone());
                }
            }
            _ => return None,
        }
    }
    if n_assign_0 != 1 || !last_assign_to_0 {
        return None;
    }
    Some(PromotedInit {
        locals: u
            .locals
            .locals
            .iter()
            .filter(|loc| loc.index != 0)
            .cloned()
            .collect(),
        statements,
        ret_rvalue: ret_rvalue?,
    })
}

struct SpliceCtx<'a> {
    table: &'a [Option<PromotedInit>],
    locals: &'a mut Locals,
    span: &'a SpanRef,
    inserts: &'a mut Vec<Statement>,
}

/// `true` when a block's raw statement or terminator kind spells a
/// `Global` place. [`splice_promoted_reads`] only rewrites those places;
/// other blocks are unchanged.
fn block_raw_has_global(bb: &BasicBlock) -> bool {
    bb.statements
        .iter()
        .any(|st| st.kind.get().contains("\"Global\""))
        || bb.terminator.kind.get().contains("\"Global\"")
}

/// Replace each inlinable promoted `Global` read with the initializer
/// statements and a local holding the init's `_0`.
///
/// Blocks with no `Global` in their raw kinds are left untouched.
/// [`Terminator::kind_value`] caches the parsed tree on the terminator, so
/// walking every cleanup block of a body that merely mentions some other
/// `Global` would retain one tree per block.
fn splice_promoted_reads(body: &mut Unstructured, table: &[Option<PromotedInit>]) {
    let mut locals = Locals {
        arg_count: body.locals.arg_count,
        locals: std::mem::take(&mut body.locals.locals),
    };
    let body_span = body.span.clone();
    for bb in &mut body.body {
        if !block_raw_has_global(bb) {
            continue;
        }
        let mut i = 0;
        while i < bb.statements.len() {
            let span = bb.statements[i].span.clone();
            let mut kind = bb.statements[i].kind_value().clone();
            let mut inserts = Vec::new();
            let mut ctx = SpliceCtx {
                table,
                locals: &mut locals,
                span: &span,
                inserts: &mut inserts,
            };
            if rewrite_value(&mut kind, &mut ctx) {
                bb.statements[i].set_kind(kind);
                let n = inserts.len();
                bb.statements.splice(i..i, inserts);
                i += n;
            }
            i += 1;
        }
        let span = bb
            .terminator
            .span
            .clone()
            .unwrap_or_else(|| body_span.clone());
        let mut kind = bb.terminator.kind_value().clone();
        let mut inserts = Vec::new();
        let mut ctx = SpliceCtx {
            table,
            locals: &mut locals,
            span: &span,
            inserts: &mut inserts,
        };
        if rewrite_value(&mut kind, &mut ctx) {
            bb.set_terminator_kind(kind);
            bb.statements.extend(inserts);
        }
    }
    body.locals.locals = locals.locals;
}

fn rewrite_value(v: &mut Value, ctx: &mut SpliceCtx<'_>) -> bool {
    if v.get("kind").is_some() && v.get("ty").is_some() {
        return rewrite_place(v, ctx);
    }
    match v {
        Value::Array(arr) => {
            let mut changed = false;
            for item in arr {
                changed |= rewrite_value(item, ctx);
            }
            changed
        }
        Value::Object(map) => {
            let mut changed = false;
            for item in map.values_mut() {
                changed |= rewrite_value(item, ctx);
            }
            changed
        }
        _ => false,
    }
}

fn rewrite_place(place: &mut Value, ctx: &mut SpliceCtx<'_>) -> bool {
    let global_id = place
        .get("kind")
        .and_then(|k| k.get("Global"))
        .and_then(|g| g.get("id"))
        .and_then(Value::as_u64);
    if let Some(id) = global_id {
        let Some(init) = ctx.table.get(id as usize).and_then(Option::as_ref) else {
            return false;
        };
        let ty = place.get("ty").cloned().unwrap_or(Value::Null);
        let t = materialize(init, ty, ctx);
        if let Some(obj) = place.as_object_mut() {
            obj.insert("kind".into(), serde_json::json!({"Local": t}));
        }
        return true;
    }
    let mut changed = false;
    if let Some(proj) = place
        .get_mut("kind")
        .and_then(|k| k.get_mut("Projection"))
        .and_then(Value::as_array_mut)
    {
        if let Some(inner) = proj.get_mut(0) {
            changed |= rewrite_place(inner, ctx);
        }
        if let Some(elem) = proj.get_mut(1) {
            changed |= rewrite_value(elem, ctx);
        }
    }
    changed
}

fn materialize(init: &PromotedInit, use_ty: Value, ctx: &mut SpliceCtx<'_>) -> u64 {
    let max_idx = init.locals.iter().map(|loc| loc.index).max().unwrap_or(0);
    let mut remap = vec![None; max_idx as usize + 1];
    for loc in &init.locals {
        let new_idx = ctx.locals.locals.len() as u64;
        if (loc.index as usize) < remap.len() {
            remap[loc.index as usize] = Some(new_idx);
        }
        let mut new_loc = loc.clone();
        new_loc.index = new_idx;
        new_loc.name = None;
        ctx.locals.locals.push(new_loc);
    }
    let t = ctx.locals.locals.len() as u64;
    remap[0] = Some(t);
    let ty = serde_json::from_value(use_ty.clone()).unwrap_or(TyRef::Other(use_ty.clone()));
    ctx.locals.locals.push(Local {
        index: t,
        name: None,
        span: ctx.span.clone(),
        ty,
    });
    for kind in &init.statements {
        let mut kind = kind.clone();
        remap_local_indices(&mut kind, &remap);
        ctx.inserts
            .push(statement_from_kind(kind, ctx.span.clone()));
    }
    let mut rv = init.ret_rvalue.clone();
    remap_local_indices(&mut rv, &remap);
    let assign = serde_json::json!({
        "Assign": [
            {"kind": {"Local": t}, "ty": use_ty},
            rv
        ]
    });
    ctx.inserts
        .push(statement_from_kind(assign, ctx.span.clone()));
    t
}

fn statement_from_kind(kind: Value, span: SpanRef) -> Statement {
    Statement {
        kind: serde_json::value::to_raw_value(&kind).expect("a JSON value serializes"),
        span,
        stmt_cache: OnceLock::new(),
        value_cache: OnceLock::from(kind),
    }
}

/// Remap `{"Local": n}` place kinds and `StorageLive` / `StorageDead`
/// payloads. Other integers (global ids, type ids, …) stay as they are.
fn remap_local_indices(v: &mut Value, remap: &[Option<u64>]) {
    match v {
        Value::Object(map) => {
            if map.len() == 1 {
                for key in ["Local", "StorageLive", "StorageDead"] {
                    if let Some(old) = map.get(key).and_then(Value::as_u64)
                        && let Some(new) = remap.get(old as usize).copied().flatten()
                    {
                        map.insert(key.to_string(), Value::from(new));
                        return;
                    }
                }
            }
            for val in map.values_mut() {
                remap_local_indices(val, remap);
            }
        }
        Value::Array(arr) => {
            for val in arr {
                remap_local_indices(val, remap);
            }
        }
        _ => {}
    }
}

fn kept_block_normal_successor_is_cleanup(blocks: &[BasicBlock]) -> bool {
    blocks.iter().any(|bb| {
        !bb.is_cleanup && term_normal_successor_is_cleanup(bb.terminator.kind_value(), blocks)
    })
}

fn is_cleanup_index(blocks: &[BasicBlock], idx: u64) -> bool {
    blocks.get(idx as usize).is_some_and(|bb| bb.is_cleanup)
}

fn term_normal_successor_is_cleanup(kind: &Value, blocks: &[BasicBlock]) -> bool {
    let Some(obj) = kind.as_object() else {
        return false;
    };
    if let Some(goto) = obj.get("Goto") {
        return goto
            .get("target")
            .and_then(Value::as_u64)
            .is_some_and(|idx| is_cleanup_index(blocks, idx));
    }
    if let Some(payload) = obj
        .get("Call")
        .or_else(|| obj.get("Drop"))
        .or_else(|| obj.get("Assert"))
    {
        return payload
            .get("target")
            .and_then(Value::as_u64)
            .is_some_and(|idx| is_cleanup_index(blocks, idx));
    }
    if let Some(switch) = obj.get("Switch") {
        return switch_normal_successor_is_cleanup(switch, blocks);
    }
    false
}

fn switch_normal_successor_is_cleanup(switch: &Value, blocks: &[BasicBlock]) -> bool {
    if let Some(branches) = switch.get("branches").and_then(Value::as_array)
        && branches
            .iter()
            .any(|v| v.as_u64().is_some_and(|idx| is_cleanup_index(blocks, idx)))
    {
        return true;
    }
    let Some(targets) = switch.get("targets") else {
        return false;
    };
    if let Some(if_arr) = targets.get("If").and_then(Value::as_array)
        && if_arr
            .iter()
            .any(|v| v.as_u64().is_some_and(|idx| is_cleanup_index(blocks, idx)))
    {
        return true;
    }
    let Some(swint) = targets.get("SwitchInt").and_then(Value::as_array) else {
        return false;
    };
    if let Some(arms) = swint.get(1).and_then(Value::as_array)
        && arms.iter().any(|arm| {
            arm.as_array()
                .and_then(|pair| pair.get(1))
                .and_then(Value::as_u64)
                .is_some_and(|idx| is_cleanup_index(blocks, idx))
        })
    {
        return true;
    }
    swint
        .get(2)
        .and_then(Value::as_u64)
        .is_some_and(|idx| is_cleanup_index(blocks, idx))
}

fn remap_block_index(old: u64, old_to_new: &[Option<u64>], resume: u64) -> u64 {
    old_to_new
        .get(old as usize)
        .copied()
        .flatten()
        .unwrap_or(resume)
}

fn remap_u64_slot(slot: &mut Value, old_to_new: &[Option<u64>], resume: u64) {
    let Some(old) = slot.as_u64() else {
        return;
    };
    *slot = Value::from(remap_block_index(old, old_to_new, resume));
}

fn remap_u64_field(value: &mut Value, field: &str, old_to_new: &[Option<u64>], resume: u64) {
    if let Some(slot) = value.get_mut(field) {
        remap_u64_slot(slot, old_to_new, resume);
    }
}

fn remap_u64_array(value: &mut Value, old_to_new: &[Option<u64>], resume: u64) {
    let Some(arr) = value.as_array_mut() else {
        return;
    };
    for slot in arr {
        remap_u64_slot(slot, old_to_new, resume);
    }
}

fn remap_term_kind_block_indices(kind: &mut Value, old_to_new: &[Option<u64>], resume: u64) {
    let Some(obj) = kind.as_object_mut() else {
        return;
    };
    if let Some(goto) = obj.get_mut("Goto") {
        remap_u64_field(goto, "target", old_to_new, resume);
        return;
    }
    for key in ["Call", "Drop", "Assert", "Panic"] {
        if let Some(payload) = obj.get_mut(key) {
            remap_u64_field(payload, "target", old_to_new, resume);
            remap_u64_field(payload, "on_unwind", old_to_new, resume);
            return;
        }
    }
    if let Some(switch) = obj.get_mut("Switch") {
        remap_switch_block_indices(switch, old_to_new, resume);
    }
}

fn remap_switch_block_indices(switch: &mut Value, old_to_new: &[Option<u64>], resume: u64) {
    if let Some(branches) = switch.get_mut("branches") {
        remap_u64_array(branches, old_to_new, resume);
    }
    let Some(targets) = switch.get_mut("targets") else {
        return;
    };
    if let Some(if_arr) = targets.get_mut("If") {
        remap_u64_array(if_arr, old_to_new, resume);
    }
    let Some(swint) = targets.get_mut("SwitchInt").and_then(Value::as_array_mut) else {
        return;
    };
    if let Some(arms) = swint.get_mut(1).and_then(Value::as_array_mut) {
        for arm in arms {
            if let Some(slot) = arm.as_array_mut().and_then(|pair| pair.get_mut(1)) {
                remap_u64_slot(slot, old_to_new, resume);
            }
        }
    }
    if let Some(default) = swint.get_mut(2) {
        remap_u64_slot(default, old_to_new, resume);
    }
}

/// A block's terminator, carrying its own span the way a [`Statement`]
/// does.  A call is a terminator rather than a statement, so the block's
/// last statement stands on a different line -- usually one that runs
/// after the call rather than at it.
#[derive(Debug, Deserialize)]
pub struct Terminator {
    /// Raw terminator-kind JSON text. Project to [`TermKind`] via
    /// [`BasicBlock::term`] so a parse error on a single terminator does
    /// not poison the whole function; [`Terminator::kind_value`] is the
    /// untyped tree.
    pub kind: Box<RawValue>,
    /// Optional for the same reason `kind` is projected rather than typed:
    /// a terminator missing its span costs the caller a fallback, and
    /// should not cost the body its parse. Charon writes one on every
    /// terminator, so this is `Some` for an artefact it produced.
    #[serde(default)]
    pub span: Option<SpanRef>,
    /// [`Terminator::kind_value`], parsed on first query.
    #[serde(skip)]
    value_cache: OnceLock<Value>,
}

impl Terminator {
    /// The terminator kind as an untyped JSON tree, parsed on first query.
    pub fn kind_value(&self) -> &Value {
        self.value_cache.get_or_init(|| {
            serde_json::from_str(self.kind.get()).expect("raw kind is a parsed JSON value")
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct Statement {
    /// Raw statement-kind JSON text. [`Statement::stmt_kind`] parses the
    /// typed kind from it; [`Statement::kind_value`] is the untyped tree.
    pub kind: Box<RawValue>,
    pub span: SpanRef,
    /// First successful or failed projection of [`kind`](Self::kind).
    /// Later [`stmt_kind`](Self::stmt_kind) calls clone this value instead
    /// of parsing the raw JSON again.
    #[serde(skip)]
    stmt_cache: OnceLock<Result<StmtKind, String>>,
    /// [`Statement::kind_value`], parsed on first query.
    #[serde(skip)]
    value_cache: OnceLock<Value>,
}

impl Statement {
    /// The statement kind as an untyped JSON tree, parsed on first query.
    pub fn kind_value(&self) -> &Value {
        self.value_cache.get_or_init(|| {
            serde_json::from_str(self.kind.get()).expect("raw kind is a parsed JSON value")
        })
    }

    /// Project to the typed [`StmtKind`] enum.
    ///
    /// The projection is stored on the statement. Every later call returns
    /// the same `Ok` value or the same `Err` string.
    pub fn stmt_kind(&self) -> Result<StmtKind, String> {
        self.stmt_cached().clone()
    }

    /// Borrow the cached [`stmt_kind`](Self::stmt_kind) projection.
    pub fn stmt_kind_ref(&self) -> Result<&StmtKind, &str> {
        match self.stmt_cached() {
            Ok(kind) => Ok(kind),
            Err(err) => Err(err.as_str()),
        }
    }

    fn stmt_cached(&self) -> &Result<StmtKind, String> {
        self.stmt_cache.get_or_init(|| {
            let kind = self.kind.get();
            serde_json::from_str::<StmtKind>(kind).map_err(|e| format!("{e}; raw kind: {kind}"))
        })
    }

    /// Replace the raw statement kind, forgetting the projection of the
    /// old one so the next [`stmt_kind`](Self::stmt_kind) reads the new kind.
    fn set_kind(&mut self, kind: Value) {
        self.kind = serde_json::value::to_raw_value(&kind).expect("a JSON value serializes");
        self.value_cache = OnceLock::from(kind);
        self.stmt_cache = OnceLock::new();
    }
}

// Statements

#[derive(Debug, Clone, Deserialize)]
pub enum StmtKind {
    /// Local enters scope.
    StorageLive(u64),
    /// Local leaves scope.
    StorageDead(u64),
    /// `place := rvalue`
    Assign(Place, Rvalue),
    /// Borrow-checker fact. No runtime effect.
    Borrowck(Value),
    /// `Assert { cond, expected, check_kind }` — inline assertion;
    /// failure terminator is the *terminator-level* `Assert` instead.
    Assert(AssertStmt),
    /// `let _ = place` style references (MIR `PlaceMention`).
    PlaceMention(Place),
    /// Anything else (e.g. `Deinit`, `SetDiscriminant`, …).
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssertStmt {
    pub cond: Operand,
    pub expected: bool,
    pub check_kind: Value,
}

// Places, operands, rvalues

#[derive(Debug, Clone, Deserialize)]
pub struct Place {
    pub kind: PlaceKind,
    pub ty: TyRef,
}

#[derive(Debug, Clone, Deserialize)]
pub enum PlaceKind {
    Local(u64),
    Projection(Box<Place>, ProjectionElem),
    /// Reference to a static / const global item.
    /// `Global { generics, id }` — `id` indexes `global_decls`.
    Global {
        generics: Value,
        id: u64,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ProjectionElem {
    /// `"Deref"` and similar atom variants.
    Atom(String),
    Tagged(Value),
}

impl ProjectionElem {
    pub fn label(&self) -> String {
        match self {
            ProjectionElem::Atom(s) => s.clone(),
            ProjectionElem::Tagged(v) => {
                if let Some(obj) = v.as_object()
                    && let Some(k) = obj.keys().next()
                {
                    return k.clone();
                }
                "?".into()
            }
        }
    }

    /// The `Field` payload of this projection element, if it is one.
    pub fn field_payload(&self) -> Option<&Value> {
        match self {
            ProjectionElem::Tagged(v) => v.as_object().and_then(|m| m.get("Field")),
            ProjectionElem::Atom(_) => None,
        }
    }
}

impl Place {
    /// Whether this place is a `Deref` projection.
    ///
    /// `front/mir.rs` records this as `FieldDescriptor::base_is_deref` on the
    /// inner of a field projection. `true` is `(*p).f`; `false` is a
    /// projection off a local aggregate (`local.f`).
    pub fn is_deref_projection(&self) -> bool {
        matches!(
            &self.kind,
            PlaceKind::Projection(_, ProjectionElem::Atom(s)) if s == "Deref"
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum Rvalue {
    /// Second value is `WithRetag` (`"Yes"` / `"No"`).
    Use(Operand, Value),
    /// `BinaryOp(op, lhs, rhs)`. `op` is a tagged variant — primitive
    /// ops are atom strings (`"Add"`, `"Eq"`, …), wrap/overflow forms
    /// are objects (`{"Shr": "Wrap"}`, `{"Add": "Wrap"}`).
    BinaryOp(Value, Operand, Operand),
    UnaryOp(Value, Operand),
    /// `Ref { place, kind, ptr_metadata }` — borrow / raw-ptr creation.
    Ref {
        place: Place,
        /// `"Shared" | "Mut" | "TwoPhaseMut" | …`
        kind: Value,
        ptr_metadata: Value,
    },
    /// `Aggregate(kind, operands)` — tuple / struct / enum-variant /
    /// array construction.
    Aggregate(Value, Vec<Operand>),
    Discriminant(Place),
    /// `Cast(kind, operand, target_ty)`.
    Cast(Value, Operand, TyRef),
    /// `Len(place)` for slice / array length.
    Len(Place),
    /// `Repeat(operand, elem_ty, count, trait_info)` for `[v; N]` literals.
    /// The last value is the `Copy`/`Clone` witness Charon now records.
    Repeat(Operand, TyRef, Value, Value),
    /// `ShallowInitBox(operand, target_ty)` — emitted by `Box::new_in`
    /// and friends to allocate the box and initialise its contents.
    ShallowInitBox(Operand, TyRef),
    /// `RawPtr { place, kind }` — raw-pointer construction (sibling of `Ref`).
    RawPtr {
        place: Place,
        kind: Value,
        ptr_metadata: Value,
    },
    /// `NullaryOp(op, type)` — `SizeOf(T)`, `AlignOf(T)`, etc.
    NullaryOp(Value, TyRef),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Const(Value),
}

/// Regular `Fun` id carried by an `Operand::Const` whose kind is `FnDef`
/// (`[{"FnDef": {"kind": {"Fun": id}, ..}}, ty]`). Literals, `VTableRef`,
/// and `TraitConst` are not function items and return `None`.
pub fn const_fn_def_regular_id(llbc: &crate::Llbc, value: &Value) -> Option<u64> {
    llbc.const_expr_kind(value)?
        .get("FnDef")?
        .get("kind")?
        .get("Fun")?
        .as_u64()
}

// Terminators

#[derive(Debug, Clone, Deserialize)]
pub enum TermKind {
    Return,
    UnwindResume,
    Abort(Value),
    /// Charon `TerminatorKind::Panic`.
    Panic {
        name: Value,
        on_unwind: u64,
    },
    Goto {
        target: u64,
    },
    Switch {
        discr: Operand,
        targets: SwitchTargets,
    },
    Call {
        call: CallPayload,
        target: u64,
        on_unwind: u64,
    },
    Assert {
        assert: AssertStmt,
        target: u64,
        on_unwind: u64,
    },
    Drop {
        /// Value whose destructor runs.  The GC liveness pass needs the place
        /// so a RootScope drop can end the bracket it opened.
        place: Place,
        /// Drop glue resolved by Charon for `place`'s type.
        fn_ptr: RegularCall,
        target: u64,
        on_unwind: u64,
    },
    /// Charon `TerminatorKind::UnwindTerminate`.
    UnwindTerminate,
    /// Charon `TerminatorKind::UndefinedBehavior`.
    UndefinedBehavior,
    #[serde(other)]
    Unknown,
}

fn decode_term_kind(kind: &RawValue, llbc: &crate::Llbc) -> Result<TermKind, String> {
    let raw = kind.get();
    // Only a `Switch` can carry `data`, whose arm constants are read off
    // the dedup tables; every other kind parses straight from the text.
    let is_switch = raw
        .trim_start()
        .strip_prefix('{')
        .is_some_and(|rest| rest.trim_start().starts_with("\"Switch\""));
    let mut term = if is_switch {
        let kind: Value = serde_json::from_str(raw).map_err(|e| format!("{e}; raw kind: {raw}"))?;
        if let Some(sw) = kind.get("Switch")
            && sw.get("data").is_some()
        {
            return decode_switch(sw, llbc).map_err(|e| format!("{e}; raw kind: {kind}"));
        }
        TermKind::deserialize(&kind).map_err(|e| format!("{e}; raw kind: {kind}"))?
    } else {
        serde_json::from_str(raw).map_err(|e| format!("{e}; raw kind: {raw}"))?
    };
    match &mut term {
        TermKind::Call {
            call:
                CallPayload {
                    func: CallFunc::Regular(reg),
                    ..
                },
            ..
        } => reg.take_instance_arguments(llbc),
        TermKind::Drop { fn_ptr, .. } => fn_ptr.take_instance_arguments(llbc),
        _ => {}
    }
    Ok(term)
}

/// `Switch { data: {scrutinee, branches, fallback}, branches }` decodes
/// straight into [`TermKind::Switch`]. Arm constants may be
/// `{"Deduplicated": id}` and are read from [`crate::Llbc::dedup_const_body`].
/// A scalar ConstantExpr folds through `const_expr_literal`. A
/// pointer/vtable arm is a bare `Ref`+`DynTrait` kind rather than
/// `[kind, ty]`; keep that JSON so the terminator decodes as
/// `SwitchInt` (`rptr.py`/`rclass.py` `ptr_nonzero` If via
/// `lower_niche_option_switch`). Any other non-scalar arm is dropped;
/// if none remain, take the fallback so the function stays in the program.
fn decode_switch(sw: &Value, llbc: &crate::Llbc) -> Result<TermKind, String> {
    let data = sw.get("data").ok_or("switch missing data")?;
    let bbs = sw
        .get("branches")
        .and_then(Value::as_array)
        .ok_or("switch missing block branches")?;
    let discr_v = data
        .get("scrutinee")
        .and_then(|s| s.get("Value"))
        .ok_or("switch missing scrutinee")?;
    let discr = Operand::deserialize(discr_v).map_err(|e| e.to_string())?;
    let arms = data
        .get("branches")
        .and_then(Value::as_array)
        .ok_or("switch missing arms")?;
    let fallback = data.get("fallback").and_then(Value::as_u64);
    let bb_of = |id: u64| bbs.get(id as usize).and_then(Value::as_u64);
    let mut decoded: Vec<(Value, u64, Option<bool>)> = Vec::new();
    for arm in arms {
        let pair = arm.as_array().ok_or("switch arm is not a pair")?;
        let target = pair
            .get(1)
            .and_then(Value::as_u64)
            .and_then(bb_of)
            .ok_or("switch arm target")?;
        let const_v = pair.first().ok_or("switch arm const")?;
        // A scalar ConstantExpr folds through `const_expr_literal`. A
        // pointer/vtable arm is a bare `Ref`+`DynTrait` kind rather than
        // `[kind, ty]`; keep that JSON so the terminator decodes as
        // `SwitchInt`. Other non-scalars are dropped (fallback Goto if
        // nothing remains).
        let lit = match llbc.const_expr_literal(const_v) {
            Some(lit) => lit,
            None => {
                let kept = llbc
                    .const_expr_kind(const_v)
                    .unwrap_or_else(|| const_v.clone());
                if kept.get("Ref").is_some() || const_v.get("Ref").is_some() {
                    kept
                } else {
                    continue;
                }
            }
        };
        let flag = lit.get("Bool").and_then(Value::as_bool);
        decoded.push((lit, target, flag));
    }
    let default = fallback.and_then(bb_of).ok_or("switch missing fallback")?;
    if decoded.is_empty() {
        return Ok(TermKind::Goto { target: default });
    }
    let all_bool = decoded.iter().all(|(_, _, flag)| flag.is_some());
    let targets = if all_bool {
        let mut then_bb = Some(default);
        let mut else_bb = None;
        for (_, target, flag) in &decoded {
            if flag == &Some(true) {
                then_bb = Some(*target);
            } else {
                else_bb = Some(*target);
            }
        }
        let then_bb = then_bb.ok_or("bool switch missing then")?;
        let else_bb = else_bb.unwrap_or(default);
        SwitchTargets::If(then_bb, else_bb)
    } else {
        let arms = decoded
            .into_iter()
            .map(|(scalar, target, _)| (scalar, target))
            .collect();
        SwitchTargets::SwitchInt(Value::Null, arms, default)
    };
    Ok(TermKind::Switch { discr, targets })
}

#[derive(Debug, Clone, Deserialize)]
pub enum SwitchTargets {
    /// Boolean switch: `[then_bb, else_bb]`.
    If(u64, u64),
    /// `SwitchInt(int_ty, [(scalar, bb)], default_bb)`.
    SwitchInt(Value, Vec<(Value, u64)>, u64),
}

#[derive(Debug, Clone, Deserialize)]
pub struct CallPayload {
    pub func: CallFunc,
    pub args: Vec<Operand>,
    pub dest: Place,
}

/// Charon's `Call.func` is one of two top-level variants:
///   - `Regular { kind, generics }`   — statically resolved
///   - `Dynamic <operand>`            — `dyn Trait` virtual call
///
/// The inner `kind` of `Regular` further distinguishes `Fun(Regular n)`
/// (monomorphized direct call), `Fun(Trait …)` (trait-bound generic
/// resolved at extraction time), or `Ptr` (function-pointer call).
#[derive(Debug, Clone, Deserialize)]
pub enum CallFunc {
    Regular(RegularCall),
    /// `dyn Trait` virtual call. The operand carries the fat pointer
    /// the dispatch reads from.
    Dynamic(Operand),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegularCall {
    pub kind: CallKind,
    pub generics: Value,
}

impl RegularCall {
    /// Spell a call to a monomorphized copy the way a call to its generic
    /// item is spelled: the callee's instance arguments become the call's
    /// `generics`.  Charon's `--monomorphize` moves the call site's type,
    /// const and trait arguments into the callee's `Instantiated` name
    /// segment and leaves the call with its regions only, so a reader of
    /// `generics` would see an unparameterized call.  The call keeps its own
    /// regions; a call that already carries arguments is left alone.
    fn take_instance_arguments(&mut self, llbc: &crate::Llbc) {
        let CallKind::Fun(FunId::Regular { id }) = &self.kind else {
            return;
        };
        let Some(args) = llbc
            .fn_by_id(*id)
            .and_then(|fd| fd.item_meta.instantiation())
        else {
            return;
        };
        let Some(generics) = self.generics.as_object_mut() else {
            return;
        };
        const KEYS: [&str; 3] = ["types", "const_generics", "trait_refs"];
        let unparameterized = KEYS.iter().all(|key| {
            generics
                .get(*key)
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
        });
        if !unparameterized {
            return;
        }
        for key in KEYS {
            if let Some(v) = args.get(key) {
                generics.insert(key.to_string(), v.clone());
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum CallKind {
    /// Statically resolved function call: `Fun { Regular(fn_id) }` or
    /// `Fun { Trait(...) }`.
    Fun(FunId),
    /// Static trait method call (post-resolution).
    Trait(Value),
    /// `Ptr` (function-pointer call).
    Ptr(Value),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone)]
pub enum FunId {
    /// `CallKind::Fun` carries the fun-decl id as a bare integer.
    Regular {
        id: u64,
    },
    Other(Value),
}

impl<'de> Deserialize<'de> for FunId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        if let Some(id) = value.as_u64() {
            return Ok(FunId::Regular { id });
        }
        Ok(FunId::Other(value))
    }
}

impl CallFunc {
    /// Bucket the call into a `CallClass` for the lowering driver to
    /// dispatch on.
    pub fn classify(&self) -> CallClass {
        match self {
            CallFunc::Regular(r) => match &r.kind {
                CallKind::Fun(FunId::Regular { .. }) => CallClass::Direct,
                CallKind::Fun(FunId::Other(_)) | CallKind::Trait(_) => CallClass::Trait,
                CallKind::Ptr(_) => CallClass::Ptr,
                CallKind::Unknown => CallClass::Unknown,
            },
            CallFunc::Dynamic(_) => CallClass::Dynamic,
            CallFunc::Unknown => CallClass::Unknown,
        }
    }
}

/// Bucket the lowering driver dispatches on for call terminators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallClass {
    /// Monomorphized direct call — emit as `direct_call`.
    Direct,
    /// Trait-bound generic — also direct (Charon already resolved).
    Trait,
    /// `dyn Trait` virtual call — emit as indirect call through fat
    /// pointer; lowering may devirtualize on type-flow.
    Dynamic,
    /// Function-pointer call.
    Ptr,
    /// Unrecognised — fail-loud at lowering site.
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item_meta(name: Value) -> ItemMeta {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "span": {"Deduplicated": 0},
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        }))
        .unwrap()
    }

    /// A monomorphized copy keeps its generic item's path; its arguments
    /// come from `instantiation`.
    #[test]
    fn instantiated_segment_is_left_out_of_the_template_path() {
        let args = serde_json::json!({
            "regions": [], "types": [{"Deduplicated": 3}],
            "const_generics": [], "trait_refs": []
        });
        let mono = item_meta(serde_json::json!([
            {"Ident": ["core", 0]},
            {"Ident": ["ptr", 0]},
            {"Ident": ["null", 0]},
            {"Instantiated": {"params": {}, "skip_binder": args.clone(), "kind": "Other"}}
        ]));
        assert_eq!(mono.name_path(), "core::ptr::null");
        assert_eq!(mono.instantiation(), Some(&args));
        assert!(matches!(
            mono.template_name().last(),
            Some(NameSeg::Ident { ident: (leaf, _) }) if leaf == "null"
        ));

        let generic = item_meta(serde_json::json!([
            {"Ident": ["core", 0]},
            {"Ident": ["ptr", 0]},
            {"Ident": ["null", 0]}
        ]));
        assert_eq!(generic.name_path(), "core::ptr::null");
        assert_eq!(generic.instantiation(), None);
    }

    #[test]
    fn builtin_path_kinds_keep_distinct_leaves() {
        let cases = [
            (r#"{"Builtin":["Closure",0]}"#, "closure"),
            (r#"{"Builtin":["Closure",2]}"#, "closure#2"),
            (r#"{"Builtin":["DropGlue",0]}"#, "drop_in_place"),
            (r#"{"Builtin":["VTable",0]}"#, "{vtable}"),
            (r#"{"Builtin":["PromotedConst",0]}"#, "promoted"),
            (r#"{"Builtin":["PromotedConst",3]}"#, "promoted#3"),
            (r#"{"Builtin":["AnonConst",0]}"#, "anon_const"),
            (r#"{"Builtin":["AnonConst",1]}"#, "anon_const#1"),
            (r#"{"Builtin":["ClosureAsFn",0]}"#, "closure_as_fn"),
            (r#"{"Builtin":["VTableMethod",0]}"#, "{vtable_method}"),
        ];
        for (json, want) in cases {
            let v: Value = serde_json::from_str(json).unwrap();
            assert_eq!(builtin_path_label(&v).as_deref(), Some(want), "{json}");
        }
        let unknown: Value = serde_json::from_str(r#"{"Builtin":["OtherKind",0]}"#).unwrap();
        assert_eq!(builtin_path_label(&unknown), None);

        let meta = item_meta(serde_json::json!([
            {"Ident": ["pyre_interpreter", 0]},
            {"Ident": ["baseobjspace", 0]},
            {"Ident": ["getattr_str_impl", 0]},
            {"Builtin": ["AnonConst", 0]}
        ]));
        assert_eq!(
            meta.name_path(),
            "pyre_interpreter::baseobjspace::getattr_str_impl::anon_const"
        );
        assert!(is_const_initializer_leaf("promoted"));
        assert!(is_const_initializer_leaf("promoted#11"));
        assert!(is_const_initializer_leaf("anon_const"));
        assert!(is_const_initializer_leaf("anon_const#1"));
        assert!(!is_const_initializer_leaf("closure"));
        assert!(!is_const_initializer_leaf("getindex_w_index"));
    }

    /// A call to a monomorphized copy reads the copy's instance arguments
    /// as its `generics`; a call that carries its own keeps them.
    #[test]
    fn a_call_to_an_instance_carries_the_instance_arguments() {
        let doc = r#"{"charon_version":"t","has_errors":false,
            "translated":{"crate_name":"c","fun_decls":[{
                "def_id":0,
                "item_meta":{"name":[{"Ident":["core",0]},{"Ident":["into",0]},
                    {"Instantiated":{"params":{},"kind":"Other","skip_binder":{
                        "regions":[],"types":[{"Deduplicated":3},{"Deduplicated":4}],
                        "const_generics":[],"trait_refs":[]}}}],
                    "span":{"Deduplicated":0},"source_text":null,
                    "attr_info":{"attributes":[],"inline":null,"rename":null,"public":true},
                    "is_local":false},
                "signature":{"is_unsafe":false,"inputs":[],"output":{"Deduplicated":3}},
                "body":null}]}}"#;
        let llbc = crate::Llbc::from_slice(doc.as_bytes()).expect("fixture parses");
        let call = |generics: Value| {
            serde_json::json!({"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": 0}, "generics": generics}},
                    "args": [],
                    "dest": {"kind": {"Local": 0}, "ty": {"Deduplicated": 3}}
                },
                "target": 1,
                "on_unwind": 2
            }})
        };
        let generics_of = |kind: &Value| match decode_term_kind(
            &serde_json::value::to_raw_value(kind).expect("a JSON value serializes"),
            &llbc,
        ) {
            Ok(TermKind::Call {
                call:
                    CallPayload {
                        func: CallFunc::Regular(reg),
                        ..
                    },
                ..
            }) => reg.generics,
            other => panic!("not a regular call: {other:?}"),
        };
        let mono = generics_of(&call(serde_json::json!({
            "regions": [{"Body": 9}], "types": [], "const_generics": [], "trait_refs": []
        })));
        assert_eq!(
            mono,
            serde_json::json!({
                "regions": [{"Body": 9}],
                "types": [{"Deduplicated": 3}, {"Deduplicated": 4}],
                "const_generics": [], "trait_refs": []
            })
        );
        let own = serde_json::json!({
            "regions": [], "types": [{"Deduplicated": 5}], "const_generics": [], "trait_refs": []
        });
        assert_eq!(generics_of(&call(own.clone())), own);
    }

    /// A positional field loses its `_N` spelling; a named `_0` keeps it.
    #[test]
    fn positional_field_has_no_name() {
        let ty = r#"{"Deduplicated": 0}"#;
        let positional: FieldDecl = serde_json::from_str(&format!(
            r#"{{"name": "_0", "is_positional": true, "ty": {ty}}}"#
        ))
        .unwrap();
        assert_eq!(positional.name, None);
        let named: FieldDecl = serde_json::from_str(&format!(
            r#"{{"name": "_0", "is_positional": false, "ty": {ty}}}"#
        ))
        .unwrap();
        assert_eq!(named.name.as_deref(), Some("_0"));
        assert_eq!(positional_field_name(0), "__pos_0");
        assert!(is_positional_field_name("__pos_0"));
        assert!(is_positional_field_name("__pos_12"));
        assert!(!is_positional_field_name("__pos_"));
        assert!(!is_positional_field_name("_0"));
    }

    /// A struct's field offsets are the single `variant_layouts[0]` entry.
    #[test]
    fn struct_layout_field_offsets() {
        let json = r#"{
            "size": 80, "align": 8,
            "discriminator": null, "uninhabited": false,
            "variant_layouts": [
                {"field_offsets": [0, 24, 48, 72, 73], "uninhabited": false, "tagger": []}
            ],
            "repr": {"repr_algo": "Rust"}
        }"#;
        let layout: TypeLayout = serde_json::from_str(json).unwrap();
        assert_eq!(layout.size, Some(80));
        assert_eq!(layout.align, Some(8));
        assert_eq!(layout.struct_field_offset(0), Some(0));
        assert_eq!(layout.struct_field_offset(3), Some(72));
        assert_eq!(layout.struct_field_offset(4), Some(73));
        assert_eq!(layout.struct_field_offset(5), None);
        assert!(!layout.repr.as_ref().unwrap().transparent);
        // single-variant type has no decodable discriminant tag
        assert_eq!(layout.discriminant_offset(), None);
    }

    #[test]
    fn transparent_repr_is_preserved() {
        let json = r#"{
            "size": 8, "align": 8,
            "variant_layouts": [{"field_offsets": [0]}],
            "repr": {"repr_algo": "Rust", "transparent": true}
        }"#;
        let layout: TypeLayout = serde_json::from_str(json).unwrap();
        assert!(layout.repr.unwrap().transparent);
    }

    /// An enum's per-variant field offsets, plus the niche tag position.
    #[test]
    fn enum_layout_per_variant_offsets_and_tag() {
        let json = r#"{
            "size": 104, "align": 8,
            "discriminator": {"Branch": {"offset": 0, "int_ty": {"Unsigned": "U8"}}},
            "uninhabited": false,
            "variant_layouts": [
                {"field_offsets": [0], "uninhabited": false, "tagger": []},
                {"field_offsets": [8], "uninhabited": false,
                 "tagger": [[0, {"Unsigned": ["U8", "6"]}]]}
            ],
            "repr": {"repr_algo": "Rust"}
        }"#;
        let layout: TypeLayout = serde_json::from_str(json).unwrap();
        assert_eq!(layout.field_offset(0, 0), Some(0));
        assert_eq!(layout.field_offset(1, 0), Some(8));
        assert_eq!(layout.field_offset(2, 0), None);
        assert_eq!(layout.discriminant_offset(), Some(0));
        assert_eq!(layout.discriminant_int_type(), Some("u8"));
        // No `children`: the offset is known, the encoding is not.
        assert!(layout.tag().is_none());
    }

    fn layout_from(discriminator: &str, variants: &str) -> TypeLayout {
        serde_json::from_str(&format!(
            r#"{{"discriminator":{discriminator},"variant_layouts":{variants}}}"#
        ))
        .unwrap()
    }

    /// Direct tags, a one-value niche, and a niche whose untagged variant
    /// sits in a hole of the span. Discriminator objects are copied from
    /// three concrete decls plus one niche whose `Invalid` child is the
    /// dead tag of the untagged index.
    #[test]
    fn branch_tag_is_direct_or_niche() {
        // Fieldless three-variant branch, fallback `Invalid`.
        let direct = layout_from(
            r#"{"Branch":{"offset":{"guarantee":null,"chosen":0},"int_ty":{"Unsigned":"U8"},"children":[[{"start":{"Unsigned":["U8","0"]},"end":{"Unsigned":["U8","0"]}},{"Known":0}],[{"start":{"Unsigned":["U8","1"]},"end":{"Unsigned":["U8","1"]}},{"Known":1}],[{"start":{"Unsigned":["U8","2"]},"end":{"Unsigned":["U8","2"]}},{"Known":2}]],"fallback":"Invalid"}}"#,
            r#"[{"field_offsets":[]},{"field_offsets":[]},{"field_offsets":[]}]"#,
        );
        assert_eq!(
            direct.tag(),
            Some(TagLayout {
                offset: 0,
                signed: false,
                bits: 8,
                encoding: TagEncoding::Direct {
                    tags: vec![0, 1, 2],
                },
            })
        );

        // Niche value 0 is variant 0; variant 1 is the untagged fallback.
        // Payload seats: variant 0 at 8, variant 1 at 0.
        let site = layout_from(
            r#"{"Branch":{"offset":{"guarantee":null,"chosen":0},"int_ty":{"Signed":"Isize"},"children":[[{"start":{"Signed":["Isize","0"]},"end":{"Signed":["Isize","0"]}},{"Known":0}]],"fallback":{"Known":1}}}"#,
            r#"[{"field_offsets":[{"guarantee":null,"chosen":8}]},{"field_offsets":[{"guarantee":null,"chosen":0}]}]"#,
        );
        assert_eq!(site.field_offset(0, 0), Some(8));
        assert_eq!(site.field_offset(1, 0), Some(0));
        assert_eq!(
            site.tag(),
            Some(TagLayout {
                offset: 0,
                signed: true,
                bits: 64,
                encoding: TagEncoding::Niche {
                    untagged_variant: 1,
                    niche_variants: 0..=0,
                    niche_start: 0,
                },
            })
        );

        // Variant 0 is tag 2 at byte 24; variant 1 is untagged. 3..=255 is
        // the unused hole. Field seats are 0/16 and 8/0.
        let range_case = layout_from(
            r#"{"Branch":{"offset":{"guarantee":null,"chosen":24},"int_ty":{"Unsigned":"U8"},"children":[[{"start":{"Unsigned":["U8","2"]},"end":{"Unsigned":["U8","2"]}},{"Known":0}],[{"start":{"Unsigned":["U8","3"]},"end":{"Unsigned":["U8","255"]}},"Invalid"]],"fallback":{"Known":1}}}"#,
            r#"[{"field_offsets":[{"guarantee":null,"chosen":0},{"guarantee":null,"chosen":16}]},{"field_offsets":[{"guarantee":null,"chosen":8},{"guarantee":null,"chosen":0}]}]"#,
        );
        assert_eq!(range_case.field_offset(0, 0), Some(0));
        assert_eq!(range_case.field_offset(0, 1), Some(16));
        assert_eq!(range_case.field_offset(1, 0), Some(8));
        assert_eq!(range_case.field_offset(1, 1), Some(0));
        assert_eq!(
            range_case.tag(),
            Some(TagLayout {
                offset: 24,
                signed: false,
                bits: 8,
                encoding: TagEncoding::Niche {
                    untagged_variant: 1,
                    niche_variants: 0..=0,
                    niche_start: 2,
                },
            })
        );

        // Generic `Option`: no concrete discriminator.
        let option = layout_from(
            "null",
            r#"[{"field_offsets":[]},{"field_offsets":[{"guarantee":{"GuaranteedAlignment":{"Deduplicated":720}},"chosen":null}]}]"#,
        );
        assert!(option.tag().is_none());

        // Untagged variant 7 sits inside 0..=8. The point `Invalid` child
        // is the dead tag `niche_start + 7`; variant 8 is `niche_start + 8`.
        let holed = layout_from(
            r#"{"Branch":{"offset":{"guarantee":null,"chosen":0},"int_ty":{"Unsigned":"U64"},"children":[[{"start":{"Unsigned":["U64","9223372036854775808"]},"end":{"Unsigned":["U64","9223372036854775808"]}},{"Known":0}],[{"start":{"Unsigned":["U64","9223372036854775809"]},"end":{"Unsigned":["U64","9223372036854775809"]}},{"Known":1}],[{"start":{"Unsigned":["U64","9223372036854775810"]},"end":{"Unsigned":["U64","9223372036854775810"]}},{"Known":2}],[{"start":{"Unsigned":["U64","9223372036854775811"]},"end":{"Unsigned":["U64","9223372036854775811"]}},{"Known":3}],[{"start":{"Unsigned":["U64","9223372036854775812"]},"end":{"Unsigned":["U64","9223372036854775812"]}},{"Known":4}],[{"start":{"Unsigned":["U64","9223372036854775813"]},"end":{"Unsigned":["U64","9223372036854775813"]}},{"Known":5}],[{"start":{"Unsigned":["U64","9223372036854775814"]},"end":{"Unsigned":["U64","9223372036854775814"]}},{"Known":6}],[{"start":{"Unsigned":["U64","9223372036854775815"]},"end":{"Unsigned":["U64","9223372036854775815"]}},"Invalid"],[{"start":{"Unsigned":["U64","9223372036854775816"]},"end":{"Unsigned":["U64","9223372036854775816"]}},{"Known":8}],[{"start":{"Unsigned":["U64","9223372036854775817"]},"end":{"Unsigned":["U64","18446744073709551615"]}},"Invalid"]],"fallback":{"Known":7}}}"#,
            r#"[{"field_offsets":[]},{"field_offsets":[]},{"field_offsets":[]},{"field_offsets":[]},{"field_offsets":[]},{"field_offsets":[]},{"field_offsets":[]},{"field_offsets":[]},{"field_offsets":[]}]"#,
        );
        assert_eq!(
            holed.tag(),
            Some(TagLayout {
                offset: 0,
                signed: false,
                bits: 64,
                encoding: TagEncoding::Niche {
                    untagged_variant: 7,
                    niche_variants: 0..=8,
                    niche_start: 9223372036854775808,
                },
            })
        );

        // An uninhabited index inside the span does not pull an untagged
        // variant that sits past the span into `niche_variants`.
        let outside = layout_from(
            r#"{"Branch":{"offset":{"chosen":0},"int_ty":{"Unsigned":"U8"},"children":[[{"start":{"Unsigned":["U8","10"]},"end":{"Unsigned":["U8","10"]}},{"Known":0}],[{"start":{"Unsigned":["U8","12"]},"end":{"Unsigned":["U8","12"]}},{"Known":2}]],"fallback":{"Known":5}}}"#,
            r#"[{"field_offsets":[]},{"field_offsets":[]},{"field_offsets":[]}]"#,
        );
        assert_eq!(
            outside.tag(),
            Some(TagLayout {
                offset: 0,
                signed: false,
                bits: 8,
                encoding: TagEncoding::Niche {
                    untagged_variant: 5,
                    niche_variants: 0..=2,
                    niche_start: 10,
                },
            })
        );
    }

    /// A fieldless enum's variant values are the tags its layout writes;
    /// a struct, or an enum with a field, has none.
    #[test]
    fn fieldless_enum_tags_come_from_the_taggers() {
        let json = r#"{
            "size": 1, "align": 1,
            "discriminator": {"Branch": {"offset": 0, "int_ty": {"Signed": "I8"}}},
            "variant_layouts": [
                {"field_offsets": [], "tagger": [[0, {"Signed": ["I8", "-1"]}]]},
                {"field_offsets": [], "tagger": [[0, {"Signed": ["I8", "4"]}]]}
            ]
        }"#;
        let layout: TypeLayout = serde_json::from_str(json).unwrap();
        assert_eq!(layout.fieldless_enum_tags(), Some(vec![-1, 4]));

        let unit_struct: TypeLayout =
            serde_json::from_str(r#"{"size": 0, "variant_layouts": [{"field_offsets": []}]}"#)
                .unwrap();
        assert_eq!(unit_struct.fieldless_enum_tags(), None);

        let payload = r#"{
            "size": 16,
            "discriminator": {"Branch": {"offset": 0, "int_ty": {"Unsigned": "U8"}}},
            "variant_layouts": [
                {"field_offsets": [], "tagger": [[0, {"Unsigned": ["U8", "0"]}]]},
                {"field_offsets": [8], "tagger": [[0, {"Unsigned": ["U8", "1"]}]]}
            ]
        }"#;
        let payload: TypeLayout = serde_json::from_str(payload).unwrap();
        assert_eq!(payload.fieldless_enum_tags(), None);
    }

    #[test]
    fn enum_discriminant_signedness_and_width_are_preserved() {
        for (int_ty, expected) in [
            (r#"{"Signed":"I16"}"#, Some("i16")),
            (r#"{"Signed":"I32"}"#, Some("i32")),
            (r#"{"Unsigned":"U64"}"#, Some("u64")),
            // Wider than a machine word: no int-bank field type spells it.
            (r#"{"Signed":"I128"}"#, None),
            (r#"{"Unsigned":"U128"}"#, None),
        ] {
            let json = format!(
                r#"{{
                    "discriminator": {{"Branch": {{"offset": 4, "int_ty": {int_ty}}}}},
                    "variant_layouts": [{{"field_offsets": []}}, {{"field_offsets": []}}]
                }}"#
            );
            let layout: TypeLayout = serde_json::from_str(&json).unwrap();
            assert_eq!(layout.discriminant_int_type(), expected);
            // A declined width is still a recorded tag: the two `None`s a
            // consumer sees from `discriminant_int_type` are distinguishable.
            assert!(layout.has_branch_discriminant());
        }
        let no_discriminator: TypeLayout =
            serde_json::from_str(r#"{"variant_layouts": [{"field_offsets": []}]}"#).unwrap();
        assert!(!no_discriminator.has_branch_discriminant());
        assert_eq!(no_discriminator.discriminant_int_type(), None);
    }

    /// Target selection: exact-match wins; a sole entry is chosen even
    /// when the requested target differs; an empty/no-match list is `None`.
    #[test]
    fn target_layout_selection() {
        let two = r#"[
            {"key": "x86_64-unknown-linux-gnu",
             "value": {"variant_layouts": [{"field_offsets": [0]}]}},
            {"key": "aarch64-apple-darwin",
             "value": {"size": 16, "variant_layouts": [{"field_offsets": [8]}]}}
        ]"#;
        let entries: Vec<TargetLayout> = serde_json::from_str(two).unwrap();
        // exact match selects the aarch64 entry (offset 8), not the first
        assert_eq!(
            select_target_layout(entries, "aarch64-apple-darwin")
                .and_then(|l| l.struct_field_offset(0)),
            Some(8)
        );

        let sole = r#"[{"key": "x86_64-unknown-linux-gnu",
            "value": {"variant_layouts": [{"field_offsets": [0]}]}}]"#;
        let entries: Vec<TargetLayout> = serde_json::from_str(sole).unwrap();
        // sole entry chosen despite a non-matching requested target
        assert_eq!(
            select_target_layout(entries, "aarch64-apple-darwin")
                .and_then(|l| l.struct_field_offset(0)),
            Some(0)
        );

        // empty list → None
        assert!(select_target_layout(Vec::new(), "aarch64-apple-darwin").is_none());
    }

    /// A union body keeps its field list. `#[serde(other)]` turns any tag this
    /// enum does not name into `Unknown`, so a missing `Union` arm would lose
    /// the fields silently rather than fail to parse — the shape `IN_ADDR_0`
    /// arrives in through the `windows-sys` dependency closure.
    #[test]
    fn union_decl_kind_keeps_its_fields() {
        let json = r#"{"Union": [
            {"name": "S_un_b", "ty": {"Deduplicated": 170}, "attr_info": null},
            {"name": "S_addr", "ty": {"Deduplicated": 42}, "attr_info": null}
        ]}"#;
        let kind: TypeDeclKind = serde_json::from_str(json).unwrap();
        let TypeDeclKind::Union(fields) = kind else {
            panic!("union body parsed as {kind:?}");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name.as_deref(), Some("S_un_b"));
        assert_eq!(fields[1].ty.label(), "ty#42");
    }

    fn span(id: u64) -> Value {
        serde_json::json!({"Deduplicated": id})
    }

    fn fun_decl_from_blocks(blocks: Vec<Value>) -> FunDecl {
        let json = serde_json::json!({
            "def_id": 0,
            "item_meta": {
                "name": [{"Ident": ["f", 0]}],
                "span": span(0),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "signature": {"is_unsafe": false, "inputs": [], "output": {"Deduplicated": 0}},
            "body": {"Unstructured": {
                "span": span(0),
                "locals": {"arg_count": 0, "locals": []},
                "body": blocks
            }}
        });
        serde_json::from_str(&json.to_string()).expect("FunDecl fixture parses")
    }

    fn bb(kind: Value, is_cleanup: bool, term_span: u64) -> Value {
        serde_json::json!({
            "statements": [],
            "terminator": {"span": span(term_span), "kind": kind},
            "is_cleanup": is_cleanup
        })
    }

    fn call_term(target: u64, on_unwind: u64) -> Value {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": 0}, "generics": {}}},
                "args": [],
                "dest": {"kind": {"Local": 0}, "ty": {"Deduplicated": 0}}
            },
            "target": target,
            "on_unwind": on_unwind
        }})
    }

    fn drop_term(target: u64, on_unwind: u64) -> Value {
        serde_json::json!({"Drop": {
            "place": {"kind": {"Local": 0}, "ty": {"Deduplicated": 0}},
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": on_unwind
        }})
    }

    fn panic_term(on_unwind: u64) -> Value {
        serde_json::json!({"Panic": {
            "name": [
                {"Ident": ["core", 0]},
                {"Ident": ["panicking", 0]},
                {"Ident": ["panic_fmt", 0]}
            ],
            "on_unwind": on_unwind
        }})
    }

    fn call_edges(bb: &BasicBlock) -> (u64, u64) {
        let call = bb.terminator.kind_value().get("Call").unwrap();
        (
            call.get("target").and_then(Value::as_u64).unwrap(),
            call.get("on_unwind").and_then(Value::as_u64).unwrap(),
        )
    }

    fn drop_edges(bb: &BasicBlock) -> (u64, u64) {
        let drop = bb.terminator.kind_value().get("Drop").unwrap();
        (
            drop.get("target").and_then(Value::as_u64).unwrap(),
            drop.get("on_unwind").and_then(Value::as_u64).unwrap(),
        )
    }

    #[test]
    fn unstructured_keeps_cleanup_blocks_and_edges() {
        let decl = fun_decl_from_blocks(vec![
            bb(call_term(1, 3), false, 0),
            bb(drop_term(2, 4), false, 0),
            bb(serde_json::json!("Return"), false, 0),
            bb(drop_term(5, 6), true, 7),
            bb(serde_json::json!("UnwindResume"), true, 0),
            bb(serde_json::json!("UnwindResume"), true, 0),
            bb(serde_json::json!("UnwindTerminate"), true, 0),
        ]);
        let body = decl.unstructured().expect("Unstructured body");
        assert_eq!(body.body.len(), 7);
        assert_eq!(call_edges(&body.body[0]), (1, 3));
        assert_eq!(drop_edges(&body.body[1]), (2, 4));
        assert_eq!(
            body.body[2].terminator.kind_value(),
            &serde_json::json!("Return")
        );
        assert!(body.body[3].is_cleanup);
        assert_eq!(drop_edges(&body.body[3]), (5, 6));
        assert!(matches!(
            body.body[3].terminator.span,
            Some(SpanRef::Deduplicated(7))
        ));
        assert!(body.body[4].is_cleanup);
        assert_eq!(
            body.body[4].terminator.kind_value(),
            &serde_json::json!("UnwindResume")
        );
        assert!(body.body[5].is_cleanup);
        assert_eq!(
            body.body[6].terminator.kind_value(),
            &serde_json::json!("UnwindTerminate")
        );
    }

    /// Cleanup blocks that never name a `Global` stay as extracted, and a
    /// later block's promoted read is still spliced. Walking those cleanup
    /// terminators is not required to find the read.
    #[test]
    fn splice_skips_cleanup_blocks_without_a_global_read() {
        let span = serde_json::json!({"Deduplicated": 0});
        let meta = |name: serde_json::Value| {
            serde_json::json!({
                "name": name,
                "span": span,
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": true
            })
        };
        let local = |index: u64| {
            serde_json::json!({
                "index": index,
                "name": null,
                "span": span,
                "ty": {"Deduplicated": 0}
            })
        };
        let bb = |statements: serde_json::Value, kind: serde_json::Value, is_cleanup: bool| {
            serde_json::json!({
                "statements": statements,
                "terminator": {"span": span, "kind": kind},
                "is_cleanup": is_cleanup
            })
        };
        let assign_global = serde_json::json!({
            "span": span,
            "kind": {"Assign": [
                {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}},
                {"Use": [
                    {"Copy": {
                        "kind": {"Global": {"generics": null, "id": 0}},
                        "ty": {"Deduplicated": 0}
                    }},
                    null
                ]}
            ]}
        });
        let init_assign = serde_json::json!({
            "span": span,
            "kind": {"Assign": [
                {"kind": {"Local": 0}, "ty": {"Deduplicated": 0}},
                {"Use": [{"Const": null}, null]}
            ]}
        });
        let fun = |def_id: u64, name: serde_json::Value, body: serde_json::Value| {
            serde_json::json!({
                "def_id": def_id,
                "item_meta": meta(name),
                "signature": {
                    "is_unsafe": false,
                    "inputs": [],
                    "output": {"Deduplicated": 0}
                },
                "body": body
            })
        };
        let subject = serde_json::json!({
            "span": span,
            "locals": {"arg_count": 0, "locals": [local(0), local(1)]},
            "body": [
                bb(serde_json::json!([]), serde_json::json!({"Goto": {"target": 2}}), true),
                bb(serde_json::json!([]), serde_json::json!("UnwindResume"), true),
                bb(serde_json::json!([assign_global]), serde_json::json!("Return"), false),
                bb(serde_json::json!([]), serde_json::json!({"Goto": {"target": 2}}), true)
            ]
        });
        let init = serde_json::json!({
            "span": span,
            "locals": {"arg_count": 0, "locals": [local(0)]},
            "body": [
                bb(serde_json::json!([init_assign]), serde_json::json!("Return"), false)
            ]
        });
        let file = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    fun(0, serde_json::json!([{"Ident": ["subject", 0]}]), serde_json::json!({"Unstructured": subject})),
                    fun(1, serde_json::json!([{"Ident": ["init", 0]}]), serde_json::json!({"Unstructured": init}))
                ],
                "global_decls": [{
                    "def_id": 0,
                    "item_meta": meta(serde_json::json!([{"Builtin": ["PromotedConst", 0]}])),
                    "global_kind": "AnonConst",
                    "value": [{"Call": [{"kind": {"Fun": 1}}]}]
                }]
            }
        });
        let llbc = crate::Llbc::from_slice(file.to_string().as_bytes()).expect("fixture");
        let body = llbc
            .fn_by_id(0)
            .unwrap()
            .unstructured()
            .expect("subject body");
        assert_eq!(body.body.len(), 4);
        assert!(body.body[0].is_cleanup && body.body[1].is_cleanup && body.body[3].is_cleanup);
        assert!(!body.body[2].is_cleanup);
        assert_eq!(
            body.body[0].terminator.kind_value(),
            &serde_json::json!({"Goto": {"target": 2}})
        );
        assert_eq!(
            body.body[1].terminator.kind_value(),
            &serde_json::json!("UnwindResume")
        );
        assert_eq!(
            body.body[3].terminator.kind_value(),
            &serde_json::json!({"Goto": {"target": 2}})
        );
        assert!(
            body.body[2].statements.len() > 1,
            "promoted read should splice initializer statements"
        );
        assert!(
            body.body[2]
                .statements
                .iter()
                .all(|st| !st.kind.get().contains("\"Global\"")),
            "promoted Global read should have been rewritten"
        );
    }

    #[test]
    fn unstructured_body_without_cleanup_blocks_is_unchanged() {
        let decl = fun_decl_from_blocks(vec![
            serde_json::json!({
                "statements": [],
                "terminator": {"span": span(0), "kind": call_term(1, 2)}
            }),
            serde_json::json!({
                "statements": [],
                "terminator": {"span": span(0), "kind": drop_term(2, 2)}
            }),
            serde_json::json!({
                "statements": [],
                "terminator": {"span": span(0), "kind": "Return"}
            }),
        ]);
        let body = decl.unstructured().expect("Unstructured body");
        assert_eq!(body.body.len(), 3);
        assert_eq!(call_edges(&body.body[0]), (1, 2));
        assert_eq!(drop_edges(&body.body[1]), (2, 2));
        assert_eq!(
            body.body[2].terminator.kind_value(),
            &serde_json::json!("Return")
        );
    }

    /// Darwin Charon 0.1.281 emits a pointer-identity Switch arm as a
    /// `Ref` / `DynTrait` place, not a ConstantExpr literal. Keep that
    /// JSON as `SwitchInt` so `lower_niche_option_switch` can close it
    /// as a `ptr_nonzero` If (`rptr.py` `rtype_bool`); decode must not
    /// fail and drop the function.
    #[test]
    fn a_pointer_identity_switch_arm_decodes() {
        let doc = r#"{"charon_version":"t","has_errors":false,
            "translated":{"crate_name":"c","fun_decls":[]}}"#;
        let llbc = crate::Llbc::from_slice(doc.as_bytes()).expect("fixture parses");
        let kind = serde_json::json!({
            "Switch": {
                "data": {
                    "scrutinee": {
                        "Value": {"Copy": {"kind": {"Local": 0}, "ty": {"Deduplicated": 3}}}
                    },
                    "branches": [[
                        {"Ref": [{"Body": 18}, {"DynTrait": {"id": 0}}, "Mut"]},
                        1
                    ]],
                    "fallback": 0
                },
                "branches": [2, 5]
            }
        });
        let term = decode_term_kind(
            &serde_json::value::to_raw_value(&kind).expect("a JSON value serializes"),
            &llbc,
        )
        .expect("Ref arm decodes");
        match term {
            TermKind::Switch {
                targets: SwitchTargets::SwitchInt(_, arms, default),
                ..
            } => {
                assert_eq!(default, 2, "fallback maps through branches[0]");
                assert_eq!(arms.len(), 1);
                assert!(arms[0].0.get("Ref").is_some(), "Ref arm is kept");
                assert_eq!(arms[0].1, 5, "arm target maps through branches[1]");
            }
            other => panic!("expected SwitchInt Ref arm: {other:?}"),
        }
    }

    #[test]
    fn panic_terminator_matches_charon_json() {
        let kind: TermKind = serde_json::from_value(panic_term(5)).expect("Panic JSON decodes");
        match kind {
            TermKind::Panic { name, on_unwind } => {
                assert_eq!(on_unwind, 5);
                assert_eq!(
                    name,
                    serde_json::json!([
                        {"Ident": ["core", 0]},
                        {"Ident": ["panicking", 0]},
                        {"Ident": ["panic_fmt", 0]}
                    ])
                );
            }
            other => panic!("expected Panic, got {other:?}"),
        }
    }

    #[test]
    fn strip_cleanup_blocks_rewrites_panic_unwind_edges() {
        let decl = fun_decl_from_blocks(vec![
            bb(panic_term(2), false, 0),
            bb(serde_json::json!("Return"), false, 0),
            bb(drop_term(3, 3), true, 7),
            bb(serde_json::json!("UnwindResume"), true, 0),
        ]);
        // FunDecl::unstructured keeps cleanup blocks. The public strip
        // still rewrites a Panic on_unwind onto the terminal resume.
        let body = strip_cleanup_blocks(decl.unstructured().expect("Unstructured body"));
        assert_eq!(body.body.len(), 3);
        let panic = body.body[0].terminator.kind_value().get("Panic").unwrap();
        assert_eq!(panic.get("on_unwind").and_then(Value::as_u64).unwrap(), 2);
        assert_eq!(
            body.body[1].terminator.kind_value(),
            &serde_json::json!("Return")
        );
        assert_eq!(
            body.body[2].terminator.kind_value(),
            &serde_json::json!("UnwindResume")
        );
    }
}
