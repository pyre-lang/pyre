//! Instantiated host tag of a niche-encoded enum.
//!
//! Generic LLBC TypeDecls (`Option`, `Cow`) record `discriminator: null`.
//! `Rvalue::Discriminant` of a raw pointer still has the use-site generic
//! arguments, so this module rebuilds rustc's layout (`Niche::reserve`,
//! `TagEncoding::Niche`) from the instantiated field types and returns the
//! `TagLayout` `emit_host_variant_index` already understands.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    TagEncoding, TagLayout, TyRef, TypeDecl, TypeDeclKind, TypeLayout, VariantDecl,
};
use serde_json::Value;

const LAYOUT_DEPTH: u8 = 16;

#[derive(Clone, Debug)]
struct ScalarNiche {
    offset: u64,
    bits: u32,
    signed: bool,
    valid_start: u128,
    valid_end: u128,
}

impl ScalarNiche {
    fn mask(&self) -> u128 {
        mask_bits(self.bits)
    }

    fn available(&self) -> u128 {
        self.valid_start
            .wrapping_sub(self.valid_end)
            .wrapping_sub(1)
            & self.mask()
    }

    /// Byte width of the niche scalar, `Scalar::size` / `Primitive::size`.
    fn size_bytes(&self) -> u64 {
        u64::from(self.bits.div_ceil(8))
    }

    /// `rustc_abi::Niche::reserve`.
    fn reserve(&self, count: u128) -> Option<(u128, ScalarNiche)> {
        if count == 0 {
            return None;
        }
        let max_value = self.mask();
        let available = self.available();
        if count > available {
            return None;
        }
        let v_start = self.valid_start;
        let v_end = self.valid_end;
        let move_start = |start: u128, end: u128| {
            let new_start = start.wrapping_sub(count) & max_value;
            (
                new_start,
                ScalarNiche {
                    offset: self.offset,
                    bits: self.bits,
                    signed: self.signed,
                    valid_start: new_start,
                    valid_end: end,
                },
            )
        };
        let move_end = |start: u128, end: u128| {
            let niche_start = end.wrapping_add(1) & max_value;
            let new_end = end.wrapping_add(count) & max_value;
            (
                niche_start,
                ScalarNiche {
                    offset: self.offset,
                    bits: self.bits,
                    signed: self.signed,
                    valid_start: start,
                    valid_end: new_end,
                },
            )
        };
        let is_bool = v_start == 0 && v_end == 1;
        let chosen = if count == 1 && !is_bool {
            let next_up = sign_extend_abs(self.bits, v_end.wrapping_add(1) & max_value);
            let next_down = sign_extend_abs(self.bits, v_start.wrapping_sub(1) & max_value);
            if next_down <= next_up {
                move_start(v_start, v_end)
            } else {
                move_end(v_start, v_end)
            }
        } else if v_start > v_end {
            move_end(v_start, v_end)
        } else {
            let distance_end_zero = max_value - v_end;
            if v_start <= distance_end_zero {
                if count <= v_start {
                    move_start(v_start, v_end)
                } else {
                    move_end(v_start, v_end)
                }
            } else {
                let end = v_end.wrapping_add(count) & max_value;
                let overshot_zero = (1..=v_end).contains(&end);
                if overshot_zero {
                    move_start(v_start, v_end)
                } else {
                    move_end(v_start, v_end)
                }
            }
        };
        Some(chosen)
    }
}

#[derive(Clone, Debug)]
struct LayoutInfo {
    size: u64,
    align: u64,
    tag: Option<TagLayout>,
    largest_niche: Option<ScalarNiche>,
}

fn mask_bits(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

fn sign_extend_abs(bits: u32, value: u128) -> u128 {
    if bits == 0 {
        return 0;
    }
    let v = value & mask_bits(bits);
    let shift = 128 - bits;
    let signed = ((v << shift) as i128) >> shift;
    signed.unsigned_abs()
}

fn target_triple() -> String {
    std::env::var("TARGET").unwrap_or_default()
}

fn word_bytes() -> u64 {
    crate::layout::target_word_size() as u64
}

fn align_up(offset: u64, align: u64) -> Option<u64> {
    super::align_up_bytes(offset, align)
}

/// Host `TagLayout` of `ty` when the instantiated layout is a niche.
///
/// `Ok(None)` is a Direct tag (or a layout with no branch): the Discriminant
/// lowering keeps the explicit-shell `__discriminant` FieldRead.
/// `Err` means the instantiated layout could not be determined.
pub(in crate::front::mir) fn instantiated_enum_host_tag(
    llbc: &Llbc,
    ty: &TyRef,
) -> Result<Option<TagLayout>, String> {
    let json = super::tyref_json(ty);
    // A generic template still names TypeVars. Its Discriminant is the
    // explicit sum shell, not a host niche that depends on the parameters.
    if value_has_typevar(llbc, &json, 0) {
        return Ok(None);
    }
    let layout = layout_of_ty(llbc, &json, &[], 0)?;
    match layout.tag {
        Some(
            tag @ TagLayout {
                encoding: TagEncoding::Niche { .. },
                ..
            },
        ) => Ok(Some(tag)),
        Some(TagLayout {
            encoding: TagEncoding::Direct { .. },
            ..
        })
        | None => Ok(None),
    }
}

fn layout_of_ty(llbc: &Llbc, ty: &Value, subst: &[Value], depth: u8) -> Result<LayoutInfo, String> {
    if depth > LAYOUT_DEPTH {
        return Err("layout recursion limit".into());
    }
    let substituted = subst_ty(llbc, ty, subst);
    let body = super::tyexpr_body(llbc, &substituted).unwrap_or(&substituted);
    if super::typevar_slot(body).is_some() {
        return Err("unsubstituted type parameter in host layout".into());
    }
    if let Some(lit) = body.get("Scalar") {
        return scalar_layout(lit);
    }
    if body.get("Never").is_some() {
        return Ok(LayoutInfo {
            size: 0,
            align: 1,
            tag: None,
            largest_niche: None,
        });
    }
    if body.get("FnPtr").is_some() {
        return thin_nonnull_ptr_layout();
    }
    if let Some(arr) = body.get("Array").and_then(Value::as_array) {
        let elem = arr
            .first()
            .ok_or_else(|| "Array without element type".to_string())?;
        let elem_layout = layout_of_ty(llbc, elem, &[], depth + 1)?;
        let len: u64 = super::charon_array_len_to_string(
            arr.get(1)
                .ok_or_else(|| "Array without length".to_string())?,
            llbc,
        )
        .parse()
        .map_err(|_| "Array length is not an integer".to_string())?;
        let size = elem_layout
            .size
            .checked_mul(len)
            .ok_or_else(|| "Array size overflow".to_string())?;
        return Ok(LayoutInfo {
            size,
            align: elem_layout.align.max(1),
            tag: None,
            largest_niche: elem_layout.largest_niche,
        });
    }
    if body.get("Slice").is_some() {
        return Err("unsized slice value has no host layout".into());
    }
    if body.get("Dyn").is_some() || body.get("TraitObject").is_some() {
        return Err("unsized dyn value has no host layout".into());
    }
    if let Some(arr) = body.get("Ref").and_then(Value::as_array) {
        let pointee = arr
            .get(1)
            .ok_or_else(|| "Ref without pointee".to_string())?;
        return ptr_layout(llbc, pointee, true);
    }
    if let Some(arr) = body.get("RawPtr").and_then(Value::as_array) {
        let pointee = arr
            .first()
            .ok_or_else(|| "RawPtr without pointee".to_string())?;
        return ptr_layout(llbc, pointee, false);
    }
    if let Some(arr) = body.get("TraitType").and_then(Value::as_array) {
        let traitref = arr
            .first()
            .ok_or_else(|| "TraitType without trait ref".to_string())?;
        let assoc = arr
            .get(1)
            .ok_or_else(|| "TraitType without assoc name".to_string())?;
        let resolved = super::resolve_trait_assoc_type_value(traitref, assoc, llbc)
            .ok_or_else(|| "unresolved TraitType projection in host layout".to_string())?;
        return layout_of_ty(llbc, resolved, &[], depth + 1);
    }
    let adt = body.get("Adt").unwrap_or(body);
    if let Some(adt_obj) = adt.as_object() {
        return adt_layout(llbc, adt_obj, subst, depth);
    }
    Err(format!(
        "no host layout for type {}",
        body.as_object()
            .and_then(|m| m.keys().next().cloned())
            .unwrap_or_else(|| "<?>".into())
    ))
}

fn value_has_typevar(llbc: &Llbc, v: &Value, depth: u8) -> bool {
    if depth > 32 {
        return false;
    }
    let peeled = super::tyexpr_body(llbc, v).unwrap_or(v);
    if super::typevar_slot(peeled).is_some() {
        return true;
    }
    match peeled {
        Value::Array(arr) => arr
            .iter()
            .any(|item| value_has_typevar(llbc, item, depth + 1)),
        Value::Object(map) => map
            .values()
            .any(|val| value_has_typevar(llbc, val, depth + 1)),
        _ => false,
    }
}

fn subst_ty(llbc: &Llbc, ty: &Value, subst: &[Value]) -> Value {
    subst_value(llbc, ty, subst, 0)
}

fn subst_value(llbc: &Llbc, v: &Value, subst: &[Value], depth: u8) -> Value {
    if depth > 32 {
        return v.clone();
    }
    let peeled = super::tyexpr_body(llbc, v).unwrap_or(v);
    if let Some(slot) = super::typevar_slot(peeled) {
        if slot == usize::MAX {
            return peeled.clone();
        }
        return subst.get(slot).cloned().unwrap_or_else(|| peeled.clone());
    }
    match peeled {
        Value::Array(arr) => Value::Array(
            arr.iter()
                .map(|item| subst_value(llbc, item, subst, depth + 1))
                .collect(),
        ),
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, val) in map {
                out.insert(key.clone(), subst_value(llbc, val, subst, depth + 1));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

fn scalar_layout(lit: &Value) -> Result<LayoutInfo, String> {
    if matches!(lit.as_str(), Some("Bool")) || lit.get("Bool").is_some() {
        return Ok(LayoutInfo {
            size: 1,
            align: 1,
            tag: None,
            largest_niche: Some(ScalarNiche {
                offset: 0,
                bits: 8,
                signed: false,
                valid_start: 0,
                valid_end: 1,
            }),
        });
    }
    if matches!(lit.as_str(), Some("Char")) || lit.get("Char").is_some() {
        return Ok(LayoutInfo {
            size: 4,
            align: 4,
            tag: None,
            largest_niche: Some(ScalarNiche {
                offset: 0,
                bits: 32,
                signed: false,
                valid_start: 0,
                valid_end: 0x10_ffff,
            }),
        });
    }
    let size = super::primitive_size_align(false, lit)
        .ok_or_else(|| "unknown scalar in host layout".to_string())?;
    let align = super::primitive_size_align(true, lit).unwrap_or(size.max(1));
    Ok(LayoutInfo {
        size,
        align,
        tag: None,
        largest_niche: None,
    })
}

fn thin_nonnull_ptr_layout() -> Result<LayoutInfo, String> {
    let word = word_bytes();
    let bits = (word * 8) as u32;
    Ok(LayoutInfo {
        size: word,
        align: word,
        tag: None,
        largest_niche: Some(ScalarNiche {
            offset: 0,
            bits,
            signed: false,
            valid_start: 1,
            valid_end: mask_bits(bits),
        }),
    })
}

fn ptr_layout(llbc: &Llbc, pointee: &Value, is_ref: bool) -> Result<LayoutInfo, String> {
    let word = word_bytes();
    let fat = pointee_is_unsized(llbc, pointee);
    let size = if fat { word * 2 } else { word };
    let niche = if is_ref {
        let bits = (word * 8) as u32;
        Some(ScalarNiche {
            offset: 0,
            bits,
            signed: false,
            valid_start: 1,
            valid_end: mask_bits(bits),
        })
    } else {
        None
    };
    Ok(LayoutInfo {
        size,
        align: word,
        tag: None,
        largest_niche: niche,
    })
}

fn pointee_is_unsized(llbc: &Llbc, pointee: &Value) -> bool {
    let Some(body) = super::tyexpr_body(llbc, pointee) else {
        return false;
    };
    if body.get("Slice").is_some() || body.get("Dyn").is_some() || body.get("TraitObject").is_some()
    {
        return true;
    }
    let adt = body.get("Adt").and_then(Value::as_object);
    if let Some(adt) = adt {
        if adt.get("builtin").and_then(Value::as_str) == Some("Str") {
            return true;
        }
        if let Some(id) = adt.get("id").and_then(Value::as_u64)
            && let Some(td) = llbc.type_by_id(id)
            && ptr_metadata_is_fat(&td.ptr_metadata)
        {
            return true;
        }
    }
    !super::json_ty_is_statically_sized(pointee, llbc)
}

fn ptr_metadata_is_fat(meta: &Value) -> bool {
    if meta.is_null() {
        return false;
    }
    if let Some(s) = meta.as_str() {
        return s != "None";
    }
    meta.get("VTable").is_some()
        || meta.get("InheritFrom").is_some()
        || meta.get("Length").is_some()
}

fn adt_layout(
    llbc: &Llbc,
    adt: &serde_json::Map<String, Value>,
    outer_subst: &[Value],
    depth: u8,
) -> Result<LayoutInfo, String> {
    let builtin = adt.get("builtin").and_then(Value::as_str);
    let args = adt_type_args(llbc, adt, outer_subst);
    if builtin == Some("Tuple") {
        return struct_from_field_types(llbc, &args, None, depth);
    }
    if builtin == Some("Str") {
        return Err("unsized str value has no host layout".into());
    }
    let id = adt
        .get("id")
        .and_then(Value::as_u64)
        .ok_or_else(|| "ADT without def_id in host layout".to_string())?;
    let td = llbc
        .type_by_id(id)
        .ok_or_else(|| format!("missing TypeDecl {id} in host layout"))?;
    match &td.kind {
        TypeDeclKind::Alias(aliased) => {
            let inner = aliased.get("aliased_ty").unwrap_or(aliased);
            layout_of_ty(llbc, inner, &args, depth + 1)
        }
        TypeDeclKind::Opaque => opaque_layout(td, llbc),
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => {
            let field_tys: Vec<Value> = fields
                .iter()
                .map(|field| subst_ty(llbc, &super::tyref_json(&field.ty), &args))
                .collect();
            let charon = td.layout_for_target(llbc, &target_triple());
            if matches!(td.kind, TypeDeclKind::Union(_)) {
                union_from_field_types(llbc, &field_tys, charon.as_ref(), depth)
            } else {
                struct_from_field_types(llbc, &field_tys, charon.as_ref(), depth)
            }
        }
        TypeDeclKind::Enum(variants) => enum_layout(llbc, td, variants, &args, depth),
        TypeDeclKind::Unknown => Err(format!(
            "unknown TypeDeclKind for {}",
            td.item_meta.name_path()
        )),
    }
}

fn adt_type_args(
    llbc: &Llbc,
    adt: &serde_json::Map<String, Value>,
    outer_subst: &[Value],
) -> Vec<Value> {
    super::type_decl_ref_generics(adt, llbc)
        .and_then(|generics| generics.get("types"))
        .and_then(Value::as_array)
        .map(|types| {
            types
                .iter()
                .map(|ty| subst_ty(llbc, ty, outer_subst))
                .collect()
        })
        .unwrap_or_default()
}

/// Pointer widths Charon records for a RawVec-shaped Opaque header.
/// Extraction is 4 or 8; the TARGET word may differ (wasm32 prepass
/// still runs on a 64-bit host and Charon then writes 24/8).
const RECORDED_WORD_WIDTHS: [u64; 2] = [4, 8];

/// Opaque univariant, size `3*recorded_word`, align `recorded_word`, a
/// single field at byte 0. That is how Charon spells a RawVec-shaped
/// header whose field types were not copied (`String`, `Vec<T>`, and
/// transparent wrappers). The recorded word is the extraction pointer
/// width, which may differ from [`word_bytes`].
fn three_word_field0_header(layout: &TypeLayout) -> bool {
    recorded_three_word_header(
        layout.size.unwrap_or(0),
        layout.align.unwrap_or(1),
        layout
            .variant_layouts
            .first()
            .map(|variant| variant.field_offsets.as_slice())
            .unwrap_or(&[]),
    )
}

fn recorded_three_word_header(size: u64, align: u64, field_offsets: &[u64]) -> bool {
    RECORDED_WORD_WIDTHS.contains(&align)
        && size == align.saturating_mul(3)
        && field_offsets == [0u64]
}

/// Same identity `tyref_is_vec_value` / `tyref_ptr_pointee_is_vec` use to
/// pick `RustVecRepr`. Charon's generic `Vec` decl is Opaque with
/// `layout: []`; the header offsets are [`crate::vec_layout::probe`].
fn type_decl_is_alloc_vec(td: &TypeDecl) -> bool {
    td.item_meta.name_path() == "alloc::vec::Vec"
}

/// TARGET-width `Vec` header from [`crate::vec_layout::probe`].
/// rustc permutes `RawVecInner.cap` (`UsizeNoHighBit`, valid
/// `0..=isize::MAX`) into the cap word; that word is the Option niche.
fn vec_header_layout() -> LayoutInfo {
    let layout = crate::vec_layout::probe();
    let word = word_bytes();
    let bits = (word * 8) as u32;
    LayoutInfo {
        size: word.saturating_mul(3),
        align: word,
        tag: None,
        largest_niche: Some(ScalarNiche {
            offset: layout.cap_offset as u64,
            bits,
            signed: false,
            valid_start: 0,
            valid_end: (1u128 << (bits - 1)) - 1,
        }),
    }
}

/// TARGET-width layout of a RawVec-shaped three-word header.
/// rustc permutes `RawVecInner.cap` (`UsizeNoHighBit`, valid
/// `0..=isize::MAX`) to offset 0. Niche bits follow the TARGET word,
/// not the extraction word. See `codeobject_layout`: a wasm32 prepass
/// still runs on the host.
fn three_word_target_layout() -> LayoutInfo {
    let word = word_bytes();
    let bits = (word * 8) as u32;
    LayoutInfo {
        size: word.saturating_mul(3),
        align: word,
        tag: None,
        largest_niche: Some(ScalarNiche {
            offset: 0,
            bits,
            signed: false,
            valid_start: 0,
            valid_end: (1u128 << (bits - 1)) - 1,
        }),
    }
}

#[cfg(test)]
thread_local! {
    static OPAQUE_THREE_WORD_HITS: std::cell::RefCell<Vec<String>> =
        std::cell::RefCell::new(Vec::new());
}

#[cfg(test)]
fn record_opaque_three_word_hit(td: &TypeDecl) {
    OPAQUE_THREE_WORD_HITS.with(|hits| {
        hits.borrow_mut().push(td.item_meta.name_path());
    });
}

#[cfg(test)]
fn take_opaque_three_word_hits() -> Vec<String> {
    OPAQUE_THREE_WORD_HITS.with(|hits| hits.replace(Vec::new()))
}

fn opaque_layout(td: &TypeDecl, llbc: &Llbc) -> Result<LayoutInfo, String> {
    if let Some(layout) = td.layout_for_target(llbc, &target_triple()) {
        let size = layout.size.ok_or_else(|| {
            format!(
                "opaque type {} has no resolved size",
                td.item_meta.name_path()
            )
        })?;
        let align = layout.align.unwrap_or(1);
        if let Some(tag) = layout.tag() {
            return Ok(LayoutInfo {
                size,
                align,
                largest_niche: remaining_niche_from_charon(&layout),
                tag: Some(tag),
            });
        }
        // rustc permutes `RawVecInner.cap` (`UsizeNoHighBit`, valid
        // `0..=isize::MAX`) to offset 0 of a three-word header. Charon records
        // that header as Opaque with size `3*recorded_word`, align
        // `recorded_word`, and one field at 0. Opaque decls have no field
        // types, so this is the layout identity — not a name match. Emit the
        // TARGET word, not the extraction word: a wasm32 prepass still runs
        // on the host and would otherwise keep 24/8. See RB101 Opaque niche
        // census and `codeobject_layout`.
        if three_word_field0_header(&layout) {
            #[cfg(test)]
            record_opaque_three_word_hit(td);
            return Ok(three_word_target_layout());
        }
        if size == 0 {
            return Ok(LayoutInfo {
                size: 0,
                align,
                tag: None,
                largest_niche: None,
            });
        }
        return Err(format!(
            "opaque type {} size={size} align={align} has no recovered scalar niche",
            td.item_meta.name_path()
        ));
    }
    // Generic Opaque: Charon writes `layout: []`. Production `Vec` is
    // this shape; `tyref_is_vec_value` already identifies it by
    // `item_meta.name_path() == "alloc::vec::Vec"` and `RustVecRepr`
    // uses `vec_layout::probe()` for the header. Recover that same
    // layout so a Discriminant of `Option<Vec<_>>` sees the cap niche.
    if type_decl_is_alloc_vec(td) {
        #[cfg(test)]
        record_opaque_three_word_hit(td);
        return Ok(vec_header_layout());
    }
    Err(format!(
        "opaque type {} has no resolved layout",
        td.item_meta.name_path()
    ))
}

fn struct_from_field_types(
    llbc: &Llbc,
    field_tys: &[Value],
    charon: Option<&TypeLayout>,
    depth: u8,
) -> Result<LayoutInfo, String> {
    let mut fields = Vec::with_capacity(field_tys.len());
    for ty in field_tys {
        fields.push(layout_of_ty(llbc, ty, &[], depth + 1)?);
    }
    if fields.is_empty() {
        let size = charon.and_then(|layout| layout.size).unwrap_or(0);
        let align = charon.and_then(|layout| layout.align).unwrap_or(1);
        return Ok(LayoutInfo {
            size,
            align: align.max(1),
            tag: None,
            largest_niche: None,
        });
    }
    let (offsets, from_charon) = match charon.and_then(|layout| layout.variant_layouts.first()) {
        Some(row) if row.field_offsets.len() == fields.len() => (row.field_offsets.clone(), true),
        _ if fields.len() == 1 => (vec![0], true),
        _ => (sequential_offsets(&fields)?, false),
    };
    let has_niche = fields.iter().any(|field| field.largest_niche.is_some());
    if has_niche && fields.len() > 1 && !from_charon {
        return Err("multi-field struct with a scalar niche but no Charon field offsets".into());
    }
    let mut best: Option<ScalarNiche> = None;
    for (field, offset) in fields.iter().zip(&offsets) {
        if let Some(mut niche) = field.largest_niche.clone() {
            niche.offset = niche
                .offset
                .checked_add(*offset)
                .ok_or_else(|| "struct niche offset overflow".to_string())?;
            if best
                .as_ref()
                .is_none_or(|cur| niche.available() > cur.available())
            {
                best = Some(niche);
            }
        }
    }
    let (seq_size, seq_align) = sequential_size_align(&fields)?;
    let size = charon.and_then(|layout| layout.size).unwrap_or(seq_size);
    let align = charon
        .and_then(|layout| layout.align)
        .unwrap_or(seq_align)
        .max(1);
    Ok(LayoutInfo {
        size,
        align,
        tag: None,
        largest_niche: best,
    })
}

fn union_from_field_types(
    llbc: &Llbc,
    field_tys: &[Value],
    charon: Option<&TypeLayout>,
    depth: u8,
) -> Result<LayoutInfo, String> {
    let mut size = 0u64;
    let mut align = 1u64;
    for ty in field_tys {
        let field = layout_of_ty(llbc, ty, &[], depth + 1)?;
        size = size.max(field.size);
        align = align.max(field.align.max(1));
    }
    let size = charon
        .and_then(|layout| layout.size)
        .or_else(|| align_up(size, align))
        .ok_or_else(|| "union size overflow".to_string())?;
    let align = charon
        .and_then(|layout| layout.align)
        .unwrap_or(align)
        .max(1);
    Ok(LayoutInfo {
        size,
        align,
        tag: None,
        // rustc_abi LayoutCalculator::layout_of_union: largest_niche is None.
        largest_niche: None,
    })
}

fn sequential_offsets(fields: &[LayoutInfo]) -> Result<Vec<u64>, String> {
    let mut offset = 0u64;
    let mut out = Vec::with_capacity(fields.len());
    for field in fields {
        let align = field.align.max(1);
        offset = align_up(offset, align).ok_or_else(|| "struct offset overflow".to_string())?;
        out.push(offset);
        offset = offset
            .checked_add(field.size)
            .ok_or_else(|| "struct size overflow".to_string())?;
    }
    Ok(out)
}

fn sequential_size_align(fields: &[LayoutInfo]) -> Result<(u64, u64), String> {
    let mut offset = 0u64;
    let mut align = 1u64;
    for field in fields {
        let fa = field.align.max(1);
        offset = align_up(offset, fa).ok_or_else(|| "struct offset overflow".to_string())?;
        offset = offset
            .checked_add(field.size)
            .ok_or_else(|| "struct size overflow".to_string())?;
        align = align.max(fa);
    }
    Ok((align_up(offset, align).unwrap_or(offset), align))
}

fn enum_layout(
    llbc: &Llbc,
    td: &TypeDecl,
    variants: &[VariantDecl],
    args: &[Value],
    depth: u8,
) -> Result<LayoutInfo, String> {
    let charon = td.layout_for_target(llbc, &target_triple());
    if let Some(layout) = charon.as_ref()
        && let Some(tag) = layout.tag()
    {
        let size = layout
            .size
            .ok_or_else(|| format!("enum {} Charon tag has no size", td.item_meta.name_path()))?;
        let align = layout.align.unwrap_or(1).max(1);
        return Ok(LayoutInfo {
            size,
            align,
            tag: Some(tag),
            largest_niche: remaining_niche_from_charon(layout),
        });
    }
    if variants.is_empty() {
        return Err(format!("enum {} has no variants", td.item_meta.name_path()));
    }
    let mut variant_layouts = Vec::with_capacity(variants.len());
    for (index, variant) in variants.iter().enumerate() {
        let field_tys: Vec<Value> = variant
            .fields
            .iter()
            .map(|field| subst_ty(llbc, &super::tyref_json(&field.ty), args))
            .collect();
        let variant_charon = charon.as_ref().and_then(|layout| {
            layout.variant_layouts.get(index).map(|row| {
                let mut one = layout.clone();
                one.variant_layouts = vec![row.clone()];
                one.discriminator = None;
                one
            })
        });
        variant_layouts.push(struct_from_field_types(
            llbc,
            &field_tys,
            variant_charon.as_ref(),
            depth,
        )?);
    }
    if variant_layouts.len() == 1 {
        return Ok(variant_layouts.remove(0));
    }
    fill_enum_niche(td, variant_layouts)
}

fn fill_enum_niche(td: &TypeDecl, variants: Vec<LayoutInfo>) -> Result<LayoutInfo, String> {
    let n = variants.len();
    let mut size = 0u64;
    let mut align = 1u64;
    for variant in &variants {
        size = size.max(variant.size);
        align = align.max(variant.align.max(1));
    }
    size = align_up(size, align).unwrap_or(size);
    let needed = (n as u128).saturating_sub(1);
    if needed == 0 {
        return Ok(variants.into_iter().next().unwrap());
    }
    let no_niche = LayoutInfo {
        size,
        align,
        tag: None,
        largest_niche: None,
    };
    // rustc_abi LayoutCalculator::layout_of_enum: largest_variant_index is
    // max size, last on a tie (`Iterator::max_by_key`).
    let largest_variant_index = variants
        .iter()
        .enumerate()
        .max_by_key(|(_index, variant)| variant.size)
        .map(|(index, _variant)| index)
        .unwrap();
    let Some(niche) = variants[largest_variant_index].largest_niche.clone() else {
        return Ok(no_niche);
    };
    if niche.available() < needed {
        return Ok(no_niche.clone());
    }
    let niche_variants = niche_variant_range(largest_variant_index, n).ok_or_else(|| {
        format!(
            "enum {} untagged variant {largest_variant_index} is not an edge of {n} variants",
            td.item_meta.name_path()
        )
    })?;
    let Some((niche_start, remaining)) = niche.reserve(needed) else {
        return Ok(no_niche.clone());
    };
    let niche_size_bytes = niche.size_bytes();
    let largest_size = variants[largest_variant_index].size;
    let others_fit = variants.iter().enumerate().all(|(index, variant)| {
        if index == largest_variant_index {
            return true;
        }
        if variant.size <= niche.offset {
            return true;
        }
        let Some(niche_end) = niche.offset.checked_add(niche_size_bytes) else {
            return false;
        };
        let Some(after) = align_up(niche_end, variant.align.max(1)) else {
            return false;
        };
        after.saturating_add(variant.size) <= largest_size
    });
    if !others_fit {
        return Ok(no_niche);
    }
    let largest_niche = (remaining.available() > 0).then_some(remaining.clone());
    Ok(LayoutInfo {
        size,
        align,
        tag: Some(TagLayout {
            offset: remaining.offset,
            signed: remaining.signed,
            bits: remaining.bits,
            encoding: TagEncoding::Niche {
                untagged_variant: largest_variant_index,
                niche_variants,
                niche_start,
            },
        }),
        largest_niche,
    })
}

fn niche_variant_range(untagged: usize, n: usize) -> Option<std::ops::RangeInclusive<usize>> {
    if n < 2 {
        return None;
    }
    if untagged == 0 {
        Some(1..=(n - 1))
    } else if untagged == n - 1 {
        Some(0..=(n - 2))
    } else {
        None
    }
}

fn remaining_niche_from_charon(layout: &TypeLayout) -> Option<ScalarNiche> {
    let branch = layout.discriminator.as_ref()?.get("Branch")?;
    let offset = layout.discriminant_offset()?;
    let (signed, bits) = branch_int_ty(branch.get("int_ty")?)?;
    let children = branch.get("children").and_then(Value::as_array)?;
    let mut invalid: Vec<(u128, u128)> = Vec::new();
    for child in children {
        let pair = child.as_array()?;
        let range = pair.first()?;
        let start = scalar_bits(range.get("start")?)?;
        let end = scalar_bits(range.get("end")?)?;
        if pair.get(1).and_then(Value::as_str) == Some("Invalid") {
            invalid.push((start, end));
        }
    }
    if invalid.len() != 1 {
        return None;
    }
    let (inv_start, inv_end) = invalid[0];
    let max_value = mask_bits(bits);
    let valid_start = inv_end.wrapping_add(1) & max_value;
    let valid_end = inv_start.wrapping_sub(1) & max_value;
    let niche = ScalarNiche {
        offset,
        bits,
        signed,
        valid_start,
        valid_end,
    };
    (niche.available() > 0).then_some(niche)
}

fn branch_int_ty(int_ty: &Value) -> Option<(bool, u32)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::front::host_layout::host_variant_index;
    use crate::front::mir::tests::{
        adt_node, fixture_item_meta, ident_path, llbc_with_types, option_adt_node, option_enum,
    };

    fn word() -> u64 {
        crate::layout::target_word_size() as u64
    }

    fn empty_generics() -> Value {
        serde_json::json!({
            "regions": [],
            "types": [],
            "const_generics": [],
            "trait_refs": []
        })
    }

    fn cow_shaped_decls() -> Vec<Value> {
        let w = word();
        let str_adt = serde_json::json!({
            "Adt": {"id": 3, "generics": empty_generics()}
        });
        let string_adt = serde_json::json!({
            "Adt": {"id": 2, "generics": empty_generics()}
        });
        let cow = serde_json::json!({
            "def_id": 1,
            "item_meta": fixture_item_meta(ident_path(&["alloc", "borrow", "Cow"])),
            "kind": {"Enum": [
                {
                    "name": "Borrowed",
                    "fields": [{
                        "name": null,
                        "is_positional": true,
                        "ty": {"Ref": ["Erased", str_adt, "Shared"]}
                    }],
                    "discriminant": {"Scalar": {"Signed": ["Isize", "0"]}}
                },
                {
                    "name": "Owned",
                    "fields": [{
                        "name": null,
                        "is_positional": true,
                        "ty": string_adt
                    }],
                    "discriminant": {"Scalar": {"Signed": ["Isize", "1"]}}
                }
            ]}
        });
        let string_decl = serde_json::json!({
            "def_id": 2,
            "item_meta": fixture_item_meta(ident_path(&["alloc", "string", "String"])),
            "kind": "Opaque",
            "layout": [{
                "key": "host",
                "value": {
                    "size": w * 3,
                    "align": w,
                    "discriminator": {"Known": 0},
                    "variant_layouts": [{"field_offsets": [0]}],
                    "repr": {"transparent": false}
                }
            }]
        });
        let str_decl = serde_json::json!({
            "def_id": 3,
            "item_meta": fixture_item_meta(ident_path(&["core", "str", "str"])),
            "kind": "Opaque",
            "ptr_metadata": "Length"
        });
        vec![option_enum(0), cow, string_decl, str_decl]
    }

    fn option_cow_ty() -> TyRef {
        TyRef::Other(option_adt_node(0, adt_node(1)))
    }

    #[test]
    fn option_cow_shaped_host_tag_is_logical_index() {
        let llbc = llbc_with_types("fixture", cow_shaped_decls(), vec![]);
        let tag = instantiated_enum_host_tag(&llbc, &option_cow_ty())
            .expect("Option of Cow-shaped enum has a determined layout")
            .expect("that layout is a niche");
        let bits = (word() * 8) as u32;
        let max = mask_bits(bits);
        assert_eq!(tag.offset, 0);
        assert_eq!(tag.bits, bits);
        match &tag.encoding {
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                assert_eq!(*untagged_variant, 1);
                assert_eq!(*niche_variants, 0..=0);
                assert_eq!(*niche_start, max.wrapping_sub(1));
            }
            other => panic!("expected Niche, got {other:?}"),
        }
        assert_eq!(host_variant_index(&tag, max.wrapping_sub(1)), 0);
        assert_eq!(host_variant_index(&tag, max), 1);
        for cap in [0u128, 1, 2, 4] {
            assert_eq!(host_variant_index(&tag, cap), 1, "Owned cap {cap}");
        }
    }

    #[test]
    fn option_i64_computed_layout_is_direct() {
        let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
        let ty = TyRef::Other(option_adt_node(
            0,
            serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}}),
        ));
        let tag = instantiated_enum_host_tag(&llbc, &ty).expect("Option<i64> layout is determined");
        assert!(tag.is_none(), "Option<i64> stays Direct, got {tag:?}");
    }

    #[test]
    fn option_char_computed_layout_is_unicode_niche() {
        let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
        let ty = TyRef::Other(option_adt_node(0, serde_json::json!({"Scalar": "Char"})));
        let tag = instantiated_enum_host_tag(&llbc, &ty)
            .expect("Option<char> layout is determined")
            .expect("Option<char> is a niche");
        assert_eq!(tag.bits, 32);
        match &tag.encoding {
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                assert_eq!(*untagged_variant, 1);
                assert_eq!(*niche_variants, 0..=0);
                assert_eq!(*niche_start, u128::from(u32::MAX));
            }
            other => panic!("expected Niche, got {other:?}"),
        }
        assert_eq!(host_variant_index(&tag, u128::from(u32::MAX)), 0);
        assert_eq!(host_variant_index(&tag, 0), 1);
        assert_eq!(host_variant_index(&tag, u128::from(u32::from('a'))), 1);
    }

    fn fixture_enum(def_id: u64, name: &str, variants: Value) -> Value {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": fixture_item_meta(ident_path(&["fixture", name])),
            "kind": {"Enum": variants}
        })
    }

    fn positional_field(ty: Value) -> Value {
        serde_json::json!({
            "name": null,
            "is_positional": true,
            "ty": ty
        })
    }

    fn enum_variant(name: &str, fields: Vec<Value>, disc: &str) -> Value {
        serde_json::json!({
            "name": name,
            "fields": fields,
            "discriminant": {"Scalar": {"Signed": ["Isize", disc]}}
        })
    }

    fn layout_of_adt(llbc: &Llbc, def_id: u64) -> LayoutInfo {
        let ty = TyRef::Other(adt_node(def_id));
        layout_of_ty(llbc, &super::super::tyref_json(&ty), &[], 0)
            .unwrap_or_else(|err| panic!("layout of fixture adt {def_id}: {err}"))
    }

    #[test]
    fn enum_bool_and_u8_payloads_are_tagged() {
        let decl = fixture_enum(
            0,
            "E",
            serde_json::json!([
                enum_variant(
                    "A",
                    vec![positional_field(serde_json::json!({"Scalar": "Bool"}))],
                    "0"
                ),
                enum_variant(
                    "B",
                    vec![positional_field(
                        serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}})
                    )],
                    "1"
                ),
            ]),
        );
        let llbc = llbc_with_types("fixture", vec![decl], vec![]);
        let ty = TyRef::Other(adt_node(0));
        let tag = instantiated_enum_host_tag(&llbc, &ty)
            .expect("enum E { A(bool), B(u8) } layout is determined");
        assert!(
            tag.is_none(),
            "equal-size bool/u8 payloads stay Direct, got {tag:?}"
        );
        let layout = layout_of_adt(&llbc, 0);
        assert!(layout.tag.is_none(), "got {:?}", layout.tag);
    }

    #[test]
    fn option_bool_shaped_enum_is_niche_encoded() {
        let decl = fixture_enum(
            0,
            "E",
            serde_json::json!([
                enum_variant("None", vec![], "0"),
                enum_variant(
                    "Some",
                    vec![positional_field(serde_json::json!({"Scalar": "Bool"}))],
                    "1"
                ),
            ]),
        );
        let llbc = llbc_with_types("fixture", vec![decl], vec![]);
        let ty = TyRef::Other(adt_node(0));
        let tag = instantiated_enum_host_tag(&llbc, &ty)
            .expect("enum E { None, Some(bool) } layout is determined")
            .expect("Option<bool>-shaped enum is a niche");
        assert_eq!(tag.offset, 0);
        assert_eq!(tag.bits, 8);
        match &tag.encoding {
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                assert_eq!(*untagged_variant, 1);
                assert_eq!(*niche_variants, 0..=0);
                assert_eq!(*niche_start, 2);
            }
            other => panic!("expected Niche, got {other:?}"),
        }
        let none: Option<bool> = None;
        let some_t = Some(true);
        let some_f = Some(false);
        assert_option_samples_match_tag(&tag, &[(none, false), (some_t, true), (some_f, true)]);
    }

    #[test]
    fn union_of_bool_and_u8_has_no_niche() {
        let decl = serde_json::json!({
            "def_id": 0,
            "item_meta": fixture_item_meta(ident_path(&["fixture", "U"])),
            "kind": {"Union": [
                {"name": "b", "ty": {"Scalar": "Bool"}, "attr_info": null},
                {
                    "name": "n",
                    "ty": {"Scalar": {"Integer": {"Unsigned": "U8"}}},
                    "attr_info": null
                }
            ]}
        });
        let llbc = llbc_with_types("fixture", vec![decl], vec![]);
        let layout = layout_of_adt(&llbc, 0);
        assert!(
            layout.largest_niche.is_none(),
            "layout_of_union has no niche, got {:?}",
            layout.largest_niche
        );
        assert!(layout.tag.is_none(), "got {:?}", layout.tag);
    }

    fn opaque_three_word_decl_at(def_id: u64, path: &[&str], recorded_word: u64) -> Value {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": fixture_item_meta(ident_path(path)),
            "kind": "Opaque",
            "layout": [{
                "key": "host",
                "value": {
                    "size": recorded_word * 3,
                    "align": recorded_word,
                    "discriminator": {"Known": 0},
                    "variant_layouts": [{"field_offsets": [0]}],
                    "repr": {"transparent": false}
                }
            }]
        })
    }

    fn opaque_three_word_decl(def_id: u64, path: &[&str]) -> Value {
        opaque_three_word_decl_at(def_id, path, word())
    }

    #[test]
    fn recorded_three_word_header_accepts_pointer_width_4_or_8() {
        assert!(recorded_three_word_header(24, 8, &[0]));
        assert!(recorded_three_word_header(12, 4, &[0]));
        assert!(!recorded_three_word_header(24, 4, &[0]));
        assert!(!recorded_three_word_header(12, 8, &[0]));
        assert!(!recorded_three_word_header(24, 8, &[]));
        assert!(!recorded_three_word_header(24, 8, &[0, 8]));
        assert!(!recorded_three_word_header(16, 8, &[0]));
        assert!(!recorded_three_word_header(24, 16, &[0]));
    }

    #[test]
    fn opaque_three_word_retargets_recorded_width_to_target_word() {
        // 12/4 is what Charon writes extracting on wasm32; 24/8 is the
        // host extraction the wasm32 prepass actually sees. Both emit
        // TARGET word*3 so Option<Cow<str>> keeps a scalar niche.
        for recorded_word in [4u64, 8] {
            let decl = opaque_three_word_decl_at(0, &["alloc", "string", "String"], recorded_word);
            let llbc = llbc_with_types("fixture", vec![decl], vec![]);
            let td = llbc.type_by_id(0).expect("String decl");
            let layout = opaque_layout(td, &llbc).unwrap_or_else(|err| {
                panic!("recorded {recorded_word}-byte word header should retarget: {err}")
            });
            let w = word();
            assert_eq!(layout.size, w * 3, "recorded_word={recorded_word}");
            assert_eq!(layout.align, w, "recorded_word={recorded_word}");
            let niche = layout
                .largest_niche
                .as_ref()
                .expect("cap niche at offset 0");
            assert_eq!(niche.offset, 0, "recorded_word={recorded_word}");
            assert_eq!(niche.bits, (w * 8) as u32, "recorded_word={recorded_word}");
            assert_eq!(
                niche.valid_end,
                (1u128 << (niche.bits - 1)) - 1,
                "recorded_word={recorded_word}"
            );
        }
    }

    fn opaque_empty_decl(def_id: u64, path: &[&str], src: Value) -> Value {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": fixture_item_meta(ident_path(path)),
            "kind": "Opaque",
            "src": src,
            "ptr_metadata": "None",
            "layout": []
        })
    }

    fn adt_node_n_type_args(def_id: u64, n: usize) -> Value {
        let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
        let types: Vec<Value> = (0..n).map(|_| u8_ty.clone()).collect();
        serde_json::json!({
            "Adt": {
                "id": def_id,
                "generics": {
                    "regions": [],
                    "types": types,
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        })
    }

    #[test]
    fn empty_layout_alloc_vec_recovers_probe_header() {
        let decl = opaque_empty_decl(0, &["alloc", "vec", "Vec"], Value::String("Normal".into()));
        let llbc = llbc_with_types("fixture", vec![decl], vec![]);
        let td = llbc.type_by_id(0).expect("Vec decl");
        let layout = opaque_layout(td, &llbc).unwrap_or_else(|err| {
            panic!("empty-layout alloc::vec::Vec should recover vec_layout::probe: {err}")
        });
        let expected = vec_header_layout();
        assert_eq!(layout.size, expected.size);
        assert_eq!(layout.align, expected.align);
        let niche = layout.largest_niche.expect("cap niche");
        let expected_niche = expected.largest_niche.expect("expected cap niche");
        assert_eq!(niche.offset, expected_niche.offset);
        assert_eq!(niche.bits, expected_niche.bits);
        assert_eq!(niche.valid_start, expected_niche.valid_start);
        assert_eq!(niche.valid_end, expected_niche.valid_end);
        assert_eq!(niche.offset, crate::vec_layout::probe().cap_offset as u64);

        let arc_decl =
            opaque_empty_decl(0, &["alloc", "sync", "Arc"], Value::String("Normal".into()));
        let llbc = llbc_with_types("fixture", vec![arc_decl], vec![]);
        let td = llbc.type_by_id(0).expect("Arc decl");
        let arc = opaque_layout(td, &llbc);
        assert!(
            arc.as_ref()
                .is_err_and(|reason| reason.contains("no resolved layout")),
            "empty-layout alloc::sync::Arc must not inherit the Vec cap niche, got {arc:?}"
        );
    }

    #[test]
    fn option_of_empty_layout_alloc_vec_matches_host_vec_niche() {
        let vec_decl =
            opaque_empty_decl(1, &["alloc", "vec", "Vec"], Value::String("Normal".into()));
        let decls = vec![option_enum(0), vec_decl];
        let llbc = llbc_with_types("fixture", decls, vec![]);
        let option_vec = TyRef::Other(option_adt_node(0, adt_node_n_type_args(1, 2)));
        let tag = instantiated_enum_host_tag(&llbc, &option_vec)
            .expect("Option of empty-layout alloc::vec::Vec")
            .expect("that layout is a niche");
        let v_none: Option<Vec<u8>> = None;
        let v_empty = Some(Vec::<u8>::new());
        let v_kw = Some(b"kw".to_vec());
        assert_option_samples_match_tag(&tag, &[(v_none, false), (v_empty, true), (v_kw, true)]);
    }

    #[test]
    fn option_cow_host_extracted_string_header_is_a_niche() {
        // Charon host extraction writes String as 24/8 regardless of
        // TARGET. The wasm32 prepass still has to recover Option<Cow>.
        let mut decls = cow_shaped_decls();
        decls[2] = opaque_three_word_decl_at(2, &["alloc", "string", "String"], 8);
        let llbc = llbc_with_types("fixture", decls, vec![]);
        let tag = instantiated_enum_host_tag(&llbc, &option_cow_ty())
            .expect("Option<Cow<str>> from a host 24/8 String header")
            .expect("that layout is a niche");
        let bits = (word() * 8) as u32;
        assert_eq!(tag.bits, bits);
        assert_eq!(tag.offset, 0);
    }

    fn tag_word<T>(value: &T, offset: u64, bits: u32) -> u128 {
        let nbytes = (bits / 8) as usize;
        assert!(
            offset as usize + nbytes <= std::mem::size_of::<T>(),
            "tag word [{offset}..+{nbytes}] overruns {}",
            std::mem::size_of::<T>()
        );
        let mut buf = [0u8; 16];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (value as *const T as *const u8).add(offset as usize),
                buf.as_mut_ptr(),
                nbytes,
            );
        }
        let mut word = 0u128;
        for (i, b) in buf[..nbytes].iter().enumerate() {
            word |= u128::from(*b) << (8 * i);
        }
        word
    }

    fn assert_option_samples_match_tag<T>(tag: &TagLayout, samples: &[(T, bool)]) {
        for (value, is_some) in samples {
            let raw = tag_word(value, tag.offset, tag.bits);
            let variant = host_variant_index(tag, raw);
            assert_eq!(
                variant != 0,
                *is_some,
                "raw={raw:#x} variant={variant} is_some={is_some} tag={tag:?}"
            );
        }
    }

    #[test]
    fn computed_layout_matches_host_option_niches() {
        use std::borrow::Cow;

        let mut decls = cow_shaped_decls();
        decls.push(opaque_three_word_decl(4, &["alloc", "vec", "Vec"]));
        let llbc = llbc_with_types("fixture", decls, vec![]);

        let option_string = TyRef::Other(option_adt_node(0, adt_node(2)));
        let tag = instantiated_enum_host_tag(&llbc, &option_string)
            .expect("Option<String> layout")
            .expect("Option<String> is a niche");
        let s_none: Option<String> = None;
        let s_empty = Some(String::new());
        let s_k = Some(String::from("k"));
        let s_kw = Some(String::from("kw"));
        let s_rest = Some(String::from("rest"));
        assert_option_samples_match_tag(
            &tag,
            &[
                (s_none, false),
                (s_empty, true),
                (s_k, true),
                (s_kw, true),
                (s_rest, true),
            ],
        );

        let option_vec = TyRef::Other(option_adt_node(0, adt_node(4)));
        let tag = instantiated_enum_host_tag(&llbc, &option_vec)
            .expect("Option<Vec<u8>> layout")
            .expect("Option<Vec<u8>> is a niche");
        let v_none: Option<Vec<u8>> = None;
        let v_empty = Some(Vec::<u8>::new());
        let v_kw = Some(b"kw".to_vec());
        assert_option_samples_match_tag(&tag, &[(v_none, false), (v_empty, true), (v_kw, true)]);

        let option_cow = option_cow_ty();
        let tag = instantiated_enum_host_tag(&llbc, &option_cow)
            .expect("Option<Cow<str>> layout")
            .expect("Option<Cow<str>> is a niche");
        let c_none: Option<Cow<'static, str>> = None;
        let c_borrowed = Some(Cow::Borrowed("kw"));
        let c_owned = Some(Cow::<str>::Owned(String::from("kw")));
        let c_owned_empty = Some(Cow::<str>::Owned(String::new()));
        assert_option_samples_match_tag(
            &tag,
            &[
                (c_none, false),
                (c_borrowed, true),
                (c_owned, true),
                (c_owned_empty, true),
            ],
        );

        let option_option_cow = TyRef::Other(option_adt_node(0, option_adt_node(0, adt_node(1))));
        let tag = instantiated_enum_host_tag(&llbc, &option_option_cow)
            .expect("Option<Option<Cow<str>>> layout")
            .expect("Option<Option<Cow<str>>> is a niche");
        let oo_none: Option<Option<Cow<'static, str>>> = None;
        let oo_some_none = Some(None);
        let oo_borrowed = Some(Some(Cow::Borrowed("kw")));
        let oo_owned = Some(Some(Cow::<str>::Owned(String::from("kw"))));
        assert_option_samples_match_tag(
            &tag,
            &[
                (oo_none, false),
                (oo_some_none, true),
                (oo_borrowed, true),
                (oo_owned, true),
            ],
        );

        let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
        let tuple = serde_json::json!({
            "Adt": {
                "id": 0,
                "builtin": "Tuple",
                "generics": {
                    "regions": [],
                    "types": [u8_ty, adt_node(1)],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        });
        let option_tuple = TyRef::Other(option_adt_node(0, tuple));
        let tuple_layout = instantiated_enum_host_tag(&llbc, &option_tuple);
        assert!(
            tuple_layout
                .as_ref()
                .is_err_and(|reason| reason.contains("no Charon field offsets")),
            "Option<(u8, Cow)> needs Charon field order, got {tuple_layout:?}"
        );
    }

    struct OpaqueHit {
        file: &'static str,
        def_id: u64,
        name: String,
        size: u64,
        align: u64,
        field_offsets: Vec<u64>,
        transparent: bool,
        discriminator: String,
    }

    fn census_opaque_three_word(llbc: &Llbc, file: &'static str) -> Vec<OpaqueHit> {
        let target = super::target_triple();
        let mut hits = Vec::new();
        for td in llbc.iter_type_decls() {
            if !matches!(td.kind, TypeDeclKind::Opaque) {
                continue;
            }
            let Some(layout) = td.layout_for_target(llbc, &target) else {
                continue;
            };
            if layout.tag().is_some() {
                continue;
            }
            if !three_word_field0_header(&layout) {
                continue;
            }
            hits.push(OpaqueHit {
                file,
                def_id: td.def_id,
                name: td.item_meta.name_path(),
                size: layout.size.unwrap_or(0),
                align: layout.align.unwrap_or(0),
                field_offsets: layout
                    .variant_layouts
                    .first()
                    .map(|row| row.field_offsets.clone())
                    .unwrap_or_default(),
                transparent: layout.repr.as_ref().is_some_and(|repr| repr.transparent),
                discriminator: layout
                    .discriminator
                    .as_ref()
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "null".into()),
            });
        }
        hits
    }

    fn adt_id_of_ty(llbc: &Llbc, ty: &TyRef) -> Option<u64> {
        let json = super::super::tyref_json(ty);
        let body = super::super::tyexpr_body(llbc, &json).unwrap_or(&json);
        body.get("Adt")?.get("id")?.as_u64()
    }

    fn option_shaped_def_ids(llbc: &Llbc) -> Vec<u64> {
        llbc.iter_type_decls()
            .filter_map(|td| {
                let TypeDeclKind::Enum(variants) = &td.kind else {
                    return None;
                };
                if variants.len() != 2 {
                    return None;
                }
                if !variants[0].fields.is_empty() || variants[1].fields.len() != 1 {
                    return None;
                }
                Some(td.def_id)
            })
            .collect()
    }

    fn option_of(def_id: u64, inner: Value) -> TyRef {
        TyRef::Other(option_adt_node(def_id, inner))
    }

    /// Production-artefact census of the Opaque 3-word field-0 identity.
    /// Gated so `cargo test --lib host_tag` stays a unit-test run.
    #[test]
    fn opaque_three_word_census() {
        if std::env::var("HOST_TAG_OPAQUE_CENSUS").ok().as_deref() != Some("1") {
            return;
        }
        use crate::runtime_names::artifacts;
        let files: &[(&str, &str)] = &[
            ("pyre-interpreter.ullbc", artifacts::INTERPRETER_ULLBC),
            ("pyre-object.ullbc", artifacts::OBJECT_ULLBC),
            ("pyre-jit.ullbc", artifacts::JIT_ULLBC),
            ("pyre-module.ullbc", artifacts::MODULE_ULLBC),
        ];
        let mut all = Vec::new();
        let mut reached: Vec<(String, Vec<String>)> = Vec::new();
        for (label, path) in files {
            if !std::path::Path::new(path).is_file() {
                println!("skip missing {path}");
                continue;
            }
            let llbc = majit_charon_reader::Llbc::load(path)
                .unwrap_or_else(|err| panic!("load {path}: {err}"));
            let hits = census_opaque_three_word(&llbc, label);
            println!("=== {label} opaque 3-word field0 count={} ===", hits.len());
            for hit in &hits {
                println!(
                    "  {} def_id={} {} size={} align={} offsets={:?} transparent={} disc={}",
                    hit.file,
                    hit.def_id,
                    hit.name,
                    hit.size,
                    hit.align,
                    hit.field_offsets,
                    hit.transparent,
                    hit.discriminator
                );
            }
            let option_ids = option_shaped_def_ids(&llbc);
            let opaque_ids: std::collections::BTreeSet<u64> =
                hits.iter().map(|h| h.def_id).collect();
            let mut enum_users: Vec<(u64, String)> = Vec::new();
            for td in llbc.iter_type_decls() {
                let TypeDeclKind::Enum(variants) = &td.kind else {
                    continue;
                };
                let uses = variants.iter().any(|variant| {
                    variant.fields.iter().any(|field| {
                        adt_id_of_ty(&llbc, &field.ty).is_some_and(|id| opaque_ids.contains(&id))
                    })
                });
                if uses {
                    enum_users.push((td.def_id, td.item_meta.name_path()));
                }
            }
            println!(
                "  enums whose variant field is a matching opaque: {}",
                enum_users.len()
            );
            for (id, name) in &enum_users {
                println!("    enum {id} {name}");
            }
            let _ = take_opaque_three_word_hits();
            for hit in &hits {
                for opt_id in &option_ids {
                    let ty = option_of(*opt_id, adt_node(hit.def_id));
                    let _ = instantiated_enum_host_tag(&llbc, &ty);
                }
            }
            for (enum_id, _) in &enum_users {
                for opt_id in &option_ids {
                    let ty = option_of(*opt_id, adt_node(*enum_id));
                    let _ = instantiated_enum_host_tag(&llbc, &ty);
                }
                let ty = TyRef::Other(adt_node(*enum_id));
                let _ = instantiated_enum_host_tag(&llbc, &ty);
            }
            let mut names = take_opaque_three_word_hits();
            names.sort();
            names.dedup();
            println!("  instantiated_enum_host_tag opaque hits: {names:?}");
            reached.push(((*label).to_string(), names));
            all.extend(hits);
        }
        println!("TOTAL opaque 3-word field0 = {}", all.len());
        for (label, names) in &reached {
            println!("REACH {label}: {names:?}");
        }
        assert!(
            !all.is_empty(),
            "expected at least String-shaped Opaque in pyre-interpreter.ullbc"
        );
    }
}
