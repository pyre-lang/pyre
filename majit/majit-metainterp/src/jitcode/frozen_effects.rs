//! Restore the descriptor mutations made by `effectinfo.compute_bitstrings`.
//!
//! RPython's translated image keeps `FieldDescr.ei_index` and the matching
//! call bitstrings together. An embedded JitCode image crosses a process
//! boundary, so its `DescrMintEntry` recipes restore the same GcCache slots
//! before any call or field is consumed. The recipes are setup input, not a
//! runtime side table; only the descriptors themselves retain the stamps.

use majit_ir::descr::{DescrRef, FieldDescr, LLType, gc_cache};
use majit_ir::effectinfo::{DescrMintEntry, DescrMintSpec, DescrSetMember};

pub(super) fn publish_mints(entries: &[DescrMintEntry]) {
    for entry in entries {
        let descr = match (&entry.member, &entry.spec) {
            (
                DescrSetMember::Field {
                    struct_id,
                    field_name,
                },
                spec,
            ) => mint_field(*struct_id, field_name, spec).map(|d| d as DescrRef),
            (DescrSetMember::Array { array_id }, spec) => mint_array(*array_id, spec),
            (
                DescrSetMember::InteriorField { array_id, name },
                DescrMintSpec::InteriorField {
                    array,
                    field_struct_id,
                    field_name,
                    field,
                },
            ) => {
                let array_descr = mint_array(*array_id, array)
                    .and_then(majit_ir::descr::descr_arc_as_array_descr)
                    .expect("an interior field has an array descriptor");
                let field_descr = mint_field(*field_struct_id, field_name, field)
                    .expect("an interior field has an element field descriptor");
                Some(gc_cache().lock().get_interiorfield_descr(
                    LLType::Array(*array_id),
                    name.clone(),
                    String::new(),
                    array_descr,
                    field_descr,
                ))
            }
            _ => None,
        }
        .expect("an embedded effect descriptor's key and recipe must agree");
        descr.set_ei_index(entry.ei_index);
    }
}

fn mint_field(
    struct_id: u64,
    name: &str,
    spec: &DescrMintSpec,
) -> Option<std::sync::Arc<dyn majit_ir::descr::FieldDescr>> {
    let DescrMintSpec::Field {
        struct_size,
        offset,
        field_size,
        field_type,
        flag,
        is_immutable,
        is_quasi_immutable,
        index_in_parent,
    } = spec
    else {
        return None;
    };
    let key = LLType::Struct(struct_id);
    let mut gc = gc_cache().lock();
    // descr.py::get_field_descr: a cache hit preserves the original layout.
    // Opcode descriptors have already published their complete parent groups.
    if let Some(existing) = gc._cache_field.get(&key).and_then(|m| m.get(name)) {
        // `descr.py` `get_field_descr` returns `cache[STRUCT][fieldname]`
        // unchanged. Upstream keys on the STRUCT itself, so a hit always has
        // the recipe's layout. `(struct_id, name)` stands in for that key;
        // these asserts only check the invariant the upstream key gives by
        // construction, so `set_ei_index` never stamps another field.
        assert_eq!(
            existing.offset(),
            *offset,
            "mint_field: cached offset {} != recipe {offset} for {name}",
            existing.offset()
        );
        assert_eq!(
            existing.field_size(),
            *field_size,
            "mint_field: cached field size {} != recipe {field_size} for {name}",
            existing.field_size()
        );
        return Some(existing.clone());
    }
    gc.get_size_descr(key.clone(), *struct_size, 0, false);
    Some(gc.get_field_descr(
        key,
        name,
        None,
        *offset,
        *field_size,
        *field_type,
        *is_immutable,
        *is_quasi_immutable,
        *flag,
        u32::MAX,
        false,
        Some(*index_in_parent),
        false,
    ))
}

fn mint_array(array_id: u64, spec: &DescrMintSpec) -> Option<DescrRef> {
    let DescrMintSpec::Array {
        base_size,
        item_size,
        flag,
        item_type,
        nolength,
        length_offset,
        is_pure,
        concrete_type,
    } = spec
    else {
        return None;
    };
    Some(gc_cache().lock().get_array_descr(
        LLType::Array(array_id),
        *base_size,
        *item_size,
        *flag,
        *item_type,
        *nolength,
        *length_offset,
        *is_pure,
        *concrete_type,
    ))
}

fn resolve(member: &DescrSetMember) -> DescrRef {
    let gc = gc_cache().lock();
    let descr = match member {
        DescrSetMember::Field {
            struct_id,
            field_name,
        } => gc
            ._cache_field
            .get(&LLType::Struct(*struct_id))
            .and_then(|m| m.get(field_name))
            .map(|d| d.clone() as DescrRef),
        DescrSetMember::Array { array_id } => {
            gc._cache_array.get(&LLType::Array(*array_id)).cloned()
        }
        DescrSetMember::InteriorField { array_id, name } => gc
            ._cache_interiorfield
            .get(&(LLType::Array(*array_id), name.clone(), String::new()))
            .cloned(),
    };
    descr.unwrap_or_else(|| panic!("embedded EffectInfo member has no descriptor: {member:?}"))
}

pub(super) fn prepare(ei: &mut majit_ir::EffectInfo) {
    let raw_set = |members: &[DescrSetMember]| {
        Some(majit_ir::effectinfo::canonicalize_descr_set(
            members.iter().map(resolve).collect(),
        ))
    };
    // `effectinfo.py` `EffectInfo.__new__` stores the `_readonly_descrs_*`
    // / `_write_descrs_*` sets, and `compute_bitstrings` writes
    // `bitstring_*` without deleting them. dispatch.rs still re-interns
    // an EffectInfo by those sets, so `prepare` keeps them. The frozen
    // partition is what `check_readonly_descr_field` reads.
    if let Some(keys) = ei.descr_set_keys.take() {
        for member in keys
            .readonly_fields
            .iter()
            .chain(&keys.write_fields)
            .chain(&keys.readonly_arrays)
            .chain(&keys.write_arrays)
            .chain(&keys.readonly_interiorfields)
            .chain(&keys.write_interiorfields)
        {
            assert_ne!(
                resolve(member).get_ei_index(),
                u32::MAX,
                "embedded EffectInfo member has no frozen partition: {member:?}"
            );
        }
        // EffectInfo.__new__ retains this one reference after compaction.
        ei.single_write_descr_array = match keys.write_arrays.as_slice() {
            [member] => Some(resolve(member)),
            _ => None,
        };
        ei._readonly_descrs_fields = raw_set(&keys.readonly_fields);
        ei._write_descrs_fields = raw_set(&keys.write_fields);
        ei._readonly_descrs_arrays = raw_set(&keys.readonly_arrays);
        ei._write_descrs_arrays = raw_set(&keys.write_arrays);
        ei._readonly_descrs_interiorfields = raw_set(&keys.readonly_interiorfields);
        ei._write_descrs_interiorfields = raw_set(&keys.write_interiorfields);
    } else {
        assert_eq!(
            ei.extraeffect,
            majit_ir::ExtraEffect::RandomEffects,
            "a concrete embedded EffectInfo needs its descriptor image"
        );
    }
}

#[test]
#[should_panic(expected = "cached offset")]
fn mint_field_rejects_a_cached_offset_that_differs_from_the_recipe() {
    fn spec(offset: usize) -> DescrMintSpec {
        DescrMintSpec::Field {
            struct_size: 16,
            offset,
            field_size: 8,
            field_type: majit_ir::value::Type::Int,
            flag: majit_ir::descr::ArrayFlag::Signed,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
        }
    }
    let name = "mint_field_offset_guard";
    let struct_id = 0xD12F_1E1D;
    assert!(mint_field(struct_id, name, &spec(0)).is_some());
    let _ = mint_field(struct_id, name, &spec(8));
}
