//! Struct descriptor helpers from `rpython/jit/codewriter/heaptracker.py`.
//!
//! Most descriptor construction already lives on [`CallControl`], because
//! Pyre's codewriter needs the same cache identity while lowering calls and
//! emitting bytecode.  This module restores the PyPy namespace and symbol
//! names, routing every descriptor operation through those existing caches
//! instead of adding a side table.

use crate::codewriter::call::{CallControl, extract_element_type_from_str, get_type_flag};
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType, Struct};

#[derive(Debug, Clone, Default)]
pub struct GcStructVTableCache<V> {
    cache_gcstruct2vtable: std::collections::HashMap<String, V>,
    testing_gcstruct2vtable: std::collections::HashMap<String, V>,
}

impl<V> GcStructVTableCache<V> {
    #[cfg(test)]
    pub(crate) fn insert_rtyper_vtable(&mut self, gcstruct: &Struct, vtable: V) {
        self.cache_gcstruct2vtable
            .insert(gcstruct._name.clone(), vtable);
    }
}

pub fn is_immutable_struct(s: &Struct) -> bool {
    s._gckind == GcKind::Gc && matches!(s._hints.get("immutable"), Some(ConstValue::Bool(true)))
}

pub fn has_gcstruct_a_vtable(gcstruct: &Struct) -> bool {
    if gcstruct._gckind != GcKind::Gc {
        return false;
    }
    if LowLevelType::Struct(Box::new(gcstruct.clone()))
        == *crate::translator::rtyper::rclass::OBJECT
    {
        return false;
    }

    let mut cursor = gcstruct.clone();
    loop {
        if matches!(cursor._hints.get("typeptr"), Some(ConstValue::Bool(true))) {
            return true;
        }
        let Some((_, first_struct)) = cursor._first_struct_owned() else {
            return false;
        };
        cursor = first_struct;
    }
}

/// `jtransform.py is_typeptr_getset` / `jtransform.rs is_typeptr_field`:
/// the class word of the object header.  Upstream keys on `typeptr` and
/// the struct's `typeptr` hint; pyre's header is `PyObject { ob_type,
/// w_class }`, and only `ob_type` is the class the tracer guards on.
pub fn is_typeptr_field(field: &crate::model::FieldDescriptor) -> bool {
    let owner_leaf = field
        .owner_root
        .as_deref()
        .map(|owner| owner.rsplit("::").next().unwrap_or(owner));
    field.name == "ob_type" && owner_leaf == Some("PyObject")
}

/// `has_gcstruct_a_vtable` over ordered field entries (`STRUCT._names`).
///
/// Follow the first field while it is an inlined by-value struct. True
/// when that chain reaches the object-header struct whose first field is
/// the typeptr. The header type itself (`rclass.OBJECT`) has no vtable.
pub fn owner_has_vtable_from_fields(
    owner: &str,
    first_field: &dyn Fn(&str) -> Option<(crate::model::FieldDescriptor, Option<String>)>,
) -> bool {
    let start = majit_ir::descr::canonical_struct_name(owner);
    let mut cursor = start.clone();
    let mut seen = std::collections::HashSet::new();
    loop {
        if !seen.insert(cursor.clone()) {
            return false;
        }
        let Some((field, nested)) = first_field(&cursor) else {
            return false;
        };
        if is_typeptr_field(&field) {
            return cursor != start;
        }
        let Some(next) = nested else {
            return false;
        };
        cursor = majit_ir::descr::canonical_struct_name(&next);
    }
}

/// `struct_field_attrs` spelling of [`owner_has_vtable_from_fields`].
pub fn attrs_have_vtable(
    owner: &str,
    attrs: &std::collections::HashMap<String, Vec<(String, crate::model::ValueType)>>,
) -> bool {
    owner_has_vtable_from_fields(owner, &|o| {
        let key = majit_ir::descr::canonical_struct_name(o);
        let rows = attrs.get(&key)?;
        let (n, ty) = rows.first()?;
        let field = crate::model::FieldDescriptor::new(n.clone(), Some(key));
        let nested = match ty {
            crate::model::ValueType::Ref(Some(s)) => {
                Some(majit_ir::descr::canonical_struct_name(s))
            }
            _ => None,
        };
        Some((field, nested))
    })
}

/// `CallControl.struct_field_entries` spelling of [`owner_has_vtable_from_fields`].
pub fn callcontrol_has_vtable(cc: &CallControl, owner: &str) -> bool {
    owner_has_vtable_from_fields(owner, &|o| {
        let key = majit_ir::descr::canonical_struct_name(o);
        let entries = cc.struct_field_entries(&key)?;
        let first = entries.first()?;
        let field = crate::model::FieldDescriptor::new(first.name.clone(), Some(key));
        let nested = cc
            .is_known_struct(&first.ty)
            .then(|| majit_ir::descr::canonical_struct_name(&first.ty));
        Some((field, nested))
    })
}

pub fn get_vtable_for_gcstruct<V: Clone>(
    gccache: &mut GcStructVTableCache<V>,
    gcstruct: &Struct,
) -> Option<V> {
    if !has_gcstruct_a_vtable(gcstruct) {
        return None;
    }
    setup_cache_gcstruct2vtable(gccache);
    gccache
        .cache_gcstruct2vtable
        .get(&gcstruct._name)
        .or_else(|| gccache.testing_gcstruct2vtable.get(&gcstruct._name))
        .cloned()
}

/// `heaptracker.py` `setup_cache_gcstruct2vtable`.
///
/// Upstream fills `_cache_gcstruct2vtable` by walking
/// `rtyper.instance_reprs` and calling `rinstance.rclass.getvtable()` for
/// each.  Pyre keeps the call site and the lookup order so
/// [`get_vtable_for_gcstruct`] reads as upstream does, but the population is
/// deliberately empty, for two reasons that must both change before it is
/// worth writing:
///
/// * [`GcStructVTableCache`] is generic over the vtable representation and
///   holds no `RPythonTyper` handle, so there is nothing here to walk.  The
///   `instance_reprs` map exists (`RPythonTyper::instance_reprs`), so
///   populating means threading the rtyper in — a signature change on every
///   holder of this cache.
/// * [`get_vtable_for_gcstruct`] has no non-test caller.  Pyre resolves an
///   instance's class word through `new_with_vtable` at rewrite time
///   instead of asking the struct for its vtable after the fact, so the
///   upstream consumer (`rewrite_op_malloc`'s vtable argument) never
///   reaches this path.
///
/// The consequence of the stub is that a caller which did appear would read
/// `None` for a struct that has a vtable, so the empty body is only sound
/// while the testing map is the sole source — which is what
/// [`set_testing_vtable_for_gcstruct`] provides today.
pub fn setup_cache_gcstruct2vtable<V>(_gccache: &mut GcStructVTableCache<V>) {}

pub fn set_testing_vtable_for_gcstruct<V>(
    gccache: &mut GcStructVTableCache<V>,
    gcstruct: &Struct,
    vtable: V,
    _name: &str,
) {
    gccache
        .testing_gcstruct2vtable
        .insert(gcstruct._name.clone(), vtable);
}

/// `heaptracker.py:64-66` `if name == 'typeptr': continue`, spelled in the
/// source field-name domain this walker actually sees.
///
/// Upstream's positional census never numbers the vtable word: `typeptr`
/// lives in `OBJECT`, the substructure every instance embeds first, and the
/// walk skips it by name. Pyre embeds that word as `ob_type` inside
/// `PyObject { ob_type, w_class }`. The other header leaf is an ordinary
/// field: `all_fielddescrs` recurses into the inlined struct and appends
/// `get_field_descr(gccache, INNER, name)` (`heaptracker.py:68-71`), so
/// `w_class` is a row of every outer object's list, keyed `(PyObject,
/// w_class)`.
pub(crate) fn is_header_word(_struct_name: &str, field_name: &str) -> bool {
    field_name == "typeptr" || field_name == "ob_type"
}

pub fn all_fielddescrs(
    gccache: &CallControl,
    struct_name: &str,
    only_gc: bool,
) -> Vec<majit_ir::descr::DescrRef> {
    let mut res = Vec::new();
    all_fielddescrs_into(gccache, struct_name, only_gc, &mut res);
    res
}

pub fn all_interiorfielddescrs(
    gccache: &CallControl,
    array_type_id: &str,
) -> Result<Vec<majit_ir::descr::DescrRef>, majit_ir::UnsupportedFieldExc> {
    let elem_name =
        extract_element_type_from_str(array_type_id).unwrap_or_else(|| array_type_id.to_string());
    if let Some(layout) = gccache.struct_layout_for(&elem_name) {
        for field in &layout.fields {
            if field.field_type == majit_ir::value::Type::Void || field.name == "typeptr" {
                continue;
            }
            if field.flag == majit_ir::descr::ArrayFlag::Struct {
                return Err(majit_ir::UnsupportedFieldExc(
                    "unexpected array(struct(struct))".to_string(),
                ));
            }
        }
    } else if let Some(fields) = gccache.struct_field_entries(&elem_name) {
        for row in fields {
            if row.name == "typeptr" || row.name.starts_with("c__pad") {
                continue;
            }
            let (_, ir_type, _) = get_type_flag(&row.ty);
            if ir_type == majit_ir::value::Type::Void {
                continue;
            }
            if gccache.is_known_struct(&row.ty) {
                return Err(majit_ir::UnsupportedFieldExc(
                    "unexpected array(struct(struct))".to_string(),
                ));
            }
        }
    } else {
        return Ok(Vec::new());
    }

    let mut res = Vec::new();
    let Some(fields) = gccache.struct_field_entries(&elem_name) else {
        if let Some(layout) = gccache.struct_layout_for(&elem_name) {
            for field in &layout.fields {
                if field.field_type == majit_ir::value::Type::Void || field.name == "typeptr" {
                    continue;
                }
                let array_id = Some(array_type_id.to_string());
                let idx = gccache
                    .descr_indices
                    .interiorfield_index(&array_id, &field.name);
                if let Some(descr) = gccache.interiorfielddescrof(idx, &array_id, &field.name) {
                    res.push(descr);
                }
            }
        }
        return Ok(res);
    };

    for row in fields {
        if row.name == "typeptr" || row.name.starts_with("c__pad") {
            continue;
        }
        let (_, ir_type, _) = get_type_flag(&row.ty);
        if ir_type == majit_ir::value::Type::Void {
            continue;
        }
        let array_id = Some(array_type_id.to_string());
        let idx = gccache
            .descr_indices
            .interiorfield_index(&array_id, &row.name);
        if let Some(descr) = gccache.interiorfielddescrof(idx, &array_id, &row.name) {
            res.push(descr);
        }
    }
    Ok(res)
}

pub fn gc_fielddescrs(gccache: &CallControl, struct_name: &str) -> Vec<majit_ir::descr::DescrRef> {
    all_fielddescrs(gccache, struct_name, true)
}

pub fn get_fielddescr_index_in(
    gccache: &CallControl,
    struct_name: &str,
    fieldname: &str,
    cur_index: isize,
) -> isize {
    let mut cur_index = cur_index;
    let Some(fields) = gccache.struct_field_entries(struct_name) else {
        return -cur_index - 1;
    };
    // A direct (non-struct) field of this STRUCT shadows the same name
    // inherited from an inlined nested struct. Upstream never hits this:
    // `OBJECT` contributes only the skipped `typeptr`, so no inner leaf
    // shares a name with an outer payload field. Pyre's extra header leaf
    // (`PyObject.w_class`) would otherwise steal `Method.w_class`'s slot.
    let has_direct = fields.iter().any(|row| {
        if row.name != fieldname {
            return false;
        }
        let (_, ir_type, _) = get_type_flag(&row.ty);
        ir_type != majit_ir::value::Type::Void && !gccache.is_known_struct(&row.ty)
    });
    for row in fields {
        let name = &row.name;
        let field_type = &row.ty;
        let (_, ir_type, _) = get_type_flag(field_type);
        if ir_type == majit_ir::value::Type::Void {
            continue;
        }
        if is_header_word(struct_name, name) {
            continue;
        }
        if gccache.is_known_struct(field_type) {
            // The dotted `outer.inner` spelling names one leaf of this
            // nested struct on the outer owner.
            if let Some(rest) = fieldname
                .strip_prefix(name.as_str())
                .and_then(|rest| rest.strip_prefix('.'))
            {
                let r = get_fielddescr_index_in(gccache, field_type, rest, cur_index);
                if r >= 0 {
                    return r;
                }
                cur_index = -r - 1;
                continue;
            }
            if has_direct {
                // Count inner leaves; do not return an inner name match.
                let r = get_fielddescr_index_in(gccache, field_type, "", cur_index);
                cur_index = -r - 1;
            } else {
                let r = get_fielddescr_index_in(gccache, field_type, fieldname, cur_index);
                if r >= 0 {
                    return r;
                }
                // the recursion was handed our own `cur_index`, so the index it
                // reports back already counts the fields walked before the nested
                // struct; adding it again would count them twice
                cur_index = -r - 1;
            }
            continue;
        }
        if name == fieldname {
            return cur_index;
        }
        cur_index += 1;
    }
    -cur_index - 1
}

fn all_fielddescrs_into(
    gccache: &CallControl,
    struct_name: &str,
    only_gc: bool,
    res: &mut Vec<majit_ir::descr::DescrRef>,
) {
    let Some(fields) = gccache
        .struct_field_entries(struct_name)
        .map(|f| f.to_vec())
    else {
        return;
    };
    for row in fields {
        let name = &row.name;
        let field_type = &row.ty;
        let (flag, ir_type, _) = get_type_flag(field_type);
        if ir_type == majit_ir::value::Type::Void {
            continue;
        }
        if name.starts_with("c__pad") || is_header_word(struct_name, name) {
            continue;
        }
        if gccache.is_known_struct(field_type) {
            all_fielddescrs_into(gccache, field_type, only_gc, res);
        } else if !only_gc || flag == majit_ir::descr::ArrayFlag::Pointer {
            let owner = Some(struct_name.to_string());
            let idx = gccache.descr_indices.field_index(&owner, name);
            if let Some(descr) = gccache.fielddescrof(idx, struct_name, None, &name) {
                res.push(descr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_hint_helpers_match_heaptracker_predicates() {
        let raw = Struct::with_hints(
            "raw",
            vec![("value".into(), LowLevelType::Signed)],
            vec![("immutable".into(), ConstValue::Bool(true))],
        );
        assert!(!is_immutable_struct(&raw));

        let gc = Struct::gc_with_hints(
            "gc",
            vec![("value".into(), LowLevelType::Signed)],
            vec![("immutable".into(), ConstValue::Bool(true))],
        );
        assert!(is_immutable_struct(&gc));
        assert!(!has_gcstruct_a_vtable(&gc));

        let object_sub = Struct::gc_with_hints(
            "object_sub",
            vec![(
                "super".into(),
                crate::translator::rtyper::rclass::OBJECT.clone(),
            )],
            vec![],
        );
        assert!(has_gcstruct_a_vtable(&object_sub));
    }

    #[test]
    fn vtable_cache_uses_rtyper_then_testing_slot() {
        let gc = Struct::gc_with_hints(
            "instance",
            vec![("typeptr".into(), LowLevelType::Signed)],
            vec![("typeptr".into(), ConstValue::Bool(true))],
        );
        let mut cache = GcStructVTableCache::default();
        set_testing_vtable_for_gcstruct(&mut cache, &gc, "testing", "Instance");
        assert_eq!(get_vtable_for_gcstruct(&mut cache, &gc), Some("testing"));
        cache.insert_rtyper_vtable(&gc, "rtyper");
        assert_eq!(get_vtable_for_gcstruct(&mut cache, &gc), Some("rtyper"));
    }

    #[test]
    fn field_list_has_vtable_when_first_struct_chain_reaches_typeptr() {
        let _registry = crate::test_support::register_struct_origins_serialized(
            std::collections::HashMap::new(),
        );
        // `attrs_have_vtable` looks up `canonical_struct_name(owner)`. A
        // sibling lib test that loaded interpreter LLBC may have registered
        // `PyObject` in `STRUCT_ORIGIN_REGISTRY`, so the leaf spelling is
        // not a stable key.
        let header_type = majit_ir::descr::canonical_struct_name("PyObject");
        let wrapper_type = majit_ir::descr::canonical_struct_name("Boxed");
        let pair_type = majit_ir::descr::canonical_struct_name("Pair");
        let mut attrs = std::collections::HashMap::new();
        attrs.insert(
            header_type,
            vec![("ob_type".to_string(), crate::model::ValueType::Ref(None))],
        );
        attrs.insert(
            wrapper_type,
            vec![
                (
                    "ob_header".to_string(),
                    crate::model::ValueType::Ref(Some("PyObject".into())),
                ),
                ("payload".to_string(), crate::model::ValueType::Int),
            ],
        );
        attrs.insert(
            pair_type,
            vec![
                ("a".to_string(), crate::model::ValueType::Int),
                ("b".to_string(), crate::model::ValueType::Int),
            ],
        );
        assert!(attrs_have_vtable("Boxed", &attrs));
        assert!(!attrs_have_vtable("PyObject", &attrs));
        assert!(!attrs_have_vtable("Pair", &attrs));
        let mut erased = std::collections::HashMap::new();
        erased.insert(
            "Frame".to_string(),
            vec![
                ("ob_header".to_string(), crate::model::ValueType::Ref(None)),
                ("payload".to_string(), crate::model::ValueType::Int),
            ],
        );
        assert!(
            !attrs_have_vtable("Frame", &erased),
            "an unresolved first-field type is not an inlined struct"
        );
    }

    #[test]
    fn callcontrol_has_vtable_walks_inlined_header() {
        let _registry = crate::test_support::register_struct_origins_serialized(
            std::collections::HashMap::new(),
        );
        let header_type = majit_ir::descr::canonical_struct_name("PyObject");
        let wrapper_type = majit_ir::descr::canonical_struct_name("Boxed");
        let mut cc = CallControl::new();
        cc.set_known_struct_names(
            [
                header_type.clone(),
                wrapper_type.clone(),
                "PyObject".to_string(),
                "Boxed".to_string(),
            ]
            .into(),
        );
        let mut fields = crate::front::StructFieldRegistry::default();
        let header = vec![("ob_type".to_string(), "usize".to_string())];
        let wrapper = vec![
            ("ob_header".to_string(), "PyObject".to_string()),
            ("payload".to_string(), "i64".to_string()),
        ];
        fields.fields.insert(header_type, header.clone());
        fields.fields.insert("PyObject".to_string(), header);
        fields.fields.insert(wrapper_type, wrapper.clone());
        fields.fields.insert("Boxed".to_string(), wrapper);
        cc.set_struct_fields(fields);
        assert!(callcontrol_has_vtable(&cc, "Boxed"));
        assert!(!callcontrol_has_vtable(&cc, "PyObject"));
    }

    #[test]
    fn field_index_recurses_through_nested_structs() {
        let mut cc = CallControl::new();
        cc.set_known_struct_names(["Inner".to_string(), "Outer".to_string()].into());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "Inner".to_string(),
            vec![
                ("a".to_string(), "i64".to_string()),
                ("b".to_string(), "i64".to_string()),
            ],
        );
        fields.fields.insert(
            "Outer".to_string(),
            vec![
                ("typeptr".to_string(), "usize".to_string()),
                ("inner".to_string(), "Inner".to_string()),
                ("c".to_string(), "i64".to_string()),
            ],
        );
        cc.set_struct_fields(fields);

        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "a", 0), 0);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "b", 0), 1);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "c", 0), 2);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "missing", 0), -4);
    }

    #[test]
    fn field_index_after_a_non_leading_nested_struct() {
        // the nested struct does not start at index 0 here, so the recursive
        // call is handed a nonzero `cur_index`.  Declaration order is
        // a, inner.x, inner.y, b -> the walk must number them 0, 1, 2, 3.
        let mut cc = CallControl::new();
        cc.set_known_struct_names(["Inner".to_string(), "Outer".to_string()].into());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "Inner".to_string(),
            vec![
                ("x".to_string(), "i64".to_string()),
                ("y".to_string(), "i64".to_string()),
            ],
        );
        fields.fields.insert(
            "Outer".to_string(),
            vec![
                ("a".to_string(), "i64".to_string()),
                ("inner".to_string(), "Inner".to_string()),
                ("b".to_string(), "i64".to_string()),
            ],
        );
        cc.set_struct_fields(fields);

        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "a", 0), 0);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "x", 0), 1);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "y", 0), 2);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "b", 0), 3);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "missing", 0), -5);
    }
}
