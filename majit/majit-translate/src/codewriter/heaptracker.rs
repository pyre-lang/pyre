//! Struct descriptor helpers from `rpython/jit/codewriter/heaptracker.py`.
//!
//! Most descriptor construction already lives on [`CallControl`], because
//! Pyre's codewriter needs the same cache identity while lowering calls and
//! emitting bytecode.  This module restores the PyPy namespace and symbol
//! names, routing every descriptor operation through those existing caches
//! instead of adding a side table.

use crate::codewriter::call::{
    CallControl, array_type_id_of_lltype, extract_element_type_from_str, get_type_flag,
};
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::lltypesystem::lltype::{Array, GcKind, LowLevelType, Struct};

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

/// Last path component is the class-word leaf (`rclass.OBJECT.typeptr`;
/// pyre's header spells it `ob_type`).
fn is_typeptr_leaf(name: &str) -> bool {
    let leaf = name.rsplit('.').next().unwrap_or(name);
    leaf == "typeptr" || leaf == "ob_type"
}

/// Ordered first field of `owner` (`STRUCT._names[0]`).
///
/// Rows come from [`CallControl::struct_field_entries`]
/// (`lookup_fields` / `field_name_at`). Nested follow uses
/// `is_known_struct`, the same test [`callcontrol_has_vtable`] uses
/// when walking `has_gcstruct_a_vtable` / `_first_struct_owned`.
fn callcontrol_first_field(cc: &CallControl, owner: &str) -> Option<(String, Option<String>)> {
    let entries = cc.struct_field_entries(owner)?;
    let first = entries.first()?;
    let nested = cc.is_known_struct(&first.ty).then(|| first.ty.clone());
    Some((first.name.clone(), nested))
}

/// Whether `field_path` on `owner` is the typeptr reached by following
/// only first fields (offset 0) from that owner.
///
/// lltype inlines a GcStruct only as the first field of another
/// GcStruct (`has_gcstruct_a_vtable` / `_first_struct_owned`).
/// `rclass.OBJECT` is a GcStruct, so the only reachable typeptr inside
/// any GC struct is the one at the end of that chain — which is why
/// `heaptracker.py all_fielddescrs` can `if name == 'typeptr': continue`
/// at every recursion level, and why `jtransform.py is_typeptr_getset` /
/// `handle_getfield_typeptr`
/// (`op.args[0].concretetype.TO._hints.get('typeptr')`) only ever see
/// the object's own class word.
///
/// Pyre can embed an object at a non-zero offset (`GcFramePrefix.frame`
/// after `gc_header`). That contained `ob_type` is an ordinary data
/// word: the census publishes it and `is_typeptr_getset` does not drop
/// the store.
pub fn is_first_field_chain_typeptr(
    owner: &str,
    field_path: &str,
    first_field: &dyn Fn(&str) -> Option<(String, Option<String>)>,
) -> bool {
    if !is_typeptr_leaf(field_path) {
        return false;
    }
    let mut cursor = majit_ir::descr::canonical_struct_name(owner);
    let mut remaining = field_path;
    loop {
        let Some((first_name, nested)) = first_field(&cursor) else {
            return false;
        };
        if remaining == first_name {
            return is_typeptr_leaf(&first_name);
        }
        let Some(rest) = remaining
            .strip_prefix(first_name.as_str())
            .and_then(|rest| rest.strip_prefix('.'))
        else {
            return false;
        };
        let Some(next) = nested else {
            return false;
        };
        cursor = majit_ir::descr::canonical_struct_name(&next);
        remaining = rest;
    }
}

/// `rclass.py OBJECT` / `heaptracker.py has_gcstruct_a_vtable`: the
/// object-header STRUCT is the one whose own first field is the typeptr.
///
/// `OBJECT = GcStruct('object', ('typeptr', CLASSTYPE), hints={...,
/// 'typeptr': True})`. `has_gcstruct_a_vtable` returns False for OBJECT
/// because that first field is the typeptr itself. A subclass inlines
/// the parent as field 0 (`struct X { struct Y super; ... }`); that
/// first field is the parent struct, not the typeptr, and the walk
/// follows `_first_struct` until the header.
///
/// Zero-length first-field chain: the owner's first field entry is the
/// typeptr leaf itself — not a nested struct and not a dotted path
/// through an inlined header (`ob_header.ob_type`).
pub(crate) fn is_object_header_owner(cc: &CallControl, owner: &str) -> bool {
    let Some((name, nested)) = callcontrol_first_field(cc, owner) else {
        return false;
    };
    if nested.is_some() || name.contains('.') {
        return false;
    }
    is_first_field_chain_typeptr(owner, &name, &|o| callcontrol_first_field(cc, o))
}

/// Host Python instance header for `InstanceRepr._setup_repr`.
///
/// `rclass.OBJECT` / `virtualref.ObjectHeader` is a one-word typeptr.
/// Pyre's instance root is that header plus the mutable class word
/// (`PyObject.ob_type`, then `w_class`). `minted_exception_instance_layout`
/// roots at that host header, not a one-word typeptr. A subclass still
/// fails [`is_object_header_owner`]: its first field is the inlined
/// parent, not the typeptr leaf.
pub(crate) fn is_host_object_header_owner(cc: &CallControl, owner: &str) -> bool {
    if !is_object_header_owner(cc, owner) {
        return false;
    }
    cc.struct_field_entries(owner)
        .is_some_and(|rows| rows.len() > 1)
}

/// `jtransform.py is_typeptr_getset` / `handle_getfield_typeptr`: the
/// class word of this owner's object header, the typeptr at the end of
/// the offset-0 first-field chain ([`is_first_field_chain_typeptr`]).
pub fn is_typeptr_field(cc: &CallControl, field: &crate::model::FieldDescriptor) -> bool {
    is_first_field_chain_typeptr(
        field.owner_root.as_deref().unwrap_or(""),
        &field.name,
        &|owner| callcontrol_first_field(cc, owner),
    )
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
        // Already on the first-field walk (`_first_struct_owned`). The
        // header struct is the one whose first field is the typeptr.
        if is_typeptr_leaf(&field.name) {
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
        let (name, nested) = callcontrol_first_field(cc, o)?;
        let field = crate::model::FieldDescriptor::new(name, Some(key));
        Some((
            field,
            nested.map(|ty| majit_ir::descr::canonical_struct_name(&ty)),
        ))
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

/// `heaptracker.py all_fielddescrs` skip of `typeptr`, spelled for the
/// first-field-chain typeptr of `struct_name` (the census owner).
///
/// Upstream skips `typeptr` by name at every recursion level because
/// lltype inlines a GcStruct only as the first field, so every
/// `typeptr` the walk meets is that owner's class word. A typeptr at a
/// non-zero offset (`GcFramePrefix.frame.ob_header.ob_type`) is not on
/// this chain and is published like any other field.
pub(crate) fn is_header_word(cc: &CallControl, struct_name: &str, field_name: &str) -> bool {
    is_first_field_chain_typeptr(struct_name, field_name, &|owner| {
        callcontrol_first_field(cc, owner)
    })
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
    array: &Array,
) -> Result<Vec<majit_ir::descr::DescrRef>, majit_ir::UnsupportedFieldExc> {
    let array_type_id = array_type_id_of_lltype(array);
    let elem_name = match &array.OF {
        LowLevelType::Struct(s) => s._name.clone(),
        _ => extract_element_type_from_str(&array_type_id).unwrap_or_else(|| array_type_id.clone()),
    };
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
            let (_, ir_type, _) = get_type_flag(&row.ty, &gccache.raw_alloc_flavor_structs);
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
                let spelling = Some(array_type_id.clone());
                let intern_id = gccache.intern_array(array);
                let idx = gccache
                    .descr_indices
                    .interiorfield_index(intern_id, &field.name);
                if let Some(descr) = gccache.interiorfielddescrof_with_array(
                    idx,
                    &spelling,
                    &Some(array.clone()),
                    &field.name,
                ) {
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
        let (_, ir_type, _) = get_type_flag(&row.ty, &gccache.raw_alloc_flavor_structs);
        if ir_type == majit_ir::value::Type::Void {
            continue;
        }
        let spelling = Some(array_type_id.clone());
        let intern_id = gccache.intern_array(array);
        let idx = gccache
            .descr_indices
            .interiorfield_index(intern_id, &row.name);
        if let Some(descr) =
            gccache.interiorfielddescrof_with_array(idx, &spelling, &Some(array.clone()), &row.name)
        {
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
    get_fielddescr_index_in_from(gccache, struct_name, struct_name, "", fieldname, cur_index)
}

fn get_fielddescr_index_in_from(
    gccache: &CallControl,
    root_owner: &str,
    struct_name: &str,
    field_prefix: &str,
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
        let (_, ir_type, _) = get_type_flag(&row.ty, &gccache.raw_alloc_flavor_structs);
        ir_type != majit_ir::value::Type::Void && !gccache.is_known_struct(&row.ty)
    });
    for row in fields {
        let name = &row.name;
        let field_type = &row.ty;
        let (_, ir_type, _) = get_type_flag(field_type, &gccache.raw_alloc_flavor_structs);
        if ir_type == majit_ir::value::Type::Void {
            continue;
        }
        let full_name = if field_prefix.is_empty() {
            name.clone()
        } else {
            format!("{field_prefix}{name}")
        };
        // Skip only the original owner's first-field-chain typeptr, not
        // an inner `ob_type` reached through a later field.
        if is_header_word(gccache, root_owner, &full_name) {
            continue;
        }
        if gccache.is_known_struct(field_type) {
            let nested_prefix = format!("{full_name}.");
            // The dotted `outer.inner` spelling names one leaf of this
            // nested struct on the outer owner.
            if let Some(rest) = fieldname
                .strip_prefix(name.as_str())
                .and_then(|rest| rest.strip_prefix('.'))
            {
                let r = get_fielddescr_index_in_from(
                    gccache,
                    root_owner,
                    field_type,
                    &nested_prefix,
                    rest,
                    cur_index,
                );
                if r >= 0 {
                    return r;
                }
                cur_index = -r - 1;
                continue;
            }
            if has_direct {
                // Count inner leaves; do not return an inner name match.
                let r = get_fielddescr_index_in_from(
                    gccache,
                    root_owner,
                    field_type,
                    &nested_prefix,
                    "",
                    cur_index,
                );
                cur_index = -r - 1;
            } else {
                let r = get_fielddescr_index_in_from(
                    gccache,
                    root_owner,
                    field_type,
                    &nested_prefix,
                    fieldname,
                    cur_index,
                );
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
    all_fielddescrs_into_from(gccache, struct_name, struct_name, "", only_gc, res);
}

fn all_fielddescrs_into_from(
    gccache: &CallControl,
    root_owner: &str,
    struct_name: &str,
    field_prefix: &str,
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
        let (flag, ir_type, _) = get_type_flag(field_type, &gccache.raw_alloc_flavor_structs);
        if ir_type == majit_ir::value::Type::Void {
            continue;
        }
        let full_name = if field_prefix.is_empty() {
            name.clone()
        } else {
            format!("{field_prefix}{name}")
        };
        if name.starts_with("c__pad") || is_header_word(gccache, root_owner, &full_name) {
            continue;
        }
        if gccache.is_known_struct(field_type) {
            let nested_prefix = format!("{full_name}.");
            all_fielddescrs_into_from(
                gccache,
                root_owner,
                field_type,
                &nested_prefix,
                only_gc,
                res,
            );
        } else if !only_gc || flag == majit_ir::descr::ArrayFlag::Pointer {
            let owner = Some(struct_name.to_string());
            let idx = gccache.descr_indices.field_index(&owner, name);
            if let Some(descr) = gccache.fielddescrof(idx, struct_name, None, name) {
                res.push(descr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_cc(rows: &[(&str, &[(&str, &str)])]) -> CallControl {
        let mut cc = CallControl::new();
        let mut fields = crate::front::StructFieldRegistry::default();
        let mut known = std::collections::HashSet::new();
        for (owner, fs) in rows {
            known.insert((*owner).to_string());
            fields.fields.insert(
                (*owner).to_string(),
                fs.iter()
                    .map(|(n, t)| ((*n).to_string(), (*t).to_string()))
                    .collect::<Vec<_>>(),
            );
        }
        cc.set_known_struct_names(known);
        cc.set_struct_fields(fields);
        cc
    }

    fn object_layout_cc() -> CallControl {
        layout_cc(&[
            (
                "PyObject",
                &[("ob_type", "*const PyType"), ("w_class", "*mut PyObject")],
            ),
            ("object", &[("typeptr", "*const PyType")]),
            (
                "W_BytesObject",
                &[("ob_header", "PyObject"), ("data", "*mut u8")],
            ),
            (
                "W_FloatObject",
                &[("ob_header", "PyObject"), ("value", "f64")],
            ),
            (
                "Method",
                &[
                    ("ob", "PyObject"),
                    ("w_function", "*mut PyObject"),
                    ("w_self", "*mut PyObject"),
                    ("w_class", "*mut PyObject"),
                    ("w_module", "*mut PyObject"),
                ],
            ),
            ("Outer", &[("typeptr", "*const PyType"), ("payload", "i64")]),
            (
                "OuterLater",
                &[("payload", "i64"), ("typeptr", "*const PyType")],
            ),
            (
                "PyFrame",
                &[("ob_header", "PyObject"), ("pycode", "*mut u8")],
            ),
            (
                "GcFramePrefix",
                &[("gc_header", "u64"), ("frame", "PyFrame")],
            ),
            (
                "pyframe::GcFramePrefix",
                &[("gc_header", "u64"), ("frame", "PyFrame")],
            ),
        ])
    }

    #[test]
    fn header_word_is_the_first_field_chain_typeptr() {
        let _registry = crate::test_support::register_struct_origins_serialized(
            std::collections::HashMap::new(),
        );
        let cc = object_layout_cc();
        assert!(is_header_word(&cc, "PyObject", "ob_type"));
        assert!(is_header_word(&cc, "object", "typeptr"));
        assert!(is_header_word(&cc, "W_BytesObject", "ob_header.ob_type"));
        assert!(is_header_word(&cc, "W_FloatObject", "ob_header.ob_type"));
        assert!(!is_header_word(&cc, "PyObject", "w_class"));
        assert!(!is_header_word(&cc, "W_BytesObject", "ob_header.w_class"));
        assert!(!is_header_word(&cc, "Method", "ob"));
        assert!(is_header_word(&cc, "Method", "ob.ob_type"));
        assert!(is_header_word(&cc, "Outer", "typeptr"));
        assert!(!is_header_word(&cc, "OuterLater", "typeptr"));
        assert!(!is_header_word(
            &cc,
            "pyframe::GcFramePrefix",
            "frame.ob_header.ob_type"
        ));
    }

    /// `rclass.py OBJECT` is the STRUCT whose own first field is the
    /// typeptr. A subclass inlines the parent (`ob_header`) or publishes
    /// a dotted path (`ob_header.ob_type`); those are the chain, not OBJECT.
    #[test]
    fn object_header_owner_is_the_struct_whose_own_first_field_is_the_typeptr() {
        let _registry = crate::test_support::register_struct_origins_serialized(
            std::collections::HashMap::new(),
        );
        let cc = object_layout_cc();
        assert!(is_object_header_owner(&cc, "PyObject"));
        assert!(is_object_header_owner(&cc, "object"));
        assert!(is_object_header_owner(&cc, "Outer"));
        assert!(is_host_object_header_owner(&cc, "PyObject"));
        assert!(!is_host_object_header_owner(&cc, "object"));
        assert!(is_host_object_header_owner(&cc, "Outer"));
        assert!(!is_object_header_owner(&cc, "W_BytesObject"));
        assert!(!is_object_header_owner(&cc, "W_FloatObject"));
        assert!(!is_object_header_owner(&cc, "Method"));
        assert!(!is_object_header_owner(&cc, "OuterLater"));
        assert!(!is_object_header_owner(&cc, "GcFramePrefix"));
        assert!(!is_object_header_owner(&cc, "pyframe::GcFramePrefix"));
        assert!(!is_host_object_header_owner(&cc, "W_BytesObject"));
        let flat = layout_cc(&[(
            "W_Flat",
            &[("ob_header.ob_type", "*const PyType"), ("value", "f64")],
        )]);
        assert!(!is_object_header_owner(&flat, "W_Flat"));
        assert!(!is_host_object_header_owner(&flat, "W_Flat"));
        assert!(is_header_word(&flat, "W_Flat", "ob_header.ob_type"));
        let leaf = crate::model::FieldDescriptor::new("ob_header.ob_type", Some("W_Flat".into()));
        assert!(is_typeptr_field(&flat, &leaf));
    }

    #[test]
    fn typeptr_field_matches_nested_header_word() {
        let _registry = crate::test_support::register_struct_origins_serialized(
            std::collections::HashMap::new(),
        );
        let cc = object_layout_cc();
        let leaf = crate::model::FieldDescriptor::new("ob_type", Some("PyObject".into()));
        assert!(is_typeptr_field(&cc, &leaf));
        let float_header =
            crate::model::FieldDescriptor::new("ob_header.ob_type", Some("W_FloatObject".into()));
        assert!(is_typeptr_field(&cc, &float_header));
        let nested = crate::model::FieldDescriptor::new(
            "frame.ob_header.ob_type",
            Some("pyframe::GcFramePrefix".into()),
        );
        assert!(!is_typeptr_field(&cc, &nested));
        let first_typeptr = crate::model::FieldDescriptor::new("typeptr", Some("Outer".into()));
        assert!(is_typeptr_field(&cc, &first_typeptr));
        let later_typeptr =
            crate::model::FieldDescriptor::new("typeptr", Some("OuterLater".into()));
        assert!(!is_typeptr_field(&cc, &later_typeptr));
        let w_class_leaf = crate::model::FieldDescriptor::new("w_class", Some("PyObject".into()));
        assert!(!is_typeptr_field(&cc, &w_class_leaf));
        let w_class = crate::model::FieldDescriptor::new(
            "frame.ob_header.w_class",
            Some("pyframe::GcFramePrefix".into()),
        );
        assert!(!is_typeptr_field(&cc, &w_class));
    }

    /// `jtransform.py is_typeptr_getset` requires the typeptr-bearing
    /// object. `GcFramePrefix.frame` sits after the GC header, so a
    /// GuardClass on the prefix base would read the wrong word.
    #[test]
    fn typeptr_field_declines_gc_frame_prefix_contained_object() {
        let _registry = crate::test_support::register_struct_origins_serialized(
            std::collections::HashMap::new(),
        );
        let cc = object_layout_cc();
        let nested = crate::model::FieldDescriptor::new(
            "frame.ob_header.ob_type",
            Some("pyframe::GcFramePrefix".into()),
        );
        assert!(!is_typeptr_field(&cc, &nested));
        assert!(!is_header_word(
            &cc,
            "pyframe::GcFramePrefix",
            "frame.ob_header.ob_type"
        ));
        let method_header = crate::model::FieldDescriptor::new("ob.ob_type", Some("Method".into()));
        assert!(is_typeptr_field(&cc, &method_header));
    }

    /// Harvest keys the full def-path; FieldWrite `owner_root` is the
    /// relative path (`pyobject::PyObject`). `lookup_fields` is the
    /// STRUCT identity (`field_name_at`); exact HashMap lookup is not.
    #[test]
    fn first_field_chain_uses_harvested_field_row_identity() {
        let mut origins = std::collections::HashMap::new();
        origins.insert("PyObject".to_string(), "pyre_object::pyobject".to_string());
        let _registry = crate::test_support::register_struct_origins_serialized(origins);
        let cc = layout_cc(&[(
            "pyre_object::pyobject::PyObject",
            &[("ob_type", "*const PyType"), ("w_class", "*mut PyObject")],
        )]);
        let leaf = crate::model::FieldDescriptor::new("ob_type", Some("pyobject::PyObject".into()));
        assert!(is_typeptr_field(&cc, &leaf));
        assert!(is_header_word(&cc, "pyobject::PyObject", "ob_type"));
        let w_class =
            crate::model::FieldDescriptor::new("w_class", Some("pyobject::PyObject".into()));
        assert!(!is_typeptr_field(&cc, &w_class));
    }

    /// `heaptracker.py all_fielddescrs` skips `typeptr` at every
    /// recursion because lltype inlines a GcStruct only as the first
    /// field. A typeptr at a non-zero offset is published.
    #[test]
    fn census_publishes_nonzero_offset_typeptr_and_skips_first_field_chain() {
        let _registry = crate::test_support::register_struct_origins_serialized(
            std::collections::HashMap::new(),
        );
        let cc = object_layout_cc();
        assert!(
            get_fielddescr_index_in(&cc, "GcFramePrefix", "frame.ob_header.ob_type", 0) >= 0,
            "contained-object typeptr is a numbered leaf of GcFramePrefix"
        );
        assert!(
            get_fielddescr_index_in(&cc, "W_FloatObject", "ob_header.ob_type", 0) < 0,
            "first-field-chain typeptr stays out of the census"
        );
        assert!(
            get_fielddescr_index_in(&cc, "PyObject", "ob_type", 0) < 0,
            "PyObject.ob_type is the owner's own typeptr"
        );
        assert_eq!(
            get_fielddescr_index_in(&cc, "W_FloatObject", "ob_header.w_class", 0),
            0
        );
    }

    /// `lookup_fields` strips `<…>` before the map lookup. The census
    /// must keep the generic harvest key so `Option<T>::Some.__pos_0`
    /// is numbered (`descr.py` `get_fielddescr_index_in`).
    #[test]
    fn census_numbers_generic_option_variant_payload() {
        let _registry = crate::test_support::register_struct_origins_serialized(
            std::collections::HashMap::new(),
        );
        let owner = "option::Option<(usize,usize,bool)>::Some";
        let cc = layout_cc(&[(owner, &[("__pos_0", "(usize, usize, bool)")])]);
        assert!(
            get_fielddescr_index_in(&cc, owner, "__pos_0", 0) >= 0,
            "generic Option::Some payload is a numbered leaf"
        );
    }

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
