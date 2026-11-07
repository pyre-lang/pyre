//! `StoreThroughPtr` → `InteriorFieldWrite` from published Variable facts.
//!
//! `rewrite_fixed_newlist` / `do_fixed_newlist` re-derive `new_array` from
//! `result.concretetype`; this pass re-derives `setinteriorfield`
//! (`rewrite_op_setinteriorfield`) from a leaf-TO `SomeInteriorPtr`
//! annotation plus the `Ptr(Interior)` words (`getinteriorfieldargs`).
//! `InteriorPtrRepr.rtype_store_through_ptr` already did that expansion on
//! the flowspace copy; jtransform walks the model graph, so the facts are
//! read here — the same call site and ordering as
//! `rewrite_fixed_list_to_vec`. A leftover `StoreThroughPtr` or
//! `InteriorPtrExpose` is a lowering error.

use std::collections::{HashMap, HashSet};

use crate::annotator::model::SomeValue;
use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{
    ConcreteType, ExitSwitch, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation,
    ValueType,
};
use crate::translator::rtyper::flowspace_adapter::LegacyToTyped;
use crate::translator::rtyper::llannotation::SomeInteriorPtr;
use crate::translator::rtyper::lltypesystem::lltype::{
    GcKind, InteriorOffset, LowLevelType, Ptr, PtrTarget, Struct,
};
use crate::translator::rtyper::rmodel::ptr_from_container_lowleveltype;

fn interior_ptr_annotation(map: &LegacyToTyped, var: &Variable) -> Option<SomeInteriorPtr> {
    for src in map.get(var).into_iter().chain(std::iter::once(var)) {
        let cell = src.annotation.borrow();
        if let Some(SomeValue::InteriorPtr(ip)) = cell.as_ref().map(|rc| rc.as_ref()) {
            return Some(ip.clone());
        }
    }
    None
}

/// `InteriorPtrRepr.getinteriorfieldargs` walks every offset in order
/// (`None` = Index unpacked from `{ptr,index}`, `Constant` = Field name).
/// `rewrite_op_setinteriorfield` then asserts the store is a GcArray of
/// Struct plus one field: `offsets == [Index(_), Field(name)]` and `name`
/// is a field of that struct.
fn setinteriorfield_field_name(
    parent: &LowLevelType,
    offsets: &[InteriorOffset],
) -> Result<String, String> {
    let struct_t = gcarray_of_struct(parent).ok_or_else(|| {
        format!(
            "PARENTTYPE is not a GcArray of Struct (rewrite_op_setinteriorfield); parent={parent:?} offsets={offsets:?}"
        )
    })?;
    let mut seen_index = false;
    let mut field_name: Option<String> = None;
    for offset in offsets {
        match offset {
            InteriorOffset::Index(_) if !seen_index && field_name.is_none() => {
                seen_index = true;
            }
            InteriorOffset::Field(name) if seen_index && field_name.is_none() => {
                if struct_t.getattr_field_type(name).is_none() {
                    return Err(format!(
                        "struct {} has no field {name:?} (rewrite_op_setinteriorfield); offsets={offsets:?}",
                        struct_t._name
                    ));
                }
                field_name = Some(name.clone());
            }
            _ => {
                return Err(format!(
                    "offsets {offsets:?} is not [Index(_), Field(name)] (rewrite_op_setinteriorfield)"
                ));
            }
        }
    }
    match (seen_index, field_name) {
        (true, Some(name)) => Ok(name),
        _ => Err(format!(
            "offsets {offsets:?} is not [Index(_), Field(name)] (rewrite_op_setinteriorfield)"
        )),
    }
}

fn gcarray_of_struct(parent: &LowLevelType) -> Option<&Struct> {
    let LowLevelType::Array(arr) = parent else {
        return None;
    };
    if arr._gckind != GcKind::Gc {
        return None;
    }
    match &arr.OF {
        LowLevelType::Struct(st) => Some(st.as_ref()),
        _ => None,
    }
}

fn array_type_id_from_parent(parent: &LowLevelType) -> Option<String> {
    match parent {
        LowLevelType::Array(arr) => Some(crate::codewriter::call::array_type_id_of_lltype(arr)),
        LowLevelType::Ptr(ptr) => match &ptr.TO {
            PtrTarget::Array(arr) => Some(crate::codewriter::call::array_type_id_of_lltype(arr)),
            _ => None,
        },
        _ => None,
    }
}

fn value_type_from_concrete(kind: ConcreteType) -> ValueType {
    match kind {
        ConcreteType::Signed | ConcreteType::Unknown => ValueType::Int,
        ConcreteType::GcRef => ValueType::Ref(None),
        ConcreteType::Float => ValueType::Float,
        ConcreteType::Void => ValueType::Void,
    }
}

fn item_ty_of(value: &Variable, ip: &SomeInteriorPtr) -> ValueType {
    match FunctionGraph::concretetype_of(value) {
        ConcreteType::Unknown | ConcreteType::Void => {
            crate::model::try_getkind(&ip.ll_ptrtype.TO, true, true, true)
                .map(value_type_from_concrete)
                .unwrap_or(ValueType::Ref(None))
        }
        kind => value_type_from_concrete(kind),
    }
}

fn interior_getitem_item_type(ip: &SomeInteriorPtr) -> Option<LowLevelType> {
    if !ip
        .ll_ptrtype
        .offsets
        .iter()
        .any(|offset| matches!(offset, InteriorOffset::Index(_)))
    {
        return None;
    }
    if ip.ll_ptrtype.TO.is_container_type() {
        return Some((*ip.ll_ptrtype.TO).clone());
    }
    match ip.ll_ptrtype.PARENTTYPE.as_ref() {
        LowLevelType::Array(arr) if arr.OF.is_container_type() => Some(arr.OF.clone()),
        LowLevelType::FixedSizeArray(arr) if arr.OF.is_container_type() => Some(arr.OF.clone()),
        _ => None,
    }
}

fn interior_ptr_field(name: &str) -> bool {
    name == "ptr" || name == "index"
}

/// `rptr.py` `pairtype(PtrRepr, IntegerRepr).rtype_getitem`.
fn interior_getitem_malloc_ops(
    result: Variable,
    array: Variable,
    index: Variable,
    item_ty: &LowLevelType,
    ip: &SomeInteriorPtr,
) -> Vec<SpaceOperation> {
    let parent = ptr_from_container_lowleveltype(&ip.ll_ptrtype);
    let interior_struct = parent._interior_ptr_type_with_index(item_ty);
    result.set_concretetype(Some(LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(interior_struct),
    }))));
    vec![
        SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::New {
                owner: "Interior".into(),
            },
        },
        SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: result.clone(),
                field: FieldDescriptor::new("ptr", Some("Interior".into())),
                value: LinkArg::Value(array),
                ty: ValueType::Ref(None),
            },
        },
        SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: result,
                field: FieldDescriptor::new("index", Some("Interior".into())),
                value: LinkArg::Value(index),
                ty: ValueType::Int,
            },
        },
    ]
}

fn array_read_interior_annotation(
    graph: &FunctionGraph,
    value_to_var: &LegacyToTyped,
    array_read_result: &Variable,
) -> Option<SomeInteriorPtr> {
    if let Some(ip) = interior_ptr_annotation(value_to_var, array_read_result)
        && interior_getitem_item_type(&ip).is_some()
    {
        return Some(ip);
    }
    for (_, op) in graph.iter_block_ops() {
        let OpKind::InteriorPtrExpose { base, .. } = &op.kind else {
            continue;
        };
        if base != array_read_result {
            continue;
        }
        let Some(exposed) = op.result.as_ref() else {
            continue;
        };
        if let Some(ip) = interior_ptr_annotation(value_to_var, exposed)
            && interior_getitem_item_type(&ip).is_some()
        {
            return Some(ip);
        }
    }
    None
}

fn rewrite_interior_getitem_producers(graph: &mut FunctionGraph, value_to_var: &LegacyToTyped) {
    let mut rewrite: Vec<(Variable, SomeInteriorPtr)> = Vec::new();
    for (_, op) in graph.iter_block_ops() {
        let (OpKind::ArrayRead { .. }, Some(result)) = (&op.kind, op.result.as_ref()) else {
            continue;
        };
        if let Some(ip) = array_read_interior_annotation(graph, value_to_var, result) {
            rewrite.push((result.clone(), ip));
        }
    }
    if rewrite.is_empty() {
        return;
    }
    let block_ids: Vec<_> = graph.iter_blocks().map(|b| b.id).collect();
    for id in block_ids {
        let ops = std::mem::take(&mut graph.block_mut(id).operations);
        let mut new_ops = Vec::with_capacity(ops.len());
        for op in ops {
            match (&op.kind, op.result.as_ref()) {
                (OpKind::ArrayRead { base, index, .. }, Some(result)) => {
                    if let Some((_, ip)) = rewrite.iter().find(|(var, _)| var == result)
                        && let Some(item_ty) = interior_getitem_item_type(ip)
                    {
                        new_ops.extend(interior_getitem_malloc_ops(
                            result.clone(),
                            base.clone(),
                            index.clone(),
                            &item_ty,
                            ip,
                        ));
                        continue;
                    }
                    new_ops.push(op);
                }
                _ => new_ops.push(op),
            }
        }
        graph.block_mut(id).operations = new_ops;
    }
}

fn rewrite_one_store(
    graph: &mut FunctionGraph,
    ptr: &Variable,
    value: &Variable,
    value_to_var: &LegacyToTyped,
) -> Result<Vec<SpaceOperation>, String> {
    let Some(ip) = interior_ptr_annotation(value_to_var, ptr) else {
        return Err("leftover StoreThroughPtr without a leaf-TO SomeInteriorPtr annotation".into());
    };
    if ip.ll_ptrtype.TO.is_container_type() {
        return Err(format!(
            "store_through_ptr requires a leaf-TO interior pointer, got {:?}",
            ip.ll_ptrtype.TO
        ));
    }
    // `rewrite_op_setinteriorfield` drops a Void value before the
    // GcArray-of-Struct assert.
    if FunctionGraph::concretetype_of(value) == ConcreteType::Void {
        return Ok(Vec::new());
    }
    let field = setinteriorfield_field_name(&ip.ll_ptrtype.PARENTTYPE, &ip.ll_ptrtype.offsets)?;
    let item_ty = item_ty_of(value, &ip);
    let array_type_id = array_type_id_from_parent(&ip.ll_ptrtype.PARENTTYPE);
    // `Ptr._interior_ptr_type_with_index` is the `{ptr, index}` GcStruct;
    // `InteriorPtrRepr.getinteriorfieldargs` unpacks those fields.
    let v_ptr = graph.alloc_value_var_with_type(ConcreteType::GcRef);
    let v_index = graph.alloc_value_var_with_type(ConcreteType::Signed);
    Ok(vec![
        SpaceOperation {
            result: Some(v_ptr.clone()),
            kind: OpKind::FieldRead {
                base: ptr.clone(),
                field: FieldDescriptor::new("ptr", Some("Interior".into())),
                ty: ValueType::Ref(None),
                pure: true,
            },
        },
        SpaceOperation {
            result: Some(v_index.clone()),
            kind: OpKind::FieldRead {
                base: ptr.clone(),
                field: FieldDescriptor::new("index", Some("Interior".into())),
                ty: ValueType::Int,
                pure: true,
            },
        },
        SpaceOperation {
            result: None,
            kind: OpKind::InteriorFieldWrite {
                base: v_ptr,
                index: v_index,
                field: FieldDescriptor::new(field, None),
                value: value.clone(),
                item_ty,
                array_type_id,
            },
        },
    ])
}

fn rewrite_block_stores(graph: &mut FunctionGraph, value_to_var: &LegacyToTyped) {
    let block_ids: Vec<_> = graph.iter_blocks().map(|b| b.id).collect();
    for id in block_ids {
        let ops = std::mem::take(&mut graph.block_mut(id).operations);
        let mut new_ops = Vec::with_capacity(ops.len());
        for op in ops {
            match &op.kind {
                OpKind::StoreThroughPtr { ptr, value } => {
                    match rewrite_one_store(graph, ptr, value, value_to_var) {
                        Ok(rewritten) => new_ops.extend(rewritten),
                        Err(msg) => {
                            panic!("rewrite_op_setinteriorfield: graph {} {msg}", graph.name)
                        }
                    }
                }
                _ => new_ops.push(op),
            }
        }
        graph.block_mut(id).operations = new_ops;
    }
}

fn concretetype_is_interior_struct(var: &Variable) -> bool {
    match var.concretetype() {
        Some(LowLevelType::Ptr(ptr)) => match &ptr.TO {
            PtrTarget::Struct(struct_t) => {
                struct_t._name == "Interior"
                    && matches!(
                        struct_t._hints.get("interior_ptr_type"),
                        Some(ConstValue::Bool(true))
                    )
            }
            _ => false,
        },
        _ => false,
    }
}

fn producer_is_interior_object(graph: &FunctionGraph, var: &Variable) -> bool {
    for (_, op) in graph.iter_block_ops() {
        if op.result.as_ref() != Some(var) {
            continue;
        }
        return match &op.kind {
            OpKind::New { owner } if owner == "Interior" => true,
            OpKind::UnaryOp {
                op: name, operand, ..
            } if name == "same_as" => producer_is_interior_object(graph, operand),
            OpKind::ArrayRead { .. } => false,
            _ => concretetype_is_interior_struct(var),
        };
    }
    concretetype_is_interior_struct(var)
}

fn identity_exposes_after_stores(graph: &mut FunctionGraph) {
    let mut to_identity: Vec<(crate::model::BlockId, usize)> = Vec::new();
    for (block_id, ops) in graph
        .iter_blocks()
        .map(|b| (b.id, b.operations.clone()))
        .collect::<Vec<_>>()
    {
        for (index, op) in ops.iter().enumerate() {
            let OpKind::InteriorPtrExpose { base, .. } = &op.kind else {
                continue;
            };
            if !producer_is_interior_object(graph, base) {
                panic!(
                    "InteriorPtrExpose of a non-Interior base in graph {}; \
                     PtrRepr.rtype_getitem must malloc the Interior {{ptr,index}} object first",
                    graph.name
                );
            }
            to_identity.push((block_id, index));
        }
    }
    for (id, index) in to_identity {
        let op = &mut graph.block_mut(id).operations[index];
        let OpKind::InteriorPtrExpose { base, ty, .. } = &op.kind else {
            continue;
        };
        // `InteriorPtrRepr.rtype_interior_ptr_expose` is identity of the
        // `{ptr, index}` words; the base is the Interior object.
        let operand = base.clone();
        let result_ty = ty.clone();
        op.kind = OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand,
            result_ty,
        };
    }
}

fn reject_leftover_interior_ops(graph: &FunctionGraph) {
    for (_, op) in graph.iter_block_ops() {
        match &op.kind {
            OpKind::StoreThroughPtr { .. } => panic!(
                "leftover StoreThroughPtr in graph {}; the rtyper must lower \
                 this store-through to setinteriorfield \
                 (InteriorPtrRepr.rtype_store_through_ptr)",
                graph.name
            ),
            OpKind::InteriorPtrExpose { .. } => panic!(
                "leftover InteriorPtrExpose in graph {}; the rtyper must lower \
                 _interior_ptr._expose before codewriter emission (InteriorPtrRepr)",
                graph.name
            ),
            _ => {}
        }
    }
}

fn remap_graph_uses(graph: &mut FunctionGraph, remap: &impl Fn(&Variable) -> Variable) {
    let block_ids: Vec<_> = graph.iter_blocks().map(|b| b.id).collect();
    for id in block_ids {
        let block = graph.block_mut(id);
        for op in &mut block.operations {
            op.kind = crate::inline::remap_op_kind(&op.kind, remap);
        }
        match &mut block.exitswitch {
            Some(ExitSwitch::Value(var)) => *var = remap(var),
            Some(ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    *arg = remap(arg);
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
        for link in &mut block.exits {
            for arg in &mut link.args {
                *arg = arg.map_value(remap);
            }
            if let Some(arg) = &mut link.last_exception {
                *arg = arg.map_value(remap);
            }
            if let Some(arg) = &mut link.last_exc_value {
                *arg = arg.map_value(remap);
            }
        }
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn resolve_alias(var: &Variable, aliases: &HashMap<Variable, Variable>) -> Variable {
    let mut cur = var.clone();
    let mut seen = HashSet::new();
    while let Some(next) = aliases.get(&cur) {
        if !seen.insert(cur.clone()) {
            break;
        }
        cur = next.clone();
    }
    cur
}

/// `removenoops.remove_same_as`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn remove_same_as_ops(graph: &mut FunctionGraph) {
    let mut aliases: HashMap<Variable, Variable> = HashMap::new();
    for (_, op) in graph.iter_block_ops() {
        let OpKind::UnaryOp {
            op: name, operand, ..
        } = &op.kind
        else {
            continue;
        };
        if name != "same_as" {
            continue;
        }
        if let Some(result) = op.result.as_ref() {
            aliases.insert(result.clone(), operand.clone());
        }
    }
    if aliases.is_empty() {
        return;
    }
    remap_graph_uses(graph, &|var| resolve_alias(var, &aliases));
    let block_ids: Vec<_> = graph.iter_blocks().map(|b| b.id).collect();
    for id in block_ids {
        graph.block_mut(id).operations.retain(|op| {
            !matches!(
                &op.kind,
                OpKind::UnaryOp { op: name, .. } if name == "same_as"
            )
        });
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn interior_malloc_results(graph: &FunctionGraph) -> HashSet<Variable> {
    graph
        .iter_block_ops()
        .filter_map(|(_, op)| match (&op.kind, op.result.as_ref()) {
            (OpKind::New { owner }, Some(result)) if owner == "Interior" => Some(result.clone()),
            _ => None,
        })
        .collect()
}

/// Forward `getfield` of Interior.ptr / Interior.index to the matching
/// `setfield` (`jtransform.py` `getinteriorfield` / `setinteriorfield`
/// read the array and index directly). Then
/// `remove_simple_mallocs` / `remove_dead_aggregates` and
/// `simplify.transform_dead_op_vars` (`prune_dead_phis`).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn forward_interior_field_reads(graph: &mut FunctionGraph) {
    let interiors = interior_malloc_results(graph);
    if interiors.is_empty() {
        return;
    }
    let mut stored: HashMap<(Variable, String), Variable> = HashMap::new();
    for (_, op) in graph.iter_block_ops() {
        let OpKind::FieldWrite {
            base, field, value, ..
        } = &op.kind
        else {
            continue;
        };
        if !interiors.contains(base) || !interior_ptr_field(&field.name) {
            continue;
        }
        if field.owner_root.as_deref() != Some("Interior") {
            continue;
        }
        if let Some(var) = value.as_variable() {
            stored.insert((base.clone(), field.name.clone()), var.clone());
        }
    }
    let mut aliases: HashMap<Variable, Variable> = HashMap::new();
    for (_, op) in graph.iter_block_ops() {
        let (OpKind::FieldRead { base, field, .. }, Some(result)) = (&op.kind, op.result.as_ref())
        else {
            continue;
        };
        if field.owner_root.as_deref() != Some("Interior") || !interior_ptr_field(&field.name) {
            continue;
        }
        if let Some(src) = stored.get(&(base.clone(), field.name.clone())) {
            aliases.insert(result.clone(), src.clone());
        }
    }
    if aliases.is_empty() {
        return;
    }
    remap_graph_uses(graph, &|var| resolve_alias(var, &aliases));
    let forwarded: HashSet<Variable> = aliases.keys().cloned().collect();
    let block_ids: Vec<_> = graph.iter_blocks().map(|b| b.id).collect();
    for id in block_ids {
        graph.block_mut(id).operations.retain(|op| {
            op.result
                .as_ref()
                .is_none_or(|result| !forwarded.contains(result))
        });
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn drop_dead_interior_mallocs(graph: &mut FunctionGraph) {
    let interiors = interior_malloc_results(graph);
    if interiors.is_empty() {
        return;
    }
    let mut real_read: HashSet<Variable> = HashSet::new();
    for block in graph.iter_blocks() {
        for op in &block.operations {
            if let OpKind::FieldWrite { base, value, .. } = &op.kind
                && interiors.contains(base)
            {
                if let Some(var) = value.as_variable() {
                    real_read.insert(var.clone());
                }
                continue;
            }
            for var in crate::inline::op_variable_refs(&op.kind) {
                real_read.insert(var);
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) => {
                real_read.insert(var.clone());
            }
            Some(ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    real_read.insert(arg.clone());
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
        for link in &block.exits {
            for arg in &link.args {
                if let Some(var) = arg.as_variable() {
                    real_read.insert(var.clone());
                }
            }
            if let Some(arg) = &link.last_exception
                && let Some(var) = arg.as_variable()
            {
                real_read.insert(var.clone());
            }
            if let Some(arg) = &link.last_exc_value
                && let Some(var) = arg.as_variable()
            {
                real_read.insert(var.clone());
            }
        }
        if block.exits.is_empty() {
            for iarg in &block.inputargs {
                real_read.insert(iarg.clone());
            }
        }
    }
    let dead: HashSet<Variable> = interiors
        .into_iter()
        .filter(|var| !real_read.contains(var))
        .collect();
    if dead.is_empty() {
        return;
    }
    let block_ids: Vec<_> = graph.iter_blocks().map(|b| b.id).collect();
    for id in block_ids {
        graph.block_mut(id).operations.retain(|op| {
            if op
                .result
                .as_ref()
                .is_some_and(|result| dead.contains(result))
            {
                return false;
            }
            if let OpKind::FieldWrite { base, .. } = &op.kind
                && dead.contains(base)
            {
                return false;
            }
            true
        });
    }
}

fn cleanup_virtual_interior_mallocs(graph: &mut FunctionGraph) {
    remove_same_as_ops(graph);
    forward_interior_field_reads(graph);
    drop_dead_interior_mallocs(graph);
}

/// Replace Interior-typed element `getitem` with malloc + `setfield`
/// `ptr` / `index` (`rptr.py` `rtype_getitem`), rewrite each
/// `StoreThroughPtr` whose `ptr` publishes a leaf-TO `SomeInteriorPtr`,
/// then identity-rename `InteriorPtrExpose` of an Interior object.
pub(crate) fn rewrite_store_through_ptr(graph: &mut FunctionGraph, value_to_var: &LegacyToTyped) {
    rewrite_interior_getitem_producers(graph, value_to_var);
    rewrite_block_stores(graph, value_to_var);
    identity_exposes_after_stores(graph);
    reject_leftover_interior_ops(graph);
    cleanup_virtual_interior_mallocs(graph);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::rc::Rc;

    use crate::annotator::model::SomeValue;
    use crate::model::ValueType;
    use crate::translator::rtyper::lltypesystem::lltype::{
        Array, GCREF, InteriorOffset, InteriorPtr, Struct,
    };

    fn gcarray_of_entry(fields: Vec<(String, LowLevelType)>) -> LowLevelType {
        let entry_struct = Struct::new("Entry", fields);
        LowLevelType::Array(Box::new(Array::gc(LowLevelType::Struct(Box::new(
            entry_struct,
        )))))
    }

    fn leaf_interior_ptr_annotation() -> SomeValue {
        SomeValue::InteriorPtr(SomeInteriorPtr::new(InteriorPtr {
            PARENTTYPE: Box::new(gcarray_of_entry(vec![("value".into(), GCREF.clone())])),
            TO: Box::new(GCREF.clone()),
            offsets: vec![
                InteriorOffset::Index(0),
                InteriorOffset::Field("value".into()),
            ],
        }))
    }

    fn store_through_graph(annotation: Option<SomeValue>) -> (FunctionGraph, Variable, Variable) {
        let mut graph = FunctionGraph::new("caller_store");
        let entry = graph.startblock;
        let ptr = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "p".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let value = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "v".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, ptr.clone());
        graph.push_inputarg_var(entry, value.clone());
        graph.push_op_var(
            entry,
            OpKind::StoreThroughPtr {
                ptr: ptr.clone(),
                value: value.clone(),
            },
            false,
        );
        graph.set_return(entry, None);
        if let Some(annotation) = annotation {
            *ptr.annotation.borrow_mut() = Some(Rc::new(annotation));
        }
        (graph, ptr, value)
    }

    fn rewrite_annotated(annotation: SomeValue) {
        let (mut graph, ptr, _value) = store_through_graph(Some(annotation));
        let mut value_to_var = HashMap::new();
        value_to_var.insert(ptr.clone(), ptr.clone());
        rewrite_store_through_ptr(&mut graph, &value_to_var);
    }

    #[test]
    fn rewrite_store_through_ptr_from_leaf_someinteriorptr_emits_interior_field_write() {
        let (mut graph, ptr, _value) = store_through_graph(Some(leaf_interior_ptr_annotation()));
        let mut value_to_var = HashMap::new();
        value_to_var.insert(ptr.clone(), ptr.clone());
        rewrite_store_through_ptr(&mut graph, &value_to_var);
        let writes: Vec<_> = graph
            .iter_block_ops()
            .filter_map(|(_, op)| match &op.kind {
                OpKind::InteriorFieldWrite { item_ty, field, .. } => {
                    Some((item_ty.clone(), field.name.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(writes, vec![(ValueType::Ref(None), "value".to_string())]);
        assert!(
            graph.iter_block_ops().any(|(_, op)| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "ptr"
            )) && graph.iter_block_ops().any(|(_, op)| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "index"
            )),
            "unpack Interior.ptr / Interior.index; ops={:?}",
            graph
                .iter_block_ops()
                .map(|(_, op)| format!("{:?}", op.kind))
                .collect::<Vec<_>>()
        );
        assert!(graph.iter_block_ops().all(|(_, op)| !matches!(
            op.kind,
            OpKind::StoreThroughPtr { .. } | OpKind::RawStore { .. }
        )));
    }

    #[test]
    #[should_panic(expected = "rewrite_op_setinteriorfield")]
    fn rewrite_store_through_ptr_nested_extra_field_fails_lowering() {
        let nested = Struct::new("Nested", vec![("value".into(), GCREF.clone())]);
        rewrite_annotated(SomeValue::InteriorPtr(SomeInteriorPtr::new(InteriorPtr {
            PARENTTYPE: Box::new(gcarray_of_entry(vec![(
                "nested".into(),
                LowLevelType::Struct(Box::new(nested)),
            )])),
            TO: Box::new(GCREF.clone()),
            offsets: vec![
                InteriorOffset::Index(0),
                InteriorOffset::Field("nested".into()),
                InteriorOffset::Field("value".into()),
            ],
        })));
    }

    #[test]
    #[should_panic(expected = "rewrite_op_setinteriorfield")]
    fn rewrite_store_through_ptr_field_only_offsets_fails_lowering() {
        rewrite_annotated(SomeValue::InteriorPtr(SomeInteriorPtr::new(InteriorPtr {
            PARENTTYPE: Box::new(gcarray_of_entry(vec![("value".into(), GCREF.clone())])),
            TO: Box::new(GCREF.clone()),
            offsets: vec![InteriorOffset::Field("value".into())],
        })));
    }

    fn container_interior_ptr_annotation() -> SomeValue {
        let entry_struct = Struct::new("Entry", vec![("value".into(), GCREF.clone())]);
        SomeValue::InteriorPtr(SomeInteriorPtr::new(InteriorPtr {
            PARENTTYPE: Box::new(gcarray_of_entry(vec![("value".into(), GCREF.clone())])),
            TO: Box::new(LowLevelType::Struct(Box::new(entry_struct))),
            offsets: vec![InteriorOffset::Index(0)],
        }))
    }

    /// Production `codewriter.rs` path: `rewrite_store_through_ptr` on the
    /// cloned model graph (`get_slot_mut` ArrayRead + expose + store).
    #[test]
    fn production_get_slot_mut_rewrites_array_read_to_setinteriorfield_on_array() {
        use crate::model::{ArrayGcKind, FieldDescriptor};

        let mut graph = FunctionGraph::new("get_slot_mut_store");
        let entry = graph.startblock;
        let entries = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "entries".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let index = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "i".to_string(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let value = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "v".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, entries.clone());
        graph.push_inputarg_var(entry, index.clone());
        graph.push_inputarg_var(entry, value.clone());
        let elem = graph
            .push_op_var(
                entry,
                OpKind::ArrayRead {
                    base: entries.clone(),
                    index: index.clone(),
                    item_ty: ValueType::Ref(None),
                    array_type_id: None,
                    nolength: false,
                    gckind: ArrayGcKind::Gc,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let exposed = graph
            .push_op_var(
                entry,
                OpKind::InteriorPtrExpose {
                    base: elem.clone(),
                    field: FieldDescriptor::new("value", Some("Entry".into())),
                    ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::StoreThroughPtr {
                ptr: exposed.clone(),
                value: value.clone(),
            },
            false,
        );
        graph.set_return(entry, None);
        *elem.annotation.borrow_mut() = Some(Rc::new(container_interior_ptr_annotation()));
        *exposed.annotation.borrow_mut() = Some(Rc::new(leaf_interior_ptr_annotation()));

        let mut value_to_var = HashMap::new();
        value_to_var.insert(elem.clone(), elem.clone());
        value_to_var.insert(exposed.clone(), exposed.clone());
        rewrite_store_through_ptr(&mut graph, &value_to_var);

        let ops: Vec<_> = graph
            .iter_block_ops()
            .map(|(_, op)| format!("{:?}", op.kind))
            .collect();
        assert!(
            graph
                .iter_block_ops()
                .all(|(_, op)| !matches!(op.kind, OpKind::ArrayRead { .. })),
            "element getitem must not remain ArrayRead; ops={ops:?}"
        );
        let writes: Vec<_> = graph
            .iter_block_ops()
            .filter_map(|(_, op)| match &op.kind {
                OpKind::InteriorFieldWrite {
                    base,
                    index: idx,
                    field,
                    ..
                } => Some((base.clone(), idx.clone(), field.name.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(writes.len(), 1, "one InteriorFieldWrite; ops={ops:?}");
        assert_eq!(writes[0].0, entries, "setinteriorfield base is the array");
        assert_eq!(writes[0].1, index, "setinteriorfield index is the index");
        assert_eq!(writes[0].2, "value");
        assert!(
            graph.iter_block_ops().all(|(_, op)| !matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "ptr" && field.owner_root.as_deref() == Some("Interior")
            )),
            "no FieldRead Interior.ptr from an ArrayRead GCREF; ops={ops:?}"
        );
        assert!(
            graph.iter_block_ops().all(|(_, op)| !matches!(
                op.kind,
                OpKind::StoreThroughPtr { .. }
                    | OpKind::InteriorPtrExpose { .. }
                    | OpKind::New { .. }
            )),
            "no leftover store/expose/Interior malloc; ops={ops:?}"
        );
    }
}
