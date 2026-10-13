//! Keep the metadata word of a fat `&dyn` / `Box<dyn>` across a call.
//!
//! `FunctionGraph::set_return` carries one value, and `jtransform.py`
//! rewrites `direct_call` to `inline_call_*` so the meta-interpreter
//! descends into the callee jitcode at runtime. A `&dyn` is two words
//! (data, vtable). The one-word return keeps the data pointer and drops
//! the vtable, so a later `method_*` read on `{vtable}` loads the slot
//! from the data pointer.
//!
//! RPython never hits that shape: the trace-time `inline_call_*` descent
//! sees the callee body, where both words are still field reads. This
//! pass does the same when that body is in this LLBC: splice it
//! (`inline::splice_direct_call`), then retarget each `method_*` read so
//! its base is the metadata word (`FatMeta`) of the fat field. The data
//! word stays the call's receiver. `dont_look_inside` and a splice cycle
//! stay residual. A cross-crate fat return has no body here and is left
//! unchanged. A size cutoff is not a substitute: the one-word return
//! would still drop the vtable, and the later `method_*` read would load
//! the slot from the data pointer.

use std::collections::{HashMap, HashSet};

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::TypeDeclKind;

use crate::flowspace::model::Variable;
use crate::front::mir::tyref_is_fat_dyn;
use crate::inline::splice_direct_call;
use crate::model::{
    BlockId, ConcreteType, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation,
    ValueType, VecFieldPart,
};

thread_local! {
    static SPLICE_STACK: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

struct SpliceGuard(u64);

impl Drop for SpliceGuard {
    fn drop(&mut self) {
        SPLICE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some(pos) = stack.iter().rposition(|id| *id == self.0) {
                stack.remove(pos);
            }
        });
    }
}

pub(crate) fn splice_in_progress(def_id: u64) -> bool {
    SPLICE_STACK.with(|stack| stack.borrow().contains(&def_id))
}

fn enter_splice(def_id: u64) -> Option<SpliceGuard> {
    if splice_in_progress(def_id) {
        return None;
    }
    SPLICE_STACK.with(|stack| stack.borrow_mut().push(def_id));
    Some(SpliceGuard(def_id))
}

struct DynCallSite {
    block: BlockId,
    op_index: usize,
    def_id: u64,
}

/// Splice recorded fat-dyn calls, then point `{vtable}` `method_*` reads
/// at the metadata word. `lower_callee` returns the callee graph already
/// run through this pass, or `None` when the body must stay a call.
pub(crate) fn splice_fat_dyn_returns(
    graph: &mut FunctionGraph,
    llbc: &Llbc,
    fun_id: u64,
    dyn_results: &[Variable],
    lower_callee: &mut dyn FnMut(u64) -> Option<FunctionGraph>,
) {
    // Most graphs never form a fat-dyn value. Skip the type-decl walk
    // unless this body recorded a fat return or already reads a vtable
    // slot (a same-function `method_*` whose base is still the data word).
    if dyn_results.is_empty() && !graph_has_vtable_method(graph) {
        return;
    }
    let Some(_guard) = enter_splice(fun_id) else {
        return;
    };
    let sites = fat_dyn_call_sites(graph, dyn_results);
    let mut prepared = Vec::new();
    for site in &sites {
        if splice_in_progress(site.def_id) {
            continue;
        }
        if let Some(callee) = lower_callee(site.def_id) {
            prepared.push((site.block, site.op_index, callee));
        }
    }
    let mut spliced = 0usize;
    for (block, op_index, callee) in prepared.into_iter().rev() {
        if splice_direct_call(graph, block, op_index, callee) {
            spliced += 1;
        }
    }
    let retargeted = retarget_vtable_method_bases(graph, llbc);
    if spliced > 0 || retargeted > 0 {
        crate::model::clear_unreachable_blocks(graph);
        crate::model::prune_dead_phis(graph);
    }
}

fn graph_has_vtable_method(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::FieldRead { field, .. } => is_vtable_method(field),
            _ => false,
        })
    })
}

fn fat_dyn_call_sites(graph: &FunctionGraph, dyn_results: &[Variable]) -> Vec<DynCallSite> {
    let ids: HashSet<u64> = dyn_results.iter().map(Variable::id).collect();
    let mut sites = Vec::new();
    for block in &graph.blocks {
        for (op_index, op) in block.operations.iter().enumerate() {
            let Some(result) = op.result.as_ref() else {
                continue;
            };
            if !ids.contains(&result.id()) {
                continue;
            }
            let OpKind::Call { target, .. } = &op.kind else {
                continue;
            };
            let Some(def_id) = target.fun_decl_id() else {
                continue;
            };
            sites.push(DynCallSite {
                block: block.id,
                op_index,
                def_id,
            });
        }
    }
    sites
}

fn cast_instance_operand(kind: &OpKind) -> Option<Variable> {
    crate::model::cast_instance_root(kind)?;
    let OpKind::Call { args, .. } = kind else {
        return None;
    };
    args.first().and_then(LinkArg::as_variable).cloned()
}

fn is_vtable_method(field: &FieldDescriptor) -> bool {
    field.name.starts_with("method_")
        && field
            .owner_root
            .as_deref()
            .is_some_and(|owner| owner == "{vtable}" || owner.ends_with("::{vtable}"))
}

/// How tightly `owner_root` names a declaration path.
///
/// `Exact` is the path or its crate-stripped spelling. `Suffix` is a
/// qualified owner (`module::Type`) that is a proper suffix of the
/// declaration. `Leaf` is a bare name (`Inner` against `a::Inner` and
/// `b::Inner`) and is not an identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnerPathRank {
    Exact,
    Suffix,
    Leaf,
}

fn owner_path_rank(decl_path: &str, owner_root: &str) -> Option<OwnerPathRank> {
    let stripped = crate::front::mir::strip_crate_prefix(decl_path);
    if decl_path == owner_root || stripped == owner_root {
        return Some(OwnerPathRank::Exact);
    }
    if owner_root.contains("::")
        && (decl_path.ends_with(&format!("::{owner_root}"))
            || stripped.ends_with(&format!("::{owner_root}")))
    {
        return Some(OwnerPathRank::Suffix);
    }
    let decl_leaf = decl_path.rsplit("::").next().unwrap_or(decl_path);
    let owner_leaf = owner_root.rsplit("::").next().unwrap_or(owner_root);
    let owner_leaf = owner_leaf.split('<').next().unwrap_or(owner_leaf);
    if owner_root == decl_leaf
        || owner_leaf == decl_leaf
        || decl_path.ends_with(&format!("::{owner_root}"))
        || stripped.ends_with(&format!("::{owner_root}"))
    {
        return Some(OwnerPathRank::Leaf);
    }
    None
}

/// Identities `resolve_adt_field` can mint for this declaration.
///
/// The field's `owner_id` is `StructId::from_canonical` of the
/// crate-stripped path, or of the full path when that stripped spelling
/// is a cross-crate tombstone. Generic arguments are not part of
/// `name_path`; a monomorphized id falls through to the path ranks.
fn decl_owner_ids(decl_path: &str) -> [majit_ir::descr::StructId; 4] {
    let stripped = crate::front::mir::strip_crate_prefix(decl_path);
    [
        majit_ir::descr::StructId::from_canonical(&stripped),
        majit_ir::descr::StructId::from_canonical(decl_path),
        majit_ir::descr::StructId::from_canonical_spelling(&stripped),
        majit_ir::descr::StructId::from_canonical_spelling(decl_path),
    ]
}

fn id_matches(owner_id: majit_ir::descr::StructId, decl_path: &str) -> bool {
    decl_owner_ids(decl_path).contains(&owner_id)
}

struct OwnerHit {
    id_hit: bool,
    rank: Option<OwnerPathRank>,
}

/// Declarations that count for one field.
///
/// An `owner_id` hit wins. When the id is set and matches nothing, only
/// an exact or qualified-suffix path counts — a shared leaf must not mark
/// the field fat. Without an id, the leaf rank is the last fallback, and
/// only when no exact or suffix hit exists.
fn choose_owner_hits(owner_id_set: bool, hits: &[OwnerHit]) -> Vec<usize> {
    let pick = |pred: fn(&OwnerHit) -> bool| -> Vec<usize> {
        hits.iter()
            .enumerate()
            .filter(|(_, hit)| pred(hit))
            .map(|(index, _)| index)
            .collect()
    };
    if owner_id_set {
        let ids = pick(|hit| hit.id_hit);
        if !ids.is_empty() {
            return ids;
        }
        let exact = pick(|hit| hit.rank == Some(OwnerPathRank::Exact));
        if !exact.is_empty() {
            return exact;
        }
        return pick(|hit| hit.rank == Some(OwnerPathRank::Suffix));
    }
    let exact = pick(|hit| hit.rank == Some(OwnerPathRank::Exact));
    if !exact.is_empty() {
        return exact;
    }
    let suffix = pick(|hit| hit.rank == Some(OwnerPathRank::Suffix));
    if !suffix.is_empty() {
        return suffix;
    }
    pick(|hit| hit.rank == Some(OwnerPathRank::Leaf))
}

fn field_is_fat_dyn(
    llbc: &Llbc,
    owner_id: Option<majit_ir::descr::StructId>,
    owner_root: &str,
    field_name: &str,
) -> bool {
    let mut hits = Vec::new();
    let mut fat = Vec::new();
    for decl in llbc.iter_type_decls() {
        if decl.published_struct_copy {
            continue;
        }
        let TypeDeclKind::Struct(fields) = &decl.kind else {
            continue;
        };
        let Some(field) = fields
            .iter()
            .find(|field| field.name.as_deref() == Some(field_name))
        else {
            continue;
        };
        let path = decl.item_meta.name_path();
        let id_hit = owner_id.is_some_and(|id| id_matches(id, &path));
        let rank = owner_path_rank(&path, owner_root);
        if !id_hit && rank.is_none() {
            continue;
        }
        hits.push(OwnerHit { id_hit, rank });
        fat.push(tyref_is_fat_dyn(&field.ty, llbc));
    }
    let chosen = choose_owner_hits(owner_id.is_some(), &hits);
    !chosen.is_empty() && chosen.into_iter().all(|index| fat[index])
}

#[derive(Clone)]
struct FatField {
    block: BlockId,
    var_id: u64,
    base: Variable,
    field: FieldDescriptor,
    pure: bool,
}

struct SameAs {
    block: BlockId,
    operand: Variable,
}

struct MetaEnv {
    fat_fields: HashMap<u64, FatField>,
    copies: HashMap<u64, SameAs>,
    fat_len: HashMap<u64, Variable>,
    memo: HashMap<(usize, u64), Variable>,
    visiting: HashSet<(usize, u64)>,
}

fn index_producers(
    graph: &FunctionGraph,
    llbc: &Llbc,
) -> (HashMap<u64, FatField>, HashMap<u64, SameAs>) {
    let mut fat_fields = HashMap::new();
    let mut copies = HashMap::new();
    let mut fat_cache: HashMap<(Option<majit_ir::descr::StructId>, String, String), bool> =
        HashMap::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(result) = op.result.as_ref() else {
                continue;
            };
            match &op.kind {
                OpKind::FieldRead {
                    base, field, pure, ..
                } if field.vec_part.is_none()
                    || field.vec_part == Some(VecFieldPart::FatVtable) =>
                {
                    // `DictStrategyRef.imp` is tagged FatVtable at lower
                    // (`tyref_is_dyn_pointer`) so heapcache can key the
                    // metadata word. That tag is the same fat field this
                    // pass used to see only while `vec_part` was still
                    // None; skip it and `method_*` stays on the `{vtable}`
                    // cast of the data pointer.
                    let is_fat = if field.vec_part == Some(VecFieldPart::FatVtable) {
                        true
                    } else {
                        let Some(owner) = field.owner_root.clone() else {
                            continue;
                        };
                        let name = field.name.clone();
                        let owner_id = field.owner_id;
                        let cache_key = (owner_id, owner.clone(), name.clone());
                        if let Some(known) = fat_cache.get(&cache_key) {
                            *known
                        } else {
                            let known = field_is_fat_dyn(llbc, owner_id, &owner, &name);
                            fat_cache.insert(cache_key, known);
                            known
                        }
                    };
                    if !is_fat {
                        continue;
                    }
                    fat_fields.entry(result.id()).or_insert(FatField {
                        block: block.id,
                        var_id: result.id(),
                        base: base.clone(),
                        field: field.clone(),
                        pure: *pure,
                    });
                }
                OpKind::UnaryOp {
                    op: name, operand, ..
                } if name == "same_as" => {
                    copies.entry(result.id()).or_insert(SameAs {
                        block: block.id,
                        operand: operand.clone(),
                    });
                }
                // `__cast_instance_intrinsic` is identity at jitcode
                // (`cast_pointer` → `same_as`). The vtable-slot base is that
                // cast of the data word, so follow it back to the fat field.
                OpKind::Call { .. } => {
                    let Some(operand) = cast_instance_operand(&op.kind) else {
                        continue;
                    };
                    copies.entry(result.id()).or_insert(SameAs {
                        block: block.id,
                        operand,
                    });
                }
                _ => {}
            }
        }
    }
    (fat_fields, copies)
}

fn retarget_vtable_method_bases(graph: &mut FunctionGraph, llbc: &Llbc) -> usize {
    let (fat_fields, copies) = index_producers(graph, llbc);
    let sites: Vec<(BlockId, u64, Variable)> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block.operations.iter().filter_map(|op| {
                let result = op.result.as_ref()?;
                let OpKind::FieldRead { base, field, .. } = &op.kind else {
                    return None;
                };
                if !is_vtable_method(field) {
                    return None;
                }
                Some((block.id, result.id(), base.clone()))
            })
        })
        .collect();
    let mut env = MetaEnv {
        fat_fields,
        copies,
        fat_len: HashMap::new(),
        memo: HashMap::new(),
        visiting: HashSet::new(),
    };
    let mut retargeted = 0usize;
    for (block, result_id, base) in sites {
        let Some(meta) = ensure_meta(graph, &mut env, block, &base) else {
            continue;
        };
        // `lookup_operand` requires a block inputarg or op result. The
        // metadata word is often a FatLen FieldRead in a predecessor, or
        // a merge phi of two such reads, and `ensure_meta` may reuse that
        // identity across a passthrough. Thread it the way
        // `ensure_variable_at_block` threads every other block-crossing
        // value (`flowspace/model.py` `checkgraph`).
        if !graph.ensure_variable_at_block(block, &meta) {
            continue;
        }
        let Some(op) = graph.blocks[block.0].operations.iter_mut().find(|op| {
            op.result
                .as_ref()
                .is_some_and(|result| result.id() == result_id)
        }) else {
            continue;
        };
        let OpKind::FieldRead { base, pure, .. } = &mut op.kind else {
            continue;
        };
        *base = meta;
        // The slot of a vtable does not change. The base is the metadata
        // word, an int (`raw` pointer); `getfield_raw_r` is emitted only
        // for a pure read (`jtransform.py` `rewrite_op_getfield`).
        *pure = true;
        retargeted += 1;
    }
    // A leftover `FatVtable` GetfieldGcR of the same slot as the
    // `FatLen` GetfieldGcI `make_equal_to`s across Box.type
    // (`imp.len` Int vs `imp.vtable` Ref).
    let fats: Vec<FatField> = env.fat_fields.values().cloned().collect();
    for fat in fats {
        let _ = emit_fat_len(graph, &mut env, &fat);
    }
    retargeted
}

fn ensure_meta(
    graph: &mut FunctionGraph,
    env: &mut MetaEnv,
    block: BlockId,
    var: &Variable,
) -> Option<Variable> {
    let key = (block.0, var.id());
    if let Some(found) = env.memo.get(&key) {
        return Some(found.clone());
    }
    if !env.visiting.insert(key) {
        return None;
    }
    let result = ensure_meta_inner(graph, env, block, var);
    env.visiting.remove(&key);
    if let Some(found) = result.clone() {
        env.memo.insert(key, found);
    }
    result
}

fn ensure_meta_inner(
    graph: &mut FunctionGraph,
    env: &mut MetaEnv,
    block: BlockId,
    var: &Variable,
) -> Option<Variable> {
    let fat = env.fat_fields.get(&var.id()).map(|fat| FatField {
        block: fat.block,
        var_id: fat.var_id,
        base: fat.base.clone(),
        field: fat.field.clone(),
        pure: fat.pure,
    });
    // The metadata word is an operand of the block that reads it
    // (`flowspace/model.py` `checkgraph`, `lookup_operand`). A fat
    // field or copy that lives in another block has to be carried in;
    // returning the producer's variable leaves a Signed base with no
    // definition here.
    if let Some(fat) = fat
        && fat.block == block
    {
        return emit_fat_len(graph, env, &fat);
    }
    let copy = env
        .copies
        .get(&var.id())
        .map(|copy| (copy.block, copy.operand.clone()));
    if let Some((copy_block, operand)) = copy
        && copy_block == block
    {
        return ensure_meta(graph, env, copy_block, &operand);
    }
    let inputarg_len = graph.blocks.get(block.0)?.inputargs.len();
    let index = graph.blocks[block.0]
        .inputargs
        .iter()
        .position(|arg| arg == var);
    let preds = graph.predecessors(block);
    if preds.is_empty() {
        return None;
    }
    let mut incoming = Vec::new();
    for pred in &preds {
        let exits = graph.blocks.get(pred.0)?.exits.clone();
        let mut saw = false;
        for (link_index, link) in exits.iter().enumerate() {
            if link.target != block {
                continue;
            }
            let src = if let Some(index) = index {
                if link.args.len() != inputarg_len {
                    return None;
                }
                link.args.get(index)?.as_variable()?.clone()
            } else {
                // Dominating use: the later block reads `var` without
                // taking it as an inputarg (`retarget_var_uses` after
                // a splice). Ask the predecessor for the same variable.
                var.clone()
            };
            incoming.push((*pred, link_index, src));
            saw = true;
        }
        if !saw {
            return None;
        }
    }
    if incoming.is_empty() {
        return None;
    }
    // `ensure_variable_at_block` records the carried value before walking
    // predecessors, so a back edge observes that definition. The metadata
    // phi is a new Signed variable. A predecessor that cannot produce one
    // drops the memo entry and does not push the inputarg.
    let phi = graph.alloc_value_var_with_type(ConcreteType::Signed);
    let key = (block.0, var.id());
    env.memo.insert(key, phi.clone());
    let mut metas = Vec::with_capacity(incoming.len());
    for (pred, link_index, src) in &incoming {
        let Some(meta) = ensure_meta(graph, env, *pred, src) else {
            env.memo.remove(&key);
            return None;
        };
        metas.push((*pred, *link_index, meta));
    }
    let first_id = metas[0].2.id();
    let used_phi = metas.iter().any(|(_, _, meta)| meta.id() == phi.id());
    if !used_phi && metas.iter().all(|(_, _, meta)| meta.id() == first_id) {
        // All predecessors produced the same metadata word, so a new
        // phi would be an identity copy. The word still has to appear
        // as an inputarg of this block: `lookup_operand` does not
        // accept a dominating use.
        let meta = metas[0].2.clone();
        env.memo.insert(key, meta.clone());
        if !graph.variable_defined_in_block(block, &meta) {
            graph.push_inputarg_var(block, meta.clone());
            for (pred, link_index, incoming) in metas {
                graph.blocks[pred.0].exits[link_index]
                    .args
                    .push(LinkArg::Value(incoming));
            }
        }
        return Some(meta);
    }
    graph.push_inputarg_var(block, phi.clone());
    for (pred, link_index, meta) in metas {
        graph.blocks[pred.0].exits[link_index]
            .args
            .push(LinkArg::Value(meta));
    }
    Some(phi)
}

fn emit_fat_len(graph: &mut FunctionGraph, env: &mut MetaEnv, fat: &FatField) -> Option<Variable> {
    if let Some(existing) = env.fat_len.get(&fat.var_id) {
        return Some(existing.clone());
    }
    // `fat.base` is the container the field was read from. After a splice
    // that container can be defined only in a predecessor. Thread it
    // (`ensure_variable_at_block`) before emitting the length word. A base
    // with no reaching definition stays the operand it already was: the
    // length word is still this field's metadata.
    if !graph.variable_defined_in_block(fat.block, &fat.base) {
        let _ = graph.try_ensure_variable_at_block(fat.block, &fat.base);
    }
    // `lookup_operand` requires the FieldRead base in this block
    // (`w_set_insert_key_checked` reads `strategy.imp` FatMeta after a
    // passthrough of the set pointer). A dominating definition is not a
    // definition (`ensure_meta` already threads the metadata word the
    // same way).
    if !graph.variable_defined_in_block(fat.block, &fat.base) {
        graph.ensure_variable_at_block(fat.block, &fat.base);
    }
    let meta = graph.alloc_value_var_with_type(ConcreteType::Signed);
    // `FatMeta` is the vtable word as Int (`GetfieldGcI`). Slice `FatLen`
    // stays `len(getattr)`; the adapter keys this tag, not the field name.
    let field = fat.field.clone().with_vec_part(VecFieldPart::FatMeta);
    let op = SpaceOperation {
        result: Some(meta.clone()),
        kind: OpKind::FieldRead {
            base: fat.base.clone(),
            field,
            ty: ValueType::Int,
            pure: fat.pure,
        },
    };
    let block = &mut graph.blocks[fat.block.0];
    let at = block.operations.iter().position(|op| {
        op.result
            .as_ref()
            .is_some_and(|result| result.id() == fat.var_id)
    });
    match at {
        Some(index) => {
            let result_ty = match &block.operations[index].kind {
                OpKind::FieldRead { field, ty, .. }
                    if field.vec_part == Some(VecFieldPart::FatVtable) =>
                {
                    Some(ty.clone())
                }
                _ => None,
            };
            if let Some(result_ty) = result_ty {
                // Same metadata word: keep the Ref users on a cast of
                // the Int load instead of a second getfield.
                block.operations[index].kind = OpKind::UnaryOp {
                    op: "cast_int_to_ptr".into(),
                    operand: meta.clone(),
                    result_ty,
                };
                block.operations.insert(index, op);
            } else {
                block.operations.insert(index + 1, op);
            }
        }
        None => block.operations.push(op),
    }
    env.fat_len.insert(fat.var_id, meta.clone());
    Some(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FieldDescriptor, FunctionGraph, OpKind, SpaceOperation, ValueType};
    use std::collections::{HashMap, HashSet};

    fn hit(id_hit: bool, rank: Option<OwnerPathRank>) -> OwnerHit {
        OwnerHit { id_hit, rank }
    }

    #[test]
    fn owner_id_beats_a_same_leaf_decl() {
        let hits = vec![
            hit(true, Some(OwnerPathRank::Leaf)),
            hit(false, Some(OwnerPathRank::Leaf)),
        ];
        assert_eq!(choose_owner_hits(true, &hits), vec![0]);
    }

    #[test]
    fn emit_fat_len_threads_base_into_the_fieldread_block() {
        // `w_set_insert_key_checked` FieldRead of `strategy.imp` FatMeta
        // after a passthrough: the owner lives in a predecessor.
        // `lookup_operand` rejects a dominating use.
        let mut graph = FunctionGraph::new("insert_key");
        let entry = graph.startblock;
        let merge = graph.create_block();
        let base = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        graph.set_goto(entry, merge, vec![]);
        assert!(
            !graph.variable_defined_in_block(merge, &base),
            "base is defined in the predecessor only"
        );
        let fat = FatField {
            block: merge,
            var_id: u64::MAX,
            base: base.clone(),
            field: FieldDescriptor::new("imp", Some("SetStrategyRef".into())),
            pure: true,
        };
        let mut env = MetaEnv {
            fat_fields: HashMap::new(),
            copies: HashMap::new(),
            fat_len: HashMap::new(),
            memo: HashMap::new(),
            visiting: HashSet::new(),
        };
        let _meta = emit_fat_len(&mut graph, &mut env, &fat);
        assert!(
            graph.variable_defined_in_block(merge, &base),
            "FatMeta FieldRead base must be an inputarg of the consuming block"
        );
    }

    #[test]
    fn owner_id_miss_does_not_fall_back_to_a_bare_leaf() {
        let hits = vec![hit(false, Some(OwnerPathRank::Leaf))];
        assert!(choose_owner_hits(true, &hits).is_empty());
    }

    #[test]
    fn missing_owner_id_keeps_the_tighter_path() {
        let hits = vec![
            hit(false, Some(OwnerPathRank::Exact)),
            hit(false, Some(OwnerPathRank::Leaf)),
        ];
        assert_eq!(choose_owner_hits(false, &hits), vec![0]);
        let leaves = vec![
            hit(false, Some(OwnerPathRank::Leaf)),
            hit(false, Some(OwnerPathRank::Leaf)),
        ];
        assert_eq!(choose_owner_hits(false, &leaves), vec![0, 1]);
    }

    #[test]
    fn qualified_owner_is_exact_after_the_crate_prefix() {
        assert_eq!(
            owner_path_rank(
                "pyre_object::dictmultiobject::DictStrategyRef",
                "dictmultiobject::DictStrategyRef"
            ),
            Some(OwnerPathRank::Exact)
        );
        assert_eq!(
            owner_path_rank("a::Inner", "b::Inner"),
            Some(OwnerPathRank::Leaf)
        );
    }

    #[test]
    fn stripped_struct_id_matches_the_field_owner() {
        let path = "pyre_object::dictmultiobject::DictStrategyRef";
        let id = majit_ir::descr::StructId::from_canonical("dictmultiobject::DictStrategyRef");
        assert!(id_matches(id, path));
        let other = majit_ir::descr::StructId::from_canonical("other::DictStrategyRef");
        assert!(!id_matches(other, path));
    }

    fn empty_env(data_id: u64, fat: Option<FatField>) -> MetaEnv {
        let mut fat_fields = HashMap::new();
        if let Some(fat) = fat {
            fat_fields.insert(data_id, fat);
        }
        MetaEnv {
            fat_fields,
            copies: HashMap::new(),
            fat_len: HashMap::new(),
            memo: HashMap::new(),
            visiting: HashSet::new(),
        }
    }

    #[test]
    fn metadata_phi_survives_a_loop_back_edge() {
        let mut graph = FunctionGraph::new("loop_meta");
        let base = graph.alloc_value_var();
        let data = graph.alloc_value_var();
        let carried = graph.alloc_value_var();
        let entry = graph.startblock;
        let header = graph.create_block();
        let field = FieldDescriptor::new("imp", Some("DictStrategyRef".into()));
        graph.blocks[entry.0].operations.push(SpaceOperation {
            result: Some(data.clone()),
            kind: OpKind::FieldRead {
                base: base.clone(),
                field: field.clone(),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        graph.push_inputarg_var(header, carried.clone());
        graph.set_goto(entry, header, vec![data.clone()]);
        graph.set_goto(header, header, vec![carried.clone()]);
        let mut env = empty_env(
            data.id(),
            Some(FatField {
                block: entry,
                var_id: data.id(),
                base,
                field,
                pure: false,
            }),
        );
        let meta = ensure_meta(&mut graph, &mut env, header, &carried).expect("metadata phi");
        assert_eq!(FunctionGraph::concretetype_of(&meta), ConcreteType::Signed);
        assert_eq!(graph.blocks[header.0].inputargs.len(), 2);
        assert_eq!(graph.blocks[header.0].inputargs[1], meta);
        assert_eq!(graph.blocks[entry.0].exits[0].args.len(), 2);
        assert_eq!(graph.blocks[header.0].exits[0].args.len(), 2);
        assert_eq!(graph.blocks[header.0].exits[0].target, header);
        let entry_meta = graph.blocks[entry.0].exits[0].args[1]
            .as_variable()
            .expect("entry metadata");
        assert_eq!(
            FunctionGraph::concretetype_of(entry_meta),
            ConcreteType::Signed
        );
        assert_eq!(
            graph.blocks[header.0].exits[0].args[1].as_variable(),
            Some(&meta)
        );
    }

    #[test]
    fn metadata_phi_is_not_pushed_when_a_predecessor_has_none() {
        let mut graph = FunctionGraph::new("loop_meta_miss");
        let data = graph.alloc_value_var();
        let entry = graph.startblock;
        let header = graph.create_block();
        graph.blocks[entry.0].operations.push(SpaceOperation {
            result: Some(data.clone()),
            kind: OpKind::ConstInt(1),
        });
        graph.push_inputarg_var(header, data.clone());
        graph.set_goto(entry, header, vec![data.clone()]);
        graph.set_goto(header, header, vec![data.clone()]);
        let mut env = empty_env(data.id(), None);
        assert!(ensure_meta(&mut graph, &mut env, header, &data).is_none());
        assert_eq!(graph.blocks[header.0].inputargs.len(), 1);
        assert_eq!(graph.blocks[entry.0].exits[0].args.len(), 1);
        assert_eq!(graph.blocks[header.0].exits[0].args.len(), 1);
        assert!(env.memo.is_empty());
    }

    #[test]
    fn metadata_follows_a_dominating_merge_into_a_later_block() {
        let mut graph = FunctionGraph::new("dominating_use");
        let base = graph.alloc_value_var();
        let data = graph.alloc_value_var();
        let carried = graph.alloc_value_var();
        let entry = graph.startblock;
        let merge = graph.create_block();
        let later = graph.create_block();
        let field = FieldDescriptor::new("imp", Some("DictStrategyRef".into()));
        graph.blocks[entry.0].operations.push(SpaceOperation {
            result: Some(data.clone()),
            kind: OpKind::FieldRead {
                base: base.clone(),
                field: field.clone(),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        graph.push_inputarg_var(merge, carried.clone());
        graph.set_goto(entry, merge, vec![data.clone()]);
        graph.set_goto(merge, later, vec![]);
        let mut env = empty_env(
            data.id(),
            Some(FatField {
                block: entry,
                var_id: data.id(),
                base,
                field,
                pure: false,
            }),
        );
        let meta = ensure_meta(&mut graph, &mut env, later, &carried).expect("dominating metadata");
        assert_eq!(FunctionGraph::concretetype_of(&meta), ConcreteType::Signed);
        assert_eq!(
            graph.blocks[later.0].inputargs.as_slice(),
            std::slice::from_ref(&meta)
        );
        assert_eq!(graph.blocks[merge.0].inputargs.len(), 2);
        assert_eq!(graph.blocks[merge.0].inputargs[1], meta);
        assert_eq!(
            graph.blocks[merge.0].exits[0].args.as_slice(),
            std::slice::from_ref(&LinkArg::Value(meta.clone()))
        );
        assert!(graph.variable_defined_in_block(later, &meta));
        assert_eq!(graph.blocks[entry.0].exits[0].args.len(), 2);
        let entry_meta = graph.blocks[entry.0].exits[0].args[1]
            .as_variable()
            .expect("entry metadata");
        assert_eq!(
            FunctionGraph::concretetype_of(entry_meta),
            ConcreteType::Signed
        );
    }

    #[test]
    fn metadata_phi_is_threaded_through_a_passthrough() {
        let mut graph = FunctionGraph::new("diamond_passthrough");
        let base = graph.alloc_value_var();
        let data_l = graph.alloc_value_var();
        let data_r = graph.alloc_value_var();
        let carried = graph.alloc_value_var();
        let left = graph.startblock;
        let right = graph.create_block();
        let merge = graph.create_block();
        let pass = graph.create_block();
        let later = graph.create_block();
        let field = FieldDescriptor::new("imp", Some("DictStrategyRef".into()));
        graph.blocks[left.0].operations.push(SpaceOperation {
            result: Some(data_l.clone()),
            kind: OpKind::FieldRead {
                base: base.clone(),
                field: field.clone(),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        graph.blocks[right.0].operations.push(SpaceOperation {
            result: Some(data_r.clone()),
            kind: OpKind::FieldRead {
                base: base.clone(),
                field: field.clone(),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        graph.push_inputarg_var(merge, carried.clone());
        graph.set_goto(left, merge, vec![data_l.clone()]);
        graph.set_goto(right, merge, vec![data_r.clone()]);
        graph.set_goto(merge, pass, vec![]);
        graph.set_goto(pass, later, vec![]);
        let mut env = empty_env(data_l.id(), None);
        env.fat_fields.insert(
            data_l.id(),
            FatField {
                block: left,
                var_id: data_l.id(),
                base: base.clone(),
                field: field.clone(),
                pure: false,
            },
        );
        env.fat_fields.insert(
            data_r.id(),
            FatField {
                block: right,
                var_id: data_r.id(),
                base,
                field,
                pure: false,
            },
        );
        let meta = ensure_meta(&mut graph, &mut env, later, &carried).expect("passthrough phi");
        assert_eq!(FunctionGraph::concretetype_of(&meta), ConcreteType::Signed);
        assert!(
            graph.variable_defined_in_block(later, &meta),
            "the method_* block must define the vtable word as an inputarg"
        );
        assert!(
            graph.variable_defined_in_block(pass, &meta),
            "the passthrough must carry the vtable word"
        );
        assert!(
            graph.variable_defined_in_block(merge, &meta),
            "the diamond merge must phi the two FatLen words"
        );
        assert_eq!(
            graph.blocks[pass.0].exits[0]
                .args
                .last()
                .and_then(LinkArg::as_variable),
            Some(&meta)
        );
        assert_eq!(
            graph.blocks[merge.0].exits[0]
                .args
                .last()
                .and_then(LinkArg::as_variable),
            Some(&meta)
        );
    }

    #[test]
    fn metadata_of_a_fat_field_is_an_inputarg_of_a_later_use() {
        let mut graph = FunctionGraph::new("later_fat");
        let base = graph.alloc_value_var();
        let data = graph.alloc_value_var();
        let entry = graph.startblock;
        let later = graph.create_block();
        let field = FieldDescriptor::new("imp", Some("DictStrategyRef".into()));
        graph.blocks[entry.0].operations.push(SpaceOperation {
            result: Some(data.clone()),
            kind: OpKind::FieldRead {
                base: base.clone(),
                field: field.clone(),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        graph.set_goto(entry, later, vec![]);
        let mut env = empty_env(
            data.id(),
            Some(FatField {
                block: entry,
                var_id: data.id(),
                base,
                field,
                pure: false,
            }),
        );
        let meta = ensure_meta(&mut graph, &mut env, later, &data).expect("later metadata");
        assert_eq!(FunctionGraph::concretetype_of(&meta), ConcreteType::Signed);
        assert_eq!(graph.blocks[later.0].inputargs.as_slice(), &[meta.clone()]);
        assert_eq!(graph.blocks[entry.0].exits[0].args.len(), 1);
        let entry_meta = graph.blocks[entry.0].exits[0].args[0]
            .as_variable()
            .expect("entry metadata");
        assert_eq!(
            FunctionGraph::concretetype_of(entry_meta),
            ConcreteType::Signed
        );
        assert_eq!(entry_meta, &meta);
    }

    #[test]
    fn metadata_phi_on_a_self_loop_terminates() {
        let mut graph = FunctionGraph::new("self_loop");
        let data = graph.alloc_value_var();
        let header = graph.create_block();
        graph.push_inputarg_var(header, data.clone());
        graph.set_goto(header, header, vec![data.clone()]);
        let mut env = empty_env(data.id(), None);
        let meta = ensure_meta(&mut graph, &mut env, header, &data).expect("self-loop phi");
        assert_eq!(graph.blocks[header.0].inputargs.len(), 2);
        assert_eq!(graph.blocks[header.0].exits[0].args.len(), 2);
        assert_eq!(
            graph.blocks[header.0].exits[0].args[1].as_variable(),
            Some(&meta)
        );
    }
}
