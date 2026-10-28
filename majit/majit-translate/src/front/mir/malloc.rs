//! Malloc / SROA (`rpython/translator/backendopt/malloc.py`).

use super::*;

/// Replace a live named-struct aggregate with its fields.
///
/// Construction is `p = malloc(S); p.f = v` (`rewrite_op_malloc`). The MIR
/// front first emits a `SyntheticTransparentCtor` so `fuse_boxing_alloc`,
/// `lower_struct_ptr_writes`, and `remove_dead_aggregates` still see the
/// construct-on-stack spelling. After those passes, a remaining named
/// struct is the JIT virtual (`OptVirtualize.make_vstruct`): field reads
/// become the stored SSA values, and a by-value copy does not exist in
/// the source language, so each escape — return, residual call argument,
/// store of the whole value, or phi/link copy — materialises one fresh
/// `New` plus `setfield`s. Mutation of one copy is then a store to that
/// object alone.
///
/// An inlined substructure is the same rule: the value is its fields,
/// never a residual constructor call (`rewrite_op_getsubstruct`,
/// `rewrite_op_getinteriorfield`).
///
/// Aggregates that still participate in a boxing cluster stay
/// constructors: the malloc argument, any phi that carries it, and nested
/// named structs stored into those (the header object fusion reads
/// `ob_type` off). They are the cluster's stack value, not the heap
/// object.
///
/// Closure environments and field-less structs also stay constructors.
/// `rewrite_op_malloc` refuses a size descr whose `all_fielddescrs` is
/// empty, and a closure has no registered layout at all.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded"
)]
pub(in crate::front::mir) fn scalar_replace_named_struct_aggregates(
    graph: &mut FunctionGraph,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> usize {
    let malloc_args = boxing_cluster_ctor_results(graph);
    let mut rewritten = 0usize;
    loop {
        let Some(site) = next_struct_aggregate_ctor(graph, &malloc_args, struct_field_attrs) else {
            break;
        };
        if !scalar_replace_one_struct_aggregate(graph, site, &malloc_args, struct_field_attrs) {
            break;
        }
        rewritten += 1;
    }
    rewritten
}

pub(in crate::front::mir) struct StructAggregateCtorSite {
    pub(in crate::front::mir) block_idx: usize,
    pub(in crate::front::mir) op_idx: usize,
    pub(in crate::front::mir) result: Variable,
    pub(in crate::front::mir) owner: String,
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded"
)]
pub(in crate::front::mir) fn next_struct_aggregate_ctor(
    graph: &FunctionGraph,
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> Option<StructAggregateCtorSite> {
    for (block_idx, block) in graph.blocks.iter().enumerate() {
        for (op_idx, op) in block.operations.iter().enumerate() {
            let Some(result) = op.result.as_ref() else {
                continue;
            };
            if malloc_args.contains(result) {
                continue;
            }
            let OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name,
                        is_struct: true,
                        ..
                    },
                args,
                result_ty,
            } = &op.kind
            else {
                continue;
            };
            if !args.is_empty() {
                continue;
            }
            if majit_charon_reader::ullbc::is_closure_leaf(name) {
                continue;
            }
            // The callee writes the cell, so the post-call field read is
            // not the pre-call SSA value. `optimizeopt/virtualize.py`
            // removes the cell once the callee is inlined.
            if is_mut_ref_cell_name(name) {
                continue;
            }
            let ValueType::Ref(Some(owner)) = result_ty else {
                continue;
            };
            if owner.is_empty() {
                continue;
            }
            let layout_empty = struct_field_attrs
                .get(owner)
                .or_else(|| {
                    owner
                        .rsplit("::")
                        .next()
                        .and_then(|leaf| struct_field_attrs.get(leaf))
                })
                .is_some_and(|rows| rows.is_empty());
            let discovered = struct_ctor_discovered_fields(graph, result);
            if discovered.is_empty() {
                continue;
            }
            if layout_empty && struct_ctor_has_whole_value_use(graph, result) {
                continue;
            }
            return Some(StructAggregateCtorSite {
                block_idx,
                op_idx,
                result: result.clone(),
                owner: owner.clone(),
            });
        }
    }
    None
}

pub(in crate::front::mir) fn struct_ctor_discovered_fields(
    graph: &FunctionGraph,
    result: &Variable,
) -> Vec<(
    crate::model::FieldDescriptor,
    crate::model::LinkArg,
    ValueType,
)> {
    let mut fields: Vec<(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )> = Vec::new();
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        let OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } = &op.kind
        else {
            continue;
        };
        if base != result {
            continue;
        }
        if let Some(existing) = fields
            .iter_mut()
            .find(|(seen, _, _)| seen.name == field.name)
        {
            *existing = (field.clone(), value.clone(), ty.clone());
        } else {
            fields.push((field.clone(), value.clone(), ty.clone()));
        }
    }
    fields
}

pub(in crate::front::mir) fn struct_ctor_has_whole_value_use(
    graph: &FunctionGraph,
    result: &Variable,
) -> bool {
    for block in &graph.blocks {
        if block.inputargs.iter().any(|arg| arg == result) {
            return true;
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) if var == result => return true,
            Some(ExitSwitch::Fused { args, .. }) if args.iter().any(|arg| arg == result) => {
                return true;
            }
            Some(ExitSwitch::LastException | ExitSwitch::Value(_) | ExitSwitch::Fused { .. })
            | None => {}
        }
        for link in &block.exits {
            if link
                .args
                .iter()
                .any(|arg| arg.as_variable() == Some(result))
            {
                return true;
            }
            if link.last_exception.as_ref().and_then(LinkArg::as_variable) == Some(result)
                || link.last_exc_value.as_ref().and_then(LinkArg::as_variable) == Some(result)
            {
                return true;
            }
        }
        for op in &block.operations {
            if struct_ctor_kind_has_whole_value_use(&op.kind, result) {
                return true;
            }
        }
    }
    false
}

pub(in crate::front::mir) fn struct_ctor_kind_has_whole_value_use(
    kind: &OpKind,
    result: &Variable,
) -> bool {
    match kind {
        OpKind::FieldRead { base, .. } if base == result => false,
        OpKind::FieldWrite { value, .. } => value.as_variable() == Some(result),
        OpKind::Call { target, args, .. } => {
            args.iter().any(|arg| arg.as_variable() == Some(result))
                && !call_target_is_gc_malloc(target)
        }
        OpKind::IndirectCall { args, .. } => args.iter().any(|arg| arg == result),
        kind => crate::inline::op_variable_refs(kind)
            .iter()
            .any(|var| var == result),
    }
}

/// `malloc.py` `LifeTime` — one union-find class of `(block, var)` pairs.
pub(in crate::front::mir) struct MallocLifeTime {
    pub(in crate::front::mir) parent: usize,
    pub(in crate::front::mir) variables: Vec<(usize, Variable)>,
    /// `("op", block, op)` or a non-op tag (`"inputargs"`, `"constant"`,
    /// `"last_exception"`, `"last_exc_value"`).
    pub(in crate::front::mir) creations: Vec<MallocCreation>,
    /// `("op", block, op, index)` or a non-op tag (`"return"`, `"except"`,
    /// `"exitswitch"`, `"dup"`).
    pub(in crate::front::mir) uses: Vec<MallocUse>,
}

pub(in crate::front::mir) enum MallocCreation {
    Op { block: usize, op_idx: usize },
    Other,
}

pub(in crate::front::mir) enum MallocUse {
    Op {
        block: usize,
        op_idx: usize,
        index: usize,
    },
    Other,
}

/// Current value of one flattened field inside `flowin`.
#[derive(Clone)]
pub(in crate::front::mir) enum FieldCur {
    /// `malloc.py` `newvarsmap` entry: a variable or a constant.
    Arg(LinkArg),
    /// Typed zero of a `Ref` / `Str` field (`ConstRefNull`).
    NullRef,
}

/// `malloc.py` `compute_lifetimes` + `_try_inline_malloc` for the lifetime
/// that contains `ctor` (`block_idx`, `result`). `true` when `flowin`
/// removed that malloc. A later `scalar_replace_named_struct_aggregates`
/// iteration is the `remove_simple_mallocs` fixpoint.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded"
)]
pub(in crate::front::mir) fn try_inline_malloc_lifetime(
    graph: &mut FunctionGraph,
    block_idx: usize,
    result: &Variable,
    owner: &str,
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> bool {
    let Some(flatnames) = struct_flatnames(owner, struct_field_attrs) else {
        return false;
    };
    if flatnames.is_empty() {
        return false;
    }
    let lifetimes = compute_lifetimes(graph);
    let Some(info_idx) = lifetimes.index.get(&(block_idx, result.clone())).copied() else {
        return false;
    };
    let info_idx = lifetimes.root_of(info_idx);
    let info = &lifetimes.nodes[info_idx];
    if !lifetime_is_removable(
        graph,
        info,
        owner,
        &flatnames,
        malloc_args,
        struct_field_attrs,
    ) {
        return false;
    }
    let mut by_block: Vec<(usize, Vec<Variable>)> = Vec::new();
    for (block, var) in &info.variables {
        if let Some((_, vars)) = by_block.iter_mut().find(|(seen, _)| seen == block) {
            if !vars.iter().any(|existing| existing == var) {
                vars.push(var.clone());
            }
        } else {
            by_block.push((*block, vec![var.clone()]));
        }
    }
    let zeros: Vec<FieldCur> = flatnames.iter().map(|(_, ty)| zero_field_cur(ty)).collect();
    // `malloc.py` `_try_inline_malloc`: inputargs first, then mallocs
    // created in the block. Each successful malloc increments progress.
    let mut progress = 0usize;
    for (block, vars) in by_block {
        let mut new_map: Option<Vec<FieldCur>> = None;
        let mut new_inputargs: Vec<Variable> = Vec::new();
        let mut inputvars: Vec<Variable> = Vec::new();
        for var in graph.blocks[block].inputargs.clone() {
            if vars.iter().any(|candidate| candidate == &var) {
                inputvars.push(var);
                if new_map.is_none() {
                    let mut map = Vec::with_capacity(flatnames.len());
                    for (_, ty) in &flatnames {
                        let fresh = graph.alloc_value_var_with_type(concretetype_for_field(ty));
                        map.push(FieldCur::Arg(LinkArg::Value(fresh.clone())));
                        new_inputargs.push(fresh);
                    }
                    new_map = Some(map);
                }
            } else {
                new_inputargs.push(var);
            }
        }
        graph.blocks[block].inputargs = new_inputargs;
        if !inputvars.is_empty() {
            flowin_malloc_block(graph, block, inputvars, new_map, &flatnames, &zeros);
        }
        let created: Vec<Variable> = graph.blocks[block]
            .operations
            .iter()
            .filter_map(|op| {
                let produced = op.result.as_ref()?;
                let still_malloc = matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor {
                            is_struct: true,
                            ..
                        },
                        args,
                        ..
                    } if args.is_empty()
                );
                if still_malloc && vars.iter().any(|candidate| candidate == produced) {
                    Some(produced.clone())
                } else {
                    None
                }
            })
            .collect();
        for var in created {
            flowin_malloc_block(graph, block, vec![var], None, &flatnames, &zeros);
            progress += 1;
        }
    }
    // Input-flowin may already have dropped the malloc (`progress` stays 0).
    // The caller treats `false` as "fall back to `New`", so any completed
    // inline reports success. `progress` still counts mallocs removed inside
    // the block, matching `malloc.py` `_try_inline_malloc`.
    let _ = progress;
    true
}

pub(in crate::front::mir) fn struct_flatnames(
    owner: &str,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> Option<Vec<(String, ValueType)>> {
    struct_field_attrs
        .get(owner)
        .or_else(|| {
            owner
                .rsplit("::")
                .next()
                .and_then(|leaf| struct_field_attrs.get(leaf))
        })
        .cloned()
}

pub(in crate::front::mir) fn concretetype_for_field(ty: &ValueType) -> crate::model::ConcreteType {
    match ty {
        ValueType::Float => crate::model::ConcreteType::Float,
        ValueType::Void => crate::model::ConcreteType::Void,
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => {
            crate::model::ConcreteType::GcRef
        }
        ValueType::State | ValueType::Unknown => crate::model::ConcreteType::Unknown,
        ValueType::Int
        | ValueType::Unsigned
        | ValueType::Bool
        | ValueType::SingleFloat
        | ValueType::Int128
        | ValueType::UInt128 => crate::model::ConcreteType::Signed,
    }
}

/// Typed zero, matching `emit_zero_constant_of_ty`. `Ref` is null.
pub(in crate::front::mir) fn zero_field_cur(ty: &ValueType) -> FieldCur {
    match ty {
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => FieldCur::NullRef,
        ValueType::Bool => FieldCur::Arg(LinkArg::Const(crate::flowspace::model::Constant::new(
            ConstValue::Bool(false),
        ))),
        ValueType::Float => FieldCur::Arg(LinkArg::Const(crate::flowspace::model::Constant::new(
            ConstValue::Float(0),
        ))),
        ValueType::Int128 => FieldCur::Arg(LinkArg::Const(crate::flowspace::model::Constant::new(
            ConstValue::Int128(0),
        ))),
        ValueType::UInt128 => FieldCur::Arg(LinkArg::Const(
            crate::flowspace::model::Constant::new(ConstValue::UInt128(0)),
        )),
        ValueType::Void => FieldCur::Arg(LinkArg::Const(crate::flowspace::model::Constant::new(
            ConstValue::None,
        ))),
        ValueType::SingleFloat => FieldCur::Arg(LinkArg::Const(
            crate::flowspace::model::Constant::new(ConstValue::Int(0)),
        )),
        ValueType::Unsigned => FieldCur::Arg(LinkArg::Const(
            crate::flowspace::model::Constant::new(ConstValue::UInt(0)),
        )),
        ValueType::Int | ValueType::State | ValueType::Unknown => FieldCur::Arg(LinkArg::Const(
            crate::flowspace::model::Constant::new(ConstValue::Int(0)),
        )),
    }
}

pub(in crate::front::mir) fn alias_op_for_field(cur: &FieldCur, ty: &ValueType) -> Option<OpKind> {
    match cur {
        FieldCur::NullRef => Some(OpKind::ConstRefNull),
        FieldCur::Arg(arg) => {
            if matches!(ty, ValueType::Unsigned)
                && matches!(arg, LinkArg::Const(constant) if matches!(constant.value, ConstValue::UInt(0) | ConstValue::Int(0)))
            {
                return Some(OpKind::ConstUInt(0));
            }
            if matches!(ty, ValueType::SingleFloat)
                && matches!(arg, LinkArg::Const(constant) if matches!(constant.value, ConstValue::Int(0)))
            {
                return Some(OpKind::ConstSingleFloat(0));
            }
            link_arg_as_alias_op(arg, ty)
        }
    }
}

pub(in crate::front::mir) fn field_value_aliasable(value: &LinkArg, ty: &ValueType) -> bool {
    alias_op_for_field(&FieldCur::Arg(value.clone()), ty).is_some()
}

/// `malloc.py` `LLTypeMallocRemover.check_malloc` for a front aggregate ctor,
/// the same selection as [`next_struct_aggregate_ctor`].
pub(in crate::front::mir) fn aggregate_ctor_owner(
    graph: &FunctionGraph,
    op: &SpaceOperation,
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> Option<String> {
    let result = op.result.as_ref()?;
    if malloc_args.contains(result) {
        return None;
    }
    let OpKind::Call {
        target:
            CallTarget::SyntheticTransparentCtor {
                name,
                is_struct: true,
                ..
            },
        args,
        result_ty,
    } = &op.kind
    else {
        return None;
    };
    if !args.is_empty() {
        return None;
    }
    if majit_charon_reader::ullbc::is_closure_leaf(name) {
        return None;
    }
    // The callee writes the cell, so the post-call field read is not the
    // pre-call SSA value. `optimizeopt/virtualize.py` removes the cell
    // once the callee is inlined.
    if is_mut_ref_cell_name(name) {
        return None;
    }
    let ValueType::Ref(Some(owner)) = result_ty else {
        return None;
    };
    if owner.is_empty() {
        return None;
    }
    let layout_empty = struct_field_attrs
        .get(owner)
        .or_else(|| {
            owner
                .rsplit("::")
                .next()
                .and_then(|leaf| struct_field_attrs.get(leaf))
        })
        .is_some_and(|rows| rows.is_empty());
    let discovered = struct_ctor_discovered_fields(graph, result);
    if discovered.is_empty() {
        return None;
    }
    if layout_empty && struct_ctor_has_whole_value_use(graph, result) {
        return None;
    }
    Some(owner.clone())
}

pub(in crate::front::mir) struct MallocLifetimes {
    pub(in crate::front::mir) index: std::collections::HashMap<(usize, Variable), usize>,
    pub(in crate::front::mir) nodes: Vec<MallocLifeTime>,
}

impl MallocLifetimes {
    pub(in crate::front::mir) fn ensure(&mut self, block: usize, var: &Variable) -> usize {
        let key = (block, var.clone());
        if let Some(&idx) = self.index.get(&key) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(MallocLifeTime {
            parent: idx,
            variables: vec![key.clone()],
            creations: Vec::new(),
            uses: Vec::new(),
        });
        self.index.insert(key, idx);
        idx
    }

    pub(in crate::front::mir) fn root_of(&self, mut idx: usize) -> usize {
        while self.nodes[idx].parent != idx {
            idx = self.nodes[idx].parent;
        }
        idx
    }

    pub(in crate::front::mir) fn find(&mut self, block: usize, var: &Variable) -> usize {
        let idx = self.ensure(block, var);
        let root = self.root_of(idx);
        let mut cursor = idx;
        while cursor != root {
            let parent = self.nodes[cursor].parent;
            self.nodes[cursor].parent = root;
            cursor = parent;
        }
        root
    }

    pub(in crate::front::mir) fn set_creation(
        &mut self,
        block: usize,
        var: &Variable,
        creation: MallocCreation,
    ) {
        let root = self.find(block, var);
        self.nodes[root].creations.push(creation);
    }

    pub(in crate::front::mir) fn set_use(
        &mut self,
        block: usize,
        var: &Variable,
        use_point: MallocUse,
    ) {
        let root = self.find(block, var);
        self.nodes[root].uses.push(use_point);
    }

    /// `malloc.py` `compute_lifetimes` `union`.
    pub(in crate::front::mir) fn union(
        &mut self,
        block1: usize,
        var1: &Variable,
        block2: usize,
        var2: &Variable,
    ) {
        let left = self.find(block1, var1);
        let right = self.find(block2, var2);
        if left == right {
            return;
        }
        let variables = std::mem::take(&mut self.nodes[right].variables);
        let creations = std::mem::take(&mut self.nodes[right].creations);
        let uses = std::mem::take(&mut self.nodes[right].uses);
        self.nodes[left].variables.extend(variables);
        self.nodes[left].creations.extend(creations);
        self.nodes[left].uses.extend(uses);
        self.nodes[right].parent = left;
    }
}

/// `malloc.py` `BaseMallocRemover.compute_lifetimes`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded"
)]
pub(in crate::front::mir) fn compute_lifetimes(graph: &FunctionGraph) -> MallocLifetimes {
    let id_to_idx: std::collections::HashMap<BlockId, usize> = graph
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    let mut lifetimes = MallocLifetimes {
        index: std::collections::HashMap::new(),
        nodes: Vec::new(),
    };
    let start = id_to_idx[&graph.startblock];
    for var in &graph.blocks[start].inputargs {
        lifetimes.set_creation(start, var, MallocCreation::Other);
    }
    let return_block = id_to_idx[&graph.returnblock];
    if let Some(var) = graph.blocks[return_block].inputargs.first() {
        lifetimes.set_use(return_block, var, MallocUse::Other);
    }
    let except_block = id_to_idx[&graph.exceptblock];
    for var in graph.blocks[except_block].inputargs.iter().take(2) {
        lifetimes.set_use(except_block, var, MallocUse::Other);
    }

    let reachable = graph.iterblocks_order();
    for block_id in &reachable {
        let Some(&block_idx) = id_to_idx.get(block_id) else {
            continue;
        };
        let block = &graph.blocks[block_idx];
        for (op_idx, op) in block.operations.iter().enumerate() {
            // `IDENTITY_OPS`: `link_arg_as_alias_op`'s plain `same_as`.
            if let OpKind::UnaryOp {
                op: name, operand, ..
            } = &op.kind
                && name == "same_as"
                && let Some(produced) = &op.result
            {
                lifetimes.union(block_idx, operand, block_idx, produced);
                continue;
            }
            for (index, var) in crate::inline::op_variable_refs(&op.kind).iter().enumerate() {
                lifetimes.set_use(
                    block_idx,
                    var,
                    MallocUse::Op {
                        block: block_idx,
                        op_idx,
                        index,
                    },
                );
            }
            if let Some(produced) = &op.result {
                lifetimes.set_creation(
                    block_idx,
                    produced,
                    MallocCreation::Op {
                        block: block_idx,
                        op_idx,
                    },
                );
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) => {
                lifetimes.set_use(block_idx, var, MallocUse::Other);
            }
            Some(ExitSwitch::Fused { args, .. }) => {
                for var in args {
                    lifetimes.set_use(block_idx, var, MallocUse::Other);
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
    }

    for block_id in &reachable {
        let Some(&block_idx) = id_to_idx.get(block_id) else {
            continue;
        };
        for link in &graph.blocks[block_idx].exits {
            if let Some(var) = link.last_exception.as_ref().and_then(LinkArg::as_variable) {
                lifetimes.set_creation(block_idx, var, MallocCreation::Other);
            }
            if let Some(var) = link.last_exc_value.as_ref().and_then(LinkArg::as_variable) {
                lifetimes.set_creation(block_idx, var, MallocCreation::Other);
            }
            let Some(&target_idx) = id_to_idx.get(&link.target) else {
                continue;
            };
            let target_args = graph.blocks[target_idx].inputargs.clone();
            let mut duplicate_roots: Vec<usize> = Vec::new();
            for (index, arg) in link.args.iter().enumerate() {
                let Some(target_var) = target_args.get(index) else {
                    continue;
                };
                match arg {
                    LinkArg::Value(var) => {
                        lifetimes.union(block_idx, var, target_idx, target_var);
                    }
                    LinkArg::Const(_) => {
                        lifetimes.set_creation(target_idx, target_var, MallocCreation::Other);
                    }
                }
                let Some(var) = arg.as_variable() else {
                    continue;
                };
                let root = lifetimes.find(block_idx, var);
                if duplicate_roots.contains(&root) && lifetimes.nodes[root].creations.len() > 1 {
                    lifetimes.set_use(block_idx, var, MallocUse::Other);
                } else {
                    duplicate_roots.push(root);
                }
            }
        }
    }
    lifetimes
}

pub(in crate::front::mir) fn lifetime_is_removable(
    graph: &FunctionGraph,
    info: &MallocLifeTime,
    owner: &str,
    flatnames: &[(String, ValueType)],
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> bool {
    // `_try_inline_malloc`: every creation point is a malloc of one owner.
    if info.creations.is_empty() {
        return false;
    }
    for creation in &info.creations {
        let MallocCreation::Op { block, op_idx } = creation else {
            return false;
        };
        let Some(op) = graph
            .blocks
            .get(*block)
            .and_then(|block| block.operations.get(*op_idx))
        else {
            return false;
        };
        if aggregate_ctor_owner(graph, op, malloc_args, struct_field_attrs).as_deref()
            != Some(owner)
        {
            return false;
        }
    }
    for use_point in &info.uses {
        let MallocUse::Op {
            block,
            op_idx,
            index,
        } = use_point
        else {
            return false;
        };
        if *index != 0 {
            return false;
        }
        let Some(op) = graph
            .blocks
            .get(*block)
            .and_then(|block| block.operations.get(*op_idx))
        else {
            return false;
        };
        let field_name = match &op.kind {
            OpKind::FieldRead { field, .. } => field.name.as_str(),
            OpKind::FieldWrite {
                field, value, ty, ..
            } => {
                if !field_value_aliasable(value, ty) {
                    return false;
                }
                field.name.as_str()
            }
            _ => return false,
        };
        if !flatnames.iter().any(|(name, _)| name == field_name) {
            return false;
        }
    }
    // A field op before the malloc (and with no incoming field vars) would
    // miss `newvarsmap`. That is not removable (`handle_unreachable` is out
    // of scope).
    let mut by_block: Vec<(usize, Vec<Variable>)> = Vec::new();
    for (block, var) in &info.variables {
        if let Some((_, vars)) = by_block.iter_mut().find(|(seen, _)| seen == block) {
            if !vars.iter().any(|existing| existing == var) {
                vars.push(var.clone());
            }
        } else {
            by_block.push((*block, vec![var.clone()]));
        }
    }
    for (block, mut vars) in by_block {
        let mut ready = graph.blocks[block]
            .inputargs
            .iter()
            .any(|var| vars.iter().any(|candidate| candidate == var));
        for op in &graph.blocks[block].operations {
            if let OpKind::UnaryOp {
                op: name, operand, ..
            } = &op.kind
                && name == "same_as"
                && vars.iter().any(|candidate| candidate == operand)
                && let Some(produced) = &op.result
            {
                vars.push(produced.clone());
                continue;
            }
            let refs = crate::inline::op_variable_refs(&op.kind);
            let base_in = refs
                .first()
                .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
            let result_in = op
                .result
                .as_ref()
                .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
            if base_in {
                if !ready {
                    return false;
                }
            } else if result_in {
                ready = true;
            }
        }
    }
    true
}

/// `malloc.py` `BaseMallocRemover.flowin` + `LLTypeMallocRemover.flowin_op`
/// (`getfield` / `setfield` / `same_as` only).
pub(in crate::front::mir) fn flowin_malloc_block(
    graph: &mut FunctionGraph,
    block: usize,
    mut vars: Vec<Variable>,
    mut newvarsmap: Option<Vec<FieldCur>>,
    flatnames: &[(String, ValueType)],
    zeros: &[FieldCur],
) {
    let ops = std::mem::take(&mut graph.blocks[block].operations);
    let mut newops: Vec<SpaceOperation> = Vec::new();
    for op in ops {
        if let OpKind::UnaryOp {
            op: name, operand, ..
        } = &op.kind
            && name == "same_as"
            && vars.iter().any(|candidate| candidate == operand)
        {
            // `flowin_op` same_as: one flattened list for both pointers.
            if let Some(produced) = &op.result {
                vars.push(produced.clone());
            }
            continue;
        }
        let refs = crate::inline::op_variable_refs(&op.kind);
        let base_in = refs
            .first()
            .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
        let result_in = op
            .result
            .as_ref()
            .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
        if base_in {
            match &op.kind {
                OpKind::FieldRead { field, ty, .. } => {
                    let Some(map) = newvarsmap.as_ref() else {
                        newops.push(op);
                        continue;
                    };
                    let Some(index) = flatnames.iter().position(|(name, _)| name == &field.name)
                    else {
                        newops.push(op);
                        continue;
                    };
                    let Some(kind) = alias_op_for_field(&map[index], ty) else {
                        newops.push(op);
                        continue;
                    };
                    newops.push(SpaceOperation {
                        result: op.result.clone(),
                        kind,
                    });
                }
                OpKind::FieldWrite { field, value, .. } => {
                    let Some(map) = newvarsmap.as_mut() else {
                        newops.push(op);
                        continue;
                    };
                    let Some(index) = flatnames.iter().position(|(name, _)| name == &field.name)
                    else {
                        newops.push(op);
                        continue;
                    };
                    map[index] = FieldCur::Arg(value.clone());
                }
                _ => newops.push(op),
            }
        } else if result_in {
            // Drop the malloc. Field vars start as the typed zeros.
            newvarsmap = Some(zeros.to_vec());
        } else {
            newops.push(op);
        }
    }
    let exits = std::mem::take(&mut graph.blocks[block].exits);
    let mut rewritten = Vec::with_capacity(exits.len());
    for mut link in exits {
        let mut appended = false;
        let mut new_args: Vec<LinkArg> = Vec::new();
        for arg in &link.args {
            if arg
                .as_variable()
                .is_some_and(|var| vars.iter().any(|candidate| candidate == var))
            {
                if !appended {
                    if let Some(map) = &newvarsmap {
                        for (index, cur) in map.iter().enumerate() {
                            new_args.push(field_cur_as_link_arg(
                                graph,
                                &mut newops,
                                cur,
                                &flatnames[index].1,
                            ));
                        }
                    }
                    appended = true;
                }
            } else {
                new_args.push(arg.clone());
            }
        }
        link.args = new_args;
        rewritten.push(link);
    }
    graph.blocks[block].operations = newops;
    graph.blocks[block].exits = rewritten;
}

pub(in crate::front::mir) fn field_cur_as_link_arg(
    graph: &mut FunctionGraph,
    newops: &mut Vec<SpaceOperation>,
    cur: &FieldCur,
    _ty: &ValueType,
) -> LinkArg {
    match cur {
        FieldCur::NullRef => {
            let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
            newops.push(SpaceOperation {
                result: Some(var.clone()),
                kind: OpKind::ConstRefNull,
            });
            LinkArg::Value(var)
        }
        FieldCur::Arg(arg) => arg.clone(),
    }
}

pub(in crate::front::mir) fn scalar_replace_one_struct_aggregate(
    graph: &mut FunctionGraph,
    site: StructAggregateCtorSite,
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> bool {
    let StructAggregateCtorSite {
        block_idx,
        op_idx,
        result,
        owner,
    } = site;
    let foreign_field_ops = graph.blocks.iter().enumerate().any(|(idx, block)| {
        idx != block_idx
            && block.operations.iter().any(|op| match &op.kind {
                OpKind::FieldRead { base, .. } | OpKind::FieldWrite { base, .. } => base == &result,
                _ => false,
            })
    });
    if foreign_field_ops {
        // `malloc.py` `_try_inline_malloc` / `flowin`: a field op in another
        // block is not an escape when the value only moves along links.
        // Fall back to one `New` per copy when the lifetime is not removable.
        if try_inline_malloc_lifetime(
            graph,
            block_idx,
            &result,
            &owner,
            malloc_args,
            struct_field_attrs,
        ) {
            return true;
        }
        let has_vtable =
            crate::codewriter::heaptracker::attrs_have_vtable(&owner, struct_field_attrs);
        graph.blocks[block_idx].operations[op_idx].kind =
            crate::model::malloc_opkind(owner.clone(), has_vtable);
        materialize_whole_value_copies_in_block(
            graph, block_idx, op_idx, &result, &owner, has_vtable,
        );
        return true;
    }

    let has_vtable = crate::codewriter::heaptracker::attrs_have_vtable(&owner, struct_field_attrs);
    let ops = std::mem::take(&mut graph.blocks[block_idx].operations);
    let mut out: Vec<crate::model::SpaceOperation> = Vec::new();
    let mut fields: Vec<(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )> = Vec::new();
    let mut field_alias: Vec<(Variable, Variable)> = Vec::new();
    for (i, op) in ops.into_iter().enumerate() {
        if i == op_idx {
            continue;
        }
        match &op.kind {
            OpKind::FieldWrite {
                base,
                field,
                value,
                ty,
            } if base == &result => {
                upsert_struct_field(&mut fields, field.clone(), value.clone(), ty.clone());
                continue;
            }
            OpKind::FieldRead {
                base, field, ty, ..
            } if base == &result => {
                if let Some((_, value, _)) =
                    fields.iter().find(|(seen, _, _)| seen.name == field.name)
                {
                    if let (Some(dest), LinkArg::Value(src)) = (&op.result, value) {
                        // The read is the stored SSA value. A block input
                        // stays a `same_as`: that parameter is distinct.
                        if !graph.blocks[block_idx]
                            .inputargs
                            .iter()
                            .any(|arg| arg == dest)
                        {
                            field_alias.push((dest.clone(), src.clone()));
                            continue;
                        }
                    }
                    if let Some(kind) = link_arg_as_alias_op(value, ty) {
                        out.push(crate::model::SpaceOperation {
                            result: op.result.clone(),
                            kind,
                        });
                        continue;
                    }
                }
            }
            _ => {}
        }
        if struct_ctor_kind_has_whole_value_use(&op.kind, &result) {
            let kind = replace_whole_value_uses_in_kind(
                graph, &mut out, &op.kind, &result, &owner, &fields, has_vtable,
            );
            out.push(crate::model::SpaceOperation {
                result: op.result.clone(),
                kind,
            });
        } else {
            out.push(op);
        }
    }

    let apply_alias = |var: &Variable| -> Variable {
        let mut current = var.clone();
        let mut seen: Vec<Variable> = Vec::new();
        loop {
            if seen.iter().any(|item| item == &current) {
                return current;
            }
            seen.push(current.clone());
            let Some((_, next)) = field_alias.iter().find(|(from, _)| from == &current) else {
                return current;
            };
            current = next.clone();
        }
    };
    for op in &mut out {
        let remap = |var: &Variable| apply_alias(var);
        op.kind = crate::inline::remap_op_kind(&op.kind, &remap);
        if let Some(result) = op.result.clone() {
            let renamed = apply_alias(&result);
            if renamed != result {
                op.result = Some(renamed);
            }
        }
    }
    let mut phi_copies: Vec<(BlockId, usize, Variable)> = Vec::new();
    let mut exits = std::mem::take(&mut graph.blocks[block_idx].exits);
    for link in &mut exits {
        for arg in &mut link.args {
            if let Some(var) = arg.as_variable() {
                let renamed = apply_alias(var);
                if &renamed != var {
                    *arg = LinkArg::Value(renamed);
                }
            }
        }
        if let Some(var) = link.last_exception.as_ref().and_then(LinkArg::as_variable) {
            let renamed = apply_alias(var);
            if &renamed != var {
                link.last_exception = Some(LinkArg::Value(renamed));
            }
        }
        if let Some(var) = link.last_exc_value.as_ref().and_then(LinkArg::as_variable) {
            let renamed = apply_alias(var);
            if &renamed != var {
                link.last_exc_value = Some(LinkArg::Value(renamed));
            }
        }
    }
    for link in &mut exits {
        for (slot, arg) in link.args.iter_mut().enumerate() {
            if arg.as_variable() != Some(&result) {
                continue;
            }
            let copy = emit_materialized_struct_copy(graph, &mut out, &owner, &fields, has_vtable);
            *arg = LinkArg::Value(copy.clone());
            // Each predecessor passes its own copy. The successor parameter
            // stays one variable; renaming it to a single predecessor's copy
            // and leaving the body on the old name made every later edge
            // read the first copy.
            phi_copies.push((link.target, slot, copy));
        }
        if link.last_exception.as_ref().and_then(LinkArg::as_variable) == Some(&result) {
            let copy = emit_materialized_struct_copy(graph, &mut out, &owner, &fields, has_vtable);
            link.last_exception = Some(LinkArg::Value(copy));
        }
        if link.last_exc_value.as_ref().and_then(LinkArg::as_variable) == Some(&result) {
            let copy = emit_materialized_struct_copy(graph, &mut out, &owner, &fields, has_vtable);
            link.last_exc_value = Some(LinkArg::Value(copy));
        }
    }
    graph.blocks[block_idx].operations = out;
    graph.blocks[block_idx].exits = exits;
    bind_struct_phi_copies(graph, &result, &phi_copies);
    true
}

/// When the successor parameter *is* the aggregate variable, point that
/// parameter and every use in the successor at one phi. Predecessors keep
/// the distinct copies already stored in their link args.
pub(in crate::front::mir) fn bind_struct_phi_copies(
    graph: &mut FunctionGraph,
    from: &Variable,
    copies: &[(BlockId, usize, Variable)],
) {
    let mut groups: Vec<(BlockId, usize, Vec<Variable>)> = Vec::new();
    for (target, slot, copy) in copies {
        if let Some((_, _, incoming)) = groups
            .iter_mut()
            .find(|(seen_target, seen_slot, _)| seen_target == target && *seen_slot == *slot)
        {
            incoming.push(copy.clone());
        } else {
            groups.push((*target, *slot, vec![copy.clone()]));
        }
    }
    for (target, slot, incoming) in groups {
        let Some(idx) = graph.blocks.iter().position(|block| block.id == target) else {
            continue;
        };
        if graph.blocks[idx].inputargs.get(slot) != Some(from) {
            continue;
        }
        let phi = if incoming.len() == 1 {
            incoming[0].clone()
        } else {
            graph.alloc_value_var()
        };
        let block = &mut graph.blocks[idx];
        block.inputargs[slot] = phi.clone();
        remap_struct_phi_uses(block, from, &phi);
    }
}

pub(in crate::front::mir) fn remap_struct_phi_uses(
    block: &mut Block,
    from: &Variable,
    to: &Variable,
) {
    let remap = |var: &Variable| {
        if var == from { to.clone() } else { var.clone() }
    };
    for op in &mut block.operations {
        op.kind = crate::inline::remap_op_kind(&op.kind, &remap);
    }
    match &mut block.exitswitch {
        Some(ExitSwitch::Value(var)) if var == from => *var = to.clone(),
        Some(ExitSwitch::Fused { args, .. }) => {
            for arg in args {
                if arg == from {
                    *arg = to.clone();
                }
            }
        }
        Some(ExitSwitch::LastException | ExitSwitch::Value(_)) | None => {}
    }
    for link in &mut block.exits {
        for arg in &mut link.args {
            retarget_link_arg(arg, from, to);
        }
        if let Some(arg) = link.last_exception.as_mut() {
            retarget_link_arg(arg, from, to);
        }
        if let Some(arg) = link.last_exc_value.as_mut() {
            retarget_link_arg(arg, from, to);
        }
    }
}

pub(in crate::front::mir) fn retarget_link_arg(arg: &mut LinkArg, from: &Variable, to: &Variable) {
    if arg.as_variable() == Some(from) {
        *arg = LinkArg::Value(to.clone());
    }
}

/// Replace uses of `from` with `to` and drop the op that defined `from`.
///
/// This is the graph `remove_same_as` builds. Returns false when `from`
/// is a block input: that parameter is the edge's own variable, and the
/// copy that feeds it has to stay a distinct `same_as`.
pub(crate) fn forward_identity(graph: &mut FunctionGraph, from: &Variable, to: &Variable) -> bool {
    if from == to {
        return true;
    }
    if graph
        .blocks
        .iter()
        .any(|block| block.inputargs.iter().any(|arg| arg == from))
    {
        return false;
    }
    let remap = |var: &Variable| {
        if var == from { to.clone() } else { var.clone() }
    };
    for block in &mut graph.blocks {
        block
            .operations
            .retain(|op| op.result.as_ref() != Some(from));
        for op in &mut block.operations {
            op.kind = crate::inline::remap_op_kind(&op.kind, &remap);
        }
        match &mut block.exitswitch {
            Some(ExitSwitch::Value(var)) if var == from => *var = to.clone(),
            Some(ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    if arg == from {
                        *arg = to.clone();
                    }
                }
            }
            Some(ExitSwitch::LastException | ExitSwitch::Value(_)) | None => {}
        }
        for link in &mut block.exits {
            for arg in &mut link.args {
                retarget_link_arg(arg, from, to);
            }
            if let Some(arg) = link.last_exception.as_mut() {
                retarget_link_arg(arg, from, to);
            }
            if let Some(arg) = link.last_exc_value.as_mut() {
                retarget_link_arg(arg, from, to);
            }
        }
    }
    true
}

pub(in crate::front::mir) fn upsert_struct_field(
    fields: &mut Vec<(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )>,
    field: crate::model::FieldDescriptor,
    value: crate::model::LinkArg,
    ty: ValueType,
) {
    if let Some(existing) = fields
        .iter_mut()
        .find(|(seen, _, _)| seen.name == field.name)
    {
        *existing = (field, value, ty);
    } else {
        fields.push((field, value, ty));
    }
}

pub(in crate::front::mir) fn link_arg_as_alias_op(
    value: &crate::model::LinkArg,
    result_ty: &ValueType,
) -> Option<OpKind> {
    match value {
        LinkArg::Value(var) => Some(OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand: var.clone(),
            result_ty: result_ty.clone(),
        }),
        LinkArg::Const(constant) => match &constant.value {
            crate::flowspace::model::ConstValue::Int(v) => Some(OpKind::ConstInt(*v)),
            crate::flowspace::model::ConstValue::UInt(v) => Some(OpKind::ConstUInt(*v)),
            crate::flowspace::model::ConstValue::Int128(v) => Some(OpKind::ConstInt128(*v)),
            crate::flowspace::model::ConstValue::UInt128(v) => Some(OpKind::ConstUInt128(*v)),
            crate::flowspace::model::ConstValue::Float(v) => Some(OpKind::ConstFloat(*v)),
            crate::flowspace::model::ConstValue::Bool(v) => Some(OpKind::ConstBool(*v)),
            crate::flowspace::model::ConstValue::None => Some(OpKind::ConstNone),
            crate::flowspace::model::ConstValue::ByteStr(v) => Some(OpKind::ConstStr(v.clone())),
            _ => None,
        },
    }
}

pub(in crate::front::mir) fn emit_materialized_struct_copy(
    graph: &mut FunctionGraph,
    out: &mut Vec<crate::model::SpaceOperation>,
    owner: &str,
    fields: &[(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )],
    has_vtable: bool,
) -> Variable {
    let result = graph.alloc_value_var();
    out.push(crate::model::SpaceOperation {
        result: Some(result.clone()),
        kind: crate::model::malloc_opkind(owner.to_string(), has_vtable),
    });
    for (field, value, ty) in fields {
        // Keep the typeptr store on the copy. `fuse_boxing_alloc`
        // (`resolve_header_plan`) reads the dominating constant `ob_type`
        // write to fill `NewWithVtable.vtable`; `jtransform.py
        // rewrite_op_setfield` / `is_typeptr_getset` drops it after that
        // capture. Dropping it here leaves `malloc_typed` residual.
        out.push(crate::model::SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: result.clone(),
                field: field.clone(),
                value: value.clone(),
                ty: ty.clone(),
            },
        });
    }
    result
}

pub(in crate::front::mir) fn replace_whole_value_uses_in_kind(
    graph: &mut FunctionGraph,
    out: &mut Vec<crate::model::SpaceOperation>,
    kind: &OpKind,
    result: &Variable,
    owner: &str,
    fields: &[(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )],
    has_vtable: bool,
) -> OpKind {
    match kind {
        OpKind::Call {
            target,
            args,
            result_ty,
        } => {
            // `FunctionReprBase.call` builds one `vlist` and passes those
            // variables to the single `direct_call` / `indirect_call`.
            let copy = emit_materialized_struct_copy(graph, out, owner, fields, has_vtable);
            let args = args
                .iter()
                .map(|arg| {
                    if arg.as_variable() == Some(result) {
                        LinkArg::Value(copy.clone())
                    } else {
                        arg.clone()
                    }
                })
                .collect();
            OpKind::Call {
                target: target.clone(),
                args,
                result_ty: result_ty.clone(),
            }
        }
        OpKind::IndirectCall {
            funcptr,
            args,
            graphs,
            family_key,
            result_ty,
        } => {
            // `get_eval_fn` is an `indirect_call` (`rpbc.py FunctionReprBase.call`).
            // `execute_generator_frame` builds `FrameResumeArgs` and passes
            // `&mut resume`; a stack address would be the walker's C frame.
            // Materialise one `New` so the residual sees a heap pointer, the
            // same escape `Call` already takes. `FunctionReprBase.call` builds
            // one `vlist` and passes those variables to the single
            // `indirect_call`; a repeated argument is the same object.
            let copy = emit_materialized_struct_copy(graph, out, owner, fields, has_vtable);
            let args = args
                .iter()
                .map(|arg| {
                    if arg == result {
                        copy.clone()
                    } else {
                        arg.clone()
                    }
                })
                .collect();
            OpKind::IndirectCall {
                funcptr: funcptr.clone(),
                args,
                graphs: graphs.clone(),
                family_key: family_key.clone(),
                result_ty: result_ty.clone(),
            }
        }
        OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } if value.as_variable() == Some(result) => OpKind::FieldWrite {
            base: base.clone(),
            field: field.clone(),
            value: LinkArg::Value(emit_materialized_struct_copy(
                graph, out, owner, fields, has_vtable,
            )),
            ty: ty.clone(),
        },
        other => {
            let copy = emit_materialized_struct_copy(graph, out, owner, fields, has_vtable);
            crate::inline::remap_op_kind(other, &|var| {
                if var == result {
                    copy.clone()
                } else {
                    var.clone()
                }
            })
        }
    }
}

pub(in crate::front::mir) fn materialize_whole_value_copies_in_block(
    graph: &mut FunctionGraph,
    block_idx: usize,
    ctor_idx: usize,
    result: &Variable,
    owner: &str,
    has_vtable: bool,
) {
    let ops = std::mem::take(&mut graph.blocks[block_idx].operations);
    let mut out: Vec<crate::model::SpaceOperation> = Vec::new();
    let mut fields: Vec<(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )> = Vec::new();
    for (i, op) in ops.into_iter().enumerate() {
        if let OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } = &op.kind
            && base == result
        {
            upsert_struct_field(&mut fields, field.clone(), value.clone(), ty.clone());
        }
        if i > ctor_idx && struct_ctor_kind_has_whole_value_use(&op.kind, result) {
            let kind = replace_whole_value_uses_in_kind(
                graph, &mut out, &op.kind, result, owner, &fields, has_vtable,
            );
            out.push(crate::model::SpaceOperation {
                result: op.result.clone(),
                kind,
            });
        } else {
            out.push(op);
        }
    }
    let mut exits = std::mem::take(&mut graph.blocks[block_idx].exits);
    for link in &mut exits {
        for arg in &mut link.args {
            if arg.as_variable() != Some(result) {
                continue;
            }
            let copy = emit_materialized_struct_copy(graph, &mut out, owner, &fields, has_vtable);
            *arg = LinkArg::Value(copy);
        }
    }
    graph.blocks[block_idx].operations = out;
    graph.blocks[block_idx].exits = exits;
}

/// Variables that still belong to an unfused boxing cluster: each
/// `lltype::malloc[_typed*]` argument, the phis that carry it, and every
/// nested named-struct stored into those aggregates.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded"
)]
pub(in crate::front::mir) fn boxing_cluster_ctor_results(
    graph: &FunctionGraph,
) -> std::collections::HashSet<Variable> {
    use std::collections::HashSet;

    let mut cluster: HashSet<Variable> = HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            if !call_target_is_gc_malloc(target) {
                continue;
            }
            cluster.extend(args.iter().filter_map(LinkArg::as_variable).cloned());
        }
    }
    let mut incoming: std::collections::HashMap<crate::model::BlockId, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    for (pred_idx, pred) in graph.blocks.iter().enumerate() {
        for (link_idx, link) in pred.exits.iter().enumerate() {
            incoming
                .entry(link.target)
                .or_default()
                .push((pred_idx, link_idx));
        }
    }
    let mut growing = true;
    while growing {
        growing = false;
        for block in &graph.blocks {
            for (slot, arg) in block.inputargs.iter().enumerate() {
                if !cluster.contains(arg) {
                    continue;
                }
                let Some(preds) = incoming.get(&block.id) else {
                    continue;
                };
                for &(pred_idx, link_idx) in preds {
                    let link = &graph.blocks[pred_idx].exits[link_idx];
                    if let Some(v) = link.args.get(slot).and_then(LinkArg::as_variable)
                        && cluster.insert(v.clone())
                    {
                        growing = true;
                    }
                }
            }
            for op in &block.operations {
                let OpKind::FieldWrite { base, value, .. } = &op.kind else {
                    continue;
                };
                if !cluster.contains(base) {
                    continue;
                }
                if let Some(v) = value.as_variable()
                    && cluster.insert(v.clone())
                {
                    growing = true;
                }
            }
        }
    }
    cluster
}
