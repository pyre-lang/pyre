//! Iterator adapters → the loop they denote.
//!
//! ## Positioning
//!
//! `Enumerate::next` / `StepBy::next` / `Wtf8CodePoints::next` are foreign
//! (`opacity: Foreign`, body `Opaque` in the extracted ULLBC) — the same
//! class as `core::slice::iter::<Impl>::next`, which `front::iter_next`
//! already rewrites at the *call site* into the native `next` op.  RPython
//! has no iterator adapters (`rlist.py` / `rrange.py` lower explicit loops),
//! so an adapter is the loop it denotes, not a residual callee and not a
//! new opcode.
//!
//! `Enumerate { iter, count }` is
//! ```text
//!     let a = self.iter.next()?;
//!     let i = self.count;
//!     self.count += 1;
//!     Some((i, a))
//! ```
//! (`core::iter::adapters::enumerate::Enumerate::next`) — over a list
//! iterator, RPython's `enumerate(lst)`.  This pass respells
//! `lst.iter().enumerate()` as `iter(__majit_enumerate(lst))`, drops the
//! Opaque constructor, and `front::iter_next` folds `Enumerate::next` into
//! the native `next` op yielding the `(index, item)` tuple the Some arm
//! reads (`rrange.py` `EnumerateIteratorRepr`).
//!
//! Fail-safe: a site whose inner iterator is not a list `iter` op, or
//! whose Option match is not the for-loop diamond, is left as the residual
//! call (census Skip).
//!
//! `Map::collect` is the same class of foreign adapter, but the predicate
//! lives on the **construction site**, not inside the polymorphic shim:
//! the caller holds a concrete closure ADT, so this pass rewrites
//! `map(it, f).collect()` into the loop it denotes and calls that
//! closure's inherent `call_mut` on a reborrow of the env.  `FilterMap::next` / `Iter::position`
//! still stay residual — they need an early-exit match on the predicate
//! result, which is a different CFG.

use crate::flowspace::model::{ConstValue, Variable};
use crate::front::iter_next::{
    BackEdges, is_iter_op_segments, is_iterator_next_target, originates_from_iter_op,
    walk_back_to_source, walk_back_with,
};
use crate::front::option_variant::{close_goto_mixed, emit_narrow, reproduce_exit_args};
use crate::front::result_exc::{back_substitute, op_operand_vars};
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FieldDescriptor, FunctionGraph, Link, LinkArg,
    OpKind, SpaceOperation, ValueType,
};

/// Recognised `Enumerate::next` residual — FunctionPath of the foreign
/// adapter impl, or a Method whose receiver leaf is `Enumerate`.
///
/// `Iterator::enumerate` at a concrete `slice::iter::Iter` is a
/// `TraitDefault` whose `call_target_segments` spell
/// `["slice", "iter", "Iter", "enumerate"]`. `Enumerate::next` is the
/// opaque impl method, spelled `iter::adapters::enumerate::Enumerate::next`
/// after crate-strip, or a Method whose owner leaf is `Enumerate`.
pub(crate) fn is_enumerate_next_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => name == "next" && receiver_root.as_deref().is_some_and(receiver_is_enumerate),
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().map(String::as_str) == Some("next")
                && (adapter_path_ends_with(segments, "enumerate", "next")
                    || segments.iter().any(|s| enumerate_adapter_leaf(s)))
        }
        _ => false,
    }
}

/// `Iterator::enumerate(it)` / `it.enumerate()` — the constructor whose
/// result is the adapter the `next` site iterates.
pub(crate) fn is_enumerate_ctor_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method { name, .. } => name == "enumerate",
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().map(String::as_str) == Some("enumerate")
        }
        CallTarget::Indirect { method_name, .. } => method_name == "enumerate",
        CallTarget::SyntheticTransparentCtor {
            name, owner_path, ..
        } => name_is_enumerate(name) || owner_path.last().is_some_and(|s| name_is_enumerate(s)),
        _ => false,
    }
}

/// `true` iff `path` names the `Enumerate` adapter ADT
/// (`core::iter::adapters::enumerate::Enumerate`).
pub(crate) fn is_enumerate_adapter_path(path: &str) -> bool {
    path.rsplit("::").next().is_some_and(enumerate_adapter_leaf)
}

fn receiver_is_enumerate(root: &str) -> bool {
    root.rsplit("::").next().is_some_and(enumerate_adapter_leaf)
}

fn name_is_enumerate(name: &str) -> bool {
    majit_ir::descr::class_root_leaf(name).split('<').next() == Some("Enumerate")
}

fn enumerate_adapter_leaf(leaf: &str) -> bool {
    majit_ir::descr::class_root_leaf(leaf).split('<').next() == Some("Enumerate")
}

fn adapter_path_ends_with(segments: &[String], adapter: &str, leaf: &str) -> bool {
    segments.last().map(String::as_str) == Some(leaf)
        && segments
            .windows(2)
            .any(|w| w[0] == "adapters" && w[1] == adapter)
}

fn field_write_value(graph: &FunctionGraph, base: &Variable, names: &[&str]) -> Option<Variable> {
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        let OpKind::FieldWrite {
            base: write_base,
            field,
            value: LinkArg::Value(value),
            ..
        } = &op.kind
        else {
            continue;
        };
        if write_base == base && names.contains(&field.name.as_str()) {
            return Some(value.clone());
        }
    }
    None
}

/// Where `lst.iter().enumerate()` is built: the `enumerate` constructor
/// and the list `iter` op the backward walk from its operand reaches.
pub(crate) struct EnumerateSite {
    /// Block and op index of `e = enumerate(it)` / `Enumerate { .. }`.
    ctor_block: usize,
    ctor_op: usize,
    ctor_var: Variable,
    /// The iterator the constructor wraps, in scope in `ctor_block`.
    pub(crate) inner: Variable,
    /// Block and result of `it = core::slice::iter(lst)`.
    iter_block: usize,
    iter_var: Variable,
    container: Variable,
}

impl EnumerateSite {
    /// `v` as [`rewrite_enumerate_to_builtin`] leaves it: the constructor's
    /// value becomes the iterator it wrapped.
    pub(crate) fn renamed(&self, v: &Variable) -> Variable {
        if *v == self.ctor_var {
            self.inner.clone()
        } else {
            v.clone()
        }
    }
}

/// Locate the `enumerate` constructor behind `enum_var` and the list `iter`
/// op beneath it.  The inner iterator may only travel from the `iter` op to
/// the constructor: once the constructor is gone that value is the
/// enumerate iterator, so any other reader would see the wrong iterator.
/// Does not mutate.
pub(crate) fn locate_enumerate_site(
    graph: &FunctionGraph,
    edges: &BackEdges,
    enum_var: &Variable,
) -> Result<EnumerateSite, String> {
    let name = &graph.name;
    // `enum_var` is often the loop-header phi; the constructor lives on
    // the entry edge.  `next(&mut e)` may pass a `same_as` reborrow of
    // that phi; peel it so the walk sees the constructor, not the copy.
    let enum_var = peel_identity_var(graph, enum_var);
    let ctor_var = walk_back_with(graph, edges, &enum_var, |op| match &op.kind {
        OpKind::Call { target, .. } if is_enumerate_ctor_target(target) => op.result.clone(),
        _ => None,
    })
    .ok_or_else(|| format!("{name}: enumerate value has no constructor origin"))?;
    let (ctor_block, ctor_op) = producer_of(graph, &ctor_var)
        .ok_or_else(|| format!("{name}: enumerate constructor origin has no producer op"))?;
    let inner = match &graph.blocks[ctor_block].operations[ctor_op].kind {
        OpKind::Call { target, args, .. }
            if args.len() == 1
                && !matches!(target, CallTarget::SyntheticTransparentCtor { .. }) =>
        {
            args[0].clone().into_variable()
        }
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { .. },
            args,
            ..
        } if args.is_empty() => {
            let block = &graph.blocks[ctor_block];
            let mut inner = None;
            for op in &block.operations {
                let OpKind::FieldWrite {
                    base, field, value, ..
                } = &op.kind
                else {
                    continue;
                };
                if *base != ctor_var {
                    continue;
                }
                match (field.name.as_str(), value) {
                    ("iter" | "__pos_0", LinkArg::Value(v)) => inner = Some(v.clone()),
                    ("count" | "__pos_1", LinkArg::Const(c))
                        if matches!(c.value, ConstValue::Int(0)) => {}
                    ("count" | "__pos_1", LinkArg::Value(v))
                        if block.operations.iter().any(|op| {
                            op.result.as_ref() == Some(v) && matches!(op.kind, OpKind::ConstInt(0))
                        }) => {}
                    _ => {
                        return Err(format!(
                            "{name}: Enumerate aggregate writes {} other than iter/count 0",
                            field.name
                        ));
                    }
                }
            }
            inner.ok_or_else(|| format!("{name}: Enumerate aggregate has no iter field"))?
        }
        _ => {
            return Err(format!(
                "{name}: enumerate origin is not enumerate(it) or an Enumerate aggregate"
            ));
        }
    };
    let inner = peel_identity_var(graph, &inner);
    let iter_var = walk_back_with(graph, edges, &inner, |op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.len() == 1 && is_iter_op_segments(segments) => op.result.clone(),
        _ => None,
    })
    .ok_or_else(|| {
        format!("{name}: Enumerate inner iterator does not originate from an iter op")
    })?;
    let (iter_block, iter_op) = producer_of(graph, &iter_var)
        .ok_or_else(|| format!("{name}: iter op result has no producer"))?;
    let container = match &graph.blocks[iter_block].operations[iter_op].kind {
        OpKind::Call { args, .. } => args[0].clone().into_variable(),
        _ => unreachable!("walk matched a call"),
    };

    // The inner iterator may only travel from the `iter` op to the
    // constructor, and the adapter value only to the `Enumerate::next`
    // sites: once the constructor is gone both are the enumerate iterator.
    // `same_as` / recast copies are the reborrow `next(&mut e)` and a
    // TraitDefault receiver already peel; they must also be legal readers
    // so the closure matches `peel_identity_var`.
    let chain = forward_only_closure(
        graph,
        &iter_var,
        "the iterator enumerate wraps",
        |bi, oi, op| {
            is_identity_copy(&op.kind)
                || (bi == ctor_block
                    && (oi == ctor_op
                        || matches!(&op.kind, OpKind::FieldWrite { base, .. } if *base == ctor_var)))
        },
    )?;
    if !chain.contains(&inner) {
        return Err(format!(
            "{name}: enumerate's operand is not the iter op's value"
        ));
    }
    // The adapter value feeds the for-loop `next`, whatever spelling
    // rustc used (`Enumerate::next`, crate-stripped
    // `Enumerate::next`, or trait-dispatch `Iterator::next`).
    forward_only_closure(graph, &ctor_var, "the enumerate value", |bi, _, op| {
        is_identity_copy(&op.kind)
            || (bi == ctor_block
                && matches!(&op.kind, OpKind::FieldWrite { base, .. } if *base == ctor_var))
            || matches!(&op.kind, OpKind::Call { target, .. } if is_iterator_next_target(target))
    })?;
    Ok(EnumerateSite {
        ctor_block,
        ctor_op,
        ctor_var,
        inner,
        iter_block,
        iter_var,
        container,
    })
}

/// Forward closure of `start` along links.  A member may be read only by an
/// op `reader_ok` accepts (block, op index, op), is never switched on, and a
/// block slot holding one is fed members only.
fn forward_only_closure(
    graph: &FunctionGraph,
    start: &Variable,
    what: &str,
    reader_ok: impl Fn(usize, usize, &SpaceOperation) -> bool,
) -> Result<Vec<Variable>, String> {
    let name = &graph.name;
    let mut chain: Vec<Variable> = vec![start.clone()];
    let mut i = 0;
    while i < chain.len() {
        let v = chain[i].clone();
        i += 1;
        for (bi, block) in graph.blocks.iter().enumerate() {
            let defines = block.inputargs.contains(&v)
                || block
                    .operations
                    .iter()
                    .any(|op| op.result.as_ref() == Some(&v));
            if !defines {
                continue;
            }
            for (oi, op) in block.operations.iter().enumerate() {
                if op_operand_vars(&op.kind).contains(&v) {
                    if !reader_ok(bi, oi, op) {
                        return Err(format!(
                            "{name}: {what} is also read by op {oi} of block {bi}"
                        ));
                    }
                    if is_identity_copy(&op.kind)
                        && let Some(r) = op.result.clone()
                        && !chain.contains(&r)
                    {
                        chain.push(r);
                    }
                }
            }
            let switched = match &block.exitswitch {
                Some(ExitSwitch::Value(sw)) => *sw == v,
                Some(ExitSwitch::Fused { args, .. }) => args.contains(&v),
                _ => false,
            };
            if switched {
                return Err(format!("{name}: {what} is switched on in block {bi}"));
            }
            for link in &block.exits {
                for (p, arg) in link.args.iter().enumerate() {
                    if matches!(arg, LinkArg::Value(a) if *a == v) {
                        let slot = graph.blocks[link.target.0].inputargs[p].clone();
                        if !chain.contains(&slot) {
                            chain.push(slot);
                        }
                    }
                }
            }
        }
    }
    for block in &graph.blocks {
        for link in &block.exits {
            for (p, arg) in link.args.iter().enumerate() {
                let Some(slot) = graph.blocks[link.target.0].inputargs.get(p) else {
                    continue;
                };
                if chain.contains(slot) && !matches!(arg, LinkArg::Value(a) if chain.contains(a)) {
                    return Err(format!(
                        "{name}: a slot holding {what} is fed another value"
                    ));
                }
            }
        }
    }
    Ok(chain)
}

fn producer_of(graph: &FunctionGraph, var: &Variable) -> Option<(usize, usize)> {
    graph.blocks.iter().enumerate().find_map(|(bi, b)| {
        b.operations
            .iter()
            .position(|op| op.result.as_ref() == Some(var))
            .map(|oi| (bi, oi))
    })
}

/// `same_as` and `__cast_instance_intrinsic` alias their operand — the
/// reborrow `next(&mut e)` and a TraitDefault receiver already peel.
fn is_identity_copy(kind: &OpKind) -> bool {
    matches!(kind, OpKind::UnaryOp { op, .. } if op == "same_as")
        || crate::model::cast_instance_root(kind).is_some()
}

/// Follow a `same_as` / recast chain to the value it copies.  `next(&mut e)`
/// and a TraitDefault `enumerate` receiver may each be such a copy of the
/// constructor / `iter` result.
fn peel_identity_var(graph: &FunctionGraph, var: &Variable) -> Variable {
    let mut current = var.clone();
    let mut seen: Vec<Variable> = vec![current.clone()];
    loop {
        let Some((bi, oi)) = producer_of(graph, &current) else {
            break;
        };
        let kind = &graph.blocks[bi].operations[oi].kind;
        let next = if let OpKind::UnaryOp { op, operand, .. } = kind
            && op == "same_as"
        {
            operand.clone()
        } else if let OpKind::Call { args, .. } = kind
            && crate::model::cast_instance_root(kind).is_some()
            && let Some(arg) = args.first()
        {
            arg.clone().into_variable()
        } else {
            break;
        };
        // SSA copy chains are acyclic; `seen` only stops a cyclic graph
        // from hanging the walk.
        if seen.iter().any(|v| *v == next) {
            break;
        }
        seen.push(next.clone());
        current = next;
    }
    current
}

/// Give `lst.iter().enumerate()` the flow-graph shape of RPython's
/// `enumerate(lst)`: `e = __majit_enumerate(lst); it = iter(e)`, with the
/// adapter constructor gone and its value renamed onto the iterator.
/// `builtin_enumerate` then types `e` as `SomeIterator(s_list,
/// "enumerate")`, `iter` of it is itself, and its `next` yields the
/// `(index, item)` tuple (`rrange.py` `EnumerateIteratorRepr`).
pub(crate) fn rewrite_enumerate_to_builtin(graph: &mut FunctionGraph, site: &EnumerateSite) {
    // Retire the constructor and, for an aggregate, its field writes.
    let ctor_var = site.ctor_var.clone();
    let mut oi = 0;
    graph.blocks[site.ctor_block].operations.retain(|op| {
        let keep = oi != site.ctor_op
            && !matches!(&op.kind, OpKind::FieldWrite { base, .. } if *base == ctor_var);
        oi += 1;
        keep
    });
    let inner = site.inner.clone();
    let rename = |v: &Variable| -> Variable {
        if *v == ctor_var {
            inner.clone()
        } else {
            v.clone()
        }
    };
    let block = &mut graph.blocks[site.ctor_block];
    for op in &mut block.operations {
        op.kind = crate::inline::remap_op_kind(&op.kind, &rename);
    }
    let (sw, exits) = crate::model::remap_control_flow_metadata_var(
        &block.exitswitch,
        &block.exits,
        rename,
        |b| b,
    );
    block.exitswitch = sw;
    block.exits = exits;

    // The iter op now walks the enumerate marker instead of the list.  The
    // removals above may have shifted it, so find it again.
    let marker = graph.alloc_value_var();
    let iter_op = graph.blocks[site.iter_block]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.iter_var))
        .expect("the iter op survives the constructor's removal");
    if let OpKind::Call { args, .. } = &mut graph.blocks[site.iter_block].operations[iter_op].kind {
        args[0] = LinkArg::Value(marker.clone());
    }
    graph.blocks[site.iter_block].operations.insert(
        iter_op,
        SpaceOperation {
            result: Some(marker),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![crate::runtime_names::shims::ENUMERATE.to_string()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![site.container.clone()]),
                result_ty: ValueType::Ref(None),
            },
        },
    );
}

/// `SomeTuple(items)` (`annotator/model.py`) and `TupleRepr`
/// (`rtyper/rtuple.py`, chosen by `rtyper.getrepr`) are one shape per
/// item-repr list. The `(count, item)` tuple uses that shape as its owner.
pub(crate) fn enumerate_yield_owner(item_ty: &ValueType) -> String {
    let atom = match item_ty {
        ValueType::Str => "String",
        ValueType::Int => "isize",
        ValueType::Unsigned => "usize",
        ValueType::Float => "f64",
        ValueType::SingleFloat => "f32",
        ValueType::Bool => "bool",
        ValueType::Int128 => "i128",
        ValueType::UInt128 => "u128",
        ValueType::Ref(Some(root)) => root.as_str(),
        ValueType::Ref(None) => "Ptr",
        ValueType::StringBuilder => "StringBuilder",
        ValueType::Void => "()",
        ValueType::State => "State",
        ValueType::Unknown => "Unknown",
    };
    format!("Tuple<usize,{atom}>")
}

/// Paint the reads of the `(index, item)` tuple `base` in `block` with the
/// shape the enumerate `next` yields: `SomeTuple((nonneg int, item))`
/// (`annotator/unaryop.py` `SomeIterator.next`), the item being the list's
/// own item repr, as `ll_listnext` hands it back rather than a pointer to it.
pub(crate) fn paint_enumerate_tuple_reads(
    graph: &mut FunctionGraph,
    block: usize,
    base: &Variable,
    item_ty: &ValueType,
) {
    let owner = enumerate_yield_owner(item_ty);
    for op in &mut graph.blocks[block].operations {
        let OpKind::FieldRead {
            base: read_base,
            field,
            ty,
            ..
        } = &mut op.kind
        else {
            continue;
        };
        if read_base != base {
            continue;
        }
        match field.name.as_str() {
            "__pos_0" => *ty = ValueType::Unsigned,
            "__pos_1" => *ty = item_ty.clone(),
            _ => continue,
        }
        field.owner_root = Some(owner.clone());
    }
}

/// Look up an `Enumerate::next` residual and, if its inner iterator is a
/// list `iter` op, return where the adapter was built.  Does not mutate;
/// the caller validates the diamond first.
pub(crate) fn enumerate_list_inner(
    graph: &FunctionGraph,
    edges: &BackEdges,
    next_target: &CallTarget,
    enum_var: &Variable,
) -> Result<Option<EnumerateSite>, String> {
    // The for-loop `next` is `Enumerate::next` when rustc names the
    // adapter impl, and `Iterator::next` when it trait-dispatches.  Both
    // wrap the same constructor; locate from the iterator value, not the
    // callee spelling.  A named `Enumerate::next` whose inner is not a
    // list `iter` still declines (the residual keeps exhaustion).
    let known_enumerate_next = is_enumerate_next_target(next_target);
    if !known_enumerate_next && !is_iterator_next_target(next_target) {
        return Ok(None);
    }
    match locate_enumerate_site(graph, edges, enum_var) {
        Ok(site) => Ok(Some(site)),
        Err(err) if known_enumerate_next => Err(err),
        Err(_) => Ok(None),
    }
}

/// A recognized `Map::collect` construction site.  Types are resolved
/// while the receiver `Map<I, F>` and destination `Vec<U>` are still
/// MIR `TyRef`s; the post-pass only needs the closure's inherent
/// `call_once` owner and the inner-item / collected-element kinds.
#[derive(Clone)]
pub(crate) struct MapCollectSite {
    /// The `collect` call result (the `Vec`) — locates block A.
    pub result_var: Variable,
    /// The closure env ADT `name_path` — the `call_once` inherent-method owner.
    pub call_once_owner: String,
    /// `I::Item` projected to a [`ValueType`] — the `next` element and the
    /// `(x,)` args-tuple payload the synthesized `call_once` writes.
    pub payload_ty: ValueType,
    /// Concrete class of a reference payload, matching
    /// `option_closure_select::ClosureSelectSite::payload_class_root`.
    pub payload_class_root: Option<String>,
    /// `<X>` suffix for the closure `Args` tuple `(payload,)`.
    pub args_tuple_suffix: String,
    /// `F::Output` / `Vec<U>` element kind — the `call_once` result and
    /// the `Vec::push` item.
    pub call_result_ty: ValueType,
    /// Element kind recorded beside the synthesized `next` so
    /// `front::iter_next` can pick the list vs range answer.
    pub inner_item_ty: ValueType,
    /// Item kind when `Vec<U>` is a one-word header. `None` keeps the GC
    /// list `vec::Vec::{new,push}`.
    pub rust_vec_kind: Option<majit_ir::rvec::VecItemKind>,
}

/// `true` iff `path` names the `Map` adapter ADT
/// (`core::iter::adapters::map::Map`).
pub(crate) fn is_map_adapter_path(path: &str) -> bool {
    path.ends_with("::iter::adapters::map::Map")
}

/// Residual `Map::collect` / `Iterator::collect` / `FromIterator::from_iter`
/// on a `Map` adapter. The capture site still proves the receiver (or the
/// `from_iter` argument) is `Map` and the dest is `Vec`; this only names
/// the leaf so a `HashMap::collect` Method is not rejected before that
/// type gate. rustc may spell `(0..n).map(f).collect()` as `from_iter`.
pub(crate) fn is_map_collect_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method { name, .. } => name == "collect" || name == "from_iter",
        CallTarget::FunctionPath { segments, .. } => matches!(
            segments.last().map(String::as_str),
            Some("collect" | "from_iter")
        ),
        _ => false,
    }
}

pub(crate) fn is_map_ctor_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method { name, .. } => name == "map",
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().map(String::as_str) == Some("map")
        }
        CallTarget::SyntheticTransparentCtor {
            name, owner_path, ..
        } => name == "Map" || owner_path.last().map(String::as_str) == Some("Map"),
        _ => false,
    }
}

fn originates_from_range_ctor(graph: &FunctionGraph, var: &Variable) -> bool {
    walk_back_to_source(graph, var, |op| match &op.kind {
        OpKind::Call {
            target:
                CallTarget::SyntheticTransparentCtor {
                    name, owner_path, ..
                },
            ..
        } if range_ctor_name(name)
            || owner_path.last().is_some_and(|leaf| range_ctor_name(leaf)) =>
        {
            Some(())
        }
        _ => None,
    })
    .is_some()
}

/// `Range` / `Range<usize>` — the aggregate capture in `front::mir` records
/// exclusive int ranges after stripping the per-instantiation suffix, and
/// this walk must see the same ctor or `map(it, f).collect()` leaves
/// `ops::range::Range::map` as an unregistered FunctionPath.
fn range_ctor_name(name: &str) -> bool {
    majit_ir::descr::class_root_leaf(name).split('<').next() == Some("Range")
}

fn inner_is_list_or_range(graph: &FunctionGraph, inner: &Variable) -> bool {
    originates_from_iter_op(graph, inner) || originates_from_range_ctor(graph, inner)
}

fn unique_predecessor(graph: &FunctionGraph, block: usize) -> Result<usize, String> {
    let mut found = None;
    for (i, b) in graph.blocks.iter().enumerate() {
        for link in &b.exits {
            if link.target.0 == block {
                if found.is_some() {
                    return Err(format!(
                        "{}: collect block {block} has multiple predecessors",
                        graph.name
                    ));
                }
                found = Some(i);
            }
        }
    }
    found.ok_or_else(|| format!("{}: collect block {block} has no predecessor", graph.name))
}

fn producer_in_block<'a>(
    graph: &'a FunctionGraph,
    block: usize,
    var: &Variable,
) -> Option<(usize, &'a SpaceOperation)> {
    graph.blocks[block]
        .operations
        .iter()
        .enumerate()
        .find(|(_, op)| op.result.as_ref() == Some(var))
}

struct MapCtor {
    block: usize,
    op_idx: usize,
    inner: Variable,
    env: Variable,
    mapped: Variable,
    aggregate: bool,
}

fn map_ctor_from_op(op: &SpaceOperation, op_idx: usize, block: usize) -> Option<MapCtor> {
    match &op.kind {
        OpKind::Call { target, args, .. } if is_map_ctor_target(target) && args.len() == 2 => {
            Some(MapCtor {
                block,
                op_idx,
                inner: args[0].clone().into_variable(),
                env: args[1].clone().into_variable(),
                mapped: op.result.clone()?,
                aggregate: false,
            })
        }
        OpKind::Call {
            target:
                CallTarget::SyntheticTransparentCtor {
                    name, owner_path, ..
                },
            ..
        } if name == "Map" || owner_path.last().map(String::as_str) == Some("Map") => {
            Some(MapCtor {
                block,
                op_idx,
                inner: Variable::new(),
                env: Variable::new(),
                mapped: op.result.clone()?,
                aggregate: true,
            })
        }
        _ => None,
    }
}

fn locate_map_ctor(
    graph: &FunctionGraph,
    collect_block: usize,
    mapped: &Variable,
) -> Result<MapCtor, String> {
    let name = graph.name.clone();
    if let Some((idx, op)) = producer_in_block(graph, collect_block, mapped)
        && let Some(mut ctor) = map_ctor_from_op(op, idx, collect_block)
    {
        if ctor.aggregate {
            ctor.inner = field_write_value(graph, &ctor.mapped, &["iter", "__pos_0"])
                .ok_or_else(|| format!("{name}: Map aggregate has no iter field write"))?;
            ctor.env = field_write_value(graph, &ctor.mapped, &["f", "__pos_1"])
                .ok_or_else(|| format!("{name}: Map aggregate has no closure field write"))?;
        }
        return Ok(ctor);
    }
    let m = unique_predecessor(graph, collect_block)?;
    let mapped_m = back_substitute(graph, &[(m, collect_block)], mapped, &name)?;
    let (idx, op) = producer_in_block(graph, m, &mapped_m).ok_or_else(|| {
        format!("{name}: Map adapter has no constructor in the collect predecessor")
    })?;
    let mut ctor = map_ctor_from_op(op, idx, m)
        .ok_or_else(|| format!("{name}: collect argument does not originate from map(it, f)"))?;
    if ctor.aggregate {
        ctor.inner = field_write_value(graph, &ctor.mapped, &["iter", "__pos_0"])
            .ok_or_else(|| format!("{name}: Map aggregate has no iter field write"))?;
        ctor.env = field_write_value(graph, &ctor.mapped, &["f", "__pos_1"])
            .ok_or_else(|| format!("{name}: Map aggregate has no closure field write"))?;
    }
    Ok(ctor)
}

fn var_in_block_scope(graph: &FunctionGraph, block: usize, var: &Variable) -> bool {
    graph.blocks[block].inputargs.iter().any(|v| v == var)
        || graph.blocks[block]
            .operations
            .iter()
            .any(|op| op.result.as_ref() == Some(var))
}

fn vec_helper_target(op: majit_ir::rvec::VecOp, kind: majit_ir::rvec::VecItemKind) -> CallTarget {
    CallTarget::function_path(majit_ir::rvec::vec_helper_path(op, kind).split("::"))
}

fn vec_new_call(result: Variable, kind: Option<majit_ir::rvec::VecItemKind>) -> SpaceOperation {
    let (target, result_ty) = match kind {
        Some(kind) => (
            vec_helper_target(majit_ir::rvec::VecOp::NewEmpty, kind),
            ValueType::Int,
        ),
        None => (
            CallTarget::function_path(["vec", "Vec", "new"]),
            ValueType::Ref(None),
        ),
    };
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target,
            args: Vec::new(),
            result_ty,
        },
    }
}

/// `FnMut::call_mut(&mut env, (payload,))`. The receiver is a `same_as`
/// reborrow of the loop-carried env, so the iteration does not consume it.
fn emit_call_mut(
    graph: &mut FunctionGraph,
    block: BlockId,
    env: Variable,
    arg: Option<(Variable, ValueType, Option<String>)>,
    call_owner: &str,
    result_ty: ValueType,
    args_tuple_suffix: &str,
) -> Variable {
    let env_borrow = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(env_borrow.clone()),
        kind: OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand: env,
            result_ty: ValueType::Ref(None),
        },
    });
    let tuple_owner = if arg.is_some() {
        format!("Tuple{args_tuple_suffix}")
    } else {
        "Tuple".to_string()
    };
    let args_tuple = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(args_tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor(&tuple_owner),
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(tuple_owner.clone())),
        },
    });
    if let Some((value, value_ty, class_root)) =
        arg.filter(|_| majit_ir::descr::is_shaped_tuple_name(&tuple_owner))
    {
        let value = emit_narrow(graph, block, value, &class_root);
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: args_tuple.clone(),
                field: FieldDescriptor {
                    name: "__pos_0".to_string(),
                    owner_root: Some(tuple_owner.clone()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(value),
                ty: value_ty,
            },
        });
    }
    let call_result = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(call_result.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("call_mut", Some(call_owner.to_string())),
            args: crate::model::call_args(vec![env_borrow, args_tuple]),
            result_ty,
        },
    });
    call_result
}

fn vec_push_call(
    result: Variable,
    out: Variable,
    item: Variable,
    kind: Option<majit_ir::rvec::VecItemKind>,
) -> SpaceOperation {
    let target = match kind {
        Some(kind) => vec_helper_target(majit_ir::rvec::VecOp::Append, kind),
        None => CallTarget::function_path(["vec", "Vec", "push"]),
    };
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target,
            args: crate::model::call_args(vec![out, item]),
            result_ty: ValueType::Void,
        },
    }
}

/// Rewrite every recorded `map(it, f).collect()` into
/// `out = <empty vec>(); loop { next(it) -> Some(x) => push(out, f(x)); None => break }`.
/// A one-word `Vec<U>` allocates with `ll_vec_newemptylist_*` and pushes
/// with `ll_vec_append_*`; every other element stays `vec::Vec::{new,push}`.
/// The synthesized `next` is returned so `front::range_iter` then
/// `front::iter_next` can fold it.  Fail-safe: a structural mismatch
/// leaves the residual `collect` (census Skip).
pub(crate) fn rewire_map_collect_sites(
    graph: &mut FunctionGraph,
    sites: &[MapCollectSite],
) -> Vec<(Variable, ValueType)> {
    let mut next_results = Vec::new();
    for site in sites {
        if site.call_once_owner.is_empty() {
            continue;
        }
        match rewire_one_map_collect_site(graph, site) {
            Ok(next_opt) => next_results.push((next_opt, site.inner_item_ty.clone())),
            Err(decline) => {
                crate::decline::record_reason(
                    crate::decline::gate::ITER_ADAPTER,
                    "map-collect-site-declined",
                    &decline,
                    &graph.name,
                );
            }
        }
    }
    next_results
}

fn rewire_one_map_collect_site(
    graph: &mut FunctionGraph,
    site: &MapCollectSite,
) -> Result<Variable, String> {
    let name = graph.name.clone();
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: Map::collect result var has no producer block"))?;
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: Map::collect call op not found in block {a}"))?;
    let ops_len = graph.blocks[a].operations.len();
    let (flow_result, _remove_upto) = if ci + 1 == ops_len {
        (site.result_var.clone(), ci)
    } else if ci + 2 == ops_len
        && crate::model::cast_instance_of(
            &graph.blocks[a].operations[ci + 1].kind,
            &site.result_var,
        )
        .is_some()
    {
        let narrowed = graph.blocks[a].operations[ci + 1]
            .result
            .clone()
            .ok_or_else(|| format!("{name}: Map::collect recast has no result"))?;
        (narrowed, ci + 1)
    } else {
        return Err(format!(
            "{name}: Map::collect call is not the last op of block {a}"
        ));
    };
    let mapped = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone().into_variable(),
        other => {
            return Err(format!(
                "{name}: Map::collect producer op is not a 1-arg call: {other:?}"
            ));
        }
    };
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: Map::collect call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: Map::collect call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;
    let ctor = locate_map_ctor(graph, a, &mapped)?;
    if !inner_is_list_or_range(graph, &ctor.inner) {
        return Err(format!(
            "{name}: Map inner iterator is neither a list iter nor an exclusive Range"
        ));
    }
    let emit_block = ctor.block;
    if !var_in_block_scope(graph, emit_block, &ctor.inner)
        || !var_in_block_scope(graph, emit_block, &ctor.env)
    {
        return Err(format!(
            "{name}: map(it, f) operands are not in scope at the constructor"
        ));
    }
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != flow_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }
    let mut carried_at_emit: Vec<Variable> = Vec::new();
    if emit_block == a {
        carried_at_emit = carried.clone();
    } else {
        for v in &carried {
            // Keep a 1:1 slot with `carried` so `reproduce_exit_args` can
            // index `done` inputs by the same position.  Two A-scope
            // values may back-substitute to one M-scope var; collapsing
            // them here made `map_source` index past `done_inputs`.
            carried_at_emit.push(back_substitute(graph, &[(emit_block, a)], v, &name)?);
        }
    }
    for v in &carried_at_emit {
        if !var_in_block_scope(graph, emit_block, v) {
            return Err(format!(
                "{name}: collect continuation value is not in scope at the map constructor"
            ));
        }
    }
    // The constructor block's exit is redirected at the loop header and
    // block A is left with no predecessor, so any op in A before `collect`
    // would never run. A may hold only the collect call and the trailing
    // recast of its result already accepted above.
    if emit_block != a && ci != 0 {
        return Err(format!(
            "{name}: Map::collect block {a} is orphaned when the constructor \
             lives in block {emit_block}; an earlier op would not run"
        ));
    }

    // All structural validation passed; mutate the graph.

    let mapped_origin = ctor.mapped.clone();
    let inner = ctor.inner.clone();
    let env = ctor.env.clone();
    let ctor_idx = ctor.op_idx;
    let aggregate = ctor.aggregate;

    if aggregate {
        graph.blocks[emit_block].operations.retain(|op| {
            !matches!(
                &op.kind,
                OpKind::FieldWrite { base, .. } if *base == mapped_origin
            )
        });
    }
    let ctor_idx = graph.blocks[emit_block]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&mapped_origin))
        .unwrap_or(ctor_idx);
    graph.blocks[emit_block].operations.remove(ctor_idx);

    // Drop the collect call and a trailing recast of its result. When
    // the constructor is in an earlier block, A is orphaned; validation
    // already refused every other op in A. An assignment in the
    // constructor's own block is not in A and still runs.
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: Map::collect call vanished before removal"))?;
    let ops_len = graph.blocks[a].operations.len();
    let last = if ci + 1 < ops_len
        && crate::model::cast_instance_of(
            &graph.blocks[a].operations[ci + 1].kind,
            &site.result_var,
        )
        .is_some()
    {
        ci + 1
    } else {
        ci
    };
    for _ in ci..=last {
        graph.blocks[a].operations.remove(ci);
    }

    let out = graph.alloc_value_var();
    graph.blocks[emit_block]
        .operations
        .push(vec_new_call(out.clone(), site.rust_vec_kind));

    let mut header_sources = vec![inner.clone(), env.clone(), out.clone()];
    for v in &carried_at_emit {
        if !header_sources.contains(v) {
            header_sources.push(v.clone());
        }
    }
    let (header, header_inputs) = graph.create_block_with_arg_vars(header_sources.len());
    let it_h = header_inputs[0].clone();

    let opt = graph.alloc_value_var();
    graph.block_mut(header).operations.push(SpaceOperation {
        result: Some(opt.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("next", None),
            args: crate::model::call_args(vec![it_h]),
            result_ty: ValueType::Ref(None),
        },
    });

    let mut switch_sources = vec![opt.clone()];
    switch_sources.extend(header_inputs.iter().cloned());
    let (switch_bb, switch_inputs) = graph.create_block_with_arg_vars(switch_sources.len());
    let opt_c = switch_inputs[0].clone();
    let it_c = switch_inputs[1].clone();
    let env_c = switch_inputs[2].clone();
    let out_c = switch_inputs[3].clone();
    close_goto_mixed(
        graph,
        header,
        switch_bb,
        switch_sources.iter().cloned().map(LinkArg::Value).collect(),
    );

    let mut body_sources = vec![opt_c.clone(), it_c.clone(), env_c.clone(), out_c.clone()];
    body_sources.extend(switch_inputs.iter().skip(4).cloned());
    let mut done_sources = Vec::with_capacity(1 + carried_at_emit.len());
    for v in &carried_at_emit {
        let idx = header_sources
            .iter()
            .position(|s| s == v)
            .expect("carried map-ctor values are header sources");
        done_sources.push(switch_inputs[1 + idx].clone());
    }
    done_sources.insert(0, out_c.clone());
    let (body_bb, body_inputs) = graph.create_block_with_arg_vars(body_sources.len());
    let (done_bb, done_inputs) = graph.create_block_with_arg_vars(done_sources.len());

    let disc = graph.alloc_value_var();
    graph.block_mut(switch_bb).operations.push(SpaceOperation {
        result: Some(disc.clone()),
        kind: OpKind::FieldRead {
            base: opt_c.clone(),
            field: FieldDescriptor::new("__discriminant", None),
            ty: ValueType::Int,
            pure: true,
        },
    });
    let none_link = Link::new_mixed(
        done_sources.iter().cloned().map(LinkArg::Value).collect(),
        done_bb,
        Some(ExitCase::Const(ConstValue::Int(0))),
    )
    .with_llexitcase_from_exitcase();
    let some_link = Link::new_mixed(
        body_sources.iter().cloned().map(LinkArg::Value).collect(),
        body_bb,
        Some(ExitCase::Const(ConstValue::Int(1))),
    )
    .with_llexitcase_from_exitcase();
    graph.set_control_flow_metadata(
        switch_bb,
        Some(ExitSwitch::Value(disc)),
        vec![none_link, some_link],
    );

    let opt_b = body_inputs[0].clone();
    let it_b = body_inputs[1].clone();
    let env_b = body_inputs[2].clone();
    let out_b = body_inputs[3].clone();
    let payload = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: opt_b,
            field: FieldDescriptor::new("__pos_0", None),
            ty: site.payload_ty.clone(),
            pure: true,
        },
    });
    let call_result = emit_call_mut(
        graph,
        body_bb,
        env_b.clone(),
        Some((
            payload,
            site.payload_ty.clone(),
            site.payload_class_root.clone(),
        )),
        &site.call_once_owner,
        site.call_result_ty.clone(),
        &site.args_tuple_suffix,
    );
    let push_result = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(vec_push_call(
        push_result,
        out_b.clone(),
        call_result,
        site.rust_vec_kind,
    ));
    let mut body_back: Vec<LinkArg> = vec![
        LinkArg::Value(it_b),
        LinkArg::Value(env_b),
        LinkArg::Value(out_b),
    ];
    body_back.extend(body_inputs.iter().skip(4).cloned().map(LinkArg::Value));
    close_goto_mixed(graph, body_bb, header, body_back);

    let out_d = done_inputs[0].clone();
    if carried.len() != done_inputs.len().saturating_sub(1) {
        return Err(format!(
            "{name}: collect continuation arity {} != done extras {}",
            carried.len(),
            done_inputs.len().saturating_sub(1)
        ));
    }
    let done_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &out_d,
        &carried,
        &done_inputs[1..],
        &name,
    )?;
    close_goto_mixed(graph, done_bb, b_target, done_link_args);

    let emit_id = graph.blocks[emit_block].id;
    close_goto_mixed(
        graph,
        emit_id,
        header,
        header_sources.into_iter().map(LinkArg::Value).collect(),
    );
    Ok(opt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, Variable};
    use crate::front::iter_next::{is_iter_op_segments, rewire_next_call_sites};
    use crate::model::{ExitCase, ExitSwitch, Link};

    #[test]
    fn range_ctor_name_accepts_crate_stripped_and_bare_range() {
        assert!(range_ctor_name("ops::range::Range<usize>"));
        assert!(range_ctor_name("Range<usize>"));
        assert!(!range_ctor_name("ops::range::RangeInclusive<usize>"));
    }

    fn enumerate_next_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "iter", "adapters", "enumerate", "<Impl>", "next"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    fn enumerate_ctor_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: [
                "core",
                "iter",
                "traits",
                "iterator",
                "Iterator",
                "enumerate",
            ]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    /// Production TraitDefault spelling of `Iterator::enumerate` at
    /// `core::slice::iter::Iter` (`trait_default_call_segments` strips
    /// the crate and keys the owner ADT).
    fn slice_iter_enumerate_ctor_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["slice", "iter", "Iter", "enumerate"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    /// Crate-stripped opaque `Enumerate::next` impl method.
    fn enumerate_impl_next_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["iter", "adapters", "enumerate", "Enumerate", "next"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    /// Trait-dispatch `Iterator::next` on the Enumerate receiver.
    fn iterator_next_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["Iterator", "next"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    fn iter_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: vec!["core".to_string(), "slice".to_string(), "iter".to_string()],
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    fn count_calls(g: &FunctionGraph, pred: impl Fn(&CallTarget) -> bool) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| matches!(&op.kind, OpKind::Call { target, .. } if pred(target)))
            .count()
    }

    fn call_result_tys(g: &FunctionGraph, pred: impl Fn(&CallTarget) -> bool) -> Vec<ValueType> {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target, result_ty, ..
                } if pred(target) => Some(result_ty.clone()),
                _ => None,
            })
            .collect()
    }

    fn is_enumerate_marker(t: &CallTarget) -> bool {
        matches!(
            t,
            CallTarget::FunctionPath { segments, .. }
                if segments == &[crate::runtime_names::shims::ENUMERATE.to_string()]
        )
    }

    fn next_result_ty(g: &FunctionGraph) -> ValueType {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    result_ty,
                    ..
                } if segments == &["__iter_next".to_string()] => Some(result_ty.clone()),
                _ => None,
            })
            .expect("native next op")
    }

    fn build_enumerate_diamond() -> (crate::model::FunctionGraph, Variable, Variable) {
        build_enumerate_diamond_with(enumerate_ctor_target(), enumerate_next_target())
    }

    fn build_enumerate_diamond_with(
        ctor: CallTarget,
        next: CallTarget,
    ) -> (crate::model::FunctionGraph, Variable, Variable) {
        let mut g = FunctionGraph::new("test_enumerate_next");
        let n = g.startblock;
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "some".to_string(),
                            "container".to_string(),
                            "make".to_string(),
                        ],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: iter_target(),
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let enumer = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: ctor,
                    args: crate::model::call_args(vec![it]),
                    result_ty: ValueType::Ref(Some("Enumerate".into())),
                },
                true,
            )
            .unwrap();

        let (h, h_args) = g.create_block_with_arg_vars(1);
        let enumer_h = h_args[0].clone();
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: next,
                    args: crate::model::call_args(vec![enumer_h.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(2);
        let opt_c = c_args[0].clone();
        let enumer_c = c_args[1].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();

        let (some_t, some_args) = g.create_block_with_arg_vars(2);
        let some_opt = some_args[0].clone();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: some_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Ref(None),
                pure: false,
            },
            true,
        );
        g.set_return(some_t, None);

        let (none_t, _none_args) = g.create_block_with_arg_vars(1);
        g.set_return(none_t, None);

        g.set_goto(n, h, vec![enumer.clone()]);
        g.set_goto(h, c, vec![opt.clone(), enumer_h]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![
                    LinkArg::Value(opt_c.clone()),
                    LinkArg::Value(enumer_c.clone()),
                ],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];
        (g, opt, enumer)
    }

    #[test]
    fn enumerate_next_path_is_recognised() {
        assert!(is_enumerate_next_target(&enumerate_next_target()));
        assert!(is_enumerate_ctor_target(&enumerate_ctor_target()));
        assert!(is_enumerate_ctor_target(&slice_iter_enumerate_ctor_target()));
        assert!(is_enumerate_next_target(&enumerate_impl_next_target()));
        assert!(!is_enumerate_next_target(&iterator_next_target()));
        assert!(!is_enumerate_next_target(&iter_target()));
        assert!(is_enumerate_adapter_path(
            "core::iter::adapters::enumerate::Enumerate"
        ));
    }

    fn assert_enumerate_lowers_to_builtin(mut g: FunctionGraph, opt: Variable, item_ty: ValueType) {
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, item_ty.clone())]);
        assert_eq!(rewritten, 1, "the enumerate for-loop must fold");
        assert_eq!(
            count_calls(&g, is_enumerate_ctor_target),
            0,
            "Iterator::enumerate residual must be gone"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_next_target),
            0,
            "Enumerate::next residual must be gone"
        );
        assert_eq!(
            count_calls(&g, is_iterator_next_target),
            0,
            "Iterator::next residual must be gone"
        );
        let marker_block = g
            .blocks
            .iter()
            .find(|b| {
                b.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::Call { target, .. } if is_enumerate_marker(target))
                })
            })
            .expect("__majit_enumerate(container)");
        let marker = marker_block
            .operations
            .iter()
            .find(
                |op| matches!(&op.kind, OpKind::Call { target, .. } if is_enumerate_marker(target)),
            )
            .and_then(|op| op.result.clone())
            .expect("__majit_enumerate result");
        assert!(
            marker_block.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if is_iter_op_segments(segments)
                    && args == &crate::model::call_args(vec![marker.clone()])
            )),
            "the iter op walks the enumerate marker"
        );
        assert_eq!(
            next_result_ty(&g),
            ValueType::Ref(Some(enumerate_yield_owner(&item_ty))),
            "next yields the (index, item) tuple"
        );
        let adds = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add"))
            .count();
        assert_eq!(adds, 0, "no count arithmetic in the front");
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::SyntheticTransparentCtor { name, .. } if name.starts_with("Tuple<")
            )),
            0,
            "the tuple is next's result, not built on the Some arm"
        );
        assert!(
            matches!(
                g.blocks.iter().find(|b| {
                    b.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::Call {
                                target: CallTarget::FunctionPath { segments, .. },
                                ..
                            } if segments == &["__iter_next".to_string()]
                        )
                    })
                }),
                Some(b) if matches!(b.exitswitch, Some(ExitSwitch::LastException))
            ),
            "next closes with StopIteration"
        );
    }

    /// `for (i, x) in xs.iter().enumerate()` becomes RPython's
    /// `for (i, x) in enumerate(xs)`: `iter(__majit_enumerate(xs))` and a
    /// native `next` that yields the `(i, x)` tuple.  The Opaque
    /// `enumerate` / `Enumerate::next` residuals must not survive, and no
    /// count arithmetic or tuple construction is spelled out in the front.
    #[test]
    fn rewrite_lowers_enumerate_to_the_enumerate_builtin() {
        let (g, opt, _enumer) = build_enumerate_diamond();
        assert_enumerate_lowers_to_builtin(g, opt, ValueType::Ref(None));
    }

    /// Production `for (i, x) in slice.iter().enumerate()` as
    /// `format_missing_err` lowers it: `<[T]>::iter` in its own block,
    /// TraitDefault `["slice", "iter", "Iter", "enumerate"]` in the next,
    /// `Enumerate::next` plus an Option recast in the header, and a Some
    /// arm that recasts to `Option::Some` twice (once per field) before
    /// reading the tuple.
    fn build_production_slice_iter_enumerate() -> (FunctionGraph, Variable) {
        let mut g = FunctionGraph::new("test_production_slice_iter_enumerate");
        let n = g.startblock;
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "some".to_string(),
                            "container".to_string(),
                            "make".to_string(),
                        ],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".to_string(),
                            "slice".to_string(),
                            "<Impl>".to_string(),
                            "iter".to_string(),
                        ],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (e, e_args) = g.create_block_with_arg_vars(1);
        let it_e = e_args[0].clone();
        let enumer = g
            .push_op_var(
                e,
                OpKind::Call {
                    target: slice_iter_enumerate_ctor_target(),
                    args: crate::model::call_args(vec![it_e]),
                    result_ty: ValueType::Ref(Some("Enumerate".into())),
                },
                true,
            )
            .unwrap();

        let (h, h_args) = g.create_block_with_arg_vars(1);
        let enumer_h = h_args[0].clone();
        let raw = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: enumerate_impl_next_target(),
                    args: crate::model::call_args(vec![enumer_h.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let opt = g
            .push_op_var(
                h,
                crate::model::cast_instance_call_result(
                    "Option<(usize,str)>",
                    raw.clone(),
                    ValueType::Ref(Some("Option<(usize,str)>".into())),
                ),
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(2);
        let opt_c = c_args[0].clone();
        let enumer_c = c_args[1].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .unwrap();

        let (some_t, some_args) = g.create_block_with_arg_vars(2);
        let some_opt = some_args[0].clone();
        let some_root = "Option<(usize,str)>::Some";
        let n1 = g
            .push_op_var(
                some_t,
                crate::model::cast_instance_call_result(
                    some_root,
                    some_opt.clone(),
                    ValueType::Ref(Some(some_root.into())),
                ),
                true,
            )
            .unwrap();
        let t1 = g
            .push_op_var(
                some_t,
                OpKind::FieldRead {
                    base: n1,
                    field: FieldDescriptor::new("__pos_0", None),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: t1,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Unsigned,
                pure: false,
            },
            true,
        );
        let n2 = g
            .push_op_var(
                some_t,
                crate::model::cast_instance_call_result(
                    some_root,
                    some_opt,
                    ValueType::Ref(Some(some_root.into())),
                ),
                true,
            )
            .unwrap();
        let t2 = g
            .push_op_var(
                some_t,
                OpKind::FieldRead {
                    base: n2,
                    field: FieldDescriptor::new("__pos_0", None),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: t2,
                field: FieldDescriptor::new("__pos_1", None),
                ty: ValueType::Str,
                pure: false,
            },
            true,
        );
        g.set_return(some_t, None);

        let (none_t, _none_args) = g.create_block_with_arg_vars(1);
        g.set_return(none_t, None);

        g.set_goto(n, e, vec![it]);
        g.set_goto(e, h, vec![enumer]);
        g.set_goto(h, c, vec![opt, enumer_h]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![
                    LinkArg::Value(opt_c.clone()),
                    LinkArg::Value(enumer_c.clone()),
                ],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];
        (g, raw)
    }

    #[test]
    fn rewrite_lowers_slice_iter_enumerate_from_the_production_op_shape() {
        let (g, raw) = build_production_slice_iter_enumerate();
        assert_enumerate_lowers_to_builtin(g, raw, ValueType::Str);
    }

    /// The tuple's item is the base iterator's item kind.  A list of ints
    /// records `Int`; the Some arm's `.1` read must not stay `Ref`.
    #[test]
    fn enumerate_tuple_element_keeps_base_item_kind() {
        let (mut g, opt, _enumer) = build_enumerate_diamond();
        let (some, payload) = g
            .blocks
            .iter()
            .enumerate()
            .find_map(|(bi, block)| {
                block.operations.iter().find_map(|op| match &op.kind {
                    OpKind::FieldRead { field, .. } if field.name == "__pos_0" => {
                        Some((bi, op.result.clone().unwrap()))
                    }
                    _ => None,
                })
            })
            .expect("some arm");
        let some_id = g.blocks[some].id;
        g.push_op_var(
            some_id,
            OpKind::FieldRead {
                base: payload,
                field: FieldDescriptor::new("__pos_1", Some("Tuple<usize,i64>".into())),
                ty: ValueType::Ref(None),
                pure: false,
            },
            true,
        );
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 1, "the enumerate for-loop must fold");
        assert_eq!(
            next_result_ty(&g),
            ValueType::Ref(Some("Tuple<usize,isize>".into()))
        );
        let reads: Vec<(Option<String>, ValueType)> = g.blocks[some]
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead { field, ty, .. } if field.name == "__pos_1" => {
                    Some((field.owner_root.clone(), ty.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            reads,
            vec![(Some("Tuple<usize,isize>".to_string()), ValueType::Int)]
        );
    }

    /// An Enumerate whose inner iterator is not a list `iter` op stays
    /// residual — packing without the StopIteration diamond would drop
    /// exhaustion.
    #[test]
    fn rewrite_declines_enumerate_over_non_list_inner() {
        let mut g = FunctionGraph::new("test_enumerate_non_list");
        let n = g.startblock;
        let foreign = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["foreign".to_string(), "chars".to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let enumer = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: enumerate_ctor_target(),
                    args: crate::model::call_args(vec![foreign]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let opt = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: enumerate_next_target(),
                    args: crate::model::call_args(vec![enumer.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(n, Some(opt.clone()));
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Ref(None))]);
        assert_eq!(rewritten, 0, "non-list inner must decline");
        assert_eq!(
            count_calls(&g, is_enumerate_next_target),
            1,
            "Enumerate::next residual survives a decline"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_ctor_target),
            1,
            "enumerate ctor is not rewritten on a decline"
        );
    }

    fn map_ctor_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "iter", "traits", "iterator", "Iterator", "map"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    fn map_collect_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["iter", "adapters", "map", "Map", "collect"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    fn collect_site(result_var: Variable) -> MapCollectSite {
        MapCollectSite {
            result_var,
            call_once_owner: "test::closure".into(),
            payload_ty: ValueType::Int,
            payload_class_root: None,
            args_tuple_suffix: "<i64>".into(),
            call_result_ty: ValueType::Ref(None),
            inner_item_ty: ValueType::Int,
            rust_vec_kind: None,
        }
    }

    /// `map(it, f).collect()` across two blocks — the Call-terminator
    /// shape Charon emits for an opaque `Iterator::map` then `collect`.
    fn build_map_collect_two_blocks() -> (FunctionGraph, Variable) {
        let mut g = FunctionGraph::new("test_map_collect");
        let n = g.startblock;
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "some".to_string(),
                            "container".to_string(),
                            "make".to_string(),
                        ],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: iter_target(),
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let env = g.push_op_var(n, OpKind::ConstInt(7), true).unwrap();
        let mapped = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: map_ctor_target(),
                    args: crate::model::call_args(vec![it, env]),
                    result_ty: ValueType::Ref(Some("Map".into())),
                },
                true,
            )
            .unwrap();

        let (a, a_args) = g.create_block_with_arg_vars(1);
        let mapped_a = a_args[0].clone();
        let collected = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: map_collect_target(),
                    args: crate::model::call_args(vec![mapped_a]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(n, a, vec![mapped.clone()]);
        g.set_goto(a, b, vec![collected.clone()]);
        (g, collected)
    }

    #[test]
    fn map_collect_path_is_recognised() {
        assert!(is_map_collect_target(&map_collect_target()));
        assert!(is_map_ctor_target(&map_ctor_target()));
        assert!(is_map_adapter_path("core::iter::adapters::map::Map"));
        assert!(!is_map_collect_target(&iter_target()));
        assert!(!is_map_adapter_path(
            "core::iter::adapters::filter_map::FilterMap"
        ));
    }

    /// The `map` constructor is in the predecessor of the collect block.
    /// An assignment in the collect block would be orphaned when that
    /// predecessor's exit is redirected at the loop header, so the
    /// rewrite declines and the assignment stays reachable from the start.
    #[test]
    fn rewrite_keeps_an_assignment_before_map_collect() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let collect_block = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call { target, .. } if is_map_collect_target(target)
                    )
                })
            })
            .expect("collect block");
        let kept = g.alloc_value_var();
        g.blocks[collect_block].operations.insert(
            0,
            SpaceOperation {
                result: Some(kept),
                kind: OpKind::ConstInt(11),
            },
        );
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert!(
            nexts.is_empty(),
            "an assignment in the orphaned collect block must decline"
        );
        assert!(
            reachable_from_start(&g).iter().any(|block| {
                g.blocks[*block]
                    .operations
                    .iter()
                    .any(|op| matches!(op.kind, OpKind::ConstInt(11)))
            }),
            "the assignment before Map::collect stays reachable"
        );
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            1,
            "Map::collect residual survives the decline"
        );
    }

    /// `xs.iter().map(f).collect()` becomes `Vec::new` + `next` +
    /// `call_once` + `Vec::push`.  The Opaque map/collect residuals
    /// must not survive, and the inner list iterator must be the
    /// `next` operand so `front::iter_next` can fold it.
    #[test]
    fn rewrite_lowers_map_collect_to_next_call_once_push() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert_eq!(nexts.len(), 1, "the map.collect chain must fold");
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            0,
            "Map::collect residual must be gone"
        );
        assert_eq!(
            count_calls(&g, is_map_ctor_target),
            0,
            "Iterator::map residual must be gone"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. }
                    if segments == &["vec".to_string(), "Vec".to_string(), "new".to_string()]
            )),
            1,
            "one Vec::new accumulator"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. }
                    if segments == &["vec".to_string(), "Vec".to_string(), "push".to_string()]
            )),
            1,
            "one Vec::push on the Some arm"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_mut")
            ),
            1,
            "the Some arm calls the closure through call_mut"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "next")
            ),
            1,
            "synthesized next on the inner iterator"
        );
        let folded = rewire_next_call_sites(&mut g, &nexts);
        assert_eq!(folded, 1, "the synthesized next diamond must fold");
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. } if segments == &["__iter_next".to_string()]
            )),
            1,
            "native next op on the inner list iterator"
        );
        assert!(
            matches!(
                g.blocks.iter().find(|b| {
                    b.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::Call {
                                target: CallTarget::FunctionPath { segments, .. },
                                ..
                            } if segments == &["__iter_next".to_string()]
                        )
                    })
                }),
                Some(b) if matches!(b.exitswitch, Some(ExitSwitch::LastException))
            ),
            "inner next closes with StopIteration"
        );
    }

    /// A one-word `Vec<U>` collect allocates the raw header and appends
    /// through `ll_vec_*`. The GC `vec::Vec::{new,push}` list stays the
    /// path when `rust_vec_kind` is absent.
    #[test]
    fn rewrite_lowers_word_vec_map_collect_to_ll_vec_helpers() {
        use majit_ir::rvec::{VecItemKind, VecOp, vec_helper_path};
        let (mut g, collected) = build_map_collect_two_blocks();
        let mut site = collect_site(collected);
        site.rust_vec_kind = Some(VecItemKind::Ref);
        let nexts = rewire_map_collect_sites(&mut g, &[site]);
        assert_eq!(nexts.len(), 1, "the map.collect chain must fold");
        let new_path = vec_helper_path(VecOp::NewEmpty, VecItemKind::Ref)
            .split("::")
            .map(str::to_string)
            .collect::<Vec<_>>();
        let append_path = vec_helper_path(VecOp::Append, VecItemKind::Ref)
            .split("::")
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. } if segments == &new_path
            )),
            1,
            "one ll_vec_newemptylist_r accumulator"
        );
        assert_eq!(
            call_result_tys(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. } if segments == &new_path
            )),
            vec![ValueType::Int],
            "the raw header is an int"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. } if segments == &append_path
            )),
            1,
            "one ll_vec_append_r on the Some arm"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. }
                    if segments == &["vec".to_string(), "Vec".to_string(), "new".to_string()]
                        || segments == &["vec".to_string(), "Vec".to_string(), "push".to_string()]
            )),
            0,
            "a one-word collect does not allocate a GC list"
        );
    }

    /// An adapter whose inner iterator is not a list `iter` and not a
    /// Range stays residual — synthesizing `next` without the
    /// StopIteration diamond would drop exhaustion.
    #[test]
    fn rewrite_declines_map_collect_over_non_list_inner() {
        let mut g = FunctionGraph::new("test_map_collect_non_list");
        let n = g.startblock;
        let foreign = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["foreign".to_string(), "chars".to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let env = g.push_op_var(n, OpKind::ConstInt(7), true).unwrap();
        let mapped = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: map_ctor_target(),
                    args: crate::model::call_args(vec![foreign, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (a, a_args) = g.create_block_with_arg_vars(1);
        let mapped_a = a_args[0].clone();
        let collected = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: map_collect_target(),
                    args: crate::model::call_args(vec![mapped_a]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_goto(n, a, vec![mapped]);
        g.set_return(a, Some(collected.clone()));
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert!(nexts.is_empty(), "non-list inner must decline");
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            1,
            "Map::collect residual survives a decline"
        );
        assert_eq!(
            count_calls(&g, is_map_ctor_target),
            1,
            "Iterator::map residual survives a decline"
        );
    }

    /// Empty `call_once_owner` means the capture could not name a
    /// concrete closure ADT.  The rewrite must not guess a callee.
    #[test]
    fn rewrite_declines_when_closure_env_is_not_a_concrete_adt() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let mut site = collect_site(collected);
        site.call_once_owner.clear();
        let nexts = rewire_map_collect_sites(&mut g, &[site]);
        assert!(nexts.is_empty(), "missing call_once owner must decline");
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            1,
            "Map::collect residual survives a missing closure ADT"
        );
    }

    /// The collect continuation may forward several live values besides
    /// the Vec.  Each A-scope slot must keep a matching `done` input —
    /// collapsing duplicate M-scope images used to index past that vec
    /// inside `reproduce_exit_args`.
    #[test]
    fn rewrite_threads_extra_live_values_to_the_continuation() {
        let mut g = FunctionGraph::new("test_map_collect_carried");
        let n = g.startblock;
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "some".to_string(),
                            "container".to_string(),
                            "make".to_string(),
                        ],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: iter_target(),
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let env = g.push_op_var(n, OpKind::ConstInt(7), true).unwrap();
        let extra = g.push_op_var(n, OpKind::ConstInt(9), true).unwrap();
        let mapped = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: map_ctor_target(),
                    args: crate::model::call_args(vec![it, env]),
                    result_ty: ValueType::Ref(Some("Map".into())),
                },
                true,
            )
            .unwrap();

        let (a, a_args) = g.create_block_with_arg_vars(2);
        let mapped_a = a_args[0].clone();
        let extra_a = a_args[1].clone();
        let collected = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: map_collect_target(),
                    args: crate::model::call_args(vec![mapped_a]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(2);
        g.set_return(b, None);
        g.set_goto(n, a, vec![mapped, extra]);
        g.set_goto(a, b, vec![collected.clone(), extra_a]);
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert_eq!(
            nexts.len(),
            1,
            "extra live values must not decline the rewrite"
        );
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            0,
            "Map::collect residual must be gone"
        );
    }

    fn reachable_from_start(g: &FunctionGraph) -> Vec<usize> {
        let mut seen = vec![false; g.blocks.len()];
        let mut stack = vec![g.startblock.0];
        while let Some(block) = stack.pop() {
            if seen.get(block).copied().unwrap_or(true) {
                continue;
            }
            seen[block] = true;
            for link in &g.blocks[block].exits {
                stack.push(link.target.0);
            }
        }
        seen.iter()
            .enumerate()
            .filter_map(|(i, on)| on.then_some(i))
            .collect()
    }

    /// `for _ in xs.iter().enumerate()` never reads the Some payload.
    /// The rewrite still folds, and the tuple carries the inner item type.
    #[test]
    fn rewrite_enumerate_unread_payload_packs_item_type() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let some_id = g.blocks[some].id;
        g.blocks[some].operations.clear();
        g.set_return(some_id, None);
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 1, "an unread Some payload still folds");
        assert_eq!(
            next_result_ty(&g),
            ValueType::Ref(Some("Tuple<usize,isize>".into())),
            "next yields the tuple of the inner item type"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_next_target),
            0,
            "Enumerate::next residual must be gone"
        );
    }

    /// Two enumerate sites with different item types do not share a tuple owner.
    #[test]
    fn two_enumerate_item_types_keep_distinct_owners() {
        let mut g = FunctionGraph::new("two_enumerate_items");
        let mut bases = Vec::new();
        for _ in 0..2 {
            let (b, a) = g.create_block_with_arg_vars(1);
            g.push_op_var(
                b,
                OpKind::FieldRead {
                    base: a[0].clone(),
                    field: FieldDescriptor::new("__pos_1", None),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            );
            bases.push((b, a[0].clone()));
        }
        paint_enumerate_tuple_reads(&mut g, bases[0].0.0, &bases[0].1, &ValueType::Int);
        paint_enumerate_tuple_reads(&mut g, bases[1].0.0, &bases[1].1, &ValueType::Str);
        let owners: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. } if field.name == "__pos_1" => {
                    field.owner_root.clone()
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            owners,
            vec![
                "Tuple<usize,isize>".to_string(),
                "Tuple<usize,String>".to_string()
            ]
        );
    }

    /// A successor that reads `__pos_0` of the forwarded Option is a live
    /// use of the Some payload the fold does not rewrite: decline.
    #[test]
    fn rewrite_declines_forward_into_a_reading_successor() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let carrier = g.blocks[some].inputargs[0].clone();
        let some_id = g.blocks[some].id;
        let (succ, succ_args) = g.create_block_with_arg_vars(1);
        g.push_op_var(
            succ,
            OpKind::FieldRead {
                base: succ_args[0].clone(),
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Unsigned,
                pure: false,
            },
            true,
        )
        .unwrap();
        g.set_return(succ, None);
        g.block_mut(some_id).exits = vec![
            Link::new_mixed(vec![LinkArg::Value(carrier)], succ, None).with_prevblock(some_id),
        ];
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Str)]);
        assert_eq!(rewritten, 0, "a live forward of the payload must decline");
        assert_eq!(count_calls(&g, is_enumerate_ctor_target), 1);
        assert_eq!(count_calls(&g, is_enumerate_marker), 0);
    }

    /// An `Enumerate { iter, count: 0 }` aggregate built before the `iter`
    /// op in the same block: removing it must not shift the marker onto
    /// the wrong op.
    #[test]
    fn rewrite_handles_an_aggregate_built_before_the_iter_op() {
        let (mut g, opt, enumer) = build_enumerate_diamond();
        let n = g.startblock.0;
        let ops = std::mem::take(&mut g.blocks[n].operations);
        let (container_op, iter_op) = (ops[0].clone(), ops[1].clone());
        let container = container_op.result.clone().unwrap();
        let it = iter_op.result.clone().unwrap();
        g.blocks[n].operations = vec![
            container_op,
            SpaceOperation {
                result: Some(enumer.clone()),
                kind: OpKind::Call {
                    target: CallTarget::synthetic_transparent_struct_ctor(
                        vec!["Enumerate".to_string()],
                        "Enumerate",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("Enumerate".into())),
                },
            },
            SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: enumer.clone(),
                    field: FieldDescriptor::new("count", Some("Enumerate".into())),
                    value: LinkArg::Const(crate::flowspace::model::Constant::new(ConstValue::Int(
                        0,
                    ))),
                    ty: ValueType::Unsigned,
                },
            },
            iter_op,
            SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: enumer.clone(),
                    field: FieldDescriptor::new("iter", Some("Enumerate".into())),
                    value: LinkArg::Value(it.clone()),
                    ty: ValueType::Ref(None),
                },
            },
        ];
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 1, "the aggregate enumerate folds");
        let ops = &g.blocks[n].operations;
        assert_eq!(ops.len(), 3, "{ops:#?}");
        let marker = ops[1].result.clone().unwrap();
        assert!(matches!(
            &ops[1].kind,
            OpKind::Call { target, args, .. }
                if is_enumerate_marker(target)
                    && args == &crate::model::call_args(vec![container])
        ));
        assert_eq!(ops[2].result.as_ref(), Some(&it));
        assert!(matches!(
            &ops[2].kind,
            OpKind::Call { target, args, .. }
                if *target == iter_target() && args == &crate::model::call_args(vec![marker])
        ));
        assert!(
            g.blocks[n].exits[0]
                .args
                .iter()
                .all(|a| !matches!(a, LinkArg::Value(v) if *v == enumer)),
            "the adapter value is renamed onto the iterator"
        );
    }

    /// Another reader of the adapter value would see the enumerate
    /// iterator after the rewrite: decline.
    #[test]
    fn rewrite_declines_an_enumerate_value_read_elsewhere() {
        let (mut g, opt, enumer) = build_enumerate_diamond();
        let n = g.startblock;
        g.push_op_var(
            n,
            OpKind::Call {
                target: CallTarget::function_path(["uses", "adapter"]),
                args: crate::model::call_args(vec![enumer]),
                result_ty: ValueType::Void,
            },
            false,
        );
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 0, "a second reader must decline");
        assert_eq!(count_calls(&g, is_enumerate_ctor_target), 1);
        assert_eq!(count_calls(&g, is_enumerate_marker), 0);
    }

    /// `Enumerate::next` reading the constructor's value in its own block
    /// reads the renamed iterator afterwards.
    #[test]
    fn rewrite_renames_a_next_reading_the_constructor_value() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let h = g
            .blocks
            .iter()
            .position(|b| {
                b.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::Call { target, .. } if is_enumerate_next_target(target))
                })
            })
            .expect("next block");
        let container = g.alloc_value_var();
        let it = g.alloc_value_var();
        let e = g.alloc_value_var();
        let mut prefix = vec![
            SpaceOperation {
                result: Some(container.clone()),
                kind: OpKind::Call {
                    target: CallTarget::function_path(["some", "container", "make"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
            },
            SpaceOperation {
                result: Some(it.clone()),
                kind: OpKind::Call {
                    target: iter_target(),
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
            },
            SpaceOperation {
                result: Some(e.clone()),
                kind: OpKind::Call {
                    target: enumerate_ctor_target(),
                    args: crate::model::call_args(vec![it.clone()]),
                    result_ty: ValueType::Ref(None),
                },
            },
        ];
        for op in &mut g.blocks[h].operations {
            if let OpKind::Call { target, args, .. } = &mut op.kind
                && is_enumerate_next_target(target)
            {
                *args = crate::model::call_args(vec![e.clone()]);
            }
        }
        prefix.append(&mut g.blocks[h].operations);
        g.blocks[h].operations = prefix;
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 1);
        let next_arg = g.blocks[h]
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments == &["__iter_next".to_string()] => Some(args[0].clone()),
                _ => None,
            })
            .expect("native next");
        assert_eq!(next_arg, LinkArg::Value(it));
        assert!(
            g.blocks[h]
                .operations
                .iter()
                .all(|op| op.result.as_ref() != Some(&e)),
            "the constructor is gone"
        );
    }

    /// The `(i, x)` tuple forwarded to a successor keeps that successor's
    /// item reads on the old shape: decline.
    #[test]
    fn rewrite_declines_a_forwarded_tuple() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let (some, payload) = g
            .blocks
            .iter()
            .enumerate()
            .find_map(|(bi, block)| {
                block.operations.iter().find_map(|op| match &op.kind {
                    OpKind::FieldRead { field, .. } if field.name == "__pos_0" => {
                        Some((bi, op.result.clone().unwrap()))
                    }
                    _ => None,
                })
            })
            .expect("some arm");
        let some_id = g.blocks[some].id;
        let (tail, _) = g.create_block_with_arg_vars(1);
        g.set_return(tail, None);
        g.set_goto(some_id, tail, vec![payload]);
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 0, "a forwarded tuple must decline");
        assert_eq!(count_calls(&g, is_enumerate_marker), 0);
    }

    /// A merge successor has another predecessor that still passes the old
    /// payload. Painting its `__pos_N` reads would retarget that edge too.
    #[test]
    fn rewrite_declines_forward_into_merge_successor() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let carrier = g.blocks[some].inputargs[0].clone();
        let some_id = g.blocks[some].id;
        let (succ, succ_args) = g.create_block_with_arg_vars(1);
        g.push_op_var(
            succ,
            OpKind::FieldRead {
                base: succ_args[0].clone(),
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Unsigned,
                pure: false,
            },
            true,
        )
        .unwrap();
        g.set_return(succ, None);
        let (other, _) = g.create_block_with_arg_vars(0);
        let other_val = g.alloc_value_var();
        g.block_mut(other).exits = vec![
            Link::new_mixed(vec![LinkArg::Value(other_val)], succ, None).with_prevblock(other),
        ];
        g.block_mut(some_id).exits = vec![
            Link::new_mixed(vec![LinkArg::Value(carrier)], succ, None).with_prevblock(some_id),
        ];
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Str)]);
        assert_eq!(rewritten, 0, "a merge successor must decline");
        assert_eq!(count_calls(&g, is_enumerate_ctor_target), 1);
    }

    /// Switching on the forwarded slot is a use of the old payload shape.
    #[test]
    fn rewrite_declines_successor_switching_on_forwarded_slot() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let carrier = g.blocks[some].inputargs[0].clone();
        let some_id = g.blocks[some].id;
        let (succ, succ_args) = g.create_block_with_arg_vars(1);
        let received = succ_args[0].clone();
        g.push_op_var(
            succ,
            OpKind::FieldRead {
                base: received.clone(),
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Unsigned,
                pure: false,
            },
            true,
        )
        .unwrap();
        g.set_return(succ, None);
        g.block_mut(succ).exitswitch = Some(ExitSwitch::Value(received));
        g.block_mut(some_id).exits = vec![
            Link::new_mixed(vec![LinkArg::Value(carrier)], succ, None).with_prevblock(some_id),
        ];
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Str)]);
        assert_eq!(
            rewritten, 0,
            "an exitswitch on the forwarded slot must decline"
        );
        assert_eq!(count_calls(&g, is_enumerate_ctor_target), 1);
    }

    /// A Some arm that uses the payload slot for something other than
    /// `__pos_0` declines before the enumerate constructor is rewritten.
    #[test]
    fn rewrite_declines_enumerate_payload_used_outside_pos0() {
        let (mut g, opt, _) = build_enumerate_diamond();
        for block in &mut g.blocks {
            for op in &mut block.operations {
                if let OpKind::FieldRead { base, field, .. } = &op.kind
                    && field.name == "__pos_0"
                {
                    let base = base.clone();
                    op.kind = OpKind::Call {
                        target: CallTarget::function_path(["uses", "payload"]),
                        args: crate::model::call_args(vec![base]),
                        result_ty: ValueType::Ref(None),
                    };
                }
            }
        }
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Ref(None))]);
        assert_eq!(rewritten, 0, "a non-__pos_0 payload use must decline");
        assert_eq!(
            count_calls(&g, is_enumerate_ctor_target),
            1,
            "enumerate ctor is not rewritten on a decline"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_marker),
            0,
            "the enumerate marker is not installed on a decline"
        );
    }

    /// A `__pos_0` read plus a forward of the same carrier is still a
    /// second reference. The rewrite must decline before the constructor
    /// is replaced.
    #[test]
    fn rewrite_declines_enumerate_payload_forwarded_beside_pos0() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let carrier = g.blocks[some].inputargs[0].clone();
        g.blocks[some].exits[0].args.push(LinkArg::Value(carrier));
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Ref(None))]);
        assert_eq!(
            rewritten, 0,
            "forwarding the carrier beside __pos_0 must decline"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_ctor_target),
            1,
            "enumerate ctor is not rewritten on a decline"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_marker),
            0,
            "the enumerate marker is not installed on a decline"
        );
    }

    /// A carrier forwarded beside `__pos_0` into a slot no block ever reads
    /// is a dead chain: it is pruned and the enumerate fold goes ahead.
    #[test]
    fn rewrite_prunes_enumerate_payload_forwarded_into_a_dead_slot() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let carrier = g.blocks[some].inputargs[0].clone();
        let (tail, _tail_args) = g.create_block_with_arg_vars(1);
        g.set_return(tail, None);
        let some_id = g.blocks[some].id;
        g.set_goto(some_id, tail, vec![carrier]);
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Ref(None))]);
        assert_eq!(
            rewritten, 1,
            "a dead forward of the carrier must not decline"
        );
        assert!(
            g.block(tail).inputargs.is_empty(),
            "the dead slot is pruned from the successor"
        );
        assert!(
            g.blocks[some].exits.iter().all(|link| link.args.is_empty()),
            "the forwarding link drops the pruned slot"
        );
        assert_eq!(count_calls(&g, is_enumerate_ctor_target), 0);
    }

    /// A trailing recast is accepted only when it recasts the collect result.
    #[test]
    fn rewrite_declines_recast_of_a_different_value_after_collect() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let collect_block = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call { target, .. } if is_map_collect_target(target)
                    )
                })
            })
            .expect("collect block");
        let other = g.alloc_value_var();
        let narrowed = g.alloc_value_var();
        g.blocks[collect_block].operations.push(SpaceOperation {
            result: Some(narrowed),
            kind: OpKind::Call {
                target: CallTarget::function_path(["__cast_instance_intrinsic"]),
                args: vec![
                    LinkArg::from(other),
                    LinkArg::from(ConstValue::byte_str("Vec")),
                ],
                result_ty: ValueType::Ref(Some("Vec".into())),
            },
        });
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert!(
            nexts.is_empty(),
            "a recast of some other value must decline"
        );
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            1,
            "Map::collect residual survives a foreign recast"
        );
    }

    /// `(0..n).map(move |_| owned.len()).collect()` reuses one closure.
    /// `FnOnce::call_once` moves the env; the header then reads that moved
    /// value. `FnMut::call_mut` takes a reborrow (`same_as` of the
    /// loop-carried env) and the back edge carries the original env.
    #[test]
    fn map_collect_calls_closure_through_call_mut_reborrow() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert_eq!(nexts.len(), 1, "the map.collect chain must fold");
        let (name, args) = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    args,
                    ..
                } if name == "call_once" || name == "call_mut" => {
                    Some((name.clone(), args.clone()))
                }
                _ => None,
            })
            .expect("the loop body must call the closure");
        assert_eq!(
            name, "call_mut",
            "the closure is FnMut, reused each iteration"
        );
        let receiver = args[0].clone().into_variable();
        let env = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match (&op.kind, &op.result) {
                (OpKind::UnaryOp { op, operand, .. }, Some(result))
                    if op == "same_as" && result == &receiver =>
                {
                    Some(operand.clone())
                }
                _ => None,
            })
            .expect("call_mut receiver must be a reborrow of the loop env");
        assert_ne!(receiver, env, "the reborrow is not the loop-carried env");
        let threads_original = g.blocks.iter().any(|b| {
            b.exits.iter().any(|link| {
                link.args
                    .iter()
                    .any(|arg| matches!(arg, LinkArg::Value(v) if v == &env))
                    && link
                        .args
                        .iter()
                        .all(|arg| !matches!(arg, LinkArg::Value(v) if v == &receiver))
            })
        });
        assert!(
            threads_original,
            "the loop header must carry the original env, not the reborrow"
        );
    }

    /// `(0..n).map(f).collect()` — Charon spells the exclusive range as
    /// `Range<usize>`, matching `front::mir`'s aggregate capture which
    /// strips the suffix. A literal `"Range"` probe left `Range::map` as
    /// an unregistered FunctionPath (`call_args_and_c_profile_args`).
    #[test]
    fn rewrite_lowers_suffixed_range_map_collect() {
        let mut g = FunctionGraph::new("test_range_map_collect");
        let n = g.startblock;
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["core".to_string(), "ops".to_string(), "range".to_string()],
                        "Range<usize>".to_string(),
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::Range".into())),
                },
                true,
            )
            .unwrap();
        let env = g.push_op_var(n, OpKind::ConstInt(7), true).unwrap();
        let mapped = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::function_path(["ops", "range", "Range", "map"]),
                    args: crate::model::call_args(vec![range, env]),
                    result_ty: ValueType::Ref(Some("Map".into())),
                },
                true,
            )
            .unwrap();
        let collected = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: map_collect_target(),
                    args: crate::model::call_args(vec![mapped]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (ret, _) = g.create_block_with_arg_vars(1);
        g.set_return(ret, None);
        g.set_goto(n, ret, vec![collected.clone()]);
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert_eq!(nexts.len(), 1, "Range<usize>.map.collect must fold");
        assert_eq!(
            count_calls(&g, is_map_ctor_target),
            0,
            "Range::map residual must be gone"
        );
    }
}
