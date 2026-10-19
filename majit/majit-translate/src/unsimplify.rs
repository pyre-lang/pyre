//! Rich-model graph transformations from `translator/unsimplify.py`.

use std::collections::{HashMap, HashSet};

use crate::codewriter::jtransform::{JitMarkerKey, jit_marker_key_from_target};
use crate::codewriter::type_state::ConcreteType;
use crate::flowspace::model::Variable;
use crate::model::{BlockId, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation};

/// Best available spelling for one variable: the `Input`-op source name if
/// the graph carries one, else the `Variable` name prefix, else its identity.
fn var_label(graph: &FunctionGraph, var: &Variable) -> String {
    graph
        .value_name_for(var)
        .or_else(|| var.renamed().then(|| var.name_prefix()))
        .unwrap_or_else(|| format!("<unnamed id={}>", var.id()))
}

/// Short human label for one operation: the opcode spelling where the model
/// carries one (`BinOp`/`UnaryOp`/`Call`/`Input`), else the `OpKind` variant
/// name.
fn op_label(op: &SpaceOperation) -> String {
    match &op.kind {
        crate::model::OpKind::BinOp { op, .. } | crate::model::OpKind::UnaryOp { op, .. } => {
            op.clone()
        }
        crate::model::OpKind::Input { name, .. } => format!("Input {name}"),
        crate::model::OpKind::Call { target, .. } => match target.path_segments() {
            Some(segments) => format!("call {}", segments.join("::")),
            None => "call".to_string(),
        },
        other => {
            // Every `OpKind` variant's `Debug` starts with its own name.
            let debug = format!("{other:?}");
            let end = debug.find([' ', '(', '{']).unwrap_or(debug.len());
            debug[..end].to_string()
        }
    }
}

/// Where `var` is produced: the operation that results in it, or the
/// block whose `inputargs` it enters through.
fn definition_site(graph: &FunctionGraph, var: &Variable) -> String {
    if let Some(site) = producing_op(graph, var) {
        return format!("defined by {site}");
    }
    for block in &graph.blocks {
        let Some(position) = block.inputargs.iter().position(|arg| arg.id() == var.id()) else {
            continue;
        };
        // A block inputarg has no defining op in this graph; the value it
        // stands for is whatever each predecessor passes in that slot, so
        // resolve one hop back through the incoming links.
        let mut incoming: Vec<String> = Vec::new();
        for pred in &graph.blocks {
            for link in &pred.exits {
                if link.target != block.id {
                    continue;
                }
                let origin = match link.args.get(position) {
                    Some(LinkArg::Value(source)) => {
                        origin_of(graph, source, &mut HashSet::new(), ORIGIN_WALK_DEPTH)
                    }
                    Some(other) => format!("{other:?}"),
                    None => "<link arity mismatch>".to_string(),
                };
                incoming.push(format!("from block {} as {origin}", pred.id.0));
            }
        }
        let mut site = format!("enters as inputarg {position} of block {}", block.id.0);
        if !incoming.is_empty() {
            site.push_str(&format!(" ({})", incoming.join("; ")));
        }
        return site;
    }
    "no definition site found in this graph".to_string()
}

/// Hop budget for [`origin_of`]. A forwarded value crosses a handful of blocks
/// at most; the bound is what keeps a loop in the link graph from recursing
/// without end, alongside the visited set.
const ORIGIN_WALK_DEPTH: usize = 16;

/// `block N op I \`label\`` for the operation whose result is `var`.
fn producing_op(graph: &FunctionGraph, var: &Variable) -> Option<String> {
    for block in &graph.blocks {
        for (index, op) in block.operations.iter().enumerate() {
            if op.result.as_ref().is_some_and(|res| res.id() == var.id()) {
                return Some(format!(
                    "`{}` at block {} op {index}",
                    op_label(op),
                    block.id.0,
                ));
            }
        }
    }
    None
}

enum IncomingSource<'a> {
    Value(&'a Variable),
    Other(&'a LinkArg),
    Missing,
}

/// One hop from a block inputarg to the values supplied by predecessor links.
fn incoming_sources<'a>(
    graph: &'a FunctionGraph,
    var: &Variable,
) -> Option<Vec<IncomingSource<'a>>> {
    let (block, position) = graph.blocks.iter().find_map(|block| {
        block
            .inputargs
            .iter()
            .position(|arg| arg.id() == var.id())
            .map(|position| (block.id, position))
    })?;
    Some(
        graph
            .blocks
            .iter()
            .flat_map(|pred| pred.exits.iter().filter(move |link| link.target == block))
            .map(|link| match link.args.get(position) {
                Some(LinkArg::Value(source)) => IncomingSource::Value(source),
                Some(other) => IncomingSource::Other(other),
                None => IncomingSource::Missing,
            })
            .collect(),
    )
}

#[derive(Default)]
struct OriginSummary {
    origins: Vec<String>,
    shared_predecessors: usize,
}

impl OriginSummary {
    fn resolved(origin: String) -> Self {
        Self {
            origins: vec![origin],
            shared_predecessors: 0,
        }
    }

    fn extend(&mut self, other: Self) {
        for origin in other.origins {
            if !self.origins.contains(&origin) {
                self.origins.push(origin);
            }
        }
        self.shared_predecessors += other.shared_predecessors;
    }
}

/// Resolve a value back to the operation that actually produces it, walking
/// through block inputargs for as long as the value is only being forwarded.
///
/// A block inputarg has no defining operation, so resolving one hop lands on
/// another inputarg whenever a value is threaded through a chain of blocks —
/// and the report then stops at a bare `id=`, which names nothing. Each hop
/// takes the incoming link argument in the same position; distinct origins are
/// reported side by side because a block with several predecessors genuinely
/// has several.
fn origin_of(
    graph: &FunctionGraph,
    var: &Variable,
    seen: &mut HashSet<u64>,
    depth: usize,
) -> String {
    let summary = origin_summary(graph, var, seen, depth);
    if summary.origins.is_empty() {
        return format!("id={}", var.id());
    }
    let mut origins = summary.origins.join(" | ");
    if summary.shared_predecessors != 0 {
        let description = if summary.origins.len() == 1 {
            "that origin"
        } else {
            "those origins"
        };
        origins.push_str(&format!(
            " (+{} predecessors sharing {description})",
            summary.shared_predecessors
        ));
    }
    origins
}

fn origin_summary(
    graph: &FunctionGraph,
    var: &Variable,
    seen: &mut HashSet<u64>,
    depth: usize,
) -> OriginSummary {
    if let Some(site) = producing_op(graph, var) {
        return OriginSummary::resolved(site);
    }
    if depth == 0 {
        return OriginSummary::resolved(format!("id={}", var.id()));
    }
    if !seen.insert(var.id()) {
        return OriginSummary {
            origins: Vec::new(),
            shared_predecessors: 1,
        };
    }
    let Some(incoming) = incoming_sources(graph, var) else {
        return OriginSummary::resolved(format!("id={}", var.id()));
    };
    let mut summary = OriginSummary::default();
    for source in incoming {
        let origin = match source {
            IncomingSource::Value(source) => origin_summary(graph, source, seen, depth - 1),
            IncomingSource::Other(other) => OriginSummary::resolved(format!("{other:?}")),
            IncomingSource::Missing => OriginSummary::resolved("<link arity mismatch>".to_string()),
        };
        summary.extend(origin);
    }
    if summary.origins.is_empty() && summary.shared_predecessors == 0 {
        OriginSummary::resolved(format!("id={}", var.id()))
    } else {
        summary
    }
}

/// Every use of a variable at or after the split point, keyed by
/// `Variable::id`: the moved operations first, then the block's outgoing
/// link arguments and exit switch.
///
/// `links` and `exitswitch` must be the pre-rename snapshots — `split_block`
/// rewrites both to the fresh copies before the `_forcelink` check runs.
fn use_sites(
    moved_source: &[SpaceOperation],
    links: &[Link],
    exitswitch: &Option<ExitSwitch>,
) -> Vec<(u64, String)> {
    let mut uses: Vec<(u64, String)> = Vec::new();
    for (index, op) in moved_source.iter().enumerate() {
        for arg in crate::inline::op_variable_refs(&op.kind) {
            uses.push((arg.id(), format!("op {index} `{}`", op_label(op))));
        }
    }
    for link in links {
        for (position, arg) in link.args.iter().enumerate() {
            if let LinkArg::Value(source) = arg {
                uses.push((
                    source.id(),
                    format!("link arg {position} to block {}", link.target.0),
                ));
            }
        }
    }
    match exitswitch {
        Some(ExitSwitch::Value(var)) => uses.push((var.id(), "exitswitch".to_string())),
        Some(ExitSwitch::Fused { opname, args }) => {
            for var in args {
                uses.push((var.id(), format!("exitswitch `{opname}`")));
            }
        }
        Some(ExitSwitch::LastException) | None => {}
    }
    uses
}

fn jit_merge_point_receiver(op: &SpaceOperation) -> Option<&Variable> {
    let OpKind::Call { target, args, .. } = &op.kind else {
        return None;
    };
    let receiver_root = target.receiver_root()?;
    let driver_roots = [receiver_root.to_string()];
    (jit_marker_key_from_target(target, &driver_roots) == Some(JitMarkerKey::JitMergePoint))
        .then(|| args.first().and_then(|a| a.as_variable()))
        .flatten()
}

fn origin_producer(graph: &FunctionGraph, var: &Variable) -> Option<SpaceOperation> {
    let mut current = var.clone();
    let mut seen = HashSet::new();
    for _ in 0..ORIGIN_WALK_DEPTH {
        if !seen.insert(current.id()) {
            return None;
        }
        if let Some(op) = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find(|op| {
                op.result
                    .as_ref()
                    .is_some_and(|result| result.id() == current.id())
            })
        {
            return Some(op.clone());
        }

        let mut incoming = incoming_sources(graph, &current)?.into_iter();
        let Some(IncomingSource::Value(source)) = incoming.next() else {
            return None;
        };
        let source = source.clone();
        if incoming
            .any(|arg| !matches!(arg, IncomingSource::Value(other) if other.id() == source.id()))
        {
            return None;
        }
        current = source;
    }
    None
}

/// RPython `split_block` recreates a Void marker receiver with `same_as`, and
/// `remove_same_as` later substitutes the constant. Before rtyping, mirror that
/// rematerialisation by copying only the constant producer used for operand 0;
/// this is the driver constant from `ExtEnterLeaveMarker.specialize_call`.
fn rematerializable_marker_receiver_producer(
    graph: &FunctionGraph,
    var: &Variable,
) -> Option<SpaceOperation> {
    let producer = origin_producer(graph, var)?;
    if !crate::inline::op_variable_refs(&producer.kind).is_empty() {
        return None;
    }
    let can_copy = match &producer.kind {
        OpKind::ConstInt(_)
        | OpKind::ConstFnAddr { .. }
        | OpKind::ConstUInt(_)
        | OpKind::ConstInt128(_)
        | OpKind::ConstUInt128(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstSymbolic { .. }
        | OpKind::ConstFloat(_)
        | OpKind::ConstStr(_)
        | OpKind::ConstInternedStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstNone
        | OpKind::ConstRefAddr(_) => true,
        OpKind::Call { args, .. } => args.is_empty(),
        _ => false,
    };
    can_copy.then_some(producer)
}

/// `translator/unsimplify.py split_block` for the rich model graph.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable Variable identity; concretetype/name cells do not participate"
)]
pub(crate) fn split_block(
    graph: &mut FunctionGraph,
    block: BlockId,
    index: usize,
    forcelink: Option<&[Variable]>,
) -> BlockId {
    let operation_len = graph.block(block).operations.len();
    assert!(
        index <= operation_len,
        "split_block: index out of range (got {}, len={})",
        index,
        operation_len,
    );
    if matches!(
        graph.block(block).exitswitch,
        Some(ExitSwitch::LastException)
    ) {
        assert!(
            index < operation_len,
            "split_block: cannot split a LastException block after its raising operation"
        );
    }

    // `unsimplify.py split_block`: `varmap.keys()` is consumed in first-
    // insertion order, so keep that order independently of HashMap iteration.
    let mut varmap_order: Vec<Variable> = Vec::new();
    let mut varmap: HashMap<Variable, Variable> = HashMap::new();
    let mut vars_produced_in_new_block: HashSet<Variable> = HashSet::new();

    fn intern_var(
        var: &Variable,
        varmap_order: &mut Vec<Variable>,
        varmap: &mut HashMap<Variable, Variable>,
        vars_produced_in_new_block: &HashSet<Variable>,
    ) {
        if vars_produced_in_new_block.contains(var) {
            return;
        }
        if !varmap.contains_key(var) {
            varmap_order.push(var.clone());
            varmap.insert(var.clone(), var.copy());
        }
    }

    fn get_new_name(
        var: &Variable,
        varmap_order: &mut Vec<Variable>,
        varmap: &mut HashMap<Variable, Variable>,
        vars_produced_in_new_block: &HashSet<Variable>,
    ) -> Variable {
        if vars_produced_in_new_block.contains(var) {
            return var.clone();
        }
        intern_var(var, varmap_order, varmap, vars_produced_in_new_block);
        varmap
            .get(var)
            .cloned()
            .expect("split_block: interned Variable missing from varmap")
    }

    let moved_source = graph.block(block).operations[index..].to_vec();
    let mut moved_operations = Vec::with_capacity(moved_source.len());
    for op in &moved_source {
        // `unsimplify.py split_block`: operands are interned before the op is
        // rewritten, and the result becomes produced only after that rewrite.
        for arg in crate::inline::op_variable_refs(&op.kind) {
            intern_var(
                &arg,
                &mut varmap_order,
                &mut varmap,
                &vars_produced_in_new_block,
            );
        }
        let remap_var = |var: &Variable| varmap.get(var).cloned().unwrap_or_else(|| var.clone());
        moved_operations.push(SpaceOperation {
            result: op.result.clone(),
            kind: crate::inline::remap_op_kind(&op.kind, &remap_var),
        });
        if let Some(result) = &op.result {
            vars_produced_in_new_block.insert(result.clone());
        }
    }

    let (mut links, exitswitch) = {
        let old = graph.block_mut(block);
        (std::mem::take(&mut old.exits), old.exitswitch.take())
    };
    // The `_forcelink` diagnostic below reports use sites in the caller's own
    // variable identities; the loops that follow rewrite `links` and
    // `exitswitch` in place to the fresh copies, so snapshot them first.
    let pre_rename_links = forcelink.is_some().then(|| links.clone());
    let pre_rename_exitswitch = forcelink.is_some().then(|| exitswitch.clone());
    for link in &mut links {
        for arg in &mut link.args {
            if Some(&*arg) == link.last_exception.as_ref()
                || Some(&*arg) == link.last_exc_value.as_ref()
            {
                continue;
            }
            if let LinkArg::Value(var) = arg {
                *var = get_new_name(
                    var,
                    &mut varmap_order,
                    &mut varmap,
                    &vars_produced_in_new_block,
                );
            }
        }
    }
    let renamed_exitswitch = exitswitch.map(|switch| match switch {
        ExitSwitch::Value(var) => ExitSwitch::Value(get_new_name(
            &var,
            &mut varmap_order,
            &mut varmap,
            &vars_produced_in_new_block,
        )),
        ExitSwitch::LastException => ExitSwitch::LastException,
        ExitSwitch::Fused { opname, args } => ExitSwitch::Fused {
            opname,
            args: crate::model::call_args(args)
                .iter()
                .map(|var| {
                    get_new_name(
                        var,
                        &mut varmap_order,
                        &mut varmap,
                        &vars_produced_in_new_block,
                    )
                })
                .collect(),
        },
    });

    let linkargs = if let Some(forcelink) = forcelink {
        assert_eq!(
            index, 0,
            "unsimplify.py split_block _forcelink requires index == 0"
        );
        let linkargs = forcelink.to_vec();
        let mut missing: Vec<&Variable> = varmap_order
            .iter()
            .filter(|var| !linkargs.contains(var))
            .collect();
        // `_forcelink` requires index 0, and `split_before_jit_merge_point` splits at
        // the marker found with configured roots. Only `moved_source[0]` can qualify;
        // using its own root satisfies the classifier's root check by construction.
        let marker_receiver_id = moved_source
            .first()
            .and_then(jit_merge_point_receiver)
            .map(Variable::id);
        let mut rematerialized_ids = Vec::new();
        for var in &missing {
            let rematerialized_kind = if FunctionGraph::concretetype_of(var) == ConcreteType::Void {
                // `unsimplify.py split_block`: recreate every omitted Void
                // value as `same_as(Constant(None, lltype.Void))`.  The rich
                // model represents the constant-producing operation directly;
                // later lowering treats `ConstNone` as lltype.Void.
                Some(OpKind::ConstNone)
            } else if marker_receiver_id == Some(var.id()) {
                // Pre-rtyping structural adaptation for the driver singleton:
                // its source graph may still expose a ref/unknown receiver
                // where upstream has already rtyped it to Void.
                rematerializable_marker_receiver_producer(graph, var).map(|producer| producer.kind)
            } else {
                None
            };
            let Some(kind) = rematerialized_kind else {
                continue;
            };
            let result = varmap
                .get(*var)
                .cloned()
                .expect("split_block: missing Variable has no fresh mapping");
            // Upstream inserts `same_as` immediately before the first operation
            // that consumes the fresh variable (or at the end when only an
            // outgoing link/exitswitch needs it).
            let insert_at = moved_operations
                .iter()
                .position(|op| crate::inline::op_variable_refs(&op.kind).contains(&result))
                .unwrap_or(moved_operations.len());
            rematerialized_ids.push(var.id());
            moved_operations.insert(
                insert_at,
                SpaceOperation {
                    result: Some(result),
                    kind,
                },
            );
        }
        missing.retain(|var| !rematerialized_ids.contains(&var.id()));
        if !missing.is_empty() {
            let uses = use_sites(
                &moved_source,
                pre_rename_links.as_deref().unwrap_or(&[]),
                pre_rename_exitswitch.as_ref().unwrap_or(&None),
            );
            let offenders: Vec<String> = missing
                .iter()
                .map(|var| {
                    let name = var_label(graph, var);
                    let mut where_used: Vec<&str> = uses
                        .iter()
                        .filter(|(id, _)| *id == var.id())
                        .map(|(_, site)| site.as_str())
                        .collect();
                    where_used.dedup();
                    format!(
                        "  {name} (id={}, kind={:?})\n    {}\n    used after the split point by: {}",
                        var.id(),
                        FunctionGraph::concretetype_of(var),
                        definition_site(graph, var),
                        if where_used.is_empty() {
                            "<none found>".to_string()
                        } else {
                            where_used.join(", ")
                        },
                    )
                })
                .collect();
            let names: Vec<String> = missing.iter().map(|var| var_label(graph, var)).collect();
            panic!(
                "The variable {} was not explicitly listed in _forcelink.  \
                 This issue can be caused by a jitdriver.jit_merge_point() where some variable \
                 containing an int or str or instance is actually known to be constant, e.g. \
                 always 42.\n\
                 Each variable below is defined before the split point and still live after it, \
                 but is neither a green nor a red of the driver; the graph must be restructured \
                 so that it is recomputed after the marker, or declared in the driver.\n\
                 Offending variables ({} of them):\n{}",
                names.join(", "),
                missing.len(),
                offenders.join("\n"),
            );
        }
        linkargs
    } else {
        varmap_order.clone()
    };

    let new_inputargs = linkargs
        .iter()
        .map(|var| {
            get_new_name(
                var,
                &mut varmap_order,
                &mut varmap,
                &vars_produced_in_new_block,
            )
        })
        .collect();
    let newblock = graph.create_block();
    {
        let new = graph.block_mut(newblock);
        // Set inputargs before constructing the old block's checked Link.
        new.inputargs = new_inputargs;
        new.operations = moved_operations;
        new.exitswitch = renamed_exitswitch;
    }
    graph.recloseblock(newblock, links);

    let link = Link::from_variables(graph, linkargs, newblock, None);
    {
        let old = graph.block_mut(block);
        old.operations.truncate(index);
        old.exitswitch = None;
    }
    graph.recloseblock(block, vec![link]);
    newblock
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::type_state::ConcreteType;
    use crate::model::{OpKind, ValueType};

    fn unary(operand: &Variable, result: &Variable) -> SpaceOperation {
        SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::UnaryOp {
                op: "int_neg".into(),
                operand: operand.clone(),
                result_ty: ValueType::Int,
            },
        }
    }

    fn graph_with_ops(
        inputargs: Vec<Variable>,
        operations: Vec<SpaceOperation>,
        returned: Variable,
    ) -> FunctionGraph {
        let mut graph = FunctionGraph::new("split_fixture");
        let start = graph.startblock;
        graph.block_mut(start).inputargs = inputargs;
        let returnblock = graph.returnblock;
        graph.block_mut(returnblock).inputargs = vec![returned.copy()];
        graph.block_mut(start).operations = operations;
        let exit = Link::from_variables(&graph, vec![returned], returnblock, None);
        graph.recloseblock(start, vec![exit]);
        graph
    }

    #[test]
    fn split_at_zero_moves_all_operations() {
        let a = Variable::named("a");
        let r = Variable::named("r");
        let mut graph = graph_with_ops(vec![a.clone()], vec![unary(&a, &r)], r);
        let old = graph.startblock;

        let new = split_block(&mut graph, old, 0, None);

        assert!(graph.block(old).operations.is_empty());
        assert_eq!(graph.block(new).operations.len(), 1);
        assert_eq!(graph.block(old).exits[0].args, vec![LinkArg::Value(a)]);
        assert_ne!(
            graph.block(new).inputargs[0].id(),
            graph.block(old).inputargs[0].id()
        );
    }

    #[test]
    fn split_in_middle_threads_intermediate() {
        let a = Variable::named("a");
        let middle = Variable::named("middle");
        let result = Variable::named("result");
        let mut graph = graph_with_ops(
            vec![a.clone()],
            vec![unary(&a, &middle), unary(&middle, &result)],
            result,
        );
        let old = graph.startblock;

        let new = split_block(&mut graph, old, 1, None);

        assert_eq!(graph.block(old).operations.len(), 1);
        assert_eq!(graph.block(new).operations.len(), 1);
        assert_eq!(
            graph.block(old).exits[0].args,
            vec![LinkArg::Value(middle.clone())]
        );
        let OpKind::UnaryOp { operand, .. } = &graph.block(new).operations[0].kind else {
            panic!("moved operation must remain UnaryOp")
        };
        assert_eq!(operand, &graph.block(new).inputargs[0]);
        assert_ne!(operand.id(), middle.id());
    }

    #[test]
    fn split_at_end_moves_original_exits() {
        let a = Variable::named("a");
        let result = Variable::named("result");
        let mut graph = graph_with_ops(vec![a.clone()], vec![unary(&a, &result)], result.clone());
        let old = graph.startblock;

        let new = split_block(&mut graph, old, 1, None);

        assert_eq!(graph.block(old).operations.len(), 1);
        assert!(graph.block(new).operations.is_empty());
        assert_eq!(graph.block(old).exits[0].args, vec![LinkArg::Value(result)]);
        assert_eq!(graph.block(new).exits.len(), 1);
    }

    #[test]
    #[should_panic(expected = "cannot split a LastException block")]
    fn split_at_end_of_last_exception_block_is_rejected() {
        let a = Variable::named("a");
        let result = Variable::named("result");
        let mut graph = graph_with_ops(vec![a.clone()], vec![unary(&a, &result)], result);
        let old = graph.startblock;
        graph.block_mut(old).exitswitch = Some(ExitSwitch::LastException);

        split_block(&mut graph, old, 1, None);
    }

    #[test]
    fn split_with_forcelink_uses_explicit_order() {
        let a = Variable::named("a");
        let b = Variable::named("b");
        let result = Variable::named("result");
        let mut graph = graph_with_ops(
            vec![a.clone(), b.clone()],
            vec![SpaceOperation {
                result: Some(result.clone()),
                kind: OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: a.clone(),
                    rhs: b.clone(),
                    result_ty: ValueType::Int,
                },
            }],
            result,
        );
        let old = graph.startblock;

        let new = split_block(&mut graph, old, 0, Some(&[b.clone(), a.clone()]));

        assert_eq!(
            graph.block(old).exits[0].args,
            vec![LinkArg::Value(b), LinkArg::Value(a)]
        );
        assert_eq!(graph.block(new).inputargs.len(), 2);
    }

    #[test]
    fn split_with_forcelink_rematerializes_every_omitted_void() {
        let kept = Variable::named("kept");
        let omitted_void = Variable::named("omitted_void");
        FunctionGraph::set_concretetype_of_inline(&omitted_void, ConcreteType::Void);
        let result = Variable::named("result");
        let mut graph = graph_with_ops(
            vec![kept.clone(), omitted_void.clone()],
            vec![SpaceOperation {
                result: Some(result.clone()),
                kind: OpKind::Call {
                    target: crate::model::CallTarget::function_path(["consume_void"]),
                    args: crate::model::call_args(vec![omitted_void]),
                    result_ty: ValueType::Void,
                },
            }],
            result,
        );
        let old = graph.startblock;

        let new = split_block(&mut graph, old, 0, Some(&[kept]));

        assert_eq!(graph.block(new).inputargs.len(), 1);
        assert!(matches!(
            graph.block(new).operations[0].kind,
            OpKind::ConstNone
        ));
        let recreated = graph.block(new).operations[0]
            .result
            .as_ref()
            .expect("ConstNone must recreate the omitted Void variable");
        assert_eq!(
            FunctionGraph::concretetype_of(recreated),
            ConcreteType::Void
        );
        let OpKind::Call { args, .. } = &graph.block(new).operations[1].kind else {
            panic!("original consumer must remain a Call")
        };
        assert_eq!(args, &[recreated.clone()]);
    }

    fn assert_split_rematerializes_marker_receiver(producer_kind: OpKind) {
        let receiver = Variable::named("pypyjitdriver");
        let marker_result = Variable::named("marker_result");
        let marker_kind = OpKind::Call {
            target: crate::model::CallTarget::method(
                "jit_merge_point",
                Some("PyPyJitDriver".into()),
            ),
            args: crate::model::call_args(vec![receiver.clone()]),
            result_ty: ValueType::Void,
        };
        let mut graph = graph_with_ops(
            vec![],
            vec![
                SpaceOperation {
                    result: Some(receiver.clone()),
                    kind: producer_kind.clone(),
                },
                SpaceOperation {
                    result: Some(marker_result.clone()),
                    kind: marker_kind,
                },
            ],
            marker_result,
        );
        let old = graph.startblock;
        let portal = split_block(&mut graph, old, 1, None);
        let portal_receiver = jit_merge_point_receiver(&graph.block(portal).operations[0])
            .expect("split portal must start with the marker receiver")
            .clone();

        let new = split_block(&mut graph, portal, 0, Some(&[]));

        assert!(graph.block(portal).exits[0].args.is_empty());
        assert!(graph.block(new).inputargs.is_empty());
        assert_eq!(graph.block(new).operations[0].kind, producer_kind);
        let rematerialized = graph.block(new).operations[0]
            .result
            .as_ref()
            .expect("copied producer must have a result");
        assert_ne!(rematerialized.id(), receiver.id());
        assert_ne!(rematerialized.id(), portal_receiver.id());
        let OpKind::Call { args, .. } = &graph.block(new).operations[1].kind else {
            panic!("moved marker must remain a Call")
        };
        assert_eq!(args, &[rematerialized.clone()]);
    }

    #[test]
    fn split_with_forcelink_rematerializes_nullary_marker_receiver() {
        assert_split_rematerializes_marker_receiver(OpKind::Call {
            target: crate::model::CallTarget::function_path(["pyre_jit", "eval", "pypyjitdriver"]),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("PyPyJitDriver".into())),
        });
    }

    #[test]
    fn split_with_forcelink_rematerializes_constref_marker_receiver() {
        assert_split_rematerializes_marker_receiver(OpKind::ConstRefAddr(8));
    }

    #[test]
    fn split_with_forcelink_does_not_rematerialize_marker_at_index_one() {
        let receiver = Variable::named("pypyjitdriver");
        let filler = Variable::named("filler");
        let marker_result = Variable::named("marker_result");
        let mut graph = graph_with_ops(
            vec![],
            vec![
                SpaceOperation {
                    result: Some(receiver.clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::function_path([
                            "pyre_jit",
                            "eval",
                            "pypyjitdriver",
                        ]),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("PyPyJitDriver".into())),
                    },
                },
                SpaceOperation {
                    result: Some(filler),
                    kind: OpKind::ConstNone,
                },
                SpaceOperation {
                    result: Some(marker_result.clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::method(
                            "jit_merge_point",
                            Some("PyPyJitDriver".into()),
                        ),
                        args: crate::model::call_args(vec![receiver]),
                        result_ty: ValueType::Void,
                    },
                },
            ],
            marker_result,
        );
        let old = graph.startblock;
        let portal = split_block(&mut graph, old, 1, None);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            split_block(&mut graph, portal, 0, Some(&[]))
        }))
        .expect_err("a marker receiver used only by operation 1 must remain an offender");
        let message = panic
            .downcast_ref::<String>()
            .expect("split_block panics with a formatted String");

        assert!(message.contains("pypyjitdriver"), "{message}");
        assert!(message.contains("op 1 `call jit_merge_point`"), "{message}");
    }

    #[test]
    fn split_with_forcelink_panic_locates_every_offender() {
        let kept = Variable::named("kept");
        let missing = Variable::named("missing_red");
        FunctionGraph::set_concretetype_of_inline(&missing, ConcreteType::Signed);
        let result = Variable::named("result");
        let mut graph = graph_with_ops(
            vec![kept.clone(), missing.clone()],
            vec![unary(&missing, &result)],
            result,
        );
        let naming_block = graph.create_block();
        graph
            .block_mut(naming_block)
            .operations
            .push(SpaceOperation {
                result: Some(missing),
                kind: OpKind::Input {
                    name: "missing_red".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
            });
        let old = graph.startblock;

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            split_block(&mut graph, old, 0, Some(&[kept]))
        }))
        .expect_err("a live-across variable outside _forcelink must fail the split");
        let message = panic
            .downcast_ref::<String>()
            .expect("split_block panics with a formatted String");

        // The diagnostic has to be actionable on its own: which value, what
        // kind it is, where it comes from, and what still needs it.
        assert!(message.contains("missing_red"), "{message}");
        assert!(message.contains("kind=Signed"), "{message}");
        assert!(
            message.contains("defined by `Input missing_red` at block"),
            "{message}"
        );
        assert!(
            message.contains("used after the split point by: op 0 `int_neg`"),
            "{message}"
        );
        assert!(
            message.contains("Offending variables (1 of them)"),
            "{message}"
        );
    }
}
