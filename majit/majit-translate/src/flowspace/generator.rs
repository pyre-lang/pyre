//! Flow-graph building for Python generators.
//!
//! RPython upstream: `rpython/flowspace/generator.py` (177 LOC).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use indexmap::IndexMap;

use super::flowcontext::{FlowContextError, FlowingError};
use super::model::{
    Block, BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc,
    Hlvalue, HostObject, Link, LinkRef, SpaceOperation, Variable, checkgraph,
};
use super::pygraph::PyGraph;

/// RPython `rpython/flowspace/generator.py AbstractPosition` — `class
/// AbstractPosition`. Marker used by generator graphs to tag
/// `Entry` / `Resume<n>` subclasses.
pub trait AbstractPosition: core::fmt::Debug {}

/// RPython `generator.py` — `get_variable_names(variables)`.
pub fn get_variable_names(variables: &[&str]) -> Vec<String> {
    use std::collections::HashSet;

    let mut seen: HashSet<String> = HashSet::new();
    let mut result: Vec<String> = Vec::new();
    for v in variables {
        let mut name: String = v.trim_matches('_').to_string();
        while seen.contains(&name) {
            name.push('_');
        }
        result.push(format!("g_{name}"));
        seen.insert(name);
    }
    result
}

fn const_host(obj: HostObject) -> Hlvalue {
    Hlvalue::Constant(Constant::new(ConstValue::HostObject(obj)))
}

fn const_str(value: &str) -> Hlvalue {
    Hlvalue::Constant(Constant::new(ConstValue::byte_str(value)))
}

fn get_variable_names_from_hlvalues(
    variables: &[Hlvalue],
) -> Result<Vec<String>, FlowContextError> {
    let mut names = Vec::with_capacity(variables.len());
    for value in variables {
        let Hlvalue::Variable(v) = value else {
            return Err(FlowContextError::Flowing(FlowingError::new(
                "generator bootstrap expects Variable inputargs",
            )));
        };
        names.push(v.name_prefix().to_string());
    }
    let refs: Vec<_> = names.iter().map(|name| name.as_str()).collect();
    Ok(get_variable_names(&refs))
}

fn tuple_of_strings(items: &[String]) -> ConstValue {
    ConstValue::Tuple(items.iter().map(ConstValue::byte_str).collect())
}

pub fn make_generatoriterator_class(func: &GraphFunc, var_names: &[String]) -> HostObject {
    let entry = HostObject::new_class_with_members(
        format!("{}.Entry", func.name),
        vec![],
        IndexMap::from([("varnames".to_string(), tuple_of_strings(var_names))]),
    );
    HostObject::new_class_with_members(
        format!("{}.GeneratorIterator", func.name),
        vec![],
        IndexMap::from([("Entry".to_string(), ConstValue::HostObject(entry))]),
    )
}

fn generator_entry_class(generator_iterator: &HostObject) -> Result<HostObject, FlowContextError> {
    match generator_iterator.class_get("Entry") {
        Some(ConstValue::HostObject(obj)) => Ok(obj),
        _ => Err(FlowContextError::Flowing(FlowingError::new(
            "generator iterator missing Entry class",
        ))),
    }
}

fn entry_varnames(entry: &HostObject) -> Result<Vec<String>, FlowContextError> {
    match entry.class_get("varnames") {
        Some(ConstValue::Tuple(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let Some(value) = item.as_text().map(str::to_owned) else {
                    return Err(FlowContextError::Flowing(FlowingError::new(
                        "generator Entry.varnames must be a tuple of strings",
                    )));
                };
                out.push(value);
            }
            Ok(out)
        }
        _ => Err(FlowContextError::Flowing(FlowingError::new(
            "generator Entry class missing varnames",
        ))),
    }
}

pub fn replace_graph_with_bootstrap(
    generator_iterator: &HostObject,
    graph: &mut FunctionGraph,
    var_names: &[String],
) -> Result<(), FlowContextError> {
    let entry = generator_entry_class(generator_iterator)?;
    let original_args = graph.startblock.borrow().inputargs.clone();
    let newblock = Block::shared(original_args.clone());
    let mut ops = Vec::new();

    let op_entry = SpaceOperation::new(
        "simple_call",
        vec![const_host(entry)],
        Hlvalue::Variable(Variable::named("entry")),
    );
    let v_entry = op_entry.result.clone();
    ops.push(op_entry);

    for (value, name) in original_args.into_iter().zip(var_names.iter()) {
        ops.push(SpaceOperation::new(
            "setattr",
            vec![v_entry.clone(), const_str(name), value],
            Hlvalue::Variable(Variable::new()),
        ));
    }

    let op_generator = SpaceOperation::new(
        "simple_call",
        vec![const_host(generator_iterator.clone()), v_entry],
        Hlvalue::Variable(Variable::named("generator")),
    );
    let generator_result = op_generator.result.clone();
    ops.push(op_generator);

    newblock.borrow_mut().operations = ops;
    let ret = Link::new(
        vec![generator_result],
        Some(graph.returnblock.clone()),
        None,
    )
    .into_ref();
    newblock.closeblock(vec![ret]);
    graph.startblock = newblock;
    Ok(())
}

pub fn attach_next_method(
    generator_iterator: &HostObject,
    graph: &FunctionGraph,
) -> Result<(), FlowContextError> {
    let func = graph.func.clone().ok_or_else(|| {
        FlowContextError::Flowing(FlowingError::new(
            "generator graph missing GraphFunc metadata",
        ))
    })?;
    let mut next_func = func.clone();
    next_func.name = format!("{}__next", next_func.name);
    next_func._generator_next_method_of_ = Some(generator_iterator.clone());
    generator_iterator.class_set(
        "next",
        ConstValue::HostObject(HostObject::new_user_function(next_func)),
    );
    Ok(())
}

fn insert_empty_startblock(graph: &mut FunctionGraph) {
    let vars: Vec<Hlvalue> = graph
        .startblock
        .borrow()
        .inputargs
        .iter()
        .map(|arg| match arg {
            Hlvalue::Variable(v) => Hlvalue::Variable(v.copy()),
            Hlvalue::Constant(c) => Hlvalue::Constant(c.clone()),
        })
        .collect();
    let newblock = Block::shared(vars.clone());
    let link = Link::new(vars, Some(graph.startblock.clone()), None).into_ref();
    newblock.closeblock(vec![link]);
    graph.startblock = newblock;
}

fn insert_reads(block: &BlockRef, varnames: &[String]) -> Result<(), FlowContextError> {
    let original_args = block.borrow().inputargs.clone();
    if original_args.len() != varnames.len() {
        return Err(FlowContextError::Flowing(FlowingError::new(
            "generator read insertion arity mismatch",
        )));
    }
    let entry = Hlvalue::Variable(Variable::named("entry"));
    let mut ops = Vec::with_capacity(original_args.len() + block.borrow().operations.len());
    for (arg, name) in original_args.iter().cloned().zip(varnames.iter()) {
        ops.push(SpaceOperation::new(
            "getattr",
            vec![entry.clone(), const_str(name)],
            arg,
        ));
    }
    ops.extend(block.borrow().operations.clone());
    let mut b = block.borrow_mut();
    b.operations = ops;
    b.inputargs = vec![entry];
    Ok(())
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn get_new_name(
    value: &Hlvalue,
    vars_produced_in_new_block: &[Variable],
    varmap: &mut HashMap<Variable, Variable>,
) -> Hlvalue {
    match value {
        Hlvalue::Constant(c) => Hlvalue::Constant(c.clone()),
        Hlvalue::Variable(v) => {
            if vars_produced_in_new_block
                .iter()
                .any(|produced| produced == v)
            {
                Hlvalue::Variable(v.clone())
            } else {
                let replacement = varmap.entry(v.clone()).or_insert_with(|| v.copy()).clone();
                Hlvalue::Variable(replacement)
            }
        }
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn split_block(block: &BlockRef, index: usize) -> Result<LinkRef, FlowContextError> {
    let mut varmap: HashMap<Variable, Variable> = HashMap::new();
    let mut vars_produced_in_new_block: Vec<Variable> = Vec::new();

    let (moved_original, exits, exitswitch) = {
        let b = block.borrow();
        if index > b.operations.len() {
            return Err(FlowContextError::Flowing(FlowingError::new(
                "split_block index out of bounds",
            )));
        }
        (
            b.operations[index..].to_vec(),
            b.exits.clone(),
            b.exitswitch.clone(),
        )
    };

    let mut moved_operations = Vec::with_capacity(moved_original.len());
    for op in moved_original {
        let repl: Vec<Hlvalue> = op
            .args
            .iter()
            .map(|arg| get_new_name(arg, &vars_produced_in_new_block, &mut varmap))
            .collect();
        let newop = SpaceOperation::with_offset(op.opname, repl, op.result.clone(), op.offset);
        if let Hlvalue::Variable(v) = &op.result {
            vars_produced_in_new_block.push(v.clone());
        }
        moved_operations.push(newop);
    }

    let mut new_exits = Vec::with_capacity(exits.len());
    for link_ref in exits {
        let link = link_ref.borrow();
        // `unsimplify.py` `split_block` renames link args with
        // `get_new_name` but leaves `link.last_exception` /
        // `link.last_exc_value` alone: the link defines them.
        let new_args = link
            .args
            .iter()
            .map(|arg| {
                let is_last_exc = match arg {
                    Some(a) => {
                        Some(a) == link.last_exception.as_ref()
                            || Some(a) == link.last_exc_value.as_ref()
                    }
                    None => false,
                };
                if is_last_exc {
                    arg.clone()
                } else {
                    arg.as_ref()
                        .map(|value| get_new_name(value, &vars_produced_in_new_block, &mut varmap))
                }
            })
            .collect();
        let mut new_link =
            Link::new_mergeable(new_args, link.target.clone(), link.exitcase.clone());
        new_link.last_exception = link.last_exception.clone();
        new_link.last_exc_value = link.last_exc_value.clone();
        new_link.llexitcase = link.llexitcase.clone();
        new_link.origin = link.origin;
        new_exits.push(Rc::new(RefCell::new(new_link)));
    }

    let linkargs: Vec<Hlvalue> = varmap.keys().cloned().map(Hlvalue::Variable).collect();
    let newblock = Block::shared(linkargs.clone());
    {
        let mut nb = newblock.borrow_mut();
        nb.operations = moved_operations;
        nb.exitswitch = exitswitch
            .as_ref()
            .map(|value| get_new_name(value, &vars_produced_in_new_block, &mut varmap));
    }
    newblock.recloseblock(new_exits);

    {
        let mut b = block.borrow_mut();
        b.operations.truncate(index);
        b.exitswitch = None;
    }
    let link = Link::new(linkargs, Some(newblock), None).into_ref();
    block.recloseblock(vec![link.clone()]);
    Ok(link)
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn eliminate_empty_blocks(graph: &mut FunctionGraph) {
    for link_ref in graph.iterlinks() {
        loop {
            let target = {
                let link = link_ref.borrow();
                link.target.clone()
            };
            let Some(target) = target else {
                break;
            };
            let can_eliminate = {
                let t = target.borrow();
                t.operations.is_empty() && t.exitswitch.is_none() && !t.exits.is_empty()
            };
            if !can_eliminate {
                break;
            }
            let exit_ref = target.borrow().exits[0].clone();
            if let Some(exit_target) = exit_ref.borrow().target.clone()
                && BlockKey::of(&exit_target) == BlockKey::of(&target)
            {
                break;
            }
            let subst: HashMap<Variable, Hlvalue> = target
                .borrow()
                .inputargs
                .iter()
                .cloned()
                .zip(link_ref.borrow().args.iter().filter_map(|arg| arg.clone()))
                .filter_map(|(lhs, rhs)| match lhs {
                    Hlvalue::Variable(v) => Some((v, rhs)),
                    Hlvalue::Constant(_) => None,
                })
                .collect();
            let exit = exit_ref.borrow();
            let new_args = exit
                .args
                .iter()
                .map(|arg| {
                    arg.as_ref().map(|value| match value {
                        Hlvalue::Variable(v) => {
                            subst.get(v).cloned().unwrap_or_else(|| value.clone())
                        }
                        Hlvalue::Constant(_) => value.clone(),
                    })
                })
                .collect();
            let mut link = link_ref.borrow_mut();
            link.args = new_args;
            link.target = exit.target.clone();
        }
    }
}

pub fn tweak_generator_body_graph(
    entry: &HostObject,
    graph: &mut FunctionGraph,
) -> Result<(), FlowContextError> {
    let entry_varnames = entry_varnames(entry)?;
    insert_empty_startblock(graph);
    insert_reads(&graph.startblock, &entry_varnames)?;

    let stopblock = Block::shared(vec![]);
    {
        let mut sb = stopblock.borrow_mut();
        let op0 = SpaceOperation::new(
            "simple_call",
            vec![const_host(
                super::model::HOST_ENV
                    .lookup_builtin("StopIteration")
                    .expect("StopIteration builtin must exist"),
            )],
            Hlvalue::Variable(Variable::named("stop_value")),
        );
        let stop_value = op0.result.clone();
        let op1 = SpaceOperation::new(
            "type",
            vec![stop_value.clone()],
            Hlvalue::Variable(Variable::named("stop_type")),
        );
        let stop_type = op1.result.clone();
        sb.operations = vec![op0, op1];
        let mut link = Link::new(
            vec![stop_type, stop_value],
            Some(graph.exceptblock.clone()),
            None,
        );
        link.exitcase = None;
        sb.closeblock(vec![link.into_ref()]);
    }

    let blocks = graph.iterblocks();
    // upstream generator.py — `mappings = [Entry]`.
    // Each entry pairs the Resume-class HostObject with the target
    // block the dispatcher will jump to when `isinstance(entry, Resume)`
    // is true. The Entry mapping's target is the graph's startblock
    // (after `_insert_reads`, i.e. the block now owning the entry var).
    let mut mappings: Vec<(HostObject, Rc<RefCell<Block>>)> =
        vec![(entry.clone(), graph.startblock.clone())];
    for block in blocks {
        for exit_ref in block.borrow().exits.clone() {
            let mut exit = exit_ref.borrow_mut();
            if let Some(target) = &exit.target
                && BlockKey::of(target) == BlockKey::of(&graph.returnblock)
            {
                exit.args = Vec::new();
                exit.target = Some(stopblock.clone());
            }
        }

        let mut index = block.borrow().operations.len();
        while index > 0 {
            index -= 1;
            let is_yield = block.borrow().operations[index].opname == "yield_";
            if !is_yield {
                continue;
            }
            let yielded_value = block.borrow().operations[index].args[0].clone();
            block.borrow_mut().operations.remove(index);
            let newlink = split_block(&block, index)?;
            let newblock = newlink
                .borrow()
                .target
                .clone()
                .expect("split_block must create a target");
            let resume_varnames = get_variable_names_from_hlvalues(
                &newlink
                    .borrow()
                    .args
                    .iter()
                    .filter_map(|arg| arg.clone())
                    .collect::<Vec<_>>(),
            )?;
            let resume = HostObject::new_class_with_members(
                format!("Resume{}", mappings.len()),
                vec![],
                IndexMap::from([("_attrs_".to_string(), tuple_of_strings(&resume_varnames))]),
            );
            // upstream generator.py — `Resume.block = newblock;
            // mappings.append(Resume)`. Carry both into our list so the
            // regular-entry dispatcher below can route each branch to
            // the matching resume block.
            mappings.push((resume.clone(), newblock.clone()));
            insert_reads(&newblock, &resume_varnames)?;

            let mut block_ops = block.borrow().operations.clone();
            let op_resume = SpaceOperation::new(
                "simple_call",
                vec![const_host(resume)],
                Hlvalue::Variable(Variable::named("resume")),
            );
            let v_resume = op_resume.result.clone();
            block_ops.push(op_resume);
            for (arg, name) in newlink
                .borrow()
                .args
                .iter()
                .filter_map(|arg| arg.clone())
                .zip(resume_varnames.iter())
            {
                block_ops.push(SpaceOperation::new(
                    "setattr",
                    vec![v_resume.clone(), const_str(name), arg],
                    Hlvalue::Variable(Variable::new()),
                ));
            }
            let op_pair = SpaceOperation::new(
                "newtuple",
                vec![v_resume, yielded_value],
                Hlvalue::Variable(Variable::named("yield_pair")),
            );
            let pair_result = op_pair.result.clone();
            block_ops.push(op_pair);
            {
                let mut b = block.borrow_mut();
                b.operations = block_ops;
            }
            let mut newlink_mut = newlink.borrow_mut();
            newlink_mut.args = vec![Some(pair_result)];
            newlink_mut.target = Some(graph.returnblock.clone());
        }
    }

    // upstream generator.py:157-169 — `regular_entry_block` cascade of
    // `isinstance(entry, Resume)` dispatchers, routing to each
    // Resume.block in turn.
    let regular_entry_block = Block::shared(vec![Hlvalue::Variable(Variable::named("entry"))]);
    let mut current = regular_entry_block.clone();
    for (resume, target_block) in &mappings {
        let input = current.borrow().inputargs[0].clone();
        let op_check = SpaceOperation::new(
            "isinstance",
            vec![input.clone(), const_host(resume.clone())],
            Hlvalue::Variable(Variable::named("is_resume")),
        );
        let check_result = op_check.result.clone();
        current.borrow_mut().operations.push(op_check);
        current.borrow_mut().exitswitch = Some(check_result.clone());
        // upstream: `link1 = Link([input], Resume.block); link1.exitcase = True`
        let yes = Link::new(
            vec![input.clone()],
            Some(target_block.clone()),
            Some(const_str("True")),
        )
        .into_ref();
        let nextblock = Block::shared(vec![Hlvalue::Variable(Variable::named("entry"))]);
        let no = Link::new(
            vec![input],
            Some(nextblock.clone()),
            Some(const_str("False")),
        )
        .into_ref();
        current.closeblock(vec![yes, no]);
        current = nextblock;
    }
    let err_cls = super::model::HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("AssertionError builtin must exist");
    let bad = Link::new(
        vec![
            const_host(err_cls.clone()),
            Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                HostObject::new_instance(
                    err_cls,
                    vec![ConstValue::byte_str("bad generator class")],
                ),
            ))),
        ],
        Some(graph.exceptblock.clone()),
        None,
    )
    .into_ref();
    current.closeblock(vec![bad]);
    graph.startblock = regular_entry_block;
    checkgraph(graph);
    eliminate_empty_blocks(graph);
    Ok(())
}

/// RPython `generator.py` — `make_generator_entry_graph(func)`.
pub fn make_generator_entry_graph(func: GraphFunc) -> Result<FunctionGraph, FlowContextError> {
    let code = func.code.as_ref().ok_or_else(|| {
        FlowContextError::Flowing(FlowingError::new(
            "generator function is missing a code object",
        ))
    })?;
    let pygraph = PyGraph::new(func.clone(), code);
    // pygraph.graph is shared via Rc for annotator compatibility; we
    // hold the only reference here, so all mutations happen behind
    // `.borrow_mut()` and we unwrap back to an owned FunctionGraph for
    // the return value.
    let var_names =
        get_variable_names_from_hlvalues(&pygraph.graph.borrow().startblock.borrow().inputargs)?;
    let generator_iterator = make_generatoriterator_class(&func, &var_names);
    replace_graph_with_bootstrap(
        &generator_iterator,
        &mut pygraph.graph.borrow_mut(),
        &var_names,
    )?;
    attach_next_method(&generator_iterator, &pygraph.graph.borrow())?;
    Ok(std::rc::Rc::try_unwrap(pygraph.graph)
        .expect("make_generator_entry_graph: PyGraph.graph should be unique")
        .into_inner())
}

/// RPython `generator.py` — `tweak_generator_graph(graph)`.
pub fn tweak_generator_graph(graph: &mut FunctionGraph) -> Result<(), FlowContextError> {
    let generator_iterator = graph
        .func
        .as_ref()
        .and_then(|func| func._generator_next_method_of_.clone())
        .ok_or_else(|| {
            FlowContextError::Flowing(FlowingError::new(
                "generator graph missing _generator_next_method_of_",
            ))
        })?;
    let entry = generator_entry_class(&generator_iterator)?;
    tweak_generator_body_graph(&entry, graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::argument::Signature;
    use crate::flowspace::bytecode::HostCode;
    use crate::flowspace::model::BlockRefExt;

    fn empty_globals() -> Constant {
        Constant::new(ConstValue::Dict(Default::default()))
    }

    fn make_host_code(name: &str) -> HostCode {
        HostCode {
            id: HostCode::fresh_identity(),
            co_name: name.to_string(),
            co_filename: "<test>".to_string(),
            co_firstlineno: 1,
            co_nlocals: 1,
            co_argcount: 1,
            co_stacksize: 0,
            co_flags: super::super::objspace::CoFlags::NEWLOCALS.bits(),
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: vec!["x".to_string()],
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(vec!["x".to_string()], None, None),
        }
    }

    #[test]
    fn get_variable_names_prefixes_inputs() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            ("prefixed", &["x", "y", "_x_"], &["g_x", "g_y", "g_x_"]),
            ("empty", &[], &[]),
        ];
        for &(name, input, want) in cases {
            assert_eq!(get_variable_names(input), want, "case {name}");
        }
    }

    /// Host: `split_at_zero_moves_all_ops_to_new_block` (`unsimplify.rs`).
    /// `unsimplify.py` `split_block` leaves `last_exc_value` unrenamed.
    #[test]
    fn split_block_leaves_last_exc_value_arg_unrenamed() {
        let a = Variable::new();
        let r = Variable::new();
        let last_exc = Variable::new();
        let last_val = Variable::new();
        let block = Block::shared(vec![Hlvalue::Variable(a.clone())]);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "int_neg",
            vec![Hlvalue::Variable(a.clone())],
            Hlvalue::Variable(r.clone()),
        ));
        let handler = Block::shared(vec![
            Hlvalue::Variable(last_exc.clone()),
            Hlvalue::Variable(last_val.clone()),
        ]);
        let mut exc_link = Link::new(
            vec![
                Hlvalue::Variable(last_exc.clone()),
                Hlvalue::Variable(last_val.clone()),
            ],
            Some(handler),
            Some(Hlvalue::Constant(Constant::new(ConstValue::builtin(
                "Exception",
            )))),
        );
        exc_link.last_exception = Some(Hlvalue::Variable(last_exc.clone()));
        exc_link.last_exc_value = Some(Hlvalue::Variable(last_val.clone()));
        block.closeblock(vec![exc_link.into_ref()]);

        let synth = split_block(&block, 0).expect("split_block");
        let newblock = synth.borrow().target.clone().expect("new block");
        let exits = newblock.borrow().exits.clone();
        assert_eq!(exits.len(), 1);
        let link = exits[0].borrow();
        let Some(Hlvalue::Variable(arg_val)) = link.args[1].as_ref() else {
            panic!("last_exc_value arg");
        };
        assert_eq!(arg_val.id(), last_val.id());
        let Some(Hlvalue::Variable(extra_val)) = link.last_exc_value.as_ref() else {
            panic!("last_exc_value extra");
        };
        assert_eq!(extra_val.id(), last_val.id());
    }

    #[test]
    fn make_generator_entry_graph_builds_bootstrap_block() {
        let mut func = GraphFunc::new("gen", empty_globals());
        func.code = Some(Box::new(make_host_code("gen")));
        let graph = make_generator_entry_graph(func).expect("bootstrap graph");
        let ops = &graph.startblock.borrow().operations;
        assert_eq!(ops[0].opname, "simple_call");
        assert_eq!(ops[1].opname, "setattr");
        assert_eq!(ops[2].opname, "simple_call");
        assert_eq!(graph.startblock.borrow().exits.len(), 1);
    }
}
