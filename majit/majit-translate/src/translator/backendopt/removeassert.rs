//! Port of `rpython/translator/backendopt/removeassert.py`.
//!
//! Mirrors upstream's module split: `all.py` imports
//! `from rpython.translator.backendopt.removeassert import remove_asserts`
//! at `all.py:11`, and the body lives in its own file. The Rust port
//! mirrors that file boundary.

use std::rc::Rc;

use crate::flowspace::model::{
    BlockRefExt, ConstValue, Constant, FunctionGraph, GraphRef, HOST_ENV, Hlvalue, LinkRef,
};
use crate::translator::backendopt::support::LOG;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::inputconst_from_lltype;
use crate::translator::rtyper::rtyper::{GenopResult, LowLevelOpList};
use crate::translator::simplify;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// RPython `removeassert.remove_asserts(translator, graphs)` at
/// `removeassert.py`.
pub fn remove_asserts(
    translator: &TranslationContext,
    graphs: &[GraphRef],
) -> Result<(), TaskError> {
    // Upstream `:9 rtyper = translator.rtyper`.
    let rtyper = translator.rtyper().ok_or_else(|| TaskError {
        message: "removeassert.py:9 remove_asserts: translator.rtyper is None".to_string(),
    })?;
    // Upstream `:10 excdata = rtyper.exceptiondata`.
    let excdata = rtyper.exceptiondata().map_err(|e| TaskError {
        message: format!("removeassert.py:10 exceptiondata: {e}"),
    })?;
    // Upstream `:11 clsdef =
    //     translator.annotator.bookkeeper.getuniqueclassdef(AssertionError)`.
    let annotator = translator.annotator().ok_or_else(|| TaskError {
        message: "removeassert.py:11 remove_asserts: translator.annotator is None".to_string(),
    })?;
    let assertion_error = HOST_ENV
        .lookup_builtin("AssertionError")
        .ok_or_else(|| TaskError {
            message: "removeassert.py:11 AssertionError missing from HOST_ENV".to_string(),
        })?;
    let clsdef = annotator
        .bookkeeper
        .getuniqueclassdef(&assertion_error)
        .map_err(|e| TaskError {
            message: format!("removeassert.py:11 getuniqueclassdef(AssertionError): {e}"),
        })?;
    // Upstream `:12 ll_AssertionError =
    //     excdata.get_standard_ll_exc_instance(rtyper, clsdef)`.
    let ll_assertion_error = excdata
        .get_standard_ll_exc_instance(&rtyper, Some(clsdef))
        .map_err(|e| TaskError {
            message: format!("removeassert.py:12 get_standard_ll_exc_instance: {e}"),
        })?;
    // Upstream `:13 total_count = [0, 0]`. The accumulator feeds the
    // final `log.removeassert(...)` summary at `:42-53`.
    let mut total_count: [usize; 2] = [0, 0];
    for graph in graphs {
        // Upstream `:16-17 count = 0; morework = True`.
        let mut count = 0usize;
        loop {
            let mut morework = false;
            // Upstream `:20-21 eliminate_empty_blocks(graph);
            // join_blocks(graph)`.
            {
                let graph_b = graph.borrow();
                simplify::eliminate_empty_blocks(&graph_b);
                simplify::join_blocks(&graph_b);
            }
            let links = graph.borrow().iterlinks();
            for link in links {
                let is_assertion_link = {
                    let graph_b = graph.borrow();
                    assertion_link_matches(&graph_b, &link, &ll_assertion_error)
                };
                if !is_assertion_link {
                    continue;
                }
                // Upstream `:26-33 if kill_assertion_link(graph, link):
                //     count += 1; morework = True; break
                // else: total_count[0] += 1
                //     if translator.config.translation.verbose:
                //         log.removeassert("cannot remove ...")`.
                //
                // `kill_assertion_link` builds a `LowLevelOpList()` to
                // emit `bool_not` / `debug_assert` (no rtyper required
                // for those ops); pyre matches upstream's
                // `LowLevelOpList()` no-arg form via
                // `LowLevelOpList::without_rtyper`.
                if kill_assertion_link(&graph.borrow(), &link)? {
                    count += 1;
                    morework = true;
                    break;
                } else {
                    total_count[0] += 1;
                    // Upstream `:32-33 if translator.config.translation.verbose:
                    //     log.removeassert("cannot remove an assert from %s" % (graph.name,))`.
                    if translator.config.translation.verbose {
                        let name = graph.borrow().name.clone();
                        LOG.method(
                            "removeassert",
                            &format!("cannot remove an assert from {name}"),
                        );
                    }
                }
            }
            if !morework {
                break;
            }
        }
        // Upstream `:34-40 if count: total_count[1] += count; ...
        // checkgraph(graph)`.
        if count != 0 {
            total_count[1] += count;
            // Upstream `:37-39 if translator.config.translation.verbose:
            //     log.removeassert("removed %d asserts in %s" % (count, graph.name))`.
            if translator.config.translation.verbose {
                let name = graph.borrow().name.clone();
                LOG.method(
                    "removeassert",
                    &format!("removed {count} asserts in {name}"),
                );
            }
            crate::flowspace::model::checkgraph(&graph.borrow());
        }
    }
    // Upstream `:41-53 total_count = tuple(total_count); if
    // total_count[0] == 0: ...; if msg is not None:
    // log.removeassert(msg)`.
    let msg = if total_count[0] == 0 {
        if total_count[1] == 0 {
            None
        } else {
            // `:46 msg = "Removed %d asserts" % (total_count[1],)`.
            Some(format!("Removed {} asserts", total_count[1]))
        }
    } else if total_count[1] == 0 {
        // `:49 msg = "Could not remove %d asserts" % (total_count[0],)`.
        Some(format!("Could not remove {} asserts", total_count[0]))
    } else {
        // `:51-52 msg = "Could not remove %d asserts, but removed %d
        // asserts." % total_count`.
        Some(format!(
            "Could not remove {} asserts, but removed {} asserts.",
            total_count[0], total_count[1]
        ))
    };
    if let Some(msg) = msg {
        // `:53 log.removeassert(msg)`.
        LOG.method("removeassert", &msg);
    }
    Ok(())
}

fn assertion_link_matches(
    graph: &FunctionGraph,
    link: &LinkRef,
    ll_assertion_error: &Constant,
) -> bool {
    let link_b = link.borrow();
    let Some(target) = &link_b.target else {
        return false;
    };
    if !Rc::ptr_eq(target, &graph.exceptblock) {
        return false;
    }
    matches!(
        link_b.args.get(1).and_then(|arg| arg.as_ref()),
        Some(Hlvalue::Constant(c)) if c == ll_assertion_error
    )
}

/// RPython `removeassert.kill_assertion_link(graph, link)` at
/// `removeassert.py`.
fn kill_assertion_link(graph: &FunctionGraph, link: &LinkRef) -> Result<bool, TaskError> {
    let block = link
        .borrow()
        .prevblock
        .as_ref()
        .and_then(|prev| prev.upgrade())
        .ok_or_else(|| TaskError {
            message: "removeassert.py `kill_assertion_link`: link.prevblock missing".to_string(),
        })?;
    let mut exits: Vec<LinkRef> = block.borrow().exits.clone();
    if exits.len() <= 1 {
        return Ok(false);
    }
    let link_index = exits
        .iter()
        .position(|candidate| Rc::ptr_eq(candidate, link))
        .ok_or_else(|| TaskError {
            message: "removeassert.py `kill_assertion_link`: link not in prevblock.exits"
                .to_string(),
        })?;
    let mut remove_condition = exits.len() == 2;
    if block.borrow().canraise() {
        if link_index == 0 {
            return Ok(false);
        }
    } else {
        let exitswitch = block.borrow().exitswitch.clone();
        if exitswitch.as_ref().and_then(hlvalue_concretetype).as_ref() != Some(&LowLevelType::Bool)
        {
            remove_condition = false;
        } else {
            if !remove_condition {
                return Err(TaskError {
                    message:
                        "removeassert.py `kill_assertion_link`: bool exitswitch without two exits"
                            .to_string(),
                });
            }
            let exitswitch = exitswitch.expect("checked above");
            // `removeassert.py` `kill_assertion_link` emits
            // `newops = LowLevelOpList()` — no rtyper
            // argument. Mirrors the no-arg form: only `bool_not` /
            // `debug_assert` are emitted, neither requires the
            // typer.
            let mut newops = LowLevelOpList::without_rtyper(None);
            let condition = if hlvalue_is_true(link.borrow().exitcase.as_ref()) {
                let inverted = newops
                    .genop(
                        "bool_not",
                        vec![exitswitch],
                        GenopResult::LLType(LowLevelType::Bool),
                    )
                    .expect("bool_not has Bool result");
                Hlvalue::Variable(inverted)
            } else {
                exitswitch
            };
            let msg = format!("assertion failed in {}", graph.name);
            let c_msg = inputconst_from_lltype(&LowLevelType::Void, &ConstValue::byte_str(msg))
                .map_err(|e| TaskError {
                    message: format!("removeassert.py:55 inputconst(Void, msg): {e}"),
                })?;
            newops.genop(
                "debug_assert",
                vec![condition, Hlvalue::Constant(c_msg)],
                GenopResult::Void,
            );
            block.borrow_mut().operations.extend(newops.ops);
        }
    }
    exits.remove(link_index);
    if remove_condition {
        block.borrow_mut().exitswitch = None;
        if let Some(first) = exits.first() {
            let mut first_b = first.borrow_mut();
            first_b.exitcase = None;
            first_b.llexitcase = None;
        }
    }
    block.recloseblock(exits);
    Ok(true)
}

fn hlvalue_concretetype(value: &Hlvalue) -> Option<LowLevelType> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

fn hlvalue_is_true(value: Option<&Hlvalue>) -> bool {
    match value {
        Some(Hlvalue::Constant(c)) => match &c.value {
            ConstValue::Bool(value) => *value,
            ConstValue::Int(value) => *value != 0,
            ConstValue::UInt(value) => *value != 0,
            ConstValue::None => false,
            _ => true,
        },
        Some(Hlvalue::Variable(_)) => true,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, LAST_EXCEPTION, Link,
        SpaceOperation, Variable,
    };
    use std::cell::RefCell;

    fn graph_ref(graph: FunctionGraph) -> GraphRef {
        Rc::new(RefCell::new(graph))
    }

    #[test]
    fn kill_assertion_link_drops_canraise_assertion_exit() {
        let _ann = RPythonAnnotator::new(None, None, None, false);
        let v = Variable::named("v");
        let result = Variable::named("result");
        let body = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", body.clone());
        body.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Variable(v)],
            Hlvalue::Variable(result.clone()),
        ));
        body.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::new(ConstValue::Atom(
            LAST_EXCEPTION.clone(),
        ))));
        let ok = Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let ll_assertion = Constant::with_concretetype(ConstValue::Int(42), LowLevelType::Signed);
        let err = Link::new(
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::builtin("AssertionError"))),
                Hlvalue::Constant(ll_assertion.clone()),
            ],
            Some(graph.exceptblock.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::builtin(
                "AssertionError",
            )))),
        )
        .into_ref();
        body.closeblock(vec![ok.clone(), err.clone()]);

        assert!(assertion_link_matches(&graph, &err, &ll_assertion));
        assert!(kill_assertion_link(&graph, &err).expect("kill assertion link"));

        assert_eq!(body.borrow().exits.len(), 1);
        assert!(body.borrow().exitswitch.is_none());
        assert!(Rc::ptr_eq(
            body.borrow().exits[0].borrow().target.as_ref().unwrap(),
            &graph.returnblock
        ));
        assert!(body.borrow().exits[0].borrow().exitcase.is_none());
    }

    #[test]
    fn kill_assertion_link_rewrites_bool_switch_to_debug_assert() {
        let _ann = RPythonAnnotator::new(None, None, None, false);
        let cond = Variable::named("cond");
        cond.set_concretetype(Some(LowLevelType::Bool));
        let body = Block::shared(vec![Hlvalue::Variable(cond.clone())]);
        let graph = FunctionGraph::new("f", body.clone());
        body.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
        let ok = Link::new(
            vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
            Some(graph.returnblock.clone()),
            Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            ))),
        )
        .into_ref();
        let ll_assertion = Constant::with_concretetype(ConstValue::Int(7), LowLevelType::Signed);
        let err = Link::new(
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::builtin("AssertionError"))),
                Hlvalue::Constant(ll_assertion.clone()),
            ],
            Some(graph.exceptblock.clone()),
            Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            ))),
        )
        .into_ref();
        body.closeblock(vec![ok.clone(), err.clone()]);
        let _ = graph_ref;

        assert!(kill_assertion_link(&graph, &err).expect("kill assertion link"));

        let body_b = body.borrow();
        assert_eq!(body_b.operations.len(), 2);
        assert_eq!(body_b.operations[0].opname, "bool_not");
        assert_eq!(body_b.operations[1].opname, "debug_assert");
        assert_eq!(body_b.exits.len(), 1);
        assert!(body_b.exitswitch.is_none());
        assert!(body_b.exits[0].borrow().exitcase.is_none());
    }
}
