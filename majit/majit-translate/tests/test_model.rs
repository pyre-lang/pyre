//! Line-by-line port of `rpython/flowspace/test/test_model.py`.
//!
//! RPython upstream (121 LOC, 9 test functions): builds the flow
//! graph corresponding to
//!
//! ```python
//! def sample_function(i):
//!     sum = 0
//!     while i > 0:
//!         sum = sum + i
//!         i = i - 1
//!     return sum
//! ```
//!
//! and asserts structural properties (checkgraph, copygraph, iter*,
//! mkentrymap, block attributes, variable renaming).

//! Deviation from upstream, per CLAUDE.md parity rule #1:
//!
//! * RPython uses module-level `graph = pieces.graph` shared by all
//!   tests. Rust rebuilds the graph per test via
//!   `Pieces::build_sample_graph()` — safer under test parallelism
//!   and avoids mutable shared state.

use std::cell::RefCell;
use std::rc::Rc;

use majit_translate::flowspace::model::{
    Block, BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc,
    HOST_ENV, Hlvalue, Link, LinkRef, SpaceOperation, Variable, c_last_exception, checkgraph,
    copygraph, mkentrymap,
};

/// Upstream `class pieces` — the manually-built graph pieces.
///
/// RPython relies on module-level `pieces = ...` with tests poking
/// at the attributes. Rust returns a struct the caller owns so each
/// test can build a fresh graph.
// some fields (ops) are only retained for parity
// with `class pieces`; downstream test-port commits
// may start reading them.
struct Pieces {
    // Variables.
    i0: Hlvalue,
    i2: Hlvalue,
    i3: Hlvalue,
    sum2: Hlvalue,
    sum3: Hlvalue,
    // Blocks.
    startblock: BlockRef,
    headerblock: BlockRef,
    whileblock: BlockRef,
    graph: FunctionGraph,
}

impl Pieces {
    /// Build the graph equivalent to upstream `pieces.graph`.
    fn build_sample_graph() -> Pieces {
        // RPython `Variable("i0")` etc.
        let i0 = Variable::named("i0");
        let i1 = Variable::named("i1");
        let i2 = Variable::named("i2");
        let i3 = Variable::named("i3");
        let sum1 = Variable::named("sum1");
        let sum2 = Variable::named("sum2");
        let sum3 = Variable::named("sum3");
        let conditionres = Variable::named("conditionres");

        // conditionop = SpaceOperation("gt", [i1, Constant(0)], conditionres)
        let conditionop = SpaceOperation::new(
            "gt",
            vec![
                Hlvalue::Variable(i1.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(0))),
            ],
            Hlvalue::Variable(conditionres.clone()),
        );
        // addop = SpaceOperation("add", [sum2, i2], sum3)
        let addop = SpaceOperation::new(
            "add",
            vec![
                Hlvalue::Variable(sum2.clone()),
                Hlvalue::Variable(i2.clone()),
            ],
            Hlvalue::Variable(sum3.clone()),
        );
        // decop = SpaceOperation("sub", [i2, Constant(1)], i3)
        let decop = SpaceOperation::new(
            "sub",
            vec![
                Hlvalue::Variable(i2.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
            ],
            Hlvalue::Variable(i3.clone()),
        );

        // startblock = Block([i0])
        let startblock = Block::shared(vec![Hlvalue::Variable(i0.clone())]);
        // headerblock = Block([i1, sum1])
        let headerblock = Block::shared(vec![
            Hlvalue::Variable(i1.clone()),
            Hlvalue::Variable(sum1.clone()),
        ]);
        // whileblock = Block([i2, sum2])
        let whileblock = Block::shared(vec![
            Hlvalue::Variable(i2.clone()),
            Hlvalue::Variable(sum2.clone()),
        ]);

        // graph = FunctionGraph("f", startblock)
        let graph = FunctionGraph::new("f", startblock.clone());

        // startblock.closeblock(Link([i0, Constant(0)], headerblock))
        let link_start_header = Rc::new(RefCell::new(Link::new(
            vec![
                Hlvalue::Variable(i0.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(0))),
            ],
            Some(headerblock.clone()),
            None,
        )));
        startblock.closeblock(vec![link_start_header]);

        // headerblock.operations.append(conditionop)
        headerblock
            .borrow_mut()
            .operations
            .push(conditionop.clone());
        // headerblock.exitswitch = conditionres
        headerblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(conditionres.clone()));
        // headerblock.closeblock(Link([sum1], graph.returnblock, False),
        //                        Link([i1, sum1], whileblock, True))
        let link_header_return = Rc::new(RefCell::new(Link::new(
            vec![Hlvalue::Variable(sum1.clone())],
            Some(graph.returnblock.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
        )));
        let link_header_while = Rc::new(RefCell::new(Link::new(
            vec![
                Hlvalue::Variable(i1.clone()),
                Hlvalue::Variable(sum1.clone()),
            ],
            Some(whileblock.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
        )));
        headerblock.closeblock(vec![link_header_return, link_header_while]);

        // whileblock.operations.append(addop)
        whileblock.borrow_mut().operations.push(addop.clone());
        // whileblock.operations.append(decop)
        whileblock.borrow_mut().operations.push(decop.clone());
        // whileblock.closeblock(Link([i3, sum3], headerblock))
        let link_while_header = Rc::new(RefCell::new(Link::new(
            vec![
                Hlvalue::Variable(i3.clone()),
                Hlvalue::Variable(sum3.clone()),
            ],
            Some(headerblock.clone()),
            None,
        )));
        whileblock.closeblock(vec![link_while_header]);

        Pieces {
            i0: Hlvalue::Variable(i0),
            i2: Hlvalue::Variable(i2),
            i3: Hlvalue::Variable(i3),
            sum2: Hlvalue::Variable(sum2),
            sum3: Hlvalue::Variable(sum3),
            startblock,
            headerblock,
            whileblock,
            graph,
        }
    }
}

// Test functions (line-by-line port of test_model.py:48-121)

#[test]
fn test_checkgraph() {
    // RPython test_model.py:48-49
    let pieces = Pieces::build_sample_graph();
    checkgraph(&pieces.graph);
}

#[test]
fn test_checkgraph_accepts_canraise_exception_links() {
    let startblock = Block::shared(vec![]);
    let graph = FunctionGraph::new("canraise", startblock.clone());
    let op_result = Hlvalue::Variable(Variable::named("res"));
    startblock
        .borrow_mut()
        .operations
        .push(SpaceOperation::new("call", vec![], op_result.clone()));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));

    let normal = Rc::new(RefCell::new(Link::new(
        vec![op_result.clone()],
        Some(graph.returnblock.clone()),
        None,
    )));
    let last_exception = Hlvalue::Variable(Variable::named("last_exception"));
    let last_exc_value = Hlvalue::Variable(Variable::named("last_exc_value"));
    let exceptional = Rc::new(RefCell::new(Link::new(
        vec![last_exception.clone(), last_exc_value.clone()],
        Some(graph.exceptblock.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
            HOST_ENV.lookup_builtin("ValueError").unwrap(),
        )))),
    )));
    exceptional
        .borrow_mut()
        .extravars(Some(last_exception.clone()), Some(last_exc_value.clone()));
    startblock.closeblock(vec![normal, exceptional]);

    checkgraph(&graph);
}

#[test]
fn test_copygraph() {
    // RPython test_model.py:51-53
    let mut pieces = Pieces::build_sample_graph();
    pieces.graph.tag = Some("tagged".into());
    pieces
        .graph
        .set_source("def sample_function(i):\n    return i\n");
    let mut func = GraphFunc::new("sample_function", Constant::new(ConstValue::Placeholder));
    func.filename = Some("sample.py".into());
    func.firstlineno = Some(1);
    pieces.graph.func = Some(func);
    let graph2 = copygraph(
        &pieces.graph,
        false,
        &std::collections::HashMap::new(),
        false,
    );
    checkgraph(&graph2);
    assert_eq!(graph2.tag.as_deref(), Some("tagged"));
    assert_eq!(
        graph2.source().unwrap(),
        "def sample_function(i):\n    return i\n"
    );
    assert_eq!(graph2.filename().unwrap(), "sample.py");
    assert_eq!(graph2.startline().unwrap(), 1);
}

#[test]
fn test_graphattributes() {
    // RPython test_model.py:55-60
    let pieces = Pieces::build_sample_graph();
    let graph = &pieces.graph;

    // assert graph.startblock is pieces.startblock
    assert!(Rc::ptr_eq(&graph.startblock, &pieces.startblock));

    // assert graph.returnblock is pieces.headerblock.exits[0].target
    let header_exit0_target = pieces.headerblock.borrow().exits[0].borrow().target.clone();
    assert!(
        header_exit0_target
            .as_ref()
            .is_some_and(|t| Rc::ptr_eq(t, &graph.returnblock)),
        "returnblock should be the first exit of the header block"
    );

    // assert graph.getargs() == [pieces.i0]
    assert_eq!(graph.getargs(), vec![pieces.i0.clone()]);

    // assert [graph.getreturnvar()] == graph.returnblock.inputargs
    assert_eq!(
        vec![graph.getreturnvar()],
        graph.returnblock.borrow().inputargs
    );

    let mut graph = pieces.graph;
    graph.set_source("def sample_function(i):\n    return i\n");
    let mut func = GraphFunc::new("sample_function", Constant::new(ConstValue::Placeholder));
    func.filename = Some("sample.py".into());
    func.firstlineno = Some(1);
    graph.func = Some(func);
    assert_eq!(
        graph.source().unwrap(),
        "def sample_function(i):\n    return i\n"
    );
    assert_eq!(graph.filename().unwrap(), "sample.py");
    assert_eq!(graph.startline().unwrap(), 1);
}

#[test]
fn test_iterblocks() {
    // RPython test_model.py:62-66
    // assert list(graph.iterblocks()) == [pieces.startblock,
    //                                     pieces.headerblock,
    //                                     graph.returnblock,
    //                                     pieces.whileblock]
    let pieces = Pieces::build_sample_graph();
    let blocks = pieces.graph.iterblocks();
    let expected = [
        pieces.startblock.clone(),
        pieces.headerblock.clone(),
        pieces.graph.returnblock.clone(),
        pieces.whileblock.clone(),
    ];
    assert_eq!(blocks.len(), expected.len());
    for (actual, want) in blocks.iter().zip(expected.iter()) {
        assert!(
            Rc::ptr_eq(actual, want),
            "iterblocks ordering diverged from upstream"
        );
    }
}

#[test]
fn test_iterlinks() {
    // RPython test_model.py:68-72
    // assert list(graph.iterlinks()) == [pieces.startblock.exits[0],
    //                                    pieces.headerblock.exits[0],
    //                                    pieces.headerblock.exits[1],
    //                                    pieces.whileblock.exits[0]]
    let pieces = Pieces::build_sample_graph();
    let links = pieces.graph.iterlinks();
    let expected: Vec<LinkRef> = vec![
        pieces.startblock.borrow().exits[0].clone(),
        pieces.headerblock.borrow().exits[0].clone(),
        pieces.headerblock.borrow().exits[1].clone(),
        pieces.whileblock.borrow().exits[0].clone(),
    ];
    assert_eq!(links.len(), expected.len());
    for (actual, want) in links.iter().zip(expected.iter()) {
        assert!(
            Rc::ptr_eq(actual, want),
            "iterlinks ordering diverged from upstream"
        );
    }
}

#[test]
fn test_mkentrymap() {
    // RPython test_model.py:74-83
    let pieces = Pieces::build_sample_graph();
    let entrymap = mkentrymap(&pieces.graph);
    // startlink is the synthetic Link the entrymap inserts for the
    // startblock; we pick it up by index and compare structurally.
    let start_entries = &entrymap[&BlockKey::of(&pieces.startblock)];
    assert_eq!(start_entries.len(), 1);
    // headerblock: start.exits[0] + while.exits[0]
    let header_entries = &entrymap[&BlockKey::of(&pieces.headerblock)];
    assert_eq!(header_entries.len(), 2);
    assert!(Rc::ptr_eq(
        &header_entries[0],
        &pieces.startblock.borrow().exits[0]
    ));
    assert!(Rc::ptr_eq(
        &header_entries[1],
        &pieces.whileblock.borrow().exits[0]
    ));
    // returnblock: header.exits[0]
    let return_entries = &entrymap[&BlockKey::of(&pieces.graph.returnblock)];
    assert_eq!(return_entries.len(), 1);
    assert!(Rc::ptr_eq(
        &return_entries[0],
        &pieces.headerblock.borrow().exits[0]
    ));
    // whileblock: header.exits[1]
    let while_entries = &entrymap[&BlockKey::of(&pieces.whileblock)];
    assert_eq!(while_entries.len(), 1);
    assert!(Rc::ptr_eq(
        &while_entries[0],
        &pieces.headerblock.borrow().exits[1]
    ));
}

#[test]
fn test_blockattributes() {
    // RPython test_model.py:85-91
    // assert block.getvariables() == [pieces.i2, pieces.sum2,
    //                                 pieces.sum3, pieces.i3]
    // assert block.getconstants() == [Constant(1)]
    let pieces = Pieces::build_sample_graph();
    let vars = pieces.whileblock.borrow().getvariables();
    let expect = vec![
        to_var(&pieces.i2),
        to_var(&pieces.sum2),
        to_var(&pieces.sum3),
        to_var(&pieces.i3),
    ];
    assert_eq!(vars, expect);

    let consts = pieces.whileblock.borrow().getconstants();
    assert_eq!(consts, vec![Constant::new(ConstValue::Int(1))]);
}

#[test]
fn test_renamevariables() {
    // RPython test_model.py:93-105
    let pieces = Pieces::build_sample_graph();
    let sum2 = to_var(&pieces.sum2);

    // v = Variable()
    let v = Variable::new();
    // block.renamevariables({pieces.sum2: v})
    #[expect(
        clippy::mutable_key_type,
        reason = "Variable hashes by immutable graph identity; mutable annotations are excluded, matching RPython identity-keyed rename maps"
    )]
    let mut mapping: std::collections::HashMap<Variable, Hlvalue> =
        std::collections::HashMap::new();
    mapping.insert(sum2.clone(), Hlvalue::Variable(v.clone()));
    pieces.whileblock.borrow_mut().renamevariables(&mapping);

    let vars = pieces.whileblock.borrow().getvariables();
    assert_eq!(
        vars,
        vec![
            to_var(&pieces.i2),
            v.clone(),
            to_var(&pieces.sum3),
            to_var(&pieces.i3)
        ]
    );

    // block.renamevariables({v: pieces.sum2})
    #[expect(
        clippy::mutable_key_type,
        reason = "Variable hashes by immutable graph identity; mutable annotations are excluded, matching RPython identity-keyed rename maps"
    )]
    let mut mapping2: std::collections::HashMap<Variable, Hlvalue> =
        std::collections::HashMap::new();
    mapping2.insert(v, Hlvalue::Variable(sum2.clone()));
    pieces.whileblock.borrow_mut().renamevariables(&mapping2);
    let vars = pieces.whileblock.borrow().getvariables();
    assert_eq!(
        vars,
        vec![
            to_var(&pieces.i2),
            sum2,
            to_var(&pieces.sum3),
            to_var(&pieces.i3),
        ]
    );
}

#[test]
fn test_variable() {
    // RPython test_model.py:107-121
    // v = Variable()
    let v = Variable::new();
    // assert v.name[0] == 'v' and v.name[1:].isdigit()
    let n = v.name();
    assert!(n.starts_with('v'));
    assert!(n[1..].chars().all(|c| c.is_ascii_digit()));
    // assert not v.renamed
    assert!(!v.renamed());
    // v.rename("foobar")
    let mut v = v;
    v.rename("foobar");
    let name1 = v.name();
    // assert name1.startswith('foobar_')
    assert!(name1.starts_with("foobar_"));
    // assert name1.split('_', 1)[1].isdigit()
    assert!(
        name1
            .split_once('_')
            .map(|x| x.1)
            .map(|s| s.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false)
    );
    // assert v.renamed
    assert!(v.renamed());
    // v.rename("not again")  -- no-op on already-renamed
    v.rename("not again");
    assert_eq!(v.name(), name1);

    // v2 = Variable(v)
    //   RPython: `Variable(v)` triggers `rename(v)` which copies
    //   _name. Our equivalent is `Variable::copy()`.
    let v2 = v.copy();
    assert!(v2.renamed());
    let n2 = v2.name();
    assert!(n2.starts_with("foobar_"));
    assert_ne!(n2, v.name());
    assert!(
        n2.split_once('_')
            .map(|x| x.1)
            .map(|s| s.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false)
    );
}

// helpers

fn to_var(w: &Hlvalue) -> Variable {
    match w {
        Hlvalue::Variable(v) => v.clone(),
        Hlvalue::Constant(_) => panic!("expected Variable Hlvalue"),
    }
}
