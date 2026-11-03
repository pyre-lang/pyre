//! Port of `rpython/translator/simplify.py::simplify_graph` onto pyre-jit's
//! flow graph (`super::flow`).
//!
//! RPython runs `simplify_graph(graph)` (`translator.py`) right after
//! `build_flow`, so every graph reaching the codewriter
//! (`perform_register_allocation → flatten_graph → compute_liveness →
//! assemble`) is already normalized.  pyre-jit's walker fuses graph-build and
//! flatten and historically skipped this normalization, which forced
//! pyre-only repairs (byte-stream TLabel rewrites, regalloc walker pins, the
//! per-PC `-live-` remap, since retired) and surfaced as the
//! `jitcode label was never marked` assembler panic.  This module brings
//! the orthodox pass list onto pyre-jit's `flow::FunctionGraph` so the graph
//! that reaches coalescing/flatten matches what PyPy produces.
//!
//! Each pass is a line-by-line port dual-referenced to
//! `rpython/translator/simplify.py` (source of truth) and the already-validated
//! port in `majit/majit-translate/src/translator/simplify.rs`.  The flow model
//! differs (`super::flow` uses `VariableId(u32)` identity and `Weak` prevblock
//! back-edges), so the structural shape is preserved while the Rust mechanics
//! follow `super::flow`'s conventions.
//!
//! Classification (issue #112 scope #4): every pass here is **direct PyPy
//! parity**.
//!
//! ## Issue #112 scope #3 — unmarked-label investigation (concluded)
//!
//! The `jitcode label was never marked` panic (`majit-metainterp/src/jitcode/
//! assembler.rs::patch_labels`) fires when flatten emits a goto/switch/forwarder
//! operand referencing a block whose `Label` is never emitted.  The only graph
//! shapes that produce it are a dead/trivial forwarder or a dead constant-switch
//! arm reaching flatten.  The walker-safe subset wired in
//! `codewriter.rs` (`eliminate_empty_blocks` + `constfold_exitswitch` +
//! `remove_trivial_links`, in `all_passes` relative order) removes exactly
//! those: `eliminate_empty_blocks` collapses dead forwarders — and the
//! port-boundary guard fails loud if a reachable link still targets a `dead`
//! block; `remove_trivial_links` merges single-entry/single-exit chains; and
//! `constfold_exitswitch` drops the dead arm through `recloseblock`, so no link
//! references it (a still-reachable arm target keeps its `Label`).
//!
//! The `all_passes` entries left unwired do **not** affect label structure:
//! `transform_dead_op_vars` / `remove_identical_vars_ssa` / `ssa_to_ssi` only
//! rename or dedup variables; `coalesce_bool` / `transform_ovfcheck` /
//! `transform_xxxitem` are op-level rewrites; and `simplify_exceptions` /
//! `remove_dead_exceptions` / `remove_assertion_errors` are documented no-ops
//! that remove no block or link on a walker graph.  So the wired subset closes
//! the gap and the assembler `never marked` panic is a fail-loud backstop, not a
//! live failure mode — empirically unreached at check.py 41/41 both backends.

use std::collections::{HashMap, HashSet};

use majit_jitcode::tool::algo::unionfind::{UnionFind, UnionFindInfo};

use super::flow::{
    BlockRef, Constant, ConstantValue, ExitSwitch, ExitSwitchElement, FlowListOfKind, FlowValue,
    FunctionGraph, LinkRef, SpaceOperation, SpaceOperationArg, Variable, mkentrymap,
};

/// `rpython/translator/simplify.py` `eliminate_empty_blocks`.
/// Port reference: `majit-translate/src/translator/simplify.rs`'s
/// `eliminate_empty_blocks`.
///
/// ```py
/// def eliminate_empty_blocks(graph):
///     for link in list(graph.iterlinks()):
///         while not link.target.operations:
///             block1 = link.target
///             if block1.exitswitch is not None:
///                 break
///             if not block1.exits:
///                 break
///             exit = block1.exits[0]
///             assert block1 is not exit.target
///             subst = dict(zip(block1.inputargs, link.args))
///             link.args = [v.replace(subst) for v in exit.args]
///             link.target = exit.target
/// ```
///
/// Collapses any empty forwarding block by retargeting each predecessor link
/// through it — the faithful operations-based predicate the orthodox
/// `simplify_graph` driver needs.  (The walker pipeline instead calls
/// `codewriter::eliminate_empty_blocks`, whose `block.dead` predicate is the
/// walker-only proxy: every walker block carries empty `operations` because the
/// SSARepr is emitted inline, so a `not operations` predicate would collapse all
/// of them.)
pub fn eliminate_empty_blocks(graph: &FunctionGraph) {
    for link in graph.iterlinks() {
        loop {
            let target = match link.borrow().target.clone() {
                Some(target) => target,
                None => break,
            };
            // `while not link.target.operations:` + the two guards
            // (`exitswitch is not None` / `not exits`).
            let exit_link = {
                let tb = target.borrow();
                if !tb.operations.is_empty() || tb.exitswitch.is_some() || tb.exits.is_empty() {
                    break;
                }
                tb.exits[0].clone()
            };
            let (exit_args, exit_target) = {
                let e = exit_link.borrow();
                match e.target.clone() {
                    Some(t) => (e.args.clone(), t),
                    None => break,
                }
            };
            assert!(
                target != exit_target,
                "eliminate_empty_blocks: the graph contains an empty infinite loop"
            );
            // `subst = dict(zip(block1.inputargs, link.args))`.
            let inputargs = target.borrow().inputargs.clone();
            let link_args = link.borrow().args.clone();
            let mut subst: HashMap<Variable, Option<FlowValue>> = HashMap::new();
            for (inputarg, arg) in inputargs.iter().zip(link_args.iter()) {
                if let FlowValue::Variable(v) = inputarg {
                    subst.insert(*v, arg.clone());
                }
            }
            // `link.args = [v.replace(subst) for v in exit.args]`.
            let new_args: Vec<Option<FlowValue>> = exit_args
                .iter()
                .map(|arg| match arg {
                    Some(FlowValue::Variable(v)) => match subst.get(v) {
                        Some(replacement) => replacement.clone(),
                        None => arg.clone(),
                    },
                    _ => arg.clone(),
                })
                .collect();
            // `link.target = exit.target`; the outer loop keeps collapsing
            // through the new target if it too is an empty forwarder.
            let mut l = link.borrow_mut();
            l.args = new_args;
            l.target = Some(exit_target);
        }
    }
}

/// `rpython/translator/simplify.py` `remove_trivial_links`.
/// Port reference: `majit-translate/src/translator/simplify.rs`'s
/// `remove_trivial_links`.
///
/// ```py
/// def remove_trivial_links(graph):
///     entrymap = mkentrymap(graph)
///     block = graph.startblock
///     seen = set([block])
///     stack = list(block.exits)
///     while stack:
///         link = stack.pop()
///         if link.target in seen:
///             continue
///         source = link.prevblock
///         target = link.target
///         if (not link.args and source.exitswitch is None and
///                 len(entrymap[target]) == 1 and
///                 target.exits):  # stop at the returnblock
///             assert len(source.exits) == 1
///             source.operations.extend(target.operations)
///             source.exitswitch = newexitswitch = target.exitswitch
///             source.recloseblock(*target.exits)
///             stack.extend(source.exits)
///         else:
///             seen.add(target)
///             stack.extend(target.exits)
/// ```
///
/// A link is trivial when it carries no args, is the single exit of its
/// source (`source.exitswitch is None` ⇒ one exit), and its target has a
/// single predecessor and is not the returnblock (`target.exits` non-empty).
/// Such a link is removed by folding `target` into `source`.
pub fn remove_trivial_links(graph: &FunctionGraph) {
    let entrymap = mkentrymap(graph);
    let startblock = graph.startblock.clone();
    // `seen = set([block])` — faithful port of a Python set; BlockRef hashes
    // by Rc pointer identity, matching RPython's object-identity membership.
    let mut seen: HashSet<BlockRef> = HashSet::new();
    seen.insert(startblock.clone());
    // `stack = list(block.exits)`; `stack.pop()` takes the last element.
    let mut stack: Vec<LinkRef> = startblock.borrow().exits.clone();

    while let Some(link) = stack.pop() {
        // `if link.target in seen: continue`.  RPython links always carry a
        // target; the `None` guard only covers half-built links, which
        // cannot appear in a graph reachable from `graph.startblock`.
        let target = match link.borrow().target.clone() {
            Some(target) => target,
            None => continue,
        };
        if seen.contains(&target) {
            continue;
        }
        // `source = link.prevblock`.
        let source = link
            .borrow()
            .prevblock_ref()
            .expect("remove_trivial_links: reachable link has a prevblock");

        let is_trivial = {
            let link_b = link.borrow();
            let source_b = source.borrow();
            let target_b = target.borrow();
            link_b.args.is_empty()
                && source_b.exitswitch.is_none()
                && entrymap.get(&target).map_or(0, Vec::len) == 1
                && !target_b.exits.is_empty() // stop at the returnblock
        };

        if is_trivial {
            // `assert len(source.exits) == 1` — guaranteed by
            // `source.exitswitch is None`, asserted to fail loudly if a
            // future graph shape breaks the invariant.
            assert_eq!(
                source.borrow().exits.len(),
                1,
                "remove_trivial_links: trivial-link source must be single-exit"
            );
            // `source.operations.extend(target.operations)`;
            // `source.exitswitch = target.exitswitch`;
            // `source.recloseblock(*target.exits)`.
            let (target_ops, target_exitswitch, target_exits) = {
                let target_b = target.borrow();
                (
                    target_b.operations.clone(),
                    target_b.exitswitch.clone(),
                    target_b.exits.clone(),
                )
            };
            {
                let mut source_b = source.borrow_mut();
                source_b.operations.extend(target_ops);
                source_b.exitswitch = target_exitswitch;
            }
            source.recloseblock(target_exits);
            // `stack.extend(source.exits)` — keep collapsing through source.
            stack.extend(source.borrow().exits.clone());
        } else {
            // `seen.add(target); stack.extend(target.exits)`.
            seen.insert(target.clone());
            stack.extend(target.borrow().exits.clone());
        }
    }
}

// ── remove_identical_vars_SSA + supporting helpers ──────────────────────

/// `simplify.py` `class Representative`.  Stores the partition's
/// representative value; `absorb` is a no-op so the info that wins the
/// weighted union keeps its `rep`.
#[derive(Clone, Debug)]
struct Representative {
    rep: FlowValue,
}

impl UnionFindInfo for Representative {
    fn absorb(&mut self, _other: &Self) {}
}

/// `simplify.py` `all_equal`.
fn all_equal(lst: &[FlowValue]) -> bool {
    match lst.first() {
        None => true,
        Some(first) => lst.iter().skip(1).all(|x| x == first),
    }
}

/// `simplify.py` `isspecialvar`.
fn isspecialvar(v: &FlowValue) -> bool {
    match v {
        FlowValue::Variable(var) => {
            let prefix = var.name_prefix();
            prefix == "last_exception_" || prefix == "last_exc_value_"
        }
        FlowValue::Constant(_) => false,
    }
}

/// `model.py Variable.replace` generalized to a `Variable → FlowValue`
/// renaming: a Variable in `mapping` becomes its mapped value, any other
/// Variable and any Constant is returned unchanged.
fn rename_value(v: &FlowValue, mapping: &HashMap<Variable, FlowValue>) -> FlowValue {
    match v {
        FlowValue::Variable(var) => mapping
            .get(var)
            .cloned()
            .unwrap_or(FlowValue::Variable(*var)),
        FlowValue::Constant(_) => v.clone(),
    }
}

fn rename_arg(
    arg: &SpaceOperationArg,
    mapping: &HashMap<Variable, FlowValue>,
) -> SpaceOperationArg {
    match arg {
        SpaceOperationArg::Value(value) => SpaceOperationArg::Value(rename_value(value, mapping)),
        SpaceOperationArg::ListOfKind(list) => SpaceOperationArg::ListOfKind(FlowListOfKind {
            kind: list.kind,
            content: list
                .content
                .iter()
                .map(|value| rename_value(value, mapping))
                .collect(),
        }),
        // `flatten.py` passes AbstractDescr / IndirectCallTargets
        // through unchanged.
        SpaceOperationArg::Descr(_) | SpaceOperationArg::IndirectCallTargets(_) => arg.clone(),
    }
}

fn rename_exitswitch(sw: &ExitSwitch, mapping: &HashMap<Variable, FlowValue>) -> ExitSwitch {
    match sw {
        ExitSwitch::Value(value) => ExitSwitch::Value(rename_value(value, mapping)),
        ExitSwitch::Tuple(elements) => ExitSwitch::Tuple(
            elements
                .iter()
                .map(|element| match element {
                    ExitSwitchElement::Value(value) => {
                        ExitSwitchElement::Value(rename_value(value, mapping))
                    }
                    ExitSwitchElement::Marker(marker) => ExitSwitchElement::Marker(marker.clone()),
                })
                .collect(),
        ),
    }
}

/// `model.py Block.renamevariables`, generalized to a
/// `Variable → FlowValue` mapping (the rep can be a Constant) and extended to
/// the `last_exception` / `last_exc_value` link extras, mirroring the
/// `majit-translate` `renamevariables_hl`.
///
/// Two deliberate adaptations matching the validated `majit-translate` port:
/// (1) `block.inputargs` are NOT renamed here — `remove_identical_vars_SSA`
/// has already rewritten them to the pruned phi-input list, and a block
/// parameter must stay a Variable (never a Constant rep); (2) unlike
/// `join_blocks`'s rename, no `indirect_call`→`direct_call` rewrite —
/// `Block.renamevariables` upstream is a plain `op.replace(mapping)`.
fn renamevariables_value(block: &BlockRef, mapping: &HashMap<Variable, FlowValue>) {
    if mapping.is_empty() {
        return;
    }
    let (new_ops, new_switch) = {
        let b = block.borrow();
        let ops: Vec<SpaceOperation> = b
            .operations
            .iter()
            .map(|op| SpaceOperation {
                opname: op.opname.clone(),
                args: op.args.iter().map(|arg| rename_arg(arg, mapping)).collect(),
                result: op.result.as_ref().map(|r| rename_value(r, mapping)),
                offset: op.offset,
            })
            .collect();
        let switch = b
            .exitswitch
            .as_ref()
            .map(|sw| rename_exitswitch(sw, mapping));
        (ops, switch)
    };
    {
        let mut b = block.borrow_mut();
        b.operations = new_ops;
        b.exitswitch = new_switch;
    }
    let exits: Vec<LinkRef> = block.borrow().exits.clone();
    for link in exits {
        let mut l = link.borrow_mut();
        l.args = l
            .args
            .iter()
            .map(|arg| arg.as_ref().map(|value| rename_value(value, mapping)))
            .collect();
        // The exception extras are Variables; a rename to a non-Variable
        // would be a contradiction, so keep only the Variable case.
        if let Some(v) = l.last_exception {
            if let FlowValue::Variable(nv) = rename_value(&FlowValue::Variable(v), mapping) {
                l.last_exception = Some(nv);
            }
        }
        if let Some(v) = l.last_exc_value {
            if let FlowValue::Variable(nv) = rename_value(&FlowValue::Variable(v), mapping) {
                l.last_exc_value = Some(nv);
            }
        }
    }
}

/// `rpython/translator/simplify.py` `remove_identical_vars_SSA`.
/// Port reference: `majit-translate/src/translator/simplify.rs`'s
/// `remove_identical_vars_SSA`.
///
/// When the same value is passed multiple times into the next block, pass it
/// only once.  Uses its own `UnionFind(Representative)` (not
/// `DataFlowFamilyBuilder`), inlining the phi-node collapse over each block's
/// inputs with the per-link `link.args[i]` as the phi sources.
pub fn remove_identical_vars_ssa(graph: &FunctionGraph) {
    let mut uf: UnionFind<FlowValue, Representative> =
        UnionFind::new(|k: &FlowValue| Representative { rep: k.clone() });

    // `entrymap = mkentrymap(graph); del entrymap[startblock];
    // entrymap.pop(returnblock, None); entrymap.pop(exceptblock, None)`.
    let mut entrymap = mkentrymap(graph);
    entrymap.remove(&graph.startblock);
    entrymap.remove(&graph.returnblock);
    entrymap.remove(&graph.exceptblock);

    // `inputs[block] = zip(block.inputargs, zip(*[link.args for link in
    // links]))` — per block, the phi for each inputarg paired with the
    // incoming-link actuals at that position.
    let mut inputs: HashMap<BlockRef, Vec<(Variable, Vec<FlowValue>)>> = HashMap::new();
    for (block, links) in entrymap.iter() {
        let inputargs = block.borrow().inputargs.clone();
        let mut phis: Vec<(Variable, Vec<FlowValue>)> = Vec::with_capacity(inputargs.len());
        for (i, input) in inputargs.iter().enumerate() {
            let FlowValue::Variable(input_v) = input else {
                continue;
            };
            let mut phi_args: Vec<FlowValue> = Vec::with_capacity(links.len());
            for link in links {
                let arg =
                    link.borrow().args.get(i).cloned().flatten().expect(
                        "remove_identical_vars_SSA: link.args position missing for inputarg",
                    );
                phi_args.push(arg);
            }
            phis.push((*input_v, phi_args));
        }
        inputs.insert(block.clone(), phis);
    }

    // `progress = True; while progress: for block in inputs: if
    // simplify_phis(block): progress = True`.
    let block_keys: Vec<BlockRef> = inputs.keys().cloned().collect();
    let mut progress = true;
    while progress {
        progress = false;
        for block in &block_keys {
            if simplify_phis_inner(&mut uf, inputs.get_mut(block).unwrap()) {
                progress = true;
            }
        }
    }

    // `renaming = dict((key, uf[key].rep) for key in uf)` — restricted to the
    // Variable keys, since only Variable storage slots are renamed.
    let renaming: HashMap<Variable, FlowValue> = {
        let keys: Vec<FlowValue> = uf.keys().cloned().collect();
        let mut out: HashMap<Variable, FlowValue> = HashMap::new();
        for key in keys {
            let FlowValue::Variable(kv) = key else {
                continue;
            };
            if let Some(info) = uf.get(&FlowValue::Variable(kv)) {
                out.insert(kv, info.rep.clone());
            }
        }
        out
    };

    // Rewrite each block's inputargs and every incoming link's args to match
    // the pruned `inputs[block]` (inputargs shrink before renamevariables).
    for (block, phis) in inputs.iter() {
        let links = entrymap
            .get(block)
            .cloned()
            .expect("entrymap lookup consistent with inputs");
        let new_inputs: Vec<FlowValue> =
            phis.iter().map(|(v, _)| FlowValue::Variable(*v)).collect();
        // `per_link_args[link_idx][phi_idx] = phis[phi_idx].1[link_idx]`.
        let per_link_args: Vec<Vec<Option<FlowValue>>> = (0..links.len())
            .map(|li| phis.iter().map(|(_, pa)| Some(pa[li].clone())).collect())
            .collect();
        block.borrow_mut().inputargs = new_inputs;
        assert_eq!(links.len(), per_link_args.len());
        for (link, args) in links.iter().zip(per_link_args.into_iter()) {
            link.borrow_mut().args = args;
        }
    }

    for block in inputs.keys() {
        renamevariables_value(block, &renaming);
    }
}

/// Inner of `remove_identical_vars_ssa`'s `simplify_phis(block)` closure
/// (`simplify.py:555-573`).
fn simplify_phis_inner(
    uf: &mut UnionFind<FlowValue, Representative>,
    phis: &mut Vec<(Variable, Vec<FlowValue>)>,
) -> bool {
    let mut to_remove: Vec<usize> = Vec::new();
    let mut unique_phis: HashMap<Vec<FlowValue>, Variable> = HashMap::new();
    for (i, (input, phi_args)) in phis.iter().enumerate() {
        // `new_args = [uf.find_rep(arg) for arg in phi_args]`.
        let new_args: Vec<FlowValue> = phi_args.iter().map(|a| uf.find_rep(a.clone())).collect();
        // `if all_equal(new_args) and not isspecialvar(new_args[0]):`
        if let Some(first) = new_args.first().cloned() {
            if all_equal(&new_args) && !isspecialvar(&first) {
                uf.union(first, FlowValue::Variable(*input));
                to_remove.push(i);
                continue;
            }
        }
        // else branch — group by identical phi-tuple.
        if let Some(existing) = unique_phis.get(&new_args).copied() {
            uf.union(FlowValue::Variable(existing), FlowValue::Variable(*input));
            to_remove.push(i);
        } else {
            unique_phis.insert(new_args, *input);
        }
    }
    // `for i in reversed(to_remove): del phis[i]`.
    for i in to_remove.iter().rev() {
        phis.remove(*i);
    }
    !to_remove.is_empty()
}

// ── constfold_exitswitch ────────────────────────────────────────────────

/// `rpython/translator/simplify.py` `replace_exitswitch_by_constant`.
/// Port reference: `majit-translate/src/translator/simplify.rs`'s
/// `replace_exitswitch_by_constant`.
///
/// ```py
/// def replace_exitswitch_by_constant(block, const):
///     newexits = [link for link in block.exits if link.exitcase == const.value]
///     if len(newexits) == 0:
///         newexits = [link for link in block.exits if link.exitcase == 'default']
///     assert len(newexits) == 1
///     newexits[0].exitcase = None
///     if hasattr(newexits[0], 'llexitcase'):
///         newexits[0].llexitcase = None
///     block.exitswitch = None
///     block.recloseblock(*newexits)
///     return newexits
/// ```
pub fn replace_exitswitch_by_constant(block: &BlockRef, const_: &Constant) -> Vec<LinkRef> {
    // `link.exitcase == const.value` — exitcase wraps a Constant carrying a
    // `ConstantValue`; compare on the inner value.
    let cases_eq = |link: &LinkRef| match &link.borrow().exitcase {
        Some(FlowValue::Constant(c)) => c.value == const_.value,
        _ => false,
    };
    let default_case = |link: &LinkRef| {
        matches!(
            &link.borrow().exitcase,
            Some(FlowValue::Constant(c)) if c.value == ConstantValue::Str("default".to_string())
        )
    };

    let exits_snapshot: Vec<LinkRef> = block.borrow().exits.clone();
    let mut newexits: Vec<LinkRef> = exits_snapshot
        .iter()
        .filter(|l| cases_eq(l))
        .cloned()
        .collect();
    if newexits.is_empty() {
        newexits = exits_snapshot
            .iter()
            .filter(|l| default_case(l))
            .cloned()
            .collect();
    }
    assert_eq!(
        newexits.len(),
        1,
        "replace_exitswitch_by_constant: no unique surviving exit"
    );
    {
        let mut l = newexits[0].borrow_mut();
        l.exitcase = None;
        l.llexitcase = None;
    }
    block.borrow_mut().exitswitch = None;
    block.recloseblock(newexits.clone());
    newexits
}

/// `rpython/translator/simplify.py` `constfold_exitswitch`.
/// Port reference: `majit-translate/src/translator/simplify.rs`'s
/// `constfold_exitswitch`.
///
/// When a block's `exitswitch` has been folded to a `Constant` (and the block
/// cannot raise), only one exit can be taken — drop the others.
pub fn constfold_exitswitch(graph: &FunctionGraph) {
    let mut seen: HashSet<BlockRef> = HashSet::new();
    seen.insert(graph.startblock.clone());
    let mut stack: Vec<LinkRef> = graph.startblock.borrow().exits.clone();

    while let Some(link) = stack.pop() {
        let target = match link.borrow().target.clone() {
            Some(target) => target,
            None => continue,
        };
        if seen.contains(&target) {
            continue;
        }
        let source = match link.borrow().prevblock_ref() {
            Some(source) => source,
            None => continue,
        };

        // `switch = source.exitswitch`; act only on a Constant switch in a
        // non-raising block (the `c_last_exception` sentinel is a Constant but
        // makes `canraise()` true, so it is correctly skipped here).
        let (const_val, is_canraise) = {
            let b = source.borrow();
            match &b.exitswitch {
                Some(ExitSwitch::Value(FlowValue::Constant(c))) => (Some(c.clone()), b.canraise()),
                _ => (None, b.canraise()),
            }
        };

        if let Some(const_val) = const_val {
            if !is_canraise {
                let new_exits = replace_exitswitch_by_constant(&source, &const_val);
                stack.extend(new_exits);
                continue;
            }
        }
        seen.insert(target.clone());
        stack.extend(target.borrow().exits.clone());
    }
}

// ── simplify_exceptions (structural adaptation — see classification) ─────

/// `rpython/translator/simplify.py` `simplify_exceptions`.
///
/// **Classification (issue #112 scope #4): structural adaptation.**
///
/// Upstream `simplify_exceptions` collapses the `except Exception:`
/// chain-of-`is_`/`issubtype`-tests that RPython's *flowspace* emits into a
/// single list of `exitcase=cls` links on the raising block.  That chain is a
/// property of RPython's flow-graph construction; the port reference
/// (`majit-translate/src/translator/simplify.rs`'s `simplify_exceptions`)
/// reproduces it because the AOT translator path consumes flowspace graphs.
///
/// pyre-jit's production codewriter is a **CPython-bytecode walker**, not the
/// RPython flowspace.  Two concrete Pyre constraints make a literal port
/// inapplicable here:
///
///  1. The walker never emits the `is_`/`issubtype` exception-dispatch chain
///     (no `is_`/`issubtype` ops exist anywhere in `pyre-jit/src/jit`).  Its
///     exception model "bakes type into per-subclass" exits
///     (`flatten.rs`'s `is_pyre_canonical_elidable_hlop`), so the input shape
///     this pass targets is never present.
///  2. The subclass oracle the pass needs — `issubclass(case, cov)` and
///     `issubclass(case, BaseException)` (`simplify.py:140,146`) — has no
///     equivalent on pyre-jit's flow `Constant`s, which carry exception types
///     as opaque host-object handles without a hierarchy.
///
/// So this is a no-op on every graph the current walker produces.  It is NOT
/// silently dropped: a tripwire panics if the `is_`/`issubtype` chain shape is
/// ever observed (e.g. once the #97 MIR front-end emits an explicit
/// flowspace-style CFG), signalling that the full collapse must then be ported
/// for real rather than relying on this adaptation.
pub fn simplify_exceptions(graph: &FunctionGraph) {
    for block in graph.iterblocks() {
        let b = block.borrow();
        if !b.canraise() {
            continue;
        }
        // The chain begins at the Exception exit's target, whose first op is
        // the `is_`/`issubtype` test.  Detect that shape without needing the
        // Exception-class oracle.
        let Some(exc) = b.exits.last() else { continue };
        let Some(query) = exc.borrow().target.clone() else {
            continue;
        };
        let q = query.borrow();
        if q.exits.len() == 2 {
            if let Some(op) = q.operations.first() {
                assert!(
                    op.opname != "is_" && op.opname != "issubtype",
                    "simplify_exceptions: an is_/issubtype exception-dispatch \
                     chain is present in the walker graph — port the full \
                     `simplify.py:110-170` collapse (issue #112) instead of \
                     relying on the structural-adaptation no-op"
                );
            }
        }
    }
}

// ── transform_dead_op_vars ──────────────────────────────────────────────

/// `rpython/translator/simplify.py` `CanRemove` — the fixed opname set
/// of side-effect-free operations.  The upstream `enum_ops_without_sideeffects()`
/// addition (the LL-operation table, `simplify.py:414-416`) has no pyre-jit
/// equivalent: pyre-jit's flow ops are CPython-bytecode-derived, not RPython LL
/// ops, so only the literal opname list ports.
const CAN_REMOVE: &[&str] = &[
    "newtuple",
    "newlist",
    "newdict",
    "bool",
    "is_",
    "id",
    "type",
    "issubtype",
    "isinstance",
    "repr",
    "str",
    "len",
    "hash",
    "getattr",
    "getitem",
    "pos",
    "neg",
    "abs",
    "hex",
    "oct",
    "ord",
    "invert",
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "div",
    "mod",
    "divmod",
    "pow",
    "lshift",
    "rshift",
    "and_",
    "or_",
    "xor",
    "int",
    "float",
    "long",
    "lt",
    "le",
    "eq",
    "ne",
    "gt",
    "ge",
    "cmp",
    "coerce",
    "contains",
    "not_contains",
    "iter",
    "get",
];

fn can_remove_opname(op: &str) -> bool {
    CAN_REMOVE.contains(&op)
}

/// `rpython/translator/simplify.py` `transform_dead_op_vars`
/// (`transform_dead_op_vars_in_blocks`).  Port reference:
/// `majit-translate/src/translator/simplify.rs`'s `transform_dead_op_vars`.
///
/// Classification (scope #4): **direct PyPy parity**.  Removes side-effect-free
/// operations whose result is never read, then drops link args / inputargs that
/// are never used.  pyre-jit's codewriter always runs the single-graph,
/// `translator=None` path, so the upstream `direct_call` / `simple_call`
/// side-effect-analysis branches (`simplify.py:489-506`, gated on
/// `translator is not None`) are not reached and are omitted.  At the
/// post-walk / pre-regalloc point the walker's `block.operations` are empty, so
/// the op-removal half is a no-op today; the link-arg / inputarg pruning half is
/// active and feeds a tighter graph to regalloc's coalescing.
pub fn transform_dead_op_vars(graph: &FunctionGraph) {
    let blocks = graph.iterblocks();
    let set_of_blocks: HashSet<BlockRef> = blocks.iter().cloned().collect();
    let startblock = graph.startblock.clone();

    // `op.opname in CanRemove and op is not block.raising_op`.  `raising_op` is
    // `operations[-1]` when `canraise()`, so positional identity matches the
    // upstream object-identity check.
    let canremove_op = |op: &SpaceOperation, block: &BlockRef, idx: usize| -> bool {
        if !can_remove_opname(&op.opname) {
            return false;
        }
        let b = block.borrow();
        if !b.canraise() {
            return true;
        }
        idx + 1 != b.operations.len()
    };

    let mut read_vars: HashSet<Variable> = HashSet::new();
    let mut dependencies: HashMap<Variable, HashSet<Variable>> = HashMap::new();

    // Pass 1: compute read_vars + dependencies.
    for block in &blocks {
        let b = block.borrow();
        for (idx, op) in b.operations.iter().enumerate() {
            if !canremove_op(op, block, idx) {
                // `read_vars.update(op.args)`.
                for arg in &op.args {
                    for v in arg.variables() {
                        read_vars.insert(v);
                    }
                }
            } else if let Some(FlowValue::Variable(rv)) = &op.result {
                // `dependencies[op.result].update(op.args)`.
                for arg in &op.args {
                    for v in arg.variables() {
                        dependencies.entry(*rv).or_default().insert(v);
                    }
                }
            }
        }
        // `if isinstance(block.exitswitch, Variable): read_vars.add(block.exitswitch)`.
        if let Some(ExitSwitch::Value(FlowValue::Variable(sw))) = &b.exitswitch {
            read_vars.insert(*sw);
        }

        if !b.exits.is_empty() {
            for link in &b.exits {
                let link_b = link.borrow();
                let Some(target) = link_b.target.as_ref() else {
                    continue;
                };
                let in_set = set_of_blocks.contains(target);
                let target_inputargs = target.borrow().inputargs.clone();
                for (arg_opt, tgt) in link_b.args.iter().zip(target_inputargs.iter()) {
                    let Some(arg) = arg_opt else { continue };
                    if !in_set {
                        if let FlowValue::Variable(av) = arg {
                            read_vars.insert(*av);
                        }
                        if let FlowValue::Variable(tv) = tgt {
                            read_vars.insert(*tv);
                        }
                    } else if let (FlowValue::Variable(tv), FlowValue::Variable(av)) = (tgt, arg) {
                        dependencies.entry(*tv).or_default().insert(*av);
                    }
                }
            }
        } else {
            // return / except blocks implicitly use their inputargs.
            for a in &b.inputargs {
                if let FlowValue::Variable(v) = a {
                    read_vars.insert(*v);
                }
            }
        }

        // A start block's inputargs are always live.
        if *block == startblock {
            for a in &b.inputargs {
                if let FlowValue::Variable(v) = a {
                    read_vars.insert(*v);
                }
            }
        }
    }

    // `flow_read_var_backward(set(read_vars))`.
    let mut pending: Vec<Variable> = read_vars.iter().copied().collect();
    while let Some(var) = pending.pop() {
        if let Some(deps) = dependencies.get(&var).cloned() {
            for prev in deps {
                if read_vars.insert(prev) {
                    pending.push(prev);
                }
            }
        }
    }

    // Pass 2: remove dead ops, then dead link args, then dead inputargs.
    for block in &blocks {
        // Backward walk over operations, removing removable ops whose result
        // is never read.  (No-op while walker blocks carry no operations.)
        let mut i = block.borrow().operations.len();
        while i > 0 {
            i -= 1;
            let (result_used, removable) = {
                let b = block.borrow();
                let op = &b.operations[i];
                let result_used = match &op.result {
                    Some(FlowValue::Variable(v)) => read_vars.contains(v),
                    // Constant or void result — never removed by this path.
                    _ => true,
                };
                (result_used, can_remove_opname(&op.opname))
            };
            if result_used {
                continue;
            }
            // Only the CanRemove branch is reachable in pyre-jit (translator is
            // always None, so simple_call/direct_call removal is skipped).
            if removable && canremove_op(&block.borrow().operations[i], block, i) {
                block.borrow_mut().operations.remove(i);
            }
        }

        // Drop output vars never used from link.args (before shrinking
        // inputargs, to keep the same-index cross-block invariant).
        let exits_snapshot: Vec<LinkRef> = block.borrow().exits.clone();
        for link in exits_snapshot {
            let target = link.borrow().target.clone();
            let Some(target) = target else { continue };
            let target_inputargs = target.borrow().inputargs.clone();
            let args_len = link.borrow().args.len();
            assert_eq!(args_len, target_inputargs.len(), "link arity mismatch");
            let mut i = args_len;
            while i > 0 {
                i -= 1;
                let drop = matches!(&target_inputargs[i], FlowValue::Variable(v) if !read_vars.contains(v));
                if drop {
                    link.borrow_mut().args.remove(i);
                }
            }
        }
    }

    // Final pass: drop unused inputargs (matching link args already removed).
    for block in &blocks {
        let inputargs = block.borrow().inputargs.clone();
        let mut i = inputargs.len();
        while i > 0 {
            i -= 1;
            let drop = matches!(&inputargs[i], FlowValue::Variable(v) if !read_vars.contains(v));
            if drop {
                block.borrow_mut().inputargs.remove(i);
            }
        }
    }
}

// ── op-scanning passes: structural-adaptation no-ops (absent walker shapes) ──
//
// `coalesce_bool`, `transform_xxxitem`, and `transform_ovfcheck` all key on a
// block's *operations* (`bool` / `getitem` raising-op / `ovfcheck` simple_call).
// pyre's walker records post-rtype shapes (`residual_call_*`, vable field ops)
// on `block.operations` directly; the pre-rtype `bool` / raising `getitem` /
// `ovfcheck` simple_call shapes these passes key on never appear.  Each is a
// documented no-op with a tripwire that panics if the target shape appears,
// signalling that the full pass must then be ported.

/// `rpython/translator/simplify.py` `coalesce_bool`.
/// Classification (scope #4): structural adaptation (no `bool` op present).
pub fn coalesce_bool(graph: &FunctionGraph) {
    for block in graph.iterblocks() {
        let b = block.borrow();
        if let Some(last) = b.operations.last() {
            assert!(
                last.opname != "bool",
                "coalesce_bool: a `bool`-terminated block is present in the \
                 walker graph — port the full `simplify.py:656-699` coalesce \
                 (issue #112) instead of the structural-adaptation no-op"
            );
        }
    }
}

/// `rpython/translator/simplify.py` `transform_xxxitem`.
/// Classification (scope #4): structural adaptation (no `getitem` raising-op,
/// and pyre-jit has no flowspace `IndexError` exitcase oracle).
pub fn transform_xxxitem(graph: &FunctionGraph) {
    for block in graph.iterblocks() {
        let b = block.borrow();
        if let Some(last_op) = b.raising_op() {
            assert!(
                last_op.opname != "getitem",
                "transform_xxxitem: a `getitem` raising-op is present in the \
                 walker graph — port the full `simplify.py:172-186` rewrite \
                 (issue #112) instead of the structural-adaptation no-op"
            );
        }
    }
}

/// `rpython/translator/simplify.py` `transform_ovfcheck`.
/// Classification (scope #4): structural adaptation.  `ovfcheck` is an RPython
/// `rlib` function the flowspace emits as a `simple_call`; the CPython-bytecode
/// walker never emits it, and `OverflowingOperation.ovfchecked()` (the `<op>_ovf`
/// variant map) has no pyre-jit equivalent.
pub fn transform_ovfcheck(graph: &FunctionGraph) {
    for block in graph.iterblocks() {
        let b = block.borrow();
        for op in &b.operations {
            assert!(
                op.opname != "simple_call",
                "transform_ovfcheck: a `simple_call` op is present in the walker \
                 graph — port the full `simplify.py:71-108` ovfcheck rewrite \
                 (issue #112) instead of the structural-adaptation no-op"
            );
        }
    }
}

// ── exception-flowspace passes: structural-adaptation no-ops ─────────────
//
// `remove_dead_exceptions` and `remove_assertion_errors` both depend on RPython
// flowspace exception shapes that pyre-jit's CPython-bytecode walker does not
// produce, and on a class-hierarchy / implicit-exception oracle that pyre-jit's
// flow `Constant`s do not carry.  They are documented no-ops, not tripwires:
// pyre-jit exception *exits* legitimately exist, so a shape tripwire would
// false-fire.

/// `rpython/translator/simplify.py` `remove_dead_exceptions`.
/// Classification (scope #4): structural adaptation.  The pass merges/prunes
/// exception exits via `issubclass(case, member)` (`simplify.py:205,210`); pyre-jit
/// represents exception exitcases as opaque host-object handles with no subclass
/// oracle on the flow graph, so the shadowing decision cannot be evaluated and the
/// pass conservatively keeps every exit.
pub fn remove_dead_exceptions(_graph: &FunctionGraph) {
    // No-op: see classification above. Keeping all exits is the safe shape when
    // the `issubclass` oracle is unavailable.
}

/// `rpython/translator/simplify.py` `remove_assertion_errors`.
/// Classification (scope #4): structural adaptation.  Removes branches that go
/// directly to `graph.exceptblock` raising a `Constant(AssertionError)` — RPython's
/// `_implicit_` exception shape (`flowcontext.py`).  The CPython-bytecode walker
/// does not emit implicit-AssertionError exits, so the upstream condition
/// (`exit.args[0] == Constant(AssertionError)`) never matches; the pass is a
/// faithful no-op on every walker graph.
pub fn remove_assertion_errors(_graph: &FunctionGraph) {
    // No-op: see classification above.
}

// ── checkgraph + the simplify_graph driver ──────────────────────────────

/// `rpython/flowspace/model.py` `checkgraph(graph)` — the pyre-jit
/// structural subset.
///
/// The full upstream checker validates SSA/SSI definition, exception-link
/// extras, and exitswitch shape; flow.rs already enforces most of those at
/// construction (`Link::new`/`settarget`/`recloseblock` arity asserts).  This
/// subset re-verifies the two invariants the simplify passes most directly
/// touch — link arity and the `link.prevblock` back-pointer — as the
/// `simplify_graph` tail sanity check.
pub fn checkgraph(graph: &FunctionGraph) {
    for block in graph.iterblocks() {
        let exits: Vec<LinkRef> = block.borrow().exits.clone();
        for link in exits {
            let l = link.borrow();
            if let Some(target) = &l.target {
                assert_eq!(
                    l.args.len(),
                    target.borrow().inputargs.len(),
                    "checkgraph: link arity does not match target inputargs"
                );
            }
            assert!(
                l.prevblock_ref().is_some_and(|prev| prev == block),
                "checkgraph: link.prevblock does not point back to its source block"
            );
        }
    }
}

/// `rpython/translator/simplify.py` `all_passes`.
///
/// The order matches upstream exactly.  This uses the faithful
/// operations-based [`eliminate_empty_blocks`] above (NOT the walker-only
/// `codewriter::eliminate_empty_blocks`, whose `block.dead` predicate suits the
/// inline walker but would collapse every block of a normal flow graph).
/// `SSA_to_SSI` lives in `ssa.rs`.
pub fn all_passes() -> &'static [fn(&FunctionGraph)] {
    &[
        transform_dead_op_vars,
        eliminate_empty_blocks,
        remove_assertion_errors,
        remove_identical_vars_ssa,
        constfold_exitswitch,
        remove_trivial_links,
        super::ssa::ssa_to_ssi,
        coalesce_bool,
        transform_ovfcheck,
        simplify_exceptions,
        transform_xxxitem,
        remove_dead_exceptions,
    ]
}

/// `rpython/translator/simplify.py` `simplify_graph(graph, passes=True)`.
///
/// ```py
/// def simplify_graph(graph, passes=True):
///     if passes is True:
///         passes = all_passes
///     for pass_ in passes:
///         pass_(graph)
///     checkgraph(graph)
/// ```
pub fn simplify_graph(graph: &FunctionGraph) {
    for pass_ in all_passes() {
        pass_(graph);
    }
    checkgraph(graph);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::flatten::Kind;
    use crate::jit::flow::{Block, Link, VariableId, push_op};

    /// A non-trivial link (source has a real exitswitch / multiple exits) must
    /// be left untouched.
    #[test]
    fn remove_trivial_links_keeps_non_trivial() {
        let start = Block::shared(vec![]);
        let mut graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();

        // A link that DOES carry an arg is not trivial.
        let v = graph.fresh_variable(Kind::Int);
        let mid = graph.new_block(vec![v.into()]);
        push_op(
            &mid,
            SpaceOperation::new("int_zero", vec![], Some(v.into()), -1),
        );
        start.closeblock(vec![
            Link::new(vec![v.into()], Some(mid.clone()), None).into_ref(),
        ]);
        mid.closeblock(vec![
            Link::new(vec![v.into()], Some(returnblock.clone()), None).into_ref(),
        ]);

        remove_trivial_links(&graph);

        // start keeps its own (empty) operations and still points at mid.
        let s = start.borrow();
        assert!(s.operations.is_empty());
        assert_eq!(s.exits.len(), 1);
        assert_eq!(s.exits[0].borrow().target, Some(mid));
    }

    /// Mirrors `majit-translate/.../simplify.rs::remove_identical_vars_ssa_dedupes_duplicate_phis`.
    /// `body` has two inputargs, each fed the SAME constant on the single
    /// incoming link, so both phis collapse and `body` loses ≥1 inputarg.
    #[test]
    fn remove_identical_vars_ssa_dedupes_duplicate_phis() {
        let v_a = Variable::new(VariableId(10), Kind::Int);
        let v_b = Variable::new(VariableId(11), Kind::Int);
        let body = Block::shared(vec![v_a.into(), v_b.into()]);
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();

        let link_sb = Link::new(
            vec![Constant::signed(1).into(), Constant::signed(1).into()],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link_sb]);

        // Build the link in its own statement so `body.borrow()` drops
        // before `closeblock` takes `borrow_mut`.
        let body_arg = body.borrow().inputargs[0].clone();
        let link_br = Link::new(vec![body_arg], Some(returnblock), None).into_ref();
        body.closeblock(vec![link_br]);

        remove_identical_vars_ssa(&graph);

        // body lost at least one inputarg.
        assert!(body.borrow().inputargs.len() < 2);
    }

    /// A block whose `exitswitch` is a constant `true` keeps only the matching
    /// (`exitcase == true`) exit; the false arm is dropped.
    #[test]
    fn constfold_exitswitch_picks_matching_exit() {
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();

        let yes = graph.new_block(vec![]);
        let no = graph.new_block(vec![]);

        // start switches on the constant `true`.
        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(Constant::bool(true).into()));
        let link_yes =
            Link::new(vec![], Some(yes.clone()), Some(Constant::bool(true).into())).into_ref();
        let link_no =
            Link::new(vec![], Some(no.clone()), Some(Constant::bool(false).into())).into_ref();
        start.closeblock(vec![link_yes, link_no]);

        // `yes`/`no` both fall through to the returnblock (1 inputarg).
        yes.closeblock(vec![
            Link::new(
                vec![Constant::signed(0).into()],
                Some(returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        no.closeblock(vec![
            Link::new(vec![Constant::signed(0).into()], Some(returnblock), None).into_ref(),
        ]);

        constfold_exitswitch(&graph);

        let s = start.borrow();
        assert!(s.exitswitch.is_none());
        assert_eq!(s.exits.len(), 1);
        assert_eq!(s.exits[0].borrow().target, Some(yes));
        // the surviving exit's case was cleared.
        assert!(s.exits[0].borrow().exitcase.is_none());
    }

    /// `transform_dead_op_vars` drops a non-start block inputarg that is never
    /// read, along with the matching incoming-link arg.
    #[test]
    fn transform_dead_op_vars_drops_unused_inputarg() {
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();

        // `middle` receives a dead inputarg that nothing uses.
        let v_dead = graph.fresh_variable(Kind::Int);
        let middle = graph.new_block(vec![v_dead.into()]);

        start.closeblock(vec![
            Link::new(vec![Constant::signed(0).into()], Some(middle.clone()), None).into_ref(),
        ]);
        middle.closeblock(vec![
            Link::new(vec![Constant::signed(0).into()], Some(returnblock), None).into_ref(),
        ]);

        transform_dead_op_vars(&graph);

        // The dead inputarg and its incoming-link arg are gone.
        assert_eq!(middle.borrow().inputargs.len(), 0);
        assert_eq!(start.borrow().exits[0].borrow().args.len(), 0);
    }

    /// The op-scanning structural-adaptation no-ops leave a walker-shaped graph
    /// (empty operations) untouched and do not trip their tripwires.
    #[test]
    fn op_scanning_noops_are_inert_on_empty_ops() {
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![Constant::signed(0).into()], Some(returnblock), None).into_ref(),
        ]);

        // None of these should panic on an empty-operations graph.
        coalesce_bool(&graph);
        transform_xxxitem(&graph);
        transform_ovfcheck(&graph);
        remove_dead_exceptions(&graph);
        remove_assertion_errors(&graph);
        simplify_exceptions(&graph);
    }

    /// The full `simplify_graph` driver runs every pass in order, collapses a
    /// trivial link, and passes `checkgraph` without panicking.
    #[test]
    fn simplify_graph_runs_all_passes_and_checkgraph() {
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();

        let v = graph.fresh_variable(Kind::Int);
        let mid = graph.new_block(vec![]);
        push_op(
            &mid,
            SpaceOperation::new("int_zero", vec![], Some(v.into()), -1),
        );
        // Trivial link start -> mid (no args).
        start.closeblock(vec![Link::new(vec![], Some(mid.clone()), None).into_ref()]);
        mid.closeblock(vec![
            Link::new(vec![v.into()], Some(returnblock.clone()), None).into_ref(),
        ]);

        simplify_graph(&graph);

        // remove_trivial_links folded mid into start; start now reaches the
        // returnblock directly and checkgraph (run inside simplify_graph) held.
        let s = start.borrow();
        assert_eq!(s.exits.len(), 1);
        assert_eq!(s.exits[0].borrow().target, Some(returnblock));
        assert_eq!(s.operations.len(), 1);
        assert_eq!(s.operations[0].opname, "int_zero");
    }

    /// The faithful (operations-based) eliminate_empty_blocks collapses a
    /// NON-dead empty forwarding block that carries args — the shape
    /// `remove_trivial_links` cannot (its incoming link has args).
    #[test]
    fn eliminate_empty_blocks_collapses_empty_forwarder_with_args() {
        // start -> empty(ve) -> next(vn);  empty has no operations and forwards
        // its inputarg.  start's link should retarget straight to `next`,
        // substituting the constant it passed into `empty`.
        let ve = Variable::new(VariableId(40), Kind::Int);
        let vn = Variable::new(VariableId(41), Kind::Int);
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();
        let empty = graph.new_block(vec![ve.into()]);
        let next = graph.new_block(vec![vn.into()]);
        // `next` carries a real op so it is NOT itself an empty forwarder
        // (otherwise the faithful pass would collapse it too).
        push_op(
            &next,
            SpaceOperation::new("keep_alive", vec![vn.into()], None, -1),
        );

        start.closeblock(vec![
            Link::new(vec![Constant::signed(7).into()], Some(empty.clone()), None).into_ref(),
        ]);
        let empty_arg = empty.borrow().inputargs[0].clone();
        empty.closeblock(vec![
            Link::new(vec![empty_arg], Some(next.clone()), None).into_ref(),
        ]);
        next.closeblock(vec![
            Link::new(vec![Constant::signed(0).into()], Some(returnblock), None).into_ref(),
        ]);

        eliminate_empty_blocks(&graph);

        // start's link now skips `empty` and targets `next`, carrying the
        // substituted constant.
        let s = start.borrow();
        assert_eq!(s.exits[0].borrow().target, Some(next));
        assert_eq!(
            s.exits[0].borrow().args,
            vec![Some(Constant::signed(7).into())]
        );
    }

    /// Issue #112 scope #3 conclusion as a regression guard: the walker-safe
    /// subset wired ahead of flatten (`eliminate_empty_blocks` +
    /// `constfold_exitswitch` + `remove_trivial_links`) removes every shape that
    /// yields an unmarked label.  A graph carrying BOTH a dead constant-switch
    /// arm and a trivial empty forwarder must, after the subset, leave no
    /// reachable link pointing at a dropped or dead block — the port-boundary
    /// invariant the assembler `patch_labels` backstop guards.
    #[test]
    fn wired_subset_leaves_no_reachable_link_to_unmarked_block() {
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();

        // The live (true) arm goes through a trivial empty forwarder; the dead
        // (false) arm reaches `dead_arm`, which only the folded-away case names.
        let fwd = graph.new_block(vec![]);
        let tail = graph.new_block(vec![]);
        let dead_arm = graph.new_block(vec![]);

        // `tail` carries a real op so it is not itself an empty forwarder.
        let v = graph.fresh_variable(Kind::Int);
        push_op(
            &tail,
            SpaceOperation::new("int_zero", vec![], Some(v.into()), -1),
        );

        // start switches on the constant `true`: true -> fwd, false -> dead_arm.
        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(Constant::bool(true).into()));
        let link_true =
            Link::new(vec![], Some(fwd.clone()), Some(Constant::bool(true).into())).into_ref();
        let link_false = Link::new(
            vec![],
            Some(dead_arm.clone()),
            Some(Constant::bool(false).into()),
        )
        .into_ref();
        start.closeblock(vec![link_true, link_false]);

        fwd.closeblock(vec![Link::new(vec![], Some(tail.clone()), None).into_ref()]);
        tail.closeblock(vec![
            Link::new(
                vec![Constant::signed(0).into()],
                Some(returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        dead_arm.closeblock(vec![
            Link::new(vec![Constant::signed(0).into()], Some(returnblock), None).into_ref(),
        ]);

        // The wired subset, in the codewriter's relative order.
        eliminate_empty_blocks(&graph);
        constfold_exitswitch(&graph);
        remove_trivial_links(&graph);

        // Port-boundary invariant: every reachable link targets a live block, so
        // flatten emits a Label for every referenced target (no unmarked label).
        for link in graph.iterlinks() {
            if let Some(target) = link.borrow().target.clone() {
                assert!(
                    !target.borrow().dead,
                    "reachable link still targets a dead block"
                );
            }
        }
        // The dropped switch arm and the collapsed forwarder are unreachable.
        let reachable = graph.iterblocks();
        assert!(!reachable.contains(&dead_arm));
        assert!(!reachable.contains(&fwd));
        // The normalized graph is flatten-ready.
        checkgraph(&graph);
    }
}
