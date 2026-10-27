//! Port of `rpython/translator/backendopt/merge_if_blocks.py`.
//!
//! Convert consecutive blocks that all compare a variable (of a
//! Primitive type) with a constant into one block with multiple
//! exits — backends materialise the result as a `switch` statement.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::{
    BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, LinkArg,
    LinkRef, Variable, mkentrymap,
};

/// RPython `is_chain_block(block, first=False)` (`merge_if_blocks.py`).
fn is_chain_block(block: &BlockRef, first: bool) -> bool {
    let b = block.borrow();
    if b.operations.is_empty() {
        return false;
    }
    if b.operations.len() > 1 && !first {
        return false;
    }
    let op = b.operations.last().expect("non-empty per :8");
    // RPython `:13-14`: only int_eq / uint_eq / char_eq / unichar_eq
    // qualify. `llong_eq` / `ullong_eq` are intentionally absent
    // (upstream comment :14-16 — switching on a `long long` is not
    // strictly C-compliant and crashes the JIT).
    if !matches!(
        op.opname.as_str(),
        "int_eq" | "uint_eq" | "char_eq" | "unichar_eq"
    ) {
        return false;
    }
    // RPython `:17 op.result != block.exitswitch`.
    if Some(&op.result) != b.exitswitch.as_ref() {
        return false;
    }
    if op.args.len() != 2 {
        return false;
    }
    // RPython `:19-22`: exactly one arg must be Variable, the other
    // Constant.
    let lhs_var = matches!(&op.args[0], Hlvalue::Variable(_));
    let rhs_var = matches!(&op.args[1], Hlvalue::Variable(_));
    if lhs_var == rhs_var {
        // Both Variable or both Constant — neither qualifies.
        return false;
    }
    // RPython `:23-30`: drop unhashable constants (`hash(value)` would
    // raise `TypeError`). Every [`ConstValue`] derives `Hash` in our
    // port, so this filter is a no-op modulo carriers we never use as
    // chain-eq keys (e.g. mutable List). Mirror upstream by rejecting
    // those structurally — the only ConstValue carriers without a
    // stable identity-hash are mutable containers.
    let const_arg = if lhs_var { &op.args[1] } else { &op.args[0] };
    if let Hlvalue::Constant(c) = const_arg
        && !is_hashable_const(&c.value)
    {
        return false;
    }
    true
}

/// Mirror of upstream's "the constant is hashable" filter at
/// `merge_if_blocks.py`. Python's `hash(value)` raises
/// `TypeError` on `list` / `dict` / `set` and on tuples that
/// transitively contain an unhashable element. Tuples of hashable
/// values (the realistic shape in eq-chain constants) hash fine
/// upstream and must hash here too — the previous "every Tuple is
/// unhashable" blanket dropped chain candidates upstream would have
/// merged.
fn is_hashable_const(value: &ConstValue) -> bool {
    match value {
        ConstValue::Dict(_) | ConstValue::List(_) => false,
        ConstValue::Tuple(items) => items.iter().all(is_hashable_const),
        _ => true,
    }
}

/// Mirror of Python `bool(value) is False` for the `ConstValue`
/// carriers that can appear as `Link.exitcase`. Upstream uses
/// `assert not falseexit.exitcase` (`merge_if_blocks.py`) which
/// folds through Python's truthiness rules: `None`, `False`, `0`,
/// `0.0`, `""`, `b""`, `()`, `[]`, `{}` are all falsy. Anything
/// else is truthy.
fn is_falsy(value: &ConstValue) -> bool {
    match value {
        ConstValue::None => true,
        ConstValue::Bool(b) => !*b,
        ConstValue::Int(n) => *n == 0,
        ConstValue::UInt(n) => *n == 0,
        ConstValue::Float(bits) => f64::from_bits(*bits) == 0.0,
        ConstValue::ByteStr(s) => s.is_empty(),
        ConstValue::UniStr(s) => s.is_empty(),
        ConstValue::Tuple(items) => items.is_empty(),
        ConstValue::List(items) => items.is_empty(),
        ConstValue::Dict(d) => d.is_empty(),
        // Truthy in Python: non-zero numbers, non-empty containers,
        // and every other carrier (Atom, Code, Function, Graphs,
        // LowLevelType, etc.). Returning `false` here mirrors
        // upstream's "anything else short-circuits the `not`".
        _ => false,
    }
}

/// RPython `merge_chain(chain, checkvar, varmap, graph)`
/// (`merge_if_blocks.py`).
///
/// * `chain` — list of `(block, case_const)` pairs, in chain order.
/// * `checkvar` — the Variable that the first block compares against
///   each case.
/// * `varmap` — substitution from "any-block-in-chain Variable" to
///   "first-block Variable" so the merged block's link args route
///   correctly.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn merge_chain(
    chain: &[(BlockRef, Hlvalue)],
    checkvar: &Hlvalue,
    varmap: &HashMap<Variable, LinkArg>,
) {
    // Closure mirroring upstream `:34-37 get_new_arg`. Upstream's
    // `varmap[var_or_const]` returns whatever was stored — a Variable
    // or a Constant (or, in pyre, a `LinkArg::None`). The
    // pre-existing port narrowed `varmap`'s value to `Hlvalue`, which
    // silently dropped the `None` case at insert time
    // (`add_to_varmap` returned without inserting). We now carry
    // `LinkArg = Option<Hlvalue>` end-to-end so the None is
    // preserved verbatim — matching upstream `:83 varmap[newvar] = var`.
    let get_new_arg = |arg: &LinkArg| -> LinkArg {
        match arg.as_ref() {
            None => None,
            Some(Hlvalue::Constant(_)) => arg.clone(),
            Some(Hlvalue::Variable(v)) => varmap.get(v).cloned().unwrap_or_else(|| arg.clone()),
        }
    };

    // RPython `:38 firstblock, _ = chain[0]`.
    let firstblock = chain[0].0.clone();
    {
        // RPython `:39-40`: drop the trailing eq op and rewire
        // `exitswitch` to checkvar.
        let mut b = firstblock.borrow_mut();
        b.operations.pop();
        b.exitswitch = Some(checkvar.clone());
    }

    // RPython `:42-46`: pull the default link off the *last* chain
    // block's False exit.
    let default_link = {
        let last = chain.last().expect("merge_chain: empty chain").0.clone();
        let last_b = last.borrow();
        last_b.exits[0].clone()
    };
    {
        let mut l = default_link.borrow_mut();
        l.exitcase = Some(Hlvalue::Constant(Constant::new(ConstValue::ByteStr(
            b"default".to_vec(),
        ))));
        l.llexitcase = None;
        l.args = l.args.iter().map(&get_new_arg).collect();
    }

    // RPython `:47-60`: walk the chain, collecting per-case True
    // links. Skip a case if its `value` was already seen (upstream
    // tolerates dup cases silently, comment :49-51). Upstream's
    // `values` dict is keyed on `case.value` (the raw Python value),
    // not on the Constant wrapper — two Constants with the same
    // value but different `concretetype` (or distinct identities for
    // identity-hashed carriers) must collapse into one case.
    let mut seen_values: HashSet<ConstValue> = HashSet::new();
    let mut links: Vec<LinkRef> = Vec::new();
    for (block, case) in chain {
        let case_const = match case {
            Hlvalue::Constant(c) => c.clone(),
            // upstream guarantees `case` is a Constant (chain
            // construction picks `case = [v for ... if isinstance(v,
            // Constant)][0]`); a non-Constant slip is a bug, not a
            // skip.
            Hlvalue::Variable(_) => {
                unreachable!("merge_if_blocks.py:91-92 case must be a Constant")
            }
        };
        if !seen_values.insert(case_const.value.clone()) {
            continue;
        }
        let link = {
            let b = block.borrow();
            b.exits[1].clone()
        };
        {
            let mut l = link.borrow_mut();
            l.exitcase = Some(Hlvalue::Constant(case_const.clone()));
            l.llexitcase = Some(Hlvalue::Constant(case_const));
            l.args = l.args.iter().map(&get_new_arg).collect();
        }
        links.push(link);
    }
    links.push(default_link);
    // RPython `:61 firstblock.recloseblock(*links)`.
    firstblock.recloseblock(links);
}

/// RPython `merge_if_blocks_once(graph)` (`merge_if_blocks.py`).
///
/// Returns `true` when a chain was merged (caller should retry).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn merge_if_blocks_once(graph: &FunctionGraph) -> bool {
    let mut candidates: Vec<BlockRef> = Vec::new();
    for block in graph.iterblocks() {
        if is_chain_block(&block, true) {
            candidates.push(block);
        }
    }
    let entrymap = mkentrymap(graph);

    for firstblock in &candidates {
        let mut chain: Vec<(BlockRef, Hlvalue)> = Vec::new();
        let mut checkvars: Vec<Hlvalue> = Vec::new();
        // upstream `varmap = {var: var for var in firstblock.exits[0|1].args}`
        // — `var` here is the link arg (Variable or Constant). The
        // pyre port stores the LinkArg verbatim so a None arg can
        // round-trip through `add_to_varmap` unchanged.
        let mut varmap: HashMap<Variable, LinkArg> = HashMap::new();
        for exit_idx in 0..=1 {
            let link = {
                let b = firstblock.borrow();
                b.exits[exit_idx].clone()
            };
            let l = link.borrow();
            for arg in &l.args {
                if let Some(Hlvalue::Variable(v)) = arg {
                    varmap.insert(v.clone(), Some(Hlvalue::Variable(v.clone())));
                }
            }
        }

        let mut current = firstblock.clone();
        loop {
            // RPython `:88-92`:
            //   checkvar = the Variable arg of the trailing eq op.
            //   resvar   = the result var of the eq op.
            //   case     = the Constant arg of the eq op.
            let (checkvar, resvar, case) = {
                let b = current.borrow();
                let op = b.operations.last().expect("is_chain_block guards op");
                let checkvar = op
                    .args
                    .iter()
                    .find_map(|a| match a {
                        Hlvalue::Variable(_) => Some(a.clone()),
                        _ => None,
                    })
                    .expect("is_chain_block guards exactly one Variable arg");
                let case = op
                    .args
                    .iter()
                    .find_map(|a| match a {
                        Hlvalue::Constant(_) => Some(a.clone()),
                        _ => None,
                    })
                    .expect("is_chain_block guards exactly one Constant arg");
                (checkvar, op.result.clone(), case)
            };
            checkvars.push(checkvar.clone());
            let (falseexit, trueexit) = {
                let b = current.borrow();
                (b.exits[0].clone(), b.exits[1].clone())
            };
            // RPython `:95 assert not falseexit.exitcase`. Upstream
            // stores `Link.exitcase` as the raw Python value (the
            // bool `False`, the int case, the string `"default"`,
            // …); `not exitcase` then uses Python truthiness. The
            // pyre port wraps the exitcase as
            // `Option<Hlvalue::Constant(...)>` so the structural
            // equivalent is "exitcase is `None`, or its
            // `ConstValue` is Python-falsy". Mirror Python truthiness
            // across every ConstValue carrier we use as exitcase
            // (None / Bool / Int / Float / ByteStr / UniStr /
            // Tuple / List / Dict).
            assert!(
                match falseexit.borrow().exitcase.as_ref() {
                    None => true,
                    Some(Hlvalue::Constant(c)) => is_falsy(&c.value),
                    Some(_) => false,
                },
                "merge_if_blocks.py:95 falseexit.exitcase must be falsy",
            );
            let target = match falseexit.borrow().target.as_ref() {
                Some(t) => t.clone(),
                None => break,
            };
            // RPython `:100-101`: if the eq's result is forwarded
            // through either exit's args, abort the chain.
            let resvar_in_falseexit = falseexit
                .borrow()
                .args
                .iter()
                .any(|a| a.as_ref() == Some(&resvar));
            let resvar_in_trueexit = trueexit
                .borrow()
                .args
                .iter()
                .any(|a| a.as_ref() == Some(&resvar));
            if resvar_in_falseexit || resvar_in_trueexit {
                break;
            }
            chain.push((current.clone(), case));
            // RPython `:103-104`: target must have exactly one
            // incoming link (else extending the chain merges in
            // an unrelated control path).
            let entries = entrymap
                .get(&BlockKey::of(&target))
                .map(Vec::len)
                .unwrap_or(0);
            if entries != 1 {
                break;
            }
            // RPython `:105-106`: checkvar must be live across the
            // False edge.
            let checkvar_pos = falseexit
                .borrow()
                .args
                .iter()
                .position(|a| a.as_ref() == Some(&checkvar));
            let Some(checkvar_pos) = checkvar_pos else {
                break;
            };
            // RPython `:107`: the next-block input arg corresponding
            // to checkvar.
            let newcheckvar = {
                let tb = target.borrow();
                tb.inputargs
                    .get(checkvar_pos)
                    .cloned()
                    .expect("inputargs aligns with link.args by index")
            };
            // RPython `:108-109`: target must continue the chain.
            if !is_chain_block(&target, false) {
                break;
            }
            // RPython `:110-111`: the new check must operate on the
            // forwarded variable.
            let next_op_args = target
                .borrow()
                .operations
                .first()
                .map(|o| o.args.clone())
                .unwrap_or_default();
            if !next_op_args.iter().any(|a| a == &newcheckvar) {
                break;
            }
            // RPython `:112-115`: extend varmap by tracking how the
            // current block's exit args land in the next block's
            // inputargs.
            for (i, var) in trueexit.borrow().args.iter().enumerate() {
                let tt = trueexit.borrow().target.clone();
                let inputargs = tt
                    .as_ref()
                    .map(|t| t.borrow().inputargs.clone())
                    .unwrap_or_default();
                if let Some(Hlvalue::Variable(newvar)) = inputargs.get(i) {
                    add_to_varmap(&mut varmap, var, newvar.clone());
                }
            }
            for (i, var) in falseexit.borrow().args.iter().enumerate() {
                let ft = falseexit.borrow().target.clone();
                let inputargs = ft
                    .as_ref()
                    .map(|t| t.borrow().inputargs.clone())
                    .unwrap_or_default();
                if let Some(Hlvalue::Variable(newvar)) = inputargs.get(i) {
                    add_to_varmap(&mut varmap, var, newvar.clone());
                }
            }
            current = target;
        }
        if chain.len() > 1 {
            // RPython `:117-118 break` — the for-else terminates
            // here, falling through to merge_chain + return True.
            merge_chain(&chain, &checkvars[0], &varmap);
            return true;
        }
    }
    // RPython `:119-120 else: return False` — the Python for-else
    // path, reached when no candidate produced a chain length > 1.
    false
}

/// Mirror of upstream's nested `add_to_varmap` (`merge_if_blocks.py`).
///
/// Upstream:
///
/// ```python
/// def add_to_varmap(var, newvar):
///     if isinstance(var, Variable):
///         varmap[newvar] = varmap[var]
///     else:
///         varmap[newvar] = var
/// ```
///
/// The else-branch stores `var` verbatim — for a Python Constant
/// that's the Constant; for pyre's `LinkArg::None` the same shape
/// must propagate. Earlier we narrowed `varmap` to
/// `HashMap<Variable, Hlvalue>` and silently dropped the None case;
/// `varmap` now carries `LinkArg` so `varmap[newvar] = var` is
/// always representable.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn add_to_varmap(varmap: &mut HashMap<Variable, LinkArg>, var: &LinkArg, newvar: Variable) {
    let value: LinkArg = match var {
        Some(Hlvalue::Variable(v)) => varmap
            .get(v)
            .cloned()
            .unwrap_or_else(|| Some(Hlvalue::Variable(v.clone()))),
        // Constant or None — upstream stores `var` verbatim.
        _ => var.clone(),
    };
    varmap.insert(newvar, value);
}

/// RPython `merge_if_blocks(graph, verbose=True)` at
/// `merge_if_blocks.py`.
///
/// `verbose` mirrors upstream's `verbose=True` keyword. Upstream
/// uses it to choose between `log("merging blocks in %s" %
/// graph.name)` and `log.dot()` over the `AnsiLogger("backendopt")`
/// at `:4` once the rewrite actually fired. pyre's translator does
/// not yet ship the `AnsiLogger`-keyed channel, so the body of each
/// branch is a no-op stub — keeping the `verbose` flag in the
/// signature lets call sites at upstream `all.py:54` /
/// `all.py:114` round-trip through the parity-shaped API
/// (`merge_if_blocks(graph, verbose=translator.config.translation
/// .verbose)`) without each caller learning a pyre-specific
/// adaptation.
pub fn merge_if_blocks(graph: &FunctionGraph, verbose: bool) {
    let mut merge = false;
    while merge_if_blocks_once(graph) {
        merge = true;
    }
    if merge {
        if verbose {
            // Upstream `:130 log("merging blocks in %s" %
            // (graph.name,))`. pyre's translator does not yet ship
            // an `AnsiLogger("backendopt")` channel; the parity stub
            // keeps the branch shape so future logger plumbing can
            // light up without touching call sites.
            let _ = graph.name.as_str();
        } else {
            // Upstream `:132 log.dot()` — progress indicator on the
            // same channel.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, ConstValue, Constant, Link, SpaceOperation};

    fn const_int(n: i64) -> Hlvalue {
        Hlvalue::Constant(Constant::new(ConstValue::Int(n)))
    }

    /// Build a 2-step chain `if x == 1: A elif x == 2: B else: C`.
    /// Block layout:
    ///   block0:  cmp1 = int_eq(x, 1); switch cmp1 → block1 (False) / block_a (True)
    ///   block1:  cmp2 = int_eq(x, 2); switch cmp2 → block_c (False) / block_b (True)
    fn build_chain_graph() -> (Variable, FunctionGraph) {
        let x = Variable::named("x");
        let cmp1 = Variable::named("cmp1");
        let cmp2 = Variable::named("cmp2");

        let block0 = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        block0.borrow_mut().operations.push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(x.clone()), const_int(1)],
            Hlvalue::Variable(cmp1.clone()),
        ));
        block0.borrow_mut().exitswitch = Some(Hlvalue::Variable(cmp1));

        let block_a = Block::shared(vec![Hlvalue::Variable(Variable::named("a_in"))]);
        let block_b = Block::shared(vec![Hlvalue::Variable(Variable::named("b_in"))]);
        let block_c = Block::shared(vec![Hlvalue::Variable(Variable::named("c_in"))]);

        let x2 = Variable::named("x2");
        let block1 = Block::shared(vec![Hlvalue::Variable(x2.clone())]);
        block1.borrow_mut().operations.push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(x2.clone()), const_int(2)],
            Hlvalue::Variable(cmp2.clone()),
        ));
        block1.borrow_mut().exitswitch = Some(Hlvalue::Variable(cmp2));

        // block0 exits: [False → block1, True → block_a]
        block0.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(x.clone())],
                Some(block1.clone()),
                Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(x.clone())],
                Some(block_a.clone()),
                Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
            )
            .into_ref(),
        ]);

        // block1 exits: [False → block_c, True → block_b]
        block1.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(x2.clone())],
                Some(block_c.clone()),
                Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(x2)],
                Some(block_b.clone()),
                Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
            )
            .into_ref(),
        ]);

        // Each leaf branch terminates by linking to the graph's
        // implicit returnblock (built by FunctionGraph::new).
        let graph = FunctionGraph::new("chain", block0);
        let return_arg = vec![Hlvalue::Variable(Variable::named("ret"))];
        for leaf in [&block_a, &block_b, &block_c] {
            leaf.closeblock(vec![
                Link::new(return_arg.clone(), Some(graph.returnblock.clone()), None).into_ref(),
            ]);
        }
        (x, graph)
    }

    #[test]
    fn merge_collapses_two_step_eq_chain_into_switch() {
        let (x, graph) = build_chain_graph();
        // Pre: block0 has 1 op, exitswitch points at cmp1.
        assert_eq!(graph.startblock.borrow().operations.len(), 1);
        assert_eq!(graph.startblock.borrow().exits.len(), 2);

        // Upstream `:124-132 merge_if_blocks(graph, verbose=True)`.
        // Pass `verbose=false` to skip the (no-op) logger branch.
        merge_if_blocks(&graph, false);

        // Post: block0 has 0 ops, exitswitch points at the
        // canonical Variable `x`, and exits now span 3 cases:
        // [int(1) → A, int(2) → B, "default" → C].
        let b = graph.startblock.borrow();
        assert_eq!(b.operations.len(), 0);
        assert_eq!(b.exits.len(), 3);
        match b.exitswitch.clone().unwrap() {
            Hlvalue::Variable(v) => assert_eq!(v.id(), x.id()),
            other => panic!("unexpected exitswitch: {other:?}"),
        }

        // Cases 0 / 1 carry the constant ints; case 2 is "default".
        let cases: Vec<_> = b
            .exits
            .iter()
            .map(|e| e.borrow().exitcase.clone().unwrap())
            .collect();
        assert!(matches!(
            &cases[0],
            Hlvalue::Constant(c) if c.value == ConstValue::Int(1)
        ));
        assert!(matches!(
            &cases[1],
            Hlvalue::Constant(c) if c.value == ConstValue::Int(2)
        ));
        match &cases[2] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::ByteStr(bs) => assert_eq!(bs.as_slice(), b"default"),
                other => panic!("expected ByteStr default, got {other:?}"),
            },
            other => panic!("expected Constant default, got {other:?}"),
        }
    }

    #[test]
    fn is_chain_block_rejects_non_eq_op() {
        let v = Variable::named("v");
        let b = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        b.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![Hlvalue::Variable(v.clone()), const_int(1)],
            Hlvalue::Variable(Variable::named("r")),
        ));
        b.borrow_mut().exitswitch = Some(Hlvalue::Variable(Variable::named("r")));
        assert!(!is_chain_block(&b, true));
    }

    #[test]
    fn is_chain_block_rejects_two_variable_args() {
        let v = Variable::named("v");
        let w = Variable::named("w");
        let b = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        b.borrow_mut().operations.push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(v), Hlvalue::Variable(w)],
            Hlvalue::Variable(Variable::named("r")),
        ));
        b.borrow_mut().exitswitch = Some(Hlvalue::Variable(Variable::named("r")));
        assert!(!is_chain_block(&b, true));
    }
}
