//! Legacy type resolution pass — transitional cutover input + reference.
//!
//! TODO(retire-legacy-resolve): flat `ConcreteType` enum with ad-hoc
//! lowering and no upstream `rpython/` counterpart — the orthodox port
//! is [`crate::translator::rtyper::rtyper::RPythonTyper`]
//! (`rtyper.py:RPythonTyper` + `rmodel.py:Repr` hierarchy), which
//! produces per-`Variable.concretetype` `LowLevelType` directly.
//!
//! This file remains because:
//!   * `dual_gate_check_with_registry`
//!     ([`crate::translator::rtyper::cutover`]) still compares the real
//!     path against `legacy_annotator::annotate(graph)` followed by
//!     `resolve_types(graph)` for parity diff, and
//!   * `transform_graph_to_jitcode`
//!     ([`crate::codewriter::codewriter`]) calls
//!     `resolve_rewritten_types(...)` to merge the post-jtransform
//!     `result_kind` declarations with `merge_synth_kinds`, and falls
//!     back to the legacy walker (`annotate(graph)` then
//!     `resolve_types(graph)`)
//!     when the dual-gate Skip-classifies.
//!
//! Retirement drops both consumers and this file together once Skip
//! categories close.
//!
//! Transforms annotated ValueTypes into concrete low-level types
//! and specializes operations accordingly.

use std::collections::HashSet;

use crate::codewriter::annotation_state::somevalue_to_valuetype;
use crate::codewriter::type_state::{
    ConcreteType, kind_char_to_concrete, valuetype_to_concrete_for_legacy,
};
use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{CallTarget, FunctionGraph, Link, LinkArg, OpKind};
use crate::translator::rtyper::rint::signed_repr;
use crate::translator::rtyper::rmodel::inputconst;

/// Resolve annotations to concrete types.
///
/// RPython equivalent: `RPythonTyper.specialize_block()` — walks
/// each block and converts annotation → Repr → lowleveltype.
///
/// Result is committed through `set_concretetype_of_inline` for
/// every populated `Variable.concretetype` cell; downstream consumers
/// read kinds via `FunctionGraph::concretetype_of(&v)`
/// (`getkind(v.concretetype)`).
pub fn resolve_types(graph: &FunctionGraph, raw_alloc_flavor_structs: &HashSet<String>) {
    resolve_types_inner(graph, raw_alloc_flavor_structs).unwrap_or_else(|e| panic!("{e}"));
}

fn resolve_types_inner(
    graph: &FunctionGraph,
    raw_alloc_flavor_structs: &HashSet<String>,
) -> Result<(), crate::translator::rtyper::error::TyperError> {
    crate::translator::rtyper::legacy_annotator::refuse_merged_frozen_pbc_reprs(graph)?;

    // RPython `rtyper.setup_block_entry` writes `Signed` for `etype`
    // and `GcRef` for `evalue` on every graph's exceptblock unconditionally
    // at rtyper time (`rpython/rtyper/rtyper.py:setup_block_entry`
    // exceptblock arm).  Run the seed before the annotation projection
    // loop, then exclude exceptblock inputargs from that loop so a stale
    // `SomeInstance(...)` annotation (left behind by a panicked real
    // rtyper run) cannot project back to `GcRef` and overwrite the
    // orthodox `Signed` on `etype`.
    let mut exceptblock_inputargs: Vec<Variable> = Vec::new();
    if let Some(exceptblock) = graph.blocks.get(graph.exceptblock.0) {
        if let Some(etype) = exceptblock.inputargs.first() {
            FunctionGraph::set_concretetype_of_inline(etype, ConcreteType::Signed);
            exceptblock_inputargs.push(etype.clone());
        }
        if let Some(evalue) = exceptblock.inputargs.get(1) {
            FunctionGraph::set_concretetype_of_inline(evalue, ConcreteType::GcRef);
            exceptblock_inputargs.push(evalue.clone());
        }
    }

    // Walk the orthodox `Variable.annotation` slot on every registered
    // graph variable (RPython `Variable.annotation`).  `Unknown`-projected
    // entries never appear because `legacy_annotator::setbinding` clears
    // the slot on `Unknown` — non-`Unknown` `valuetype_to_concrete`
    // outputs always produce a concrete kind, so every populated cell
    // commits a real
    // `FunctionGraph::set_concretetype_of_inline` write.
    // `rtyper.py setup_block_entry` always takes the inputarg repr from
    // its annotation (`rclass.py getinstancerepr`); a classless
    // `SomeInstance` is the root instance, flavor `'gc'`.
    for var in graph.iter_variables() {
        if exceptblock_inputargs.contains(&var) {
            continue;
        }
        let ann = var.annotation.borrow();
        if let Some(rc_some) = ann.as_ref() {
            let concrete = concrete_from_annotation(rc_some, raw_alloc_flavor_structs)?;
            commit_concretetype(&var, concrete);
        }
    }

    // Resolve from ops with explicit type info
    for block in &graph.blocks {
        // Resolve inputargs (Phi nodes) from `Variable.annotation`
        for var in &block.inputargs {
            if FunctionGraph::concretetype_of(var) == ConcreteType::Unknown {
                let concrete = match var.annotation.borrow().as_ref() {
                    Some(rc) => concrete_from_annotation(rc, raw_alloc_flavor_structs)?,
                    None => valuetype_to_concrete_for_legacy(
                        &crate::model::ValueType::Unknown,
                        raw_alloc_flavor_structs,
                    ),
                };
                if concrete != ConcreteType::Unknown {
                    commit_concretetype(var, concrete);
                }
            }
        }
        for op in &block.operations {
            if let Some(result_var) = op.result.as_ref() {
                let inferred = infer_concrete_from_op(&op.kind, raw_alloc_flavor_structs);
                if inferred != ConcreteType::Unknown {
                    // The op carries authoritative result kind (post-
                    // jtransform `float_*` rewrite, explicit Call /
                    // FieldRead / etc. result_ty).  Override any
                    // annotation default — annrpython defaults BinOp /
                    // UnaryOp result to Int, which would shadow a later
                    // Float inference for `float_add` / `float_neg` /
                    // etc. produced by jtransform's float-operand
                    // rewrite arm.
                    commit_concretetype(result_var, inferred);
                }
                seed_raw_ptr_cast_source(&op.kind, raw_alloc_flavor_structs);
            }
        }
    }

    // Cross-block: propagate through Link args → target inputargs.
    // Keep the exception-link split explicit, mirroring upstream's
    // `_convert_link()` handling of `last_exception` /
    // `last_exc_value` before the per-arg conversion loop.
    for block in &graph.blocks {
        for link in &block.exits {
            if link_is_raise_like(link) {
                convert_raise_link(graph, link);
            } else {
                convert_link(graph, link);
            }
        }
    }

    // Backward-constraint: an op whose RPython decorator declares
    // integer operands (`@arguments("i", ...)` in `blackhole.py`) cannot
    // run on `Ref`-classified operands — upstream `rtyper.py specialize`
    // determines operand kind from the op's `concretetype`, not from a
    // fallback.  Pyre's front-end leaves many `OpKind::Input` values
    // with `ty: Unknown` (path references not yet bound by the annotator);
    // without this pass they stay Unknown and `setconcretetype` fails
    // (`rtyper.py bindingrepr`). Upgrade Unknown operands of pure
    // integer ops to `Signed`, matching RPython's mandatory kind
    // assignment (`rint.py IntegerRepr` `hop.inputargs(self, self)`).
    // Void is Impossible/None (`IntegerRepr.convert_const` refuses None)
    // and stays Void. `ptr_eq` / `ptr_ne` remain Ref-Ref (rewritten in
    // `jtransform` per `jtransform.py _rewrite_cmp_ptrs`) and are skipped.
    fn is_int_unop(op: &str) -> bool {
        matches!(op, "neg" | "invert" | "not")
    }
    fn is_identity_unop(op: &str) -> bool {
        matches!(op, "same_as")
    }
    // Iterate: a backward-inferred Signed on one Variable may feed
    // another op's operand through Link propagation, so run until
    // fixed-point.  RPython's `rtyper.py:specialize` is a single
    // forward pass because every Variable has a `concretetype` from
    // annotation; pyre's partial typing means downstream ops can
    // depend on upstream inferences that only this pass supplies.
    loop {
        let mut changed = false;
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::BinOp {
                        op: opname,
                        lhs,
                        rhs,
                        ..
                    } if canonical_int_binop(opname).is_some() => {
                        // Only upgrade Unknown operands.  Forcing
                        // GcRef → Signed here is unsound: a value
                        // classified Ref may alias with pointer uses
                        // elsewhere in the graph, and demoting it to
                        // Int would misroute those other uses
                        // (regalloc puts the value in an int register,
                        // subsequent field reads dereference garbage).
                        // The remaining pyre-only `int_*/{ri,ir,rr}`
                        // keys need `cast_ptr_to_int` insertion in
                        // jtransform, not a type override in the
                        // rtyper. Void is Impossible/None; do not
                        // repair it to Signed.
                        changed |= maybe_seed_int_operand(lhs);
                        changed |= maybe_seed_int_operand(rhs);
                        if let Some(result) = op.result.as_ref() {
                            // RPython rtyper resolves an `add` whose
                            // operands are both `lltype.Float` to
                            // `float_add` directly (`rfloat.py
                            // rtype_add`); the result inherits Float
                            // (or Int for comparisons).  When pyre's
                            // upstream typing already classified both
                            // operands as Float, mirror that: seed the
                            // result as Float for arithmetic ops and
                            // Int for comparisons.  jtransform will
                            // then rewrite the BinOp's `op` from `add`
                            // to `float_add` (etc.) and insert
                            // `cast_int_to_float` for mixed int/float
                            // operands, keeping IR/regalloc consistent.
                            let lhs_ty = FunctionGraph::concretetype_of(lhs);
                            let rhs_ty = FunctionGraph::concretetype_of(rhs);
                            let any_float =
                                lhs_ty == ConcreteType::Float || rhs_ty == ConcreteType::Float;
                            let both_numeric =
                                matches!(lhs_ty, ConcreteType::Signed | ConcreteType::Float)
                                    && matches!(rhs_ty, ConcreteType::Signed | ConcreteType::Float);
                            // Mirror jtransform's float-rewrite set
                            // (`codewriter/jtransform.rs`):
                            // arithmetic `add/sub/mul/div` →
                            // `float_*` returns Float; comparisons
                            // `lt/le/gt/ge/eq/ne` → `float_*` returns
                            // Int (Bool — RPython `rfloat.py:133`,
                            // `blackhole.py bhimpl_float_eq`).
                            // `mod` is also Float when a Float operand is
                            // present, but jtransform lowers it to the
                            // residual `ll_math::math_fmod` call rather than a
                            // non-RPython `float_mod` opcode.
                            let is_compare =
                                matches!(opname.as_str(), "lt" | "le" | "gt" | "ge" | "eq" | "ne");
                            let is_arith =
                                matches!(opname.as_str(), "add" | "sub" | "mul" | "div" | "mod");
                            if any_float && both_numeric && (is_arith || is_compare) {
                                let target = if is_compare {
                                    ConcreteType::Signed
                                } else {
                                    ConcreteType::Float
                                };
                                if FunctionGraph::concretetype_of(result) != target {
                                    FunctionGraph::set_concretetype_of_inline(result, target);
                                    changed = true;
                                }
                            } else if !any_float || !both_numeric {
                                changed |= maybe_seed_int_operand(result);
                            }
                        }
                    }
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if is_int_unop(opname) => {
                        changed |= maybe_seed_int_operand(operand);
                        if let Some(result) = op.result.as_ref() {
                            // Same Float-operand override as the BinOp
                            // arm above.  Unary `neg` on a Float
                            // returns Float (`float_neg`).
                            let operand_float =
                                FunctionGraph::concretetype_of(operand) == ConcreteType::Float;
                            if operand_float && opname == "neg" {
                                if FunctionGraph::concretetype_of(result) != ConcreteType::Float {
                                    FunctionGraph::set_concretetype_of_inline(
                                        result,
                                        ConcreteType::Float,
                                    );
                                    changed = true;
                                }
                            } else {
                                changed |= maybe_seed_int_operand(result);
                            }
                        }
                    }
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if opname == "cast_int_to_float" => {
                        // RPython `rfloat.py` rtype_int → calls
                        // `hop.inputarg(Signed, ...)` before emitting
                        // `cast_int_to_float`, so the operand's
                        // concretetype is Signed by construction.
                        // jtransform's `coerce_operand_to_float`
                        // (codewriter/jtransform.rs) emits this
                        // op only when get_value_kind(operand) == 'i'
                        // in pass 1.  Pass 2 runs on the rewritten
                        // graph from a fresh state and may lose that
                        // backward-constraint upgrade if the operand
                        // lacks a definitive def-site classification
                        // (e.g., Call with Unknown result_ty), which
                        // would surface as `cast_int_to_float/r>f`
                        // at the assembler.  Re-seed Signed here so
                        // pass 2 converges to the same operand kind.
                        changed |= maybe_seed_int_operand(operand);
                    }
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if is_identity_unop(opname) => {
                        if let Some(result) = op.result.as_ref() {
                            let operand_ty = FunctionGraph::concretetype_of(operand);
                            let result_ty = FunctionGraph::concretetype_of(result);
                            changed |= maybe_seed_concrete_type(result, operand_ty);
                            changed |= maybe_seed_concrete_type(operand, result_ty);
                        }
                    }
                    _ => {}
                }
            }
            // Re-propagate along links after each op round so backward
            // inferences reach the linked Variables. RPython has concrete
            // types on both ends by this point and `_convert_link()` emits
            // conversions where needed; this legacy pass has no conversion
            // insertion, so Unknown values joined by a link must converge to
            // the known low-level type before regalloc/flatten.
            for link in &block.exits {
                changed |= if link_is_raise_like(link) {
                    converge_raise_link(graph, link)
                } else {
                    converge_link(graph, link)
                };
            }
        }
        if !changed {
            break;
        }
    }

    // `RPythonTyper.setconcretetype` writes
    // `v.concretetype = bindingrepr(v).lowleveltype` (`rtyper.py`).
    // There is no default: `bindingrepr` calls `annotator.binding`,
    // which is a KeyError when the Variable has no annotation. A
    // leftover Unknown is that typer bug — do not invent `GcRef`.
    #[expect(
        clippy::mutable_key_type,
        reason = "Variable equality and hashing use immutable identity IDs while annotations are excluded; this exactly models RPython object-identity dictionary keys without a divergent side table"
    )]
    let mut seen: std::collections::HashSet<Variable> = std::collections::HashSet::new();
    for block in &graph.blocks {
        collect_block_variables(block, &mut seen);
    }
    // A graph whose body never reaches `returnblock` still owns
    // `getreturnvar()`. `SomeImpossibleValue.rtyper_makerepr` is
    // `impossible_repr` (`rtyper.py` / `annmodel.py`) — Void, not a
    // heap object. Seed that before leftover Unknown is a TyperError.
    // Only a live predecessor counts (`FunctionGraph.iterblocks` walks
    // from `startblock`; `RPythonTyper.specialize_more_blocks` walks
    // `annotator.annotated`). A disconnected untyped link into
    // returnblock does not keep the return var live.
    let live_flags = live_block_flags(graph);
    let returnblock_has_pred = graph.blocks.iter().enumerate().any(|(idx, block)| {
        live_flags[idx]
            && block.id != graph.returnblock
            && block
                .exits
                .iter()
                .any(|link| link.target == graph.returnblock)
    });
    if !returnblock_has_pred
        && let Some(ret) = graph
            .blocks
            .get(graph.returnblock.0)
            .and_then(|block| block.inputargs.first())
        && FunctionGraph::concretetype_of(ret) == ConcreteType::Unknown
    {
        FunctionGraph::set_concretetype_of_inline(ret, ConcreteType::Void);
    }

    // `RPythonTyper.specialize` only walks `annotator.annotated` after
    // `annotator.simplify`. Skip-arm does not delete disconnected
    // blocks; their Variables have no binding. They are
    // `SomeImpossibleValue` → `impossible_repr` (Void). Live blocks are
    // `FunctionGraph.iterblocks` from startblock plus
    // `secondary_entry_roots`. Do not invent GcRef.
    #[expect(
        clippy::mutable_key_type,
        reason = "Variable equality and hashing use immutable identity IDs while annotations are excluded; this exactly models RPython object-identity dictionary keys without a divergent side table"
    )]
    let mut live_vars: std::collections::HashSet<Variable> = std::collections::HashSet::new();
    for (idx, block) in graph.blocks.iter().enumerate() {
        if live_flags[idx] {
            collect_block_variables(block, &mut live_vars);
        }
    }
    for var in &seen {
        if FunctionGraph::concretetype_of(var) == ConcreteType::Unknown && !live_vars.contains(var)
        {
            FunctionGraph::set_concretetype_of_inline(var, ConcreteType::Void);
        }
    }

    let leftover: Vec<&Variable> = live_vars
        .iter()
        .filter(|var| FunctionGraph::concretetype_of(var) == ConcreteType::Unknown)
        .collect();
    if let Some(var) = leftover.first() {
        let names: Vec<String> = leftover.iter().map(|v| v.name()).take(12).collect();
        panic!(
            "setconcretetype: no binding for {} in graph {:?} ({} leftover \
             Unknown Variable(s): {}; rtyper.py setconcretetype / bindingrepr)",
            var.name(),
            graph.name,
            leftover.len(),
            names.join(", "),
        );
    }

    // RPython parity: every `FunctionGraph::set_concretetype_of_inline(&var, ct)`
    // above publishes the resolved kind on each Variable's
    // `concretetype` cell, matching `rtyper.py setconcretetype v.concretetype = ...`.
    // Downstream consumers read kinds via
    // `FunctionGraph::concretetype_of(&v)` (i.e.
    // `getkind(v.concretetype)`) directly without a separate
    // `apply_to_graph` publish step.
    Ok(())
}

/// Reachability from `startblock` (`FunctionGraph.iterblocks`) plus
/// `secondary_entry_roots`. Those extra roots stand in for
/// `simplify.transform_dead_op_vars_in_blocks`'s every-graph-start root
/// when one `FunctionGraph` carries a second start (the `generated::*` /
/// `eval_loop_jit` body block). An unreached returnblock or exceptblock
/// is not live; its Unknown variables take `impossible_repr` (Void).
fn live_block_flags(graph: &FunctionGraph) -> Vec<bool> {
    let mut live = vec![false; graph.blocks.len()];
    let mut worklist: Vec<crate::model::BlockId> = vec![graph.startblock];
    worklist.extend(crate::model::secondary_entry_roots(graph));
    while let Some(block_id) = worklist.pop() {
        if std::mem::replace(&mut live[block_id.0], true) {
            continue;
        }
        for link in &graph.blocks[block_id.0].exits {
            worklist.push(link.target);
        }
    }
    live
}

fn collect_block_variables(block: &crate::model::Block, into: &mut HashSet<Variable>) {
    for var in &block.inputargs {
        into.insert(var.clone());
    }
    for op in &block.operations {
        for var in crate::inline::op_variable_refs(&op.kind) {
            into.insert(var);
        }
        if let Some(result) = op.result.as_ref() {
            into.insert(result.clone());
        }
    }
    for link in &block.exits {
        for arg in &link.args {
            if let Some(var) = arg.as_variable() {
                into.insert(var.clone());
            }
        }
        for arg in link.last_exception.iter().chain(link.last_exc_value.iter()) {
            if let Some(var) = arg.as_variable() {
                into.insert(var.clone());
            }
        }
    }
}

fn concrete_from_annotation(
    s: &crate::annotator::model::SomeValue,
    raw_alloc_flavor_structs: &HashSet<String>,
) -> Result<ConcreteType, crate::translator::rtyper::error::TyperError> {
    use crate::annotator::description::DescEntry;
    use crate::annotator::model::SomeValue;
    if let SomeValue::PBC(pbc) = s {
        let all_frozen = pbc
            .descriptions
            .values()
            .all(|d| matches!(d, DescEntry::Frozen(_)));
        if all_frozen {
            if pbc.descriptions.len() > 1 || pbc.can_be_none {
                return Err(
                    crate::translator::rtyper::legacy_annotator::multiple_frozen_pbc_unsupported(),
                );
            }
            // `getFrozenPBCRepr` → `SingleFrozenPBCRepr.lowleveltype = Void`.
            return Ok(ConcreteType::Void);
        }
    }
    let vtype = somevalue_to_valuetype(s);
    Ok(valuetype_to_concrete_for_legacy(
        &vtype,
        raw_alloc_flavor_structs,
    ))
}

fn const_value_to_concrete(value: &ConstValue) -> ConcreteType {
    match value {
        ConstValue::Int(_)
        | ConstValue::UInt(_)
        | ConstValue::Bool(_)
        | ConstValue::SpecTag(_)
        | ConstValue::AddressOffset(_)
        | ConstValue::InheritanceId { .. }
        | ConstValue::LLAddress(_)
        | ConstValue::AddressAsInt(_)
        | ConstValue::UniChar(_) => ConcreteType::Signed,
        ConstValue::Int128(_) | ConstValue::UInt128(_) => ConcreteType::Unknown,
        ConstValue::Float(_) | ConstValue::LongFloat(_) => ConcreteType::Float,
        ConstValue::SingleFloat(_) => ConcreteType::Unknown,
        ConstValue::Placeholder => ConcreteType::Unknown,
        ConstValue::Atom(_)
        | ConstValue::Dict(_)
        | ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::None
        | ConstValue::Code(_)
        | ConstValue::LLPtr(_)
        | ConstValue::Function(_)
        | ConstValue::HostObject(_)
        | ConstValue::Opaque(_) => ConcreteType::GcRef,
        // Bare-constant repr selection after annotation: one FrozenDesc
        // (`getFrozenPBCRepr` → `SingleFrozenPBCRepr`).
        // `SingleFrozenPBCRepr.lowleveltype` is `Void`; `history.getkind(Void)`
        // is `"void"`. `refuse_merged_frozen_pbc_reprs` refuses a dest
        // whose `follow_link` / `mergeinputargs` union is not a single
        // FrozenDesc (`getFrozenPBCRepr`), or — when the dest is
        // unannotated — distinct Repr identities or a mixed non-Repr arg.
        ConstValue::Repr(_) => ConcreteType::Void,
    }
}

fn link_is_raise_like(link: &Link) -> bool {
    link.last_exception.is_some() && link.last_exc_value.is_some()
}

fn convert_link(graph: &FunctionGraph, link: &Link) {
    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        // A `LinkArg::Const(_)` whose `concretetype` carries a
        // construction-site repr (e.g. `set_return`'s
        // `Constant(None, concretetype=Void)`) is authoritative —
        // mirrors RPython `_convert_link()` materialising
        // `inputconst(r_to, value)` whose own `concretetype` is the
        // destination repr (`rpython/rtyper/rmodel.py:inputconst`
        // + `rpython/rtyper/rnone.py convert_from_to`).  Force-write so an
        // earlier annotation-derived kind (e.g. `Ref → GcRef` from
        // `valuetype_to_concrete` on the upstream `SomeNone → Ref`
        // projection) does not shadow it.
        if let LinkArg::Const(value) = src
            && let Some(lltype) = value.concretetype.as_ref()
        {
            let incoming = crate::model::getkind(lltype);
            // `insert_link_conversions` converts the Constant INTO dest
            // (`rtyper.py`: `link.args[i] = inputconst(r_a2, a1.value)`).
            // Dest stays Signed; `IntegerRepr.convert_const` (`rint.py`)
            // raises TyperError for None.
            if incoming == ConcreteType::Void
                && FunctionGraph::concretetype_of(dst) == ConcreteType::Signed
            {
                let _ = inputconst(signed_repr().as_ref(), &value.value)
                    .unwrap_or_else(|e| panic!("{e}"));
                continue;
            }
            FunctionGraph::set_concretetype_of_inline(dst, incoming);
            continue;
        }
        // `insert_link_conversions` (`rtyper.py`) converts each Constant
        // into the dest repr. A bare Repr constant is `SingleFrozenPBCRepr`
        // (`getFrozenPBCRepr`). `refuse_merged_frozen_pbc_reprs` already
        // refused a dest whose `follow_link` / `mergeinputargs` union is
        // not a single FrozenDesc, or — when the dest is unannotated —
        // distinct Repr identities or a mixed non-Repr arg.
        let _ = maybe_seed_concrete_type(dst, link_arg_concrete_type(src));
    }
}

fn convert_raise_link(graph: &FunctionGraph, link: &Link) {
    if let Some(value) = link.last_exception.as_ref().and_then(|a| a.as_variable()) {
        let _ = maybe_seed_concrete_type(value, ConcreteType::Signed);
    }
    if let Some(value) = link.last_exc_value.as_ref().and_then(|a| a.as_variable()) {
        let _ = maybe_seed_concrete_type(value, ConcreteType::GcRef);
    }

    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        let src_ty = if Some(src) == link.last_exception.as_ref() {
            ConcreteType::Signed
        } else if Some(src) == link.last_exc_value.as_ref() {
            ConcreteType::GcRef
        } else {
            link_arg_concrete_type(src)
        };
        let _ = maybe_seed_concrete_type(dst, src_ty);
    }
}

fn converge_link(graph: &FunctionGraph, link: &Link) -> bool {
    let mut changed = false;
    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        match src {
            LinkArg::Value(src_var) => {
                let src_ty = FunctionGraph::concretetype_of(src_var);
                let dst_ty = FunctionGraph::concretetype_of(dst);
                changed |= maybe_seed_concrete_type(dst, src_ty);
                changed |= maybe_seed_concrete_type(src_var, dst_ty);
            }
            LinkArg::Const(_) => {
                // Route through `link_arg_concrete_type` so the
                // `Constant.concretetype` construction-site hint
                // (e.g. `set_return`'s `concretetype=Void`) is
                // honoured here too — bypassing it would let
                // `const_value_to_concrete(&value.value)` default
                // `None → GcRef` and shadow the Void write.
                changed |= maybe_seed_concrete_type(dst, link_arg_concrete_type(src));
            }
        }
    }
    changed
}

fn converge_raise_link(graph: &FunctionGraph, link: &Link) -> bool {
    let mut changed = false;
    if let Some(value) = link.last_exception.as_ref().and_then(|a| a.as_variable()) {
        changed |= maybe_seed_concrete_type(value, ConcreteType::Signed);
    }
    if let Some(value) = link.last_exc_value.as_ref().and_then(|a| a.as_variable()) {
        changed |= maybe_seed_concrete_type(value, ConcreteType::GcRef);
    }

    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        let src_ty = if Some(src) == link.last_exception.as_ref() {
            ConcreteType::Signed
        } else if Some(src) == link.last_exc_value.as_ref() {
            ConcreteType::GcRef
        } else {
            link_arg_concrete_type(src)
        };
        changed |= maybe_seed_concrete_type(dst, src_ty);
        if let LinkArg::Value(src_var) = src {
            let dst_ty = FunctionGraph::concretetype_of(dst);
            changed |= maybe_seed_concrete_type(src_var, dst_ty);
        }
    }
    changed
}

fn link_arg_concrete_type(src: &LinkArg) -> ConcreteType {
    match src {
        LinkArg::Value(var) => FunctionGraph::concretetype_of(var),
        // RPython `pairtype(Repr, NoneRepr).convert_from_to`
        // (`rpython/rtyper/rnone.py`) emits `inputconst(Void, None)`
        // when None flows into a `NoneRepr` target; the symmetric
        // `pairtype(NoneRepr, Repr).convert_from_to`
        // (`rpython/rtyper/rnone.py`) emits `inputconst(r_to, None)`
        // which is a null pointer for `Ptr`/ref targets.  Pyre's
        // construction sites that already know the target repr write
        // it onto `Constant.concretetype` at the construction site
        // (e.g. `set_return` wires `Constant(None, concretetype=Void)`
        // per `flowcontext.py RETURN_VALUE` + `:1232-1236`); honour that
        // construction-site hint here.  Falls back to the value-only
        // default (`getkind(Ptr) == GcRef` for None) when the
        // construction site did not set a target repr.
        LinkArg::Const(value) => match value.concretetype.as_ref() {
            Some(lltype) => crate::model::getkind(lltype),
            None => const_value_to_concrete(&value.value),
        },
    }
}

/// A raw `Ptr(Array)` (`CArrayPtr` / `CCHARP`) is the concretetype
/// `direct_ptradd` scales by. A later kind-only commit (`Signed` or
/// `GcRef`) must not replace it with the canonical scalar.
fn commit_concretetype(var: &Variable, concrete: ConcreteType) {
    if var.concretetype().as_ref().is_some_and(|ty| {
        matches!(
            ty,
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Ptr(ptr)
                if matches!(
                    ptr.TO,
                    crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Array(_)
                )
        )
    }) {
        return;
    }
    FunctionGraph::set_concretetype_of_inline(var, concrete);
}

/// Operations whose RPython decorator requires integer operands on
/// both sides (`blackhole.py+ @arguments("i", "i", returns="i")`).
/// Unlike the comparison ops `lt`/`le`/`gt`/`ge`, no ptr-specific
/// variant exists in RPython — these must see Signed operands.
/// `eq`/`ne` may still be rewritten to `ptr_eq`/`ptr_ne` in
/// `jtransform.rs` when both operands are Ref (`_rewrite_cmp_ptrs`);
/// they are omitted so that pass observes the original Ref-Ref shape.
pub(crate) fn canonical_int_binop(op: &str) -> Option<&'static str> {
    match op {
        "add" | "add_assign" => Some("add"),
        "sub" | "sub_assign" => Some("sub"),
        "mul" | "mul_assign" => Some("mul"),
        "div" | "div_assign" => Some("div"),
        "mod" | "mod_assign" => Some("mod"),
        "and" | "bitand" | "bitand_assign" => Some("and"),
        "or" | "bitor" | "bitor_assign" => Some("or"),
        "xor" | "bitxor" | "bitxor_assign" => Some("xor"),
        "rshift" | "shr" | "rshift_assign" => Some("rshift"),
        "lshift" | "shl" | "lshift_assign" => Some("lshift"),
        // `_int_to_int_cast` narrow signed target. Both operands are
        // signed words (`int_signext(v, nbytes)`).
        "signext" => Some("signext"),
        "lt" => Some("lt"),
        "le" => Some("le"),
        "gt" => Some("gt"),
        "ge" => Some("ge"),
        _ => None,
    }
}

/// `rint.py IntegerRepr` `hop.inputarg(self)` types the operand Signed.
/// Unknown is the only kind this pass fills in. Void is Impossible/None
/// (`IntegerRepr.convert_const` refuses None) and stays Void.
/// GcRef/Float stay put: demoting a Ref is unsound.
fn maybe_seed_int_operand(dst: &Variable) -> bool {
    match FunctionGraph::concretetype_of(dst) {
        ConcreteType::Unknown => {
            FunctionGraph::set_concretetype_of_inline(dst, ConcreteType::Signed);
            true
        }
        _ => false,
    }
}

fn maybe_seed_concrete_type(dst: &Variable, src_ty: ConcreteType) -> bool {
    if FunctionGraph::concretetype_of(dst) == ConcreteType::Unknown
        && src_ty != ConcreteType::Unknown
    {
        // RPython parity: `rtyper.py setconcretetype v.concretetype = ...` writes the
        // resolved kind inline on the Variable as soon as the resolver
        // knows it.  Pyre's iterative build mirrors that by publishing
        // through the Variable cell so subsequent `concretetype_of(dst)`
        // sees the kind during the same pass.
        FunctionGraph::set_concretetype_of_inline(dst, src_ty);
        true
    } else {
        false
    }
}

/// A FieldRead of a raw pointer is often `Ref(None)`; the following
/// `__cast_instance_intrinsic` carries `Ref(Some(intern_key))`.
/// `history.py getkind` of that raw `Ptr` is `'int'`. An untyped
/// source shares the dest bank. A GC source (`*const ()` from
/// `w_code_get_ptr`) stays Ref: `rewrite_op_direct_call` emits
/// `cast_ptr_to_int` for that crossing (`lltype.py`).
fn seed_raw_ptr_cast_source(kind: &OpKind, raw_alloc_flavor_structs: &HashSet<String>) {
    let OpKind::Call {
        target,
        args,
        result_ty,
    } = kind
    else {
        return;
    };
    let CallTarget::FunctionPath { segments, .. } = target else {
        return;
    };
    if segments.last().map(String::as_str) != Some("__cast_instance_intrinsic") {
        return;
    }
    if valuetype_to_concrete_for_legacy(result_ty, raw_alloc_flavor_structs) != ConcreteType::Signed
    {
        return;
    }
    let Some(src) = args.first().and_then(LinkArg::as_variable) else {
        return;
    };
    if FunctionGraph::concretetype_of(src) == ConcreteType::Unknown {
        FunctionGraph::set_concretetype_of_inline(src, ConcreteType::Signed);
    }
}

fn infer_concrete_from_op(
    kind: &OpKind,
    raw_alloc_flavor_structs: &HashSet<String>,
) -> ConcreteType {
    match kind {
        OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } => ConcreteType::Signed,
        // Unsigned shares the JIT integer register bank with Signed; the
        // annotation/repr distinction is carried separately.
        OpKind::ConstUInt(_) => ConcreteType::Signed,
        // RPython `getkind(lltype.Bool) == 'int'` (flatten.py:getkind);
        // codewriter folds Bool storage to int kind.
        OpKind::ConstBool(_) => ConcreteType::Signed,
        // `_we_are_jitted` symbolic carries `Bool` concretetype, whose
        // `getkind` folds to int kind — same legacy concrete as
        // `ConstBool` so the dual-gate projection stays convergent.
        OpKind::ConstSymbolic { .. } => ConcreteType::Signed,
        OpKind::ConstFloat(_) => ConcreteType::Float,
        // RPython `rpython/annotator/annrpython.py` types every Variable
        // at annotation time, so `OpKind::Input` reaching rtyper has a
        // concrete type.  pyre's `front::mir` emits each graph parameter
        // as an `OpKind::Input { ty }` whose `ty` is projected from the
        // MIR local's declared type, which is `ValueType::Unknown`
        // whenever that type cannot be resolved.  Leave Unknown so the
        // integer-op backward-constraint pass (`resolve_types`) can
        // upgrade operands of pure integer ops to `Signed` before
        // leftover Unknown fails `setconcretetype`.
        OpKind::Input { ty, .. } => valuetype_to_concrete_for_legacy(ty, raw_alloc_flavor_structs),
        // Field / array reads whose declared `ty` is pyre-only Unknown
        // default to `ConcreteType::GcRef` so the value reaches
        // regalloc. RPython's rtyper resolves the field / element type
        // from the struct's `concretetype`; pyre's struct registry can
        // lack the entry (Rust generics, unsupported declarations),
        // leaving `ValueType::Unknown`.  Ref is the conservative
        // default — if the underlying value is actually Int, the
        // canonical `getfield_gc_i/rd>i` key still carries the
        // correct result kind at assembler emit time.
        OpKind::FieldRead { ty, .. } => {
            let c = valuetype_to_concrete_for_legacy(ty, raw_alloc_flavor_structs);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::ArrayRead { item_ty, .. } => {
            let c = valuetype_to_concrete_for_legacy(item_ty, raw_alloc_flavor_structs);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::InteriorFieldRead { item_ty, .. } => {
            let c = valuetype_to_concrete_for_legacy(item_ty, raw_alloc_flavor_structs);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::Call { result_ty, .. } => {
            valuetype_to_concrete_for_legacy(result_ty, raw_alloc_flavor_structs)
        }
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => kind_char_to_concrete(*result_kind),
        OpKind::UnaryOp { op, result_ty, .. } if op == "same_as" => {
            valuetype_to_concrete_for_legacy(result_ty, raw_alloc_flavor_structs)
        }
        // For `BinOp` / `UnaryOp` whose `result_ty` is still
        // `ValueType::Unknown`, return `ConcreteType::Unknown` rather
        // than defaulting to `Signed`.  The override at line ~55 only
        // fires when `inferred != Unknown`; previously the `Signed`
        // fallback overrode the annotator's value (e.g. the `neg(Float)
        // → Float` annotation produced by the dedicated `op == "neg"`
        // arm in `infer_op_type`), poisoning downstream phi-merge
        // inputargs whose link arg types are correctly Float.  With
        // `Unknown` here the override is skipped and the annotator's
        // value persists; the loop below still upgrades the concrete
        // type to `Float` for `neg(Float)` etc. via the special arms.
        OpKind::BinOp { result_ty, .. } | OpKind::UnaryOp { result_ty, .. } => {
            valuetype_to_concrete_for_legacy(result_ty, raw_alloc_flavor_structs)
        }
        // Vtable funcptr extraction returns an integer pointer (RPython
        // `op.args[0]` of `indirect_call` is `Ptr(FuncType)`).
        OpKind::VtableMethodPtr { .. } => ConcreteType::Signed,
        OpKind::IndirectCall { result_ty, .. } => {
            valuetype_to_concrete_for_legacy(result_ty, raw_alloc_flavor_structs)
        }
        // Virtualizable field / array reads inherit the RPython
        // canonical result kind from the declared field/element type.
        OpKind::VableFieldRead { ty, .. } => {
            let c = valuetype_to_concrete_for_legacy(ty, raw_alloc_flavor_structs);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::VableArrayRead { item_ty, .. } => {
            let c = valuetype_to_concrete_for_legacy(item_ty, raw_alloc_flavor_structs);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        // A length, not an element — no `Unknown`-to-`GcRef` fallback applies.
        // `arraylen_gc/rd>i` is the same length word as `arraylen_vable`.
        OpKind::ArrayLen { .. } | OpKind::VableArrayLen { .. } => ConcreteType::Signed,
        // `OpKind::Abort` for unsupported syntax — macros, unsupported
        // literals, fallback expressions.  Fall back to GcRef so these values
        // still get a regalloc coloring and the assembler's
        // `lookup_reg_with_kind` covers every operand. RPython has no
        // analogue; porting each producer path individually eliminates
        // the `abort/` wire keys.
        OpKind::Abort { .. } => ConcreteType::GcRef,
        _ => ConcreteType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeInstance, SomeValue};
    use crate::model::{
        CallTarget, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType, exception_exitcase,
    };
    use crate::translator::rtyper::legacy_annotator as annotate;
    use std::rc::Rc;

    #[test]
    fn resolves_int_types() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v_var = graph
            .push_op_var(entry, OpKind::ConstInt(42), true)
            .unwrap();
        graph.set_return(entry, Some(v_var.clone()));

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(FunctionGraph::concretetype_of(&v_var), ConcreteType::Signed);
    }

    #[test]
    fn resolves_ref_field() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let base_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "obj".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let v_var = graph
            .push_op_var(
                entry,
                OpKind::FieldRead {
                    base: base_var,
                    field: crate::model::FieldDescriptor::new("obj", None),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v_var.clone()));

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(FunctionGraph::concretetype_of(&v_var), ConcreteType::GcRef);
    }

    #[test]
    fn resolves_phi_through_link_args() {
        let mut graph = FunctionGraph::new("phi");
        let entry = graph.startblock;
        let val_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let (target, phi_args) = graph.create_block_with_arg_vars(1);
        let phi_var = phi_args[0].clone();
        graph.set_goto(entry, target, vec![val_var]);
        graph.set_return(target, Some(phi_var.clone()));

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(
            FunctionGraph::concretetype_of(&phi_var),
            ConcreteType::Signed
        );
    }

    #[test]
    fn instance_phi_resolves_from_classdef_flavor() {
        use crate::annotator::classdesc::ClassDef;

        fn build(name: &str) -> (FunctionGraph, crate::flowspace::model::Variable) {
            let mut graph = FunctionGraph::new(name);
            let entry = graph.startblock;
            let call_var = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::function_path(["frame", "code"]),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("CodeObject".into())),
                    },
                    true,
                )
                .unwrap();
            let (target, phi_args) = graph.create_block_with_arg_vars(1);
            let phi_var = phi_args[0].clone();
            graph.set_goto(entry, target, vec![call_var.clone()]);
            graph.set_return(target, Some(phi_var.clone()));
            (graph, phi_var)
        }

        let raw = HashSet::from(["bytecode::CodeObject".to_string()]);
        let (graph, phi) = build("classed_raw_phi");
        let classdef = ClassDef::new_standalone("bytecode::CodeObject", None);
        let classed = Rc::new(SomeValue::Instance(SomeInstance::new(
            Some(classdef),
            false,
            Default::default(),
        )));
        *phi.annotation.borrow_mut() = Some(classed);
        resolve_types(&graph, &raw);
        assert_eq!(FunctionGraph::concretetype_of(&phi), ConcreteType::Signed);

        let (graph, phi) = build("classless_phi");
        let classless = Rc::new(SomeValue::Instance(SomeInstance::new(
            None,
            false,
            Default::default(),
        )));
        *phi.annotation.borrow_mut() = Some(classless);
        resolve_types(&graph, &raw);
        assert_eq!(FunctionGraph::concretetype_of(&phi), ConcreteType::GcRef);
    }

    #[test]
    fn backward_constraint_types_int_ops_as_signed() {
        let cases = [
            ("add", "int_backprop", "add"),
            ("bitxor", "bitxor_backprop", "bitxor"),
        ];
        for (name, graph_name, op) in cases {
            let mut graph = FunctionGraph::new(graph_name);
            let entry = graph.startblock;
            let lhs_var = graph
                .push_op_var(
                    entry,
                    OpKind::Input {
                        name: "lhs".to_string(),
                        ty: ValueType::Unknown,
                        class_root: None,
                    },
                    true,
                )
                .unwrap();
            let rhs_var = graph
                .push_op_var(
                    entry,
                    OpKind::Input {
                        name: "rhs".to_string(),
                        ty: ValueType::Unknown,
                        class_root: None,
                    },
                    true,
                )
                .unwrap();
            let result_var = graph
                .push_op_var(
                    entry,
                    OpKind::BinOp {
                        op: op.to_string(),
                        lhs: lhs_var.clone(),
                        rhs: rhs_var.clone(),
                        result_ty: ValueType::Int,
                    },
                    true,
                )
                .unwrap();
            graph.set_return(entry, Some(result_var.clone()));

            annotate::annotate(&graph);
            resolve_types(&graph, &HashSet::new());
            assert_eq!(
                FunctionGraph::concretetype_of(&lhs_var),
                ConcreteType::Signed,
                "case {name} lhs"
            );
            assert_eq!(
                FunctionGraph::concretetype_of(&rhs_var),
                ConcreteType::Signed,
                "case {name} rhs"
            );
            assert_eq!(
                FunctionGraph::concretetype_of(&result_var),
                ConcreteType::Signed,
                "case {name} result"
            );
        }
    }

    #[test]
    fn integer_repr_leaves_a_void_int_sub_operand() {
        // Void is Impossible/None. `IntegerRepr.convert_const` refuses
        // None; this pass must not repair the operand to Signed.
        let mut graph = FunctionGraph::new("int_sub_void_rhs");
        let entry = graph.startblock;
        let lhs = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&rhs, ConcreteType::Void);
        let result = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result.clone()));

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(FunctionGraph::concretetype_of(&lhs), ConcreteType::Signed);
        assert_eq!(FunctionGraph::concretetype_of(&rhs), ConcreteType::Void);
        assert_eq!(
            FunctionGraph::concretetype_of(&result),
            ConcreteType::Signed
        );
    }

    #[test]
    #[should_panic(expected = "not an integer")]
    fn convert_link_does_not_smash_signed_dest_with_void_const() {
        // `insert_link_conversions` `inputconst(r_a2, a1.value)`:
        // dest stays Signed; `IntegerRepr.convert_const` refuses None.
        let mut graph = FunctionGraph::new("signed_phi_void_const");
        let entry = graph.startblock;
        let (target, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        FunctionGraph::set_concretetype_of_inline(&phi, ConcreteType::Signed);
        let void_none = crate::flowspace::model::Constant::with_concretetype(
            crate::flowspace::model::ConstValue::None,
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
        );
        graph.blocks[entry.0].exits.push(Link::new_mixed(
            vec![LinkArg::Const(void_none)],
            target,
            None,
        ));
        convert_link(&graph, &graph.blocks[entry.0].exits[0]);
    }

    #[test]
    #[should_panic(expected = "not an integer")]
    fn convert_link_mixed_int_value_and_void_none_return() {
        // One predecessor returns an int Variable, another uses
        // `set_return(_, None)` (`Constant(None, Void)`). Annotation
        // unions Int ∪ Ref → Unknown; convert_link seeds Signed from
        // the int edge then `inputconst(signed_repr, None)` raises.
        let mut graph = FunctionGraph::new("mixed_int_void_return");
        let int_bb = graph.create_block();
        let void_bb = graph.create_block();
        let v = graph
            .push_op_var(int_bb, OpKind::ConstInt(1), true)
            .unwrap();
        graph.set_return(int_bb, Some(v));
        graph.set_return(void_bb, None);

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
    }

    fn repr_const(
        handle: &std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr>,
    ) -> ConstValue {
        ConstValue::Repr(crate::flowspace::model::ReprConst::new(
            std::sync::Arc::clone(handle),
        ))
    }

    fn branch_start_to_preds(
        graph: &mut FunctionGraph,
        pred1: crate::model::BlockId,
        pred2: crate::model::BlockId,
    ) {
        let cond = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "switch".into(),
                    ty: ValueType::Bool,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_branch(graph.startblock, cond, pred1, vec![], pred2, vec![]);
    }

    fn two_pred_const_phi(
        left: ConstValue,
        right: ConstValue,
    ) -> (FunctionGraph, crate::flowspace::model::Variable) {
        let mut graph = FunctionGraph::new("repr_phi");
        let pred1 = graph.create_block();
        let pred2 = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        branch_start_to_preds(&mut graph, pred1, pred2);
        graph.set_goto_mixed(
            pred1,
            join,
            vec![LinkArg::Const(crate::flowspace::model::Constant::new(left))],
        );
        graph.set_goto_mixed(
            pred2,
            join,
            vec![LinkArg::Const(crate::flowspace::model::Constant::new(
                right,
            ))],
        );
        graph.set_return(join, Some(phi.clone()));
        (graph, phi)
    }

    #[test]
    fn same_repr_const_phi_is_void() {
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let (graph, phi) = two_pred_const_phi(repr_const(&handle), repr_const(&handle));
        annotate::annotate(&graph);
        resolve_types_inner(&graph, &HashSet::new()).expect("same ReprConst is Void");
        assert_eq!(FunctionGraph::concretetype_of(&phi), ConcreteType::Void);
    }

    #[test]
    fn distinct_repr_const_phi_is_typer_error() {
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle_a: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let handle_b: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let (graph, _phi) = two_pred_const_phi(repr_const(&handle_a), repr_const(&handle_b));
        let err =
            resolve_types_inner(&graph, &HashSet::new()).expect_err("distinct ReprConst merge");
        let msg = err.to_string();
        assert!(
            msg.contains("MultipleFrozenPBCRepr"),
            "error should name MultipleFrozenPBCRepr, got {msg}"
        );
        assert!(
            msg.contains("getFrozenPBCRepr"),
            "error should cite getFrozenPBCRepr, got {msg}"
        );
    }

    #[test]
    fn repr_const_mixed_with_non_repr_is_typer_error() {
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let (graph, _phi) = two_pred_const_phi(repr_const(&handle), ConstValue::Int(1));
        let err =
            resolve_types_inner(&graph, &HashSet::new()).expect_err("Repr mixed with non-Repr");
        let msg = err.to_string();
        assert!(
            msg.contains("MultipleFrozenPBCRepr"),
            "error should name MultipleFrozenPBCRepr, got {msg}"
        );
        assert!(
            msg.contains("getFrozenPBCRepr"),
            "error should cite getFrozenPBCRepr, got {msg}"
        );
    }

    #[test]
    fn resolve_types_accepts_repr_const_and_same_singleton_variable() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let c = repr_const(&handle);
        let bk = Rc::new(Bookkeeper::new());
        let s_pbc = bk
            .immutablevalue(&c)
            .expect("immutablevalue of Repr is SomePBC");

        let mut graph = FunctionGraph::new("repr_const_and_var");
        let pred1 = graph.create_block();
        let pred2 = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();

        let src_var = graph.alloc_value_var();
        *src_var.annotation.borrow_mut() = Some(Rc::new(s_pbc));

        branch_start_to_preds(&mut graph, pred1, pred2);
        graph.set_goto_mixed(
            pred1,
            join,
            vec![LinkArg::Const(crate::flowspace::model::Constant::new(c))],
        );
        graph.set_goto(pred2, join, vec![src_var]);
        graph.set_return(join, Some(phi.clone()));

        annotate::annotate_with_bookkeeper(&graph, Some(&bk));
        resolve_types(&graph, &HashSet::new());
        assert_eq!(FunctionGraph::concretetype_of(&phi), ConcreteType::Void);
    }

    #[test]
    fn resolve_types_refuses_distinct_repr_consts_unannotated() {
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle_a: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let handle_b: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let (graph, _phi) = two_pred_const_phi(repr_const(&handle_a), repr_const(&handle_b));
        let err = resolve_types_inner(&graph, &HashSet::new())
            .expect_err("distinct ReprConst unannotated");
        let msg = err.to_string();
        assert!(
            msg.contains("MultipleFrozenPBCRepr"),
            "error should name MultipleFrozenPBCRepr, got {msg}"
        );
    }

    #[test]
    fn same_as_preserves_ref_classification() {
        let mut graph = FunctionGraph::new("same_as_ref");
        let entry = graph.startblock;
        let value_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "obj".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let alias_var = graph
            .push_op_var(
                entry,
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand: value_var.clone(),
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(alias_var.clone()));

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(
            FunctionGraph::concretetype_of(&value_var),
            ConcreteType::GcRef
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&alias_var),
            ConcreteType::GcRef
        );
    }

    #[test]
    fn same_as_propagates_signed_without_forcing_unknown_identity_to_int() {
        let mut graph = FunctionGraph::new("same_as_int");
        let entry = graph.startblock;
        let value_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let alias_var = graph
            .push_op_var(
                entry,
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand: value_var.clone(),
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(alias_var.clone()));

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(
            FunctionGraph::concretetype_of(&value_var),
            ConcreteType::Signed
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&alias_var),
            ConcreteType::Signed
        );
    }

    #[test]
    fn backward_constraint_propagates_signed_back_through_link_source() {
        let mut graph = FunctionGraph::new("link_backprop");
        let entry = graph.startblock;
        let src_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "x".to_string(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let one_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let (target, phi_args) = graph.create_block_with_arg_vars(1);
        let phi_var = phi_args[0].clone();
        graph.set_goto(entry, target, vec![src_var.clone()]);
        let result_var = graph
            .push_op_var(
                target,
                OpKind::BinOp {
                    op: "add".to_string(),
                    lhs: phi_var.clone(),
                    rhs: one_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(target, Some(result_var.clone()));

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(
            FunctionGraph::concretetype_of(&phi_var),
            ConcreteType::Signed
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&src_var),
            ConcreteType::Signed
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&result_var),
            ConcreteType::Signed
        );
    }

    #[test]
    fn unreached_returnblock_is_void_despite_disconnected_predecessor() {
        // startblock raises and never reaches returnblock.
        // A disconnected block (no predecessor, no `OpKind::Input`
        // inputargs) still exits to returnblock with an untyped
        // Variable. `RPythonTyper.specialize_more_blocks` walks
        // `annotator.annotated`; `FunctionGraph.iterblocks` walks
        // from `startblock` — neither sees the disconnected block.
        // The return var is `SomeImpossibleValue` → `impossible_repr`
        // (`rtyper.py` / `annmodel.py`): Void.
        let mut graph = FunctionGraph::new("dead_return_pred");
        let entry = graph.startblock;
        graph.set_raise(entry, "");
        let dead = graph.create_block();
        let ghost = graph.alloc_value_var();
        graph.set_return(dead, Some(ghost));
        let return_var = graph.blocks[graph.returnblock.0].inputargs[0].clone();

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(
            FunctionGraph::concretetype_of(&return_var),
            ConcreteType::Void
        );
    }

    #[test]
    fn unreached_disconnected_phi_is_impossible_void() {
        // `annotator.simplify` / `clear_unreachable_blocks` drop a
        // no-predecessor phi that is not an Input-shaped secondary
        // entry. Skip-arm keeps the block; `SomeImpossibleValue` →
        // `impossible_repr` types it Void.
        let mut graph = FunctionGraph::new("dead_phi");
        let entry = graph.startblock;
        let live = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        graph.set_return(entry, Some(live.clone()));
        let (_dead, args) = graph.create_block_with_arg_vars(1);
        let ghost = args[0].clone();

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(FunctionGraph::concretetype_of(&live), ConcreteType::Signed);
        assert_eq!(FunctionGraph::concretetype_of(&ghost), ConcreteType::Void);
    }

    #[test]
    fn unreachable_return_predecessor_does_not_panic() {
        // startblock loops and never reaches returnblock.
        // A disconnected block (no predecessor, no `OpKind::Input`
        // inputargs) still exits to returnblock with an untyped
        // Variable. `FunctionGraph.iterblocks` walks from
        // `startblock`; `RPythonTyper.specialize_more_blocks` walks
        // `annotator.annotated` — neither sees the disconnected block
        // or the unreached return. The return var is
        // `SomeImpossibleValue` → `impossible_repr` (`rtyper.py` /
        // `annmodel.py`): Void.
        let mut graph = FunctionGraph::new("loop_dead_return_pred");
        let entry = graph.startblock;
        graph.set_goto(entry, entry, vec![]);
        let dead = graph.create_block();
        let ghost = graph.alloc_value_var();
        graph.set_return(dead, Some(ghost));
        let return_var = graph.blocks[graph.returnblock.0].inputargs[0].clone();

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        let live = live_block_flags(&graph);
        assert!(live[entry.0]);
        assert!(!live[dead.0]);
        assert!(!live[graph.returnblock.0]);
        assert!(!live[graph.exceptblock.0]);
        assert_eq!(
            FunctionGraph::concretetype_of(&return_var),
            ConcreteType::Void
        );
    }

    #[test]
    fn secondary_entry_block_stays_live() {
        // `simplify.transform_dead_op_vars_in_blocks` treats every
        // graph start as a root. A no-predecessor block whose
        // inputargs are all defined by matching `OpKind::Input` ops
        // is pyre's stand-in (`secondary_entry_roots`) when one
        // `FunctionGraph` carries a second start (`generated::*` /
        // `eval_loop_jit`).
        let mut graph = FunctionGraph::new("secondary_entry");
        let entry = graph.startblock;
        let live_ret = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        graph.set_return(entry, Some(live_ret));
        let orphan = graph.create_block();
        let param = graph
            .push_op_var(
                orphan,
                OpKind::Input {
                    name: "captured".to_string(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(orphan, param.clone());

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert!(
            live_block_flags(&graph)[orphan.0],
            "Input-shaped secondary entry must stay live"
        );
        assert_eq!(FunctionGraph::concretetype_of(&param), ConcreteType::Signed);
    }

    #[test]
    fn input_op_without_secondary_entry_shape_is_not_live() {
        // `secondary_entry_roots` is not "contains `OpKind::Input`".
        // A disconnected block with an Input op stays dead when it
        // has a predecessor, or when an inputarg is not defined by
        // a matching Input.

        let mut with_pred = FunctionGraph::new("input_with_pred");
        with_pred.set_raise(with_pred.startblock, "");
        let pred = with_pred.create_block();
        let target = with_pred.create_block();
        let param = with_pred
            .push_op_var(
                target,
                OpKind::Input {
                    name: "x".to_string(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        with_pred.push_inputarg_var(target, param.clone());
        let ghost = with_pred.alloc_value_var();
        with_pred.set_goto(pred, target, vec![ghost]);

        annotate::annotate(&with_pred);
        resolve_types(&with_pred, &HashSet::new());
        let live = live_block_flags(&with_pred);
        assert!(!live[pred.0]);
        assert!(!live[target.0]);
        assert_eq!(FunctionGraph::concretetype_of(&param), ConcreteType::Void);

        let mut unmatched = FunctionGraph::new("inputarg_not_from_input");
        unmatched.set_raise(unmatched.startblock, "");
        let (orphan, args) = unmatched.create_block_with_arg_vars(1);
        let phi = args[0].clone();
        let extra = unmatched
            .push_op_var(
                orphan,
                OpKind::Input {
                    name: "extra".to_string(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();

        annotate::annotate(&unmatched);
        resolve_types(&unmatched, &HashSet::new());
        let live = live_block_flags(&unmatched);
        assert!(!live[orphan.0]);
        assert_eq!(FunctionGraph::concretetype_of(&phi), ConcreteType::Void);
        assert_eq!(FunctionGraph::concretetype_of(&extra), ConcreteType::Void);
    }

    #[test]
    #[should_panic(expected = "setconcretetype: no binding")]
    fn unknown_input_without_integer_constraint_is_typer_error() {
        // `rtyper.py setconcretetype` / `bindingrepr` has no GcRef default.
        let mut graph = FunctionGraph::new("unknown_backfill");
        let entry = graph.startblock;
        let value_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "obj".to_string(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(value_var));

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
    }

    #[test]
    fn resolves_raise_link_exception_pair() {
        let mut graph = FunctionGraph::new("raise_link");
        let entry = graph.startblock;
        let exc_block = graph.exceptblock;
        let (etype_var, evalue_var) = {
            let inputargs = &graph.block(exc_block).inputargs;
            (inputargs[0].clone(), inputargs[1].clone())
        };
        let last_exception_var = graph.alloc_value_var();
        let last_exc_value_var = graph.alloc_value_var();
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![
                Link::from_variables(
                    &graph,
                    vec![last_exception_var.clone(), last_exc_value_var.clone()],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(last_exception_var.clone())),
                    Some(LinkArg::Value(last_exc_value_var.clone())),
                ),
            ],
        );

        annotate::annotate(&graph);
        resolve_types(&graph, &HashSet::new());
        assert_eq!(
            FunctionGraph::concretetype_of(&last_exception_var),
            ConcreteType::Signed
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&last_exc_value_var),
            ConcreteType::GcRef
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&etype_var),
            ConcreteType::Signed
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&evalue_var),
            ConcreteType::GcRef
        );
    }
}
