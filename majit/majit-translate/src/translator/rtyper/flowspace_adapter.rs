//! `model::FunctionGraph` (pyre surface DSL) →
//! `flowspace::FunctionGraph` (RPython-orthodox) adapter.
//!
//! TODO(retire-this-adapter): this file has no upstream RPython
//! counterpart. RPython's pipeline has only one graph model — the
//! annotator builds its `FunctionGraph` (`rpython/flowspace/model.py`)
//! directly and the rtyper consumes it in place.  Pyre carries two
//! graph models in parallel: `crate::model::FunctionGraph` for the
//! surface DSL emitted by `parse → front → SemanticProgram`, and
//! `crate::flowspace::model::FunctionGraph` for the real
//! `translator/rtyper/` pipeline.  This adapter bridges the gap.
//! Retire when the surface DSL is replaced with `flowspace`-native
//! producers; until then the adapter remains the per-graph entry
//! into `RPythonAnnotator` / `RPythonTyper::specialize`.
//!
//! ## Why not `PyGraph`?
//!
//! `PyGraph` (`flowspace::pygraph::PyGraph`) wraps a `FunctionGraph`
//! with `GraphFunc` / `HostCode` / `Signature` / `defaults` —
//! Python-runtime function metadata that pyre's surface DSL does not
//! produce (`parse → front → SemanticProgram` operates on Rust source,
//! not CPython callables). `RPythonTyper::specialize`
//! (`rtyper.rs`) does NOT consume `PyGraph` directly — it iterates
//! `RPythonAnnotator.annotated` / `all_blocks`, which
//! `specialize_legacy_graph` will populate with the
//! [`FlowspaceAdapterOutput`] this adapter returns. Skipping the PyGraph
//! wrapping avoids fabricating fake `GraphFunc` / `HostCode` instances.
//!
//! ## Layout
//!
//! The adapter performs three responsibilities, all line-by-line at
//! `function_graph_to_flowspace`:
//!
//! 1. **Annotation lift** — clone each legacy `Variable.annotation`
//!    cell (`Rc<RefCell<Option<Rc<SomeValue>>>>`) onto a
//!    freshly-allocated `flowspace::Variable`. Variable identity is
//!    block-local per `flowspace/model.py:checkgraph`; the adapter
//!    keeps a `legacy Variable → typed Variable` representative map
//!    (keyed on Variable object identity) for post-specialize readback.
//! 2. **Per-OpKind translation** — `translate_op` maps each pyre
//!    `model::SpaceOperation` to a `flowspace::SpaceOperation` over
//!    `Hlvalue` operands.  Pre-rtyper variants (`Input`, `ConstInt`,
//!    `ConstFloat`, `BinOp`, `Call`, `FieldRead`, `ArrayRead`, ...)
//!    have explicit arms; post-rtyper jtransform variants are
//!    classified by [`post_rtyper_jtransform_variant_name`] and
//!    fail-loud with a stage-mismatch message.
//! 3. **Block topology** — wires `flowspace::Block` per legacy
//!    `Block`, translates `exits` / `exitcase` / `exitswitch`,
//!    designates `startblock` / `returnblock` / `exceptblock`, and
//!    assembles a `flowspace::FunctionGraph`.  `getreturnvar`
//!    (`rtyper.rs`) is non-degenerate because the
//!    returnblock's inputarg is materialised as the canonical
//!    flowspace return `Variable`.
//!
//! [`crate::translator::rtyper::cutover::specialize_legacy_graph_with_registry_returning_value_to_var`]
//! drives this adapter, runs `RPythonTyper::specialize`, and returns
//! the legacy-`Variable`-keyed typed-`Variable` map + the per-`Variable`
//! `Constant.concretetype` `LowLevelType` table that consumers project to
//! `ConcreteType` on demand.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::{
    self as flowspace_model, Block as FlowspaceBlock, BlockRef, ConstValue, Constant,
    FunctionGraph as FlowspaceGraph, HOST_ENV, Hlvalue, HostObject, Link as FlowspaceLink,
    SpaceOperation as FlowspaceOp, Variable, c_last_exception,
};
use crate::model::{BlockId, ExitCase, ExitSwitch, FunctionGraph, LinkArg, OpKind, SpaceOperation};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::llannotation::lltype_to_annotation;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

/// Map from a legacy graph `Variable` to the representative typed
/// `flowspace::Variable` the adapter created for readback.
///
/// This is not the graph's identity model. RPython `checkgraph` requires
/// block inputargs and operation results to be defined in exactly one
/// block, so [`function_graph_to_flowspace`] uses block-local Variables
/// while translating the actual graph. The map is keyed by the **legacy
/// graph Variable's object identity** (RPython keys cross-graph dicts on
/// Variable objects, never on an integer slot): `legacy_var -> typed_var`
/// lets consumers read `typed_var.concretetype` / `.annotation` back onto
/// the legacy Variable they already hold, with no slot-index round-trip.
pub type LegacyToTyped = HashMap<Variable, Variable>;
/// Every block-local flowspace Variable derived from one legacy MIR
/// Variable. The legacy MIR can reuse one Variable identity as an op result
/// and as inputargs of several successor blocks; orthodox RPython flowspace
/// requires each block inputarg to be a distinct definition. `LegacyToTyped`
/// retains the preferred projection representative, while this multimap lets
/// the cutover replace an untyped representative removed by simplify with a
/// typed block-local peer after rtyping.
pub type LegacyToTypedCandidates = HashMap<Variable, Vec<Variable>>;

pub use crate::codewriter::annotation_state::valuetype_to_someshell;

/// `__getslice_*` markers may carry an item-kind suffix as a second segment.
/// The rtyper expands the marker to `getslice` and learns the element from
/// the annotated list, so the suffix is ignored here.
fn is_getslice_marker(segments: &[String], name: &str) -> bool {
    segments.first().map(String::as_str) == Some(name) && segments.len() <= 2
}

/// Allocate a fresh `flowspace::Variable` and attach the projected
/// `SomeValue` shell to its `annotation` slot.
///
/// The legacy `Variable.id` does NOT carry over to the fresh
/// Variable's id — `Variable::new` allocates a fresh process-wide
/// identity (`flowspace/model.rs`). Identity correspondence is
/// preserved out-of-band by [`LegacyToTyped`].
/// The high-level operations one `ll_vec_*` helper call stands for:
/// `newrustvec(kind[, hint])`, `len`, `getitem`, `setitem`, or a
/// `getattr` + `simple_call` of the `append` / `reverse` / `free` / `items` /
/// `extend_from_slice` / `extend` method.
fn rust_vec_helper_ops(
    op: majit_ir::rvec::VecOp,
    kind: majit_ir::rvec::VecItemKind,
    arg_hls: Vec<Hlvalue>,
    result: Hlvalue,
) -> Result<Vec<FlowspaceOp>, TyperError> {
    use majit_ir::rvec::VecOp;
    let path = majit_ir::rvec::vec_helper_path(op, kind);
    let arity = match op {
        VecOp::NewEmpty => 0,
        VecOp::NewHint | VecOp::Length | VecOp::Reverse | VecOp::Free | VecOp::Items => 1,
        VecOp::GetItem | VecOp::Append | VecOp::AllocAndSet | VecOp::Extend => 2,
        VecOp::SetItem | VecOp::ExtendFromSlice => 3,
    };
    if arg_hls.len() != arity {
        return Err(TyperError::message(format!(
            "{path} takes {arity} args, got {}",
            arg_hls.len()
        )));
    }
    let (opname, method) = match op {
        VecOp::NewEmpty | VecOp::NewHint | VecOp::AllocAndSet => {
            let kind_letter = Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                &kind.kind_char().to_string(),
            )));
            let mut args = Vec::with_capacity(arg_hls.len() + 1);
            args.push(kind_letter);
            args.extend(arg_hls);
            return Ok(vec![FlowspaceOp::new("newrustvec", args, result)]);
        }
        VecOp::Length => ("len", None),
        VecOp::GetItem => ("getitem", None),
        VecOp::SetItem => ("setitem", None),
        VecOp::Append => ("simple_call", Some("append")),
        VecOp::Reverse => ("simple_call", Some("reverse")),
        VecOp::Free => ("simple_call", Some("free")),
        VecOp::Items => ("simple_call", Some("items")),
        VecOp::ExtendFromSlice => ("simple_call", Some("extend_from_slice")),
        VecOp::Extend => ("simple_call", Some("extend")),
    };
    let Some(method) = method else {
        return Ok(vec![FlowspaceOp::new(opname, arg_hls, result)]);
    };
    let mut args = arg_hls.into_iter();
    let receiver = args.next().expect("arity checked");
    let bound_method = Hlvalue::Variable(Variable::new());
    let mut call_args = vec![bound_method.clone()];
    call_args.extend(args);
    Ok(vec![
        FlowspaceOp::new(
            "getattr",
            vec![
                receiver,
                Hlvalue::Constant(Constant::new(ConstValue::byte_str(method))),
            ],
            bound_method,
        ),
        FlowspaceOp::new(opname, call_args, result),
    ])
}

fn seed_variable(legacy_var: &Variable) -> Variable {
    let var = Variable::new();
    // Copy the precise per-`Variable.annotation` `SomeValue` shell
    // from the legacy Variable onto the freshly minted one,
    // matching upstream `_setbinding(v, s_value)` semantics
    // (`rpython/annotator/annrpython.py`).
    //
    // Source: tests that hand-seed annotations call
    // `legacy_annotator::setbinding(&legacy_var, ty)` before
    // reaching here;
    // the production `addpendingblock` flowin path leaves
    // `legacy_var.annotation` empty so the fresh Variable
    // starts unannotated and flowin populates it via `setbinding`.
    // The dual-gate baseline that calls `legacy_annotator::annotate`
    // runs AFTER specialize completes (cutover.rs:dual_gate_check /
    // dual_gate_check_with_registry baseline section), so the wider
    // legacy lift never reaches this site.
    //
    // An empty `legacy_var.annotation` slot (unpopulated or
    // `ValueType::Unknown`) leaves the fresh `Variable.annotation`
    // empty — the rtyper then fails at `bindingrepr` with `KeyError:
    // no binding for arg` on first touch, surfacing the producer-
    // side gap rather than silently bridging to `GcRef` via a
    // fabricated `SomeInstance(None)` shell.
    if let Some(s) = legacy_var.annotation.borrow().as_ref() {
        *var.annotation.borrow_mut() = Some(s.clone());
    }
    var
}

/// Build the `legacy Variable → typed flowspace::Variable` map for every
/// value reachable from `legacy.blocks`.
///
/// Three reference-site classes seed the map:
///
/// 1. **Definitions** — `block.inputargs` (RPython-orthodox phi nodes)
///    and `op.result`. Every operand referenced via `op.args` /
///    `link.args` / `exitswitch` resolves to a definition site in the
///    same graph (legacy `FunctionGraph` is mostly SSA), so seeding
///    definitions covers most of the value set.
///
/// 2. **Link-side sentinels** — `link.args` / `link.last_exception` /
///    `link.last_exc_value`. RPython `flowspace/model.py` and
///    pyre's front (`front::mir`) allow a `Link.args` slot
///    to carry a *fresh* prevblock-side `Variable` whose only "defining
///    site" is the link itself — the value flows into the target
///    block's inputarg via this synthetic Variable. The adapter must
///    seed a `Variable` for each such slot so the link
///    translation can resolve the operand without tripping the
///    "undefined operand" invariant in `lookup_operand`.
///
/// 3. **Exitswitch values** — `block.exitswitch = Some(ExitSwitch::Value(vid))`
///    sometimes references a slot defined in a successor block's
///    inputarg context (rarely but legitimately in legacy graphs).
///    Seeded for the same reason.
///
/// Each legacy Variable is seeded exactly once via
/// `entry().or_insert_with`, preserving operand identity across
/// multiple readers — the op translator looks up the same Variable
/// instance for every reader of a given operand, matching upstream
/// Python's reference semantics where `op.args[i]` and `op2.args[j]`
/// may be the same `Variable` object.
///
/// **Restricted to the adapter / its tests.**  `function_graph_to_flowspace`
/// builds a *block-local* `legacy Variable -> typed Variable` map per
/// block in the topology assembly pass, mirroring `RPython` `checkgraph`'s
/// per-block Variable invariant
/// (`rpython/flowspace/model.py`: a Variable must be defined in
/// exactly one block).  Using this whole-graph helper as the source of
/// truth for the adapter's main path would violate that invariant by
/// reusing a single `Variable` across blocks.  The helper stays in the
/// crate solely to back the adapter's regression tests
/// (`build_value_to_variable_map_*`); production cutover code must use
/// the per-block maps owned by `function_graph_to_flowspace`.
#[cfg(test)]
pub(crate) fn build_value_to_variable_map(legacy: &FunctionGraph) -> LegacyToTyped {
    // Callers that hand-seed test fixtures must write directly to
    // `legacy.variable(vid).annotation` via
    // `legacy_annotator::setbinding(&var, ty)` before invoking this so
    // downstream `seed_variable` reads through the orthodox
    // `Variable.annotation` carrier.
    let mut map: LegacyToTyped = HashMap::new();
    for block in &legacy.blocks {
        // Class 1a — block-inputarg definitions.
        for var in &block.inputargs {
            map.entry(var.clone()).or_insert_with(|| seed_variable(var));
        }

        // Per-block name → inputarg-Variable lookup for `OpKind::Input`
        // rebind-aliasing. Pyre's surface front (`front::mir`) emits a
        // *leading* `Input{name, ty}` op for each named parameter whose
        // `op.result` matches a `block.inputargs` entry, and may emit
        // *additional* `Input{same name}` ops with fresh `op.result`
        // slots for body-side rebinds. RPython's flowspace has no
        // such Input op machinery — the parameter Variable IS the
        // inputarg. Without aliasing the rebind result to the
        // canonical inputarg Variable, `setup_block_entry` writes
        // `concretetype` only on the inputarg's Rc<RefCell> and the
        // body's BinOp lookup hits a fresh Variable with `None`
        // concretetype, tripping `genop`'s "wrong level!" assertion.
        let mut name_to_inputarg_var: HashMap<&str, Variable> = HashMap::new();
        for op in &block.operations {
            if let OpKind::Input { name, .. } = &op.kind
                && let Some(result_var) = op.result.as_ref()
                && block.inputargs.contains(result_var)
                && let Some(var) = map.get(result_var)
            {
                name_to_inputarg_var
                    .entry(name.as_str())
                    .or_insert_with(|| var.clone());
            }
        }

        // Class 1b — op-result definitions, with Input rebind aliasing.
        //
        // `OpKind::Abort` (pyre-only front-end marker for unsupported
        // expression forms) is intentionally NOT seeded into
        // `value_to_var`.  `translate_op`'s Abort arm emits no
        // flowspace op (`OpKind::Abort { .. }
        // => Ok(Vec::new())`), so seeding the result_var here would
        // hand consumer ops a `Hlvalue::Variable` that never gets
        // *defined* by any emitted flowspace op — `checkgraph`
        // (`flowspace/model.rs::checkgraph`) then panics with
        // "variable used before definition" at the consumer's arg
        // slot, NOT at the missing-operand site.  Skipping the seed
        // here forces the first consumer's `lookup_operand` to fail
        // with "undefined operand" instead (`is_known_unported`
        // already matches that substring; the dual gate Skip-classifies
        // the graph cleanly at the producer-adjacent site).
        for op in &block.operations {
            let Some(result_var) = op.result.as_ref() else {
                continue;
            };
            if matches!(op.kind, OpKind::Abort { .. }) {
                continue;
            }
            if map.contains_key(result_var) {
                continue;
            }
            let var = if let OpKind::Input { name, .. } = &op.kind {
                name_to_inputarg_var
                    .get(name.as_str())
                    .cloned()
                    .unwrap_or_else(|| seed_variable(result_var))
            } else {
                seed_variable(result_var)
            };
            map.insert(result_var.clone(), var);
        }
        // Class 3 — exitswitch-referenced values.
        if let Some(crate::model::ExitSwitch::Value(var)) = &block.exitswitch {
            map.entry(var.clone()).or_insert_with(|| seed_variable(var));
        }
        // Class 2 — link-side sentinels.
        for link in &block.exits {
            for arg in &link.args {
                if let Some(var) = arg.as_variable() {
                    map.entry(var.clone()).or_insert_with(|| seed_variable(var));
                }
            }
            if let Some(arg) = link.last_exception.as_ref()
                && let Some(var) = arg.as_variable()
            {
                map.entry(var.clone()).or_insert_with(|| seed_variable(var));
            }
            if let Some(arg) = link.last_exc_value.as_ref()
                && let Some(var) = arg.as_variable()
            {
                map.entry(var.clone()).or_insert_with(|| seed_variable(var));
            }
        }
    }
    map
}

/// result-`Variable` → `Hlvalue` map combining the [`LegacyToTyped`]
/// map with constant-inlining of `OpKind::ConstInt` / `ConstFloat`
/// define-ops.
///
/// RPython's flowspace inlines constants natively as `Hlvalue::Constant`
/// in `op.args` (`flowspace/operation.py` `simple_call(target,
/// *args)` — `target` and each `arg` is either a `Variable` or
/// `Constant`). Pyre's legacy graph splits constants into define-ops
/// (`OpKind::ConstInt(n)` produces a fresh result consumed
/// elsewhere). The adapter must recombine: every result defined as a
/// const becomes a `Hlvalue::Constant`; every other defined result
/// remains a `Hlvalue::Variable` from the variable map.
///
/// Constants are wrapped with their low-level concretetype attached,
/// matching RPython's `Constant.concretetype` shape.
///
/// Test-only: production const-inlining is carried by the `value_map`
/// / `value_to_var` Variable-keyed path; this result-Variable-keyed
/// builder survives solely as the mirror the
/// `build_value_to_hlvalue_map_inlines_const_defines` fixture diffs
/// against, so it is gated `#[cfg(test)]`.
#[cfg(test)]
pub fn build_value_to_hlvalue_map(
    legacy: &FunctionGraph,
    value_to_var: &LegacyToTyped,
) -> HashMap<crate::flowspace::model::Variable, Hlvalue> {
    let mut map: HashMap<crate::flowspace::model::Variable, Hlvalue> = value_to_var
        .iter()
        .map(|(legacy_var, var)| (legacy_var.clone(), Hlvalue::Variable(var.clone())))
        .collect();

    for block in &legacy.blocks {
        for op in &block.operations {
            let Some(result) = op.result.clone() else {
                continue;
            };
            match &op.kind {
                OpKind::ConstInt(n) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Int(*n),
                            LowLevelType::Signed,
                        )),
                    );
                }
                OpKind::ConstFnAddr { value, .. } => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Int(*value),
                            LowLevelType::Signed,
                        )),
                    );
                }
                OpKind::ConstUInt(n) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Int(*n as i64),
                            LowLevelType::Unsigned,
                        )),
                    );
                }
                OpKind::ConstBool(b) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Bool(*b),
                            LowLevelType::Bool,
                        )),
                    );
                }
                OpKind::ConstFloat(bits) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Float(*bits),
                            LowLevelType::Float,
                        )),
                    );
                }
                OpKind::ConstRefNull => {
                    map.insert(result, Hlvalue::Constant(const_ref_gcref_constant(None)));
                }
                OpKind::ConstNone => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::None,
                            LowLevelType::Void,
                        )),
                    );
                }
                OpKind::ConstRefAddr(addr) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(const_ref_gcref_constant(Some(*addr))),
                    );
                }
                _ => {}
            }
        }
    }
    map
}

/// Fold a `ConstRefAddr` / `ConstRefNull` opkind to its constant.
///
/// Both opkinds carry a `PyObjectRef` (object-space singleton address
/// from `static_addrs.refs`, or the `PY_NULL` null reference) — a GC
/// reference in the walker's kind system.  The constant is a
/// `GCREF`-typed `_ptr` (`Ptr(GcOpaque("GCREF"))`, the lltype of
/// erased GC references) carrying the raw address as
/// `_ptr_obj::IntCast`, so `SomePtr(GCREF)` annotation and `PtrRepr`
/// rtyping give the slot its GcRef kind.  Folding to
/// `LLAddress`/`Address` instead made every graph returning such a
/// constant diverge at the dual-gate (`legacy=GcRef, real=Signed`):
/// `llmemory.Address` is int-kinded.  `cast_int_to_ptr` is not usable
/// here — it asserts odd (tagged) integers (lltype.py) and
/// host addresses are even — so the `_ptr` is built directly.
fn const_ref_gcref_constant(addr: Option<i64>) -> Constant {
    use crate::translator::rtyper::lltypesystem::lltype::{_ptr, _ptr_obj, GCREF, LowLevelType};
    let LowLevelType::Ptr(gcref_t) = GCREF.clone() else {
        panic!("GCREF must be a Ptr lowleveltype");
    };
    let p = match addr {
        None | Some(0) => _ptr::new(*gcref_t, Ok(None)),
        Some(a) => _ptr::new_with_solid(*gcref_t, Ok(Some(_ptr_obj::IntCast(a))), true),
    };
    Constant::with_concretetype(ConstValue::LLPtr(Box::new(p)), GCREF.clone())
}

/// A Void-typed operand is spelled as a Void `Constant`, never as a
/// `Variable`: `do_rename` inside `Transformer.optimize_block`
/// (jtransform.py) renames every Void var to
/// `Constant(None, lltype.Void)` before it can be read, and
/// `pairtype(Repr, VoidRepr).convert_from_to` (rmodel.py)
/// converts into Void as `inputconst(lltype.Void, None)`.
///
/// PRE-EXISTING-ADAPTATION: tolerate legacy graphs with undefined Void
/// operands. MIR's erased-call results and fieldless-enum projections now
/// emit actual ConstNone definitions: this fallback must not replace those
/// producers, because it runs after framestate and loses their provenance.
/// Remove it once the other input-graph producers and their fixtures enforce
/// the upstream defined-operand invariant, not by treating Unknown as Void.
///
/// Only Void is treated this way. A missing operand of any other kind
/// still fails loud: it names a value some op was supposed to define.
fn void_constant_for(v: &Variable) -> Option<Hlvalue> {
    matches!(
        v.concretetype.borrow().as_ref(),
        Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void)
    )
    .then(|| {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
        ))
    })
}

/// Look up the `Hlvalue` for an operand `Variable`. Surfaces a
/// fail-loud `TyperError` when the operand is undefined (every
/// referenced operand `Variable` must have been seeded by
/// [`build_value_to_variable_map`] or shadowed by
/// `build_value_to_hlvalue_map`'s const inlining).
///
/// The error message embeds the enclosing `SpaceOperation` (variant
/// name + result `Variable`) and the role of the failing argument (e.g.
/// `"lhs"`, `"rhs"`, `"base"`, `"index"`, `"value"`, `"operand"`,
/// `"args[i]"`, `"result"`) so per-graph diagnosis can locate the
/// broken op without re-traversing the graph. The required substring
/// `"undefined operand"` is preserved verbatim so the dual
/// gate's `is_known_unported` predicate (`cutover.rs`) keeps
/// matching this category.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn lookup_operand(
    value_map: &HashMap<Variable, Hlvalue>,
    operand: &Variable,
    op: &SpaceOperation,
    arg_role: &str,
) -> Result<Hlvalue, TyperError> {
    value_map
        .get(operand)
        .cloned()
        .or_else(|| void_constant_for(operand))
        .ok_or_else(|| {
            let result_label = match op.result.as_ref() {
                Some(var) => format!("Some({var:?})"),
                None => "None".to_string(),
            };
            TyperError::message(format!(
                "translate_op: undefined operand {operand:?} as {arg_role} of {opkind} \
             (result {result_label}) — adapter invariant broken (every referenced \
             operand must be defined as a block inputarg or op result)",
                opkind = opkind_variant_name(&op.kind),
            ))
        })
}

/// RPython `SpaceOperation.args` is a mixed `Variable | Constant` list
/// (`flowspace/model.py`).  `LinkArg::Const` is already the flowspace
/// Constant — lift it as `Hlvalue::Constant` with no SSA lookup.
fn lookup_link_arg(
    value_map: &HashMap<Variable, Hlvalue>,
    operand: &LinkArg,
    op: &SpaceOperation,
    arg_role: &str,
) -> Result<Hlvalue, TyperError> {
    match operand {
        LinkArg::Value(var) => lookup_operand(value_map, var, op, arg_role),
        LinkArg::Const(c) => Ok(Hlvalue::Constant(c.clone())),
    }
}

/// Resolve the `Hlvalue` result slot for a legacy op. When the op has
/// no result (`Option::None`), allocate a fresh anonymous Variable per
/// RPython convention (every `SpaceOperation.result` slot is non-None
/// upstream — model.py:432-438; void-result ops use a throwaway
/// `Variable()`).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn resolve_result_hlvalue(
    op: &SpaceOperation,
    value_map: &HashMap<Variable, Hlvalue>,
) -> Result<Hlvalue, TyperError> {
    match op.result.as_ref() {
        Some(var) => lookup_operand(value_map, var, op, "result"),
        None => Ok(Hlvalue::Variable(Variable::new())),
    }
}

/// Reconstruct the item `LowLevelType` from an [`OpKind::RawLoad`] /
/// [`OpKind::RawStore`] descr.  `ValueType` collapses integer widths;
/// signedness selects `Signed` vs `Unsigned`.  A Rust `u8` is
/// `rffi.UCHAR` (an unsigned `lltype.Number`), not `lltype.Char`
/// (`SomeChar`); width stays on the original `OpKind` (`itemsize`)
/// for `jtransform` `rewrite_op_raw_load` / `rewrite_op_raw_store`.
/// `_LLOpEntry.compute_result_annotation` then binds
/// `lltype_to_annotation` of this type.
fn raw_item_lltype(
    item_ty: &crate::model::ValueType,
    _itemsize: usize,
    is_item_signed: bool,
) -> Result<LowLevelType, TyperError> {
    use crate::model::ValueType;
    Ok(match item_ty {
        ValueType::Float => LowLevelType::Float,
        ValueType::SingleFloat => LowLevelType::SingleFloat,
        ValueType::Bool => LowLevelType::Bool,
        ValueType::Int128 => LowLevelType::SignedLongLongLong,
        ValueType::UInt128 => LowLevelType::UnsignedLongLongLong,
        ValueType::Unsigned => LowLevelType::Unsigned,
        ValueType::Int if is_item_signed => LowLevelType::Signed,
        ValueType::Int => LowLevelType::Unsigned,
        ValueType::Ref(_) => crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
        ValueType::Void => LowLevelType::Void,
        ValueType::Str | ValueType::StringBuilder | ValueType::State | ValueType::Unknown => {
            return Err(TyperError::message(format!(
                "translate_op: RawLoad/RawStore item_ty {item_ty:?} has no \
                 lltype_to_annotation mapping from the descr"
            )));
        }
    })
}

/// `_LLOpEntry.compute_result_annotation` writes
/// `lltype_to_annotation(RESULTTYPE)` onto the result Variable.  The
/// adapter does that here because the FlowspaceOp cannot carry the
/// OpKind descr as a TYPE constant (`Bookkeeper.immutablevalue`
/// rejects `ConstValue::LowLevelType`).
fn seed_raw_item_annotation(result: &Hlvalue, lltype: LowLevelType) {
    if let Hlvalue::Variable(v) = result {
        *v.annotation.borrow_mut() = Some(Rc::new(lltype_to_annotation(lltype)));
    }
}

/// Map a pyre-frontend unary op name (`front::mir::unary_op_label`)
/// onto the RPython flowspace operator name
/// (`rpython/flowspace/operation.py`).
///
/// `neg` and `bool` pass through (registered upstream as
/// `add_operator('neg', 1, ..)` at line 466 and `add_operator('bool',
/// 1, ..)` at line 467).
///
/// Typed numeric / ptr / Unsigned casts have no `OpKind::UnaryOp`
/// route — the frontend routes typed casts through
/// `simple_call(<host_callable>, v)` per
/// upstream `__builtin__.int/float/bool` /
/// `lltype.cast_ptr_to_int` / `lltype.cast_int_to_ptr` /
/// `rarithmetic.intmask` / `rarithmetic.r_uint`.  Only `same_as`
/// remains on the `OpKind::UnaryOp` route, emitted by the
/// identity / source-type-unknown cast fallback and
/// dispatched by `RPythonTyper::translate_operation` to
/// `rbuiltin::rtype_same_as` (verbatim port of `rtyper.py`).
/// `same_as` is also generated by `unsimplify::split_block` Void-
/// variable recreation and the backendopt pipeline.
///
/// `not` and `deref` are the only fail-loud arms: pyre's frontend
/// eliminates both at the source (`UnOp::Not` desugar / `Deref`
/// pass-through).  Reaching either arm means a synthetic graph
/// injected the op directly.  RPython distinguishes logical `not`
/// (UNARY_NOT, lowered as `bool(operand)` + branch —
/// `flowcontext.py`) from bitwise `invert` (UNARY_INVERT —
/// `flowcontext.py` → `op.invert`); without static type info,
/// the adapter cannot discriminate, so fail-loud is the only safe
/// choice.
fn normalize_unary_op_name(source_name: &str) -> Result<String, TyperError> {
    match source_name {
        "neg" => Ok("neg".to_string()),
        // RPython `bool` is registered as a unary op at
        // `operation.py add_operator('bool', 1, ..)` and emitted
        // by `flowcontext.py UNARY_NOT` /
        // `:766-777 JUMP_IF_*_OR_POP` as the discriminator before a
        // `guessbool` fork.  Pyre emits it from
        // `model::FunctionGraph::set_branch`, which wraps every branch
        // condition in `OpKind::UnaryOp { op: "bool", .. }` before
        // installing it as the exitswitch.  Pass through unchanged.
        "bool" => Ok("bool".to_string()),
        // `invert` — PyPy `add_operator('invert', 1, .., pure=True)` at
        // `operation.py:474`, emitted by `flowcontext.py:188-191
        // UNARY_INVERT` and dispatched through
        // `RPythonTyper::translate_op`'s `"invert"` arm
        // (`rtyper.rs`) into `IntegerRepr::rtype_invert`
        // (`rint.py` → `rint.rs`). Pyre's frontend
        // emits `OpKind::UnaryOp { op: "invert", .. }` directly for the
        // integer bitwise-complement form (the case Rust's `!42_i64`
        // denotes).  Without this arm the
        // literal-int parity path Skip-classifies in the real rtyper.
        "invert" => Ok("invert".to_string()),
        // `same_as` — RPython's internal rtyper renaming op
        // (`rtyper.py:478-481`).  Defensively kept on the unary-op
        // dispatch path so the rtyper can re-enter `translate_operation`
        // on graphs that carry `same_as` from any source: identity /
        // source-type-unknown cast lowering in the frontend,
        // `unsimplify::split_block`'s
        // Void-variable recreation (`unsimplify.rs`), and the
        // backendopt pipeline's block-prefix `same_as` insertion
        // (`backendopt/constfold.rs`, `backendopt/all.rs`,
        // `removenoops.rs`, `storesink.rs`).  All other typed
        // `(source, target)` casts have no unary-op route — they
        // route through `simple_call(<host_callable>, v)` per upstream
        // `__builtin__.int/float/bool` / `lltype.cast_*` /
        // `rarithmetic.intmask` / `rarithmetic.r_uint`.
        "same_as" => Ok("same_as".to_string()),
        // `str` — RPython `add_operator('str', 1, ..)` (`operation.py`),
        // dispatched at `rtyper.rs "str" => rtype_str` into the per-repr
        // `ll_str` lowering and annotated `SomeString` (`unaryop.rs str =>
        // SomeString()`).  `front::mir`'s `format!`-chain expansion
        // (`collapse_fmt_chains`) renders each `{}` placeholder argument
        // with `OpKind::UnaryOp { op: "str", .. }` — the orthodox
        // string-build lowering (`str(arg)` ++ `ll_strconcat`) in place of
        // the graph-less `fmt::rt::Argument::new_display` chain.
        "str" => Ok("str".to_string()),
        // `abs` — `operation.py add_operator('abs', 1, dispatch=1,
        // pyfunc=abs, pure=True)`, dispatched at `rtyper.rs "abs"` into
        // `FloatRepr.rtype_abs` (`float_abs`) / `IntegerRepr.rtype_abs`.
        // The front lowers `f64::abs` to this unary op.
        "abs" => Ok("abs".to_string()),
        // `cast_float_to_int` is what `rfloat.py FloatRepr.rtype_int`
        // emits for `int(x)` on a float, so the flowspace op it stands for
        // is `int` (`operation.py add_operator('int', 1, dispatch=1,
        // pyfunc=int)`).  The front emits it for
        // `f64::to_int_unchecked::<i64>` and the float-to-int scalar cast.
        "cast_float_to_int" => Ok("int".to_string()),
        other => Err(TyperError::missing_rtype_operation(format!(
            "normalize_unary_op_name: pyre UnaryOp `{other}` has no \
             flowspace counterpart (operation.py registers \
             `pos` / `neg` / `invert` / `bool` / `abs` and the ported \
             `str` as unary ops, and `cast_float_to_int` stands for `int`; \
             `same_as` is rtyper's internal renaming op per \
             rtyper.py:478-481; all 13 typed cast names retired \
             across Slices A.3 / B.1 / A.4a / A.4b / A.4c — frontend \
             routes typed casts through \
             `simple_call(<host_callable>, v)` per upstream \
             `__builtin__.int/float/bool` / `lltype.cast_*` / \
             `rarithmetic.intmask` / `rarithmetic.r_uint`).  Frontend \
             must distinguish bitwise `invert` from logical `not` \
             and remove `deref` before reaching the rtyper."
        ))),
    }
}

/// Map a pyre-frontend binary op name (`front::mir::binop_label`)
/// onto the RPython flowspace operator name
/// (`rpython/flowspace/operation.py add_operator(...)`).
///
/// Rust-side identifiers (`bitand`, `bitor`, `bitxor`, `add_assign`,
/// ...) become the trailing-underscore / `inplace_*` forms RPython
/// registers and `RPythonTyper::translate_op_with_map`
/// (`rtyper.rs`) dispatches on.  Names already matching
/// RPython (`add`, `sub`, `mul`, `mod`, `lshift`, `rshift`, `lt`, ...)
/// pass through unchanged.
///
/// A bare `and` / `or` arriving here is always the *bitwise* operator:
/// `front::mir::binop_label` maps Charon `BitAnd` / `BitOr` to
/// `"and"` / `"or"` (matching the trace pipeline's blackhole handler
/// vocabulary), and rustc lowers the short-circuit `&&` / `||` forms
/// into MIR branch forks before Charon ever sees them — Python's
/// short-circuit `and`/`or` are likewise control flow upstream
/// (`operation.py:475-510` registers no such binary operator), so the
/// keyword-suffixed `and_` / `or_` names below are unambiguously the
/// bitwise registrations.
fn normalize_binop_name(source_name: &str) -> Result<String, TyperError> {
    let normalized = match source_name {
        "and" | "bitand" => "and_",
        "or" | "bitor" => "or_",
        "bitxor" => "xor",
        "add_assign" => "inplace_add",
        "sub_assign" => "inplace_sub",
        "mul_assign" => "inplace_mul",
        "div_assign" => "inplace_div",
        "mod_assign" => "inplace_mod",
        "bitand_assign" => "inplace_and",
        "bitor_assign" => "inplace_or",
        "bitxor_assign" => "inplace_xor",
        "lshift_assign" => "inplace_lshift",
        "rshift_assign" => "inplace_rshift",
        other => other,
    };
    Ok(normalized.to_string())
}

/// Translate a single legacy `model::SpaceOperation` into zero or more
/// `flowspace::SpaceOperation`s.
///
/// Returns `Ok(Vec::new())` when the op is **fully consumed by other
/// adapter infrastructure** — `OpKind::Input` (handled by block
/// topology assembly, where the result `Variable` becomes a
/// `block.inputargs` entry) and `OpKind::ConstInt` / `ConstFloat`
/// (inlined by the `value_map` const-folding path at every consuming
/// op's args site, mirrored in tests by `build_value_to_hlvalue_map`).
///
/// Returns `Err(TyperError)` for variants that are not yet lowered.
/// The error message names the specific variant so the dual-gate
/// failure cleanly identifies which unported variant it hit.
/// Format `op.result` for diagnostic messages.  Renders the result
/// `Variable`'s identity name directly through its `Display`
/// (`Variable.__repr__`, the `{_name}{_nr}` shape) so fail-loud
/// output stays readable without a backing-slot projection.
fn fmt_op_result(op: &SpaceOperation) -> String {
    match op.result.as_ref() {
        Some(var) => format!("Some({var})"),
        None => "None".to_string(),
    }
}

/// `true` iff `kind` is a 0-arg unit-variant transparent ctor
/// (`StepResult::Continue`, `LoopResult::Done`, …) that [`translate_op`]
/// pre-folds to a `Constant` and emits no `FlowspaceOp` (the unit-variant
/// guard at the top of `translate_op`).
///
/// [`op_canraise`] and [`translate_op`] consult the SAME predicate —
/// `op_canraise` is false exactly when `translate_op` emits no op.
fn is_elided_unit_variant_ctor(kind: &OpKind) -> bool {
    if let OpKind::Call {
        target:
            crate::model::CallTarget::SyntheticTransparentCtor {
                name, owner_path, ..
            },
        args,
        ..
    } = kind
        && args.is_empty()
    {
        let mut segments = owner_path.clone();
        segments.push(name.clone());
        return crate::translator::rtyper::unit_variant_fold::is_synthetic_unit_variant_path(
            &segments,
        );
    }
    false
}

/// A fixed-size Rust array aggregate emitted by `front::mir`.
///
/// The semantic/legacy graph keeps the exact `Array` transparent-ctor plus
/// `__pos_i` field writes required by legacy code generation and format-array
/// recognizers.  The RPython annotator has no distinct Rust array class:
/// arrays, slices, and Vec values are list-shaped.  The block adapter therefore
/// reconstructs this source aggregate as one `newlist(items...)` operation
/// while leaving the original model graph untouched.
fn is_array_aggregate_ctor(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call {
            target: crate::model::CallTarget::SyntheticTransparentCtor {
                name,
                owner_path,
                ..
            },
            ..
        } if owner_path.is_empty()
            && majit_ir::descr::strip_instantiation_suffix(name) == "Array"
    )
}

fn array_aggregate_elements(
    operations: &[SpaceOperation],
    array: &Variable,
) -> Option<Vec<Variable>> {
    let mut by_index = Vec::new();
    for op in operations {
        if let OpKind::FieldWrite {
            base, field, value, ..
        } = &op.kind
            && base.id() == array.id()
        {
            let index = field.name.strip_prefix("__pos_")?.parse::<usize>().ok()?;
            by_index.push((index, value.as_variable()?.clone()));
        }
    }
    by_index.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in by_index.iter().enumerate() {
        if *actual != expected {
            return None;
        }
    }
    Some(by_index.into_iter().map(|(_, value)| value).collect())
}

/// A shaped Rust tuple aggregate emitted by `front::mir`: the
/// `Tuple<A,B,..>` transparent ctor followed by its `__pos_i` writes.
///
/// A Rust tuple is an RPython tuple: `newtuple(items...)` builds a
/// `SomeTuple` (`annotator/unaryop.py`), `t[i]` reads an item and the
/// rtyper lowers both through `TupleRepr` (`rtyper/rtuple.py`).  As with
/// [`is_array_aggregate_ctor`], only this flowspace view is rebuilt; the
/// semantic graph keeps the ctor and writes for legacy code generation.
fn tuple_aggregate_shape(kind: &OpKind) -> Option<&str> {
    match kind {
        OpKind::Call {
            target:
                crate::model::CallTarget::SyntheticTransparentCtor {
                    name, owner_path, ..
                },
            args,
            ..
        } if owner_path.is_empty()
            && args.is_empty()
            && majit_ir::descr::is_shaped_tuple_name(name) =>
        {
            Some(name.as_str())
        }
        _ => None,
    }
}

/// A tuple aggregate rebuilt as `newtuple`: its items in position order,
/// and the index of the block operation the `newtuple` replaces.
struct TupleAggregate {
    items: Vec<crate::model::LinkArg>,
    at: usize,
}

/// The items of the tuple `shape` built by the ctor at `ctor_at`, whose
/// result is `tuple`.  The front computes an item after the ctor and
/// before its `__pos_i` write, so `newtuple` takes the place of the last
/// write.  The front skips the write of a zero-sized `()` item, whose value
/// is `None`; any other missing or repeated position declines.
fn tuple_aggregate_items(
    operations: &[SpaceOperation],
    ctor_at: usize,
    tuple: &Variable,
    shape: &str,
) -> Option<TupleAggregate> {
    let inner = shape.strip_prefix("Tuple<")?.strip_suffix('>')?;
    let spellings = crate::front::mir::split_top_level_type_args(inner);
    let mut items: Vec<Option<crate::model::LinkArg>> = vec![None; spellings.len()];
    let mut at = ctor_at;
    for (op_index, op) in operations.iter().enumerate() {
        if let OpKind::FieldWrite {
            base, field, value, ..
        } = &op.kind
            && base.id() == tuple.id()
        {
            let index = field.name.strip_prefix("__pos_")?.parse::<usize>().ok()?;
            let slot = items.get_mut(index)?;
            if slot.is_some() {
                return None;
            }
            *slot = Some(value.clone());
            at = at.max(op_index);
        }
    }
    let items = items
        .into_iter()
        .zip(spellings)
        .map(|(item, spelling)| {
            item.or_else(|| {
                (spelling == "()")
                    .then(|| crate::model::LinkArg::Const(Constant::new(ConstValue::None)))
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(TupleAggregate { items, at })
}

/// `true` iff `kind` is `front::mir`'s synthetic string-literal
/// define-op (`Call(["__str_const", <text>])`, `mir.rs`).
/// Upstream flowspace carries a string literal as a bare
/// `Constant('text')` SSA value, so the pre-pass
/// ([`legacy_const_define_hlvalue`]) folds the op to that Constant and
/// [`translate_op`] emits nothing; a Constant is not an operation and
/// raises nothing, so [`op_canraise`] is false — same contract as the
/// unit-variant ctor elision above.
fn is_str_const_define(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.is_empty() && segments.len() == 2 && segments[0] == "__str_const"
    )
}

/// `true` iff `kind` is `front::mir`'s synthetic function-item constant
/// define-op (`Call(["__fn_const", <real path>...])`).  Like
/// [`is_str_const_define`], this is a constant carrier rather than a
/// flowspace call operation.
fn is_fn_const_define(kind: &OpKind) -> bool {
    match kind {
        OpKind::Call { target, args, .. } if args.is_empty() => {
            crate::model::fn_const_segments(target).is_some()
        }
        _ => false,
    }
}

/// `true` iff `kind` is `front::mir`'s synthetic prebuilt-constant-array
/// define-op (`Call(["__const_int_array", <i0>, <i1>, …])`,
/// `fold_named_const_int_array_global`).  A module-level constant array
/// is carried as a bare `Constant(list)` SSA value, so the pre-pass
/// ([`legacy_const_define_hlvalue`]) folds the op to that Constant and
/// [`translate_op`] emits nothing — same contract as
/// [`is_str_const_define`].
fn is_const_int_array_define(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.is_empty() && segments.len() >= 2 && segments[0] == "__const_int_array"
    )
}

/// Decode the baked integers of a `__const_int_array` define-op's
/// segments (`["__const_int_array", "0", "2", …]`) into the
/// `ConstValue::Int` element list.  `None` if any segment past the tag
/// fails to parse (a malformed synthesis) or the op is not the
/// define-op shape.
fn const_int_array_items(kind: &OpKind) -> Option<Vec<ConstValue>> {
    let OpKind::Call {
        target: crate::model::CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return None;
    };
    segments
        .get(1..)?
        .iter()
        .map(|s| s.parse::<i64>().ok().map(ConstValue::Int))
        .collect()
}

/// `true` iff `kind` is `front::mir`'s synthetic prebuilt-string-array
/// define-op (`Call(["__const_str_array", <s0>, <s1>, …])`,
/// `fold_named_const_str_array_global`). Same Constant(list) contract as
/// [`is_const_int_array_define`].
fn is_const_str_array_define(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.is_empty() && segments.len() >= 2 && segments[0] == "__const_str_array"
    )
}

fn const_str_array_items(kind: &OpKind) -> Option<Vec<ConstValue>> {
    let OpKind::Call {
        target: crate::model::CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return None;
    };
    Some(
        segments
            .get(1..)?
            .iter()
            .map(|s| ConstValue::ByteStr(s.clone().into_bytes()))
            .collect(),
    )
}

fn is_const_prebuilt_array_define(kind: &OpKind) -> bool {
    is_const_int_array_define(kind) || is_const_str_array_define(kind)
}

fn const_prebuilt_array_items(kind: &OpKind) -> Option<Vec<ConstValue>> {
    const_int_array_items(kind).or_else(|| const_str_array_items(kind))
}

/// The pure, non-raising flowspace opname a `core::<family>::<leaf>`
/// method-call bridge lowers to in [`translate_op`], or `None` when the
/// path is not such a bridge.  The single source of truth shared by
/// [`translate_op`] (which emits the op) and [`op_canraise`] (which must
/// agree the emitted op closes no exception edge), so the two tables
/// cannot drift.  `core::cmp::{min,max}` is deliberately excluded: it
/// lowers to a `simple_call(<builtin>)` that keeps the ordinary raising
/// `CallOp` classification.
fn nonraising_core_bridge_opname(segments: &[String], arg_count: usize) -> Option<&str> {
    if segments.len() < 3 || segments[0] != "core" {
        return None;
    }
    let family = segments[1].as_str();
    let leaf = segments[segments.len() - 1].as_str();
    match (family, leaf) {
        // Rich comparisons map to the like-named flowspace op; plain
        // comparisons carry an empty `canraise` (operation.py).
        ("cmp", "eq" | "ne" | "lt" | "le" | "gt" | "ge") if arg_count == 2 => Some(leaf),
        ("slice", "len") if arg_count == 1 => Some("len"),
        ("slice", "iter") if arg_count == 1 => Some("iter"),
        // `wrapping_mul` / `wrapping_sub` are the Rust spelling of
        // upstream's plain `*` / `-` (lltype `int_mul` / `int_sub`
        // wrap); the plain `mul` / `sub` ops raise nothing.
        ("num", "wrapping_mul") if arg_count == 2 => Some("mul"),
        ("num", "wrapping_sub") if arg_count == 2 => Some("sub"),
        _ => None,
    }
}

/// `true` iff `segments` is the `core::slice::<Impl>::reverse` FunctionPath
/// that `front::mir` emits for a Rust `slice.reverse()` call (which has no
/// source body to register).  Unlike the bridge ops above, `reverse` has no
/// rtyper *operator* arm, so it is routed in [`translate_op`] to the
/// `getattr` + `simple_call` *method* shape that reaches
/// `FixedSizeListRepr.rtype_method("reverse")` (`rlist.py`
/// `rtype_method_reverse` → `ll_reverse`).  Shared with [`op_canraise`],
/// which classifies the originating Call non-raising
/// (`rlist.py hop.exception_cannot_occur()`).
fn is_slice_reverse_segments(segments: &[String]) -> bool {
    segments.len() == 4
        && segments[0] == "core"
        && segments[1] == "slice"
        && segments[2] == "<Impl>"
        && segments[3] == "reverse"
}

/// `Vec::with_capacity(n)` / `Vec::new()` (Rust MIR `vec::Vec::{with_capacity,
/// new}`) — the resizable-list constructor. Routed in [`translate_op`] to the
/// `newlist` operation with no element args, the shape `rtype_newlist`
/// (`rlist.py`) lowers through `newlist(llops, r_list, [])`.
///
/// The capacity operand is dropped: upstream keeps a size hint only through
/// `objectmodel.newlist_hint` (`objectmodel.py`), whose
/// `rtype_newlist(hop, v_sizehint)` path our `rtype_newlist` does not take —
/// the hint is an allocation optimization, not a semantic difference, so an
/// empty `newlist` is the faithful lowering of both constructors.
///
/// Recognizing the constructor is what lets [`is_vec_push_segments`] and
/// [`is_slice_reverse_segments`] work at all: they emit `getattr(recv, ...)`
/// method shapes that require the receiver to annotate as a `SomeList`. While
/// the constructor stayed a residual host call its result was the
/// classdef-less opaque `SomeInstance` that `foreign_container_ctor` types,
/// and the `getattr` against it aborted annotation.
///
/// Like [`is_vec_push_segments`], this recognizer fires only on the
/// `CallTarget::FunctionPath` shape, which is complete for the foreign `vec`
/// crate: `front::mir::impl_method_owner` cannot resolve `vec::Vec` to a
/// classdef-bound owner, so every constructor call arrives as these segments.
fn is_vec_ctor_segments(segments: &[String]) -> bool {
    segments.len() == 3
        && segments[0] == "vec"
        && segments[1] == "Vec"
        && (segments[2] == "with_capacity" || segments[2] == "new")
}

/// `pyre_interpreter::builtins::try_pyobject_vec_with_capacity(sizehint)` is
/// the Rust source spelling of RPython's `objectmodel.newlist_hint(sizehint)`.
///
/// The helper is deliberately an opaque source-translation boundary, just as
/// upstream's function is handled by an ExtRegistryEntry instead of tracing
/// its Python body.  Its ordinary interpreter body still performs the
/// fallible Rust reservation; translated callers take this exact path into
/// `rtype_newlist_hint` and therefore retain both the capacity hint and the
/// MemoryError edge.
fn is_runtime_newlist_hint_segments(segments: &[String]) -> bool {
    segments.len() >= 2
        && segments[segments.len() - 2] == "builtins"
        && segments[segments.len() - 1] == "try_pyobject_vec_with_capacity"
}

/// `alloc::vec::from_elem(elem, count)` (Rust MIR `vec![elem; count]`) —
/// the repeated resizable-list constructor.  Routed in [`translate_op`] to
/// `newlist(elem)` + `mul(list, count)`, the same `[a] * b` shape that
/// `transform_allocate` (`transform.py`) collapses to
/// `alloc_and_set` and `rtype_alloc_and_set` (`rlist.py`) lowers
/// through `ll_alloc_and_set` (`rlist.py`).
///
/// Charon emits this call as `alloc::vec::from_elem`, while the constructor
/// recognizer above omits the foreign crate prefix. Match the trailing
/// `vec::from_elem` spelling defensively, but require the exact two-argument
/// `(elem, count)` shape.
fn is_vec_from_elem_segments(segments: &[String], arg_count: usize) -> bool {
    segments.len() >= 2
        && segments[segments.len() - 2] == "vec"
        && segments[segments.len() - 1] == "from_elem"
        && arg_count == 2
}

/// `Vec::push(l, item)` (Rust MIR `vec::Vec::push`) — the resizable-list
/// append. Routed to the resized `ListRepr.rtype_method("append")`
/// (`rlist.py:185`) via the `getattr(recv, "append") + simple_call` method
/// shape, exactly like [`is_slice_reverse_segments`]; the Rust method name
/// `push` maps to the RPython list method `append`.
///
/// This recognizer fires only on the `CallTarget::FunctionPath` shape, and
/// that is complete: `vec::Vec` is a foreign type with no extracted LLBC ADT,
/// so `front::mir::impl_method_owner` cannot resolve it to a classdef-bound
/// owner and returns `None`, which forces every `Vec::push` call to the
/// `[vec, Vec, push]` FunctionPath segments rather than `CallTarget::Method`.
/// The generic Method arm (which would `getattr(recv, "push")` against a list
/// that has no `push`) is therefore unreachable for it — only user-defined
/// classdef-bound receivers route through Method.
fn is_vec_push_segments(segments: &[String]) -> bool {
    segments.len() == 3 && segments[0] == "vec" && segments[1] == "Vec" && segments[2] == "push"
}

/// `RDict::insert` / `RDict::get` (`rordereddict.py` `ll_dict_setitem` /
/// `ll_dict_lookup`). The owner segment is the type; `unspecialized_leaf`
/// sees through the monomorphized copy. The hash in the leaf is not part
/// of the match.
fn rdict_insert_or_get_leaf(segments: &[String]) -> Option<&str> {
    if segments.len() < 3 || segments[segments.len() - 2] != "RDict" {
        return None;
    }
    let leaf = crate::front::clause_spec::unspecialized_leaf(segments.last()?);
    matches!(leaf, "insert" | "get").then_some(leaf)
}

/// `Vec::extend_from_slice(l, slice)` (Rust MIR `vec::Vec::extend_from_slice`)
/// — appends every element of `slice` to the resizable list. Routed to the
/// resized `ListRepr.rtype_method("extend")` (`rlist.py`) via the
/// `getattr(recv, "extend") + simple_call` method shape, exactly like
/// [`is_vec_push_segments`]; the Rust method name `extend_from_slice` maps to
/// the RPython list method `extend` (whose argument is the slice, a
/// non-resized list — `list_method_extend` takes the list-list path).
fn is_vec_extend_from_slice_segments(segments: &[String]) -> bool {
    segments.len() == 3
        && segments[0] == "vec"
        && segments[1] == "Vec"
        && segments[2] == "extend_from_slice"
}

/// Test-only mirror of whether the flowspace op(s) this `OpKind`
/// lowers to carry a non-empty `canraise` (`operation.py`).
///
/// The production path is `translate_op`, which emits real flowspace
/// operations and lets their upstream-modeled `canraise` metadata drive
/// exception edges.  This predicate is retained only for tests that keep
/// the local lowering table honest: a non-raising tail op (a transparent
/// ctor, `same_as`, a pure cast / binop, getattr / setattr, a const)
/// must not be classified as canraise.
///
/// KEEP IN SYNC with [`translate_op`]'s `OpKind` -> flowspace-opname arms.
#[cfg(test)]
pub(crate) fn op_canraise(kind: &OpKind) -> bool {
    match kind {
        // getitem / setitem -> `[IndexError, KeyError, Exception]`
        // (operation.py:727-730).
        OpKind::ArrayRead { .. } | OpKind::ArrayWrite { .. } => true,
        // `len` is a pure flowspace operation; the rtyper routes it through
        // the receiver repr (`rtype_len`) without an exception edge.
        OpKind::ArrayLen { .. } => false,
        // `InteriorField*` unfolds in `translate_op` into a chained
        // `getitem(base, index)` followed by `getattr` / `setattr`, so it
        // carries the getitem's `[IndexError, KeyError, Exception]`
        // (operation.py).  The getattr / setattr step is itself
        // non-raising, but the getitem makes the op raise.
        OpKind::InteriorFieldRead { .. } | OpKind::InteriorFieldWrite { .. } => true,
        // A transparent `Ok(x)` / `Some(x)` / `Err(e)` ctor lowers to
        // `simple_call(HostClass(qualname), x…)` (the
        // `SyntheticTransparentCtor` arm in `translate_op`).  The `?` /
        // Result / Option transparent-ctor elision is a Rust-specific
        // adaptation with no RPython counterpart, but the simple_call it
        // emits is classified by the ordinary `CallOp.canraise` rule: a
        // Constant class callable outside `__builtin__` / `exceptions`
        // raises `[Exception]` (operation.py:648-661), so a non-unit ctor
        // raises like any non-builtin call.  The sole non-raising case is
        // the 0-arg unit-variant ctor, which `translate_op` pre-folds to a
        // `Constant` and emits no op — `op_canraise` is false exactly when
        // that happens.  Matched before the general `Call` arm.
        kind @ OpKind::Call {
            target: crate::model::CallTarget::SyntheticTransparentCtor { .. },
            ..
        } => !is_elided_unit_variant_ctor(kind),
        // A string-literal define-op pre-folds to `Constant('text')`
        // and emits no op — a Constant raises nothing.  Matched before
        // the general `Call` arm, same as the unit-variant elision.
        kind if is_str_const_define(kind) => false,
        // A function-item constant define-op is a bare constant carrier,
        // not a callee invocation, so it raises nothing.
        kind if is_fn_const_define(kind) => false,
        // A prebuilt-constant-array define-op pre-folds to `Constant(list)`
        // and emits no op — a Constant raises nothing.  Matched before the
        // general `Call` arm, same as the string-literal elision.
        kind if is_const_prebuilt_array_define(kind) => false,
        // A `hint(x, **kwds)` op (`OpKind::Hint`) lowers to a non-raising
        // `same_as(value)` (`rtyper.py:478-481` internal renaming) in
        // `translate_op` — it emits no raising op.  Matched before the
        // general `Call` arm, same as the elisions above.
        OpKind::Hint { .. } => false,
        // `raw_load` / `raw_store` are llops with empty `canraise`
        // (`LL_OPERATIONS['raw_load']` / `['raw_store']`).
        OpKind::RawLoad { .. } | OpKind::RawStore { .. } => false,
        // `core::cmp::{eq..ge}` / `core::slice::{len,iter}` /
        // `core::num::wrapping_mul` lower to pure, non-raising flowspace
        // ops (see `nonraising_core_bridge_opname`); classify the
        // originating Call the same way so a `?` tail op does not install
        // an exception edge the lowered op cannot take.  Matched before
        // the general `Call` arm.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if nonraising_core_bridge_opname(segments, args.len()).is_some() => false,
        // `__strlen` rewrites to the rtyper `len` op (`is_len_op`);
        // `StringRepr.rtype_len` → `ll_strlen` raises nothing.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if segments.as_slice() == ["__strlen"] && args.len() == 1 => false,
        // `core::slice::<Impl>::reverse` lowers (in `translate_op`) to a
        // `getattr` + `simple_call` method shape whose `rtype_method_reverse`
        // does `hop.exception_cannot_occur()` (`rlist.py`); classify the
        // originating Call non-raising so a `?` tail op installs no exception
        // edge the lowered op cannot take.  (`reverse` returns `()`, so it is
        // never actually a `?` operand — this keeps the table faithful.)
        // Matched before the general `Call` arm.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            ..
        } if is_slice_reverse_segments(segments) => false,
        // `vec::Vec::{with_capacity,new}` lowers (in `translate_op`) to the
        // `newlist` operation, whose `canraise` is `[]` (operation.py
        // `class NewList`); classify the originating Call non-raising so a `?`
        // tail op installs no exception edge the lowered op cannot take.
        // Matched before the general `Call` arm.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if is_vec_ctor_segments(segments) || is_vec_from_elem_segments(segments, args.len()) => {
            false
        }
        // `[__iter_next]` (`front::iter_next`) lowers to the raising
        // flowspace `next` op (`operation.rs` `OpKind::Next.can_only_throw
        // = [StopIteration, RuntimeError]`).  Matched before the general
        // `Call` arm, mirroring [`translate_op`]'s `[__iter_next]` arm.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.len() == 1 && crate::front::iter_next::is_iter_next_segments(segments) => true,
        // simple_call -> `CallOp.canraise` is `[Exception]` for a
        // non-builtin callable (operation.py).  Constant builtin
        // callables (int / float / chr / unicode) carry the narrower
        // `builtins_exceptions` set, but a `?` operand is Result/Option-
        // typed so over-approximating those few builtins is inert.  The
        // jtransform-generated call variants cannot appear as a front-end
        // `Expr::Try` tail op (they are produced by a later pass); listing
        // them only keeps the predicate faithful to `CallOp`.
        OpKind::Call { .. }
        | OpKind::IndirectCall { .. }
        | OpKind::CallElidable { .. }
        | OpKind::CallResidual { .. }
        | OpKind::CallMayForce { .. } => true,
        // Plain binops: div / mod / divmod / truediv / floordiv / pow carry
        // ZeroDivisionError, and pow / lshift / rshift carry ValueError,
        // even without the `_ovf` suffix (operation.py:751-756); plain
        // add / sub / mul / cmp / bitops are `[]`.  Compound-assign names
        // (`*_assign`) are seen here BEFORE `normalize_binop_name` maps
        // them to `inplace_*`, so they are classified by their `inplace_*`
        // canraise: `inplace_div/mod/lshift/rshift` keep the plain
        // ZeroDivisionError/ValueError, and `inplace_add/sub/mul` carry
        // OverflowError (they have no `_ovf` variant) (operation.py:751-756);
        // `inplace_and/or/xor` are `[]`.  The Rust front end emits the
        // plain div / mod / lshift / rshift and the `*_assign` family; the
        // other plain names are faithful but never produced.
        OpKind::BinOp { op, .. } => matches!(
            op.as_str(),
            "div"
                | "mod"
                | "divmod"
                | "truediv"
                | "floordiv"
                | "pow"
                | "lshift"
                | "rshift"
                // The `_ovf` arithmetic twins carry `[OverflowError]`
                // (operation.py `_add_except_ovf`;
                // `OpKind::{Add,Sub,Mul}Ovf.canraise()`).  The front-end
                // emits them only at a `LastException` block whose
                // `raising_op` is the `_ovf` op (`front::checked_arith`),
                // so they must classify as raising.
                | "add_ovf"
                | "sub_ovf"
                | "mul_ovf"
                | "add_assign"
                | "sub_assign"
                | "mul_assign"
                | "div_assign"
                | "mod_assign"
                | "lshift_assign"
                | "rshift_assign"
        ),
        // getattr / setattr / neg / not / type / same_as / Const* /
        // Guard* / VableForce / Input / Abort (JIT-abort marker) -> all
        // `canraise = []`.
        _ => false,
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
/// Census-only accumulation of a graph's *whole* unresolvable-callee set.
///
/// The adapter is first-fail: [`function_graph_to_flowspace`] returns on the
/// first op it cannot translate, so `MAJIT_RTYPER_VERBOSE`'s
/// `[PREPASS phaseA fail]` row names one wall per graph even when the graph
/// stands behind several.  Ranking walls by that row therefore ranks *first*
/// walls, and closing the top one relocates its graphs to their next wall
/// instead of lifting them — measured on `Wtf8Buf::push_str`, where the wall
/// fell 15 → 5 and the census total did not move.
///
/// `MAJIT_RTYPER_FRONTIER` keeps the scan going past a failed **call** op so
/// the row can name the whole set.  Upstream spells the same distinction
/// `keepgoing` (`annrpython.py`): its `flowin` records the error in
/// `self.errors`, adds the block to `self.failed_blocks` and returns instead of
/// raising, and `complete` reports every recorded error together.  Pyre's
/// annotator ports that flag (`RPythonAnnotator::keepgoing`), but the walls
/// counted here are raised one layer earlier — in this adapter, before the
/// annotator sees the graph — so the same behaviour is spelled here.
///
/// **Only call ops resume.**  A call's resolution failure is a function of the
/// callee path and the registry alone, so skipping it invents no wall; any
/// other op's failure may be a consequence of an operand this scan skipped,
/// and resuming past it would manufacture walls that do not exist.  The
/// skipped call's result is bound to a placeholder so later ops still resolve
/// their operands — a census-only graph that is discarded, since the adapter
/// still returns the first error and the subject still fails exactly as it
/// does with the gate off.
mod frontier {
    use super::TyperError;
    use std::cell::RefCell;

    #[derive(Default)]
    struct Frame {
        /// Every distinct wall seen in this adapter run, in discovery order.
        walls: Vec<String>,
        /// The wall the gate-off run would have stopped at.
        first: Option<TyperError>,
    }

    thread_local! {
        /// One frame per active adapter run.  A run is re-entrant: resolving a
        /// call can lift the callee's source, which runs the adapter again, and
        /// a flat slot would let the inner run adopt the outer run's `first`.
        static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
        /// Walls of the last outermost run, awaiting [`super::take_frontier_walls`].
        static LAST: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }

    /// Whether `MAJIT_RTYPER_FRONTIER` is on.  Same `== "1"` contract as
    /// `MAJIT_RTYPER_VERBOSE`, so a literal `0` stays off.
    pub(super) fn enabled() -> bool {
        std::env::var_os("MAJIT_RTYPER_FRONTIER").is_some_and(|v| v == "1")
    }

    /// Guard for one adapter run.  Collecting is scoped to its lifetime.
    pub(super) struct Guard {
        active: bool,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            if !self.active {
                return;
            }
            let frame = STACK.with(|s| s.borrow_mut().pop()).unwrap_or_default();
            STACK.with(|s| {
                let mut stack = s.borrow_mut();
                match stack.last_mut() {
                    // An inner run's walls block the outer run too — its
                    // failure is what the outer call reports — so they join the
                    // caller's set rather than being dropped.
                    Some(outer) => {
                        for wall in frame.walls {
                            if !outer.walls.contains(&wall) {
                                outer.walls.push(wall);
                            }
                        }
                    }
                    None => LAST.with(|l| *l.borrow_mut() = frame.walls),
                }
            });
        }
    }

    /// Begin collecting for one adapter run.
    pub(super) fn enter() -> Guard {
        let active = enabled();
        if active {
            STACK.with(|s| {
                let mut stack = s.borrow_mut();
                // An outermost run owns the next census row: drop the previous
                // run's set now, so a subject whose adapter is never reached
                // reports nothing rather than its predecessor's walls.
                if stack.is_empty() {
                    LAST.with(|l| l.borrow_mut().clear());
                }
                stack.push(Frame::default());
            });
        }
        Guard { active }
    }

    /// Whether `err`, raised on `op`, may be skipped and the run resumed.
    /// Records the wall as a side effect when it may.
    pub(super) fn resume(op: &super::SpaceOperation, err: &TyperError) -> bool {
        if !matches!(op.kind, super::OpKind::Call { .. }) {
            return false;
        }
        STACK.with(|s| {
            let mut stack = s.borrow_mut();
            let Some(frame) = stack.last_mut() else {
                return false;
            };
            // One line per wall: the registry message is a paragraph, and a
            // multi-line census record is unparseable without a banner to split
            // on.
            let text = err
                .to_string()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            if !frame.walls.contains(&text) {
                frame.walls.push(text);
            }
            frame.first.get_or_insert_with(|| err.clone());
            true
        })
    }

    /// The error the gate-off run would have returned, if this run skipped a
    /// wall.
    pub(super) fn pending_error() -> Option<TyperError> {
        STACK.with(|s| s.borrow().last().and_then(|f| f.first.clone()))
    }

    /// Take the wall set of the last completed outermost run.
    pub(super) fn take_last() -> Vec<String> {
        LAST.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }
}

/// Read the wall set [`frontier`] collected for the subject just translated.
/// Empty unless `MAJIT_RTYPER_FRONTIER` is on.
pub fn take_frontier_walls() -> Vec<String> {
    frontier::take_last()
}

/// [`legacy_const_define_hlvalue`], resuming past a skippable call wall under
/// [`frontier`].
///
/// The unresolvable fn-const is stood in for by a
/// [`ConstValue::Placeholder`] constant rather than reported as "not a const
/// define": the const-define pass is graph-wide, so demoting the value to a
/// block-local Variable would leave every other block using it undefined.
fn legacy_const_define_hlvalue_or_frontier(
    op: &SpaceOperation,
    call_registry: Option<&crate::translator::rtyper::call_registry::CallRegistry>,
) -> Result<Option<Hlvalue>, TyperError> {
    match legacy_const_define_hlvalue(op, call_registry) {
        Err(err) if frontier::resume(op, &err) => Ok(Some(Hlvalue::Constant(
            constant_from_constvalue(ConstValue::Placeholder),
        ))),
        other => other,
    }
}

/// [`translate_op`], resuming past a skippable call wall under [`frontier`].
///
/// The skipped call is replaced by a `direct_call` on a
/// [`ConstValue::Placeholder`] callee, not dropped.  Both halves are forced by
/// [`crate::flowspace::model`]'s structural check: the caller has already
/// seeded the call's result Variable and a later block reads it through a
/// Link, so some op must define it; and a `canraise` block's last op may not
/// be `same_as`, so the stand-in has to stay a call.  It is census-only —
/// this graph is discarded, since [`function_graph_to_flowspace`] reports the
/// wall.
fn translate_op_or_frontier(
    op: &SpaceOperation,
    value_map: &HashMap<Variable, Hlvalue>,
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
) -> Result<Vec<FlowspaceOp>, TyperError> {
    match translate_op(op, value_map, call_registry) {
        Err(err) if frontier::resume(op, &err) => {
            let Some(result) = op.result.as_ref() else {
                return Ok(Vec::new());
            };
            let result = lookup_operand(value_map, result, op, "result")?;
            // Already stood in for by
            // [`legacy_const_define_hlvalue_or_frontier`] — the constant needs
            // no defining op.
            if matches!(result, Hlvalue::Constant(_)) {
                return Ok(Vec::new());
            }
            let placeholder = Hlvalue::Constant(constant_from_constvalue(ConstValue::Placeholder));
            Ok(vec![FlowspaceOp::new(
                "direct_call",
                vec![placeholder],
                result,
            )])
        }
        other => other,
    }
}

/// Lower a `dyn Trait` virtual call to the pre-rtyper shape:
/// `getattr(receiver, method_name)` then `simple_call(bound_method, args[1..])`,
/// with the receiver narrowed to the trait family's base `ClassDef` first when
/// the trait is a registered dispatch family.
///
/// `args[0]` is the receiver; the rationale for this shape is on the
/// `CallTarget::Indirect` arm of [`translate_op`], which is one of the two
/// callers. The other is that arm's pyre-internal twin, `OpKind::IndirectCall`
/// carrying a `family_key`: an `indirect_call` op has no pre-rtyper spelling
/// upstream — `rpbc.py` mints it inside `rtype_simple_call`, so
/// everything the annotator sees is still a `simple_call` — and the adapter is
/// the entry into annotation, so both spellings must arrive here as one.
fn dyn_trait_dispatch_ops(
    arg_hls: Vec<Hlvalue>,
    trait_root: &str,
    method_name: &str,
    result: Hlvalue,
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
) -> Result<Vec<FlowspaceOp>, TyperError> {
    let mut iter = arg_hls.into_iter();
    let receiver = iter.next().ok_or_else(|| {
        TyperError::message(
            "Call::Indirect has empty args: dyn-Trait receiver must be args[0]".to_string(),
        )
    })?;
    let base_root = call_registry
        .bookkeeper()
        .registered_trait_family_base_root(trait_root);
    let mut ops = Vec::with_capacity(3);
    // Narrow the receiver to the family base classdef when the
    // trait is a registered dispatch family.
    let getattr_receiver = if let Some(base_root) = base_root {
        let callable_host = HOST_ENV
            .lookup_builtin(crate::runtime_names::shims::CAST_INSTANCE)
            .ok_or_else(|| {
                TyperError::message(
                    "__cast_instance_intrinsic missing from HOST_ENV bootstrap".to_string(),
                )
            })?;
        let narrowed = Hlvalue::Variable(Variable::new());
        ops.push(FlowspaceOp::new(
            "simple_call",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::HostObject(callable_host))),
                receiver,
                Hlvalue::Constant(Constant::new(ConstValue::byte_str(&base_root))),
            ],
            narrowed.clone(),
        ));
        narrowed
    } else {
        receiver
    };
    let bound_method = Hlvalue::Variable(Variable::new());
    let mut call_args = Vec::with_capacity(iter.size_hint().0 + 1);
    call_args.push(bound_method.clone());
    call_args.extend(iter);
    ops.push(FlowspaceOp::new(
        "getattr",
        vec![
            getattr_receiver,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str(method_name))),
        ],
        bound_method,
    ));
    ops.push(FlowspaceOp::new("simple_call", call_args, result));
    Ok(ops)
}

/// `objectmodel.instantiate(cls)`: allocate an instance of the constant
/// class `cls` without calling its `__init__` (`robjmodel_instantiate`,
/// `rtype_instantiate`).
fn instantiate_op(class_host: HostObject, result: Hlvalue) -> Result<FlowspaceOp, TyperError> {
    let instantiate = HOST_ENV
        .import_module("rpython.rlib.objectmodel")
        .and_then(|module| module.module_get("instantiate"))
        .ok_or_else(|| {
            TyperError::message("objectmodel.instantiate missing from HOST_ENV bootstrap")
        })?;
    Ok(FlowspaceOp::new(
        "simple_call",
        vec![
            Hlvalue::Constant(Constant::new(ConstValue::HostObject(instantiate))),
            Hlvalue::Constant(Constant::new(ConstValue::HostObject(class_host))),
        ],
        result,
    ))
}

pub fn translate_op(
    op: &SpaceOperation,
    value_map: &HashMap<Variable, Hlvalue>,
    // The call registry is consulted by the `OpKind::Call::FunctionPath`
    // arm to resolve a registered `(HostObject, FunctionDesc)` pair
    // and emit a flowspace `simple_call` (`operation.py`,
    // `rpbc.rs`'s `FunctionRepr::rtype_simple_call`).  Empty registry
    // callsites surface a distinct fail-loud message; producers
    // must pre-register every reachable FunctionPath.
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
) -> Result<Vec<FlowspaceOp>, TyperError> {
    // Unit-variant ctors (`StepResult::Continue`, `LoopResult::Done`, …)
    // pre-fold to `Hlvalue::Constant(HostObject(prebuilt_instance))` in the
    // pre-pass (see `legacy_const_define_hlvalue`).  Skip translation here
    // so they do not double-emit as `simple_call(HostClass(qualname))` —
    // matches the `ConstInt`/`ConstBool`/`ConstFloat` pattern below (the
    // pre-pass owns the slot's `Hlvalue::Constant`, translate_op emits no
    // FlowspaceOp).  `op_canraise` consults the same predicate, so it
    // reports these — and only these — transparent ctors as non-raising.
    if is_elided_unit_variant_ctor(&op.kind) {
        return Ok(Vec::new());
    }
    // String-literal define-ops pre-fold to `Constant('text')` the same
    // way (see `is_str_const_define`); the pre-pass owns the slot's
    // `Hlvalue::Constant`, translate_op emits no FlowspaceOp.
    if is_str_const_define(&op.kind) {
        return Ok(Vec::new());
    }
    // Function-item define-ops pre-fold to a constant carrier the same
    // way (see `is_fn_const_define`); consumers that need the address
    // recover the real path by stripping the synthetic head.
    if is_fn_const_define(&op.kind) {
        return Ok(Vec::new());
    }
    // Prebuilt-constant-array define-ops pre-fold to `Constant(list)` the
    // same way (see `is_const_int_array_define`); the pre-pass owns the
    // slot's `Hlvalue::Constant`, translate_op emits no FlowspaceOp.
    if is_const_prebuilt_array_define(&op.kind) {
        return Ok(Vec::new());
    }
    match &op.kind {
        // ─── Skipped: fully consumed by other adapter infrastructure ───
        OpKind::Input { .. } => Ok(Vec::new()),
        OpKind::ConstInt(_)
        | OpKind::ConstFnAddr { .. }
        | OpKind::ConstUInt(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstFloat(_)
        | OpKind::ConstStr(_)
        | OpKind::ConstInternedStr(_)
        | OpKind::ConstRefNull
        | OpKind::ConstNone
        | OpKind::ConstRefAddr(_) => Ok(Vec::new()),
        // ─── Skipped: pyre JIT trace markers without a flowspace peer ───
        // `GuardTrue` / `GuardFalse` / `GuardValue` are JIT-side
        // assertions emitted by pyre's tracer — they constrain the
        // runtime value of an existing SSA operand and produce no new
        // SSA result.  `VableForce` is a virtualizable-flush hint
        // (`hint_force_virtualizable`, no operands, no result).
        // RPython's flowspace abstract interpreter does not have any
        // of these at the high-level rtyper input
        // (`rpython/flowspace/operation.py:475-510`); the equivalent
        // checks appear later in `pyjitpl` / `codewriter` after the
        // rtyper has already lowered to lltype.  For the type
        // resolution pass driven by `specialize_legacy_graph` they
        // are pure no-ops: skipping them preserves the SSA chain
        // (any operand they read is defined elsewhere; the absence
        // of a result means no consumer is left unsatisfied).
        OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::VableForce { .. } => Ok(Vec::new()),

        // `hint(x, **kwds)` (`OpKind::Hint`) is an identity outside the JIT:
        // the flowspace oracle types its result as `same_as(value)` — the
        // `rtyper.py:478-481` internal-renaming op — exactly as RPython's
        // `hint` op is a no-op in the genc/non-JIT build, while the JIT
        // codewriter (`jtransform::rewrite_op_hint`) is what rewrites it to
        // the `<kind>_guard_value` family.  `jit::promote(x)` and
        // `#[elidable_promote]`'s `hint_promote_or_string` both land here.
        OpKind::Hint { value, .. } => {
            let value_hl = lookup_operand(value_map, value, op, "value")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("same_as", vec![value_hl], result)])
        }

        // ─── pyre-only `OpKind::Abort` marker ───
        // Front-end `lower_expr::stop_unsupported` / `continue_with_unknown`
        // emit this when the surface Rust DSL hits an unsupported
        // expression form (unsupported lit, ForLoop, Closure, Macro,
        // …).  RPython upstream raises `FlowingError`
        // (`flowspace/flowcontext.py`) and drops the function
        // before annotator/rtyper see it, so there is no upstream
        // analogue.  A `SomeInstance(None) -> Ptr -> GcRef`
        // synthesis is not used because the `classdef=None` result
        // trips downstream `find_attribute` lookups.
        //
        // Post-`FORCE_ATTRIBUTES_INTO_CLASSES` pre-population (struct
        // field projection by `register_struct_fields`) impl-method
        // `self` narrows to a populated
        // ClassDef, so the original classdef:None cascade is no longer
        // triggered by the receiver projection.  Abort's own
        // result_var is still un-narrowed, but every front-end
        // emit-site falls into one of two shapes:
        //
        //   (a) `stop_unsupported` — pushes Abort and returns
        //       `Err(FlowingError::Unsupported)`; the parent `?`
        //       ladder aborts the body before any operand reads the
        //       result_var.  No downstream consumer ⇒ skipping is
        //       safe.
        //
        //   (b) `continue_with_unknown` — pushes Abort and returns
        //       the result_var as a `Lowered.value`.  Callers may
        //       consume it; for those, the absent translate output
        //       leaves the result_var unmapped in `value_to_var`,
        //       and the first consumer surfaces a fail-loud
        //       "undefined operand" message that
        //       `is_known_unported` classifies as Skip — same
        //       outcome as the prior "post-rtyper jtransform
        //       variant" Skip, just at a more localised site.
        //
        // Each emit-site is retired by lowering the
        // specific expression form properly (`ConstStr`, `Range`,
        // `Closure`, etc.); until each is lowered, this
        // arm absorbs the placeholder silently so the dual-gate
        // doesn't have to round-trip through a TyperError just to
        // re-classify as Skip.
        OpKind::Abort { .. } => Ok(Vec::new()),

        // ─── `newtuple` — RPython `BUILD_TUPLE` / `space.newtuple` ───
        // `PureOperation` (`operation.py`).  Each `args[i]`
        // Variable is routed through `value_map` so the legacy
        // flowspace SpaceOperation references the same Hlvalue
        // identities the graph validator (`checkgraph`) tracks; using
        // raw model-side Variables here would trip
        // "variable used before definition" when an earlier op
        // remapped its result to a different legacy Variable.
        OpKind::NewTuple { args } => {
            let mut hl_args: Vec<Hlvalue> = Vec::with_capacity(args.len());
            for (i, var) in args.iter().enumerate() {
                let role = format!("arg{i}");
                hl_args.push(lookup_operand(value_map, var, op, &role)?);
            }
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("newtuple", hl_args, result)])
        }

        // ─── `newlist` — RPython `BUILD_LIST` / `space.newlist` ───
        // `PureOperation`.  Same operand-routing discipline as
        // `newtuple`: each `args[i]` Variable goes through `value_map`
        // so the legacy SpaceOperation references the Hlvalue identities
        // `checkgraph` tracks.
        OpKind::NewList { args } => {
            let mut hl_args: Vec<Hlvalue> = Vec::with_capacity(args.len());
            for (i, var) in args.iter().enumerate() {
                let role = format!("arg{i}");
                hl_args.push(lookup_operand(value_map, var, op, &role)?);
            }
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("newlist", hl_args, result)])
        }

        // ─── `getslice` — RPython `l[start:stop]` / `space.getslice` ───
        // `PureOperation` (`operation.py`).  The three operands are
        // `(list, start, stop)`, each routed through `value_map` so the
        // legacy SpaceOperation references the Hlvalue identities
        // `checkgraph` tracks.  The annotator's `getslice` handler
        // (`unaryop.py`) builds a fresh `listdef.offspring`, and the
        // rtyper's `rtype_getslice` (`rlist.py`) lowers it to a
        // `gendirectcall` of the per-kind `ll_listslice_*` helper.
        OpKind::GetSlice { args } => {
            let mut hl_args: Vec<Hlvalue> = Vec::with_capacity(args.len());
            for (i, var) in args.iter().enumerate() {
                let role = format!("arg{i}");
                hl_args.push(lookup_operand(value_map, var, op, &role)?);
            }
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("getslice", hl_args, result)])
        }

        // ─── `New` / `NewWithVtable` — boxing GC allocation ───
        // The model-graph op carries the boxing struct leaf `owner` and flows
        // straight to the codewriter/assembler (`new_with_vtable`).  For the
        // ephemeral annotation / rtype type-oracle it mirrors the
        // `SyntheticTransparentCtor` struct path: `instantiate(cls)` against
        // the interned class host annotates the result as a fresh
        // `SomeInstance(owner)`.  `getuniqueclassdef_for_struct_root` first
        // forces the struct's field rows to be projected so the trailing
        // payload `FieldWrite(result, …)` resolves.
        OpKind::New { owner } | OpKind::NewWithVtable { owner, .. } => {
            let bk = call_registry.bookkeeper();
            bk.getuniqueclassdef_for_struct_root(owner).map_err(|e| {
                TyperError::message(format!(
                    "translate_op: allocation owner {owner:?} is not a known struct root: {e}"
                ))
            })?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![instantiate_op(
                bk.intern_class_by_qualname(owner),
                result,
            )?])
        }

        // ─── `LoadStatic` — single-segment static lookup ─
        // Pyre-only marker emitted by the frontend when a path
        // expression resolves to a crate-level `static` declaration
        // (SHOUTY_CASE constant like `GC_WEAKREF_BOX_TYPE`).  RPython
        // peer: `LOAD_GLOBAL` (`flowspace/flowcontext.py`)
        // resolves the name lookup to a `Constant(value)` directly
        // — no SpaceOperation is emitted, and the bound `Variable`
        // *is* the graph-level definition.  Pyre always emits an op
        // here so cross-block reads have a defined producer (the
        // `checkgraph` defining-var set requires every operand to
        // trace to an op result or `Block.inputargs`).
        //
        // When `extract_static_decls` could fold the static's
        // RHS to a `ConstValue` (`bool` / integer / float / string
        // literals + `const { LIT }` block wrapper), the adapter emits
        // `same_as(Constant(value))` — the concrete `Constant` shape PyPy
        // `LOAD_GLOBAL` pushes.  Unresolved RHS values are rejected here:
        // allowing a path-string sentinel to survive would create a
        // `same_as/*` JitCode opcode, but RPython's blackhole only has
        // `int_same_as/i>i` for test hints.
        OpKind::LoadStatic {
            segments, value, ..
        } => {
            let Some(v) = value else {
                return Err(TyperError::message(format!(
                    "translate_op: unresolved LoadStatic {segments:?} has no RPython \
                     JitCode counterpart; fold the static to a Constant before rtyper"
                )));
            };
            let constant = Hlvalue::Constant(Constant::new(v.clone()));
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("same_as", vec![constant], result)])
        }

        // ─── Pre-rtyper opname normalization ───
        // `front::mir::binop_label` emits Rust-side
        // names (`bitand`, `bitor`, `bitxor`, `add_assign`, ...).
        // RPython flowspace registers operators via
        // `add_operator('and_', 2, ...)` etc.
        // (`rpython/flowspace/operation.py`): `and_`, `or_`,
        // `xor`, `inplace_add`, `inplace_sub`, ...  Translate the
        // pyre-side name to its RPython counterpart so the rtyper's
        // `translate_op` arm matching (`rtyper.rs`) finds
        // the proper `pair_rtype_*` dispatch.
        OpKind::BinOp {
            op: opname,
            lhs,
            rhs,
            result_ty,
        } => {
            let l = lookup_operand(value_map, lhs, op, "lhs")?;
            let r = lookup_operand(value_map, rhs, op, "rhs")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            let normalized = normalize_binop_name(opname)?;
            if result_ty == &crate::model::ValueType::Unsigned && !opname.ends_with("_ovf") {
                // The model graph retains the Rust result type, while the
                // RPython annotator derives arithmetic signedness solely from
                // operand annotations. This matters for checked Rust MIR:
                // `(usize, bool)` is scalarized to its `.0` value, but `len`
                // and an integer literal otherwise make `sub` infer a signed
                // SomeInteger. Re-express the result's Rust `usize` type as
                // the orthodox RPython `r_uint(value)` conversion. Its
                // extregistry annotation is
                // `SomeInteger(unsigned=True, knowntype=Ruint)`, which also
                // carries non-negativity by type (`unsigned or nonneg`).
                let module = HOST_ENV
                    .import_module("rpython.rlib.rarithmetic")
                    .ok_or_else(|| {
                        TyperError::message(
                            "rpython.rlib.rarithmetic missing from HOST_ENV bootstrap".to_string(),
                        )
                    })?;
                let callable_host = module.module_get("r_uint").ok_or_else(|| {
                    TyperError::message(
                        "rarithmetic.r_uint missing from HOST_ENV bootstrap".to_string(),
                    )
                })?;
                let scalar = Variable::new();
                Ok(vec![
                    FlowspaceOp::new(normalized, vec![l, r], Hlvalue::Variable(scalar.clone())),
                    FlowspaceOp::new(
                        "simple_call",
                        vec![
                            Hlvalue::Constant(Constant::new(ConstValue::HostObject(callable_host))),
                            Hlvalue::Variable(scalar),
                        ],
                        result,
                    ),
                ])
            } else {
                Ok(vec![FlowspaceOp::new(normalized, vec![l, r], result)])
            }
        }

        // ─── Pre-rtyper opname normalization for unary ops ───
        // `front::mir::unary_op_label` emits Rust-side
        // names (`neg`, `not`, `deref`).  RPython flowspace registers
        // unary operators via `add_operator('neg', 1, ..)` /
        // `add_operator('invert', 1, ..)` /
        // `add_operator('pos', 1, ..)` etc.
        // (`rpython/flowspace/operation.py`).  Translate the
        // pyre-side name to its RPython counterpart so the rtyper's
        // unary dispatch (`rtyper.rs` `translate_op_*`) finds
        // the proper `unary_rtype_*` arm.
        OpKind::UnaryOp {
            op: opname,
            operand,
            ..
        } => {
            let v = lookup_operand(value_map, operand, op, "operand")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                normalize_unary_op_name(opname)?,
                vec![v],
                result,
            )])
        }

        // ─── `isinstance` — RPython `space.isinstance(obj, cls)` ───
        // `flowspace/operation.py:259 isinstance → OpKind::IsInstance`.
        // Emitted pre-rtyper by `front/ast.rs` at `TupleStruct` /
        // composite match-cascade payload sites where a unit-variant
        // ptr_eq does not suffice. The rtyper dispatches `"isinstance"`
        // at `rtyper.rs`'s `translate_unary_operation` →
        // `InstanceRepr::rtype_isinstance`, which mints either a
        // per-class `ll_isinstance_const_*` helper (Constant
        // `class_carrier`) or the generic `ll_isinstance` helper
        // (Variable `class_carrier`).
        OpKind::IsInstance {
            obj, class_carrier, ..
        } => {
            let obj_hl = lookup_operand(value_map, obj, op, "obj")?;
            let cls_hl = lookup_operand(value_map, class_carrier, op, "class_carrier")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "isinstance",
                vec![obj_hl, cls_hl],
                result,
            )])
        }

        // ─── FieldRead / FieldWrite ports ───
        // RPython `flowspace/operation.py GetAttr.opname = 'getattr'`
        // and `setattr` (operation.py: same module). The high-level
        // attribute-access op carries the field name as a
        // `ConstValue::ByteStr` (Python 2 `str`), matching the rtyper's
        // `rtype_getattr` / `rtype_setattr` dispatch
        // (`rtyper.rs`'s `translate_operation`). InstanceRepr later lowers the
        // `getattr`/`setattr` op into a `getfield_*` / `setfield_*`
        // bytecode keyed on the field's lltype kind.
        OpKind::FieldRead { base, field, .. }
            if field.name == "length" && field.owner_root.as_deref() == Some("Utf8Str") =>
        {
            // `Utf8Str` carries the `rstr.py STR` layout, which the bookkeeper
            // annotates as `SomeString`: its length word is `len(s)`
            // (`LLHelpers.ll_strlen`), not a `getattr` the string repr has no
            // attribute for.
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("len", vec![base_hl], result)])
        }
        OpKind::FieldRead { base, field, .. } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            // `retarget_vec_operand` reuses this FieldRead's result as the
            // length word (`ValueType::Int`). A bare `getattr` of the Vec
            // field is the list pointer (`SomeList` → `Ptr(GcStruct)` →
            // GcRef). RPython `ll_length` (`lltypesystem/rlist.py`) is
            // `return l.length`, a Signed, via `ListRepr.rtype_len`.
            // `Box<[T]>` length is the metadata word. The result is that
            // count (`ll_length` / `rtype_len`), typed by `len` of the
            // field, not by `getattr` of the slice itself.
            if field.vec_part == Some(crate::model::VecFieldPart::FatLen) {
                let list = Hlvalue::Variable(Variable::new());
                return Ok(vec![
                    FlowspaceOp::new(
                        "getattr",
                        vec![
                            base_hl,
                            Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                        ],
                        list.clone(),
                    ),
                    FlowspaceOp::new("len", vec![list], result),
                ]);
            }
            if field.vec_part == Some(crate::model::VecFieldPart::Len) {
                // Inline: the read's base is the parent struct, so the list
                // is `getattr(parent, field)` and `len` is `ll_length`.
                // A pointer to the Vec is already the list (`ll_length(l)`).
                if field.inline_vec {
                    let list = Hlvalue::Variable(Variable::new());
                    return Ok(vec![
                        FlowspaceOp::new(
                            "getattr",
                            vec![
                                base_hl,
                                Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                            ],
                            list.clone(),
                        ),
                        FlowspaceOp::new("len", vec![list], result),
                    ]);
                }
                return Ok(vec![FlowspaceOp::new("len", vec![base_hl], result)]);
            }
            // `t.N` of a Rust tuple is `getitem(t, N)` on the `SomeTuple`
            // that `newtuple` built (`rtuple.py` `rtype_getitem`).
            if field
                .owner_root
                .as_deref()
                .is_some_and(majit_ir::descr::is_shaped_tuple_name)
                && let Some(index) = field
                    .name
                    .strip_prefix("__pos_")
                    .and_then(|n| n.parse::<i64>().ok())
            {
                return Ok(vec![FlowspaceOp::new(
                    "getitem",
                    vec![
                        base_hl,
                        Hlvalue::Constant(Constant::new(ConstValue::Int(index))),
                    ],
                    result,
                )]);
            }
            // A variant payload is not a field of the enum base. Narrow to
            // the variant class before `getattr`, so `__pos_0` keeps that
            // variant's type when another variant uses the same name.
            // Only an owner whose parent is an enum base; a struct path
            // also contains `::`.
            let variant_owner = field.owner_root.as_deref().filter(|root| {
                let Some((parent, _)) = root.rsplit_once("::") else {
                    return false;
                };
                call_registry
                    .bookkeeper()
                    .struct_fields
                    .borrow()
                    .as_ref()
                    .is_some_and(|reg| reg.is_enum_base(parent))
            });
            if let Some(owner) = variant_owner {
                let callable_host = HOST_ENV
                    .lookup_builtin(crate::runtime_names::shims::CAST_INSTANCE)
                    .ok_or_else(|| {
                        TyperError::message(
                            "__cast_instance_intrinsic missing from HOST_ENV bootstrap".to_string(),
                        )
                    })?;
                let narrowed = Hlvalue::Variable(Variable::new());
                return Ok(vec![
                    FlowspaceOp::new(
                        "simple_call",
                        vec![
                            Hlvalue::Constant(Constant::new(ConstValue::HostObject(callable_host))),
                            base_hl,
                            Hlvalue::Constant(Constant::new(ConstValue::byte_str(owner))),
                        ],
                        narrowed.clone(),
                    ),
                    FlowspaceOp::new(
                        "getattr",
                        vec![
                            narrowed,
                            Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                        ],
                        result,
                    ),
                ]);
            }
            Ok(vec![FlowspaceOp::new(
                "getattr",
                vec![
                    base_hl,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                ],
                result,
            )])
        }
        OpKind::FieldWrite {
            base, field, value, ..
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            // The stored value is an `AbstractValue` — a `Variable`
            // (resolve through the SSA map) or an inline `Constant`
            // (carries its own value), mirroring `Hlvalue` directly.
            let value_hl = match value {
                crate::model::LinkArg::Value(var) => lookup_operand(value_map, var, op, "value")?,
                crate::model::LinkArg::Const(c) => Hlvalue::Constant(c.clone()),
            };
            let result = resolve_result_hlvalue(op, value_map)?;
            let name = Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name)));
            let variant_owner = field.owner_root.as_deref().filter(|root| {
                let Some((parent, _)) = root.rsplit_once("::") else {
                    return false;
                };
                call_registry
                    .bookkeeper()
                    .struct_fields
                    .borrow()
                    .as_ref()
                    .is_some_and(|reg| reg.is_enum_base(parent))
            });
            if let Some(owner) = variant_owner {
                let callable_host = HOST_ENV
                    .lookup_builtin(crate::runtime_names::shims::CAST_INSTANCE)
                    .ok_or_else(|| {
                        TyperError::message(
                            "__cast_instance_intrinsic missing from HOST_ENV bootstrap".to_string(),
                        )
                    })?;
                let narrowed = Hlvalue::Variable(Variable::new());
                return Ok(vec![
                    FlowspaceOp::new(
                        "simple_call",
                        vec![
                            Hlvalue::Constant(Constant::new(ConstValue::HostObject(callable_host))),
                            base_hl,
                            Hlvalue::Constant(Constant::new(ConstValue::byte_str(owner))),
                        ],
                        narrowed.clone(),
                    ),
                    FlowspaceOp::new("setattr", vec![narrowed, name, value_hl], result),
                ]);
            }
            Ok(vec![FlowspaceOp::new(
                "setattr",
                vec![base_hl, name, value_hl],
                result,
            )])
        }

        // ─── ArrayRead / ArrayLen / ArrayWrite ports ───
        // RPython `flowspace/operation.py: GetItem.opname = 'getitem'`
        // and `setitem`. The base[index] form maps directly to
        // `getitem(base, index)` / `setitem(base, index, value)`; the
        // rtyper's `rtype_getitem` / `rtype_setitem` later route through
        // ListRepr / TupleRepr / Fixed-array repr based on the receiver's
        // resolved type, lowering to `getarrayitem_gc_*` /
        // `setarrayitem_gc_*` bytecodes.
        OpKind::ArrayRead { base, index, .. } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let index_hl = lookup_operand(value_map, index, op, "index")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "getitem",
                vec![base_hl, index_hl],
                result,
            )])
        }
        OpKind::ArrayLen { base, .. } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("len", vec![base_hl], result)])
        }
        OpKind::ArrayWrite {
            base, index, value, ..
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let index_hl = lookup_operand(value_map, index, op, "index")?;
            // The stored value is an `AbstractValue` — a `Variable`
            // (resolve through the SSA map) or an inline `Constant`.
            let value_hl = match value {
                crate::model::LinkArg::Value(var) => lookup_operand(value_map, var, op, "value")?,
                crate::model::LinkArg::Const(c) => Hlvalue::Constant(c.clone()),
            };
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "setitem",
                vec![base_hl, index_hl, value_hl],
                result,
            )])
        }

        // ─── InteriorFieldRead / InteriorFieldWrite ports ───
        // RPython `effectinfo.py add_interiorfield` notes that `getinteriorfield_gc`
        // implicitly carries both a `readarray` and a `readinteriorfield`
        // effect — the array-of-structs pattern is fundamentally a
        // chained `getitem(base, index) -> elem` followed by
        // `getattr(elem, field_name)` (or `setattr` for writes). Pyre's
        // legacy IR collapses these into a single `InteriorField*` op
        // for direct lowering convenience, but the rtyper sees the
        // chained form, so unfold here into two flowspace ops with an
        // intermediate `Variable` carrying the array element.
        OpKind::InteriorFieldRead {
            base, index, field, ..
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let index_hl = lookup_operand(value_map, index, op, "index")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            let elem_var = Hlvalue::Variable(Variable::new());
            Ok(vec![
                FlowspaceOp::new("getitem", vec![base_hl, index_hl], elem_var.clone()),
                FlowspaceOp::new(
                    "getattr",
                    vec![
                        elem_var,
                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                    ],
                    result,
                ),
            ])
        }
        OpKind::InteriorFieldWrite {
            base,
            index,
            field,
            value,
            ..
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let index_hl = lookup_operand(value_map, index, op, "index")?;
            let value_hl = lookup_operand(value_map, value, op, "value")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            let elem_var = Hlvalue::Variable(Variable::new());
            Ok(vec![
                FlowspaceOp::new("getitem", vec![base_hl, index_hl], elem_var.clone()),
                FlowspaceOp::new(
                    "setattr",
                    vec![
                        elem_var,
                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                        value_hl,
                    ],
                    result,
                ),
            ])
        }

        // ─── Call port (CallTarget per variant) ───
        // RPython `flowspace/operation.py SimpleCall.opname =
        // 'simple_call'`. The first arg is a Constant wrapping the
        // callable (or a Variable carrying a runtime function pointer).
        // Each `CallTarget` variant maps to a different shape:
        //
        //   FunctionPath { segments, .. }     — direct call to `path::func`.
        //                                    Wrap the joined qualname in
        //                                    a `HostObject::new_opaque(...)`
        //                                    Constant; rtyper's
        //                                    `rtype_simple_call` dispatches
        //                                    on the callable's resolved
        //                                    repr (PBCRepr / etc.).
        //   SyntheticTransparentCtor      — Rust struct ctor `Class { .. }`.
        //                                    Same shape as FunctionPath:
        //                                    opaque host wrapping the type
        //                                    qualname; the rtyper-equivalent
        //                                    layer routes the ctor to its
        //                                    InstanceRepr.
        //   Method { name, .. }           — `obj.method(args)` — chains
        //                                    `getattr(args[0], name) → meth`
        //                                    into `simple_call(meth, args[1..])`,
        //                                    mirroring `flowspace/
        //                                    flowcontext.py:LOAD_ATTR +
        //                                    CALL_FUNCTION` shape.
        //   Indirect { trait_root, name } — `dyn Trait` dispatch. Pyre's
        //                                    `rclass.rs` rewrites this into
        //                                    a `VtableMethodPtr` followed
        //                                    by an `IndirectCall`; reaching
        //                                    the adapter means rclass.rs
        //                                    didn't run, so fail-loud.
        //   UnsupportedExpr               — frontend coverage gap; fail-loud
        //                                    surfaces the missing
        //                                    `front::mir` arm.
        OpKind::Call { target, args, .. } => {
            use crate::model::CallTarget;
            let arg_hls: Result<Vec<_>, _> = args
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let role = format!("args[{i}]");
                    lookup_link_arg(value_map, v, op, &role)
                })
                .collect();
            let arg_hls = arg_hls?;
            let result = resolve_result_hlvalue(op, value_map)?;
            match target {
                // `FunctionPath` resolves through
                // `CallRegistry`, returning the registry entry's
                // `HostObject::UserFunction` instead of an opaque
                // wrapper. The rtyper's `pair_simple_call` then
                // short-circuits on `bookkeeper.descs` (pre-populated
                // by the registry) and routes through
                // `FunctionRepr::call(hop)` (`rpbc.py`).
                CallTarget::FunctionPath { segments, .. } => {
                    // RPython `SimpleCall.opname = 'simple_call'`
                    // (`flowspace/operation.py`); `SimpleCall.eval`
                    // reads `w_callable, args_w = self.args[0],
                    // self.args[1:]`.  `front::exc_from_raise` emits
                    // that mixed list directly (`const(exc_class)` at
                    // args[0]).  The path names the op, not a second
                    // callable — do not wrap it again.
                    //
                    // A call of an `ll_vec_*` helper is the `Vec` operation
                    // it lowers, seen at the high level so `RustVecRepr`
                    // chooses the helper again from the one table.
                    if let Some((op, kind)) =
                        majit_ir::rvec::vec_helper_for_path(&segments.join("::"))
                    {
                        return rust_vec_helper_ops(op, kind, arg_hls, result);
                    }
                    if segments.as_slice() == ["type"] {
                        if arg_hls.len() != 1 {
                            return Err(TyperError::message(
                                "translate_op: FunctionPath [\"type\"] requires \
                                 one operand (flowspace/operation.py Type)"
                                    .to_string(),
                            ));
                        }
                        return Ok(vec![FlowspaceOp::new("type", arg_hls, result)]);
                    }
                    if segments.as_slice() == ["simple_call"]
                        || segments.as_slice() == ["__dyn_call"]
                    {
                        // `__dyn_call` is the front's spelling of a call
                        // through a function pointer (`CallClass::Dynamic`
                        // whose operand is not a recovered vtable slot).
                        // The annotator sees a `simple_call` whose args[0]
                        // is that pointer; `PtrRepr.rtype_simple_call`
                        // then emits `indirect_call`.
                        if arg_hls.is_empty() {
                            return Err(TyperError::message(
                                "translate_op: FunctionPath [\"simple_call\"] / \
                                 [\"__dyn_call\"] requires args[0] as the callable \
                                 (flowspace/operation.py SimpleCall.eval)"
                                    .to_string(),
                            ));
                        }
                        return Ok(vec![FlowspaceOp::new("simple_call", arg_hls, result)]);
                    }
                    // RPython `VirtualizableInstanceRepr.hook_access_field`
                    // inserts the `jit_force_virtualizable` LLOp during
                    // rtyping; it never asks the annotator to traverse a
                    // runtime force-helper body.  Rust cannot attach the
                    // hook to a field repr, so pyre-interpreter spells the
                    // marker as an ordinary source call.  Reify that call as
                    // an ExtRegistry builtin before CallRegistry lookup: its
                    // `specialize_call` emits the identical LLOp and prevents
                    // a failed lift of the runtime `FORCE_FRAME_HOOK` chain
                    // from poisoning every descriptor gateway caller.
                    if segments.as_slice()
                        == [
                            "pyre_interpreter",
                            "executioncontext",
                            "jit_force_virtualizable",
                        ]
                    {
                        let callable = Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                            HostObject::new_builtin_callable(
                                "pyre_interpreter.executioncontext.jit_force_virtualizable",
                            ),
                        )));
                        let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                        call_args.push(callable);
                        call_args.extend(arg_hls);
                        return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                    }
                    // `core` method spellings of upstream operations:
                    // pyre source writes `a.min(b)` /
                    // `a != b`-via-`PartialEq::ne` / `v.len()` /
                    // `x.wrapping_mul(y)` where upstream Python
                    // writes `min(a, b)` / the flowspace ops `ne` /
                    // `len` / `mul` (flowspace/operation.py tables;
                    // lltype `int_mul` wraps, so `wrapping_mul` is
                    // the Rust spelling of upstream's plain `*` —
                    // e.g. the `x * 1000003` green-key hash).
                    // front::mir records the monomorphized core path
                    // when the call survives as a trait/inherent
                    // method call (primitive BinOps lower to
                    // `OpKind::BinOp` and never reach here); map the
                    // path back to the upstream spelling so the
                    // chain reaches unaryop.py / binaryop.py /
                    // rbuiltin.py instead of failing FunctionPath
                    // resolution.
                    // Pure bridges (cmp comparisons / slice len|iter /
                    // num wrapping_mul) → a like-named non-raising
                    // flowspace op.  `nonraising_core_bridge_opname` is
                    // the shared table `op_canraise` mirrors.
                    if let Some(opname) = nonraising_core_bridge_opname(segments, arg_hls.len()) {
                        return Ok(vec![FlowspaceOp::new(opname, arg_hls, result)]);
                    }
                    // `slice[..slice.len() - 1]` is emitted by
                    // `front::slice_index` as an unregistered marker rather
                    // than frontend `getslice`/`ConstInt` ops.  The marker is
                    // expanded here, after annotation, to the upstream
                    // `getslice(slice, 0, -1)` contract.  This mirrors the
                    // frontend/adapter split used by `__array_repeat`: an
                    // untyped graph falling to the legacy walker retains an
                    // ordinary residual call and cannot carry an unwired
                    // `getslice` op to the assembler.
                    //
                    // Upstream: `decompose_slice_args` (`rtyper.rs`)
                    // selects `SliceKind::MinusOne` for annotated constants
                    // `0` and `-1`, then `rlist.py:906-911` calls
                    // `ll_listslice_minusone` (asserting, not clamping).
                    if is_getslice_marker(segments, "__getslice_minusone") && arg_hls.len() == 1 {
                        let slice = arg_hls.into_iter().next().expect("slice arg");
                        return Ok(vec![FlowspaceOp::new(
                            "getslice",
                            vec![
                                slice,
                                Hlvalue::Constant(Constant::new(ConstValue::Int(0))),
                                Hlvalue::Constant(Constant::new(ConstValue::Int(-1))),
                            ],
                            result,
                        )]);
                    }
                    if is_getslice_marker(segments, "__getslice_rangeto") && arg_hls.len() == 2 {
                        // slice, end -> getslice(slice, 0, end)
                        let mut args = arg_hls.into_iter();
                        let slice = args.next().expect("slice arg");
                        let end = args.next().expect("RangeTo end arg");
                        return Ok(vec![FlowspaceOp::new(
                            "getslice",
                            vec![
                                slice,
                                Hlvalue::Constant(Constant::new(ConstValue::Int(0))),
                                end,
                            ],
                            result,
                        )]);
                    }
                    if is_getslice_marker(segments, "__getslice_rangefrom") && arg_hls.len() == 2 {
                        // slice, start -> getslice(slice, start, None).
                        // RPython `decompose_slice_args` selects StartOnly;
                        // the frontend records only SliceIndex<usize>
                        // consumers, which prove the runtime start nonnegative.
                        let mut args = arg_hls.into_iter();
                        let slice = args.next().expect("slice arg");
                        let start = args.next().expect("RangeFrom start arg");
                        return Ok(vec![FlowspaceOp::new(
                            "getslice",
                            vec![
                                slice,
                                start,
                                Hlvalue::Constant(Constant::new(ConstValue::None)),
                            ],
                            result,
                        )]);
                    }
                    if is_getslice_marker(segments, "__getslice_range") && arg_hls.len() == 3 {
                        // slice, start, end -> getslice(slice, start, end)
                        // This is the direct RPython flow-space shape for the
                        // interpreter sites ported from `buffer[start:end]`.
                        return Ok(vec![FlowspaceOp::new("getslice", arg_hls, result)]);
                    }
                    // Rust `Wtf8::as_bytes()[i]` is the byte-string operation
                    // PyPy writes as `ord(utf8[i])` throughout rutf8.py. The
                    // string repr's `getitem` yields SomeChar; preserve that
                    // intermediate and let the ordinary `ord` rtyper turn it
                    // into Signed.
                    if segments.as_slice() == ["__string_byte_getitem"] && arg_hls.len() == 2 {
                        let mut args = arg_hls.into_iter();
                        let string = args.next().expect("string byte-view arg");
                        let index = args.next().expect("string byte index arg");
                        let byte = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new("getitem", vec![string, index], byte.clone()),
                            FlowspaceOp::new("ord", vec![byte], result),
                        ]);
                    }
                    // `[v; N]` array literal.  `front::mir` lowers
                    // `Rvalue::Repeat` to the `__array_repeat` synthetic
                    // carrying `(fill, count)` whenever the count decodes.
                    // Expand it into the `[a] * b` shape upstream builds —
                    // `newlist(a)` then `mul(list, b)` — which
                    // `transform_allocate` (transform.py) collapses
                    // into `alloc_and_set(b, a)`, the operation
                    // `rtype_alloc_and_set` (rlist.py) lowers
                    // through `ll_alloc_and_set` (rlist.py) to
                    // `_ll_alloc_and_clear` / `_ll_alloc_and_set_nonnull`.
                    //
                    // The list must be `mul`'s arg 0: `transform_allocate`
                    // matches on `op.args[0] in length1_lists` and reads the
                    // count back out of `op.args[1]`, which is the order
                    // `rtype_alloc_and_set` then consumes as
                    // `hop.inputargs(Signed, r_list.item_repr)`.  Upstream's
                    // `pairtype(SomeInteger, SomeList).mul` (binaryop.py)
                    // exists but `transform_allocate` does not match it.
                    // Both ops go out together so they share a block, which
                    // is the scope `transform_allocate` scans.
                    //
                    // Emitting here rather than in `front::mir` is
                    // deliberate: this adapter is only reached on the way to
                    // the rtyper, so neither op can survive on a graph that
                    // fell back to the legacy walker.
                    if segments.len() == 1 && segments[0] == "__array_repeat" && arg_hls.len() == 2
                    {
                        let mut iter = arg_hls.into_iter();
                        let fill = iter.next().expect("array repeat fill arg");
                        let count = iter.next().expect("array repeat count arg");
                        let list = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new("newlist", vec![fill], list.clone()),
                            FlowspaceOp::new("mul", vec![list, count], result),
                        ]);
                    }
                    // `[__iter_next]` (`front::iter_next`) → the raising
                    // flowspace `next` op (`operation.rs` `OpKind::Next`,
                    // `can_only_throw = [StopIteration, RuntimeError]`).  The
                    // block's `LastException` exits — set by the front-end
                    // `next`-diamond rewrite — carry the exception edges, so
                    // no edge-building happens here (unlike a `?`-tail op).
                    if crate::front::iter_next::is_iter_next_segments(segments)
                        && arg_hls.len() == 1
                    {
                        return Ok(vec![FlowspaceOp::new("next", arg_hls, result)]);
                    }
                    // `min`/`max` are the exception: they lower to a
                    // `simple_call(<builtin>)` (raising like any builtin
                    // call), so they stay out of the pure-bridge table.
                    if segments.len() >= 3
                        && segments[0] == "core"
                        && segments[1] == "cmp"
                        && arg_hls.len() == 2
                    {
                        let leaf = segments[segments.len() - 1].as_str();
                        if matches!(leaf, "min" | "max") {
                            let builtin = HOST_ENV.lookup_builtin(leaf).ok_or_else(|| {
                                TyperError::message(format!(
                                    "builtin `{leaf}` missing from HOST_ENV"
                                ))
                            })?;
                            let callable =
                                Hlvalue::Constant(Constant::new(ConstValue::HostObject(builtin)));
                            let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                            call_args.push(callable);
                            call_args.extend(arg_hls);
                            return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                        }
                    }
                    // `__cast_pointer` marker (`front::mir`
                    // `cast_pointer_call`) — pyre's carrier for the
                    // upstream `cast_pointer(PTRTYPE, ptr)` downcast
                    // (lltype.py).  The root is a trailing ByteStr
                    // Constant; intern it by qualname so every cast
                    // site shares one `HostObject` Arc (`getdesc`
                    // dedups on Arc identity).
                    if segments.as_slice() == ["__cast_pointer"] {
                        if arg_hls.len() != 2 {
                            return Err(TyperError::message(format!(
                                "__cast_pointer requires (operand, constant_root), got {}",
                                arg_hls.len()
                            )));
                        }
                        let callable_host = HOST_ENV
                            .import_module("rpython.rtyper.lltypesystem.lltype")
                            .and_then(|m| m.module_get("cast_pointer"))
                            .ok_or_else(|| {
                                TyperError::message(
                                    "HOST_ENV lltype module must expose cast_pointer".to_string(),
                                )
                            })?;
                        let root = match &arg_hls[1] {
                            Hlvalue::Constant(c) => c.value.as_pystr().ok_or_else(|| {
                                TyperError::message(
                                    "__cast_pointer root must be a constant string".to_string(),
                                )
                            })?,
                            _ => {
                                return Err(TyperError::message(
                                    "__cast_pointer root must be a Constant".to_string(),
                                ));
                            }
                        };
                        let class_host = call_registry.bookkeeper().intern_class_by_qualname(root);
                        let mut call_args = Vec::with_capacity(3);
                        call_args.push(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                            callable_host,
                        ))));
                        call_args.push(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                            class_host,
                        ))));
                        call_args.push(arg_hls[0].clone());
                        return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                    }
                    // `__cast_instance_intrinsic` — front-end pointer-downcast
                    // narrow (#298).  `front::mir` emits
                    // `Call(["__cast_instance_intrinsic"], [operand, const(root)])`
                    // for `obj as *const RegisteredStruct`.  The analyzer
                    // reads the trailing `ByteStr` root to type the result
                    // `SomeInstance(root)`, and the typer lowers the call
                    // to a `cast_pointer`.  The callable resolves through
                    // the `__cast_instance_intrinsic` HOST_ENV singleton so its
                    // Arc identity matches the `BUILTIN_TYPER` key.
                    if segments.as_slice() == [crate::runtime_names::shims::CAST_INSTANCE] {
                        if arg_hls.len() != 2 {
                            return Err(TyperError::message(format!(
                                "__cast_instance_intrinsic requires (operand, constant_root), got {}",
                                arg_hls.len()
                            )));
                        }
                        let callable_host = HOST_ENV
                            .lookup_builtin(crate::runtime_names::shims::CAST_INSTANCE)
                            .ok_or_else(|| {
                                TyperError::message(
                                    "__cast_instance_intrinsic missing from HOST_ENV bootstrap"
                                        .to_string(),
                                )
                            })?;
                        let callable =
                            Hlvalue::Constant(Constant::new(ConstValue::HostObject(callable_host)));
                        let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                        call_args.push(callable);
                        call_args.extend(arg_hls);
                        return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                    }
                    // `__cast_address_intrinsic` — the erasing twin of the narrow
                    // above (`front::mir` emits it for `p as *mut u8`).  It
                    // carries no root, so the reconstruction is the plain
                    // `simple_call(callable, operand)`; resolving through the
                    // HOST_ENV singleton keeps the Arc identity that keys its
                    // `BUILTIN_TYPER` entry, which a single-segment path
                    // consulting the `PyreCallRegistry` first would lose.
                    if segments.len() == 1
                        && segments[0] == crate::runtime_names::shims::CAST_ADDRESS
                    {
                        if arg_hls.len() != 1 {
                            return Err(TyperError::message(format!(
                                "__cast_address_intrinsic requires exactly one operand, got {}",
                                arg_hls.len()
                            )));
                        }
                        let callable_host = HOST_ENV
                            .lookup_builtin(crate::runtime_names::shims::CAST_ADDRESS)
                            .ok_or_else(|| {
                                TyperError::message(
                                    "__cast_address_intrinsic missing from HOST_ENV bootstrap"
                                        .to_string(),
                                )
                            })?;
                        let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                        call_args.push(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                            callable_host,
                        ))));
                        call_args.extend(arg_hls);
                        return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                    }
                    // Exact RPython `rgc.ll_arraymove` call reconstructed by
                    // the MIR list-storage adapter from Rust `ptr::copy`.
                    if segments.len() == 1
                        && segments[0] == crate::runtime_names::shims::LL_ARRAYMOVE
                    {
                        if arg_hls.len() != 4 {
                            return Err(TyperError::message(format!(
                                "__majit_ll_arraymove requires exactly four args, got {}",
                                arg_hls.len()
                            )));
                        }
                        let callable_host = HOST_ENV
                            .lookup_builtin(crate::runtime_names::shims::LL_ARRAYMOVE)
                            .ok_or_else(|| {
                                TyperError::message(
                                    "__majit_ll_arraymove missing from HOST_ENV bootstrap"
                                        .to_string(),
                                )
                            })?;
                        let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                        call_args.push(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                            callable_host,
                        ))));
                        call_args.extend(arg_hls);
                        return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                    }
                    // `front::range_iter`'s synthesized `range(start, stop)`
                    // (the exclusive int-`Range` for-loop divert).  It carries
                    // the reserved `__majit_range` spelling rather than a bare
                    // `["range"]` because the registry ladder below resolves a
                    // single-segment path against the `CallRegistry`
                    // *first* (the `frame.globals`-before-`__builtin__` order
                    // of `flowcontext.py find_global`), and pyre's registry is one
                    // flat namespace keyed by leaf-only `CallPath`s.  A pyre
                    // free function whose leaf happens to be a builtin name —
                    // `pyre_interpreter::module::_ast::convert::range`
                    // registers as `["range"]` — would therefore bind this
                    // synthesized call to *its* `HostObject`, and since
                    // `BUILTIN_TYPER` is keyed by Arc identity
                    // (`rbuiltin.rs`) the rtyper then fails
                    // `findbltintyper` with "don't know about built-in
                    // function <host range>".  Resolving the reserved
                    // spelling straight to the `HOST_ENV` singleton keeps the
                    // Arc identity that keys `rtype_builtin_range`
                    // (`rrange.py:96-126`), the same discipline the
                    // `__cast_instance_intrinsic` arm above documents.
                    // `__majit_enumerate(lst)` is the front's spelling of
                    // `enumerate(lst)` and resolves the same way.
                    let builtin = match segments.as_slice() {
                        [leaf] if leaf == crate::runtime_names::shims::RANGE => Some("range"),
                        [leaf] if leaf == crate::runtime_names::shims::ENUMERATE => {
                            Some("enumerate")
                        }
                        _ => None,
                    };
                    if let Some(builtin) = builtin {
                        let callable_host = HOST_ENV.lookup_builtin(builtin).ok_or_else(|| {
                            TyperError::message(format!(
                                "{builtin} missing from HOST_ENV bootstrap"
                            ))
                        })?;
                        let callable =
                            Hlvalue::Constant(Constant::new(ConstValue::HostObject(callable_host)));
                        let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                        call_args.push(callable);
                        call_args.extend(arg_hls);
                        return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                    }
                    // `front::mir`'s builder-mode rewrite emits three reserved
                    // single-segment markers for a *lifted* string-accumulator
                    // (pass 2 of the two-pass lift gate).  A dropped graph
                    // keeps the concat form and never reaches this lift-only
                    // adapter, so the markers only ever appear where
                    // `SomeStringBuilder` is fully typed.  Each lowers to the
                    // flowspace shape the annotator's `find_method` /
                    // `newstringbuilder` arms already consume, mirroring the
                    // `Vec::push` / `slice.reverse` method arms below.
                    //
                    // `__majit_stringbuilder_new` → `newstringbuilder`, forwarding
                    // the ctor operands verbatim: `new()` carries none and
                    // `with_capacity(n)` carries the size `n`.  The operand count
                    // mirrors `len(hop.args_v)` in
                    // `AbstractStringBuilderRepr.rtyper_new` (rtyper/rbuilder.py),
                    // which `rtype_newstringbuilder` honours — no arg selects
                    // `ll_new(INIT_SIZE)`, the size arg threads `ll_new(n)`.
                    if segments.len() == 1
                        && segments[0] == crate::runtime_names::shims::STRINGBUILDER_NEW
                    {
                        return Ok(vec![FlowspaceOp::new("newstringbuilder", arg_hls, result)]);
                    }
                    // `__majit_stringbuilder_append(recv, piece)` →
                    // `getattr(recv, "append") + simple_call(bound, piece)`,
                    // reaching `StringBuilderRepr::rtype_method("append")`
                    // (rstring.py) — identical to the `Vec::push` arm.
                    if segments.len() == 1
                        && segments[0] == crate::runtime_names::shims::STRINGBUILDER_APPEND
                    {
                        if arg_hls.len() != 2 {
                            return Err(TyperError::message(format!(
                                "__majit_stringbuilder_append requires exactly two args \
                                 (receiver, piece), got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message(
                                "__majit_stringbuilder_append requires a receiver arg".to_string(),
                            )
                        })?;
                        let piece = iter.next().ok_or_else(|| {
                            TyperError::message(
                                "__majit_stringbuilder_append requires a piece arg".to_string(),
                            )
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver,
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                        "append",
                                    ))),
                                ],
                                bound_method.clone(),
                            ),
                            FlowspaceOp::new("simple_call", vec![bound_method, piece], result),
                        ]);
                    }
                    // `__majit_stringbuilder_build(recv)` →
                    // `getattr(recv, "build") + simple_call(bound)`, reaching
                    // `StringBuilderRepr::rtype_method("build")`
                    // (rstring.py) → `SomeString` — identical to the
                    // `slice.reverse` arm (no extra args on the bound method).
                    if segments.len() == 1
                        && segments[0] == crate::runtime_names::shims::STRINGBUILDER_BUILD
                    {
                        if arg_hls.len() != 1 {
                            return Err(TyperError::message(format!(
                                "__majit_stringbuilder_build requires exactly one receiver arg, \
                                 got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message(
                                "__majit_stringbuilder_build requires a receiver arg".to_string(),
                            )
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver,
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str("build"))),
                                ],
                                bound_method.clone(),
                            ),
                            FlowspaceOp::new("simple_call", vec![bound_method], result),
                        ]);
                    }
                    // The `len` operation in its three spellings: the
                    // `__len` synthetic `front/mir.rs` lowers `Rvalue::Len`
                    // (and the `<str>::is_empty` decomposition) to; the
                    // slice-receiver `core::slice::<Impl>::len`; and the
                    // `<str>::len` method.  Rust lowers `slice.len()` /
                    // `s.len()` to MIR calls to those intrinsics, which
                    // have no source body to register.  Route all three to
                    // the rtyper's `len` operation (`rtyper.rs "len"
                    // arm` → `Repr.rtype_len`), the same dispatch upstream
                    // `op.len(v)` reaches via `unaryop.py`.  The
                    // rtyper dispatches on the receiver repr: a slice maps
                    // to `SomeList` (`ll_length`), a `&str` to `SomeString`
                    // (`StringRepr.rtype_len` → `ll_strlen`).  The helper
                    // is registered as an opname graph and lowered to the
                    // `strlen`/`arraylen_gc` blackhole op
                    // (`codewriter::jtransform_opname::lower_graph`), so
                    // these are real `len` ops, not symbolic residuals.
                    //
                    // `__strlen` is a fourth frontend spelling
                    // (`front/mir.rs` plants it for `Rvalue::Len` /
                    // `slice::len` / `Wtf8::len` on a
                    // `string_byte_view_locals` place).  A receiver that
                    // is already the rstr takes this `len` op
                    // (`StringRepr.rtype_len` → `ll_strlen`, `rstr.py`).
                    // A `Wtf8`/`Str` cast of a wrapper is not that rstr:
                    // [`rewrite_wtf8_view_strlen`] turns it into
                    // `getattr` of the wrapper's one string field
                    // (`W_UnicodeObject._utf8`, `len(w._utf8)`) before
                    // this arm runs.  `ll_strlen` on the wrapper would
                    // read `PyObject.w_class` at offset 8.
                    let is_len_op = (segments.len() == 1
                        && (segments[0] == "__len" || segments[0] == "__strlen"))
                        || (segments.len() == 4
                            && segments[0] == "core"
                            && segments[1] == "slice"
                            && segments[2] == "<Impl>"
                            && segments[3] == "len")
                        || (segments.len() >= 3
                            && segments[segments.len() - 3] == "str"
                            && segments[segments.len() - 2] == "<Impl>"
                            && segments[segments.len() - 1] == "len");
                    if is_len_op {
                        if arg_hls.len() != 1 {
                            return Err(TyperError::message(format!(
                                "len operation requires exactly one receiver arg, got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let arg = iter.next().ok_or_else(|| {
                            TyperError::message("len operation requires a receiver arg".to_string())
                        })?;
                        return Ok(vec![FlowspaceOp::new("len", vec![arg], result)]);
                    }
                    // `slice.reverse()` lowers (in Rust MIR) to a call to
                    // `core::slice::<Impl>::reverse`, arriving as the
                    // FunctionPath with no source body to register.  Unlike
                    // `len`/`iter`, `reverse` has no rtyper operator arm, so
                    // emit the *method* shape — `getattr(recv, "reverse")` +
                    // `simple_call(bound_method)` — that the annotator's
                    // `find_method("reverse")` (`unaryop.rs`) and
                    // `BuiltinMethodRepr::rtype_simple_call` →
                    // `rtype_method("reverse")` consume, identical to the
                    // `CallTarget::Method` arm below.  `rtype_method_reverse`
                    // (`rlist.py`) `gendirectcall`s `ll_reverse`
                    // (`rlist.py:677-686`).
                    if is_slice_reverse_segments(segments) {
                        if arg_hls.len() != 1 {
                            return Err(TyperError::message(format!(
                                "slice.reverse requires exactly one receiver arg, got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message("slice.reverse requires a receiver arg".to_string())
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver,
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                        "reverse",
                                    ))),
                                ],
                                bound_method.clone(),
                            ),
                            FlowspaceOp::new("simple_call", vec![bound_method], result),
                        ]);
                    }
                    // `Vec::with_capacity(n)` / `Vec::new()` lower (in Rust
                    // MIR) to a call to `vec::Vec::{with_capacity,new}`.  Emit
                    // the `newlist` operation with no element args, which
                    // `rtype_newlist` (`rlist.py`) lowers through
                    // `newlist(llops, r_list, [])`.  The capacity operand is
                    // dropped — see [`is_vec_ctor_segments`].
                    if is_vec_ctor_segments(segments) {
                        return Ok(vec![FlowspaceOp::new("newlist", vec![], result)]);
                    }
                    // `try_pyobject_vec_with_capacity(sizehint)?` is the
                    // fallible Rust spelling of
                    // `objectmodel.newlist_hint(sizehint)`. The
                    // Result-to-exception pass has already made this call's
                    // result the normal-edge Vec payload, so the synthetic op
                    // has precisely the upstream one-argument/result shape.
                    if is_runtime_newlist_hint_segments(segments) {
                        if arg_hls.len() != 1 {
                            return Err(TyperError::message(format!(
                                "try_pyobject_vec_with_capacity requires exactly one size hint, got {}",
                                arg_hls.len()
                            )));
                        }
                        return Ok(vec![FlowspaceOp::new("newlist_hint", arg_hls, result)]);
                    }
                    // `vec![elem; count]` lowers (in Rust MIR) to a call to
                    // `alloc::vec::from_elem(elem, count)`. Emit the same
                    // `newlist(elem)` + `mul(list, count)` shape as the
                    // `__array_repeat` synthetic above; see
                    // [`is_vec_from_elem_segments`] for the upstream
                    // `transform_allocate` / `rtype_alloc_and_set` path.
                    if is_vec_from_elem_segments(segments, arg_hls.len()) {
                        let mut iter = arg_hls.into_iter();
                        let fill = iter.next().expect("from_elem fill arg");
                        let count = iter.next().expect("from_elem count arg");
                        let list = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new("newlist", vec![fill], list.clone()),
                            FlowspaceOp::new("mul", vec![list, count], result),
                        ]);
                    }
                    // `Vec::push(recv, item)` lowers (in Rust MIR) to a call
                    // to `vec::Vec::push`. Emit the *method* shape
                    // `getattr(recv, "append") + simple_call(bound_method,
                    // item)` that the annotator's `find_method("append")`
                    // (`unaryop.rs`) and `BuiltinMethodRepr::rtype_simple_call`
                    // → `ListRepr::rtype_method("append")` consume —
                    // identical to the `slice.reverse` arm above, but the
                    // bound method takes the appended `item` arg.
                    if is_vec_push_segments(segments) {
                        if arg_hls.len() != 2 {
                            return Err(TyperError::message(format!(
                                "Vec::push requires exactly two args (receiver, item), got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message("Vec::push requires a receiver arg".to_string())
                        })?;
                        let item = iter.next().ok_or_else(|| {
                            TyperError::message("Vec::push requires an item arg".to_string())
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver,
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                        "append",
                                    ))),
                                ],
                                bound_method.clone(),
                            ),
                            FlowspaceOp::new("simple_call", vec![bound_method, item], result),
                        ]);
                    }
                    // `Vec::extend_from_slice(recv, slice)` — same method
                    // shape as the `Vec::push` arm, but the bound method is
                    // `extend` and its arg is the source slice (a non-resized
                    // list). Reaches `ListRepr::rtype_method("extend")`.
                    if is_vec_extend_from_slice_segments(segments) {
                        if arg_hls.len() != 2 {
                            return Err(TyperError::message(format!(
                                "Vec::extend_from_slice requires exactly two args \
                                 (receiver, slice), got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message(
                                "Vec::extend_from_slice requires a receiver arg".to_string(),
                            )
                        })?;
                        let source = iter.next().ok_or_else(|| {
                            TyperError::message(
                                "Vec::extend_from_slice requires a source-slice arg".to_string(),
                            )
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver,
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                        "extend",
                                    ))),
                                ],
                                bound_method.clone(),
                            ),
                            FlowspaceOp::new("simple_call", vec![bound_method, source], result),
                        ]);
                    }
                    // `RDict::get(recv, key)` is `dict.get`
                    // (`OrderedDictRepr.rtype_method_get` → `ll_dict_get`
                    // → `ll_dict_lookup`). Two arguments: the missing key
                    // answers the `None` default that method pushes.
                    // `RDict::insert(recv, key, value)` returns the previous
                    // value (`Option`), while `ll_dict_setitem` is void
                    // (`rtype_setitem`). Read with `get` into the call's
                    // result, then `setitem` (`ll_dict_setitem`). `get`
                    // runs first so the displaced value is the one insert
                    // returned.
                    if let Some(leaf) = rdict_insert_or_get_leaf(segments) {
                        if leaf == "get" {
                            if arg_hls.len() != 2 {
                                return Err(TyperError::message(format!(
                                    "RDict::get requires exactly two args (receiver, key), got {}",
                                    arg_hls.len()
                                )));
                            }
                            let mut iter = arg_hls.into_iter();
                            let receiver = iter.next().ok_or_else(|| {
                                TyperError::message(
                                    "RDict::get requires a receiver arg".to_string(),
                                )
                            })?;
                            let key = iter.next().ok_or_else(|| {
                                TyperError::message("RDict::get requires a key arg".to_string())
                            })?;
                            let bound_method = Hlvalue::Variable(Variable::new());
                            return Ok(vec![
                                FlowspaceOp::new(
                                    "getattr",
                                    vec![
                                        receiver,
                                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                            "get",
                                        ))),
                                    ],
                                    bound_method.clone(),
                                ),
                                FlowspaceOp::new("simple_call", vec![bound_method, key], result),
                            ]);
                        }
                        if arg_hls.len() != 3 {
                            return Err(TyperError::message(format!(
                                "RDict::insert requires exactly three args \
                                 (receiver, key, value), got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message("RDict::insert requires a receiver arg".to_string())
                        })?;
                        let key = iter.next().ok_or_else(|| {
                            TyperError::message("RDict::insert requires a key arg".to_string())
                        })?;
                        let value = iter.next().ok_or_else(|| {
                            TyperError::message("RDict::insert requires a value arg".to_string())
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        let stored = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver.clone(),
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str("get"))),
                                ],
                                bound_method.clone(),
                            ),
                            FlowspaceOp::new(
                                "simple_call",
                                vec![bound_method, key.clone()],
                                result,
                            ),
                            FlowspaceOp::new("setitem", vec![receiver, key, value], stored),
                        ]);
                    }
                    // Fail-closed on an UNFUSED
                    // `lltype::malloc[_typed/_managed/_stable]`. A GC
                    // struct gets its `NewWithVtable` lowering before the rtyper
                    // runs only where `fuse_boxing_alloc` (`model.rs`) could
                    // resolve the cluster's header: a constant `ob_type`
                    // type-pointer whose `w_class` is `get_instantiate` of that
                    // same type.  A cluster it declined — no `ob_header` field at
                    // all, a type-pointer that is not a constant, or a `w_class`
                    // naming a different type, which is a subclass construction
                    // (`w_long_from_raw` pairs `&LONG_TYPE` with `&INT_TYPE`) —
                    // survives to here as a call, with no ported
                    // `jtransform.rewrite_op_malloc` general path
                    // (`jtransform.py`).  Layer-1 misses it (cutover skips
                    // registering `malloc_typed`), so it would resolve to the
                    // HOST_ENV builtin at Layer-3b and match a residual `simple_call`
                    // carrying a symbolic fnaddr the executor cannot run.  Reject it
                    // with a `TyperError` classified in `is_known_unported` so the
                    // graph census-Skips to the legacy walker instead of silently
                    // matching a wrong residual call.  See the companion note at
                    // `cutover::populate_call_registry_from_call_graphs` (the
                    // `malloc_typed` registration skip).
                    if segments.len() >= 2
                        && segments[segments.len() - 2] == "lltype"
                        && matches!(
                            segments[segments.len() - 1].as_str(),
                            "malloc"
                                | "malloc_typed"
                                | "malloc_typed_managed"
                                | "malloc_typed_stable"
                        )
                    {
                        return Err(TyperError::message(
                            "`lltype::malloc[_typed/_managed/_stable]` survived fuse_boxing_alloc unfused; \
                             the cluster's header did not resolve to a constant type \
                             pointer standing for its own `w_class`, so no safe \
                             malloc->new lowering could be proven"
                                .to_string(),
                        ));
                    }
                    let key =
                        crate::translator::rtyper::call_registry::FunctionPathKey::from_segments(
                            segments.iter().cloned(),
                        );
                    // Method-routing for a registered dispatch family
                    // (issue #346).  `front::mir`'s `CallKind::Trait` arm
                    // lowers `<Trait>::method(receiver, …)` to a
                    // `FunctionPath [<trait leaf>, method]`.  The
                    // direct-path binder (`lib.rs`) registers only trait
                    // default bodies and single-impl devirtualizations; a
                    // REQUIRED method with `>=2` impls is intentionally
                    // left unregistered ("keep failing loud until
                    // receiver-driven resolution lands").  When the trait
                    // is an opted-in family and the receiver carries the
                    // family base `ClassDef` (seeded by
                    // `derive_subject_inputcells`), lower the call as a
                    // `getattr` + `simple_call` so the attrfamily merge
                    // resolves the impl `MethodDesc` family — exactly as
                    // `CallTarget::Method`.  Gated on an exact registry
                    // miss so default bodies and single-impl devirt (both
                    // registered) keep their direct FunctionPath.
                    if segments.len() == 2
                        && call_registry.lookup(&key).is_none()
                        && call_registry
                            .bookkeeper()
                            .is_registered_trait_family_leaf(&segments[0])
                    {
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message(format!(
                                "translate_op: family method call {segments:?} \
                                 has no receiver arg (receiver must be args[0])"
                            ))
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        let mut call_args = Vec::with_capacity(iter.size_hint().0 + 1);
                        call_args.push(bound_method.clone());
                        call_args.extend(iter);
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver,
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                        &segments[1],
                                    ))),
                                ],
                                bound_method,
                            ),
                            FlowspaceOp::new("simple_call", call_args, result),
                        ]);
                    }
                    // Three resolution layers, matching upstream's three
                    // dispatch shapes for a dotted call site:
                    //
                    // 1. `CallRegistry` — user functions registered
                    //    by the production builder.  Analogous to
                    //    `flowspace/flowcontext.py:LOAD_GLOBAL` reading
                    //    `frame.globals` (user globals) first.
                    //
                    // 2. Single-segment HOST_ENV builtin
                    //    (`HOST_ENV.lookup_builtin(name)`) — analogous
                    //    to `flowcontext.py getattr(__builtin__,
                    //    varname)`, the second stage of
                    //    `find_global` (`flowcontext.py`).
                    //
                    // 3. Multi-segment HOST_ENV module attribute
                    //    (`HOST_ENV.import_module(prefix).module_get(\
                    //    attr)`) — analogous to the bytecode chain
                    //    `LOAD_GLOBAL <module>` (resolving via
                    //    `find_global`) followed by `LOAD_ATTR <attr>`
                    //    (`flowcontext.py`).  Pyre folds the
                    //    chain into a single `FunctionPath` carrier
                    //    because the frontend already records the
                    //    fully-qualified dotted path; the host
                    //    resolution still respects the same scope
                    //    boundary.
                    //
                    // Layer 3 — host module attribute lookup.  Upstream
                    // `LOAD_GLOBAL <module>` + `LOAD_ATTR <attr>` chain
                    // (`flowcontext.py`) consults the caller's
                    // per-function `frame.globals` first, falling back
                    // to `builtins` if absent.  Resolution order:
                    //
                    // 3b. `segments` already
                    //     spell out the fully-qualified Rust path
                    //     (`rpython::rtyper::lltypesystem::lltype::
                    //     cast_ptr_to_int`) without a matching `use`
                    //     statement — Rust compiles such paths
                    //     directly, so pyre source frequently writes
                    //     them inline.  Upstream has no exact analog
                    //     (Python source uniformly imports before
                    //     calling), so the HOST_ENV fallback
                    //     stays in place for existing pyre callsites.
                    //     A production callsite relies on this branch
                    //     (a tracing-time `OpKind::Call::FunctionPath`
                    //     with segments spelling a curated HOST_ENV
                    //     module), so removing the branch fails the
                    //     strict gate (cranelift fib_recursive +
                    //     fannkuch TIMEOUT).
                    //
                    // 3c. Unknown prefix — `TyperError` (caller must
                    //     register the path or import the prefix).
                    let callable_host = if let Some(entry) = call_registry.lookup(&key) {
                        entry.host_object.clone()
                    } else if let Some(host) = flowspace_model::host_env_callable(segments) {
                        // Branch 3b — builtin leaf or fully-qualified inline
                        // path. The PRE-EXISTING-ADAPTATION is item `3b.` of
                        // this arm's Layer 3 resolution-order list.
                        host
                    } else if let Some(entry) = call_registry.lookup_with_leaf_match(&key) {
                        // Fuzzy leaf-match is the last registry fallback.
                        // Exact registry entries and HOST_ENV
                        // module paths must win first so external
                        // stubs such as `BigInt::from`, `Vec::new`, and
                        // `Box::new` cannot be captured by an unrelated
                        // user function with the same leaf.
                        //
                        // This is the resting point for the former
                        // caller-scoped `use`-import resolution: MIR callsites
                        // carry the fully-qualified crate-relative path, so the
                        // exact `lookup(&key)` above is the primary binder and
                        // no per-caller import scope is needed to disambiguate.
                        // The leaf-scan itself is wrong-bind-safe —
                        // `lookup_with_leaf_match` returns `Some` only when the
                        // same-leaf matches converge on a single `host_object`
                        // identity, otherwise `None` falls through to the hard
                        // error below.
                        entry.host_object.clone()
                    } else {
                        return Err(TyperError::message(format!(
                            "translate_op: OpKind::Call::FunctionPath {{ segments: {:?} }} \
                             not registered in CallRegistry, not in HOST_ENV \
                             `__builtin__`, and not a known module-qualified host attribute — \
                             the production builder (a SemanticProgram walker, or a test \
                             fixture building the registry directly) must register the path \
                             with its parameter Signature before specialize_legacy_graph \
                             consults the rtyper. Result slot = {}",
                            segments,
                            fmt_op_result(op),
                        )));
                    };
                    let callable =
                        Hlvalue::Constant(Constant::new(ConstValue::HostObject(callable_host)));
                    let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                    call_args.push(callable);
                    call_args.extend(arg_hls);
                    Ok(vec![FlowspaceOp::new("simple_call", call_args, result)])
                }
                CallTarget::SyntheticTransparentCtor {
                    name, owner_path, ..
                } => {
                    // RPython parity: tagged-union ctor `Foo(x)` annotates as
                    // `SomePBC([ClassDesc(Foo)])` then `pair_simple_call`
                    // constructs `SomeInstance(classdef)` (`bookkeeper.py:
                    // 315-316`).  Wrapping the ctor name as
                    // `HostObject::new_class(qualname, [])` routes through
                    // the existing `is_class()` arm in
                    // [`crate::annotator::bookkeeper::Bookkeeper::immutablevalue_hostobject`]
                    // (`bookkeeper.rs`) → `getdesc` → `ClassDesc::new`
                    // (`classdesc.rs`) → `SomePBC([ClassDesc])`, instead
                    // of falling through to the "Don't know how to represent"
                    // error that `HostObject::new_opaque` produces.  The
                    // resulting `SomeInstance(classdef)` projects to
                    // `ConcreteType::GcRef`, matching legacy
                    // `resolve_types(Unknown) → GcRef`.  Post-jtransform
                    // [`crate::codewriter::jtransform`] still unwraps
                    // the simple_call to its inner value (the transparent
                    // semantics survive at the codewriter layer).
                    //
                    // `owner_path` qualifies the ctor identity so two
                    // distinct enums sharing a leaf (e.g.
                    // `StepResult::Continue` vs `JitAction::Continue`)
                    // produce different ClassDescs.  Falls back to the
                    // bare leaf when no owner was recorded (Ok/Err/Some).
                    // Intern the ctor class by its qualified qualname so
                    // every site shares one `HostObject` Arc — the
                    // singleton-class identity `getdesc` dedups on
                    // (`bookkeeper.rs`, keyed by `Arc::ptr_eq`).
                    // Without interning each site mints a fresh Arc → a
                    // fresh `ClassDesc` per occurrence, so a class can
                    // never be numbered once (its `minid` would differ per
                    // site) nor instantiated consistently across graphs.
                    // The bare-leaf fallback (`owner_path` empty, only
                    // reached when the type failed to resolve) is NOT
                    // interned: a bare variant name like `Ok` is ambiguous
                    // across enums, so interning it would conflate distinct
                    // classes onto one `ClassDesc`.  Exception: the fixed
                    // aggregate-kind placeholder tags named by id atom
                    // (`Tuple` for tuples / unresolved ADTs, `Array`,
                    // `Closure` — see `front::mir::aggregate_ctor_name` —
                    // and `MutRef`, the one-field GC cell for a `&mut` GC
                    // reference) are not variant names; they name ONE
                    // universal placeholder each (no enum ambiguity), and the
                    // construction-side FieldWrite chain shares the same tag
                    // as `owner_root`
                    // with the field-projection side (which resolves it
                    // through `getuniqueclassdef_for_struct_root` →
                    // `intern_class_by_qualname`).  Minting a fresh Arc per
                    // site here splits the placeholder into one `ClassDef`
                    // per occurrence, so the value's classdef can never match
                    // the field repr's interned classdef (rtyper
                    // convert_from_to has no common base → typer error), and
                    // two `()` values meeting at a join can never union.
                    let class_host = if owner_path.is_empty() {
                        // Match the aggregate-kind placeholder tag by its
                        // instantiation-stripped root so a per-shape tuple
                        // (`Tuple<f64,f64>`, `Tuple<BigInt,BigInt>`) interns
                        // by its full suffixed qualname exactly as bare
                        // `Tuple` does — `canonical_struct_name` keys the
                        // cache by the whole `Tuple<…>` string, so every
                        // construction site of one shape shares one Arc and
                        // matches the field-projection side's interned class.
                        // Without this a suffixed tuple falls to the fresh-Arc
                        // branch below and mints a distinct `Tuple<f64,f64>`
                        // per site → `commonbase` None → UnionError at a join.
                        // `MutRef<T>` is the one-field GC cell for a `&mut`
                        // GC reference: one universal placeholder per pointee
                        // shape, interned under that same suffixed qualname.
                        if matches!(
                            majit_ir::descr::strip_instantiation_suffix(name),
                            "Tuple" | "Array" | "Closure" | "MutRef"
                        ) {
                            call_registry.bookkeeper().intern_class_by_qualname(name)
                        } else {
                            HostObject::new_class(name.clone(), Vec::new())
                        }
                    } else {
                        let bk = call_registry.bookkeeper();
                        // A variant ctor's owner tail is the enum type
                        // itself (`resolve_aggregate_adt` pushes the
                        // enum leaf onto the variant's owner path); a
                        // struct ctor's owner tail is its module.  The
                        // enum base registers as a flat class whose only
                        // row is the synthetic `__discriminant` tag
                        // (`front::mir` metadata collection) — no struct
                        // carries one — so `is_enum_base` discriminates an
                        // enum variant ctor from a struct ctor.
                        //
                        // An enum variant interns through the SAME
                        // primitive the discriminant-narrowing path uses
                        // ([`Bookkeeper::intern_enum_variant_host`]), keyed
                        // by `canonical_struct_name(enum_leaf)::variant`.
                        // A constructed `Some(x)` and a matched `Some(x)`
                        // then resolve to ONE class object (`rclass.py:
                        // 82-88` single-class-per-variant), so they union
                        // to that variant (not the base) and agree on the
                        // payload attr / field owner.  The variant is a
                        // subclass of the discriminant-only enum base, so
                        // sibling variants still union to
                        // `SomeInstance(enum)` through `ClassDef::commonbase`
                        // (classdesc.py) instead of failing
                        // `mergeinputargs` with "no common base class".
                        // `canonical_struct_name` keeps two enums sharing a
                        // leaf (`StepResult::Continue` vs
                        // `JitAction::Continue`) distinct.
                        //
                        // A struct ctor keeps its dotted qualname: the
                        // FORCE-attr projection in `_init_classdef`
                        // (`struct_force_key_from_dotted_qualname`) strips
                        // the crate to the `module::Type` key the
                        // struct-field FORCE table uses.  The bare-leaf
                        // case above (`owner_path` empty) is unchanged: a
                        // bare variant name like `Ok` is ambiguous across
                        // enums, so it is not interned.
                        // Probe the QUALIFIED enum-base spelling
                        // (`owner_path.join("::")` = the enum's full
                        // `name_path`) before the bare tail.  The qualified
                        // key survives `harden_duplicate_leaf_metadata`'s
                        // bare-leaf withdrawal, so a variant ctor of an enum
                        // whose leaf collides across modules
                        // (`StepResult::Continue` vs `JitAction::Continue`)
                        // still routes through `intern_enum_variant_host`
                        // instead of falling through to the dotted struct-ctor
                        // branch — which would mint a class unrelated to the
                        // matched variant's, breaking enum base/variant class
                        // identity (RPython keys identity on the live class
                        // object, never a name, so this cannot arise upstream).
                        // Keep the qualified owner when interning too. A
                        // withdrawn or absent bare-leaf alias must not split
                        // the constructor from the class used by field reads
                        // and the common-base join (Bookkeeper.getuniqueclassdef).
                        let owner_tail = owner_path.last();
                        let owner_qual = owner_path.join("::");
                        // A duplicate leaf (`StepResult` vs `JitAction`) makes
                        // the fully qualified generic spelling ambiguous, so
                        // `is_enum_base` is false and the ctor used to mint a
                        // dotted class.  Walk path suffixes until one is the
                        // registered discriminant-only key
                        // (`pyopcode::StepResult`) and intern the variant
                        // under that class.
                        let enum_key = bk.struct_fields.borrow().as_ref().and_then(|reg| {
                            let mut candidate = owner_qual.as_str();
                            loop {
                                if let Some(key) = reg.enum_base_registry_key(candidate) {
                                    return Some(key);
                                }
                                if reg.is_enum_base(candidate) {
                                    return Some(candidate.to_string());
                                }
                                match candidate.split_once("::") {
                                    Some((_, rest)) => candidate = rest,
                                    None => break,
                                }
                            }
                            owner_tail.and_then(|tail| {
                                reg.enum_base_registry_key(tail)
                                    .or_else(|| reg.is_enum_base(tail).then(|| tail.clone()))
                            })
                        });
                        let in_variant_table = enum_key.is_some() || {
                            let map = bk.enum_variant_by_discriminant.borrow();
                            map.as_ref().is_some_and(|map| {
                                let bare = owner_qual.split('<').next().unwrap_or(&owner_qual);
                                let mut candidate = bare;
                                loop {
                                    let stripped = crate::front::mir::strip_crate_prefix(candidate);
                                    if map.contains_key(candidate)
                                        || map.contains_key(stripped.as_str())
                                    {
                                        return true;
                                    }
                                    match candidate.split_once("::") {
                                        Some((_, rest)) => candidate = rest,
                                        None => return false,
                                    }
                                }
                            })
                        };
                        if in_variant_table {
                            // Keep `owner_qual`, generics included.  Interning
                            // under the bare registry key collapses every
                            // `Result<T, E>::Ok` onto one class and their
                            // `__pos_0` payloads union.
                            bk.intern_enum_variant_host(&owner_qual, name)
                        } else {
                            // A closure env ctor.  Normal struct ctors keep the
                            // dotted qualname (`_init_classdef`'s
                            // `struct_force_key_from_dotted_qualname` reduces it
                            // to the `module::Type` FORCE key), but a closure's
                            // deep `<Impl>::fn::closure` path does not reduce to
                            // a field-registry key, so the dotted class mints
                            // fieldless.  When the full `::` `name_path` IS a
                            // struct-root registry key, intern under it and
                            // project the captured fields via
                            // `getuniqueclassdef_for_struct_root`, matching the
                            // fielded classdef the closure param-typing path
                            // (`tyref_input_class_root`) derives.  RPython
                            // forbids closures.
                            let colon_qual = format!("{owner_qual}::{name}");
                            let is_closure_root = majit_charon_reader::ullbc::is_closure_leaf(name)
                                && bk
                                    .struct_fields
                                    .borrow()
                                    .as_ref()
                                    .is_some_and(|reg| reg.fields.contains_key(&colon_qual));
                            if is_closure_root {
                                let _ = bk.getuniqueclassdef_for_struct_root(&colon_qual);
                                bk.intern_class_by_qualname(&colon_qual)
                            } else {
                                let qualname = format!("{}.{}", owner_path.join("."), name);
                                bk.intern_class_by_qualname(&qualname)
                            }
                        }
                    };
                    // Rust `T { fields }` allocates without running any
                    // `__init__`; the field initializers are the FieldWrite
                    // chain that follows.
                    if !arg_hls.is_empty() {
                        return Err(TyperError::message(format!(
                            "translate_op: SyntheticTransparentCtor {name:?} carries {} \
                             operands; a Rust aggregate is an allocation followed by \
                             its field writes. Result slot = {}",
                            arg_hls.len(),
                            fmt_op_result(op),
                        )));
                    }
                    Ok(vec![instantiate_op(class_host, result)?])
                }
                // The untranslated marker body is a no-op. The codewriter
                // reads the legacy call; annotating it as getattr on the
                // driver sentinel (a GCREF ConstRefAddr) panics.
                CallTarget::Method { name, .. }
                    if name == "jit_merge_point" || name == "can_enter_jit" =>
                {
                    let result = resolve_result_hlvalue(op, value_map)?;
                    let constant = Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::None,
                        LowLevelType::Void,
                    ));
                    Ok(vec![FlowspaceOp::new("same_as", vec![constant], result)])
                }
                CallTarget::Method { name, .. } => {
                    let mut iter = arg_hls.into_iter();
                    let receiver = iter.next().ok_or_else(|| {
                        TyperError::message(
                            "Call::Method has empty args: receiver must be args[0]".to_string(),
                        )
                    })?;
                    let bound_method = Hlvalue::Variable(Variable::new());
                    let mut call_args = Vec::with_capacity(iter.size_hint().0 + 1);
                    call_args.push(bound_method.clone());
                    call_args.extend(iter);
                    Ok(vec![
                        FlowspaceOp::new(
                            "getattr",
                            vec![
                                receiver,
                                Hlvalue::Constant(Constant::new(ConstValue::byte_str(name))),
                            ],
                            bound_method,
                        ),
                        FlowspaceOp::new("simple_call", call_args, result),
                    ])
                }
                // A `dyn Trait` virtual call. `front::mir` routes the
                // inline-Field vtable-slot dispatch here with the receiver already in
                // `args[0]`.  The production codewriter lowers it through
                // `rpbc::lower_indirect_calls` (`VtableMethodPtr` +
                // `IndirectCall`), but pyre has no IR vtable struct, so the
                // funcptr projection cannot carry a `Ptr(Func)` lltype that
                // `PtrRepr::rtype_simple_call` (rmodel.rs) needs —
                // synthesising a `SomePtr(Func)` symbolically is not
                // available in the two-phase annotate/rtype prepass.
                //
                // Instead lower to the same `getattr(receiver,
                // method_name)` + `simple_call(bound_method, args[1..])`
                // shape as `CallTarget::Method`.  The receiver `args[0]` is
                // a `dyn Trait` call-result / deref TEMP annotated
                // `SomeInstance(classdef=None)` — `dyn Trait` has no
                // `class_root`, and `derive_subject_inputcells` (this file,
                // the #346 receiver seeder) only touches startblock
                // inputargs, not temps — so the method getattr would block
                // on the classdef-less shell.  Narrow the receiver first to
                // the trait-family base `ClassDef` via the
                // `__cast_instance_intrinsic` mechanism (same as the `mir.rs`
                // path, lowered by `translate_op`'s `__cast_instance_intrinsic`
                // arm): emit
                // `simple_call(__cast_instance_intrinsic, receiver,
                // Constant(byte_str(base_root)))` producing a narrowed
                // receiver, then `getattr(narrowed, method_name)`.  The
                // family base's impl subclasses carry the methods through
                // attrfamily merge, so the getattr resolves the impl
                // `MethodDesc` family and the trailing `simple_call` rtypes
                // through the ordinary bound-method path
                // (`rpbc.py FunctionRepr.call`); dispatch stays virtual.
                //
                // The cast root must resolve through `struct_root_classes`
                // to the base classdef `register_trait_family` minted under
                // `canonical_struct_name(base_root)`.  `trait_root` here is
                // the bare trait leaf (`dyn_indirect_target`), but the family
                // base is keyed by the trait's full `name_path()`, so map the
                // leaf back through `registered_trait_family_base_root`.  When
                // no base is registered (unregistered / foreign trait, or an
                // ambiguous same-leaf collision) the lookup returns `None`
                // and the plain getattr is emitted — the receiver stays
                // classdef-less and fails loud exactly as before, no cast to
                // a non-existent root.
                CallTarget::Indirect {
                    method_name,
                    trait_root,
                } => {
                    dyn_trait_dispatch_ops(arg_hls, trait_root, method_name, result, call_registry)
                }
                CallTarget::UnsupportedExpr => Err(TyperError::message(format!(
                    "translate_op: Call with CallTarget::UnsupportedExpr at \
                     result={} — frontend coverage gap; the `front::mir` \
                     arm that emitted this Call must classify the call shape \
                     before the rtyper sees it",
                    fmt_op_result(op),
                ))),
            }
        }

        // ─── Pyre-internal: IndirectCall ───
        // An `indirect_call` op has no pre-rtyper spelling. Upstream mints it
        // inside the rtyper — `rpbc.py` `rtype_simple_call(self, hop):
        // return self.call(hop)`, which reaches the emit in
        // `FunctionReprBase.call`:
        // ```python
        // vlist.append(hop.inputconst(Void, row_of_graphs.values()))
        // v = hop.genop('indirect_call', vlist, resulttype=rresult)
        // ```
        // Everything the annotator sees is still a `simple_call`
        // (`SimpleCall` in `flowspace/operation.py`,
        // `simple_call_SomeObject` in `annotator/unaryop.py`),
        // and this adapter is the entry into annotation. So the op cannot be
        // lowered to a flowspace `indirect_call` before arriving here: the
        // trailing `c_graphs` Constant that `GraphAnalyzer.analyze`
        // (graphanalyze.py) reads out
        // of `op.args[-1]` is a product of rtyping, which has not run yet.
        //
        // `family_key` carries the pre-rtyper identity of the vtable slot, so a
        // dyn-trait dispatch lowers to the same `getattr` + `simple_call` shape
        // as its `CallTarget::Indirect` twin. `funcptr` is dropped: it is the
        // `Field[Adt(vtable), slot]` projection, which the `getattr` re-derives
        // from the receiver at the level the annotator works in.
        OpKind::IndirectCall {
            args,
            family_key: Some((trait_root, method_name)),
            ..
        } => {
            let arg_hls = args
                .iter()
                .enumerate()
                .map(|(i, v)| lookup_operand(value_map, v, op, &format!("args[{i}]")))
                .collect::<Result<Vec<_>, _>>()?;
            let result = resolve_result_hlvalue(op, value_map)?;
            dyn_trait_dispatch_ops(arg_hls, trait_root, method_name, result, call_registry)
        }
        // A stored function pointer — no trait receiver and no method name, so
        // there is nothing for a `getattr` to name. It keeps failing loud.
        OpKind::IndirectCall { .. } => Err(TyperError::message(format!(
            "translate_op: IndirectCall at result={} carries no `family_key`, \
             so it names no dyn-trait vtable slot the adapter can lower to the \
             pre-rtyper `getattr` + `simple_call` shape; a raw function-pointer \
             callee has no such shape before rtyping",
            fmt_op_result(op),
        ))),

        // ─── Pyre-internal: VtableMethodPtr ───
        // TODO(rclass-vtable-rework): pyre-only adaptation of
        // `rclass.py getclsfield()`.  Emitted by
        // `translator/rtyper/rclass.rs` to project the function
        // pointer out of a `dyn Trait` receiver's vtable. It exists
        // only *inside* the rtyper pipeline (rclass produces it;
        // jtransform consumes it), so reaching the flowspace adapter
        // input means an rtyper-stage layer missed its own emit/
        // consume invariant.
        OpKind::VtableMethodPtr { .. } => Err(TyperError::message(format!(
            "translate_op: VtableMethodPtr at result={} is rtyper-internal \
             (TODO(rclass-vtable-rework) of rclass.py:371-377); rclass.rs \
             emits it and the jtransform layer consumes it before flowspace \
             adapter input — reaching here means the rclass→jtransform \
             pipeline broke",
            fmt_op_result(op),
        ))),

        // `(lo..=hi).contains(&x)` lowered by `front::range_contains`.
        // `int_between(n, m, p)` is the llop `n <= m < p`.  It is not a
        // flowspace operator, so it is carried through under its llop
        // name; `flowin` binds the bool result and `translate_operation`
        // emits the same llop.
        OpKind::LoweredBlackholeOp { opname, args }
            if opname == "int_between" && args.len() == 3 =>
        {
            let mut hl_args = Vec::with_capacity(args.len());
            for (i, var) in args.iter().enumerate() {
                hl_args.push(lookup_operand(value_map, var, op, &format!("arg{i}"))?);
            }
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("int_between", hl_args, result)])
        }

        // `ptr::write` / `*p = v` / `*p` on raw memory.  Upstream the
        // high-level form is an llop call (`llop.raw_load(TYPE, addr,
        // offset)` / `llop.raw_store(addr, offset, value)`);
        // `_LLOpEntry.compute_result_annotation` binds
        // `lltype_to_annotation(RESULTTYPE)` and
        // `_LLOpEntry.specialize_call` re-emits the same-named llop.
        // The descr (`item_ty` / `itemsize` / `is_item_signed`) lives
        // on the original `OpKind` that `jtransform` `rewrite_op_raw_load`
        // / `rewrite_op_raw_store` still consumes.  A `ConstValue::LowLevelType`
        // extra arg cannot carry TYPE through `hop.setup` —
        // `Bookkeeper.immutablevalue` rejects that variant — so the
        // adapter writes `lltype_to_annotation` of the descr onto the
        // result Variable; `flowin` takes ownership via `setbinding`.
        OpKind::RawLoad {
            base,
            offset,
            item_ty,
            itemsize,
            is_item_signed,
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let offset_hl = lookup_operand(value_map, offset, op, "offset")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            seed_raw_item_annotation(
                &result,
                raw_item_lltype(item_ty, *itemsize, *is_item_signed)?,
            );
            Ok(vec![FlowspaceOp::new(
                "raw_load",
                vec![base_hl, offset_hl],
                result,
            )])
        }
        OpKind::RawStore {
            base,
            offset,
            value,
            ..
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let offset_hl = lookup_operand(value_map, offset, op, "offset")?;
            let value_hl = lookup_operand(value_map, value, op, "value")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            seed_raw_item_annotation(&result, LowLevelType::Void);
            Ok(vec![FlowspaceOp::new(
                "raw_store",
                vec![base_hl, offset_hl, value_hl],
                result,
            )])
        }

        // ─── Stage-invariant fail-loud catch-all ───
        // No remaining variants reach here legitimately: every legitimate
        // pre-rtyper input shape has an explicit arm above, every
        // post-rtyper jtransform-emitted variant is enumerated in
        // `post_rtyper_jtransform_variant_name` and short-circuits with
        // a stage-mismatch message before this fall-through, and every
        // pyre-internal rtyper-cutover variant (`IndirectCall`,
        // `VtableMethodPtr`) has its own targeted fail-loud arm.  Hitting
        // this catch-all means a brand-new `OpKind` was added without
        // updating either the explicit translate arm OR the variant-name
        // table — fail-loud with a clear pointer at where the missing
        // arm should land.
        other => {
            let variant = opkind_variant_name(other);
            if let Some(stage_msg) = post_rtyper_jtransform_variant_name(other) {
                Err(TyperError::message(format!(
                    "translate_op: post-rtyper jtransform variant `{stage_msg}` \
                     reached the flowspace adapter at result={}.  RPython \
                     `rpython/jit/codewriter/jtransform.py` runs *after* the \
                     rtyper has lowered every high-level op, so this variant \
                     must NEVER appear at the rtyper input.  Source of the \
                     leak is upstream — check `rpbc.rs` / `rclass.rs` / the \
                     pre-rtyper graph builder for an emit site that should \
                     have produced a pre-rtyper shape (e.g. `FieldRead` / \
                     `ArrayRead` / `Call`) instead of `{variant}`.",
                    fmt_op_result(op),
                )))
            } else {
                Err(TyperError::message(format!(
                    "translate_op: OpKind variant `{variant}` has no \
                     translate arm and no stage-invariant classification.  \
                     A new pyre-internal variant was added to \
                     `model::OpKind` without updating \
                     `flowspace_adapter::translate_op` or \
                     `opkind_variant_name`.  Add an explicit translate arm \
                     above (lower to flowspace) or, if the variant is \
                     post-rtyper-only, list it in \
                     `post_rtyper_jtransform_variant_name` so the \
                     stage-mismatch message fires.  result={}",
                    fmt_op_result(op),
                )))
            }
        }
    }
}

/// Stable variant name for fail-loud messages. Matches the RPython
/// convention of identifying ops by their opname stem so dual-gate
/// failures are immediately greppable.
fn opkind_variant_name(kind: &OpKind) -> &'static str {
    #[allow(unreachable_patterns)]
    match kind {
        OpKind::Input { .. } => "Input",
        OpKind::ConstInt(_) => "ConstInt",
        OpKind::ConstFnAddr { .. } => "ConstFnAddr",
        OpKind::ConstUInt(_) => "ConstUInt",
        OpKind::ConstBool(_) => "ConstBool",
        OpKind::ConstSymbolic { .. } => "ConstSymbolic",
        OpKind::ConstFloat(_) => "ConstFloat",
        OpKind::ConstInt128(_) => "ConstInt128",
        OpKind::ConstUInt128(_) => "ConstUInt128",
        OpKind::ConstSingleFloat(_) => "ConstSingleFloat",
        OpKind::ConstStr(_) => "ConstStr",
        OpKind::ConstRef(_) => "ConstRef",
        OpKind::ConstRefNull => "ConstRefNull",
        OpKind::ConstNone => "ConstNone",
        OpKind::ConstRefAddr(_) => "ConstRefAddr",
        OpKind::FieldRead { .. } => "FieldRead",
        OpKind::FieldWrite { .. } => "FieldWrite",
        OpKind::ArrayRead { .. } => "ArrayRead",
        OpKind::ArrayLen { .. } => "ArrayLen",
        OpKind::ArrayWrite { .. } => "ArrayWrite",
        OpKind::InteriorFieldRead { .. } => "InteriorFieldRead",
        OpKind::InteriorFieldWrite { .. } => "InteriorFieldWrite",
        OpKind::Call { .. } => "Call",
        OpKind::GuardTrue { .. } => "GuardTrue",
        OpKind::GuardFalse { .. } => "GuardFalse",
        OpKind::GuardValue { .. } => "GuardValue",
        OpKind::VtableMethodPtr { .. } => "VtableMethodPtr",
        OpKind::IndirectCall { .. } => "IndirectCall",
        OpKind::BinOp { .. } => "BinOp",
        OpKind::UnaryOp { .. } => "UnaryOp",
        OpKind::VableForce { .. } => "VableForce",
        OpKind::VableFieldRead { .. } => "VableFieldRead",
        OpKind::VableFieldWrite { .. } => "VableFieldWrite",
        OpKind::VableArrayRead { .. } => "VableArrayRead",
        OpKind::VableArrayWrite { .. } => "VableArrayWrite",
        OpKind::VableArrayLen { .. } => "VableArrayLen",
        OpKind::CallElidable { .. } => "CallElidable",
        OpKind::CallResidual { .. } => "CallResidual",
        OpKind::CallMayForce { .. } => "CallMayForce",
        OpKind::InlineCall { .. } => "InlineCall",
        OpKind::RecursiveCall { .. } => "RecursiveCall",
        OpKind::JitDebug { .. } => "JitDebug",
        OpKind::AssertGreen { .. } => "AssertGreen",
        OpKind::CurrentTraceLength => "CurrentTraceLength",
        OpKind::IsConstant { .. } => "IsConstant",
        OpKind::IsVirtual { .. } => "IsVirtual",
        OpKind::GuardClass { .. } => "GuardClass",
        OpKind::IsInstance { .. } => "IsInstance",
        OpKind::ConditionalCall { .. } => "ConditionalCall",
        OpKind::ConditionalCallValue { .. } => "ConditionalCallValue",
        OpKind::RecordKnownResult { .. } => "RecordKnownResult",
        OpKind::RecordQuasiImmutField { .. } => "RecordQuasiImmutField",
        OpKind::Live => "Live",
        OpKind::JitMergePoint { .. } => "JitMergePoint",
        OpKind::LoopHeader { .. } => "LoopHeader",
        OpKind::Abort { .. } => "Abort",
        OpKind::NewArrayClear { .. } => "NewArrayClear",
        OpKind::NewListClear { .. } => "NewListClear",
        OpKind::New { .. } => "New",
        OpKind::RawMalloc { .. } => "RawMalloc",
        OpKind::RawFree { .. } => "RawFree",
        OpKind::NewWithVtable { .. } => "NewWithVtable",
        OpKind::NewArray { .. } => "NewArray",
        OpKind::NewTuple { .. } => "NewTuple",
        OpKind::NewList { .. } => "NewList",
        OpKind::GetSlice { .. } => "GetSlice",
        OpKind::RawLoad { .. } => "RawLoad",
        OpKind::RawStore { .. } => "RawStore",
        OpKind::Hint { .. } => "Hint",
        OpKind::LoweredBlackholeOp { .. } => "LoweredBlackholeOp",
        OpKind::LoadStatic { .. } => "LoadStatic",
        // Catch-all for variants pyre may add without bumping this
        // table — surfaces as `<unknown>` in the fail-loud message
        // rather than a misleading variant tag.  The catch-all message
        // (above) instructs the reader to extend this table.
        _ => "<unknown OpKind variant>",
    }
}

/// Identify whether `kind` is a post-rtyper jtransform-emitted variant
/// (i.e., emitted by `rpython/jit/codewriter/jtransform.py` AFTER the
/// rtyper has lowered every high-level op).  These variants must NEVER
/// reach the flowspace adapter — a leak indicates the upstream pre-
/// rtyper graph builder (or pyre's `rpbc.rs` / `rclass.rs`) emitted the
/// post-rtyper shape directly instead of routing through the rtyper-
/// equivalent lowering step.  Returns `Some(name)` for fail-loud messages.
fn post_rtyper_jtransform_variant_name(kind: &OpKind) -> Option<&'static str> {
    Some(match kind {
        OpKind::VableFieldRead { .. } => "VableFieldRead (jtransform.py:651-927)",
        OpKind::VableFieldWrite { .. } => "VableFieldWrite (jtransform.py:651-927)",
        OpKind::VableArrayRead { .. } => "VableArrayRead (jtransform.py:651-927)",
        OpKind::VableArrayWrite { .. } => "VableArrayWrite (jtransform.py:651-927)",
        OpKind::VableArrayLen { .. } => "VableArrayLen (jtransform.py:808-817)",
        OpKind::CallElidable { .. } => "CallElidable (jtransform.py:414-435 rewrite_call)",
        OpKind::CallResidual { .. } => "CallResidual (jtransform.py:414-435 rewrite_call)",
        OpKind::CallMayForce { .. } => "CallMayForce (jtransform.py:414-435 rewrite_call)",
        OpKind::InlineCall { .. } => "InlineCall (jtransform.py:473-482)",
        OpKind::RecursiveCall { .. } => "RecursiveCall (jtransform.py:522-534)",
        OpKind::JitDebug { .. } => "JitDebug (jtransform.py:1731-1743)",
        OpKind::AssertGreen { .. } => "AssertGreen (jtransform.py:1731-1743)",
        OpKind::CurrentTraceLength => "CurrentTraceLength (jtransform.py:1731-1743)",
        OpKind::IsConstant { .. } => "IsConstant (jtransform.py:1731-1743)",
        OpKind::IsVirtual { .. } => "IsVirtual (jtransform.py:1731-1743)",
        OpKind::GuardClass { .. } => "GuardClass (jtransform.py:1004-1010)",
        OpKind::ConditionalCall { .. } => "ConditionalCall (jtransform.py:1665-1688)",
        OpKind::ConditionalCallValue { .. } => "ConditionalCallValue (jtransform.py:1665-1688)",
        OpKind::RecordKnownResult { .. } => "RecordKnownResult (jtransform.py:292-313)",
        OpKind::RecordQuasiImmutField { .. } => "RecordQuasiImmutField (jtransform.py:901-903)",
        OpKind::Live => "Live (jtransform.py:469,481,533)",
        OpKind::JitMergePoint { .. } => "JitMergePoint (jtransform.py:1690-1718)",
        OpKind::LoopHeader { .. } => "LoopHeader (jtransform.py:1690-1718)",
        OpKind::NewArray { .. } => "NewArray (jtransform.py do_fixed_newlist)",
        OpKind::NewArrayClear { .. } => "NewArrayClear (jtransform.py do_fixed_newlist_clear)",
        OpKind::NewListClear { .. } => {
            "NewListClear (jtransform.py:1938-1943 do_resizable_newlist_clear)"
        }
        // `OpKind::Abort` is pyre-only — RPython raises `FlowingError`
        // (`flowspace/flowcontext.py`) and drops the function
        // before reaching the rtyper.  Now handled by an explicit
        // `Ok(Vec::new())` arm in `translate_op`, but the diagnostic
        // name is retained here so the post-rtyper variant table can
        // still surface the marker if a leak ever reaches it.
        // Each front-end `stop_unsupported` / `continue_with_unknown`
        // emit-site is retired by lowering its specific expression
        // form (`ConstStr`, `Range`, `Closure`, etc.).
        OpKind::Abort { .. } => "Abort (runtime extension marker)",
        _ => return None,
    })
}

/// Output of [`function_graph_to_flowspace`] — the assembled
/// `flowspace::FunctionGraph` plus enough side tables for
/// `specialize_legacy_graph` to drive `RPythonTyper::specialize`
/// against pyre's annotator surface and read back per-`Variable`
/// concretetypes.
#[derive(Debug)]
pub struct FlowspaceAdapterOutput {
    /// Assembled `flowspace::FunctionGraph` carrying every legacy block
    /// translated to a `flowspace::Block` over `Hlvalue` operands.
    /// Wrapped in `Rc<RefCell<_>>` to match RPython's
    /// `FunctionDesc.cache` ownership shape — handed to
    /// `RPythonAnnotator` directly.
    pub graph: Rc<RefCell<FlowspaceGraph>>,
    /// Legacy graph `Variable` → typed flowspace `Variable`; each typed
    /// Variable's `concretetype` is read after `specialize` returns.
    pub value_to_var: LegacyToTyped,
    /// All block-local typed representatives for each legacy Variable.
    /// See [`LegacyToTypedCandidates`].
    pub value_to_var_candidates: LegacyToTypedCandidates,
    /// Const-define result `Variable` -> `Constant.concretetype`.
    /// Materialised at lift time from `OpKind::ConstInt` / `ConstFloat`
    /// via `Constant::with_concretetype`,
    /// matching RPython's `Constant.concretetype` ground truth.  The
    /// per-`Variable` `LowLevelType` is read directly so the projector
    /// does not have to reconstruct the kind from the reduced legacy
    /// `ValueType` view.
    pub constant_concretetypes: HashMap<Variable, LowLevelType>,
    /// Actual constant representatives, retained for the Constant arm of
    /// simplify.py::remove_identical_vars_SSA's phi renaming. Type equality
    /// alone is not evidence that two incoming constants are equal.
    pub constant_hlvalues: HashMap<Variable, Hlvalue>,
    /// `BlockId → flowspace::BlockRef` mapping. Includes the canonical
    /// `returnblock` and `exceptblock` (mapped to the
    /// `FunctionGraph::with_return_var`-allocated final blocks) so any
    /// legacy Link targeting them resolves correctly.
    #[cfg(test)]
    pub block_map: HashMap<BlockId, BlockRef>,
}

/// Translate a legacy `ExitCase` into the `Hlvalue` slot expected by
/// `flowspace::Link.exitcase`. RPython encodes the discriminating value
/// as a `Hlvalue::Constant` carrying the matched bool / Python value
/// (`flowspace/model.py`).
fn exitcase_to_hlvalue(
    exitcase: Option<&ExitCase>,
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
) -> Result<Option<Hlvalue>, TyperError> {
    Ok(match exitcase {
        None => None,
        Some(ExitCase::Bool(b)) => Some(Hlvalue::Constant(constant_from_constvalue(
            ConstValue::Bool(*b),
        ))),
        Some(ExitCase::Const(cv)) => Some(Hlvalue::Constant(constant_from_constvalue(cv.clone()))),
        // `except OperationError`: the carrier's class object, the one the
        // bookkeeper mints as an `Exception` subclass.
        Some(ExitCase::ErrorCarrier) => {
            let class = call_registry
                .bookkeeper()
                .exception_carrier_class()
                .ok_or_else(|| {
                    TyperError::message(
                        "ExitCase::ErrorCarrier on a pipeline that named no error carrier",
                    )
                })?;
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                class,
            ))))
        }
    })
}

fn constant_from_constvalue(value: ConstValue) -> Constant {
    match value {
        ConstValue::Int(n) => Constant::with_concretetype(ConstValue::Int(n), LowLevelType::Signed),
        ConstValue::Int128(n) => {
            Constant::with_concretetype(ConstValue::Int128(n), LowLevelType::SignedLongLongLong)
        }
        ConstValue::UInt128(n) => {
            Constant::with_concretetype(ConstValue::UInt128(n), LowLevelType::UnsignedLongLongLong)
        }
        ConstValue::Bool(b) => Constant::with_concretetype(ConstValue::Bool(b), LowLevelType::Bool),
        ConstValue::Float(bits) => {
            Constant::with_concretetype(ConstValue::Float(bits), LowLevelType::Float)
        }
        other => Constant::new(other),
    }
}

fn legacy_const_define_hlvalue(
    op: &SpaceOperation,
    call_registry: Option<&crate::translator::rtyper::call_registry::CallRegistry>,
) -> Result<Option<Hlvalue>, TyperError> {
    // Const-define ops always carry a result Variable; bail on the
    // (malformed) result-less op so the caller can key the const
    // concretetype by `op.result` identity.
    if op.result.is_none() {
        return Ok(None);
    }
    match &op.kind {
        OpKind::ConstInt(n) => Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(*n),
            LowLevelType::Signed,
        )))),
        OpKind::ConstFnAddr { value, .. } => Ok(Some(Hlvalue::Constant(
            Constant::with_concretetype(ConstValue::Int(*value), LowLevelType::Signed),
        ))),
        OpKind::ConstUInt(n) => Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(*n as i64),
            LowLevelType::Unsigned,
        )))),
        OpKind::ConstInt128(n) => Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int128(*n),
            LowLevelType::SignedLongLongLong,
        )))),
        OpKind::ConstUInt128(n) => Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::UInt128(*n),
            LowLevelType::UnsignedLongLongLong,
        )))),
        OpKind::ConstBool(b) => Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Bool(*b),
            LowLevelType::Bool,
        )))),
        OpKind::ConstFloat(bits) => Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Float(*bits),
            LowLevelType::Float,
        )))),
        OpKind::ConstRefNull => Ok(Some(Hlvalue::Constant(const_ref_gcref_constant(None)))),
        // The annotator `None` constant — `Constant(NoneType)`,
        // `concretetype = Void`.  `immutableconstant` annotates it
        // `SomeValue::None_` (bookkeeper `immutablevalue(&ConstValue::None)`
        // → `s_none()`), the shape `decompose_slice_args`'s `stop_is_none`
        // selection reads for a `[start:]` getslice.  Same `(None, Void)`
        // carrier `set_return` uses for a void return.
        OpKind::ConstNone => Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        )))),
        OpKind::ConstRefAddr(addr) => Ok(Some(Hlvalue::Constant(const_ref_gcref_constant(Some(
            *addr,
        ))))),
        // Prebuilt-constant-array define-op.  A module-level constant
        // array (`OP_STACKDEL: [i32; N]`) is carried as a bare immutable
        // `Constant(list)`; `immutableconstant` annotates it `SomeList`
        // and its `arr[i]` read is a foldable getarrayitem.  `front::mir`
        // has no ConstList opkind and synthesises a 0-arg
        // `Call(["__const_int_array", <i0>, …])` instead
        // (`fold_named_const_int_array_global`); re-fold that define-op to
        // the upstream Constant shape here, the same way the
        // `__str_const` arm below re-folds a string literal.
        kind if is_const_prebuilt_array_define(kind) => Ok(const_prebuilt_array_items(kind)
            .map(|items| Hlvalue::Constant(Constant::new(ConstValue::List(items))))),
        // String-literal constant.  Upstream flowspace carries a string
        // literal as a bare `Constant('text')` SSA value (annotated
        // `SomeString` by `immutablevalue`); `front::mir` has no
        // ConstStr opkind and synthesises a 0-arg
        // `Call(["__str_const", <text>])` instead (`mir.rs`).
        // Re-fold that define-op to the upstream Constant shape here.
        // The stamped lltype is `Ptr(STR)` — fixed for every string
        // constant, same as the primitive arms above (the rtyper's
        // `StringRepr.convert_const` re-derives the same `Ptr(STR)`
        // per use; `inputconst` always converts from `c.value`, so
        // the ByteStr payload is never read through this stamp).
        // Without the stamp the define-result Variable has no kind,
        // and the dual-gate projection reports the slot as a
        // `legacy=GcRef, real=Unknown` divergence on every graph
        // containing a string literal.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.is_empty() && segments.len() == 2 && segments[0] == "__str_const" => {
            Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::ByteStr(segments[1].clone().into_bytes()),
                crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone(),
            ))))
        }
        OpKind::Call { target, args, .. }
            if args.is_empty() && crate::model::fn_const_segments(target).is_some() =>
        {
            let segments = crate::model::fn_const_segments(target).expect("guard checked");
            let call_registry = call_registry.ok_or_else(|| {
                TyperError::message(format!(
                    "fn const {:?} requires a CallRegistry to materialise getfunctionptr",
                    segments
                ))
            })?;
            let key = crate::translator::rtyper::call_registry::FunctionPathKey(segments.to_vec());
            let entry = call_registry.lookup_with_leaf_match(&key).ok_or_else(|| {
                TyperError::message(format!(
                    "fn const {:?} did not resolve to a registered callee",
                    key.segments()
                ))
            })?;
            // A fn-const is address-taken data for a static method table
            // (`typedef` `[(name, fn, arity)]`).  Its LL carrier is
            // unconditionally `ConstFnAddr` (the const-define arm below),
            // it is never JIT-called in the compiled portal, so the `FuncType`
            // is pure typing bookkeeping with no machine-code channel.  Taking
            // the address does NOT need the callee's body — only its declared
            // signature.  Three populate-time realities block the precise
            // `getfunctionptr`, and all three fall back to the callee's
            // DECLARED LLBC signature (projected at the populate pre-pass; same
            // fidelity class as `FOREIGN_STDLIB_EXTERNALS`) so the caller's
            // lift succeeds instead of recording a lazy-failure that poisons
            // every attr-lookup caller:
            //   1. no graph is rtyped yet (every arg `concretetype` is `None`);
            //   2. the callee's graph is still being built (a fn-const
            //      cycle), so its pygraph is not cached;
            //   3. the callee's BODY does not lift — e.g. an unregistered
            //      iterator/container adapter, a separate front-lowering gap
            //      (#65).  The address is still valid; only JIT-compiling the
            //      body is blocked, and this entry is never JIT-called.
            // A callee whose signature is not fully projectable leaves the slot
            // `None`; the site then fails closed with the most specific
            // diagnosis, exactly as before.
            use crate::translator::rtyper::lltypesystem::lltype;
            let graphs = entry.function_desc.borrow().getgraphs();
            let lift_error = entry.lift_error();
            let maybe_graph = graphs.into_iter().next();
            // The declared LLBC signature is the one `FuncType` for this
            // Rust `fn` type (`getfunctionptr` / `FuncType` from the
            // graph's declared args). Prefer it over a rtyped
            // `getfunctionptr`: after rtyping, a slice arg's
            // concretetype is a list pointer, which is a different
            // `ll_ptrtype` from the parameter-seed spelling of the same
            // `fn(&[PyObjectRef]) -> …`. Keep the graph backlink when a
            // pygraph is cached.
            let func_ptr = match entry.declared_funcptr_type() {
                Some(ft) => match &maybe_graph {
                    Some(graph) => lltype::functionptr_for_graph_with_type(&graph.graph, ft),
                    None => lltype::functionptr(ft, &key.segments().join("::"), None, None),
                },
                None => {
                    let precise = match (&maybe_graph, &lift_error) {
                        (Some(graph), None) => {
                            lltype::getfunctionptr(&graph.graph, lltype::_getconcretetype).ok()
                        }
                        _ => None,
                    };
                    match precise {
                        Some(func_ptr) => func_ptr,
                        None => {
                            return Err(TyperError::message(match lift_error {
                                Some(err) => format!(
                                    "fn const {:?} resolved to an unbuildable callee: {err}",
                                    key.segments()
                                ),
                                None => format!(
                                    "fn const {:?} resolved to a registry entry without a cached graph",
                                    key.segments()
                                ),
                            }));
                        }
                    }
                }
            };
            let func_ptr_type = crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Ptr(
                Box::new(func_ptr._TYPE.clone()),
            );
            Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(func_ptr)),
                func_ptr_type,
            ))))
        }
        // RPython parity: unit-variant ctors (`StepResult::Continue`,
        // `LoopResult::Done`, …) are pre-built singleton instances at
        // the rtyper layer (`rclass.InstanceRepr.
        // get_reusable_prebuilt_instance`), so the codewriter never
        // sees a call op for them — the rtyper folds them to
        // `Constant(prebuilt_ptr)` before `jtransform` runs.
        //
        // Pyre's frontend (`front::mir`) lowers a unit-variant
        // path expression to `OpKind::Call { target:
        // SyntheticTransparentCtor, args: [], result_ty: Unknown }`;
        // without this pre-fold the args=[] call falls through to
        // `handle_residual_call` and leaves a `residual_call_r/d>r`
        // op in the walker arm body that breaks the walker's arm
        // dispatch.
        //
        // The `is_synthetic_unit_variant_path` allowlist
        // (StepResult, LoopResult, JitAction, CompareOp variants) is
        // the same set consulted here — both layers agree on which
        // paths are unit-variant singletons.
        OpKind::Call {
            target:
                crate::model::CallTarget::SyntheticTransparentCtor {
                    name,
                    owner_path,
                    variant_tag,
                    ..
                },
            args,
            ..
        } if args.is_empty() => {
            let mut segments = owner_path.clone();
            segments.push(name.clone());
            if !crate::translator::rtyper::unit_variant_fold::is_synthetic_unit_variant_path(
                &segments,
            ) {
                return Ok(None);
            }
            // PyPy rtyper folds unit-variant PBC constructors into a
            // singleton instance pointer before jtransform sees them
            // (`rtyper/rpbc.py::SingleFrozenPBCRepr`).  The
            // pre-fold here materialises the same shape inside the
            // flowspace graph so the per-graph annotator surfaces a
            // `Hlvalue::Constant(HostObject(prebuilt_instance))` to
            // downstream rtyper passes.  This only
            // affects graphs that go through the rtyper Match arm
            // (`dual_gate_publish_concretetypes`).  Graphs registered
            // directly through `register_function_graph` can take the
            // Skip arm and bypass this pre-fold;
            // the residual `OpKind::Call` survives into jtransform
            // and is emitted as a `residual_call_r_r` wrapper there.
            // Closing that gap requires either an early-pass on
            // `FunctionGraph` ahead of jtransform or extending
            // `is_synthetic_result_option_ctor` to handle the args=0
            // case.
            let qualname = segments.join(".");
            // Reuse the bookkeeper's interned-class prebuilt instance so this
            // legacy fold path produces the same `HostObject` Arc as the
            // pre-jtransform `fold_unit_variant_ctors` pass — mirrors
            // `InstanceRepr.get_reusable_prebuilt_instance` caching on
            // the per-rtyper `instance_reprs` map
            // (`rpython/rtyper/rclass.py`).  Without this, two
            // graphs that reach the same unit variant via different
            // gate arms would resolve to distinct singletons.
            let Some(registry) = call_registry else {
                return Ok(None);
            };
            let instance =
                crate::translator::rtyper::unit_variant_fold::intern_unit_variant_prebuilt_instance(
                    registry.bookkeeper(),
                    &qualname,
                    *variant_tag,
                );
            let Some(instance) = instance else {
                return Ok(None);
            };
            Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::HostObject(instance),
                crate::translator::rtyper::rclass::OBJECTPTR.clone(),
            ))))
        }
        _ => Ok(None),
    }
}

/// Translate a single legacy `LinkArg` into a `Hlvalue`. `LinkArg::Value`
/// resolves through `value_map` (which carries Variable identities for
/// regular operands and inlined constants for `OpKind::ConstInt` /
/// `ConstFloat` defines, mirrored in tests by
/// `build_value_to_hlvalue_map`). `LinkArg::Const` materialises a
/// fresh `Hlvalue::Constant`.
///
/// `source_block_id` / `target_block_id` / `arg_index` carry the
/// surrounding context for fail-loud diagnostics — when the lookup
/// misses, the message embeds the predecessor and successor block ids
/// plus the `arg_index` into `Link.args`, so per-graph diagnosis can
/// locate the broken link without re-traversing the graph.  Mirrors
/// the role-bearing enrichment of `lookup_operand` (variant name +
/// arg role).  The required substring `"undefined operand"`
/// is preserved verbatim for `is_known_unported`
/// (`cutover.rs`).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn link_arg_to_hlvalue(
    arg: &LinkArg,
    value_map: &HashMap<Variable, Hlvalue>,
    source_block_id: BlockId,
    target_block_id: BlockId,
    arg_index: usize,
) -> Result<Hlvalue, TyperError> {
    match arg {
        // `LinkArg::Const` now carries the full upstream-orthodox
        // `Constant` (id + value + concretetype) directly — no need
        // to round-trip through `constant_from_constvalue` and
        // mint a fresh id.
        LinkArg::Const(cv) => Ok(Hlvalue::Constant(cv.clone())),
        LinkArg::Value(var) => value_map
            .get(var)
            .cloned()
            .or_else(|| void_constant_for(var))
            .ok_or_else(|| {
                TyperError::message(format!(
                    "translate_op: undefined operand {var:?} as Link.args[{arg_index}] entry \
                 (source block {source_block_id:?} -> target block {target_block_id:?}) — \
                 adapter invariant broken (every referenced operand must be \
                 defined as a block inputarg or op result)",
                ))
            }),
    }
}

/// Translate an exception-link extra variable.
///
/// RPython `flowspace/model.py` defines `link.last_exception`
/// and `link.last_exc_value` in the link scope before checking
/// `link.args`; those same Variables may then appear in `link.args`.
/// Pyre's legacy graph represents them as fresh slots whose only
/// definition site is the link, so the adapter must materialise them in
/// a per-link map instead of requiring a block-local definition.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn link_extravar_to_hlvalue(
    arg: &LinkArg,
    name: &str,
    value_map: &mut HashMap<Variable, Hlvalue>,
    value_to_var: &mut LegacyToTyped,
) -> Result<Hlvalue, TyperError> {
    match arg {
        LinkArg::Value(legacy_var) => {
            if let Some(existing) = value_map.get(legacy_var).cloned() {
                return Ok(existing);
            }
            let mut var = seed_variable(legacy_var);
            // flowcontext.py BlockRecorder.guessexception gives these
            // link-scoped definitions their last_exception/last_exc_value
            // names. SSA_to_SSI excludes those prefixes from ordinary
            // cross-block uses; anonymous variables would be threaded back
            // before the raising operation that actually defines them.
            var.rename(name);
            value_to_var
                .entry(legacy_var.clone())
                .or_insert_with(|| var.clone());
            let hlvalue = Hlvalue::Variable(var);
            value_map.insert(legacy_var.clone(), hlvalue.clone());
            Ok(hlvalue)
        }
        // `LinkArg::Const` now carries the full upstream-orthodox
        // `Constant` (id + value + concretetype) directly — no need
        // to round-trip through `constant_from_constvalue` and
        // mint a fresh id.
        LinkArg::Const(cv) => Ok(Hlvalue::Constant(cv.clone())),
    }
}

/// Derive per-inputarg `SomeValue` cells for a subject's startblock,
/// preferring an explicit `Variable.annotation` seed (test-fixture
/// hand-built graphs without front-end Input ops) and falling through
/// to the `OpKind::Input { name, ty }` ops the front-end (`front::mir`)
/// emits for the receiver `self` and each typed param.
///
/// Returns one `SomeValue` per `startblock.inputargs` entry, in
/// position order.
///
/// Resolution order per inputarg `vid`:
/// 1. Matching `OpKind::Input { ty, class_root }` op result == `vid` at the
///    startblock — the source-typed production graph from `front::mir`.
/// 2. `legacy.variable(vid).annotation` — fallback for minimal fixtures that
///    have no production-shape Input op and explicitly seed the Variable via
///    `legacy_annotator::setbinding(&var, ty)`.
///
/// Errors:
///
/// - Both sources miss for an inputarg — front-end producer
///   divergence (every typed param emits the Input op alongside the
///   inputargs registration in the front pass; a missing Input op
///   means the producer wired the inputarg without declaring its
///   type and no `Variable.annotation` shell was supplied either).
/// - `valuetype_to_someshell(ty)` returns `None` for the resolved
///   `ValueType` (only `ValueType::Unknown`) — the inputarg's type
///   is an annotation gap; the helper surfaces it the same way
///   `seed_variable` does.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub(crate) fn derive_subject_inputcells(
    legacy: &FunctionGraph,
    // Used to seed a `Ref` receiver with its cached, identity-stable
    // struct-root `ClassDef` via `getuniqueclassdef_for_struct_root`
    // (see the `OpKind::Input` arm below).  `None` (test fixtures) keeps
    // every `Ref` classdef-less, narrowed later by annotation.
    bookkeeper: Option<&Rc<crate::annotator::bookkeeper::Bookkeeper>>,
) -> Result<Vec<crate::annotator::model::SomeValue>, TyperError> {
    // Id-keyed lookup, not the dense `blocks[id.0]` projection — block
    // ids need not be index-aligned (see `reachable_block_ids`).
    let startblock = legacy
        .blocks
        .iter()
        .find(|b| b.id == legacy.startblock)
        .ok_or_else(|| {
            TyperError::message(format!(
                "derive_subject_inputcells: startblock {:?} not present in graph blocks",
                legacy.startblock
            ))
        })?;
    let mut input_by_result: HashMap<
        crate::flowspace::model::Variable,
        (&crate::model::ValueType, &Option<String>),
    > = HashMap::new();
    for op in &startblock.operations {
        if let (Some(result), OpKind::Input { ty, class_root, .. }) = (op.result.as_ref(), &op.kind)
        {
            input_by_result.insert(result.clone(), (ty, class_root));
        }
    }
    let mut cells = Vec::with_capacity(startblock.inputargs.len());
    for (idx, var) in startblock.inputargs.iter().enumerate() {
        // 1. Front-end Input op at the startblock.  This is the source-type
        // authority, just as RPython seeds a function from its signature and
        // then lets call propagation widen that cell.  In particular, do not
        // let a classdef-less annotation left on the legacy graph erase a
        // richer `PyObjectRef`/W_Root class_root: that stale slot is an input
        // from the transitional walker, not an RPython annotation source.
        if let Some(&(ty, class_root)) = input_by_result.get(var) {
            let shell = valuetype_to_someshell(ty).ok_or_else(|| {
                TyperError::message(format!(
                    "derive_subject_inputcells: startblock.inputargs[{idx}] \
                     ({var:?}) has `ValueType::{ty:?}` (from Input op) whose \
                     `valuetype_to_someshell` projection is `None` (annotation gap — \
                     only `ValueType::Unknown` lacks a SomeValue shell)"
                ))
            })?;
            // A `Ref` inputarg projects to the abstract `SomeInstance(None)`
            // shell from `valuetype_to_someshell`.  When the front-end
            // resolved the param's struct root (`OpKind::Input.class_root`,
            // populated from `type_root_ident` at front/ast.rs)
            // and the struct-field registry knows that root, seed the
            // receiver with the *cached* `ClassDef` from
            // `getuniqueclassdef_for_struct_root` (bookkeeper.rs). The
            // cache makes the classdef identity-stable across repeated
            // lookups (`Rc::ptr_eq`), so the annotation fixpoint stays
            // independent of graph-processing order — unlike the earlier
            // eager seed that minted a fresh struct-root ClassDef per site
            // (identity divergence → order-dependent fixpoint).  Host structs
            // with no RPython classdef (e.g. `PyFrame::locals_cells_stack_w:
            // *mut FixedObjectArray`) are never a call argument, so
            // call-propagation never narrows them; the seed is the only path
            // that gives them a populated receiver.  Other `Ref` variants
            // (no `class_root`, unknown root, or no bookkeeper) keep the
            // classdef-less shell, narrowed by call-propagation as before
            // (`description.py FunctionDesc.pycall`).
            // A `Vec` of one-word items is the raw `{ptr, len, cap}` header
            // `RustVecRepr` lowers; the front end types it `Int` and names it
            // by its spelling.
            if let (Some(bk), Some(root)) = (bookkeeper, class_root.as_deref())
                && let Some(s_vec) = bk.project_rust_vec(root)
            {
                cells.push(s_vec);
                continue;
            }
            // A function pointer is the same Int bank (`history.getkind` of
            // `Ptr(FuncType)`) named `fn(inputs) -> output` /
            // `Option<fn(inputs) -> output>`. `valuetype_to_someshell` would
            // seed `SomeInteger`; seed `SomePtr` so a `fn` argument unions
            // with a reified fn item, a `fn` field read, and a typed
            // `fn_null_constant` (`llannotation.py`
            // `pairtype(SomePtr, SomePtr).union`). The coarse `Option<fn>`
            // marker still unions with an untyped `fn_null_constant`.
            if let Some(root) = class_root.as_deref()
                && let Some(s_fn) = crate::annotator::bookkeeper::fn_ptr_somevalue_for_spelling_in(
                    root,
                    bookkeeper.map(|bk| bk.as_ref()),
                )
            {
                cells.push(s_fn);
                continue;
            }
            // A list-typed param (`Vec<T>`, `&[T]`, Charon `Slice<T>`) is
            // the annotator's list model regardless of the Input's
            // register bank: a one-word item slice can arrive as `Int`
            // (fat-pointer address) and still needs `SomeList` so
            // `args.len()` is `len` of a list, not `getattr("__len__")`
            // on a classdef-less `SomeInstance`.
            if let (Some(bk), Some(root)) = (bookkeeper, class_root.as_deref())
                && majit_ir::descr::is_list_container_spelling(root)
            {
                cells.push(bk.project_struct_field_type(root));
                continue;
            }
            if matches!(ty, crate::model::ValueType::Ref(_)) {
                // Rust's `gc_roots::pin_root(PyObjectRef)` parameter is the
                // source spelling of RPython GC-transformer's opaque root
                // slot.  The front marks that exact input as `GCREF`; seed a
                // SomePtr(GCREF), not the W_Root-like SomeInstance fallback,
                // so heterogeneous StringRepr / InstanceRepr livevars meet
                // only after their ordinary cast_opaque_ptr conversion.
                if class_root.as_deref() == Some("GCREF") {
                    cells.push(
                        crate::translator::rtyper::llannotation::lltype_to_annotation(
                            crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
                        ),
                    );
                    continue;
                }
                // A tuple param carries its `Tuple<A,B>` shape; its `.N`
                // reads are `getitem`s on the RPython tuple (`rtuple.py`),
                // so seed the `SomeTuple`, not the shape's nominal classdef.
                if let (Some(bk), Some(root)) = (bookkeeper, class_root.as_deref())
                    && majit_ir::descr::is_shaped_tuple_name(root)
                {
                    cells.push(bk.project_shaped_tuple(root));
                    continue;
                }
                // String-typed params are string values, not class
                // instances: `String` and `str` both map to the byte
                // string type (`s_str0` = `SomeString(no_nul=True)`,
                // matching `project_struct_field_type`).  String literals
                // lower through `__str_const` to `ConstValue::ByteStr`
                // (flowspace_adapter, stamped `Ptr(STR)`/`StringRepr`),
                // so a literal flowing into a `&str`/`String` param must
                // meet the same byte `SomeString` — seeding `s_unicode0`
                // here instead raised `str ∪ unicode` at `mergeinputargs`.
                // The foreign `alloc::string::String` TypeDecl also
                // registers struct-field rows, so the registry path below
                // would otherwise seed a `SomeInstance(String)` shell whose
                // field writes poison classdef attr cells with
                // instance-annotated strings.
                if class_root.as_deref() == Some("String") || class_root.as_deref() == Some("str") {
                    cells.push(crate::annotator::model::s_str0());
                    continue;
                }
                // `BytesBlock` is the low-level owner of Python bytes'
                // RPython-shaped `STR.chars` storage, not a user-visible
                // nominal instance.  The MIR front folds
                // `bytes_block_chars`'s exact `(chars, length)` raw-slice
                // view back to this header, so seed the header with the same
                // `SomeString` cell that `project_struct_field_type` assigns it.
                // A generic raw-slice owner does not take this path.
                if class_root.as_deref() == Some("BytesBlock") {
                    cells.push(crate::annotator::model::s_str0());
                    continue;
                }
                if let (Some(root), Some(bk)) = (class_root.as_ref(), bookkeeper) {
                    // A registered receiver-driven dispatch family (issue
                    // #346): a `dyn Trait` receiver whose bound-trait
                    // `class_root` names a family opted in through
                    // `register_trait_families` seeds the family base
                    // `ClassDef`.  Its impl subclasses carry the methods
                    // (attrfamily merge), so a method getattr on the
                    // receiver resolves the impl `MethodDesc` family
                    // instead of blocking on the classdef-less shell.
                    // Checked before the unique-impl substitution below —
                    // a multi-impl family has no unique impl to fold to.
                    let family_base = bk.trait_family_bases.borrow().get(root.as_str()).cloned();
                    if let Some(base_host) = family_base {
                        let cd = bk.getuniqueclassdef(&base_host).map_err(|e| {
                            TyperError::message(format!(
                                "derive_subject_inputcells: startblock.inputargs[{idx}] \
                                 ({var:?}) trait-family base for {root:?} failed \
                                 ClassDef registration: {e:?}"
                            ))
                        })?;
                        cells.push(crate::annotator::model::SomeValue::Instance(
                            crate::annotator::model::SomeInstance::new(
                                Some(cd),
                                false,
                                std::collections::BTreeMap::new(),
                            ),
                        ));
                        continue;
                    }
                    // A generic param (`&T` where `T: Trait`, incl. a
                    // trait default body's `&Self`) carries the bound
                    // trait's qualified path as `class_root`
                    // (`tyref_generic_trait_bound_root`).  When the
                    // analyzed world has exactly one concrete impl of
                    // that trait, the receiver's only possible shape is
                    // that impl type — substitute its struct root and
                    // seed below as if the param were typed concretely.
                    let root = bk
                        .trait_unique_impls
                        .borrow()
                        .get(root)
                        .cloned()
                        .unwrap_or_else(|| root.clone());
                    let root = &root;
                    let known = bk
                        .struct_fields
                        .borrow()
                        .as_ref()
                        .is_some_and(|reg| reg.fields.contains_key(root))
                        // A `MutRef<T>` cell is a synthetic one-field class with no
                        // struct-field rows, interned by qualname like its constructor.
                        || majit_ir::descr::strip_instantiation_suffix(root) == "MutRef";
                    if known {
                        // A `&FixedObjectArray` receiver models as its
                        // `_items` element list, not the wrapping struct
                        // (project_struct_field_type), so an `arr[idx]` access
                        // resolves as a list `getitem` rather than rewriting
                        // to `getattr("__getitem__")`.
                        if majit_ir::descr::canonical_struct_name(root)
                            == "object_array::FixedObjectArray"
                        {
                            cells.push(bk.project_struct_field_type(root));
                            continue;
                        }
                        // A `&RawStruct` is the address of that struct
                        // (`Ptr` of a raw `Struct`, `history.getkind`).
                        // FieldRead of the same owner already seeds
                        // SomePtr (`project_struct_field_type`); minting
                        // SomeInstance here made mergeinputargs raise
                        // Instance(X) ∪ Ptr(X) (`llannotation.py`
                        // pairtype(SomePtr, SomeObject).union).
                        if let Some(cell) = bk.raw_struct_ptr_annotation(root) {
                            cells.push(cell);
                            continue;
                        }
                        let cd = bk.getuniqueclassdef_for_struct_root(root).map_err(|e| {
                            TyperError::message(format!(
                                "derive_subject_inputcells: startblock.inputargs[{idx}] \
                                 ({var:?}) `Ref` receiver root {root:?} failed struct-root \
                                 ClassDef registration: {e:?}"
                            ))
                        })?;
                        cells.push(crate::annotator::model::SomeValue::Instance(
                            crate::annotator::model::SomeInstance::new(
                                Some(cd),
                                false,
                                std::collections::BTreeMap::new(),
                            ),
                        ));
                        continue;
                    }
                }
            }
            if crate::translator::rtyper::cutover::rtyper_verbose_enabled()
                && matches!(ty, crate::model::ValueType::Ref(_))
            {
                let payload = match ty {
                    crate::model::ValueType::Ref(p) => p.as_deref(),
                    _ => None,
                };
                let canon = class_root
                    .as_deref()
                    .map(majit_ir::descr::canonical_struct_name);
                let (raw_known, canon_known) = match (bookkeeper, class_root.as_deref()) {
                    (Some(bk), Some(root)) => {
                        let guard = bk.struct_fields.borrow();
                        let raw_known = guard
                            .as_ref()
                            .is_some_and(|reg| reg.fields.contains_key(root));
                        let canon_known = canon.as_ref().is_some_and(|c| {
                            guard.as_ref().is_some_and(|reg| reg.fields.contains_key(c))
                        });
                        (raw_known, canon_known)
                    }
                    _ => (false, false),
                };
                crate::codewriter::annotation_state::record_classdef_less_input(
                    &legacy.name,
                    &var.name_prefix(),
                    payload,
                    class_root.as_deref(),
                    canon.as_deref(),
                    raw_known,
                    canon_known,
                    bookkeeper.is_some(),
                );
            }
            // An `Int` input whose `class_root` names a Raw owner is the
            // address of that struct (`Ptr` of a raw `Struct`,
            // `history.getkind`). Fieldless enums are also `Int` and carry
            // a class root; `raw_struct_ptr_annotation` leaves those as the
            // integer shell.
            if matches!(ty, crate::model::ValueType::Int)
                && let Some(root) = class_root.as_deref()
                && let Some(bk) = bookkeeper
                && let Some(cell) = bk.raw_struct_ptr_annotation(root)
            {
                cells.push(cell);
                continue;
            }
            cells.push(shell);
            continue;
        }
        // 2. Explicit SomeValue fallback published onto `var.annotation`.
        // Production front graphs always took the Input arm above; only
        // hand-built fixtures without Input ops rely on this path.
        if let Some(rc) = var.annotation.borrow().as_ref() {
            cells.push((**rc).clone());
            continue;
        }
        // No further fallback: every typed parameter emits the Input
        // op alongside the inputargs registration; reaching here implies
        // the inputarg has neither a published `Variable.annotation`
        // shell nor a startblock Input op.
        return Err(TyperError::message(format!(
            "derive_subject_inputcells: startblock.inputargs[{idx}] \
             ({var:?}) has no matching `OpKind::Input {{ ty }}` op at \
             the startblock and no `Variable.annotation` shell — \
             front-end producer divergence (every typed parameter emits \
             the Input op alongside the inputargs registration; see \
             `front::mir`)"
        )));
    }
    Ok(cells)
}

/// `remove_dead_blocks` (`translator/simplify.py`) — the set of blocks
/// reachable from `startblock` by following `Block.exits`.  The rtyper
/// only annotates reachable blocks, so the flowspace adapter must drop
/// everything else.  The monotonic front-end leaves model-unreachable
/// `on_unwind` cleanup blocks in the graph: `lower_call` / `Drop` /
/// `Assert` forward past the Rust panic-table unwind edge
/// (`front::mir`), and the orphan cleanup block is closed by `set_raise`
/// (`model.rs::set_raise`).  `remove_dead_blocks` drops it before rtyping,
/// exactly as RPython annotates only the startblock-reachable closure;
/// without this prune the orphan block would be rtyped even though it can
/// never execute.  The `Block.dead`
/// flag covers only the framestate path's explicit marking
/// (`mir.rs::lower_framestate`); startblock reachability is the general
/// case the `dead` skip sites below also honour.
/// `__strlen` on a `Wtf8`/`Str` cast reads the wrapper. `W_UnicodeObject._utf8`
/// is the rstr (`unicodeobject.py`); the length is `len` of that one
/// string-typed field. A layout with any other count of string fields
/// declines — `ll_strlen` on the wrapper reads `PyObject.w_class`.
///
/// A block inputarg is that cast when every predecessor `Link` resolves to
/// the same Wtf8 source. `Ok(None)` when this call is not that cast, so a
/// real string receiver still lowers as `len`.
fn rewrite_wtf8_view_strlen(
    legacy: &FunctionGraph,
    op: &SpaceOperation,
    value_map: &HashMap<Variable, Hlvalue>,
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
) -> Result<Option<Vec<FlowspaceOp>>, TyperError> {
    let Some(view) = strlen_call_arg(&op.kind) else {
        return Ok(None);
    };
    let reachable = reachable_block_ids(legacy);
    let obj = match resolve_strlen_view(legacy, view, 0, &mut Vec::new(), &reachable)? {
        StrlenView::Cast(obj) => obj,
        StrlenView::NotCast | StrlenView::Cycle => return Ok(None),
    };
    let Some(class_name) =
        op_defining(legacy, &obj).and_then(|producer| input_class_name(&producer.kind))
    else {
        return Err(TyperError::message(
            "__strlen on a Wtf8 view: receiver class unresolved \
             (wtf8-strlen-unknown-receiver)",
        ));
    };
    let field_name = unique_string_field(call_registry, class_name).map_err(|why| {
        TyperError::message(format!(
            "__strlen on a Wtf8 view of {class_name} has no unique string field ({why})"
        ))
    })?;
    let receiver = lookup_operand(value_map, &obj, op, "strlen receiver")?;
    let field = Hlvalue::Variable(Variable::new());
    let result = resolve_result_hlvalue(op, value_map)?;
    Ok(Some(vec![
        FlowspaceOp::new(
            "getattr",
            vec![
                receiver,
                Hlvalue::Constant(Constant::new(ConstValue::byte_str(field_name))),
            ],
            field.clone(),
        ),
        FlowspaceOp::new("len", vec![field], result),
    ]))
}

fn strlen_call_arg(kind: &OpKind) -> Option<&Variable> {
    let OpKind::Call {
        target: crate::model::CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = kind
    else {
        return None;
    };
    if segments.as_slice() != ["__strlen"] || args.len() != 1 {
        return None;
    }
    args.first().and_then(LinkArg::as_variable)
}

enum StrlenView {
    Cast(Variable),
    NotCast,
    /// Re-entered phi. It adds no producer of its own.
    Cycle,
}

fn strlen_unresolved() -> TyperError {
    TyperError::message(
        "__strlen on a Wtf8 view: receiver class unresolved \
         (wtf8-strlen-unknown-receiver)",
    )
}

/// Operation result, or the one Wtf8 cast every predecessor link carries.
/// A phi that is not a Wtf8 cast stays `NotCast` so a real string `len`
/// still lowers. A cast that does not unify across incoming edges declines.
fn resolve_strlen_view(
    legacy: &FunctionGraph,
    var: &Variable,
    depth: usize,
    stack: &mut Vec<Variable>,
    reachable: &std::collections::HashSet<BlockId>,
) -> Result<StrlenView, TyperError> {
    if depth > 32 {
        return Err(strlen_unresolved());
    }
    if stack.iter().any(|seen| seen == var) {
        return Ok(StrlenView::Cycle);
    }
    if let Some(op) = op_defining(legacy, var) {
        return Ok(match wtf8_str_cast_source(&op.kind) {
            Some(obj) => StrlenView::Cast(obj.clone()),
            None => StrlenView::NotCast,
        });
    }
    let mut incoming: Vec<crate::model::LinkArg> = Vec::new();
    let mut saw_input = false;
    for block in &legacy.blocks {
        // `function_graph_to_flowspace_inner` drops the same blocks:
        // `dead` or not in `reachable_block_ids`. A dead predecessor
        // must not feed the phi.
        if block.dead || !reachable.contains(&block.id) {
            continue;
        }
        let Some(idx) = block.inputargs.iter().position(|input| input == var) else {
            continue;
        };
        saw_input = true;
        for pred in &legacy.blocks {
            if pred.dead || !reachable.contains(&pred.id) {
                continue;
            }
            for link in &pred.exits {
                if link.target != block.id {
                    continue;
                }
                match link.args.get(idx) {
                    Some(arg) => incoming.push(arg.clone()),
                    None => return Err(strlen_unresolved()),
                }
            }
        }
    }
    if !saw_input || incoming.is_empty() {
        return Ok(StrlenView::NotCast);
    }
    stack.push(var.clone());
    let mut cast_src: Option<Variable> = None;
    let mut saw_not = false;
    for arg in &incoming {
        let view = match arg {
            crate::model::LinkArg::Value(value) => {
                resolve_strlen_view(legacy, value, depth + 1, stack, reachable)?
            }
            crate::model::LinkArg::Const(_) => StrlenView::NotCast,
        };
        match view {
            StrlenView::Cast(obj) => {
                if saw_not {
                    return Err(strlen_unresolved());
                }
                if let Some(prev) = &cast_src {
                    if prev != &obj {
                        return Err(strlen_unresolved());
                    }
                } else {
                    cast_src = Some(obj);
                }
            }
            StrlenView::NotCast => {
                if cast_src.is_some() {
                    return Err(strlen_unresolved());
                }
                saw_not = true;
            }
            StrlenView::Cycle => {}
        }
    }
    stack.pop();
    if let Some(obj) = cast_src {
        Ok(StrlenView::Cast(obj))
    } else {
        Ok(StrlenView::NotCast)
    }
}

fn op_defining<'a>(legacy: &'a FunctionGraph, var: &Variable) -> Option<&'a SpaceOperation> {
    legacy.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(var))
    })
}

fn wtf8_str_cast_source(kind: &OpKind) -> Option<&Variable> {
    if crate::model::cast_instance_root(kind) != Some("Wtf8") {
        return None;
    }
    let OpKind::Call {
        result_ty, args, ..
    } = kind
    else {
        return None;
    };
    if *result_ty != crate::model::ValueType::Str {
        return None;
    }
    args.first().and_then(LinkArg::as_variable)
}

fn input_class_name(kind: &OpKind) -> Option<&str> {
    let OpKind::Input { class_root, ty, .. } = kind else {
        return None;
    };
    if let Some(name) = class_root.as_deref().filter(|name| !name.is_empty()) {
        return Some(name);
    }
    match ty {
        crate::model::ValueType::Ref(Some(name)) => Some(name.as_str()),
        _ => None,
    }
}

fn unique_string_field(
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
    class_name: &str,
) -> Result<String, &'static str> {
    let bk = call_registry.bookkeeper();
    let rows = {
        let borrowed = bk.struct_fields.borrow();
        let Some(reg) = borrowed.as_ref() else {
            return Err("no struct field layout");
        };
        let Some(rows) = layout_rows(reg, class_name) else {
            return Err("class has no registered fields");
        };
        rows.to_vec()
    };
    let mut found: Option<String> = None;
    for (name, ty) in &rows {
        if !matches!(
            bk.project_struct_field_type(ty),
            crate::annotator::model::SomeValue::String(_)
        ) {
            continue;
        }
        if found.is_some() {
            return Err("more than one string field");
        }
        found = Some(name.clone());
    }
    found.ok_or("no string field")
}

fn layout_rows<'a>(
    reg: &'a crate::front::StructFieldRegistry,
    class_name: &str,
) -> Option<&'a [(String, String)]> {
    if class_name.is_empty() {
        return None;
    }
    if let Some(rows) = reg.fields.get(class_name) {
        return Some(rows.as_slice());
    }
    let leaf = class_name.rsplit("::").next().unwrap_or(class_name);
    let canonical = majit_ir::descr::canonical_struct_name(leaf);
    if let Some(rows) = reg.fields.get(&canonical) {
        return Some(rows.as_slice());
    }
    let mut found: Option<&[(String, String)]> = None;
    for (key, rows) in &reg.fields {
        let suffix = key.ends_with(&format!("::{class_name}"))
            || class_name.ends_with(&format!("::{key}"))
            || key == leaf;
        if !suffix {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(rows.as_slice());
    }
    found
}

fn reachable_block_ids(legacy: &FunctionGraph) -> std::collections::HashSet<BlockId> {
    // Index lookup instead of `FunctionGraph::block` (a dense
    // `blocks[id.0]` projection): final blocks reached as link targets
    // need no `Block` entry of their own (test fixtures omit them), and
    // a final block contributes no exits anyway.  Mirrors `iterblocks`'
    // `exits[::-1]` stack order.
    let by_id: HashMap<BlockId, &crate::model::Block> =
        legacy.blocks.iter().map(|b| (b.id, b)).collect();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![legacy.startblock];
    while let Some(id) = stack.pop() {
        if seen.insert(id)
            && let Some(block) = by_id.get(&id)
        {
            stack.extend(block.exits.iter().rev().map(|e| e.target));
        }
    }
    seen
}

/// `RPythonTyper.specialize`'s per-graph entry: build the flowspace view of one
/// legacy graph.
///
/// Under `MAJIT_RTYPER_FRONTIER` the run resumes past every unresolvable call
/// instead of stopping at the first, so [`take_frontier_walls`] can report the
/// subject's whole call-wall set.  Resuming is census-only: the graph it builds
/// is discarded, and the error reported here is the one the gate-off run would
/// have returned, so the subject fails identically either way.
///
/// **Only call ops resume.**  A call's resolution failure is a function of the
/// callee path and the registry alone, so skipping it invents no wall; any
/// other op's failure may be a consequence of an operand the run skipped, and
/// resuming past it would manufacture walls that do not exist.  Such an op
/// still stops the run — its error is then replaced by the recorded first wall,
/// which is where the gate-off run stopped.
///
/// Measured on the `pyre-jit-trace` prepass census: the gate-off phase-A
/// failure set is reproducible (1681 subjects, twice, byte-identical), and the
/// gate moves exactly one of them — `_io::stringio::W_StringIO::default` stops
/// panicking in annotate and reaches phase B instead.  Read the ranking as
/// carrying that one-subject error bar.
pub fn function_graph_to_flowspace(
    legacy: &FunctionGraph,
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
) -> Result<FlowspaceAdapterOutput, TyperError> {
    let _guard = frontier::enter();
    let result = function_graph_to_flowspace_inner(legacy, call_registry);
    match frontier::pending_error() {
        Some(first) => Err(first),
        None => result,
    }
}

/// Translate one legacy graph, reporting the first wall it hits.
///
/// Under [`frontier`] the run resumes past a skippable call wall, so the error
/// this returns need not be the one it stopped at —
/// [`function_graph_to_flowspace`] restores it.
///
/// One-way conversion from the legacy `crate::model::FunctionGraph`
/// into a `flowspace::FunctionGraph` whose blocks carry `Hlvalue`
/// operands and per-value `SomeValue` annotations on its `Variable`s.
///
/// Two-pass topology assembly:
///
/// 1. **Pass 1** — allocate one `flowspace::BlockRef` per legacy
///    non-final block, allocating fresh `Variable`s for each block's
///    inputargs. Assemble the `flowspace::FunctionGraph` via
///    `FunctionGraph::with_return_var`, supplying the canonical
///    returnblock inputarg so the rtyper's `getreturnvar`
///    (`rtyper.rs`) finds a real return `Variable`.
/// 2. **Pass 2** — for each non-final block, translate `operations` via
///    [`translate_op`], translate `exits` (link args + targets +
///    exitcase) via [`link_arg_to_hlvalue`] / [`exitcase_to_hlvalue`],
///    and translate `exitswitch` via the `value_map`.
///
/// Topology assembly delegates per-OpKind operation translation to
/// [`translate_op`], which means any
/// legacy graph carrying an
/// unported OpKind variant surfaces a fail-loud `TyperError` from this
/// function. Trivial graphs (only `Input` / `ConstInt` / `ConstFloat`
/// op definitions) flow through cleanly.
///
/// Addpendingblock conversion — production path no longer
/// pre-seeds `Variable.annotation` from `legacy_annotator::annotate`.
/// Once the cutover entry queues the subject's startblock onto the
/// orthodox `addpendingblock` queue
/// (`cutover.rs:specialize_legacy_graph_with_registry_returning_value_to_var`),
/// `complete_pending_blocks` drives `flowin` which writes
/// `Variable.annotation` for every reachable inputarg and op result.
/// Carrying the legacy pre-seed alongside flowin caused
/// `setbinding: new value does not contain old` panics at
/// `annrpython.rs` whenever flowin's `follow_link` computed a
/// narrower annotation (e.g., constant-tracking `SomeInteger{const,
/// nonneg}`) than legacy_annotator's wider lift.
///
/// Test fixtures that hand-roll minimal SSA graphs without
/// production-shape `OpKind::Input { ty }` ops must seed
/// `legacy.variable(vid).annotation` directly via
/// `legacy_annotator::setbinding(&var, ValueType::…)` before calling
/// this function so `seed_variable` reads the right shell.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn function_graph_to_flowspace_inner(
    legacy: &FunctionGraph,
    // Call resolution plumbing — see [`translate_op`].
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
) -> Result<FlowspaceAdapterOutput, TyperError> {
    let mut value_to_var: LegacyToTyped = HashMap::new();
    let mut value_to_var_candidates: LegacyToTypedCandidates = HashMap::new();
    let mut constant_hlvalues: HashMap<Variable, Hlvalue> = HashMap::new();
    let mut constant_concretetypes: HashMap<Variable, LowLevelType> = HashMap::new();

    // `remove_dead_blocks` reachable set — blocks not reachable from
    // `startblock` are dropped from every pass below alongside the
    // explicit `dead` flag.  See [`reachable_block_ids`].
    let reachable = reachable_block_ids(legacy);

    for legacy_block in &legacy.blocks {
        // Dead / unreachable blocks are pruned from the flowspace
        // (`remove_dead_blocks` parity), so their const-defines are never
        // referenced — skip them here too rather than stamp concretetypes
        // onto orphan cells.
        if legacy_block.dead || !reachable.contains(&legacy_block.id) {
            continue;
        }
        for legacy_op in &legacy_block.operations {
            if let Some(hlvalue) =
                legacy_const_define_hlvalue_or_frontier(legacy_op, Some(call_registry))?
            {
                // `legacy_const_define_hlvalue` only returns `Some` for ops
                // with a result Variable, so this is always present here.
                let Some(result_var) = legacy_op.result.as_ref() else {
                    continue;
                };
                if let Hlvalue::Constant(c) = &hlvalue
                    && let Some(ct) = &c.concretetype
                {
                    let legacy_ct = if is_fn_const_define(&legacy_op.kind) {
                        // The rtyper-side carrier is a real LLPtr so
                        // direct-call resolution can materialise
                        // `getfunctionptr` from the function graph.  The
                        // legacy/codewriter side must still see the
                        // synthetic define's declared Int slot: later
                        // jtransform materialises funcptr values as
                        // `ConstFnAddr` and the assembler encodes
                        // them through the `i` argcode.
                        LowLevelType::Signed
                    } else {
                        ct.clone()
                    };
                    constant_concretetypes.insert(result_var.clone(), legacy_ct.clone());
                    // Also stamp the lltype onto the legacy graph's
                    // orphan Variable cell for this const-define
                    // result.  The rtyper consumes `Hlvalue::Constant`
                    // surfaces for const-defines and never reads the
                    // legacy Variable cell directly, so the write is
                    // additive — `RPythonTyper.specialize` won't
                    // overwrite this slot.  Downstream consumers
                    // reading `FunctionGraph::concretetype_of(&v)`
                    // (RPython parity for `getkind(v.concretetype)`)
                    // then see the const kind inline, without
                    // depending on the post-rtyper
                    // `apply_to_graph(constant_concretetypes, …)`
                    // bridge.
                    result_var.set_concretetype(Some(legacy_ct));
                }
                constant_hlvalues.insert(result_var.clone(), hlvalue);
            }
        }
    }

    // A flowspace graph contains only blocks reachable from
    // `startblock`: upstream builds graphs by abstract interpretation,
    // so an unreachable block cannot exist, and every downstream
    // consumer (checkgraph / annotator / rtyper) walks `iterblocks()`
    // — a reachability DFS (`flowspace/model.rs`, model.py).  The
    // legacy MIR graph, by contrast, keeps lowered-but-unreachable
    // blocks: every `on_unwind` edge is dropped at lowering while the
    // `UnwindResume`/`Abort` terminator still lowers via `set_raise`,
    // leaving a predecessor-less block whose orphan `[etype, evalue]`
    // Link.args are defined by no inputarg and no op result.  Convert
    // only the reachable closure — the same `reachable` set the const
    // prepass above filtered on, so a block is translated iff its
    // const-defines were seeded.  A *reachable* block with the same
    // orphan shape still fails the operand-definedness check in
    // `link_arg_to_hlvalue` — SSA-definedness is not relaxed.

    // ──────────────────────────────────────────────────────────────
    // Pass 1 — allocate fresh `flowspace::BlockRef` for every legacy
    // reachable non-final block. The legacy `returnblock` and
    // `exceptblock` are skipped here; `FunctionGraph::with_return_var`
    // allocates the canonical flowspace finals, and the block_map is
    // populated with those after graph construction.
    // ──────────────────────────────────────────────────────────────

    let mut block_map: HashMap<BlockId, BlockRef> = HashMap::new();
    let mut block_inputarg_vars: HashMap<BlockId, Vec<(Variable, Variable)>> = HashMap::new();

    for legacy_block in &legacy.blocks {
        // `dead` / unreachable blocks are removed before annotation
        // (`remove_dead_blocks` parity).  The framestate path marks
        // model-unreachable orphan `on_unwind` blocks `dead`; the
        // monotonic path leaves them unmarked but startblock-unreachable,
        // so `reachable` is what prunes them there.  Skipping Pass 1 alloc
        // keeps them out of `block_map`, which Pass 2 must mirror (it
        // indexes `block_map`).  No reachable block can target a pruned
        // block — a target is reachable by construction — so the mirror
        // stays consistent.
        if legacy_block.dead
            || !reachable.contains(&legacy_block.id)
            || legacy_block.id == legacy.returnblock
            || legacy_block.id == legacy.exceptblock
        {
            continue;
        }
        let mut local_inputs: Vec<(Variable, Variable)> =
            Vec::with_capacity(legacy_block.inputargs.len());
        let mut inputargs: Vec<Hlvalue> = Vec::with_capacity(legacy_block.inputargs.len());
        for legacy_var in legacy_block.inputargs.iter() {
            let var = seed_variable(legacy_var);
            value_to_var_candidates
                .entry(legacy_var.clone())
                .or_default()
                .push(var.clone());
            value_to_var
                .entry(legacy_var.clone())
                .or_insert_with(|| var.clone());
            local_inputs.push((legacy_var.clone(), var.clone()));
            inputargs.push(Hlvalue::Variable(var));
        }
        block_inputarg_vars.insert(legacy_block.id, local_inputs);
        block_map.insert(legacy_block.id, FlowspaceBlock::shared(inputargs));
    }

    let startblock = block_map.get(&legacy.startblock).cloned().ok_or_else(|| {
        TyperError::message(format!(
            "function_graph_to_flowspace: legacy.startblock {:?} not in legacy.blocks",
            legacy.startblock
        ))
    })?;

    // Resolve the returnblock's inputarg as a fresh final-block
    // Variable. Even when the legacy graph reuses the source slot
    // here, RPython's checkgraph treats target inputargs as definitions
    // in the target block, not as the predecessor's Variable object.
    //
    // RPython `model.py` `FunctionGraph.getreturnvar`
    // returns `self.returnblock.inputargs[0]` unconditionally — there
    // is no fallback for an empty `inputargs` list, and a malformed
    // graph raises `IndexError` at this site rather than fabricating a
    // fresh Variable.  Pyre's `model::FunctionGraph::with_return_var`
    // builds the returnblock with `inputargs:
    // vec![return_value]` by invariant, so the lookup is guaranteed to
    // succeed; surface the violation as a `TyperError` instead of
    // silently producing a `Variable::new()` placeholder.
    let return_var = legacy
        .blocks
        .iter()
        .find(|b| b.id == legacy.returnblock)
        .and_then(|b| {
            let legacy_var = b.inputargs.first()?;
            Some(legacy_var.clone())
        })
        .map(|legacy_var| {
            let var = seed_variable(&legacy_var);
            value_to_var_candidates
                .entry(legacy_var.clone())
                .or_default()
                .push(var.clone());
            value_to_var
                .entry(legacy_var)
                .or_insert_with(|| var.clone());
            Hlvalue::Variable(var)
        })
        .ok_or_else(|| {
            TyperError::message(format!(
                "function_graph_to_flowspace: legacy graph {:?} has no \
                 returnblock {:?} with at least one inputarg — \
                 `model::FunctionGraph::with_return_var` \
                 builds the returnblock with `inputargs: vec![return_value]` \
                 by invariant; matches RPython `model.py` `getreturnvar` \
                 which indexes `returnblock.inputargs[0]` \
                 without a fallback",
                legacy.name, legacy.returnblock,
            ))
        })?;

    let graph = FlowspaceGraph::with_return_var(legacy.name.clone(), startblock, return_var);
    let returnblock_ref = graph.returnblock.clone();
    let exceptblock_ref = graph.exceptblock.clone();

    if let Some(legacy_exceptblock) = legacy.blocks.iter().find(|b| b.id == legacy.exceptblock)
        && legacy_exceptblock.inputargs.len() == 2
    {
        let mut except_inputargs = Vec::with_capacity(2);
        for legacy_var in legacy_exceptblock.inputargs.iter() {
            let var = seed_variable(legacy_var);
            value_to_var_candidates
                .entry(legacy_var.clone())
                .or_default()
                .push(var.clone());
            value_to_var
                .entry(legacy_var.clone())
                .or_insert_with(|| var.clone());
            except_inputargs.push(Hlvalue::Variable(var));
        }
        exceptblock_ref.borrow_mut().inputargs = except_inputargs;
    }

    let graph_ref = graph.into_ref();

    // Map the canonical finals so any legacy Link targeting them
    // resolves to the flowspace finals constructed above.
    block_map.insert(legacy.returnblock, returnblock_ref);
    block_map.insert(legacy.exceptblock, exceptblock_ref);

    // ──────────────────────────────────────────────────────────────
    // Pass 2 — fill operations + exits + exitswitch for each reachable
    // non-final legacy block. Final blocks (returnblock / exceptblock)
    // are already terminal in flowspace — `mark_final()` was set by
    // `FunctionGraph::with_return_var`.
    // ──────────────────────────────────────────────────────────────

    for legacy_block in &legacy.blocks {
        // Mirror Pass 1's skip set exactly — a `dead` / unreachable block
        // has no `block_map` entry, so translating it would panic at the
        // index below.  See the Pass 1 comment for the `remove_dead_blocks`
        // rationale.
        if legacy_block.dead
            || !reachable.contains(&legacy_block.id)
            || legacy_block.id == legacy.returnblock
            || legacy_block.id == legacy.exceptblock
        {
            continue;
        }
        let block_ref = block_map[&legacy_block.id].clone();
        let mut value_map = constant_hlvalues.clone();
        let mut name_to_value: HashMap<String, Hlvalue> = HashMap::new();

        if let Some(inputs) = block_inputarg_vars.get(&legacy_block.id) {
            for (legacy_var, var) in inputs {
                let hlvalue = Hlvalue::Variable(var.clone());
                value_map.insert(legacy_var.clone(), hlvalue.clone());
                if let Some(name) = legacy.value_name_for(legacy_var) {
                    name_to_value.entry(name.to_string()).or_insert(hlvalue);
                }
            }
        }
        for legacy_op in &legacy_block.operations {
            if let (
                Some(result_var),
                OpKind::Input {
                    name,
                    ty: _,
                    class_root: _,
                },
            ) = (legacy_op.result.as_ref(), &legacy_op.kind)
                && legacy_block.inputargs.contains(result_var)
                && let Some(existing) = value_map.get(result_var).cloned()
            {
                name_to_value.entry(name.clone()).or_insert(existing);
            }
        }

        // Translate operations.
        let mut translated_ops: Vec<FlowspaceOp> = Vec::new();
        let tuple_aggregates: HashMap<Variable, TupleAggregate> = legacy_block
            .operations
            .iter()
            .enumerate()
            .filter_map(|(index, op)| {
                let shape = tuple_aggregate_shape(&op.kind)?;
                let result = op.result.as_ref()?.clone();
                let aggregate =
                    tuple_aggregate_items(&legacy_block.operations, index, &result, shape)?;
                Some((result, aggregate))
            })
            .collect();
        let array_list_elements: HashMap<Variable, Vec<Variable>> = legacy_block
            .operations
            .iter()
            .filter_map(|op| {
                if !is_array_aggregate_ctor(&op.kind) {
                    return None;
                }
                let result = op.result.as_ref()?.clone();
                let elements = array_aggregate_elements(&legacy_block.operations, &result)?;
                Some((result, elements))
            })
            .collect();
        for (op_index, legacy_op) in legacy_block.operations.iter().enumerate() {
            if let Some(hlvalue) =
                legacy_const_define_hlvalue_or_frontier(legacy_op, Some(call_registry))?
            {
                if let Some(result_var) = legacy_op.result.as_ref() {
                    value_map.insert(result_var.clone(), hlvalue.clone());
                    if let Some(name) = legacy.value_name_for(result_var) {
                        name_to_value.insert(name.to_string(), hlvalue);
                    }
                }
                translated_ops.extend(translate_op_or_frontier(
                    legacy_op,
                    &value_map,
                    call_registry,
                )?);
                continue;
            }

            if let (
                Some(result_var),
                OpKind::Input {
                    name,
                    ty: _,
                    class_root: _,
                },
            ) = (legacy_op.result.as_ref(), &legacy_op.kind)
            {
                if !value_map.contains_key(result_var) {
                    if let Some(alias) = name_to_value.get(name).cloned() {
                        // Same-block name match: alias the body `Input`
                        // result to the prior `Hlvalue` for `name`.
                        // Mirrors `front::mir`'s same-block
                        // LOAD_FAST dedup that the front already
                        // enforces (no SSA divergence at this site).
                        if let Hlvalue::Variable(var) = &alias {
                            value_to_var
                                .entry(result_var.clone())
                                .or_insert_with(|| var.clone());
                        }
                        value_map.insert(result_var.clone(), alias);
                    } else {
                        // Cross-block body `Input`: name not in the
                        // current block's `name_to_value`.  RPython
                        // flowspace has no analogue for this — every
                        // local cross-block reference is threaded via
                        // the predecessor `Link.args` and the target
                        // block's `inputargs` (`flowcontext.py
                        // LOAD_FAST` writes the local into
                        // `self.locals_w`; cross-block reads always go
                        // through the target block's pre-allocated
                        // `inputargs[]`, never via a fresh Variable).
                        //
                        // Fail-loud here surfaces the producer-side
                        // gap: either the target block's inputargs were
                        // not extended to carry `name`, or the
                        // predecessor link's args do not include the
                        // slot producer.  The dual-gate at
                        // `cutover.rs`'s `is_known_unported` matches
                        // the substring `"adapter cross-block body
                        // Input"` and Skip-classifies the graph,
                        // routing it through `legacy_state` until
                        // cross-block locals threading covers
                        // every shape.
                        return Err(TyperError::message(format!(
                            "translate_op: adapter cross-block body Input — \
                             name {name:?} (result {result_var:?}) was not threaded \
                             through Link.args / target inputargs by the \
                             predecessor block.  RPython has no body-`Input` \
                             op (`flowcontext.py` `LOAD_FAST` writes locals \
                             into self.locals_w; cross-block reads go via the \
                             target block's pre-allocated inputargs).  Producer \
                             gap — either Cat 2.1 cross-block locals threading \
                             missed this shape, or the front-end's body-`Input` \
                             emission needs to be extended to predeclare the \
                             name in the predecessor link."
                        )));
                    }
                }
                translated_ops.extend(translate_op_or_frontier(
                    legacy_op,
                    &value_map,
                    call_registry,
                )?);
                continue;
            }

            // Skip Abort here for the same reason
            // `build_value_to_variable_map` skips it — `translate_op`
            // emits no flowspace op for Abort, so seeding its result
            // would leave the consumer's flowspace arg referencing a
            // never-defined Variable.  Letting `lookup_operand` fail
            // at the first consumer surfaces the orthodox
            // "undefined operand" message that
            // `is_known_unported` classifies as Skip at the
            // producer-adjacent site.
            if let Some(result_var) = legacy_op.result.as_ref()
                && !value_map.contains_key(result_var)
                && !matches!(legacy_op.kind, OpKind::Abort { .. })
            {
                // The op result is `result_var`'s sole definition site, so
                // its freshly-seeded flowspace Variable is the authority for
                // `value_to_var` — `insert`, not `or_insert_with`. A use of
                // `result_var` reached earlier in block-storage order (e.g.
                // a `*_ovf` raising op whose result the `checked_arith` front
                // rewrite also threads through a link) may already have
                // seeded a *different*, block-local typed Variable under the
                // same legacy key. Keeping that earlier entry splits the
                // identity: `specialize_block` writes the `concretetype` onto
                // this op-result Variable (the one `value_map` carries and
                // `translate_op` emits), while the dual gate reads the stale
                // earlier entry — a spurious `real=Unknown` divergence. The
                // per-block-Variable invariant still holds: this seed stays
                // local to the defining block; only the legacy→typed map is
                // repointed to the definition.
                let var = seed_variable(result_var);
                value_to_var_candidates
                    .entry(result_var.clone())
                    .or_default()
                    .push(var.clone());
                value_to_var.insert(result_var.clone(), var.clone());
                value_map.insert(result_var.clone(), Hlvalue::Variable(var));
            }
            // RPython has one list model for Rust fixed arrays, slices, and
            // Vec values.  Rebuild the front-end's exact Array ctor +
            // positional writes as `newlist(items...)` only in this
            // flowspace view; the semantic graph stays intact for legacy
            // fallback/code generation.
            if let Some(result_var) = legacy_op.result.as_ref()
                && let Some(elements) = array_list_elements.get(result_var)
            {
                let mut args = Vec::with_capacity(elements.len());
                for (index, element) in elements.iter().enumerate() {
                    args.push(lookup_operand(
                        &value_map,
                        element,
                        legacy_op,
                        &format!("array_item_{index}"),
                    )?);
                }
                let result = resolve_result_hlvalue(legacy_op, &value_map)?;
                translated_ops.push(FlowspaceOp::new("newlist", args, result));
                continue;
            }
            if let OpKind::FieldWrite { base, .. } = &legacy_op.kind
                && array_list_elements.contains_key(base)
            {
                continue;
            }
            // A Rust tuple aggregate is `newtuple(items...)` in this view,
            // placed where its last item is written.
            let tuple_var = match &legacy_op.kind {
                OpKind::FieldWrite { base, .. } if tuple_aggregates.contains_key(base) => {
                    Some(base)
                }
                _ => legacy_op
                    .result
                    .as_ref()
                    .filter(|result| tuple_aggregates.contains_key(*result)),
            };
            if let Some(tuple_var) = tuple_var {
                let aggregate = &tuple_aggregates[tuple_var];
                if aggregate.at == op_index {
                    let mut args = Vec::with_capacity(aggregate.items.len());
                    for (index, item) in aggregate.items.iter().enumerate() {
                        args.push(match item {
                            crate::model::LinkArg::Value(var) => lookup_operand(
                                &value_map,
                                var,
                                legacy_op,
                                &format!("tuple_item_{index}"),
                            )?,
                            crate::model::LinkArg::Const(c) => Hlvalue::Constant(c.clone()),
                        });
                    }
                    let result = lookup_operand(&value_map, tuple_var, legacy_op, "tuple")?;
                    translated_ops.push(FlowspaceOp::new("newtuple", args, result));
                }
                continue;
            }
            if let Some(ops) =
                rewrite_wtf8_view_strlen(legacy, legacy_op, &value_map, call_registry)?
            {
                translated_ops.extend(ops);
                if let Some(result_var) = legacy_op.result.as_ref()
                    && let Some(name) = legacy.value_name_for(result_var)
                    && let Some(value) = value_map.get(result_var).cloned()
                {
                    name_to_value.insert(name.to_string(), value);
                }
                continue;
            }
            translated_ops.extend(translate_op_or_frontier(
                legacy_op,
                &value_map,
                call_registry,
            )?);
            if let Some(result_var) = legacy_op.result.as_ref()
                && let Some(name) = legacy.value_name_for(result_var)
                && let Some(value) = value_map.get(result_var).cloned()
            {
                name_to_value.insert(name.to_string(), value);
            }
        }

        // Translate exits.
        let mut translated_exits: Vec<flowspace_model::LinkRef> =
            Vec::with_capacity(legacy_block.exits.len());
        for legacy_link in &legacy_block.exits {
            let mut link_value_map = value_map.clone();
            let last_exception = legacy_link
                .last_exception
                .as_ref()
                .map(|arg| {
                    link_extravar_to_hlvalue(
                        arg,
                        "last_exception",
                        &mut link_value_map,
                        &mut value_to_var,
                    )
                })
                .transpose()?;
            let last_exc_value = legacy_link
                .last_exc_value
                .as_ref()
                .map(|arg| {
                    link_extravar_to_hlvalue(
                        arg,
                        "last_exc_value",
                        &mut link_value_map,
                        &mut value_to_var,
                    )
                })
                .transpose()?;
            let target = block_map.get(&legacy_link.target).cloned().ok_or_else(|| {
                TyperError::message(format!(
                    "function_graph_to_flowspace: legacy link target {:?} not found in \
                     legacy.blocks (block id={:?})",
                    legacy_link.target, legacy_block.id,
                ))
            })?;
            let mut args: Vec<Hlvalue> = legacy_link
                .args
                .iter()
                .enumerate()
                .map(|(idx, arg)| {
                    link_arg_to_hlvalue(
                        arg,
                        &link_value_map,
                        legacy_block.id,
                        legacy_link.target,
                        idx,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            // RPython `FlowContext.exc_from_raise` always computes
            // `w_type = op.type(w_value)` before constructing the
            // `(w_type, w_value)` link to `exceptblock`
            // (`flowspace/flowcontext.py`).  Pyre's pre-flowspace
            // model historically elided that operation and used the
            // exception value in both slots because `flatten.make_return`
            // only emits the second slot.  The rtyper observes the link
            // earlier, however: `_convert_link` assigns
            // `ExceptionData.r_exception_type` to slot 0 and must therefore
            // receive the class representation, not the instance
            // representation.
            //
            // Restore the upstream flowspace shape at this adapter boundary.
            // Limit the reconstruction to the exact legacy adaptation:
            // a direct, non-LastException link to `exceptblock` whose two
            // arguments are the same Variable.  Ordinary exception edges
            // already carry their distinct link-scoped `(last_exception,
            // last_exc_value)` variables and must remain untouched.
            let duplicated_direct_raise = legacy_link.target == legacy.exceptblock
                && legacy_link.last_exception.is_none()
                && legacy_link.last_exc_value.is_none()
                && matches!(
                    legacy_link.args.as_slice(),
                    [LinkArg::Value(etype), LinkArg::Value(evalue)] if etype == evalue
                );
            if duplicated_direct_raise {
                let evalue = args[1].clone();
                let etype = Hlvalue::Variable(Variable::new());
                translated_ops.push(FlowspaceOp::new("type", vec![evalue], etype.clone()));
                args[0] = etype;
            }
            let exitcase = exitcase_to_hlvalue(legacy_link.exitcase.as_ref(), call_registry)?;
            let mut link = FlowspaceLink::new(args, Some(target), exitcase);
            // RPython `Link.__init__` (`flowspace/model.rs:Link::new`) leaves
            // `llexitcase` unset; `RPythonTyper._convert_link`
            // (`rpython/rtyper/rtyper.py`) populates it during
            // specialization via `r_case.convert_const(link.exitcase)`.
            // Pre-seeding `llexitcase` here would diverge from upstream
            // `pre-rtyper` graph shape; the rtyper overwrites it anyway.
            link.extravars(last_exception, last_exc_value);
            link.prevblock = Some(Rc::downgrade(&block_ref));
            translated_exits.push(link.into_ref());
        }

        // Translate exitswitch.
        let translated_exitswitch = match &legacy_block.exitswitch {
            None => None,
            Some(ExitSwitch::Value(var)) => {
                Some(value_map.get(var).cloned().ok_or_else(|| {
                    // Inline counterpart of `lookup_operand` for the
                    // block.exitswitch path (no enclosing
                    // SpaceOperation). Required substring
                    // `"undefined operand"` is preserved
                    // verbatim for `is_known_unported`
                    // (`cutover.rs`).
                    TyperError::message(format!(
                        "translate_op: undefined operand {var:?} as block.exitswitch — \
                         adapter invariant broken (every referenced operand must be \
                         defined as a block inputarg or op result)",
                    ))
                })?)
            }
            Some(ExitSwitch::LastException) => Some(Hlvalue::Constant(c_last_exception())),
            // `ExitSwitch::Fused` is produced only by the JIT codewriter's
            // `jtransform::optimize_goto_if_not` (post-rtyper); the
            // pre-rtyper legacy graph this adapter translates never carries
            // a fused exitswitch.
            Some(ExitSwitch::Fused { .. }) => {
                unreachable!("fused exitswitch cannot reach the pre-rtyper flowspace adapter")
            }
        };

        // Commit to the flowspace::Block. Borrow scope is tight to
        // avoid alias-clash with link.prevblock's Weak above.
        {
            let mut block = block_ref.borrow_mut();
            block.operations = translated_ops;
            block.exits = translated_exits;
            block.exitswitch = translated_exitswitch;
        }
    }

    flowspace_model::checkgraph(&graph_ref.borrow());

    Ok(FlowspaceAdapterOutput {
        graph: graph_ref,
        value_to_var,
        value_to_var_candidates,
        constant_concretetypes,
        constant_hlvalues,
        #[cfg(test)]
        block_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::model::{KnownType, SomeObjectTrait, SomeValue};
    use crate::model::{
        Block, BlockId, FunctionGraph as LegacyGraph, OpKind, SpaceOperation, ValueType,
    };
    use crate::translator::rtyper::call_registry::CallRegistry;
    use crate::translator::rtyper::legacy_annotator::setbinding;

    /// Mint `n` fresh values on `graph`, returned indexed `0..n` so
    /// fixtures can refer to operands / results / inputargs positionally.
    fn mint_vars(graph: &mut LegacyGraph, n: usize) -> Vec<crate::flowspace::model::Variable> {
        (0..n).map(|_| graph.alloc_value_var()).collect()
    }

    /// Project positional indices to held Variables for
    /// `Block { inputargs, .. }` literals.
    fn block_inputargs(
        vars: &[crate::flowspace::model::Variable],
        vids: &[usize],
    ) -> Vec<crate::flowspace::model::Variable> {
        vids.iter().map(|&i| vars[i].clone()).collect()
    }

    /// Helper: empty `CallRegistry` for tests that don't exercise
    /// the Call resolution path.  The registry's
    /// bookkeeper is freshly minted because translate_op tests don't
    /// share state with an enclosing annotator.
    fn empty_call_registry() -> CallRegistry {
        CallRegistry::new(Rc::new(Bookkeeper::new()))
    }

    #[test]
    fn valuetype_int_lifts_to_someinteger_default() {
        let s = valuetype_to_someshell(&ValueType::Int).expect("Int must project");
        match s {
            SomeValue::Integer(i) => {
                assert_eq!(i.knowntype(), KnownType::Int);
                assert!(!i.nonneg, "default SomeInteger.nonneg must be false");
                assert!(!i.unsigned, "default SomeInteger.unsigned must be false");
            }
            other => panic!("ValueType::Int must lift to SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn wide_valuetypes_and_constants_keep_exact_rpython_lowlevel_types() {
        for (value_type, expected_known) in [
            (ValueType::Int128, KnownType::LongLongLong),
            (ValueType::UInt128, KnownType::ULongLongLong),
        ] {
            let s = valuetype_to_someshell(&value_type).expect("wide ValueType must project");
            let SomeValue::Integer(integer) = s else {
                panic!("wide ValueType must lift to SomeInteger")
            };
            assert_eq!(integer.knowntype(), expected_known);
        }

        for (value, expected_lltype) in [
            (
                ConstValue::Int128(i128::MIN),
                LowLevelType::SignedLongLongLong,
            ),
            (
                ConstValue::UInt128(u128::MAX),
                LowLevelType::UnsignedLongLongLong,
            ),
        ] {
            let constant = constant_from_constvalue(value.clone());
            assert_eq!(constant.value, value);
            assert_eq!(constant.concretetype, Some(expected_lltype));
        }
    }

    #[test]
    fn valuetype_float_lifts_to_somefloat_default() {
        let s = valuetype_to_someshell(&ValueType::Float).expect("Float must project");
        match s {
            SomeValue::Float(f) => {
                assert_eq!(f.knowntype(), KnownType::Float);
                assert!(f.immutable(), "SomeFloat is immutable per model.py:164-183");
            }
            other => panic!("ValueType::Float must lift to SomeFloat, got {other:?}"),
        }
    }

    #[test]
    fn valuetype_ref_lifts_to_someinstance_classdef_none() {
        let s = valuetype_to_someshell(&ValueType::Ref(None)).expect("Ref must project");
        match s {
            SomeValue::Instance(inst) => {
                assert!(
                    inst.classdef.is_none(),
                    "Ref placeholder must carry classdef=None (model.py:438 default)"
                );
                assert!(
                    !inst.can_be_none,
                    "SomeInstance.__init__ default `can_be_None=False` (model.py:438)"
                );
                assert!(
                    inst.flags.is_empty(),
                    "SomeInstance.__init__ default `flags={{}}` (model.py:438)"
                );
            }
            other => panic!(
                "ValueType::Ref must lift to SomeValue::Instance(classdef=None), got {other:?}"
            ),
        }
    }

    #[test]
    fn valuetype_void_lifts_to_impossible_lattice_bottom() {
        let s = valuetype_to_someshell(&ValueType::Void).expect("Void must project");
        assert!(
            matches!(s, SomeValue::Impossible),
            "ValueType::Void must lift to SomeValue::Impossible, got {s:?}"
        );
    }

    #[test]
    fn valuetype_state_lifts_to_someinstance_classdef_none() {
        // TODO(state-shell): pyre-only `State` (JIT state pointer)
        // — temporary fallback shell.
        let s = valuetype_to_someshell(&ValueType::State).expect("State must project");
        match s {
            SomeValue::Instance(inst) => {
                assert!(
                    inst.classdef.is_none(),
                    "State placeholder must carry classdef=None"
                );
            }
            other => panic!(
                "ValueType::State must lift to SomeValue::Instance(classdef=None), got {other:?}"
            ),
        }
    }

    #[test]
    fn valuetype_unknown_returns_none_for_failloud_at_bindingrepr() {
        // `Unknown` is an annotation gap with no
        // annotation-stage shell.  Returning `None` leaves
        // `Variable.annotation` empty so `bindingrepr` panics with
        // `KeyError: no binding for arg`
        // (`annotator/annrpython.rs`), surfacing the producer-side
        // gap instead of silently bridging it to `GcRef` via a
        // fabricated `SomeInstance(None)` shell — that bridging
        // conflated the annotation-stage lattice node with the
        // **legacy** `resolve_types(Unknown) -> ConcreteType::Unknown
        // -> GcRef` resolver-stage backfill.
        assert!(
            valuetype_to_someshell(&ValueType::Unknown).is_none(),
            "ValueType::Unknown must return None — annotation gap, no \
             annotation-stage shell"
        );
    }

    #[test]
    fn seed_variable_attaches_lifted_annotation_observable_via_clone() {
        let mut graph = LegacyGraph::new("seed_test");
        let vars = mint_vars(&mut graph, 8); // vars[0..8]
        let legacy_var = vars[7].clone();
        setbinding(&legacy_var, ValueType::Int);
        let var = seed_variable(&legacy_var);

        // Reference semantics: the annotation Rc-shares across clones
        // (flowspace/model.rs), so a clone observes the same
        // shell instance.
        let clone = var.clone();
        let clone_ann = clone.annotation.borrow();
        let shell = clone_ann
            .as_ref()
            .expect("seed_variable must populate annotation slot");
        assert!(
            matches!(shell.as_ref(), SomeValue::Integer(_)),
            "annotation must round-trip the lifted SomeInteger shell"
        );
    }

    #[test]
    fn seed_variable_unannotated_input_leaves_annotation_empty_for_failloud() {
        // When the legacy Variable carries no published
        // SomeValue shell, the seed MUST NOT fabricate a
        // SomeInstance(classdef=None) — that would silently bridge an
        // annotation gap to GcRef via the resolver-stage backfill at
        // the wrong layer. Instead, leave Variable.annotation empty so
        // `bindingrepr` panics with `KeyError: no binding for arg`
        // (annotator/annrpython.rs), surfacing the producer-side
        // gap as a fail-loud signal.
        let legacy_var = Variable::new();
        let var = seed_variable(&legacy_var);
        let ann = var.annotation.borrow();
        assert!(
            ann.is_none(),
            "Unannotated input must leave annotation empty (Cat 2.4 fail-loud), \
             got {:?}",
            ann.as_ref()
        );
    }

    fn legacy_graph_with_inputarg_and_result(
        input: usize,
        result: usize,
    ) -> (LegacyGraph, Vec<crate::flowspace::model::Variable>) {
        let mut graph = LegacyGraph::new("test");
        let vars = mint_vars(&mut graph, input.max(result) + 1);
        let inputargs = block_inputargs(&vars, &[input]);
        let result_var = vars[result].clone();
        let mut block = Block {
            id: BlockId(0),
            inputargs,
            operations: vec![SpaceOperation {
                result: Some(result_var),
                kind: OpKind::ConstInt(0),
            }],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];
        (graph, vars)
    }

    #[test]
    fn build_value_to_variable_map_seeds_inputargs_and_op_results() {
        let (graph, vars) = legacy_graph_with_inputarg_and_result(1, 2);

        setbinding(&vars[1], ValueType::Int);
        setbinding(&vars[2], ValueType::Ref(None));
        let map = build_value_to_variable_map(&graph);

        assert_eq!(
            map.len(),
            2,
            "map must seed both the inputarg and the op result"
        );
        assert!(
            matches!(
                map[&vars[1]]
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|s| s.as_ref().clone()),
                Some(SomeValue::Integer(_))
            ),
            "inputarg slot 1 (Int) must be seeded with SomeInteger"
        );
        assert!(
            matches!(
                map[&vars[2]]
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|s| s.as_ref().clone()),
                Some(SomeValue::Instance(_))
            ),
            "op-result slot 2 (Ref) must be seeded with SomeInstance(classdef=None)"
        );
    }

    #[test]
    fn build_value_to_variable_map_dedupes_by_value_id() {
        // Two ops both reading the same inputarg (legacy graphs are SSA
        // — every slot has one definition, but multiple readers).
        // Must produce one Variable identity per slot.
        let mut graph = LegacyGraph::new("dedup_test");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let inputargs = block_inputargs(&vars, &[1]);
        let result2_var = vars[2].clone();
        let result3_var = vars[3].clone();
        let mut block = Block {
            id: BlockId(0),
            inputargs,
            operations: vec![
                SpaceOperation {
                    result: Some(result2_var),
                    kind: OpKind::ConstInt(7),
                },
                SpaceOperation {
                    result: Some(result3_var),
                    kind: OpKind::ConstInt(11),
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        setbinding(&vars[1], ValueType::Int);
        setbinding(&vars[2], ValueType::Int);
        setbinding(&vars[3], ValueType::Int);
        let map = build_value_to_variable_map(&graph);

        assert_eq!(map.len(), 3, "three distinct slots → three Variables");
        // The identity invariant: the inputarg's Variable is one fresh
        // identity, the two op results are two more fresh identities, and
        // they don't collide.
        assert_ne!(map[&vars[1]], map[&vars[2]]);
        assert_ne!(map[&vars[1]], map[&vars[3]]);
        assert_ne!(map[&vars[2]], map[&vars[3]]);
    }

    #[test]
    fn build_value_to_variable_map_aliases_input_rebind_to_inputarg() {
        // Pyre's surface front emits a leading `Input{name}` op whose
        // result IS a block.inputarg, plus follow-up `Input{same name}`
        // ops with FRESH result slots for body-side rebinds. The
        // adapter must alias the rebind result to the canonical
        // inputarg Variable so `setup_block_entry`'s
        // `concretetype` write reaches both — otherwise the body's
        // BinOp lookup hits a fresh Variable with no concretetype and
        // trips genop's "wrong level!" assertion.
        let mut graph = LegacyGraph::new("rebind_alias");
        let vars = mint_vars(&mut graph, 3); // vars[0..3]
        let mut block = Block {
            id: BlockId(0),
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![
                // Leading definition: result IS the inputarg.
                SpaceOperation {
                    result: Some(vars[1].clone()),
                    kind: OpKind::Input {
                        name: "x".to_string(),
                        ty: ValueType::Int,
                        class_root: None,
                    },
                },
                // Rebind: result is fresh; same name → alias to vars[1]'s Variable.
                SpaceOperation {
                    result: Some(vars[2].clone()),
                    kind: OpKind::Input {
                        name: "x".to_string(),
                        ty: ValueType::Int,
                        class_root: None,
                    },
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        setbinding(&vars[1], ValueType::Int);
        setbinding(&vars[2], ValueType::Int);
        let map = build_value_to_variable_map(&graph);
        assert_eq!(
            map[&vars[1]], map[&vars[2]],
            "Input rebind result must alias to inputarg Variable identity"
        );
    }

    // ───── dispatcher + skip arms + fail-loud ─────

    #[test]
    fn build_value_to_hlvalue_map_inlines_const_defines() {
        let mut graph = LegacyGraph::new("const_inline");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let inputargs = block_inputargs(&vars, &[1]);
        let result2_var = vars[2].clone();
        let result3_var = vars[3].clone();
        let mut block = Block {
            id: BlockId(0),
            inputargs,
            operations: vec![
                SpaceOperation {
                    result: Some(result2_var),
                    kind: OpKind::ConstInt(42),
                },
                SpaceOperation {
                    result: Some(result3_var),
                    kind: OpKind::ConstFloat(0xC000_0000_0000_0000), // f64::from_bits → -2.0
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        setbinding(&vars[1], ValueType::Int);
        setbinding(&vars[2], ValueType::Int);
        setbinding(&vars[3], ValueType::Float);
        let var_map = build_value_to_variable_map(&graph);
        let hl_map = build_value_to_hlvalue_map(&graph, &var_map);

        // Inputarg keeps its Variable identity.
        assert!(matches!(hl_map[&vars[1]], Hlvalue::Variable(_)));

        // ConstInt define is inlined as Hlvalue::Constant(Int).
        match &hl_map[&vars[2]] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(n) => assert_eq!(*n, 42),
                other => panic!("result 2 must be ConstValue::Int, got {other:?}"),
            },
            other => panic!("result 2 must be inlined as Hlvalue::Constant, got {other:?}"),
        }

        // ConstFloat define is inlined as Hlvalue::Constant(Float).
        match &hl_map[&vars[3]] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Float(bits) => assert_eq!(*bits, 0xC000_0000_0000_0000),
                other => panic!("result 3 must be ConstValue::Float, got {other:?}"),
            },
            other => panic!("result 3 must be inlined as Hlvalue::Constant, got {other:?}"),
        }
    }

    #[test]
    fn translate_op_skips_input_define() {
        let value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::Input {
                name: "x".to_string(),
                ty: ValueType::Int,
                class_root: None,
            },
        };
        let result = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Input must translate to skip");
        assert!(
            result.is_empty(),
            "Input define has no SpaceOperation analogue (handled by block \
             topology via block.inputargs); translate_op must yield empty Vec"
        );
    }

    #[test]
    fn derive_subject_inputcells_projects_each_typed_input_op() {
        let mut graph = LegacyGraph::new("subject");
        let entry = graph.startblock;
        let x_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "x".to_string(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let y_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "y".to_string(),
                    ty: ValueType::Float,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let z_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "z".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, x_var);
        graph.push_inputarg_var(entry, y_var);
        graph.push_inputarg_var(entry, z_var);

        let cells = derive_subject_inputcells(&graph, None)
            .expect("typed Input ops must project to definite SomeValue cells");
        assert_eq!(cells.len(), 3);
        assert!(
            matches!(cells[0], SomeValue::Integer(_)),
            "x: Int -> SomeInteger, got {:?}",
            cells[0],
        );
        assert!(
            matches!(cells[1], SomeValue::Float(_)),
            "y: Float -> SomeFloat, got {:?}",
            cells[1],
        );
        assert!(
            matches!(cells[2], SomeValue::Instance(_)),
            "z: Ref -> SomeInstance(classdef=None), got {:?}",
            cells[2],
        );
    }

    #[test]
    fn derive_subject_inputcells_seeds_fn_ptr_int_input_as_someptr() {
        let mut graph = LegacyGraph::new("takes_fn");
        let entry = graph.startblock;
        let f_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "func".to_string(),
                    ty: ValueType::Int,
                    class_root: Some("fn".to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, f_var);

        let cells = derive_subject_inputcells(&graph, None)
            .expect("fn-pointer Input must project to a definite SomeValue cell");
        assert_eq!(cells.len(), 1);
        assert!(
            matches!(cells[0], SomeValue::Ptr(_)),
            "Int + class_root fn must seed SomePtr(FuncType), got {:?}",
            cells[0],
        );
    }

    #[test]
    fn derive_subject_inputcells_seeds_option_fn_int_input_as_someptr() {
        let mut graph = LegacyGraph::new("takes_option_fn");
        let entry = graph.startblock;
        let f_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "func".to_string(),
                    ty: ValueType::Int,
                    class_root: Some("Option<fn>".to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, f_var);

        let cells = derive_subject_inputcells(&graph, None)
            .expect("Option<fn> Input must project to a definite SomeValue cell");
        assert_eq!(cells.len(), 1);
        assert!(
            matches!(cells[0], SomeValue::Ptr(_)),
            "Int + class_root Option<fn> must seed SomePtr(FuncType), got {:?}",
            cells[0],
        );
    }

    #[test]
    fn derive_subject_inputcells_prefers_source_root_over_legacy_shell() {
        let mut graph = LegacyGraph::new("set_locals_w");
        let entry = graph.startblock;
        let value = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "value".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: Some("PyObject".to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, value.clone());

        // Transitional legacy annotation loses the pointee class.  The
        // source Input still says PyObjectRef, corresponding to W_Root in
        // PyPy, and must be the authoritative initial cell.
        setbinding(&value, ValueType::Ref(None));
        let bk = Rc::new(Bookkeeper::new());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "PyObject".to_string(),
            vec![("type_ptr".to_string(), "usize".to_string())],
        );
        bk.set_struct_fields(Rc::new(fields));

        let cells = derive_subject_inputcells(&graph, Some(&bk))
            .expect("source-typed PyObject input must seed a concrete class");
        let SomeValue::Instance(value) = &cells[0] else {
            panic!("PyObjectRef must seed SomeInstance, got {:?}", cells[0])
        };
        let classdef = value
            .classdef
            .as_ref()
            .expect("source class_root must replace the classdef-less legacy shell");
        assert_eq!(classdef.borrow().name, "PyObject");
    }

    #[test]
    fn derive_subject_inputcells_seeds_a_tuple_input_as_sometuple() {
        let mut graph = LegacyGraph::new("closure::call_once");
        let entry = graph.startblock;
        let args = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "args".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: Some("Tuple<i64,f64>".to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, args);
        let bk = Rc::new(Bookkeeper::new());
        // The shape is also a registered positional class; the tuple arm
        // must win over its nominal classdef.
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "Tuple<i64,f64>".to_string(),
            vec![
                ("__pos_0".to_string(), "i64".to_string()),
                ("__pos_1".to_string(), "f64".to_string()),
            ],
        );
        bk.set_struct_fields(Rc::new(fields));

        let cells =
            derive_subject_inputcells(&graph, Some(&bk)).expect("a tuple-shaped input must seed");
        let SomeValue::Tuple(tuple) = &cells[0] else {
            panic!("a Tuple<..> input must seed SomeTuple, got {:?}", cells[0])
        };
        assert_eq!(tuple.items.len(), 2);
        assert!(matches!(tuple.items[0], SomeValue::Integer(_)));
        assert!(matches!(tuple.items[1], SomeValue::Float(_)));
    }

    #[test]
    fn derive_subject_inputcells_seeds_ref_of_raw_struct_as_someptr() {
        use crate::translator::rtyper::lltypesystem::lltype::{GcKind, PtrTarget};

        let owner = "rawref::Scope";
        let owner_id = majit_ir::descr::StructId::from_canonical(owner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (owner.to_string(), Some(owner_id)),
            ]));

        let mut graph = LegacyGraph::new("scope_get");
        let entry = graph.startblock;
        let receiver = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "scope".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: Some(owner.to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, receiver);

        let bk = Rc::new(Bookkeeper::new());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            owner.to_string(),
            vec![("save_point".to_string(), "usize".to_string())],
        );
        fields.raw_word_owners.insert(owner.to_string());
        bk.set_struct_fields(Rc::new(fields));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            owner_id,
            crate::call::StructLayout {
                size: 8,
                align: 8,
                gckind: GcKind::Raw,
                fields: vec![crate::call::StructFieldLayout {
                    name: "save_point".into(),
                    offset: 0,
                    size: 8,
                    flag: majit_ir::descr::ArrayFlag::Unsigned,
                    field_type: majit_ir::value::Type::Int,
                    rank: None,
                }],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        bk.set_struct_layouts(cc.struct_layouts_handle());

        let cells =
            derive_subject_inputcells(&graph, Some(&bk)).expect("&RawStruct input must seed");
        let SomeValue::Ptr(ptr) = &cells[0] else {
            panic!("&RawStruct must seed SomePtr, got {:?}", cells[0]);
        };
        let PtrTarget::Struct(st) = &ptr.ll_ptrtype.TO else {
            panic!("raw Ref input must point at a Struct");
        };
        assert_eq!(st._name, owner);
        assert_eq!(st._gckind, GcKind::Raw);
    }

    #[test]
    fn derive_subject_inputcells_seeds_a_slice_of_gc_refs_as_somelist() {
        let mut graph = LegacyGraph::new("ctypefunc::call");
        let entry = graph.startblock;
        let args = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "args_w".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: Some("[PyObjectRef]".to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, args);
        let bk = Rc::new(Bookkeeper::new());
        let cells =
            derive_subject_inputcells(&graph, Some(&bk)).expect("a [PyObjectRef] input must seed");
        assert!(
            matches!(cells[0], SomeValue::List(_)),
            "args_w: &[PyObjectRef] must seed SomeList, got {:?}",
            cells[0]
        );
    }

    #[test]
    fn derive_subject_inputcells_seeds_charon_slice_class_root_as_somelist() {
        let mut graph = LegacyGraph::new("wrapper_args");
        let entry = graph.startblock;
        let args = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "args".to_string(),
                    ty: ValueType::Int,
                    class_root: Some("Slice<PyObjectRef>".to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, args);
        let bk = Rc::new(Bookkeeper::new());
        let cells = derive_subject_inputcells(&graph, Some(&bk))
            .expect("a Slice<PyObjectRef> input must seed");
        assert!(
            matches!(cells[0], SomeValue::List(_)),
            "Charon Slice<T> must seed SomeList even on an Int-colored input, got {:?}",
            cells[0]
        );
    }

    #[test]
    fn derive_subject_inputcells_records_no_fallthrough_when_verbose_off() {
        assert!(
            !crate::translator::rtyper::cutover::rtyper_verbose_enabled(),
            "this pin requires MAJIT_RTYPER_VERBOSE to be unset or not \"1\""
        );
        let mut graph = LegacyGraph::new("verbose_off_fallthrough");
        let entry = graph.startblock;
        let value = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "value".to_string(),
                    ty: ValueType::Ref(Some("UnregisteredRoot".to_string())),
                    class_root: Some("UnregisteredRoot".to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, value);
        let bk = Rc::new(Bookkeeper::new());
        bk.set_struct_fields(Rc::new(crate::front::StructFieldRegistry::default()));

        let cells = derive_subject_inputcells(&graph, Some(&bk))
            .expect("unregistered Ref input keeps the classdef-less shell");
        assert!(
            matches!(&cells[0], SomeValue::Instance(inst) if inst.classdef.is_none()),
            "unregistered root must keep the classdef-less shell, got {:?}",
            cells[0],
        );
    }

    #[test]
    fn derive_subject_inputcells_fails_loud_on_inputarg_without_input_op() {
        let mut graph = LegacyGraph::new("subject");
        let entry = graph.startblock;
        let orphan = graph.alloc_value_var();
        graph.push_inputarg_var(entry, orphan);
        let err = derive_subject_inputcells(&graph, None)
            .expect_err("inputarg without matching Input op must surface as TyperError");
        let msg = format!("{err}");
        assert!(
            msg.contains("no matching `OpKind::Input"),
            "error must name the missing Input op invariant, got: {msg}"
        );
    }

    #[test]
    fn derive_subject_inputcells_fails_loud_on_unknown_valuetype() {
        let mut graph = LegacyGraph::new("subject");
        let entry = graph.startblock;
        let var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "u".to_string(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, var);
        let err = derive_subject_inputcells(&graph, None)
            .expect_err("ValueType::Unknown has no SomeValue projection");
        let msg = format!("{err}");
        assert!(
            msg.contains("Unknown") && msg.contains("None"),
            "error must mention Unknown + missing projection, got: {msg}"
        );
    }

    #[test]
    fn strlen_on_w_str_get_wtf8_identity_alias_receiver_is_instance_not_string() {
        // This fixture is the OLD identity-alias graph: `__strlen` on a
        // PyObject Input.  Real `w_str_get_wtf8` sites no longer produce
        // it — dest is the `W_UnicodeObject.value` read (see
        // `strlen_arg_at_real_w_str_get_wtf8_sites_is_the_utf8_field`).
        // Pin the annotation / repr at this argument: it is still
        // SomeInstance / InstanceRepr, which has no `rtype_len` and whose
        // annotator `len_SomeInstance` is `getattr(__len__)`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::pairtype::ReprClassId;
        use crate::translator::rtyper::rmodel::Repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};

        let mut graph = LegacyGraph::new("strlen_wtf8_identity_alias");
        let vars = mint_vars(&mut graph, 4);
        let obj = vars[1].clone();
        let len_result = vars[2].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![
                SpaceOperation {
                    result: Some(obj.clone()),
                    kind: OpKind::Input {
                        name: "obj".to_string(),
                        ty: ValueType::Ref(None),
                        class_root: Some("PyObject".to_string()),
                    },
                },
                SpaceOperation {
                    result: Some(len_result.clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath {
                            segments: vec!["__strlen".to_string()],
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![obj.clone()]),
                        result_ty: ValueType::Int,
                    },
                },
            ],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(len_result.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let bk = Rc::new(Bookkeeper::new());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "PyObject".to_string(),
            vec![("type_ptr".to_string(), "usize".to_string())],
        );
        fields.fields.insert(
            "pyobject::PyObject".to_string(),
            vec![("type_ptr".to_string(), "usize".to_string())],
        );
        bk.set_struct_fields(Rc::new(fields));

        let cells = derive_subject_inputcells(&graph, Some(&bk))
            .expect("PyObject Input must seed a SomeValue cell");
        let SomeValue::Instance(inst) = &cells[0] else {
            panic!(
                "w_str_get_wtf8 identity-alias receiver must be SomeInstance, got {:?}",
                cells[0]
            );
        };
        assert!(
            !matches!(cells[0], SomeValue::String(_)),
            "identity-alias of w_str_get_wtf8(obj) must not seed SomeString"
        );
        let classdef = inst
            .classdef
            .as_ref()
            .expect("PyObject class_root must populate classdef");
        assert_eq!(classdef.borrow().name, "PyObject");

        let projected_root_instance = bk.project_struct_field_type("PyObjectRef");
        assert!(
            matches!(projected_root_instance, SomeValue::Instance(_)),
            "PyObjectRef projects to SomeInstance(pyobject::PyObject), got {projected_root_instance:?}"
        );
        assert!(
            !matches!(projected_root_instance, SomeValue::String(_)),
            "PyObjectRef must not project to SomeString"
        );

        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        value_map.insert(obj.clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(len_result.clone(), Hlvalue::Variable(Variable::new()));
        let strlen_op = SpaceOperation {
            result: Some(len_result.clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__strlen".to_string()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![obj.clone()]),
                result_ty: ValueType::Int,
            },
        };
        // `__strlen` now rewrites to `len` (the real-site dest is
        // SomeString).  This fixture is still an Instance receiver, so
        // the rewrite is the same spelling — honesty is the later
        // `rtype_len` dispatch below, which still fails.
        let translated_strlen = translate_op(&strlen_op, &value_map, &empty_call_registry())
            .expect("__strlen is a len spelling");
        assert_eq!(translated_strlen.len(), 1);
        assert_eq!(translated_strlen[0].opname, "len");

        // Contrast: the existing `__len` spelling is the rtyper `len` op.
        let len_op = SpaceOperation {
            result: Some(len_result.clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__len".to_string()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![obj.clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&len_op, &value_map, &empty_call_registry())
            .expect("__len is the existing len spelling");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "len");

        let ann = RPythonAnnotator::new(None, None, Some(bk.clone()), false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("exceptiondata for getinstancerepr");
        let repr = rtyper
            .getrepr(&cells[0])
            .expect("SomeInstance(PyObject) must make InstanceRepr");
        assert_eq!(Repr::class_name(repr.as_ref()), "InstanceRepr");
        assert_eq!(
            Repr::repr_class_id(repr.as_ref()),
            ReprClassId::InstanceRepr
        );
        assert_ne!(Repr::repr_class_id(repr.as_ref()), ReprClassId::StringRepr);

        let spaceop = crate::flowspace::model::SpaceOperation::new(
            crate::flowspace::operation::OpKind::Len.opname(),
            vec![],
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        let err = Repr::rtype_len(repr.as_ref(), &hop).expect_err("InstanceRepr has no rtype_len");
        assert!(err.is_missing_rtype_operation());
        let msg = err.to_string();
        assert!(
            msg.contains("unimplemented operation: 'len'"),
            "InstanceRepr.rtype_len must be the Repr default, got: {msg}"
        );
    }

    #[test]
    fn w_str_get_wtf8_cast_projects_dest_as_somestring() {
        // The `Wtf8` string-root cast
        // `cast_instance_call_result("Wtf8", obj, ValueType::Str)`: dest
        // is a new Variable whose annotation is SomeString; obj stays
        // SomeInstance(PyObject).  StringRepr.rtype_len → ll_strlen is
        // then the honest length of that dest.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::SomeInstance;
        use crate::flowspace::model::ConstValue;
        use crate::translator::rtyper::pairtype::ReprClassId;
        use crate::translator::rtyper::rmodel::Repr;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let mut graph = LegacyGraph::new("w_str_get_wtf8_str_projection");
        let vars = mint_vars(&mut graph, 4);
        let obj = vars[1].clone();
        let dest = vars[2].clone();
        let cast = crate::model::cast_instance_call_result("Wtf8", obj.clone(), ValueType::Str);
        assert_eq!(
            crate::model::cast_instance_root(&cast),
            Some("Wtf8"),
            "the producer root is the Wtf8 string root"
        );
        let OpKind::Call { result_ty, .. } = &cast else {
            panic!("cast_instance_call_result must be a Call");
        };
        assert_eq!(
            result_ty,
            &ValueType::Str,
            "dest must be painted ValueType::Str, not the receiver Ref"
        );

        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![
                SpaceOperation {
                    result: Some(obj.clone()),
                    kind: OpKind::Input {
                        name: "obj".to_string(),
                        ty: ValueType::Ref(None),
                        class_root: Some("PyObject".to_string()),
                    },
                },
                SpaceOperation {
                    result: Some(dest.clone()),
                    kind: cast,
                },
            ],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(dest.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let bk = Rc::new(Bookkeeper::new());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "PyObject".to_string(),
            vec![("type_ptr".to_string(), "usize".to_string())],
        );
        bk.set_struct_fields(Rc::new(fields));

        let cells = derive_subject_inputcells(&graph, Some(&bk))
            .expect("PyObject Input must seed a SomeValue cell");
        assert!(
            matches!(&cells[0], SomeValue::Instance(_)),
            "the receiver stays SomeInstance, got {:?}",
            cells[0]
        );

        let s_obj = SomeValue::Instance(SomeInstance::new(
            Some(ClassDef::new_standalone("pyobject::PyObject", None)),
            false,
            Default::default(),
        ));
        let s_root = bk
            .immutablevalue(&ConstValue::byte_str("Wtf8"))
            .expect("Wtf8 root constant");
        let s_dest = crate::annotator::builtin::call_builtin(
            &bk,
            crate::runtime_names::shims::CAST_INSTANCE,
            &[Some(s_obj), Some(s_root)],
            &std::collections::HashMap::new(),
        )
        .expect("string-root cast must accept a PyObject instance");
        assert!(
            matches!(s_dest, SomeValue::String(_)),
            "w_str_get_wtf8 dest must annotate SomeString, got {s_dest:?}"
        );

        let shell = valuetype_to_someshell(&ValueType::Str).expect("Str projects");
        assert!(
            matches!(shell, SomeValue::String(_)),
            "ValueType::Str seeds SomeString, got {shell:?}"
        );

        let ann = RPythonAnnotator::new(None, None, Some(bk.clone()), false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("exceptiondata for getrepr");
        let repr = rtyper
            .getrepr(&s_dest)
            .expect("SomeString must make StringRepr");
        assert_eq!(Repr::class_name(repr.as_ref()), "StringRepr");
        assert_eq!(Repr::repr_class_id(repr.as_ref()), ReprClassId::StringRepr);
        assert_ne!(
            Repr::repr_class_id(repr.as_ref()),
            ReprClassId::InstanceRepr
        );
    }

    fn wtf8_strlen_fixture(
        rows: Vec<(String, String)>,
    ) -> (LegacyGraph, CallRegistry, crate::flowspace::model::Variable) {
        let registry = empty_call_registry();
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert("W_UnicodeObject".to_string(), rows);
        registry.bookkeeper().set_struct_fields(Rc::new(fields));

        let mut graph = LegacyGraph::new("strlen_wtf8_dest");
        let vars = mint_vars(&mut graph, 4);
        let obj = vars[1].clone();
        let dest = vars[2].clone();
        let len_result = vars[3].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![
                SpaceOperation {
                    result: Some(obj.clone()),
                    kind: OpKind::Input {
                        name: "obj".to_string(),
                        ty: ValueType::Ref(Some("W_UnicodeObject".to_string())),
                        class_root: Some("W_UnicodeObject".to_string()),
                    },
                },
                SpaceOperation {
                    result: Some(dest.clone()),
                    kind: crate::model::cast_instance_call_result("Wtf8", obj, ValueType::Str),
                },
                SpaceOperation {
                    result: Some(len_result.clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath {
                            segments: vec!["__strlen".to_string()],
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![dest.clone()]),
                        result_ty: ValueType::Int,
                    },
                },
            ],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(len_result)],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[3]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];
        (graph, registry, dest)
    }

    fn flow_ops(output: &FlowspaceAdapterOutput) -> Vec<crate::flowspace::model::SpaceOperation> {
        let graph = output.graph.borrow();
        graph
            .iterblocks()
            .into_iter()
            .flat_map(|block| block.borrow().operations.clone())
            .collect()
    }

    fn hl_var_id(value: &Hlvalue) -> Option<u64> {
        match value {
            Hlvalue::Variable(var) => Some(var.id()),
            _ => None,
        }
    }

    /// `w_str_get_wtf8` paints its dest `ValueType::Str`, but the pointer
    /// is still the `W_UnicodeObject`. `Wtf8::len` is the byte length of
    /// `_utf8` (`W_UnicodeObject.value`, an rstr `STR`), so `__strlen`
    /// must `getattr` that string field and `len` the field — `ll_strlen`
    /// on the wrapper reads `PyObject.w_class`.
    #[test]
    fn strlen_on_wtf8_cast_reads_the_string_field_not_the_wrapper() {
        use crate::flowspace::model::ConstValue;

        let (graph, registry, dest) = wtf8_strlen_fixture(vec![
            ("ob_header".to_string(), "PyObject".to_string()),
            ("value".to_string(), "*mut Utf8Str".to_string()),
            ("byte_len".to_string(), "usize".to_string()),
            ("len".to_string(), "usize".to_string()),
        ]);
        let output =
            function_graph_to_flowspace(&graph, &registry).expect("proven string field must lower");
        let ops = flow_ops(&output);
        let getattr = ops
            .iter()
            .find(|op| op.opname == "getattr")
            .expect("length must getattr the string field, not len the wrapper");
        let Hlvalue::Constant(field) = &getattr.args[1] else {
            panic!("getattr field must be a constant, got {:?}", getattr.args);
        };
        assert!(
            matches!(&field.value, ConstValue::ByteStr(bytes) if bytes == b"value"),
            "the unique string-typed field is `value` (_utf8), not byte_len/len, got {:?}",
            field.value
        );
        let field_id = hl_var_id(&getattr.result).expect("getattr result");
        let len_ops: Vec<_> = ops.iter().filter(|op| op.opname == "len").collect();
        assert_eq!(len_ops.len(), 1, "one len, of the string field: {ops:?}");
        assert_eq!(hl_var_id(&len_ops[0].args[0]), Some(field_id));
        let wrapper = output
            .value_to_var
            .get(&dest)
            .expect("cast dest representative");
        assert_ne!(
            hl_var_id(&len_ops[0].args[0]),
            Some(wrapper.id()),
            "len must not read the W_UnicodeObject wrapper"
        );
        assert!(
            !ops.iter().any(|op| {
                matches!(op.opname.as_str(), "int_add" | "direct_ptradd" | "adr_add")
            }),
            "the length is a field then ll_strlen, not an address: {ops:?}"
        );
    }

    /// The cast result reaches `__strlen` as the successor block's inputarg.
    /// Resolving only operation results would `len` the wrapper.
    #[test]
    fn strlen_on_wtf8_cast_phi_reads_the_string_field() {
        use crate::flowspace::model::ConstValue;

        let registry = empty_call_registry();
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "W_UnicodeObject".to_string(),
            vec![
                ("ob_header".to_string(), "PyObject".to_string()),
                ("value".to_string(), "*mut Utf8Str".to_string()),
                ("byte_len".to_string(), "usize".to_string()),
                ("len".to_string(), "usize".to_string()),
            ],
        );
        registry.bookkeeper().set_struct_fields(Rc::new(fields));

        let mut graph = LegacyGraph::new("strlen_wtf8_phi");
        let vars = mint_vars(&mut graph, 5);
        let obj = vars[1].clone();
        let dest = vars[2].clone();
        let view = vars[3].clone();
        let len_result = vars[4].clone();
        let phi = BlockId(2);
        let ret = graph.returnblock;
        graph.exceptblock = BlockId(3);
        graph.blocks = vec![
            Block {
                id: graph.startblock,
                inputargs: block_inputargs(&vars, &[1]),
                operations: vec![
                    SpaceOperation {
                        result: Some(obj.clone()),
                        kind: OpKind::Input {
                            name: "obj".to_string(),
                            ty: ValueType::Ref(Some("W_UnicodeObject".to_string())),
                            class_root: Some("W_UnicodeObject".to_string()),
                        },
                    },
                    SpaceOperation {
                        result: Some(dest.clone()),
                        kind: crate::model::cast_instance_call_result(
                            "Wtf8",
                            obj.clone(),
                            ValueType::Str,
                        ),
                    },
                ],
                exitswitch: None,
                exits: vec![link_to_returnblock(
                    vec![LinkArg::Value(obj.clone()), LinkArg::Value(dest)],
                    phi,
                )],
                framestate: None,
                dead: false,
            },
            Block {
                id: ret,
                inputargs: vec![len_result.clone()],
                operations: vec![],
                exitswitch: None,
                exits: vec![],
                framestate: None,
                dead: false,
            },
            Block {
                id: phi,
                inputargs: vec![obj, view.clone()],
                operations: vec![SpaceOperation {
                    result: Some(len_result.clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath {
                            segments: vec!["__strlen".to_string()],
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![view]),
                        result_ty: ValueType::Int,
                    },
                }],
                exitswitch: None,
                exits: vec![link_to_returnblock(vec![LinkArg::Value(len_result)], ret)],
                framestate: None,
                dead: false,
            },
        ];
        let output = function_graph_to_flowspace(&graph, &registry)
            .expect("a phi of one Wtf8 cast must lower");
        let ops = flow_ops(&output);
        let getattr = ops
            .iter()
            .find(|op| op.opname == "getattr")
            .expect("phi strlen must getattr the string field");
        let Hlvalue::Constant(field) = &getattr.args[1] else {
            panic!("getattr field must be a constant, got {:?}", getattr.args);
        };
        assert!(
            matches!(&field.value, ConstValue::ByteStr(bytes) if bytes == b"value"),
            "got {:?}",
            field.value
        );
    }

    /// A dead predecessor that feeds a non-cast into the phi is not an
    /// incoming edge. The live cast still resolves to the string field.
    #[test]
    fn strlen_on_wtf8_cast_phi_ignores_dead_predecessor() {
        use crate::flowspace::model::ConstValue;

        let registry = empty_call_registry();
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "W_UnicodeObject".to_string(),
            vec![
                ("ob_header".to_string(), "PyObject".to_string()),
                ("value".to_string(), "*mut Utf8Str".to_string()),
                ("byte_len".to_string(), "usize".to_string()),
                ("len".to_string(), "usize".to_string()),
            ],
        );
        registry.bookkeeper().set_struct_fields(Rc::new(fields));

        let mut graph = LegacyGraph::new("strlen_wtf8_phi_dead_pred");
        let vars = mint_vars(&mut graph, 6);
        let obj = vars[1].clone();
        let dest = vars[2].clone();
        let view = vars[3].clone();
        let len_result = vars[4].clone();
        let not_cast = vars[5].clone();
        let phi = BlockId(2);
        let dead_id = BlockId(4);
        let ret = graph.returnblock;
        graph.exceptblock = BlockId(3);
        graph.blocks = vec![
            Block {
                id: graph.startblock,
                inputargs: block_inputargs(&vars, &[1]),
                operations: vec![
                    SpaceOperation {
                        result: Some(obj.clone()),
                        kind: OpKind::Input {
                            name: "obj".to_string(),
                            ty: ValueType::Ref(Some("W_UnicodeObject".to_string())),
                            class_root: Some("W_UnicodeObject".to_string()),
                        },
                    },
                    SpaceOperation {
                        result: Some(dest.clone()),
                        kind: crate::model::cast_instance_call_result(
                            "Wtf8",
                            obj.clone(),
                            ValueType::Str,
                        ),
                    },
                ],
                exitswitch: None,
                exits: vec![link_to_returnblock(
                    vec![LinkArg::Value(obj.clone()), LinkArg::Value(dest)],
                    phi,
                )],
                framestate: None,
                dead: false,
            },
            Block {
                id: ret,
                inputargs: vec![len_result.clone()],
                operations: vec![],
                exitswitch: None,
                exits: vec![],
                framestate: None,
                dead: false,
            },
            Block {
                id: phi,
                inputargs: vec![obj.clone(), view.clone()],
                operations: vec![SpaceOperation {
                    result: Some(len_result.clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath {
                            segments: vec!["__strlen".to_string()],
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![view]),
                        result_ty: ValueType::Int,
                    },
                }],
                exitswitch: None,
                exits: vec![link_to_returnblock(vec![LinkArg::Value(len_result)], ret)],
                framestate: None,
                dead: false,
            },
            Block {
                id: dead_id,
                inputargs: vec![],
                operations: vec![SpaceOperation {
                    result: Some(not_cast.clone()),
                    kind: OpKind::Input {
                        name: "other".to_string(),
                        ty: ValueType::Int,
                        class_root: None,
                    },
                }],
                exitswitch: None,
                exits: vec![link_to_returnblock(
                    vec![LinkArg::Value(obj), LinkArg::Value(not_cast)],
                    phi,
                )],
                framestate: None,
                dead: true,
            },
        ];
        let output = function_graph_to_flowspace(&graph, &registry)
            .expect("a dead non-cast predecessor must not block the live Wtf8 cast");
        let ops = flow_ops(&output);
        let getattr = ops
            .iter()
            .find(|op| op.opname == "getattr")
            .expect("phi strlen must still getattr the string field");
        let Hlvalue::Constant(field) = &getattr.args[1] else {
            panic!("getattr field must be a constant, got {:?}", getattr.args);
        };
        assert!(
            matches!(&field.value, ConstValue::ByteStr(bytes) if bytes == b"value"),
            "got {:?}",
            field.value
        );
    }

    /// No string-typed field in the registered layout: do not accept
    /// `len` of the wrapper. That answer is `ll_strlen` at offset 8,
    /// which is `PyObject.w_class`.
    #[test]
    fn strlen_on_wtf8_cast_without_string_field_declines() {
        let (graph, registry, _dest) = wtf8_strlen_fixture(vec![
            ("ob_header".to_string(), "PyObject".to_string()),
            ("byte_len".to_string(), "usize".to_string()),
            ("len".to_string(), "usize".to_string()),
        ]);
        let err = function_graph_to_flowspace(&graph, &registry)
            .expect_err("unproven Wtf8 view must not become len of the wrapper");
        let msg = err.to_string();
        assert!(
            msg.contains("string field") || msg.contains("__strlen"),
            "decline must name the unproven strlen, got {msg}"
        );
        assert!(
            !msg.contains("wtf8-strlen-unknown-receiver"),
            "a known class with no string field is not an unresolved receiver"
        );
    }

    #[test]
    fn unresolved_wtf8_strlen_receiver_is_classified_for_the_legacy_walker() {
        let msg = "__strlen on a Wtf8 view: receiver class unresolved \
                   (wtf8-strlen-unknown-receiver)";
        assert_eq!(
            crate::translator::rtyper::cutover::unported_category(msg),
            Some("wtf8-strlen-unknown-receiver")
        );
    }

    fn describe_op_kind(kind: &OpKind) -> String {
        if let OpKind::Input { ty, class_root, .. } = kind {
            return format!("Input ty={ty:?} class_root={class_root:?}");
        }
        if let OpKind::FieldRead { field, ty, .. } = kind {
            return format!(
                "FieldRead {}.{} ty={ty:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name
            );
        }
        if let Some(root) = crate::model::cast_instance_root(kind) {
            let result_ty = match kind {
                OpKind::Call { result_ty, .. } => format!("{result_ty:?}"),
                _ => "?".to_string(),
            };
            return format!("cast_instance root={root} result_ty={result_ty}");
        }
        format!("{kind:?}")
    }

    fn describe_strlen_arg_producer(
        graph: &LegacyGraph,
        arg: &crate::flowspace::model::Variable,
    ) -> String {
        fn walk(
            graph: &LegacyGraph,
            arg: &crate::flowspace::model::Variable,
            depth: usize,
        ) -> String {
            if depth > 32 {
                return "phi-walk-too-deep".to_string();
            }
            for block in &graph.blocks {
                for op in &block.operations {
                    if op.result.as_ref() == Some(arg) {
                        return describe_op_kind(&op.kind);
                    }
                }
            }
            let mut incoming = Vec::new();
            for block in &graph.blocks {
                let Some(idx) = block.inputargs.iter().position(|v| v == arg) else {
                    continue;
                };
                for pred in &graph.blocks {
                    for link in &pred.exits {
                        if link.target != block.id {
                            continue;
                        }
                        match link.args.get(idx) {
                            Some(crate::model::LinkArg::Value(v)) => {
                                incoming.push(walk(graph, v, depth + 1));
                            }
                            Some(crate::model::LinkArg::Const(c)) => {
                                incoming.push(format!("const {:?}", c.value));
                            }
                            None => incoming.push("missing-link-arg".to_string()),
                        }
                    }
                }
            }
            incoming.sort();
            incoming.dedup();
            match incoming.as_slice() {
                [] => "no-producer".to_string(),
                [one] => one.clone(),
                many => format!("phi{{{}}}", many.join(" | ")),
            }
        }
        walk(graph, arg, 0)
    }

    fn collect_strlen_sites(graph: &LegacyGraph) -> Vec<(String, bool, bool)> {
        let mut sites = Vec::new();
        for block in &graph.blocks {
            for op in &block.operations {
                let OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } = &op.kind
                else {
                    continue;
                };
                if segments.as_slice() != ["__strlen"] {
                    continue;
                }
                let Some(arg) = args.first().and_then(|a| a.as_variable()) else {
                    sites.push(("const-or-missing-arg".to_string(), false, false));
                    continue;
                };
                let desc = describe_strlen_arg_producer(graph, arg);
                let is_utf8_read = desc == "FieldRead W_UnicodeObject.value ty=Str";
                let is_instance_input =
                    desc.starts_with("Input ") && desc.contains("class_root=Some");
                sites.push((desc, is_utf8_read, is_instance_input));
            }
        }
        sites
    }

    /// Real-site measurement: the `__strlen` argument at a
    /// `w_str_get_wtf8` byte view.  The identity-alias fixture
    /// (`strlen_on_w_str_get_wtf8_identity_alias_receiver_is_instance_not_string`)
    /// does not exercise the frontend, so this lowers the object LLBC.
    /// The view is `w_obj._utf8`: the `value` getfield of the receiver
    /// narrowed to `W_UnicodeObject`, typed `Str`, so `strlen` reads the
    /// rstr `STR` payload and not the wrapper.
    #[test]
    fn strlen_arg_at_real_w_str_get_wtf8_sites_is_the_utf8_field() {
        use majit_charon_reader::Llbc;

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../build/llbc/pyre-object.ullbc"
        );
        let llbc = Llbc::load(path).expect("load real pyre-object LLBC");
        let mut all_sites = Vec::new();
        for name in ["w_str_cut", "str_byte_window"] {
            let graph = crate::front::mir::lower_function(&llbc, name)
                .unwrap_or_else(|err| panic!("lower {name}: {err}"));
            let sites = collect_strlen_sites(&graph);
            assert!(
                !sites.is_empty(),
                "{name} must plant at least one __strlen (w_str_get_wtf8 dest is a byte view)"
            );
            for (desc, is_utf8_read, is_instance) in &sites {
                all_sites.push((name, desc.clone(), *is_utf8_read, *is_instance));
            }
        }

        let instance_sites: Vec<_> = all_sites
            .iter()
            .filter(|(_, _, _, is_instance)| *is_instance)
            .collect();
        assert!(
            instance_sites.is_empty(),
            "__strlen still sees an Instance Input at real sites: {instance_sites:?}"
        );
        let not_utf8: Vec<_> = all_sites
            .iter()
            .filter(|(_, _, is_utf8_read, _)| !*is_utf8_read)
            .collect();
        assert!(
            not_utf8.is_empty(),
            "__strlen argument is not the W_UnicodeObject.value read at: {not_utf8:?}"
        );
    }

    #[test]
    fn translate_op_skips_const_int_define() {
        let value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::ConstInt(7),
        };
        let result = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ConstInt must translate to skip");
        assert!(
            result.is_empty(),
            "ConstInt define is inlined by build_value_to_hlvalue_map; \
             translate_op must yield empty Vec"
        );
    }

    #[test]
    fn translate_op_skips_const_float_define() {
        let value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::ConstFloat(0),
        };
        let result = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ConstFloat must translate to skip");
        assert!(result.is_empty());
    }

    #[test]
    fn translate_op_raw_load_lowers_to_raw_load_spaceop() {
        // RawLoad carries through as the `raw_load` llop (`llop.raw_load`);
        // the descr seeds the result annotation via `lltype_to_annotation`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_raw_load_fixture");
        let vars = mint_vars(&mut graph, 4);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::RawLoad {
                base: vars[1].clone(),
                offset: vars[2].clone(),
                item_ty: ValueType::Int,
                itemsize: 8,
                is_item_signed: true,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("RawLoad arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "raw_load");
        assert_eq!(lowered.args.len(), 2);
        let Hlvalue::Variable(result) = &lowered.result else {
            panic!("RawLoad result must be a Variable");
        };
        match result.annotation.borrow().as_ref().map(|rc| (**rc).clone()) {
            Some(SomeValue::Integer(i)) => {
                assert!(!i.unsigned, "i64 raw_load is signed");
                assert_eq!(i.knowntype(), KnownType::Int);
            }
            other => panic!("i64 RawLoad must seed SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn translate_op_raw_store_lowers_to_raw_store_spaceop() {
        // RawStore carries through as the `raw_store` llop; the result is
        // the Void annotation (`lltype_to_annotation(Void)`).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_raw_store_fixture");
        let vars = mint_vars(&mut graph, 4);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::RawStore {
                base: vars[1].clone(),
                offset: vars[2].clone(),
                value: vars[3].clone(),
                item_ty: ValueType::Int,
                itemsize: 8,
                is_item_signed: true,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("RawStore arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "raw_store");
        assert_eq!(lowered.args.len(), 3);
        let Hlvalue::Variable(result) = &lowered.result else {
            panic!("RawStore result must be a Variable");
        };
        match result.annotation.borrow().as_ref().map(|rc| (**rc).clone()) {
            Some(SomeValue::None_(_)) => {}
            other => panic!("RawStore must seed s_None, got {other:?}"),
        }
    }

    fn flowin_raw_load_item_annotation(
        item_ty: ValueType,
        itemsize: usize,
        is_item_signed: bool,
    ) -> SomeValue {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::Block as FlowspaceBlock;
        use crate::flowspace::model::FunctionGraph as FlowspaceGraph;

        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("raw_load_ann_fixture");
        let vars = mint_vars(&mut graph, 4);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::RawLoad {
                base: vars[1].clone(),
                offset: vars[2].clone(),
                item_ty,
                itemsize,
                is_item_signed,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("RawLoad arm must lower");
        let ann = RPythonAnnotator::new(None, None, None, false);
        let startblock = FlowspaceBlock::shared(Vec::new());
        startblock.borrow_mut().operations = translated;
        let fs_graph = Rc::new(std::cell::RefCell::new(FlowspaceGraph::new(
            "raw_load_ann_fixture",
            startblock.clone(),
        )));
        ann.processblock(&fs_graph, &startblock)
            .expect("flowin of raw_load must succeed");
        let blk = startblock.borrow();
        let Hlvalue::Variable(result) = &blk.operations[0].result else {
            panic!("raw_load result must be a Variable");
        };
        result
            .annotation
            .borrow()
            .as_ref()
            .map(|rc| (**rc).clone())
            .expect("flowin must bind the raw_load result")
    }

    #[test]
    fn raw_load_i64_u8_f64_flowin_binds_lltype_to_annotation() {
        match flowin_raw_load_item_annotation(ValueType::Int, 8, true) {
            SomeValue::Integer(i) => {
                assert!(!i.unsigned, "i64 is signed");
                assert_eq!(i.knowntype(), KnownType::Int);
            }
            other => panic!("i64 raw_load must bind SomeInteger, got {other:?}"),
        }
        match flowin_raw_load_item_annotation(ValueType::Unsigned, 1, false) {
            SomeValue::Integer(i) => {
                assert!(i.unsigned, "u8 is unsigned");
                assert_eq!(i.knowntype(), KnownType::Ruint);
            }
            other => panic!("u8 raw_load must bind SomeInteger unsigned, got {other:?}"),
        }
        match flowin_raw_load_item_annotation(ValueType::Int, 1, false) {
            SomeValue::Integer(i) => {
                assert!(
                    i.unsigned,
                    "1-byte unsigned Int is rffi.UCHAR, not lltype.Char"
                );
                assert_eq!(i.knowntype(), KnownType::Ruint);
            }
            other => {
                panic!("1-byte unsigned Int raw_load must bind SomeInteger unsigned, got {other:?}")
            }
        }
        match flowin_raw_load_item_annotation(ValueType::Float, 8, false) {
            SomeValue::Float(_) => {}
            other => panic!("f64 raw_load must bind SomeFloat, got {other:?}"),
        }
    }

    #[test]
    fn translate_op_binop_lowers_to_passthrough_spaceop() {
        // BinOp arm: `add` / `sub` / `lt` / ... pass through to a
        // flowspace SpaceOperation with the same opname; lhs/rhs args
        // get resolved via lookup_operand and the result Hlvalue via
        // resolve_result_hlvalue.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let lhs_var = Hlvalue::Variable(Variable::new());
        let rhs_var = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), lhs_var.clone());
        value_map.insert(vars[2].clone(), rhs_var.clone());
        value_map.insert(vars[3].clone(), result_var.clone());

        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: vars[1].clone(),
                rhs: vars[2].clone(),
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("BinOp arm must lower");
        assert_eq!(translated.len(), 1, "BinOp lowers to exactly one SpaceOp");
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "add", "opname passes through unchanged");
        assert_eq!(lowered.args.len(), 2);
    }

    #[test]
    fn translate_op_getslice_lowers_to_getslice_spaceop() {
        // GetSlice arm: the three `(list, start, stop)` operands route
        // through lookup_operand and the result via resolve_result_hlvalue,
        // producing a flowspace `getslice` SpaceOperation the annotator's
        // list handler (unaryop.rs) consumes.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_getslice_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        for slot in [1, 2, 3, 4] {
            value_map.insert(vars[slot].clone(), Hlvalue::Variable(Variable::new()));
        }

        let op = SpaceOperation {
            result: Some(vars[4].clone()),
            kind: OpKind::GetSlice {
                args: vec![vars[1].clone(), vars[2].clone(), vars[3].clone()],
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("GetSlice arm must lower");
        assert_eq!(
            translated.len(),
            1,
            "GetSlice lowers to exactly one SpaceOp"
        );
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "getslice", "opname is getslice");
        assert_eq!(
            lowered.args.len(),
            3,
            "getslice carries (list, start, stop)"
        );
    }

    #[test]
    fn translate_op_binop_undefined_lhs_surfaces_invariant_break() {
        // BinOp arm threads operand lookups through lookup_operand, so a
        // missing lhs surfaces the "adapter invariant broken" message
        // rather than a silent panic.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 100); // vars[0..100]
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: vars[99].clone(), // not in value_map
                rhs: vars[2].clone(),
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("undefined BinOp operand must surface invariant break");
        let msg = format!("{err}");
        assert!(msg.contains("undefined operand"));
    }

    #[test]
    fn translate_op_call_function_path_lowers_to_simple_call() {
        // Call::FunctionPath → `simple_call(callable_host, args...)` per
        // `flowspace/operation.py SimpleCall.opname = 'simple_call'`.
        // The callable Constant wraps the `CallRegistry` entry's
        // synthetic `HostObject::UserFunction` so the rtyper's
        // `bookkeeper.getdesc` short-circuits onto the registered
        // FunctionDesc.
        use crate::flowspace::argument::Signature;
        use crate::translator::rtyper::call_registry::FunctionPathKey;
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let registry = empty_call_registry();
        registry.get_or_register(
            FunctionPathKey::from_segments(["a", "b"]),
            Signature::new(vec!["x".into()], None, None),
        );
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["a".into(), "b".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &registry).expect("Call::FunctionPath must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        assert_eq!(lowered.args.len(), 2, "callable + 1 arg");
        let Hlvalue::Constant(ref callable) = lowered.args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        // Synthetic GraphFunc takes the last path segment as its
        // `__name__`, mirroring upstream `func.__name__` (the leaf
        // identifier, not the dotted module path).
        assert_eq!(host.qualname(), "b");
    }

    #[test]
    fn translate_op_jit_force_virtualizable_bypasses_runtime_helper_graph() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_force_virtualizable_fixture");
        let vars = mint_vars(&mut graph, 3);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![
                        "pyre_interpreter".into(),
                        "executioncontext".into(),
                        "jit_force_virtualizable".into(),
                    ],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone()]),
                result_ty: ValueType::Void,
            },
        };

        // No CallRegistry entry is intentional: upstream rtyping emits this
        // force as an LLOp and does not annotate its runtime implementation.
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("jit_force_virtualizable must lower through ExtRegistry");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        let Hlvalue::Constant(callable) = &lowered.args[0] else {
            panic!("marker callable must be constant");
        };
        let ConstValue::HostObject(host) = &callable.value else {
            panic!("marker callable must be a HostObject");
        };
        assert_eq!(
            host.qualname(),
            "pyre_interpreter.executioncontext.jit_force_virtualizable"
        );
        assert!(crate::translator::rtyper::extregistry::is_registered(
            &callable.value
        ));
    }

    #[test]
    fn translate_op_array_repeat_expands_to_newlist_then_mul() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_array_repeat_fixture");
        let vars = mint_vars(&mut graph, 4);
        let fill = Hlvalue::Variable(Variable::new());
        let count = Hlvalue::Variable(Variable::new());
        let result = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[1].clone(), fill.clone());
        value_map.insert(vars[2].clone(), count.clone());
        value_map.insert(vars[3].clone(), result.clone());

        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__array_repeat".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone(), vars[2].clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("two-arg __array_repeat must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "newlist");
        assert_eq!(translated[0].args, vec![fill.clone()]);
        assert_eq!(translated[1].opname, "mul");
        assert_eq!(translated[1].args[0], translated[0].result);
        assert_eq!(translated[1].args[1], count);
        assert_eq!(translated[1].result, result);
    }

    #[test]
    fn translate_runtime_capacity_helper_becomes_newlist_hint() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_newlist_hint_fixture");
        let vars = mint_vars(&mut graph, 2);
        let hint = Hlvalue::Variable(Variable::new());
        let result = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[0].clone(), hint.clone());
        value_map.insert(vars[1].clone(), result.clone());
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![
                        "pyre_interpreter".into(),
                        "builtins".into(),
                        "try_pyobject_vec_with_capacity".into(),
                    ],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[0].clone()]),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("try_pyobject_vec_with_capacity must lower as the newlist_hint intrinsic");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "newlist_hint");
        assert_eq!(translated[0].args, vec![hint]);
        assert_eq!(translated[0].result, result);
        assert!(op_canraise(&op.kind));
    }

    /// The frontend cannot see the adapter's expansion: `lower_function`
    /// still contains the raw `alloc::vec::from_elem` call, while the
    /// flowspace graph must carry `newlist` + `mul` instead.  This real-LLBC
    /// anchor therefore runs both stages and checks the adapter output.
    #[test]
    #[ignore]
    fn bind_kwargs_to_signature_has_no_residual_vec_from_elem_after_adaptation() {
        let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
        let llbc = majit_charon_reader::Llbc::load(path).expect("load real LLBC");
        let legacy = crate::front::mir::lower_function(&llbc, "bind_kwargs_to_signature")
            .expect("lower bind_kwargs_to_signature");
        let raw_from_elem = legacy
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath { segments, .. },
                        ..
                    } if is_vec_from_elem_segments(segments, 2)
                )
            })
            .count();
        assert!(raw_from_elem > 0, "fixture must contain vec::from_elem");

        let var_map = build_value_to_variable_map(&legacy);
        let value_map = build_value_to_hlvalue_map(&legacy, &var_map);
        let registry = empty_call_registry();
        let expansions = legacy
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath { segments, .. },
                        ..
                    } if is_vec_from_elem_segments(segments, 2)
                )
            })
            .map(|op| translate_op(op, &value_map, &registry).expect("adapt from_elem"))
            .collect::<Vec<_>>();
        assert!(!expansions.is_empty());
        for translated in expansions {
            assert_eq!(translated.len(), 2);
            assert_eq!(translated[0].opname, "newlist");
            assert_eq!(translated[1].opname, "mul");
        }
    }

    #[test]
    fn translate_op_getslice_minusone_expands_after_annotation() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_getslice_minusone_fixture");
        let vars = mint_vars(&mut graph, 2);
        let slice = Hlvalue::Variable(Variable::new());
        let result = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[0].clone(), slice.clone());
        value_map.insert(vars[1].clone(), result.clone());
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__getslice_minusone".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[0].clone()]),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("minus-one marker must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "getslice");
        assert_eq!(translated[0].args[0], slice);
        assert_eq!(
            translated[0].args[1],
            Hlvalue::Constant(Constant::new(ConstValue::Int(0)))
        );
        assert_eq!(
            translated[0].args[2],
            Hlvalue::Constant(Constant::new(ConstValue::Int(-1)))
        );
        assert_eq!(translated[0].result, result);
    }

    #[test]
    fn translate_op_getslice_rangeto_expands_after_annotation() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_getslice_rangeto_fixture");
        let vars = mint_vars(&mut graph, 3);
        let slice = Hlvalue::Variable(Variable::new());
        let end = Hlvalue::Variable(Variable::new());
        let result = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[0].clone(), slice.clone());
        value_map.insert(vars[1].clone(), end.clone());
        value_map.insert(vars[2].clone(), result.clone());
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__getslice_rangeto".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[0].clone(), vars[1].clone()]),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("RangeTo marker must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "getslice");
        assert_eq!(translated[0].args[0], slice);
        assert_eq!(
            translated[0].args[1],
            Hlvalue::Constant(Constant::new(ConstValue::Int(0)))
        );
        assert_eq!(translated[0].args[2], end);
        assert_eq!(translated[0].result, result);
    }

    #[test]
    fn translate_op_getslice_rangefrom_expands_after_annotation() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_getslice_rangefrom_fixture");
        let vars = mint_vars(&mut graph, 3);
        let slice = Hlvalue::Variable(Variable::new());
        let start = Hlvalue::Variable(Variable::new());
        let result = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[0].clone(), slice.clone());
        value_map.insert(vars[1].clone(), start.clone());
        value_map.insert(vars[2].clone(), result.clone());
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__getslice_rangefrom".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[0].clone(), vars[1].clone()]),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("RangeFrom marker must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "getslice");
        assert_eq!(translated[0].args[0], slice);
        assert_eq!(translated[0].args[1], start);
        assert_eq!(
            translated[0].args[2],
            Hlvalue::Constant(Constant::new(ConstValue::None))
        );
        assert_eq!(translated[0].result, result);
    }

    #[test]
    fn translate_op_call_function_path_exact_registry_wins_over_leaf_collision() {
        // Two user fns share the bare leaf `shared_leaf` under different
        // module paths, so `lookup_with_leaf_match` is ambiguous (distinct
        // HostObjects → it returns None).  The exact `call_registry.lookup`
        // runs first in the resolution order, so a FunctionPath spelling
        // the full `othermod::shared_leaf` path resolves to that exact
        // entry; were resolution to fall through to the leaf-match
        // fallback it would error instead.  Locks in the
        // exact-before-leaf-match precedence the resolver depends on.
        use crate::flowspace::argument::Signature;
        use crate::translator::rtyper::call_registry::FunctionPathKey;
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let registry = empty_call_registry();
        let other_entry = registry.get_or_register(
            FunctionPathKey::from_segments(["othermod", "shared_leaf"]),
            Signature::new(vec!["x".into()], None, None),
        );
        let mymod_entry = registry.get_or_register(
            FunctionPathKey::from_segments(["mymod", "shared_leaf"]),
            Signature::new(vec!["x".into()], None, None),
        );
        assert_ne!(
            other_entry.host_object, mymod_entry.host_object,
            "colliding-leaf entries must be distinct HostObjects"
        );
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["othermod".into(), "shared_leaf".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &registry).expect("exact FunctionPath must lower");
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        let Hlvalue::Constant(ref callable) = lowered.args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        assert_eq!(
            host, &other_entry.host_object,
            "exact registry lookup must win over the leaf-match fallback"
        );
        assert_ne!(
            host, &mymod_entry.host_object,
            "must not resolve to the colliding-leaf sibling"
        );
    }

    #[test]
    fn translate_op_call_type_lowers_to_type_spaceop() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 4);
        let operand = Variable::new();
        value_map.insert(vars[1].clone(), Hlvalue::Variable(operand.clone()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::function_path(["type"]),
                args: crate::model::call_args(vec![vars[1].clone()]),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("FunctionPath [type] must lower to the type op");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "type");
        assert_eq!(translated[0].args.len(), 1);
        assert!(matches!(&translated[0].args[0], Hlvalue::Variable(v) if *v == operand));
    }

    #[test]
    fn translate_op_call_simple_call_uses_const_callable_at_args_0() {
        // RPython `flowcontext.py exc_from_raise` emits
        // `op.simple_call(const(exc_class), *args)`.  The frontend
        // now carries that Constant at `Call.args[0]`; FunctionPath
        // `["simple_call"]` is the op name, not a second callable.
        // A colliding registry leaf must not be prepended.
        use crate::flowspace::argument::Signature;
        use crate::translator::rtyper::call_registry::FunctionPathKey;
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let registry = empty_call_registry();
        registry.get_or_register(
            FunctionPathKey::from_segments(["mymod", "simple_call"]),
            Signature::new(vec!["msg".into()], None, None),
        );
        let expected = HOST_ENV
            .lookup_builtin("ValueError")
            .expect("bootstrap_builtin_exceptions must register ValueError");
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["simple_call".into()],
                    fun_decl_id: None,
                },
                args: vec![
                    LinkArg::from(ConstValue::HostObject(expected.clone())),
                    LinkArg::from(vars[1].clone()),
                ],
                result_ty: ValueType::Ref(None),
            },
        };
        let translated =
            translate_op(&op, &value_map, &registry).expect("simple_call(const(class)) must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "simple_call");
        assert_eq!(translated[0].args.len(), 2);
        let Hlvalue::Constant(ref callable) = translated[0].args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("callable must be ConstValue::HostObject");
        };
        assert_eq!(
            host, &expected,
            "callable must be the args[0] class Constant, not a wrapped path"
        );
        assert!(
            matches!(&translated[0].args[1], Hlvalue::Variable(_)),
            "message arg must stay a Variable"
        );
    }

    #[test]
    fn translate_op_dyn_call_lowers_to_simple_call() {
        // `front::mir` mints `Call(["__dyn_call"], [funcptr, *args])` for
        // a Dynamic call whose operand is not a recovered vtable slot.
        // That is a call through a function pointer: the pre-rtyper
        // shape is `simple_call(funcptr, *args)`, the same op
        // `PtrRepr.rtype_simple_call` later emits `indirect_call` for.
        // args[0] is the callable Variable, not a wrapped HostObject.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 4);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__dyn_call".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone(), vars[2].clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("__dyn_call marker must lower to simple_call");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "simple_call");
        assert_eq!(translated[0].args.len(), 2);
        assert!(
            matches!(&translated[0].args[0], Hlvalue::Variable(_)),
            "callable must be the funcptr Variable, not a wrapped path"
        );
        assert!(
            matches!(&translated[0].args[1], Hlvalue::Variable(_)),
            "call argument must stay a Variable"
        );
    }

    #[test]
    fn translate_op_call_function_path_falls_back_to_host_env_builtin() {
        // Single-segment FunctionPath unregistered in CallRegistry
        // falls back to HOST_ENV.lookup_builtin(name), letting frontend
        // `Expr::Cast` lowering emit
        // `Call { target: FunctionPath { segments: vec!["int"], fun_decl_id: None }, args }`
        // and route through `BuiltinFunctionRepr.rtype_simple_call →
        // BUILTIN_TYPER["int"] → rtype_builtin_int`.  Mirrors upstream
        // `flowspace/flowcontext.py:LOAD_GLOBAL` resolving against
        // `__builtin__.__dict__` after `frame.globals` misses.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["int".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Call::FunctionPath single-segment must fall back to HOST_ENV");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        assert_eq!(lowered.args.len(), 2);
        let Hlvalue::Constant(ref callable) = lowered.args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        // HOST_ENV.lookup_builtin("int") returns the shared `int` class
        // HostObject — the same identity BUILTIN_TYPER keys its
        // `rtype_builtin_int` entry on.
        let expected = HOST_ENV
            .lookup_builtin("int")
            .expect("HOST_ENV bootstrap must register __builtin__.int");
        assert_eq!(host, &expected);
    }

    #[test]
    fn qualified_builtin_cast_ignores_a_same_leaf_registry_function() {
        // RPython carries the actual `float` object in the flowspace
        // Constant, so `BUILTIN_TYPER[float]` cannot be captured by an
        // unrelated function with the same `__name__`.  Pyre's MIR carrier
        // is a path; the explicit `__builtin__` owner is how the cast path
        // preserves that object identity across the flat CallRegistry.
        use crate::flowspace::argument::Signature;
        use crate::translator::rtyper::call_registry::FunctionPathKey;

        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("qualified_builtin_cast_fixture");
        let vars = mint_vars(&mut graph, 3);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));

        let registry = empty_call_registry();
        registry.get_or_register(
            FunctionPathKey::from_segments(["float"]),
            Signature::new(vec!["value".into()], None, None),
        );
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__builtin__".into(), "float".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone()]),
                result_ty: ValueType::Float,
            },
        };

        let translated = translate_op(&op, &value_map, &registry)
            .expect("qualified builtin cast must resolve through HOST_ENV");
        let Hlvalue::Constant(ref callable) = translated[0].args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        let expected = HOST_ENV
            .lookup_builtin("float")
            .expect("HOST_ENV bootstrap must register __builtin__.float");
        assert_eq!(host, &expected);
        assert!(
            !host.is_user_function(),
            "the same-leaf registry function must not capture the builtin cast"
        );
    }

    #[test]
    fn translate_op_call_function_path_resolves_host_module_attr_for_lltype_cast() {
        // Multi-segment FunctionPath unregistered in CallRegistry
        // falls back to Layer 3 (HOST_ENV.import_module + module_get).
        // Mirrors upstream `LOAD_GLOBAL lltype` → `LOAD_ATTR cast_ptr_\
        // to_int` chain (`flowcontext.py`).  The resolved
        // HostObject is the same shared identity that BUILTIN_TYPER
        // keys its `rtype_cast_ptr_to_int` typer on
        // (`rbuiltin.py`).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![
                        "rpython".into(),
                        "rtyper".into(),
                        "lltypesystem".into(),
                        "lltype".into(),
                        "cast_ptr_to_int".into(),
                    ],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Layer 3 host-module attr lookup must resolve lltype.cast_ptr_to_int");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "simple_call");
        let Hlvalue::Constant(ref callable) = translated[0].args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        let expected = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("cast_ptr_to_int"))
            .expect("populate_host_env must register lltype.cast_ptr_to_int");
        assert_eq!(host, &expected);
    }

    #[test]
    fn translate_op_call_function_path_rejects_unregistered_multi_segment_path() {
        // Defense-in-depth: an arbitrary multi-segment user path
        // (`some.unknown.module.path`) that misses every layer —
        // CallRegistry, HOST_ENV single-segment builtin, and
        // HOST_ENV module attr (because `some.unknown.module` is not
        // curated in `populate_host_env`) — surfaces a `TyperError`
        // rather than a silent host-attribute resolution.  This pins
        // the implicit HOST_ENV-curation bound that the Layer 3
        // fallback relies on.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![
                        "some".into(),
                        "unknown".into(),
                        "module".into(),
                        "path".into(),
                    ],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("Unregistered FunctionPath must surface TyperError, not silently resolve");
        let msg = format!("{err}");
        assert!(
            msg.contains("not registered in CallRegistry"),
            "error must name the missing-registration invariant, got: {msg}"
        );
    }

    #[test]
    fn translate_op_cast_pointer_marker_rebuilds_two_arg_upstream_call() {
        // `__cast_pointer` marker reconstructs the upstream 2-arg
        // `cast_pointer(PTRTYPE, ptr)` shape (lltype.py):
        // constant callable + interned target class, then the
        // pointer operand.  The frontend carries the root as a
        // trailing ByteStr Constant.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 10);
        let operand = Variable::new();
        value_map.insert(vars[1].clone(), Hlvalue::Variable(operand.clone()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: crate::model::cast_pointer_call("W_CastTarget", vars[1].clone()),
        };
        let registry = empty_call_registry();
        let translated = translate_op(&op, &value_map, &registry)
            .expect("__cast_pointer marker must lower to simple_call");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        assert_eq!(lowered.args.len(), 3);
        let Hlvalue::Constant(ref callable) = lowered.args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("cast_pointer callable must be ConstValue::HostObject");
        };
        let expected = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("cast_pointer"))
            .expect("populate_host_env must register lltype.cast_pointer");
        assert_eq!(host, &expected);
        let Hlvalue::Constant(ref class_const) = lowered.args[1] else {
            panic!("target class must be a Constant");
        };
        let ConstValue::HostObject(ref class_host) = class_const.value else {
            panic!("target class must be ConstValue::HostObject");
        };
        // Interned by qualname — every cast site shares one HostObject
        // identity, so `getdesc` resolves them to one ClassDesc.
        let interned = registry
            .bookkeeper()
            .intern_class_by_qualname("W_CastTarget");
        assert_eq!(class_host, &interned);
        assert!(matches!(lowered.args[2], Hlvalue::Variable(ref v) if *v == operand));
    }

    #[test]
    fn translate_op_cast_instance_uses_trailing_root_constant() {
        // `cast_instance_call` puts the target root at args[1] as a
        // ByteStr Constant.  The adapter prepends the HOST_ENV
        // callable and leaves that mixed list intact.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 10);
        let operand = Variable::new();
        value_map.insert(vars[1].clone(), Hlvalue::Variable(operand.clone()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: crate::model::cast_instance_call("W_CastTarget", vars[1].clone()),
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("cast_instance_call must lower to simple_call");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "simple_call");
        assert_eq!(translated[0].args.len(), 3);
        let Hlvalue::Constant(ref callable) = translated[0].args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("callable must be HostObject");
        };
        let expected = HOST_ENV
            .lookup_builtin(crate::runtime_names::shims::CAST_INSTANCE)
            .expect("HOST_ENV must register __cast_instance_intrinsic");
        assert_eq!(host, &expected);
        assert!(
            matches!(&translated[0].args[1], Hlvalue::Variable(v) if *v == operand),
            "operand stays args[1] of simple_call"
        );
        let Hlvalue::Constant(ref root) = translated[0].args[2] else {
            panic!("root must stay a trailing Constant");
        };
        assert_eq!(root.value.as_pystr(), Some("W_CastTarget"));
    }

    #[test]
    fn translate_op_call_synthetic_transparent_ctor_lowers_to_instantiate() {
        // Call::SyntheticTransparentCtor mirrors Rust's `Class { fields }`
        // ctor: an allocation without `__init__`, i.e.
        // `objectmodel.instantiate(Class)`; the field writes follow it.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let ctor = |args: Vec<Variable>| SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::synthetic_transparent_ctor("Point"),
                args: crate::model::call_args(args),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&ctor(Vec::new()), &value_map, &empty_call_registry())
            .expect("Call::SyntheticTransparentCtor must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "simple_call");
        let [
            Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(ref callable),
                ..
            }),
            Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(ref class),
                ..
            }),
        ] = translated[0].args[..]
        else {
            panic!("expected simple_call(instantiate, class)");
        };
        assert_eq!(callable.qualname(), "rpython.rlib.objectmodel.instantiate");
        assert!(class.is_class());
        assert_eq!(class.qualname(), "Point");

        translate_op(
            &ctor(vec![vars[1].clone()]),
            &value_map,
            &empty_call_registry(),
        )
        .expect_err("a Rust aggregate ctor carries no call operands");
    }

    /// Two graphs that construct the same `MutRef<T>` cell share one class
    /// host. The cell is one placeholder per pointee shape, so a callee's
    /// field read and every caller's constructor resolve the same classdef.
    #[test]
    fn mut_ref_cell_graphs_share_one_class_host() {
        let name = "MutRef<*mut PyObject>";
        let registry = empty_call_registry();
        let host_of = |label: &str| {
            let mut graph = LegacyGraph::new(label);
            let vars = mint_vars(&mut graph, 4);
            let startblock = Block {
                id: graph.startblock,
                inputargs: block_inputargs(&vars, &[1]),
                operations: vec![SpaceOperation {
                    result: Some(vars[2].clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::synthetic_transparent_struct_ctor(
                            Vec::new(),
                            name,
                        ),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some(name.to_string())),
                    },
                }],
                exitswitch: None,
                exits: vec![link_to_returnblock(
                    vec![LinkArg::Value(vars[2].clone())],
                    graph.returnblock,
                )],
                framestate: None,
                dead: false,
            };
            let returnblock = Block {
                id: graph.returnblock,
                inputargs: block_inputargs(&vars, &[3]),
                operations: vec![],
                exitswitch: None,
                exits: vec![],
                framestate: None,
                dead: false,
            };
            graph.blocks = vec![startblock, returnblock];
            let output = function_graph_to_flowspace(&graph, &registry)
                .expect("MutRef construction must adapt");
            let flowspace_graph = output.graph.borrow();
            let startblock = flowspace_graph.startblock.borrow();
            let op = startblock
                .operations
                .iter()
                .find(|op| op.opname == "simple_call")
                .expect("MutRef ctor lowers to instantiate");
            let Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(class),
                ..
            }) = &op.args[1]
            else {
                panic!("instantiate's class argument must be a HostObject");
            };
            assert_eq!(class.qualname(), name);
            class.clone()
        };
        assert_eq!(host_of("mut_ref_caller_a"), host_of("mut_ref_caller_b"));
    }

    /// A `MutRef<T>` parameter has no struct-field rows, but its input cell
    /// must still be the constructor's interned class — the same host
    /// `SyntheticTransparentCtor` registers through `intern_class_by_qualname`.
    #[test]
    fn mut_ref_input_cell_seeds_constructor_class_host() {
        let name = "MutRef<*mut PyObject>";
        let registry = empty_call_registry();
        let bk = registry.bookkeeper().clone();
        bk.set_struct_fields(Rc::new(crate::front::StructFieldRegistry::default()));

        let mut graph = LegacyGraph::new("mut_ref_callee");
        let entry = graph.startblock;
        let cell = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "cell".to_string(),
                    ty: ValueType::Ref(Some(name.to_string())),
                    class_root: Some(name.to_string()),
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, cell);

        let cells = derive_subject_inputcells(&graph, Some(&bk))
            .expect("MutRef input must seed a classdef");
        let SomeValue::Instance(inst) = &cells[0] else {
            panic!("MutRef input must be SomeInstance, got {:?}", cells[0]);
        };
        let classdef = inst
            .classdef
            .as_ref()
            .expect("MutRef input cell must carry a classdef");
        let seeded = classdef.borrow().classdesc.borrow().pyobj.clone();

        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut ctor_graph = LegacyGraph::new("mut_ref_ctor");
        let vars = mint_vars(&mut ctor_graph, 2);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::synthetic_transparent_struct_ctor(
                    Vec::new(),
                    name,
                ),
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Ref(Some(name.to_string())),
            },
        };
        let translated = translate_op(&op, &value_map, &registry).expect("MutRef ctor must lower");
        let Hlvalue::Constant(Constant {
            value: ConstValue::HostObject(ctor_host),
            ..
        }) = &translated[0].args[1]
        else {
            panic!("instantiate's class argument must be a HostObject");
        };
        assert_eq!(ctor_host.qualname(), name);
        assert_eq!(&seeded, ctor_host);
    }

    /// A call of each `ll_vec_*` helper lowers to the high-level `Vec`
    /// operation it implements.
    #[test]
    fn translate_op_ll_vec_helper_calls_lower_to_high_level_ops() {
        use majit_ir::rvec::{VecItemKind, VecOp, vec_helper_path};
        for kind in VecItemKind::ALL {
            for op in VecOp::ALL {
                let nargs = match op {
                    VecOp::NewEmpty => 0,
                    VecOp::NewHint
                    | VecOp::Length
                    | VecOp::Reverse
                    | VecOp::Free
                    | VecOp::Items => 1,
                    VecOp::GetItem | VecOp::Append | VecOp::AllocAndSet | VecOp::Extend => 2,
                    VecOp::SetItem | VecOp::ExtendFromSlice => 3,
                };
                let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
                let mut graph = LegacyGraph::new("translate_op_fixture");
                let vars = mint_vars(&mut graph, 11);
                let result_var = Hlvalue::Variable(Variable::new());
                value_map.insert(vars[1].clone(), result_var.clone());
                let mut arg_hls = Vec::new();
                for var in &vars[2..2 + nargs] {
                    let hl = Hlvalue::Variable(Variable::new());
                    value_map.insert(var.clone(), hl.clone());
                    arg_hls.push(hl);
                }
                let path = vec_helper_path(op, kind);
                let spaceop = SpaceOperation {
                    result: Some(vars[1].clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath {
                            segments: path.split("::").map(String::from).collect(),
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vars[2..2 + nargs].to_vec()),
                        result_ty: ValueType::Int,
                    },
                };
                let translated = translate_op(&spaceop, &value_map, &empty_call_registry())
                    .unwrap_or_else(|err| panic!("{path} must lower: {err:?}"));
                let last = translated.last().expect("at least one op");
                assert_eq!(last.result, result_var, "{path}");
                match op {
                    VecOp::NewEmpty | VecOp::NewHint | VecOp::AllocAndSet => {
                        assert_eq!(translated.len(), 1);
                        assert_eq!(last.opname, "newrustvec");
                        let Hlvalue::Constant(letter) = &last.args[0] else {
                            panic!("{path}: kind letter must be a Constant");
                        };
                        assert_eq!(
                            letter.value.as_text(),
                            Some(kind.kind_char().to_string().as_str())
                        );
                        assert_eq!(&last.args[1..], arg_hls.as_slice());
                    }
                    VecOp::Length | VecOp::GetItem | VecOp::SetItem => {
                        assert_eq!(translated.len(), 1);
                        let opname = match op {
                            VecOp::Length => "len",
                            VecOp::GetItem => "getitem",
                            _ => "setitem",
                        };
                        assert_eq!(last.opname, opname);
                        assert_eq!(last.args, arg_hls);
                    }
                    VecOp::Append
                    | VecOp::Reverse
                    | VecOp::Free
                    | VecOp::Items
                    | VecOp::ExtendFromSlice
                    | VecOp::Extend => {
                        let method = match op {
                            VecOp::Append => "append",
                            VecOp::Reverse => "reverse",
                            VecOp::Items => "items",
                            VecOp::ExtendFromSlice => "extend_from_slice",
                            VecOp::Extend => "extend",
                            _ => "free",
                        };
                        assert_eq!(translated.len(), 2);
                        assert_eq!(translated[0].opname, "getattr");
                        assert_eq!(translated[0].args[0], arg_hls[0]);
                        let Hlvalue::Constant(name) = &translated[0].args[1] else {
                            panic!("{path}: method name must be a Constant");
                        };
                        assert_eq!(name.value.as_text(), Some(method));
                        assert_eq!(last.opname, "simple_call");
                        assert_eq!(last.args[0], translated[0].result);
                        assert_eq!(&last.args[1..], &arg_hls[1..]);
                    }
                }
            }
        }
    }

    #[test]
    fn translate_op_enumerate_marker_calls_the_enumerate_builtin() {
        // `__majit_enumerate(lst)` (the front's spelling of
        // `lst.iter().enumerate()`) → `simple_call(enumerate, lst)`, so
        // `builtin_enumerate` types it `SomeIterator(s_list, "enumerate")`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 2);
        let lst = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[0].clone(), lst.clone());
        value_map.insert(vars[1].clone(), result_var.clone());
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![crate::runtime_names::shims::ENUMERATE.into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[0].clone()]),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("__majit_enumerate marker must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "simple_call");
        let Hlvalue::Constant(ref callable) = translated[0].args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("callable must be ConstValue::HostObject");
        };
        assert_eq!(
            *host,
            HOST_ENV
                .lookup_builtin("enumerate")
                .expect("enumerate builtin")
        );
        assert_eq!(translated[0].args[1..], [lst]);
        assert_eq!(translated[0].result, result_var);
    }

    #[test]
    fn translate_op_stringbuilder_new_marker_lowers_to_newstringbuilder() {
        // Bare `new()` `__majit_stringbuilder_new` (front builder-mode ctor
        // rewrite) → one `newstringbuilder` op with no operands; the annotator
        // types its result `SomeStringBuilder` and `rtype_newstringbuilder`
        // picks `ll_new(INIT_SIZE)` (`len(hop.args_v) == 0`).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11);
        let result_var = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[1].clone(), result_var.clone());
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![crate::runtime_names::shims::STRINGBUILDER_NEW.into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("__majit_stringbuilder_new marker must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "newstringbuilder");
        assert!(translated[0].args.is_empty());
        assert_eq!(translated[0].result, result_var);
    }

    #[test]
    fn translate_op_stringbuilder_new_marker_forwards_with_capacity_size() {
        // `with_capacity(n)` `__majit_stringbuilder_new` → `newstringbuilder`
        // carrying the size operand `n`.  `rtype_newstringbuilder` then takes
        // the `len(hop.args_v) != 0` branch and threads `ll_new(n)`
        // (rtyper/rbuilder.py) instead of the `INIT_SIZE` default.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11);
        let result_var = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[1].clone(), result_var.clone());
        let size = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[2].clone(), size.clone());
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__majit_stringbuilder_new".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[2].clone()]),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("__majit_stringbuilder_new marker must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "newstringbuilder");
        assert_eq!(translated[0].args.len(), 1);
        assert_eq!(translated[0].args[0], size);
        assert_eq!(translated[0].result, result_var);
    }

    #[test]
    fn translate_op_stringbuilder_append_marker_chains_getattr_simple_call() {
        // `__majit_stringbuilder_append(recv, piece)` → `getattr(recv,
        // "append") + simple_call(bound, piece)`, the same method shape as
        // `Vec::push`; reaches `StringBuilderRepr::rtype_method("append")`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11);
        let recv = Hlvalue::Variable(Variable::new());
        let piece = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[1].clone(), recv.clone());
        value_map.insert(vars[2].clone(), piece.clone());
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![crate::runtime_names::shims::STRINGBUILDER_APPEND.into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone(), vars[2].clone()]),
                result_ty: ValueType::Void,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("__majit_stringbuilder_append marker must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getattr");
        assert_eq!(translated[0].args.len(), 2);
        assert_eq!(translated[0].args[0], recv);
        let Hlvalue::Constant(ref name_const) = translated[0].args[1] else {
            panic!("append getattr name must be a Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"append"
        ));
        let bound = &translated[0].result;
        assert_eq!(translated[1].opname, "simple_call");
        assert_eq!(translated[1].args.len(), 2);
        assert_eq!(&translated[1].args[0], bound);
        assert_eq!(translated[1].args[1], piece);
    }

    #[test]
    fn translate_op_stringbuilder_build_marker_chains_getattr_simple_call() {
        // `__majit_stringbuilder_build(recv)` → `getattr(recv, "build") +
        // simple_call(bound)`, the same shape as `slice.reverse` (no extra
        // arg); reaches `StringBuilderRepr::rtype_method("build")` →
        // `SomeString`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11);
        let recv = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        value_map.insert(vars[1].clone(), recv.clone());
        value_map.insert(vars[2].clone(), result_var.clone());
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![crate::runtime_names::shims::STRINGBUILDER_BUILD.into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone()]),
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("__majit_stringbuilder_build marker must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getattr");
        assert_eq!(translated[0].args[0], recv);
        let Hlvalue::Constant(ref name_const) = translated[0].args[1] else {
            panic!("build getattr name must be a Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"build"
        ));
        let bound = &translated[0].result;
        assert_eq!(translated[1].opname, "simple_call");
        assert_eq!(translated[1].args.len(), 1);
        assert_eq!(&translated[1].args[0], bound);
        assert_eq!(translated[1].result, result_var);
    }

    #[test]
    fn translate_op_isinstance_lowers_to_flowspace_isinstance() {
        // OpKind::IsInstance arrives from the tuple-struct match
        // cascade (`front/ast.rs`).  Emit a single
        // `isinstance(obj, cls)` flowspace op so the rtyper dispatches
        // to `InstanceRepr::rtype_isinstance`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let obj_hl_var = Variable::new();
        let cls_hl_var = Variable::new();
        let result_hl_var = Variable::new();
        value_map.insert(vars[1].clone(), Hlvalue::Variable(obj_hl_var.clone()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(cls_hl_var.clone()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(result_hl_var.clone()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::IsInstance {
                obj: vars[1].clone(),
                class_carrier: vars[2].clone(),
                result_ty: ValueType::Bool,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("OpKind::IsInstance must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(
            translated[0].opname, "isinstance",
            "IsInstance must emit the flowspace `isinstance` opname",
        );
        assert_eq!(translated[0].args.len(), 2, "isinstance args: [obj, cls]");
        match &translated[0].args[0] {
            Hlvalue::Variable(v) => assert_eq!(v, &obj_hl_var, "args[0] must be obj"),
            other => panic!("args[0] must be Variable, got {other:?}"),
        }
        match &translated[0].args[1] {
            Hlvalue::Variable(v) => {
                assert_eq!(v, &cls_hl_var, "args[1] must be class_carrier")
            }
            other => panic!("args[1] must be Variable, got {other:?}"),
        }
        match &translated[0].result {
            Hlvalue::Variable(v) => {
                assert_eq!(v, &result_hl_var, "result must follow value_map mapping")
            }
            other => panic!("result must be Variable, got {other:?}"),
        }
    }

    #[test]
    fn class_helper_calls_use_registered_bodies_not_high_level_rewrites() {
        use crate::codewriter::call::GraphStore;
        use crate::parse::CallPath;
        use crate::translator::rtyper::call_registry::FunctionPathKey;
        use crate::translator::rtyper::cutover::populate_call_registry_from_call_graphs;

        for (name, arity) in [
            ("ll_issubclass", 2),
            ("ll_isinstance", 2),
            ("ll_issubclass_const", 3),
        ] {
            let segments = vec![
                crate::runtime_names::crates::OBJECT.to_string(),
                "pyobject".to_string(),
                name.to_string(),
            ];
            let mut caller = LegacyGraph::new("caller");
            let vars = mint_vars(&mut caller, arity + 1);
            let value_map: HashMap<Variable, Hlvalue> = vars
                .iter()
                .map(|v| (v.clone(), Hlvalue::Variable(Variable::new())))
                .collect();
            let op = SpaceOperation {
                result: Some(vars[arity].clone()),
                kind: OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: segments.clone(),
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vars[..arity].iter().cloned()),
                    result_ty: ValueType::Bool,
                },
            };

            // With no body, fail as any unregistered call does: never turn a
            // helper name into isinstance/issubtype to conceal missing input.
            let missing = translate_op(&op, &value_map, &empty_call_registry())
                .expect_err("an absent class helper must not become a high-level operation");
            assert!(
                missing
                    .to_string()
                    .contains("not registered in CallRegistry"),
                "{missing}"
            );

            // Deliberately simple fixture body: this test pins registration
            // and exact callable identity, not subclass arithmetic. Use the
            // PRODUCTION population path so restoring its name-based skip
            // would fail even if manually registered calls still worked.
            let mut body = LegacyGraph::new(name);
            let inputs = mint_vars(&mut body, arity);
            body.block_mut(body.startblock).inputargs = inputs.clone();
            body.set_return(body.startblock, Some(inputs[0].clone()));
            let mut graphs = GraphStore::default();
            graphs.insert(CallPath::from_segments(segments.clone()), body);
            let registry = std::rc::Rc::new(empty_call_registry());
            populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry)
                .expect("register ordinary helper body");
            let entry = registry
                .lookup(&FunctionPathKey::from_segments(segments))
                .expect("helper must not be excluded by name");
            assert!(entry.lift_error().is_none(), "{:?}", entry.lift_error());
            let translated =
                translate_op(&op, &value_map, &registry).expect("registered class helper call");
            assert_eq!(translated.len(), 1);
            assert_eq!(translated[0].opname, "simple_call");
            assert_eq!(translated[0].args.len(), arity + 1);
            let Hlvalue::Constant(callable) = &translated[0].args[0] else {
                panic!("registered callable constant required");
            };
            assert_eq!(
                callable.value,
                ConstValue::HostObject(entry.host_object.clone())
            );
            for (actual, source) in translated[0].args[1..].iter().zip(&vars[..arity]) {
                assert_eq!(actual, &value_map[source]);
            }
            assert_eq!(translated[0].result, value_map[&vars[arity]]);
        }
    }

    /// Layer-1 `call_registry.lookup` beats HOST_ENV. A registered
    /// residual body is the callable; the `majit_rlib.jit.conditional_call*`
    /// builtin is not consulted. `registration_decline` exists so populate
    /// never leaves that body in the registry.
    #[test]
    fn conditional_call_registered_body_beats_host_env() {
        use crate::codewriter::call::GraphStore;
        use crate::parse::CallPath;
        use crate::translator::rtyper::call_registry::FunctionPathKey;
        use crate::translator::rtyper::cutover::populate_call_registry_from_call_graphs;

        let segments = vec![
            "majit_rlib".to_string(),
            "jit".to_string(),
            "conditional_call_elidable1".to_string(),
        ];
        let mut caller = LegacyGraph::new("caller");
        let vars = mint_vars(&mut caller, 4);
        let value_map: HashMap<Variable, Hlvalue> = vars
            .iter()
            .map(|v| (v.clone(), Hlvalue::Variable(Variable::new())))
            .collect();
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: segments.clone(),
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vars[..3].iter().cloned()),
                result_ty: ValueType::Ref(None),
            },
        };

        let missing = translate_op(&op, &value_map, &empty_call_registry())
            .expect("HOST_ENV Layer-3b resolves the helper when the registry is empty");
        assert_eq!(missing[0].opname, "simple_call");
        let Hlvalue::Constant(host_callable) = &missing[0].args[0] else {
            panic!("HOST_ENV callable constant required");
        };
        let host = crate::flowspace::model::host_env_callable(&segments)
            .expect("majit_rlib.jit.conditional_call_elidable1 is a HOST_ENV builtin");
        assert_eq!(host_callable.value, ConstValue::HostObject(host.clone()));

        // An LLBC body in GraphStore is declined, so populate still
        // leaves Layer-3b as the resolver.
        let mut body = LegacyGraph::new("conditional_call_elidable1");
        let inputs = mint_vars(&mut body, 3);
        body.block_mut(body.startblock).inputargs = inputs.clone();
        body.set_return(body.startblock, Some(inputs[0].clone()));
        let mut graphs = GraphStore::default();
        graphs.insert(CallPath::from_segments(segments.clone()), body);
        let registry = std::rc::Rc::new(empty_call_registry());
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry)
            .expect("populate");
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments(segments.clone()))
                .is_none(),
            "populate must decline the residual body so HOST_ENV wins"
        );
        let translated =
            translate_op(&op, &value_map, &registry).expect("declined helper uses HOST_ENV");
        let Hlvalue::Constant(callable) = &translated[0].args[0] else {
            panic!("callable constant required");
        };
        assert_eq!(callable.value, ConstValue::HostObject(host));
    }

    #[test]
    fn translate_op_call_method_chains_getattr_simple_call() {
        // Call::Method `obj.method(args)` → 2-op chain `[getattr(obj,
        // "method") -> meth, simple_call(meth, args[1..])]`, mirroring
        // `flowspace/flowcontext.py: LOAD_ATTR + CALL_FUNCTION` shape.
        // args[0] is the receiver (matches Rust method-call lowering).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new())); // receiver
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new())); // arg
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new())); // result
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::method("push", Some("Vec".into())),
                args: crate::model::call_args(vec![vars[1].clone(), vars[2].clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("Call::Method must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getattr");
        assert_eq!(translated[1].opname, "simple_call");
        // getattr's name arg is the method name as a byte string.
        let Hlvalue::Constant(ref name_const) = translated[0].args[1] else {
            panic!("getattr method-name arg must be a Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"push"
        ));
        // Bound-method Variable identity threads from getattr.result into
        // simple_call.args[0].
        let Hlvalue::Variable(ref m1) = translated[0].result else {
            panic!("getattr result must be Variable");
        };
        let Hlvalue::Variable(ref m2) = translated[1].args[0] else {
            panic!("simple_call's first arg must be Variable (bound method)");
        };
        assert_eq!(
            m1.id(),
            m2.id(),
            "bound method Variable identity must thread"
        );
        // simple_call args = [bound_method, args[1..]]
        assert_eq!(translated[1].args.len(), 2);
    }

    #[test]
    fn translate_op_call_indirect_chains_getattr_simple_call() {
        // Call::Indirect (`dyn Trait` virtual call) lowers to the same
        // `[getattr(receiver, method_name) -> meth, simple_call(meth,
        // args[1..])]` chain as `CallTarget::Method`.  The receiver
        // `args[0]` is seeded (by `derive_subject_inputcells`) as the
        // trait-family base ClassDef, so the method getattr resolves the
        // impl `MethodDesc` family in the two-phase prepass instead of
        // failing loud.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new())); // receiver
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new())); // arg
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new())); // result
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::Indirect {
                    trait_root: "MyTrait".into(),
                    method_name: "do_it".into(),
                },
                args: crate::model::call_args(vec![vars[1].clone(), vars[2].clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Call::Indirect must lower to getattr + simple_call");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getattr");
        assert_eq!(translated[1].opname, "simple_call");
        // getattr's name arg is the method name as a byte string.
        let Hlvalue::Constant(ref name_const) = translated[0].args[1] else {
            panic!("getattr method-name arg must be a Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"do_it"
        ));
        // Bound-method Variable identity threads from getattr.result into
        // simple_call.args[0].
        let Hlvalue::Variable(ref m1) = translated[0].result else {
            panic!("getattr result must be Variable");
        };
        let Hlvalue::Variable(ref m2) = translated[1].args[0] else {
            panic!("simple_call's first arg must be Variable (bound method)");
        };
        assert_eq!(
            m1.id(),
            m2.id(),
            "bound method Variable identity must thread"
        );
        // simple_call args = [bound_method, args[1..]]
        assert_eq!(translated[1].args.len(), 2);
    }

    #[test]
    fn translate_op_indirect_call_without_family_key_fails_loud() {
        // An IndirectCall carrying no `family_key` is a stored function
        // pointer: it names no trait receiver and no method, so there is
        // nothing for a `getattr` to spell, and a function pointer has no
        // pre-rtyper shape at all — upstream only ever mints `indirect_call`
        // inside `rtype_simple_call` and `FunctionReprBase.call` (rpbc.py),
        // after rtyping. Fail-loud is the
        // parity-correct behaviour for that arm.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::IndirectCall {
                funcptr: vars[1].clone(),
                args: vec![vars[2].clone()],
                graphs: None,
                family_key: None,
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("a family-key-less IndirectCall must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("IndirectCall") && msg.contains("family_key"),
            "fail-loud must name IndirectCall and the missing family_key, got: {msg}"
        );
    }

    /// An IndirectCall that does carry a `family_key` names a dyn-trait vtable
    /// slot, and lowers to the same pre-rtyper shape as its `CallTarget::
    /// Indirect` twin: `getattr(receiver, method)` then `simple_call`. The
    /// annotator never sees an `indirect_call` — `rpbc.py` mints that
    /// inside `rtype_simple_call`, downstream of here.
    #[test]
    fn translate_op_indirect_call_with_family_key_lowers_to_getattr_simple_call() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11);
        for v in &vars[1..4] {
            value_map.insert(v.clone(), Hlvalue::Variable(Variable::new()));
        }
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::IndirectCall {
                funcptr: vars[1].clone(),
                args: vec![vars[2].clone()],
                graphs: None,
                family_key: Some(("Handler".to_string(), "run".to_string())),
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("a family-keyed IndirectCall lowers to the dyn-dispatch shape");
        let names: Vec<&str> = translated.iter().map(|o| o.opname.as_str()).collect();
        assert_eq!(names, vec!["getattr", "simple_call"]);
        let Hlvalue::Constant(ref name_const) = translated[0].args[1] else {
            panic!("getattr's second arg must be the method-name Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"run"
        ));
        // The bound method threads from getattr.result into simple_call.args[0],
        // and args[0] of the op — the receiver — is consumed by the getattr.
        let Hlvalue::Variable(ref m1) = translated[0].result else {
            panic!("getattr result must be Variable");
        };
        let Hlvalue::Variable(ref m2) = translated[1].args[0] else {
            panic!("simple_call's first arg must be the bound method Variable");
        };
        assert_eq!(m1.id(), m2.id());
        assert_eq!(translated[1].args.len(), 1, "the receiver is not re-passed");
    }

    #[test]
    fn translate_op_vec_len_word_lowers_to_len() {
        // `VecFieldPart::Len` is `rtype_len`: the result is the count, not
        // the list. An inline field is `len(getattr(owner, field))`; a
        // pointer to the Vec is `len(vec)`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 4);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let inline = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::FieldRead {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new("entries", Some("RDict".into()))
                    .with_inline_vec(true)
                    .with_vec_part(crate::model::VecFieldPart::Len),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let translated =
            translate_op(&inline, &value_map, &empty_call_registry()).expect("inline Vec len read");
        let names: Vec<&str> = translated.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(names, vec!["getattr", "len"]);
        let Hlvalue::Variable(ref list_v) = translated[0].result else {
            panic!("getattr result must be the list");
        };
        let Hlvalue::Variable(ref len_arg) = translated[1].args[0] else {
            panic!("len argument must be the list");
        };
        assert_eq!(list_v.id(), len_arg.id());
        let Hlvalue::Variable(ref len_res) = translated[1].result else {
            panic!("len result must be a variable");
        };
        let Hlvalue::Variable(mapped) = value_map.get(&vars[2]).unwrap() else {
            panic!("field-read result must map to a variable");
        };
        assert_eq!(len_res.id(), mapped.id());

        let word = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::FieldRead {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new("len", Some("alloc::vec::Vec".into()))
                    .with_vec_part(crate::model::VecFieldPart::Len),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let translated =
            translate_op(&word, &value_map, &empty_call_registry()).expect("Vec len word");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "len");
    }

    #[test]
    fn translate_op_fat_len_lowers_to_getattr_len() {
        // FatLen is the inline-vec shape: `len(getattr(owner, field))`.
        // The field-read result is the `len` result, an int. FatData stays
        // `getattr` of the field (the data pointer).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 5);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let len_op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::FieldRead {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new(
                    "varnames",
                    Some("bytecode::CodeObject".into()),
                )
                .with_vec_part(crate::model::VecFieldPart::FatLen),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let translated =
            translate_op(&len_op, &value_map, &empty_call_registry()).expect("fat len");
        let names: Vec<&str> = translated.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(names, vec!["getattr", "len"]);
        let Hlvalue::Constant(ref name_const) = translated[0].args[1] else {
            panic!("getattr name");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"varnames"
        ));
        let Hlvalue::Variable(ref len_res) = translated[1].result else {
            panic!("len result");
        };
        let Hlvalue::Variable(mapped) = value_map.get(&vars[2]).unwrap() else {
            panic!("mapped result");
        };
        assert_eq!(len_res.id(), mapped.id());
        let Hlvalue::Variable(ref getattr_res) = translated[0].result else {
            panic!("getattr result");
        };
        assert_ne!(getattr_res.id(), mapped.id());

        let data_op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::FieldRead {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new(
                    "localspluskinds",
                    Some("bytecode::CodeObject".into()),
                )
                .with_vec_part(crate::model::VecFieldPart::FatData),
                ty: ValueType::Ref(None),
                pure: false,
            },
        };
        let translated =
            translate_op(&data_op, &value_map, &empty_call_registry()).expect("fat data");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "getattr");
        let Hlvalue::Variable(ref data_res) = translated[0].result else {
            panic!("data result");
        };
        let Hlvalue::Variable(mapped) = value_map.get(&vars[3]).unwrap() else {
            panic!("mapped data");
        };
        assert_eq!(data_res.id(), mapped.id());
    }

    #[test]
    fn translate_op_field_read_lowers_to_getattr() {
        // FieldRead → flowspace `getattr(base, ConstValue::ByteStr(name))`
        // mirroring `flowspace/operation.py GetAttr.opname = 'getattr'`.
        // The rtyper later dispatches via `rtype_getattr` based on the
        // base operand's resolved repr (InstanceRepr / etc.).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let base_var = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), base_var.clone());
        value_map.insert(vars[2].clone(), result_var.clone());

        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::FieldRead {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new("f", Some("Owner".into())),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("FieldRead arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "getattr");
        assert_eq!(lowered.args.len(), 2);
        let Hlvalue::Constant(ref name_const) = lowered.args[1] else {
            panic!("FieldRead lowering must pass field name as Hlvalue::Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"f"
        ));
    }

    #[test]
    fn translate_op_inline_vec_len_lowers_to_ll_length() {
        // `self.entries.len()` retargets the field read in place
        // (`retarget_vec_operand`). The result must be `len` of that
        // list — `ListRepr.rtype_len` / `ll_length` → Signed — not
        // `getattr` of the Vec field (the list pointer, GcRef).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let base_var = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11);
        value_map.insert(vars[1].clone(), base_var);
        value_map.insert(vars[2].clone(), result_var.clone());
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::FieldRead {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new("entries", Some("RDict".into()))
                    .with_inline_vec(true)
                    .with_vec_part(crate::model::VecFieldPart::Len),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("inline Vec length field read must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getattr");
        assert_eq!(translated[1].opname, "len");
        let Hlvalue::Variable(ref list) = translated[0].result else {
            panic!("getattr result must be the list");
        };
        let Hlvalue::Variable(ref len_arg) = translated[1].args[0] else {
            panic!("len receiver must be the list");
        };
        assert_eq!(list.id(), len_arg.id());
        let Hlvalue::Variable(ref len_res) = translated[1].result else {
            panic!("len result must be a Variable");
        };
        let Hlvalue::Variable(ref expect) = result_var else {
            panic!("fixture result must be a Variable");
        };
        assert_eq!(len_res.id(), expect.id());
    }

    #[test]
    fn translate_op_str_length_read_lowers_to_len() {
        // `LLHelpers.ll_strlen`: the length word of an rstr `STR` is `len(s)`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 3);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::FieldRead {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new("length", Some("Utf8Str".into())),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("STR length must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "len");
        assert_eq!(translated[0].args.len(), 1);
    }

    #[test]
    fn translate_op_field_write_lowers_to_setattr() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new("g", Some("Owner".into())),
                value: crate::model::LinkArg::Value(vars[2].clone()),
                ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("FieldWrite arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "setattr");
        assert_eq!(lowered.args.len(), 3);
        let Hlvalue::Constant(ref name_const) = lowered.args[1] else {
            panic!("FieldWrite lowering must pass field name as Hlvalue::Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"g"
        ));
    }

    #[test]
    fn translate_op_array_read_lowers_to_getitem() {
        // ArrayRead → flowspace `getitem(base, index)` mirroring
        // `flowspace/operation.py: GetItem.opname = 'getitem'`. RTyper's
        // `rtype_getitem` dispatches via the receiver's resolved repr
        // (ListRepr / TupleRepr / FixedSizeArrayRepr) and lowers to
        // `getarrayitem_gc_*`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::ArrayRead {
                base: vars[1].clone(),
                index: vars[2].clone(),
                item_ty: ValueType::Int,
                array_type_id: None,
                nolength: false,
                pure: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ArrayRead arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "getitem");
        assert_eq!(lowered.args.len(), 2);
    }

    #[test]
    fn translate_op_array_len_lowers_to_len() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 4);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::ArrayLen {
                base: vars[1].clone(),
                array_type_id: Some("[PyObjectRef]".to_string()),
                nolength: false,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("ArrayLen arm must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "len");
        assert_eq!(translated[0].args.len(), 1);
        assert!(!op_canraise(&op.kind));
    }

    #[test]
    fn translate_op_strlen_lowers_to_len() {
        // `front::mir` rewrites `Rvalue::Len` / `<[u8]>::len` on a string
        // byte-view to `Call(["__strlen"])`.  Real-site receivers are the
        // Wtf8/`Str` dest (`SomeString` / `StringRepr`), so this is the
        // rtyper `len` op — `StringRepr.rtype_len` → `ll_strlen`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 3);
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__strlen".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![vars[1].clone()]),
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("__strlen is a len spelling");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "len");
        assert_eq!(translated[0].args.len(), 1);
        assert!(!op_canraise(&op.kind));
    }

    #[test]
    fn translate_op_array_write_lowers_to_setitem() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::ArrayWrite {
                base: vars[1].clone(),
                index: vars[2].clone(),
                value: crate::model::LinkArg::Value(vars[3].clone()),
                item_ty: ValueType::Int,
                array_type_id: None,
                nolength: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ArrayWrite arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "setitem");
        assert_eq!(lowered.args.len(), 3);
    }

    #[test]
    fn translate_op_interior_field_read_unfolds_to_getitem_getattr_chain() {
        // InteriorFieldRead → `getitem(base, index)` chained into
        // `getattr(elem, field_name)`, mirroring `effectinfo.py add_interiorfield`'s
        // implicit `readarray + readinteriorfield` effects. Two flowspace
        // ops surface from one legacy op; the rtyper sees the chain.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::InteriorFieldRead {
                base: vars[1].clone(),
                index: vars[2].clone(),
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                item_ty: ValueType::Int,
                array_type_id: None,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("InteriorFieldRead arm must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getitem");
        assert_eq!(translated[0].args.len(), 2);
        assert_eq!(translated[1].opname, "getattr");
        assert_eq!(translated[1].args.len(), 2);
        // The second op's first arg must be the element variable
        // produced by the first op.
        let Hlvalue::Variable(ref elem_v1) = translated[0].result else {
            panic!("getitem result must be a Variable");
        };
        let Hlvalue::Variable(ref elem_v2) = translated[1].args[0] else {
            panic!("getattr base arg must be the chained element Variable");
        };
        assert_eq!(
            elem_v1.id(),
            elem_v2.id(),
            "elem Variable identity must thread"
        );
    }

    #[test]
    fn op_canraise_covers_getitem_bearing_and_inplace_binops() {
        // `InteriorField*` unfolds to a getitem (raising) + getattr/setattr,
        // so as a `?` tail it carries the getitem's `[IndexError, KeyError,
        // Exception]` (operation.py:727-730).
        let interior = OpKind::InteriorFieldRead {
            base: Variable::new(),
            index: Variable::new(),
            field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
            item_ty: ValueType::Int,
            array_type_id: None,
        };
        assert!(
            op_canraise(&interior),
            "InteriorFieldRead carries the unfolded getitem's canraise"
        );

        // Compound-assign names reach `op_canraise` BEFORE
        // `normalize_binop_name` maps them to `inplace_*`.
        // `inplace_div`/`inplace_add`/`inplace_lshift` carry
        // ZeroDivisionError/OverflowError/ValueError; `inplace_and` and
        // plain `add` are `[]` (operation.py:751-756).
        let binop = |name: &str| OpKind::BinOp {
            op: name.to_string(),
            lhs: Variable::new(),
            rhs: Variable::new(),
            result_ty: ValueType::Int,
        };
        assert!(op_canraise(&binop("div_assign")));
        assert!(op_canraise(&binop("add_assign")));
        assert!(op_canraise(&binop("lshift_assign")));
        assert!(op_canraise(&binop("div")));
        assert!(!op_canraise(&binop("bitand_assign")));
        assert!(!op_canraise(&binop("add")));

        // `core::*` method bridges that `translate_op` lowers to pure
        // flowspace ops must classify as non-raising, so a `?` tail does
        // not install an unreachable exception edge.  Drift here is what
        // `nonraising_core_bridge_opname` exists to prevent.
        let core_call = |segs: &[&str], argc: usize| OpKind::Call {
            target: crate::model::CallTarget::FunctionPath {
                segments: segs.iter().map(|s| s.to_string()).collect(),
                fun_decl_id: None,
            },
            args: crate::model::call_args((0..argc).map(|_| Variable::new())),
            result_ty: ValueType::Int,
        };
        assert!(!op_canraise(&core_call(
            &["core", "cmp", "PartialEq", "eq"],
            2
        )));
        assert!(!op_canraise(&core_call(
            &["core", "cmp", "PartialOrd", "lt"],
            2
        )));
        assert!(!op_canraise(&core_call(&["core", "slice", "len"], 1)));
        assert!(!op_canraise(&core_call(&["core", "slice", "iter"], 1)));
        assert!(!op_canraise(&core_call(
            &["core", "num", "wrapping_mul"],
            2
        )));
        // `min`/`max` lower to a raising `simple_call`, and a wrong arg
        // count falls through to the general raising `Call` arm.
        assert!(op_canraise(&core_call(&["core", "cmp", "min"], 2)));
        assert!(op_canraise(&core_call(&["core", "cmp", "max"], 2)));
        assert!(op_canraise(&core_call(&["core", "slice", "len"], 2)));
    }

    #[test]
    fn translate_op_interior_field_write_unfolds_to_getitem_setattr_chain() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::InteriorFieldWrite {
                base: vars[1].clone(),
                index: vars[2].clone(),
                field: crate::model::FieldDescriptor::new("y", Some("Point".into())),
                value: vars[3].clone(),
                item_ty: ValueType::Int,
                array_type_id: None,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("InteriorFieldWrite arm must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getitem");
        assert_eq!(translated[1].opname, "setattr");
        assert_eq!(translated[1].args.len(), 3);
    }

    #[test]
    fn translate_op_undefined_operand_surfaces_invariant_break() {
        // The lookup_operand helper is shared across every arm with
        // operands. Validate it surfaces a clear "adapter
        // invariant broken" message and embeds the enriched diagnostic
        // context (op variant + arg role).
        let value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 101); // vars[0..101]
        let op = SpaceOperation {
            result: Some(vars[100].clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: vars[99].clone(),
                rhs: vars[0].clone(),
                result_ty: ValueType::Int,
            },
        };
        let err = lookup_operand(&value_map, &vars[99], &op, "lhs")
            .expect_err("undefined operand lookup must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("undefined operand") && msg.contains("invariant"),
            "fail-loud message must explain the invariant, got: {msg}"
        );
        assert!(
            msg.contains("as lhs of BinOp") && msg.contains("result Some(Variable("),
            "verbose diagnostic must include arg role + op variant + result variable, got: {msg}"
        );
    }

    #[test]
    fn translate_unsigned_binop_carries_rust_type_through_r_uint() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_unsigned_binop");
        let vars = mint_vars(&mut graph, 3);
        let lhs = Variable::new();
        let rhs = Variable::new();
        let result = Variable::new();
        value_map.insert(vars[0].clone(), Hlvalue::Variable(lhs));
        value_map.insert(vars[1].clone(), Hlvalue::Variable(rhs));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(result.clone()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::BinOp {
                op: "sub".to_string(),
                lhs: vars[0].clone(),
                rhs: vars[1].clone(),
                result_ty: ValueType::Unsigned,
            },
        };

        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("unsigned BinOp must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "sub");
        assert_eq!(translated[1].opname, "simple_call");
        let Hlvalue::Variable(ref scalar_result) = translated[0].result else {
            panic!("sub result must be a temporary Variable");
        };
        let Hlvalue::Variable(ref cast_arg) = translated[1].args[1] else {
            panic!("r_uint operand must be the sub result Variable");
        };
        assert_eq!(scalar_result.id(), cast_arg.id());
        let Hlvalue::Variable(ref cast_result) = translated[1].result else {
            panic!("r_uint result must be the model result Variable");
        };
        assert_eq!(cast_result.id(), result.id());
        let Hlvalue::Constant(callable) = &translated[1].args[0] else {
            panic!("r_uint callable must be a Constant");
        };
        let ConstValue::HostObject(host) = &callable.value else {
            panic!("r_uint callable must be a HostObject");
        };
        let expected = HOST_ENV
            .import_module("rpython.rlib.rarithmetic")
            .expect("HOST_ENV bootstrap must register rpython.rlib.rarithmetic")
            .module_get("r_uint")
            .expect("rarithmetic module must expose r_uint");
        assert_eq!(
            host, &expected,
            "callable must be the registered r_uint object"
        );
    }

    // ───── topology assembly ─────

    fn link_to_returnblock(args: Vec<LinkArg>, returnblock_id: BlockId) -> crate::model::Link {
        let mut link = crate::model::Link::new_mixed(args, returnblock_id, None);
        link.prevblock = None;
        link
    }

    fn legacy_minimal_identity_return_graph()
    -> (LegacyGraph, Vec<crate::flowspace::model::Variable>) {
        // Smallest valid legacy graph: one inputarg, returns it
        // directly. Must produce a flowspace::FunctionGraph
        // whose startblock has the single inputarg Variable,
        // exits→returnblock, and the returnblock's inputarg is the same
        // Variable identity (so RPythonTyper.getreturnvar resolves
        // correctly).
        //
        // RPython convention: returnblock canonically has one inputarg
        // (`flowspace/model.py`). True void returns use a
        // `SomeNone` / `Void`-typed argument; pyre's legacy graph
        // mirrors that by always emitting a single slot in the
        // returnblock's inputargs.
        let mut graph = LegacyGraph::new("identity_return");
        let vars = mint_vars(&mut graph, 2); // vars[0..2]
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[1].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];
        (graph, vars)
    }

    #[test]
    fn function_graph_to_flowspace_minimal_identity_return_assembles_graph() {
        let (legacy, vars) = legacy_minimal_identity_return_graph();

        let output = function_graph_to_flowspace(&legacy, &empty_call_registry())
            .expect("minimal graph must assemble");

        // value_to_var must contain the inputarg.
        assert!(
            output.value_to_var.contains_key(&vars[1]),
            "value_to_var must seed the legacy inputarg"
        );

        // block_map must contain startblock + returnblock + exceptblock.
        assert!(output.block_map.contains_key(&legacy.startblock));
        assert!(output.block_map.contains_key(&legacy.returnblock));
        assert!(output.block_map.contains_key(&legacy.exceptblock));

        // The flowspace graph's startblock is the same BlockRef as the
        // mapped legacy.startblock.
        let graph = output.graph.borrow();
        assert!(Rc::ptr_eq(
            &graph.startblock,
            &output.block_map[&legacy.startblock],
        ));

        // The flowspace graph's returnblock is the mapped legacy.returnblock.
        assert!(Rc::ptr_eq(
            &graph.returnblock,
            &output.block_map[&legacy.returnblock],
        ));

        // The startblock has exactly one exit, targeting the returnblock.
        let startblock = graph.startblock.borrow();
        assert_eq!(startblock.exits.len(), 1);
        let exit = startblock.exits[0].borrow();
        assert!(Rc::ptr_eq(
            exit.target.as_ref().expect("link must have target"),
            &graph.returnblock,
        ));
    }

    #[test]
    fn function_graph_to_flowspace_rebuilds_array_aggregate_as_rpython_list() {
        let mut graph = LegacyGraph::new("array_to_list");
        let vars = mint_vars(&mut graph, 6);
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![
                SpaceOperation {
                    result: Some(vars[2].clone()),
                    kind: OpKind::ConstInt(10),
                },
                SpaceOperation {
                    result: Some(vars[3].clone()),
                    kind: OpKind::ConstInt(20),
                },
                SpaceOperation {
                    result: Some(vars[4].clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::synthetic_transparent_ctor("Array"),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("Array".to_string())),
                    },
                },
                SpaceOperation {
                    result: None,
                    kind: OpKind::FieldWrite {
                        base: vars[4].clone(),
                        field: crate::model::FieldDescriptor::new(
                            "__pos_0",
                            Some("Array".to_string()),
                        ),
                        value: LinkArg::Value(vars[2].clone()),
                        ty: ValueType::Int,
                    },
                },
                SpaceOperation {
                    result: None,
                    kind: OpKind::FieldWrite {
                        base: vars[4].clone(),
                        field: crate::model::FieldDescriptor::new(
                            "__pos_1",
                            Some("Array".to_string()),
                        ),
                        value: LinkArg::Value(vars[3].clone()),
                        ty: ValueType::Int,
                    },
                },
            ],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[4].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[5]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("array aggregate must adapt");
        let graph = output.graph.borrow();
        let startblock = graph.startblock.borrow();
        assert_eq!(startblock.operations.len(), 1);
        let op = &startblock.operations[0];
        assert_eq!(op.opname, "newlist");
        assert_eq!(op.args.len(), 2);
    }

    /// A shaped tuple ctor whose item is computed after the ctor becomes one
    /// `newtuple` at its last item write, and a `__pos_N` read becomes
    /// `getitem(t, N)`.
    #[test]
    fn function_graph_to_flowspace_rebuilds_tuple_aggregate_as_rpython_tuple() {
        let mut graph = LegacyGraph::new("tuple_to_newtuple");
        let vars = mint_vars(&mut graph, 7);
        let owner = "Tuple<i64,i64>".to_string();
        let pos = |index: usize| {
            crate::model::FieldDescriptor::new(&format!("__pos_{index}"), Some(owner.clone()))
        };
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![
                SpaceOperation {
                    result: Some(vars[2].clone()),
                    kind: OpKind::ConstInt(10),
                },
                SpaceOperation {
                    result: Some(vars[4].clone()),
                    kind: OpKind::Call {
                        target: crate::model::CallTarget::synthetic_transparent_ctor(&owner),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some(owner.clone())),
                    },
                },
                SpaceOperation {
                    result: None,
                    kind: OpKind::FieldWrite {
                        base: vars[4].clone(),
                        field: pos(0),
                        value: LinkArg::Value(vars[2].clone()),
                        ty: ValueType::Int,
                    },
                },
                SpaceOperation {
                    result: Some(vars[3].clone()),
                    kind: OpKind::ConstInt(20),
                },
                SpaceOperation {
                    result: None,
                    kind: OpKind::FieldWrite {
                        base: vars[4].clone(),
                        field: pos(1),
                        value: LinkArg::Value(vars[3].clone()),
                        ty: ValueType::Int,
                    },
                },
                SpaceOperation {
                    result: Some(vars[5].clone()),
                    kind: OpKind::FieldRead {
                        base: vars[4].clone(),
                        field: pos(1),
                        ty: ValueType::Int,
                        pure: false,
                    },
                },
            ],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[5].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[6]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("tuple aggregate must adapt");
        let graph = output.graph.borrow();
        let startblock = graph.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["newtuple", "getitem"]);
        assert_eq!(startblock.operations[0].args.len(), 2);
        assert!(matches!(
            &startblock.operations[1].args[1],
            Hlvalue::Constant(c) if c.value == ConstValue::Int(1)
        ));
    }

    #[test]
    fn function_graph_to_flowspace_returnvar_identity_preserved() {
        // When the returnblock has an inputarg slot, the flowspace
        // graph's returnblock must use the SAME Variable identity (so
        // RPythonTyper.getreturnvar finds the right Variable —
        // rtyper.rs).
        let mut graph = LegacyGraph::new("with_return_var");
        let vars = mint_vars(&mut graph, 3); // vars[0..3]
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[1].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("graph must assemble");

        // The flowspace returnblock's single inputarg must be the
        // Variable we seeded for slot 2.
        let flowspace_graph = output.graph.borrow();
        let returnblock_ref = &flowspace_graph.returnblock;
        let returnblock = returnblock_ref.borrow();
        assert_eq!(
            returnblock.inputargs.len(),
            1,
            "returnblock must carry the single return-value inputarg"
        );
        match &returnblock.inputargs[0] {
            Hlvalue::Variable(v) => {
                let expected = &output.value_to_var[&vars[2]];
                assert_eq!(
                    v, expected,
                    "returnblock inputarg must preserve Variable identity from value_to_var"
                );
            }
            other => panic!("returnblock inputarg must be a Variable, got {other:?}"),
        }
    }

    #[test]
    fn function_graph_to_flowspace_const_define_op_inlined_in_link_args() {
        // ConstInt(7) defines vars[2]. build_value_to_hlvalue_map
        // inlines it into Link.args as Hlvalue::Constant — link
        // translation must use that
        // mapping rather than wrapping the unused Variable.
        let mut graph = LegacyGraph::new("const_link_arg");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![SpaceOperation {
                result: Some(vars[2].clone()),
                kind: OpKind::ConstInt(7),
            }],
            exitswitch: None,
            // Return vars[2], the ConstInt define.
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[2].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[3]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("graph must assemble");

        let flowspace_graph = output.graph.borrow();
        let startblock = flowspace_graph.startblock.borrow();
        // ConstInt define is a skip arm — operations must be empty.
        assert!(
            startblock.operations.is_empty(),
            "ConstInt define has no flowspace::SpaceOperation analogue"
        );
        // The exit's link arg must be the inlined Constant.
        let exit = startblock.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        match exit.args[0].as_ref().expect("link arg is Some") {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(n) => assert_eq!(*n, 7),
                other => panic!("link arg constant must be Int, got {other:?}"),
            },
            other => panic!("link arg must be Hlvalue::Constant, got {other:?}"),
        }
    }

    #[test]
    fn function_graph_to_flowspace_exception_link_materialises_extravars_before_args() {
        // RPython checkgraph defines exception-link extravars before
        // validating link.args. Pyre's legacy graph represents those
        // as fresh slots whose only definition site is the link.
        let mut graph = LegacyGraph::new("canraise_with_extravars");
        let vars = mint_vars(&mut graph, 12); // vars[0..12], extravars live at 10/11
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1, 2]),
            operations: vec![SpaceOperation {
                result: Some(vars[3].clone()),
                kind: OpKind::BinOp {
                    op: "add".to_string(),
                    lhs: vars[1].clone(),
                    rhs: vars[2].clone(),
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: Some(crate::model::ExitSwitch::LastException),
            exits: vec![
                link_to_returnblock(vec![LinkArg::Value(vars[3].clone())], graph.returnblock),
                crate::model::Link::new_mixed(
                    vec![
                        LinkArg::Value(vars[10].clone()),
                        LinkArg::Value(vars[11].clone()),
                    ],
                    graph.exceptblock,
                    Some(crate::model::exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(vars[10].clone())),
                    Some(LinkArg::Value(vars[11].clone())),
                ),
            ],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[4]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        let exceptblock = Block {
            id: graph.exceptblock,
            inputargs: block_inputargs(&vars, &[10, 11]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock, exceptblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("exception graph assembles");
        let flowspace_graph = output.graph.borrow();
        let startblock = flowspace_graph.startblock.borrow();
        let exc_link = startblock.exits[1].borrow();
        assert!(exc_link.last_exception.is_some());
        assert!(exc_link.last_exc_value.is_some());
        assert_eq!(
            exc_link.args[0].as_ref(),
            exc_link.last_exception.as_ref(),
            "exception type arg must reuse link.last_exception Variable"
        );
        assert_eq!(
            exc_link.args[1].as_ref(),
            exc_link.last_exc_value.as_ref(),
            "exception value arg must reuse link.last_exc_value Variable"
        );
        drop(exc_link);
        drop(startblock);
        drop(flowspace_graph);

        // SSA_to_SSI deliberately ignores the startblock when collecting
        // cross-block uses. Put the raising operation in a successor so this
        // test actually reaches its exception-extravar handling.
        let entry_args = vec![
            Hlvalue::Variable(Variable::new()),
            Hlvalue::Variable(Variable::new()),
        ];
        let entry = flowspace_model::Block::shared(entry_args.clone());
        let old_start = output.graph.borrow().startblock.clone();
        let mut link = FlowspaceLink::new(entry_args, Some(old_start), None);
        link.prevblock = Some(Rc::downgrade(&entry));
        entry.borrow_mut().closeblock(vec![link.into_ref()]);
        output.graph.borrow_mut().startblock = entry;
        let graph = output.graph.borrow();
        flowspace_model::checkgraph(&graph);
        crate::translator::backendopt::ssa::ssa_to_ssi(&graph, None);
        flowspace_model::checkgraph(&graph);
    }

    #[test]
    fn function_graph_to_flowspace_restores_type_op_for_duplicated_direct_raise() {
        // `front::exc_from_raise` / `front::result_exc` historically encode a
        // direct raise as `(evalue, evalue)` because the final `raise`
        // bytecode reads only slot 1.  RPython's rtyper sees this graph before
        // flattening and requires slot 0 to be the result of `type(evalue)`.
        let mut graph = LegacyGraph::new("direct_raise");
        let vars = mint_vars(&mut graph, 5);
        setbinding(&vars[1], ValueType::Ref(None));

        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![
                    LinkArg::Value(vars[1].clone()),
                    LinkArg::Value(vars[1].clone()),
                ],
                graph.exceptblock,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        let exceptblock = Block {
            id: graph.exceptblock,
            inputargs: block_inputargs(&vars, &[3, 4]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock, exceptblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("direct raise graph assembles");
        let flowspace_graph = output.graph.borrow();
        let startblock = flowspace_graph.startblock.borrow();
        let [type_op] = startblock.operations.as_slice() else {
            panic!("direct raise must gain exactly one type operation");
        };
        assert_eq!(type_op.opname, "type");
        let exc_link = startblock.exits[0].borrow();
        assert_eq!(type_op.args, vec![exc_link.args[1].clone().unwrap()]);
        assert_eq!(exc_link.args[0].as_ref(), Some(&type_op.result));
        assert_ne!(exc_link.args[0], exc_link.args[1]);
    }

    #[test]
    fn function_graph_to_flowspace_unported_opkind_surfaces_failloud() {
        // A graph carrying a still-fail-loud OpKind
        // (Call::UnsupportedExpr — a frontend coverage gap that
        // `front::mir` must classify before the rtyper sees it) must
        // surface that op's translate_op error from inside Pass 2, not
        // silently emit a partial graph.
        let mut graph = LegacyGraph::new("unported_op");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let inputargs = block_inputargs(&vars, &[1]);
        let arg_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![SpaceOperation {
                result: Some(vars[2].clone()),
                kind: OpKind::Call {
                    target: crate::model::CallTarget::UnsupportedExpr,
                    args: crate::model::call_args(vec![arg_var]),
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[2].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[3]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let err = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect_err("unported OpKind must surface as TyperError");
        let msg = format!("{err}");
        assert!(
            msg.contains("UnsupportedExpr") && msg.contains("front::mir"),
            "unported-op error must propagate translate_op's fail-loud message, got: {msg}"
        );
    }

    #[test]
    fn function_graph_to_flowspace_skips_unreachable_raise_block() {
        // A predecessor-less unwind block carrying `set_raise`'s orphan
        // `[etype, evalue]` Link.args — the MIR lowering shape left
        // behind when every inbound `on_unwind` edge is dropped — must
        // not reject the graph: only the reachable closure is
        // converted, mirroring upstream `iterblocks` (a flowspace
        // graph cannot contain unreachable blocks at all).
        let (mut legacy, _vars) = legacy_minimal_identity_return_graph();
        let orphan_etype = legacy.alloc_value_var();
        let orphan_evalue = legacy.alloc_value_var();
        let exceptblock = legacy.exceptblock;
        legacy.blocks.push(Block {
            id: BlockId(3),
            inputargs: vec![],
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(orphan_etype), LinkArg::Value(orphan_evalue)],
                exceptblock,
                None,
            )],
            framestate: None,
            dead: false,
        });

        let output = function_graph_to_flowspace(&legacy, &empty_call_registry())
            .expect("unreachable raise block must not reject the graph");
        // start → return only; the orphan block was never converted.
        assert_eq!(
            output.graph.borrow().iterblocks().len(),
            2,
            "converted graph must contain exactly the reachable closure"
        );
    }

    #[test]
    fn function_graph_to_flowspace_rejects_reachable_orphan_raise_args() {
        // A *reachable* block whose exception-link args are defined by
        // no inputarg and no op result must still fail loud — the
        // reachable-closure conversion does not relax SSA-definedness.
        let mut legacy = LegacyGraph::new("reachable_orphan_raise");
        let vars = mint_vars(&mut legacy, 9);
        let orphan_etype = vars[7].clone();
        let orphan_evalue = vars[8].clone();
        let exceptblock = legacy.exceptblock;
        let startblock = Block {
            id: legacy.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(orphan_etype), LinkArg::Value(orphan_evalue)],
                exceptblock,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: legacy.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        legacy.blocks = vec![startblock, returnblock];

        let err = function_graph_to_flowspace(&legacy, &empty_call_registry())
            .expect_err("reachable orphan raise args must stay fail-loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("undefined operand"),
            "reachable orphan must keep the operand-definedness error, got: {msg}"
        );
    }

    #[test]
    fn opkind_variant_name_covers_core_variants() {
        // Sanity: variant_name maps each core OpKind to a stable string.
        // Any new OpKind variant added without a corresponding arm here
        // surfaces as "<unknown OpKind variant>" in fail-loud messages,
        // which prompts the developer to extend this table.
        assert_eq!(opkind_variant_name(&OpKind::ConstInt(0)), "ConstInt");
        assert_eq!(opkind_variant_name(&OpKind::ConstFloat(0)), "ConstFloat");
        assert_eq!(
            opkind_variant_name(&OpKind::Input {
                name: "x".into(),
                ty: ValueType::Int,
                class_root: None,
            }),
            "Input"
        );
        assert_eq!(
            opkind_variant_name(&OpKind::New {
                owner: "PlainObject".into(),
            }),
            "New"
        );
        assert_eq!(
            opkind_variant_name(&OpKind::NewWithVtable {
                owner: "TypedObject".into(),
                vtable: 1,
            }),
            "NewWithVtable"
        );
    }

    #[test]
    fn translate_op_accepts_both_boxing_allocation_variants() {
        let registry = empty_call_registry();
        registry
            .bookkeeper()
            .set_struct_fields(Rc::new(crate::front::StructFieldRegistry {
                fields: HashMap::from([
                    ("PlainObject".to_string(), vec![]),
                    ("TypedObject".to_string(), vec![]),
                ])
                .into(),
                ..crate::front::StructFieldRegistry::default()
            }));
        let mut graph = LegacyGraph::new("allocation_variants_fixture");

        for kind in [
            OpKind::New {
                owner: "PlainObject".into(),
            },
            OpKind::NewWithVtable {
                owner: "TypedObject".into(),
                vtable: 1,
            },
        ] {
            let result = graph.alloc_value_var();
            let value_map = HashMap::from([(result.clone(), Hlvalue::Variable(Variable::new()))]);
            let translated = translate_op(
                &SpaceOperation {
                    result: Some(result),
                    kind,
                },
                &value_map,
                &registry,
            )
            .expect("both allocation variants must reach the rtyper as a typed constructor");
            assert_eq!(translated.len(), 1);
            assert_eq!(translated[0].opname, "simple_call");
            let crate::flowspace::model::Hlvalue::Constant(ref callable) = translated[0].args[0]
            else {
                panic!("allocation callable must be a Constant");
            };
            let crate::flowspace::model::ConstValue::HostObject(ref host) = callable.value else {
                panic!("allocation callable must be a HostObject");
            };
            assert_eq!(
                host.qualname(),
                "rpython.rlib.objectmodel.instantiate",
                "New/NewWithVtable allocate through objectmodel.instantiate"
            );
        }
    }

    /// The prebuilt-constant-array define-op (`front::mir`'s
    /// `__const_int_array` synthetic Call) is recognized, decodes to its
    /// baked integer list, and pre-folds to a `Constant(List)` the same
    /// way `__str_const` folds to `Constant(ByteStr)` — so `translate_op`
    /// emits nothing and the slot carries the prebuilt list constant.
    #[test]
    fn const_int_array_define_folds_to_list_constant() {
        use crate::model::CallTarget;

        let define = OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec![
                    "__const_int_array".to_string(),
                    "0".to_string(),
                    "2".to_string(),
                    "-3".to_string(),
                ],
                fun_decl_id: None,
            },
            args: Vec::new(),
            result_ty: ValueType::Ref(None),
        };
        assert!(is_const_int_array_define(&define));
        assert_eq!(
            const_int_array_items(&define),
            Some(vec![
                ConstValue::Int(0),
                ConstValue::Int(2),
                ConstValue::Int(-3),
            ])
        );

        let op = SpaceOperation {
            result: Some(Variable::new()),
            kind: define,
        };
        // `translate_op` emits no FlowspaceOp — the pre-pass owns the slot.
        assert!(
            translate_op(&op, &HashMap::new(), &empty_call_registry())
                .expect("const-array define translates")
                .is_empty()
        );
        // The pre-pass folds the define to the prebuilt list constant.
        match legacy_const_define_hlvalue(&op, None).expect("const-array define folds") {
            Some(Hlvalue::Constant(c)) => assert_eq!(
                c.value,
                ConstValue::List(vec![
                    ConstValue::Int(0),
                    ConstValue::Int(2),
                    ConstValue::Int(-3),
                ])
            ),
            other => panic!("expected Constant(List), got {other:?}"),
        }

        // A `__str_const` call and an ordinary path are not mistaken for it.
        let str_const = OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__str_const".to_string(), "hi".to_string()],
                fun_decl_id: None,
            },
            args: Vec::new(),
            result_ty: ValueType::Ref(None),
        };
        assert!(!is_const_int_array_define(&str_const));
    }

    #[test]
    fn const_str_array_define_folds_to_list_constant() {
        use crate::model::CallTarget;

        let define = OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec![
                    "__const_str_array".to_string(),
                    "re.TEMPLATE".to_string(),
                    "re.IGNORECASE".to_string(),
                ],
                fun_decl_id: None,
            },
            args: Vec::new(),
            result_ty: ValueType::Ref(None),
        };
        assert!(is_const_str_array_define(&define));
        assert!(is_const_prebuilt_array_define(&define));
        assert!(!is_const_int_array_define(&define));
        assert_eq!(
            const_str_array_items(&define),
            Some(vec![
                ConstValue::ByteStr(b"re.TEMPLATE".to_vec()),
                ConstValue::ByteStr(b"re.IGNORECASE".to_vec()),
            ])
        );

        let op = SpaceOperation {
            result: Some(Variable::new()),
            kind: define,
        };
        assert!(
            translate_op(&op, &HashMap::new(), &empty_call_registry())
                .expect("const-str-array define translates")
                .is_empty()
        );
        match legacy_const_define_hlvalue(&op, None).expect("const-str-array define folds") {
            Some(Hlvalue::Constant(c)) => assert_eq!(
                c.value,
                ConstValue::List(vec![
                    ConstValue::ByteStr(b"re.TEMPLATE".to_vec()),
                    ConstValue::ByteStr(b"re.IGNORECASE".to_vec()),
                ])
            ),
            other => panic!("expected Constant(List), got {other:?}"),
        }
    }

    #[test]
    fn fn_const_define_is_constant_carrier() {
        use crate::model::CallTarget;

        let registry = empty_call_registry();
        let key = crate::translator::rtyper::call_registry::FunctionPathKey::from_segments([
            "module", "function",
        ]);
        let func = crate::flowspace::model::GraphFunc::new(
            "function",
            Constant::new(ConstValue::Dict(Default::default())),
        );
        let ret = Variable::new();
        ret.annotation.replace(Some(Rc::new(
            crate::annotator::model::SomeValue::Impossible,
        )));
        ret.set_concretetype(Some(
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
        ));
        let startblock = crate::flowspace::model::Block::shared(vec![]);
        let pygraph = Rc::new(crate::flowspace::pygraph::PyGraph {
            graph: Rc::new(RefCell::new(
                crate::flowspace::model::FunctionGraph::with_return_var(
                    "function",
                    startblock,
                    Hlvalue::Variable(ret),
                ),
            )),
            func,
            signature: RefCell::new(crate::flowspace::argument::Signature::new(
                Vec::<String>::new(),
                None,
                None,
            )),
            defaults: RefCell::new(Some(Vec::new())),
            access_directly: std::cell::Cell::new(false),
        });
        registry.register_callee(
            key,
            crate::flowspace::argument::Signature::new(Vec::<String>::new(), None, None),
            pygraph,
        );

        let define = OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec![
                    crate::model::FN_CONST_HEAD.to_string(),
                    "module".to_string(),
                    "function".to_string(),
                ],
                fun_decl_id: None,
            },
            args: Vec::new(),
            result_ty: ValueType::Int,
        };
        assert!(is_fn_const_define(&define));
        assert!(!is_str_const_define(&define));
        assert!(!is_const_int_array_define(&define));

        let op = SpaceOperation {
            result: Some(Variable::new()),
            kind: define.clone(),
        };
        assert!(
            translate_op(&op, &HashMap::new(), &empty_call_registry())
                .expect("fn const define translates")
                .is_empty()
        );
        match legacy_const_define_hlvalue(&op, Some(&registry)).expect("fn const define folds") {
            Some(Hlvalue::Constant(c)) => {
                assert!(matches!(c.value, ConstValue::LLPtr(_)));
                assert!(matches!(
                    c.concretetype,
                    Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Ptr(_))
                ));
            }
            other => panic!("expected LLPtr funcptr Constant carrier, got {other:?}"),
        }

        let mut legacy = LegacyGraph::new("fn_const_carrier");
        let carrier = legacy.alloc_value_var();
        let startblock = Block {
            id: legacy.startblock,
            inputargs: vec![],
            operations: vec![SpaceOperation {
                result: Some(carrier.clone()),
                kind: define,
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(carrier.clone())],
                legacy.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: legacy.returnblock,
            inputargs: vec![carrier.clone()],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        legacy.blocks = vec![startblock, returnblock];

        let output = function_graph_to_flowspace(&legacy, &registry)
            .expect("fn const carrier graph converts");
        assert_eq!(
            output.constant_concretetypes.get(&carrier),
            Some(&crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed),
            "legacy/codewriter carrier type must stay Int/Signed"
        );
        assert_eq!(
            carrier.concretetype(),
            Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed),
            "legacy Variable cell must stay Int/Signed"
        );
    }
}
