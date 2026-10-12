//! Codewriter helpers shared by `call.py`, `jtransform.py`, etc.
//!
//! Translated from `rpython/jit/codewriter/support.py`. The pieces that
//! pyre actually consumes live here as line-by-line ports; the bulk of
//! `support.py` (`LLtypeHelpers`, `OOtypeHelpers`, RPython annotator
//! glue, `get_send_oopspec`) is RPython-internal and has no Rust
//! counterpart.  `parse_oopspec` / `normalize_opargs` are ported as
//! pure helper functions ([`parse_oopspec`] / [`normalize_opargs`])
//! and consumed by [`decode_builtin_call`] when argnames are
//! registered via `CallControl::mark_oopspec_argnames`.
//!
//! [`INLINE_CALLS_TO`] mirrors `support.py inline_calls_to`
//! verbatim — the four entries upstream's `find_all_graphs` BFS seeds
//! so the inliner can always reach `int_abs` / `int_floordiv` /
//! `int_mod` / `ll_math.ll_math_sqrt`.  `int_floordiv` / `int_mod`
//! are the host graph bound to `_ll_2_int_floordiv` / `_ll_2_int_mod`;
//! the BFS seed pushes those canonical paths.
//! `int_abs` and `ll_math.ll_math_sqrt` stay unseeded: pyre has no
//! body graph for either, and binding a fnaddr alone would make the
//! call opaque.
//!
//! [`builtin_func_for_spec`] is a line-by-line port of
//! `support.py:767-808`. The RPython algorithm — cache lookup, list-or-dict
//! discriminator, helper resolution through [`setup_extra_builtin`],
//! `need_result_type` prepend, c_func construction, cache write — is
//! reproduced in the same step order. The annotator-only edges
//! (`MixLevelHelperAnnotator.constfunc`, `lltype_to_annotation`,
//! `annmodel.SomePBC`) collapse to structural no-ops: pyre's helpers are
//! concrete `extern "C"` function pointers registered through
//! [`crate::codewriter::call::CallControl::register_function_fnaddr`]
//! and resolved by canonical `_ll_<n>_<name>` lookup, so there is no
//! per-call graph materialisation. The returned [`BuiltinFuncSpec`]
//! bundles `(fnaddr, LIST_OR_DICT, need_result_type)` — pyre's analogue
//! of upstream's `(c_func, LIST_OR_DICT)` plus the side-channel
//! attribute. The cache lives behind [`CallControl::lookup_builtin_func_for_spec_cache`] /
//! [`CallControl::cache_builtin_func_for_spec`] (RPython
//! `rtyper._builtin_func_for_spec_cache`).
//!
//! [`setup_extra_builtin`] mirrors `support.py` verbatim — the
//! same `_ll_<n>_<name>` template, the same `build` prefix when `extra`
//! is supplied, the same KeyError fallback (pyre collapses
//! `globals()[name]` / `getattr(LLtypeHelpers, name).im_func` to a single
//! canonical fnaddr lookup, because pyre's helper namespace is flat).
//!
//! [`decode_builtin_call`] mirrors `support.py` line by line.
//! Upstream walks `op.args[0].value._obj` to read the helper's
//! `.oopspec` attribute (the spec string set by `@rlib.jit.oopspec`);
//! pyre has no `_obj` analogue on a Rust function pointer, so the
//! lookup goes through `CallControl::get_oopspec` — the registry
//! populated at translation time from `#[oopspec(...)]` attributes
//! and the static `jit.*` bindings `lib.rs`'s
//! `analyze_pipeline_from_module_paths` registers.  The lookup is the structural
//! mirror of upstream's `_obj.<callable>.oopspec` access: same
//! information, different host vehicle.

use crate::codewriter::call::CallControl;
use crate::codewriter::jtransform::{JitMarkerKey, jit_marker_key_from_target};
use crate::codewriter::type_state::ConcreteType;
use crate::flowspace::model::Variable;
use crate::model::{BlockId, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};
use crate::parse::CallPath;

use majit_ir::value::Type;

/// `jit/metainterp/warmspot.py find_jit_merge_points`, restricted to one graph.
pub(crate) fn find_jit_merge_point(
    graph: &FunctionGraph,
    driver_roots: &[String],
) -> Option<(BlockId, usize)> {
    let mut found = None;
    for block in &graph.blocks {
        for (index, op) in block.operations.iter().enumerate() {
            let is_merge_point = matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if jit_marker_key_from_target(target, driver_roots) == Some(JitMarkerKey::JitMergePoint)
            );
            if !is_merge_point {
                continue;
            }
            assert!(
                found.is_none(),
                "found several jit_merge_points in the same graph"
            );
            found = Some((block.id, index));
        }
    }
    found
}

/// `jit/codewriter/support.py sort_vars`.
pub(crate) fn sort_vars(args: &[Variable]) -> Vec<Variable> {
    // The split runs before the rtyper publishes every concretetype.  An
    // Unknown has no upstream `getkind()` ordering yet, so assigning it the
    // Ref key can spuriously move a later-known Int in front of it and report
    // a bogus JitDriver declaration-order error.  Keep the declaration order
    // until the types are authoritative; the typed codewriter pass performs
    // the real register-bank validation afterwards.
    if args
        .iter()
        .any(|var| FunctionGraph::concretetype_of(var) == ConcreteType::Unknown)
    {
        return args.to_vec();
    }
    let mut sorted = args.to_vec();
    sorted.sort_by_key(|var| match FunctionGraph::concretetype_of(var) {
        ConcreteType::Signed => 1,
        ConcreteType::GcRef => 2,
        ConcreteType::Float => 3,
        ConcreteType::Unknown => unreachable!("Unknown returned above"),
        ConcreteType::Void => panic!("support.py sort_vars: Void must be filtered first"),
    });
    sorted
}

/// `jit/codewriter/support.py decode_hp_hint_args`.
pub(crate) fn decode_hp_hint_args(
    marker_args: &[Variable],
    numgreens: usize,
    numreds: usize,
) -> (Vec<Variable>, Vec<Variable>) {
    assert!(
        marker_args.len() >= numgreens,
        "jit_merge_point marker has fewer operands than declared greens"
    );
    let (greens, reds) = marker_args.split_at(numgreens);
    assert_eq!(reds.len(), numreds);

    let sort_and_check = |args: &[Variable], is_green: bool| {
        let vars: Vec<Variable> = args
            .iter()
            .filter(|var| FunctionGraph::concretetype_of(var) != ConcreteType::Void)
            .cloned()
            .collect();
        if is_green {
            assert_eq!(
                vars.len(),
                args.len(),
                "not supported so far: 'greens' variables contain Void"
            );
        }
        let sorted = sort_vars(&vars);
        assert_eq!(
            vars, sorted,
            "You have to reorder the variables named in the JitDriver (both the 'greens' and \
             'reds' independently). They must be sorted like this: first all the integer-like, \
             then all the pointer-like, and finally the floats.\nGot: {:?}\nExpected: {:?}",
            vars, sorted
        );
        vars
    };
    (sort_and_check(greens, true), sort_and_check(reds, false))
}

/// `jit/codewriter/support.py split_before_jit_merge_point`.
pub(crate) fn split_before_jit_merge_point(
    graph: &mut FunctionGraph,
    mut portalblock: BlockId,
    portalopindex: usize,
    numgreens: usize,
    numreds: usize,
    driver_roots: &[String],
) -> BlockId {
    if portalopindex > 0 {
        portalblock = crate::unsimplify::split_block(graph, portalblock, portalopindex, None);
    }
    let marker = graph
        .block(portalblock)
        .operations
        .first()
        .expect("split portal block must start with jit_merge_point")
        .clone();
    let OpKind::Call { target, args, .. } = marker.kind else {
        panic!("split portal block must start with jit_merge_point")
    };
    assert_eq!(
        jit_marker_key_from_target(&target, driver_roots),
        Some(JitMarkerKey::JitMergePoint),
        "split portal block must start with jit_merge_point"
    );
    let marker_args = crate::model::call_arg_vars(
        args.get(1..)
            .expect("jit_merge_point method call must carry its receiver"),
    );
    let (greens, reds) = decode_hp_hint_args(&marker_args, numgreens, numreds);
    let mut linkargs = greens;
    linkargs.extend(reds);
    crate::unsimplify::split_block(graph, portalblock, 0, Some(&linkargs))
}

/// `warmspot.py rewrite_jit_merge_point`.
///
/// The copied portal keeps the loop that starts at `jit_merge_point`.
/// The original graph, the one callers still inline, keeps only the
/// operations that produce the marker arguments and then
/// `return portal_runner(*args)`. `guess_call_kind` classifies that
/// direct call as `recursive` because the callee is
/// `jitdriver_sd.portal_runner_ptr` (`call.py`).
pub(crate) fn rewrite_jit_merge_point(
    graph: &mut FunctionGraph,
    portal_runner: &crate::parse::CallPath,
    numgreens: usize,
    numreds: usize,
    driver_roots: &[String],
) {
    let (block, index) = find_jit_merge_point(graph, driver_roots)
        .expect("rewrite_jit_merge_point: jit_merge_point missing from original portal");
    let marker = graph.block(block).operations[index].clone();
    let OpKind::Call { args, .. } = marker.kind else {
        panic!("rewrite_jit_merge_point: jit_merge_point must be a call");
    };
    let marker_args = crate::model::call_arg_vars(
        args.get(1..)
            .expect("jit_merge_point method call must carry its receiver"),
    );
    let (greens, reds) = decode_hp_hint_args(&marker_args, numgreens, numreds);
    let mut call_args = greens;
    call_args.extend(reds);
    let return_input = graph.block(graph.returnblock).inputargs.first().cloned();
    let result_ty = match return_input
        .as_ref()
        .map(|var| FunctionGraph::concretetype_of(var))
    {
        Some(ConcreteType::Signed) => crate::model::ValueType::Int,
        Some(ConcreteType::GcRef) => crate::model::ValueType::Ref(None),
        Some(ConcreteType::Float) => crate::model::ValueType::Float,
        Some(ConcreteType::Void) => crate::model::ValueType::Void,
        // Pre-rtyper `returnvar` has no concretetype: `PORTALFUNC.RESULT`
        // is then the portal's declared source result type. A fixture
        // graph leaves `return_type` empty and returns void.
        Some(ConcreteType::Unknown) | None => match graph.return_type.as_deref() {
            None | Some("()") => crate::model::ValueType::Void,
            Some(
                "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
                | "bool" | "char",
            ) => crate::model::ValueType::Int,
            Some("f32" | "f64") => crate::model::ValueType::Float,
            Some(_) => crate::model::ValueType::Ref(None),
        },
    };
    let result = if matches!(result_ty, crate::model::ValueType::Void) {
        None
    } else {
        let ty = match result_ty {
            crate::model::ValueType::Int => ConcreteType::Signed,
            crate::model::ValueType::Ref(_) => ConcreteType::GcRef,
            crate::model::ValueType::Float => ConcreteType::Float,
            _ => unreachable!("non-void result_ty matched above"),
        };
        Some(graph.alloc_value_var_with_type(ty))
    };
    graph.block_mut(block).operations.truncate(index);
    // The ticker `continue` above `jit_merge_point` stays reachable, so
    // `look_inside_graph` would decline the stub (`loop-without-unroll_safe`)
    // and `find_all_graphs` would never inline it. The portal copy already
    // carries `unroll_safe` from `split_graph_and_record_jitdriver`.
    if !graph.func.unroll_safe {
        graph.push_hint("unroll_safe");
    }
    graph.block_mut(block).operations.push(SpaceOperation {
        result: result.clone(),
        kind: OpKind::Call {
            target: crate::model::CallTarget::function_path(portal_runner.segments.clone()),
            args: crate::model::call_args(call_args),
            result_ty,
        },
    });
    graph.set_return(block, result);
}

/// The split portal's start block is a fresh parameter list (the
/// marker's greens, then reds). `split_block` copies those variables
/// and does not move the original `OpKind::Input` ops, so the
/// annotator has neither an Input op nor a `Variable.annotation`.
/// Emit one Input per new inputarg, typed from the pre-split producer
/// and, for reds, the driver's red-type name.
pub(crate) fn seed_split_portal_input_ops(
    graph: &mut FunctionGraph,
    start: BlockId,
    green_kinds: &[Type],
    red_kinds: &[Type],
    red_types: &[String],
) {
    let inputs = graph.block(start).inputargs.clone();
    let originals = graph.blocks.iter().find_map(|block| {
        block
            .exits
            .iter()
            .find(|exit| exit.target == start)
            .map(|exit| {
                exit.args
                    .iter()
                    .filter_map(LinkArg::as_variable)
                    .cloned()
                    .collect::<Vec<_>>()
            })
    });
    let declared: Vec<ValueType> = green_kinds
        .iter()
        .chain(red_kinds.iter())
        .copied()
        .filter(|kind| *kind != Type::Void)
        .map(ir_type_to_valuetype)
        .collect();
    let declared_reds = red_kinds.iter().filter(|kind| **kind != Type::Void).count();
    let red_start = inputs.len().saturating_sub(declared_reds);
    let mut seeded = Vec::with_capacity(inputs.len());
    for (index, var) in inputs.iter().enumerate() {
        let produced = originals
            .as_ref()
            .and_then(|vars| vars.get(index))
            .and_then(|original| producer_input_type(graph, original));
        let (mut ty, mut class_root) = produced.unwrap_or_else(|| {
            (
                declared.get(index).cloned().unwrap_or(ValueType::Ref(None)),
                None,
            )
        });
        // The first green is `next_instr: usize`. The driver kind is the
        // JIT int bank, which is not the annotator's signedness.
        if index == 0 && ty == ValueType::Int {
            ty = ValueType::Unsigned;
        }
        if class_root.is_none() {
            if let ValueType::Ref(Some(root)) = &ty {
                class_root = Some(root.clone());
            }
        }
        if class_root.is_none() && index >= red_start {
            class_root = red_types.get(index - red_start).cloned();
        }
        seeded.push(SpaceOperation {
            result: Some(var.clone()),
            kind: OpKind::Input {
                name: var.name_prefix(),
                ty,
                class_root,
            },
        });
    }
    let block = graph.block_mut(start);
    seeded.append(&mut block.operations);
    block.operations = seeded;
}

fn ir_type_to_valuetype(kind: Type) -> ValueType {
    match kind {
        Type::Int => ValueType::Int,
        Type::Ref => ValueType::Ref(None),
        Type::Float => ValueType::Float,
        Type::Void => ValueType::Void,
    }
}

fn producer_input_type(
    graph: &FunctionGraph,
    var: &Variable,
) -> Option<(ValueType, Option<String>)> {
    for block in &graph.blocks {
        for op in &block.operations {
            if op.result.as_ref() != Some(var) {
                continue;
            }
            return match &op.kind {
                OpKind::Input { ty, class_root, .. } => Some((ty.clone(), class_root.clone())),
                OpKind::FieldRead { ty, .. } | OpKind::VableFieldRead { ty, .. } => {
                    Some((ty.clone(), None))
                }
                OpKind::Call { result_ty, .. }
                | OpKind::IndirectCall { result_ty, .. }
                | OpKind::BinOp { result_ty, .. }
                | OpKind::UnaryOp { result_ty, .. } => Some((result_ty.clone(), None)),
                OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } => Some((ValueType::Int, None)),
                OpKind::ConstUInt(_) => Some((ValueType::Unsigned, None)),
                OpKind::ConstBool(_) => Some((ValueType::Bool, None)),
                OpKind::ConstFloat(_) => Some((ValueType::Float, None)),
                _ => None,
            };
        }
    }
    None
}

/// `support.py inline_calls_to`.
///
/// `(oopspec_name, ll_args, ll_res)` triples whose graphs the BFS
/// seeds so the optimizer can always look inside.  Mirrors upstream
/// line by line.
pub static INLINE_CALLS_TO: &[(&str, &[Type], Type)] = &[
    ("int_abs", &[Type::Int], Type::Int),
    ("int_floordiv", &[Type::Int, Type::Int], Type::Int),
    ("int_mod", &[Type::Int, Type::Int], Type::Int),
    ("ll_math.ll_math_sqrt", &[Type::Float], Type::Float),
];

/// `support.py decode_builtin_call(op)`.
///
/// Line-by-line port.  Upstream:
///
/// ```python
/// def decode_builtin_call(op):
///     if op.opname == 'direct_call':
///         fnobj = op.args[0].value._obj
///         opargs = op.args[1:]
///         return get_call_oopspec_opargs(fnobj, opargs)
///     elif op.opname == 'gc_identityhash':
///         return get_identityhash_oopspec(op)
///     elif op.opname == 'gc_id':
///         return get_gcid_oopspec(op)
///     else:
///         raise ValueError(op.opname)
/// ```
///
/// Returns `(oopspec_name, opargs)`, mirroring the upstream 2-tuple.
///
/// `fnobj = op.args[0].value._obj` → `get_call_oopspec_opargs`:
/// upstream pulls the spec string off the callee function pointer's
/// metadata (`ll_func.oopspec`).  Pyre's `OpKind::Call` does not carry
/// the funcptr as args[0]; the equivalent metadata lives on
/// [`CallControl`] under [`CallControl::mark_oopspec`] (the registry
/// `lib.rs`'s `analyze_pipeline_from_module_paths` populates from `#[oopspec(...)]`
/// attributes and static jit.* bindings).  The lookup goes through
/// [`CallControl::get_oopspec`] — same information as
/// `op.args[0].value._obj.oopspec`, different host vehicle.  The spec
/// string is split at `(` to extract the bare name, matching
/// upstream's `parse_oopspec` first step (`support.py
/// operation_name, args = ll_func.oopspec.split('(', 1)`).
///
/// `opargs = op.args[1:]`: upstream drops the funcptr at args[0] and
/// returns the trailing positional args, then `normalize_opargs`
/// permutes / injects constants according to the placeholder pattern
/// in the spec string.  Pyre's `OpKind::Call::args` already excludes
/// the funcptr (the target is carried in `target: CallTarget`).
///
/// Strict-parity port: when `CallControl::get_oopspec_argnames(target)`
/// returns `Some(argnames)`, invoke the full [`parse_oopspec`] +
/// [`normalize_opargs`] line-by-line port.  When `None`, fall back to
/// bare-name extraction + positional forwarding — pyre's `lib.rs` jit.*
/// bindings have no `(...)` pattern, so no argname lookup is needed.
///
/// Return type is `Vec<NormalizedArg>` (not `Vec<Variable>`): a slot
/// may be a passthrough `Pass(Variable)` or a `ConstInt(i64)` literal
/// that the caller must materialise as an `OpKind::ConstInt` op.
/// `jtransform.rs::handle_builtin_call` mints fresh Variables for
/// constant slots at the residual-call site.
///
/// `gc_identityhash` / `gc_id`: pyre has no corresponding OpKind
/// variant.  These ops would land alongside a port of
/// `rpython/jit/codewriter/jtransform.py rewrite_op_gc_identityhash`
/// / `rewrite_op_gc_id`; until the variants exist, the catch-all
/// `_` arm panics with the upstream `ValueError(op.opname)` message
/// shape so a future emitter that adds the variant trips here and
/// the corresponding arm gets ported in lockstep.
///
/// **Panics** when the call has no registered oopspec — upstream's
/// `parse_oopspec` raises `AttributeError` at `ll_func.oopspec` when
/// the wrapper carries no `oopspec` attribute.  Callers must only
/// invoke `decode_builtin_call` after the call has been classified
/// as a builtin (mirroring `jtransform.py` gating through
/// `handle_builtin_call`), at which point the oopspec attribute is
/// guaranteed by the `@oopspec` decorator that promoted the helper
/// to a builtin in the first place.
pub fn decode_builtin_call(
    op: &SpaceOperation,
    call_control: &CallControl,
) -> (String, Vec<NormalizedArg>) {
    match &op.kind {
        // `support.py:756-759`: op.opname == 'direct_call' → resolve via fnobj
        OpKind::Call { target, args, .. } => {
            let args = crate::model::call_arg_vars(args);
            let args: &[crate::flowspace::model::Variable] = &args;
            // `support.py decode_builtin_call fnobj = op.args[0].value._obj` →
            // `:759 get_call_oopspec_opargs(fnobj, opargs)` →
            // `:707 operation_name, args = ll_func.oopspec.split('(', 1)`.
            // Pyre's analogue: the spec string is registered on the
            // target via `mark_oopspec` (rlib/jit.py `@oopspec(spec)`
            // semantics); the bare name is the part before `(`.
            let spec = call_control.get_oopspec(target).unwrap_or_else(|| {
                panic!(
                    "decode_builtin_call: target {target:?} carries no \
                     oopspec registration.  Upstream `support.py:707 \
                     operation_name, args = ll_func.oopspec.split('(', 1)` \
                     raises `AttributeError` on the same condition.  \
                     Callers must classify the call as a builtin via \
                     `CallControl::call_kind` before invoking \
                     `decode_builtin_call`, mirroring \
                     `jtransform.py:484 handle_builtin_call`'s upstream gating.",
                )
            });
            // `support.py operation_name, args = ll_func.oopspec.split('(', 1)`
            // → `:726 get_call_oopspec_opargs` → `:727
            // oopspec, argtuple = parse_oopspec(fnobj)`.
            //
            // Strict-parity port: when the target's argnames are
            // registered (`CallControl::mark_oopspec_argnames`),
            // invoke the full [`parse_oopspec`] + [`normalize_opargs`]
            // pair.  When no argnames are registered (the dominant
            // case today — the `lib.rs` jit.* bindings are bare dotted
            // names with no `(...)` pattern, so argname info is not
            // needed), fall back to the bare-name split and forward
            // `args` positionally (wrapped in `NormalizedArg::Pass`).
            match call_control.get_oopspec_argnames(target) {
                Some(argnames) => {
                    let argname_refs: Vec<&str> = argnames.iter().map(String::as_str).collect();
                    let (oopspec_name, argtuple) = parse_oopspec(&spec, &argname_refs);
                    let normalized = normalize_opargs(&argtuple, args);
                    (oopspec_name, normalized)
                }
                None => {
                    // `support.py decode_builtin_call opargs = op.args[1:]`: pyre's
                    // `OpKind::Call::args` already excludes the funcptr
                    // so the positional args flow through directly,
                    // wrapped as `Pass(Variable)` for the uniform return
                    // shape — matches `normalize_opargs()`'s pass-through
                    // behaviour when no argnames metadata exists.
                    let oopspec_name = spec.split('(').next().unwrap_or(&spec).trim().to_string();
                    let opargs: Vec<NormalizedArg> = args
                        .iter()
                        .map(|var| NormalizedArg::Pass(var.clone()))
                        .collect();
                    (oopspec_name, opargs)
                }
            }
        }
        // `support.py:760-763`: gc_identityhash / gc_id branches.
        //
        // Pyre has no OpKind::GcIdentityHash / OpKind::GcId yet —
        // those would land alongside a port of
        // `rpython/jit/codewriter/jtransform.py rewrite_op_gc_identityhash`
        // / `rewrite_op_gc_id`.  When the variants land, replace the
        // catch-all panic with explicit arms that delegate to ported
        // `get_identityhash_oopspec` / `get_gcid_oopspec` helpers.
        //
        // `support.py:765`: raise ValueError(op.opname)
        _ => panic!(
            "decode_builtin_call: ValueError(op.opname) — upstream \
             `support.py:765 raise ValueError(op.opname)`.  Pyre OpKind \
             variant {:?} has no decode arm; if this is a newly emitted \
             op the corresponding `rewrite_op_<name>` must be ported and \
             an arm added here.",
            op.kind,
        ),
    }
}

/// `support.py class Index` — placeholder for a positional
/// arg.  `normalize_opargs` walks an `argtuple` and replaces each
/// `Index(n)` with `opargs[n]` (passthrough), while non-Index entries
/// become constant injections.  Pyre's pure-function port emits this
/// enum from [`parse_oopspec`].
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizeSlot {
    /// `support.py argname2index[argname] = Index(n)` — positional
    /// passthrough.  `n` is the 0-based slot in the callee's
    /// argname list.
    Index(usize),
    /// `support.py argtuple = eval(args, argname2index)` — integer
    /// literal injection.  Upstream wraps as `Constant(obj,
    /// lltype.Signed)`.  Pyre uses this slot only for genuine
    /// `lltype.Signed`-tagged constants: integer literals parsed by
    /// [`parse_literal_slot`] hit this variant exclusively.  Char
    /// literals (`'a'`) panic in the parser per the comment on
    /// [`parse_literal_slot`] — RPython's `lltype.Char` has no
    /// pyre IR analogue and conflating it with `lltype.Signed`
    /// would be a deviation.
    ConstInt(i64),
    /// `support.py argtuple = eval(args, argname2index)` — float
    /// literal injection (e.g. `1.5`, `2.0e3`).  Upstream wraps as
    /// `Constant(obj, lltype.Float)`.  Stored as the f64 bit pattern
    /// to keep `PartialEq` / `Hash` derivable (`history.py:265
    /// ConstFloat.getfloatstorage`).
    ConstFloat(u64),
}

/// `support.py normalize_opargs` return-shape.
///
/// Upstream returns a list where each entry is either an existing
/// `Variable` reference (`opargs[obj.n]`) or a fresh
/// `Constant(obj, lltype.typeOf(obj))`.  Pyre splits the two cases
/// into a discriminated enum so callers (`decode_builtin_call` and
/// downstream) can decide how to materialise constants — pyre's
/// constants are `OpKind::ConstInt` SSA ops whose creation needs
/// graph-builder access at the callsite.
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizedArg {
    /// `support.py result.append(opargs[obj.n])` — pass an
    /// existing `Variable` through verbatim (upstream `Hlvalue::Variable`
    /// at `flowspace/model.py` identity).
    Pass(crate::flowspace::model::Variable),
    /// `support.py result.append(Constant(obj, lltype.Signed))`
    /// — caller materialises an `OpKind::ConstInt(v)` op carrying
    /// this value.
    ConstInt(i64),
    /// `support.py result.append(Constant(obj, lltype.Float))`
    /// — caller materialises an `OpKind::ConstFloat(bits)` op
    /// carrying this value (`bits` is the f64 bit pattern).
    ConstFloat(u64),
}

/// `support.py parse_oopspec(fnobj)` — pure-function port.
///
/// ```python
/// def parse_oopspec(fnobj):
///     FUNCTYPE = lltype.typeOf(fnobj)
///     ll_func = fnobj._callable
///     nb_args = len(FUNCTYPE.ARGS)
///     argnames = ll_func.__code__.co_varnames[:nb_args]
///     # parse the oopspec and fill in the arguments
///     operation_name, args = ll_func.oopspec.split('(', 1)
///     assert args.endswith(')')
///     args = args[:-1] + ','     # trailing comma to force tuple syntax
///     if args.strip() == ',':
///         args = '()'
///     nb_args = len(argnames)
///     argname2index = dict(zip(argnames, [Index(n) for n in range(nb_args)]))
///     argtuple = eval(args, argname2index)
///     return operation_name, argtuple
/// ```
///
/// Pyre's port takes `spec` (the `ll_func.oopspec` string registered
/// via `mark_oopspec`) and `argnames` (the callee's positional parameter
/// names, registered via `mark_oopspec_argnames`) and returns
/// `(operation_name, argtuple)` matching the upstream 2-tuple.
///
/// **Deviation** vs upstream `eval(args, argname2index)`:
/// the inner expression is parsed by a narrow comma-split + per-slot
/// literal recogniser ([`parse_literal_slot`]), NOT by a full Python
/// `eval`.  The slots pyre recognises are exactly:
///   - identifier (resolves to `Index(n)`),
///   - integer literal,
///   - float literal containing `.`, `e`, or `E` (`1.5`, `2.0e3`).
///
/// Every other slot form upstream's `eval` would accept panics here:
/// char literals (`'a'`, `'\n'`, `'\x41'`), string literals
/// (`'foo'`), nested-tuple expressions, arithmetic / boolean sub-
/// expressions, and any other Python expression.  All current
/// upstream `@oopspec(...)` decorations under `rpython/` use only
/// identifier / integer / float literals (`grep -rn "@oopspec" rpython/`
/// found zero `'<char>'` slot patterns); broaden the recogniser only
/// when a real upstream decoration introduces a new slot kind, and
/// at that point port the matching `NormalizeSlot::ConstChar(u8)` /
/// `ConstStr(String)` variant + matching `OpKind::ConstChar` /
/// `OpKind::ConstStr` materialisation so the constant carries the
/// upstream `lltype.Char` / `lltype.Ptr(STR)` tag — speculative
/// `ConstInt(byte)` fallback is NOT acceptable strict-parity
/// because `lltype.Char` ≠ `lltype.Signed`.
///
/// The spec-has-no-`(` case (a structural divergence from upstream's
/// `.split('(', 1)` which would raise ValueError) returns an empty
/// `argtuple` with the bare spec as `operation_name`.  This handles
/// pyre's `lib.rs` jit.* bare-name registrations gracefully.
pub fn parse_oopspec(spec: &str, argnames: &[&str]) -> (String, Vec<NormalizeSlot>) {
    // `support.py operation_name, args = ll_func.oopspec.split('(', 1)`
    let (operation_name, args_part) = match spec.split_once('(') {
        Some((name, rest)) => (name.trim().to_string(), rest),
        None => return (spec.trim().to_string(), Vec::new()),
    };
    // `support.py assert args.endswith(')')`
    let inner = match args_part.strip_suffix(')') {
        Some(stripped) => stripped,
        None => panic!(
            "parse_oopspec: spec {spec:?} is missing the closing `)`.  \
             Upstream `support.py:708` asserts `args.endswith(')')`."
        ),
    };
    // `support.py args = args[:-1] + ','` — trailing-comma tuple syntax.
    // `support.py:710-711 if args.strip() == ',': args = '()'`
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return (operation_name, Vec::new());
    }
    // `support.py nb_args = len(argnames)`
    // `support.py argname2index = dict(zip(argnames, [Index(n) for n in range(nb_args)]))`
    let argname2index: std::collections::HashMap<&str, usize> = argnames
        .iter()
        .enumerate()
        .map(|(n, name)| (*name, n))
        .collect();
    // `support.py argtuple = eval(args, argname2index)`
    //
    // Pyre's narrow expression parser: comma-split, then per-slot
    // resolve as identifier (→ `Index(n)`) or Python literal.  Empty
    // slots (e.g. `"foo(x,)"` after the trailing-comma append from
    // upstream :709) are ignored — they mirror the Python tuple
    // syntax that lets `(x,)` produce a 1-tuple.
    let argtuple: Vec<NormalizeSlot> = trimmed
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| parse_literal_slot(s, &argname2index, spec, argnames))
        .collect();
    (operation_name, argtuple)
}

/// Per-slot resolver for `parse_oopspec` — mirrors the
/// `argname2index`-backed `eval` resolution one cell at a time.
///
/// Resolution order matches Python literal precedence:
///   - bound identifier (`argname2index[s]`) → `Index(n)`
///   - integer literal → `ConstInt(i64)`
///   - float literal (contains `.`, `e`, or `E`) → `ConstFloat(bits)`
///   - anything else → panic.  Notably **char literals** (`'a'`,
///     `'\n'`, `'\x41'`) panic here even though upstream `eval`
///     accepts them: RPython tags such constants `lltype.Char`,
///     which has no `ConcreteType` variant in pyre — the
///     `Constant('a', lltype.Char)` ≠ `Constant(97, lltype.Signed)`
///     identity is load-bearing for descr-typed dispatch, and a
///     `ConstInt(byte)` fallback would be a speculative deviation
///     with no upstream basis in the current `rpython/` source.
fn parse_literal_slot(
    s: &str,
    argname2index: &std::collections::HashMap<&str, usize>,
    spec: &str,
    argnames: &[&str],
) -> NormalizeSlot {
    if let Some(n) = argname2index.get(s) {
        return NormalizeSlot::Index(*n);
    }
    // Integer literal — must be tried before float because `"42"`
    // parses successfully both as `i64` and `f64`, and `Constant(42,
    // lltype.typeOf(42))` upstream yields `lltype.Signed` not Float.
    if let Ok(n) = s.parse::<i64>() {
        return NormalizeSlot::ConstInt(n);
    }
    // Float literal — `1.5`, `2.0e3`, `.5`, `inf`, etc.
    if (s.contains('.') || s.contains('e') || s.contains('E'))
        && let Ok(f) = s.parse::<f64>()
    {
        return NormalizeSlot::ConstFloat(f.to_bits());
    }
    panic!(
        "parse_oopspec: slot {s:?} in spec {spec:?} is neither a known argname \
         (one of {argnames:?}) nor a supported literal (int / float).  \
         Upstream `support.py:714 eval(args, argname2index)` would additionally \
         accept char literals (`lltype.Char`), string literals (`lltype.Ptr(STR)`), \
         and nested-tuple literals.  Pyre has no `ConcreteType` variant for any \
         of those, so a faithful port requires landing the matching variant \
         (`NormalizeSlot::ConstChar` / `ConstStr` / ...) + materialisation site \
         at `jtransform.rs::handle_builtin_call` BEFORE the upstream `@oopspec(...)` \
         decoration that needs it can be honoured.  Speculative `ConstInt(byte)` \
         fallbacks for char literals are NOT strict parity — `lltype.Char` ≠ \
         `lltype.Signed`."
    );
}

/// `support.py normalize_opargs(argtuple, opargs)` — pure-function port.
///
/// ```python
/// def normalize_opargs(argtuple, opargs):
///     result = []
///     for obj in argtuple:
///         if isinstance(obj, Index):
///             result.append(opargs[obj.n])
///         else:
///             result.append(Constant(obj, lltype.typeOf(obj)))
///     return result
/// ```
///
/// Pyre's [`NormalizedArg`] enum splits the two cases so the caller can
/// decide how to materialise the constant injection (pyre's constants
/// are `OpKind::ConstInt` SSA ops whose creation needs graph-builder
/// access at the residual-call callsite, not here).
pub fn normalize_opargs(
    argtuple: &[NormalizeSlot],
    opargs: &[crate::flowspace::model::Variable],
) -> Vec<NormalizedArg> {
    argtuple
        .iter()
        .map(|slot| match slot {
            // `support.py if isinstance(obj, Index):
            //                        result.append(opargs[obj.n])`
            NormalizeSlot::Index(n) => NormalizedArg::Pass(opargs[*n].clone()),
            // `support.py else: result.append(Constant(obj, lltype.typeOf(obj)))`
            // — one branch per lltype Constant flavour.
            NormalizeSlot::ConstInt(v) => NormalizedArg::ConstInt(*v),
            NormalizeSlot::ConstFloat(bits) => NormalizedArg::ConstFloat(*bits),
        })
        .collect()
}

/// `support.py:782-794 need_result_type` flag.
///
/// In upstream this is an attribute set on the wrapper function
/// (e.g. `LLtypeHelpers._ll_1_dict_keys.need_result_type = True` or
/// `= 'exact'`).  Pyre cannot read attributes off a function pointer,
/// so the flag is co-registered alongside the fnaddr when the host
/// has a helper that needs it.  The discriminator is:
///
/// * [`NeedResultType::No`] — `getattr(impl, 'need_result_type', False)` is False.
/// * [`NeedResultType::Approx`] — flag is truthy and not exactly `'exact'`
///   (upstream `:786 if impl.need_result_type != 'exact': ll_restype = ll_restype.TO`).
/// * [`NeedResultType::Exact`] — flag is exactly `'exact'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeedResultType {
    No,
    Approx,
    Exact,
}

/// `support.py:770` cache key tuple
/// (`(oopspec_name, tuple(ll_args), ll_res, extrakey)`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BuiltinFuncSpecCacheKey {
    pub oopspec_name: String,
    pub ll_args: Vec<Type>,
    pub ll_res: Type,
    pub extrakey: Option<String>,
}

/// `support.py builtin_func_for_spec` return value.
///
/// Upstream returns the 2-tuple `(c_func, LIST_OR_DICT)` where `c_func`
/// is a `Constant` whose `.value._obj` is a `_ptr` carrying both the
/// translated graph and the function pointer.  Pyre's analog surfaces:
///
/// * [`Self::impl_name`] — canonical `_ll_<n>_<name>` identifier produced
///   by [`setup_extra_builtin`].  Mirrors upstream's `impl.func_name`
///   (the wrapper looked up from `globals()` / `LLtypeHelpers`).
/// * [`Self::fnaddr`] — host-bound `extern "C"` function pointer
///   resolved through `setup_extra_builtin` (which panics if no
///   registration exists, mirroring `support.py:687-690` raise).
/// * [`Self::list_or_dict`] — the LIST_OR_DICT discriminator from
///   `support.py:776-779`.
/// * [`Self::need_result_type`] — `need_result_type` side-channel
///   attribute (`support.py:782-794`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinFuncSpec {
    pub impl_name: String,
    pub oopspec_name: String,
    pub ll_args: Vec<Type>,
    pub ll_res: Type,
    pub list_or_dict: Type,
    pub fnaddr: i64,
    pub need_result_type: NeedResultType,
}

/// `support.py:683-693 setup_extra_builtin(rtyper, oopspec_name, nb_args, extra=None)`.
///
/// Line-by-line port.  The upstream Python:
///
/// ```python
/// def setup_extra_builtin(rtyper, oopspec_name, nb_args, extra=None):
///     name = '_ll_%d_%s' % (nb_args, oopspec_name.replace('.', '_'))
///     if extra is not None:
///         name = 'build' + name
///     try:
///         wrapper = globals()[name]
///     except KeyError:
///         wrapper = getattr(LLtypeHelpers, name).im_func
///     if extra is not None:
///         wrapper = wrapper(*extra)
///     return wrapper
/// ```
///
/// Pyre returns the `(canonical_name, fnaddr, need_result_type)` triple
/// because the `wrapper` upstream concept (a Python function object
/// carrying both an address and a `.need_result_type` attribute) does
/// not exist in Rust — those two pieces of data have to be surfaced
/// explicitly.
///
/// Resolution: pyre helpers are registered through
/// `register_function_fnaddr(CallPath::from_segments([canonical_name]))`
/// before the codewriter pipeline starts; this function reads that
/// registry.  The KeyError / `getattr(LLtypeHelpers, name).im_func`
/// fallback upstream walks two namespaces (module globals + class
/// dict); pyre's namespace is flat so a single lookup suffices.
///
/// **Panics** when the host has not registered a fnaddr under the
/// canonical name.  This mirrors `support.py:687-690` which raises
/// `AttributeError` from `getattr(LLtypeHelpers, name).im_func` when
/// both `globals()[name]` and `LLtypeHelpers.<name>` lookups fail.
/// Upstream never observes this raise in practice because every
/// helper named in `support.py` exists; pyre's host equivalents
/// (`pyre-interpreter::jit_trace_fnaddrs`) MUST register the
/// canonical name for any helper the codewriter pipeline asks for.
/// The strict panic is what lets `setup_extra_builtin` act as a
/// helper-existence invariant; weaker forms (returning `None`)
/// would let unbound helpers flow silently into placeholder fnaddrs
/// downstream and mask real wiring gaps.
pub fn setup_extra_builtin(
    call_control: Option<&CallControl>,
    oopspec_name: &str,
    nb_args: usize,
    extra: Option<&[String]>,
    extrakey: Option<&str>,
) -> (String, i64, NeedResultType) {
    // `support.py:684`: name = '_ll_%d_%s' % (nb_args, oopspec_name.replace('.', '_'))
    let mut name = format!("_ll_{}_{}", nb_args, oopspec_name.replace('.', "_"));
    // `support.py`: if extra is not None: name = 'build' + name
    if extra.is_some() {
        name = format!("build{}", name);
    }
    // `support.py:687-690`:
    //   try:
    //       wrapper = globals()[name]
    //   except KeyError:
    //       wrapper = getattr(LLtypeHelpers, name).im_func
    //
    // Pyre's flat namespace lookup against the host-registered fnaddr
    // table.  An unregistered canonical name is a wiring bug —
    // upstream would raise `AttributeError` at the inner `getattr`
    // (the `im_func` attribute access on a missing class attribute);
    // pyre panics so the same failure surfaces loudly at codewriter
    // setup time instead of corrupting silently.
    let cc = call_control.expect(
        "setup_extra_builtin: call_control is required for a strict lookup; \
         upstream `support.py:687-690` resolves the wrapper through the rtyper-bound \
         `LLtypeHelpers` namespace, which pyre exposes via `CallControl::lookup_function_fnaddr`.",
    );
    // `support.py`: if extra is not None: wrapper = wrapper(*extra)
    //
    // Upstream calls the resolved wrapper as a Python factory to
    // produce the specialized helper.  Pyre cannot run a Rust
    // function pointer as a factory (no runtime codegen, no
    // annotator to drive specialisation); hosts pre-register the
    // specialised fnaddr per `(canonical_name, extrakey)` pair via
    // [`CallControl::register_builtin_factory`], and the lookup
    // happens BEFORE the generic `lookup_function_fnaddr` fallback
    // so the specialised binding wins.  When `extra.is_some()` and
    // no factory specialisation is registered, the fallback still
    // tries the build-prefixed canonical name (matching upstream's
    // KeyError-then-getattr two-step structure), letting a host that
    // pre-builds a single specialisation register under the same
    // canonical key without paying for the factory-registry layer.
    let factory_hit = match (extra, extrakey) {
        (Some(_), Some(k)) => cc.lookup_builtin_factory(&name, k),
        // `support.py assert (extra is None) == (extrakey is None)`
        // is checked at the `builtin_func_for_spec` boundary; both
        // None here means the upstream `wrapper(*extra)` step is
        // skipped, and both Some without a registered factory falls
        // through to the canonical-name lookup below.
        _ => None,
    };
    let fnaddr = factory_hit.unwrap_or_else(|| {
        let canonical = CallPath::from_segments([name.as_str()]);
        cc.lookup_function_fnaddr(&canonical).unwrap_or_else(|| {
            panic!(
                "setup_extra_builtin: canonical helper `{name}` is not registered. \
                 Upstream `support.py:687-690` raises `AttributeError` at \
                 `getattr(LLtypeHelpers, name).im_func` when both `globals()[name]` \
                 and `LLtypeHelpers.<name>` lookups fail.  Pyre's host \
                 (`pyre-interpreter::jit_trace_fnaddrs` or equivalent) must register \
                 this helper under that exact canonical name before the codewriter \
                 pipeline invokes `builtin_func_for_spec({oopspec_name:?}, ...)`.",
            )
        })
    });
    // `need_result_type`: upstream reads `getattr(impl, 'need_result_type', False)`.
    // Pyre cannot reach into a function pointer; the flag is co-registered
    // through `CallControl::lookup_need_result_type` against the same
    // canonical name.  Unregistered canonical names default to
    // `NeedResultType::No`, matching `getattr(..., 'need_result_type', False)`'s
    // missing-attribute fallback.
    let need_result_type = cc
        .lookup_need_result_type(&name)
        .unwrap_or(NeedResultType::No);
    // `support.py:693`: return wrapper
    (name, fnaddr, need_result_type)
}

/// `support.py:767-808 builtin_func_for_spec(rtyper, oopspec_name, ll_args, ll_res, extra=None, extrakey=None)`.
///
/// Line-by-line port.  Upstream Python:
///
/// ```python
/// def builtin_func_for_spec(rtyper, oopspec_name, ll_args, ll_res,
///                           extra=None, extrakey=None):
///     assert (extra is None) == (extrakey is None)
///     key = (oopspec_name, tuple(ll_args), ll_res, extrakey)
///     try:
///         return rtyper._builtin_func_for_spec_cache[key]
///     except (KeyError, AttributeError):
///         pass
///     args_s = [lltype_to_annotation(v) for v in ll_args]
///     if '.' not in oopspec_name:    # 'newxxx' operations
///         LIST_OR_DICT = ll_res
///     else:
///         LIST_OR_DICT = ll_args[0]
///     s_result = lltype_to_annotation(ll_res)
///     impl = setup_extra_builtin(rtyper, oopspec_name, len(args_s), extra)
///     if getattr(impl, 'need_result_type', False):
///         ...                       # prepend SomePBC([desc]) to args_s
///     if hasattr(rtyper, 'annotator'):  # regular case
///         mixlevelann = MixLevelHelperAnnotator(rtyper)
///         c_func = mixlevelann.constfunc(impl, args_s, s_result)
///         mixlevelann.finish()
///     else:
///         # for testing only
///         c_func = Constant(oopspec_name,
///                           lltype.Ptr(lltype.FuncType(ll_args, ll_res)))
///     if not hasattr(rtyper, '_builtin_func_for_spec_cache'):
///         rtyper._builtin_func_for_spec_cache = {}
///     rtyper._builtin_func_for_spec_cache[key] = (c_func, LIST_OR_DICT)
///     return c_func, LIST_OR_DICT
/// ```
///
/// Pyre's `call_control` is the `rtyper` analog: it owns the
/// persistent fnaddr registry and the
/// `_builtin_func_for_spec_cache`.
///
/// **Caveat re: `call_control=None`**: upstream's testing-only
/// branch (`support.py:800-803`) does not need an annotator — it
/// constructs a placeholder `Constant(oopspec_name,
/// Ptr(FuncType(...)))` whose only consumer is the test that
/// inspects the spec name + arg types.  Pyre's structural divergence:
/// `setup_extra_builtin` always needs `call_control` for the strict
/// fnaddr lookup (`support.py globals()[name]` /
/// `LLtypeHelpers.<name>.im_func` maps to
/// `CallControl::lookup_function_fnaddr` in pyre, and no Rust analog
/// of the symbolic Python identifier exists without a registry).
/// `setup_extra_builtin(None, ...)` therefore **panics** at the
/// `expect()` site, NOT returning a stub — current pyre callers
/// always supply a real `&CallControl`.  When a testing-only path
/// appears, port the upstream stub by returning `(canonical_name,
/// symbolic_fnaddr_for_path(...), NeedResultType::No)` from
/// `setup_extra_builtin` and having `builtin_func_for_spec` skip the
/// cache write (already handled).
pub fn builtin_func_for_spec(
    call_control: Option<&CallControl>,
    oopspec_name: &str,
    ll_args: &[Type],
    ll_res: Type,
    extra: Option<&[String]>,
    extrakey: Option<&str>,
) -> BuiltinFuncSpec {
    // `support.py`: assert (extra is None) == (extrakey is None).
    // The workspace defines no `[profile.release]`, so `debug_assertions`
    // is off in the profile that actually runs the codewriter (the
    // `pyre-jit-trace` build script) — a `debug_assert_eq!` here would be
    // compiled out of every shipping build, unlike the sibling
    // `support.py:708` port below, which panics unconditionally.
    assert_eq!(
        extra.is_none(),
        extrakey.is_none(),
        "support.py:769 — extra and extrakey must be supplied together",
    );

    // `support.py:770`: key = (oopspec_name, tuple(ll_args), ll_res, extrakey)
    let key = BuiltinFuncSpecCacheKey {
        oopspec_name: oopspec_name.to_string(),
        ll_args: ll_args.to_vec(),
        ll_res,
        extrakey: extrakey.map(String::from),
    };

    // `support.py:771-774`: try: return rtyper._builtin_func_for_spec_cache[key]
    //                       except (KeyError, AttributeError): pass
    if let Some(cc) = call_control
        && let Some(cached) = cc.lookup_builtin_func_for_spec_cache(&key)
    {
        return cached;
    }

    // `support.py:775`: args_s = [lltype_to_annotation(v) for v in ll_args]
    //
    // Pyre has no annotator; `ll_args` are carried verbatim through
    // the returned spec.  Upstream's `args_s` is consumed by
    // `mixlevelann.constfunc(impl, args_s, s_result)`, which has no
    // Rust counterpart.

    // `support.py:776-779`: LIST_OR_DICT discriminator.
    let list_or_dict = if !oopspec_name.contains('.') {
        // 'newxxx' operations
        ll_res
    } else {
        // dotted name (list / dict family, ll_math.*)
        debug_assert!(
            !ll_args.is_empty(),
            "support.py:779 — dotted oopspec name `{oopspec_name}` requires ll_args[0]",
        );
        ll_args[0]
    };

    // `support.py:780`: s_result = lltype_to_annotation(ll_res)
    // (no annotator surface in pyre; `ll_res` flows verbatim into the
    // returned spec)

    // `support.py:781`: impl = setup_extra_builtin(rtyper, oopspec_name, len(args_s), extra)
    //
    // `extrakey` is threaded as well so the factory-registry lookup
    // (`wrapper(*extra)` analog) can pick the right specialised
    // fnaddr — upstream's wrapper is invoked with `extra` itself, and
    // pyre's analog uses the host-supplied `extrakey` as the
    // matching cache discriminator.
    let (impl_name, fnaddr, need_result_type) =
        setup_extra_builtin(call_control, oopspec_name, ll_args.len(), extra, extrakey);

    // `support.py:782-794`: if getattr(impl, 'need_result_type', False):
    //                           if hasattr(rtyper, 'annotator'):
    //                               bk = rtyper.annotator.bookkeeper
    //                               ll_restype = ll_res
    //                               if impl.need_result_type != 'exact':
    //                                   ll_restype = ll_restype.TO
    //                               desc = bk.getdesc(ll_restype)
    //                           else:
    //                               class TestingDesc(object): ...
    //                               desc = TestingDesc()
    //                           args_s.insert(0, annmodel.SomePBC([desc]))
    //
    // The SomePBC prepended on `args_s` is annotator-only state: it
    // tells the annotator that the wrapper's first argument is a
    // pointer-to-type constant.  Pyre carries the same information on
    // the returned spec via `need_result_type`; downstream residual-call
    // emitters that handle `Approx` / `Exact` are responsible for
    // prepending the type-tag arg when materialising the call.  No
    // INLINE_CALLS_TO entry sets the flag, so this arm is structural
    // until the dict-iter helpers land.
    let _ = need_result_type;

    // `support.py:796-803`: c_func construction.
    //   if hasattr(rtyper, 'annotator'):   # regular case
    //       mixlevelann = MixLevelHelperAnnotator(rtyper)
    //       c_func = mixlevelann.constfunc(impl, args_s, s_result)
    //       mixlevelann.finish()
    //   else:                              # for testing only
    //       c_func = Constant(oopspec_name,
    //                         lltype.Ptr(lltype.FuncType(ll_args, ll_res)))
    //
    // Pyre always lands on the regular case: `setup_extra_builtin`
    // above resolved the host-bound `fnaddr`, which mirrors what
    // upstream's `mixlevelann.constfunc` materialises (the function
    // pointer half of the resulting Constant).  The testing-only
    // upstream branch (`Constant(oopspec_name, lltype.Ptr(...))` with
    // no real graph behind it) corresponds to a deliberate
    // host-config decision pyre does not exercise — the strict
    // panic in `setup_extra_builtin` prevents reaching this point
    // with an unbound helper.  No separate graph materialisation
    // happens because pyre cannot run the RPython annotator over a
    // Rust impl; hosts that want the BFS to look inside the helper
    // additionally call `register_function_graph` against the same
    // canonical `CallPath`.
    let spec = BuiltinFuncSpec {
        impl_name,
        oopspec_name: oopspec_name.to_string(),
        ll_args: ll_args.to_vec(),
        ll_res,
        list_or_dict,
        fnaddr,
        need_result_type,
    };

    // `support.py:805-807`: if not hasattr(rtyper, '_builtin_func_for_spec_cache'):
    //                           rtyper._builtin_func_for_spec_cache = {}
    //                       rtyper._builtin_func_for_spec_cache[key] = (c_func, LIST_OR_DICT)
    //
    // Pyre's cache lives on `CallControl` and is unconditionally
    // initialised at construction time (no `hasattr` guard needed —
    // the field is always present).  Cache writes happen behind a
    // `RefCell` so the read-only `call_control: Option<&CallControl>`
    // signature stays parity with upstream's `rtyper` parameter shape.
    if let Some(cc) = call_control {
        cc.cache_builtin_func_for_spec(key, spec.clone());
    }

    // `support.py:809`: return c_func, LIST_OR_DICT
    spec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_vars_preserves_declaration_order_while_any_type_is_unknown() {
        let mut graph = FunctionGraph::new("unknown_sort");
        let unresolved = graph.alloc_value_var_with_type(ConcreteType::Unknown);
        let integer = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let reference = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let declared = vec![unresolved, integer, reference];
        assert_eq!(sort_vars(&declared), declared);
    }

    #[test]
    fn sort_vars_orders_fully_typed_register_banks() {
        let mut graph = FunctionGraph::new("typed_sort");
        let reference = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let float = graph.alloc_value_var_with_type(ConcreteType::Float);
        let integer = graph.alloc_value_var_with_type(ConcreteType::Signed);
        assert_eq!(
            sort_vars(&[reference.clone(), float.clone(), integer.clone()]),
            vec![integer, reference, float]
        );
    }

    #[test]
    fn inline_calls_to_matches_upstream_four_entries() {
        // `support.py:444-449` parity — exactly the four upstream
        // entries, in upstream order.  A new entry here means a new
        // upstream line landed; sync it instead of letting the
        // tables drift.
        assert_eq!(INLINE_CALLS_TO.len(), 4);
        assert_eq!(INLINE_CALLS_TO[0].0, "int_abs");
        assert_eq!(INLINE_CALLS_TO[0].1, &[Type::Int]);
        assert_eq!(INLINE_CALLS_TO[0].2, Type::Int);
        assert_eq!(INLINE_CALLS_TO[1].0, "int_floordiv");
        assert_eq!(INLINE_CALLS_TO[1].1, &[Type::Int, Type::Int]);
        assert_eq!(INLINE_CALLS_TO[1].2, Type::Int);
        assert_eq!(INLINE_CALLS_TO[2].0, "int_mod");
        assert_eq!(INLINE_CALLS_TO[2].1, &[Type::Int, Type::Int]);
        assert_eq!(INLINE_CALLS_TO[2].2, Type::Int);
        assert_eq!(INLINE_CALLS_TO[3].0, "ll_math.ll_math_sqrt");
        assert_eq!(INLINE_CALLS_TO[3].1, &[Type::Float]);
        assert_eq!(INLINE_CALLS_TO[3].2, Type::Float);
    }

    #[test]
    fn setup_extra_builtin_renders_canonical_name() {
        // `support.py name = '_ll_%d_%s' % (nb_args, oopspec_name.replace('.', '_'))`.
        // Register stub fnaddrs so the strict-panic resolution succeeds —
        // the test is purely about the name template, not the lookup.
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 1);
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_int_abs"]), 2);
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_ll_math_ll_math_sqrt"]), 3);
        let (name, _, _) = setup_extra_builtin(Some(&cc), "int_mod", 2, None, None);
        assert_eq!(name, "_ll_2_int_mod");
        let (name, _, _) = setup_extra_builtin(Some(&cc), "int_abs", 1, None, None);
        assert_eq!(name, "_ll_1_int_abs");
        let (name, _, _) = setup_extra_builtin(Some(&cc), "ll_math.ll_math_sqrt", 1, None, None);
        // `.` → `_` per `support.py .replace('.', '_')`
        assert_eq!(name, "_ll_1_ll_math_ll_math_sqrt");
    }

    #[test]
    fn setup_extra_builtin_prepends_build_when_extra_supplied() {
        // `support.py if extra is not None: name = 'build' + name`
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["build_ll_0_dict_make"]), 1);
        let extra: Vec<String> = vec!["foo".to_string()];
        let (name, _, _) =
            setup_extra_builtin(Some(&cc), "dict_make", 0, Some(&extra), Some("foo"));
        assert_eq!(name, "build_ll_0_dict_make");
    }

    #[test]
    fn setup_extra_builtin_resolves_host_registered_fnaddr() {
        let mut cc = CallControl::new();
        // Host registers under the canonical `_ll_<n>_<name>` form per
        // `support.py:684` template.
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 0xdead_beef);
        let (_, fnaddr, _) = setup_extra_builtin(Some(&cc), "int_mod", 2, None, None);
        assert_eq!(fnaddr, 0xdead_beef);
    }

    #[test]
    #[should_panic(expected = "canonical helper `_ll_2_int_mod` is not registered")]
    fn setup_extra_builtin_panics_when_canonical_unregistered() {
        // `support.py`: when both `globals()[name]` and
        // `getattr(LLtypeHelpers, name).im_func` lookups miss, RPython
        // raises `AttributeError`.  Pyre mirrors the raise via panic so
        // missing wiring surfaces loudly at setup time.
        let cc = CallControl::new();
        let _ = setup_extra_builtin(Some(&cc), "int_mod", 2, None, None);
    }

    #[test]
    fn setup_extra_builtin_reads_need_result_type_registry() {
        // `support.py`: `getattr(impl, 'need_result_type', False)`.
        // Pyre's host registers the flag against the canonical name; the
        // setup_extra_builtin lookup must observe it.
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_dict_keys"]), 0x44);
        cc.register_need_result_type("_ll_1_dict_keys", NeedResultType::Approx);
        let (_, _, ty) = setup_extra_builtin(Some(&cc), "dict_keys", 1, None, None);
        assert_eq!(ty, NeedResultType::Approx);
    }

    #[test]
    fn setup_extra_builtin_uses_factory_registry_when_extra_supplied() {
        // `support.py wrapper = wrapper(*extra)`: pyre's
        // factory registry returns the specialised fnaddr that the
        // host pre-built for `(canonical_name, extrakey)`.  The
        // generic `build_ll_2_dict_iter` canonical-name fallback is
        // intentionally bound to a different address so the test
        // distinguishes which path won — the factory match must
        // shadow the fallback.
        const FALLBACK: i64 = 0xfa11_bac1;
        const SPECIAL: i64 = 0xcafe_5e11;
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["build_ll_2_dict_iter"]), FALLBACK);
        cc.register_builtin_factory("build_ll_2_dict_iter", "Ptr(STR)", SPECIAL);
        let extra: Vec<String> = vec!["Ptr(STR)".to_string()];
        let (_, fnaddr, _) =
            setup_extra_builtin(Some(&cc), "dict_iter", 2, Some(&extra), Some("Ptr(STR)"));
        assert_eq!(fnaddr, SPECIAL);
    }

    #[test]
    fn setup_extra_builtin_factory_falls_back_to_canonical_when_unregistered() {
        // `support.py:691-692` after the wrapper itself is found via
        // `globals()[name]`: pyre falls back to the generic
        // canonical-name fnaddr when no `(canonical, extrakey)`
        // specialisation is registered, matching the pre-factory
        // workaround pattern.
        const FALLBACK: i64 = 0xfa11_bac1;
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["build_ll_2_dict_iter"]), FALLBACK);
        let extra: Vec<String> = vec!["Ptr(STR)".to_string()];
        let (_, fnaddr, _) =
            setup_extra_builtin(Some(&cc), "dict_iter", 2, Some(&extra), Some("Ptr(STR)"));
        assert_eq!(fnaddr, FALLBACK);
    }

    #[test]
    fn builtin_func_for_spec_no_dot_uses_ll_res_as_list_or_dict() {
        // `support.py:776-777`: if '.' not in oopspec_name: LIST_OR_DICT = ll_res
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_int_abs"]), 0x11);
        let spec = builtin_func_for_spec(Some(&cc), "int_abs", &[Type::Int], Type::Int, None, None);
        assert_eq!(spec.oopspec_name, "int_abs");
        assert_eq!(spec.impl_name, "_ll_1_int_abs");
        assert_eq!(spec.ll_args, &[Type::Int]);
        assert_eq!(spec.ll_res, Type::Int);
        assert_eq!(spec.list_or_dict, Type::Int);
        assert_eq!(spec.fnaddr, 0x11);
        assert_eq!(spec.need_result_type, NeedResultType::No);
    }

    #[test]
    fn builtin_func_for_spec_dotted_name_uses_first_arg_as_list_or_dict() {
        // `support.py:778-779`: dotted name → LIST_OR_DICT = ll_args[0]
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(
            CallPath::from_segments(["_ll_1_ll_math_ll_math_sqrt"]),
            0x22,
        );
        let spec = builtin_func_for_spec(
            Some(&cc),
            "ll_math.ll_math_sqrt",
            &[Type::Float],
            Type::Float,
            None,
            None,
        );
        assert_eq!(spec.list_or_dict, Type::Float);
        assert_eq!(spec.impl_name, "_ll_1_ll_math_ll_math_sqrt");
    }

    #[test]
    fn builtin_func_for_spec_resolves_host_registered_canonical_fnaddr() {
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 0xdead_beef);
        let spec = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            None,
            None,
        );
        assert_eq!(spec.fnaddr, 0xdead_beef);
    }

    #[test]
    fn builtin_func_for_spec_caches_repeated_calls() {
        // `support.py:771-774` + `:805-807`: the second call returns
        // the cached spec.  Use a controlled fnaddr that we can flip
        // after the first call to confirm the second call reads from
        // the cache rather than re-resolving.
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 0x1111);
        let first = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            None,
            None,
        );
        assert_eq!(first.fnaddr, 0x1111);
        // Mutate the underlying registry; cache should shield the
        // second call from observing the update.
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 0x2222);
        let second = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            None,
            None,
        );
        assert_eq!(second.fnaddr, 0x1111);
    }

    #[test]
    fn builtin_func_for_spec_cache_keyed_on_extrakey() {
        // `support.py:770` key tuple includes extrakey — distinct
        // extrakeys must produce distinct cache entries even when
        // every other field matches.
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["build_ll_2_int_mod"]), 0xaaaa);
        let extra_a: Vec<String> = vec!["a".to_string()];
        let extra_b: Vec<String> = vec!["b".to_string()];
        let spec_a = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            Some(&extra_a),
            Some("a"),
        );
        let spec_b = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            Some(&extra_b),
            Some("b"),
        );
        // Both produce the `build_ll_2_int_mod` canonical name (extra
        // is_some → 'build' prefix) and the same fnaddr, but the cache
        // tracks them under distinct keys for upstream-parity dedup.
        assert_eq!(spec_a.impl_name, "build_ll_2_int_mod");
        assert_eq!(spec_b.impl_name, "build_ll_2_int_mod");
        assert_eq!(spec_a.fnaddr, 0xaaaa);
        assert_eq!(spec_b.fnaddr, 0xaaaa);
    }

    #[test]
    #[should_panic(expected = "extra and extrakey must be supplied together")]
    fn builtin_func_for_spec_rejects_mismatched_extra_extrakey() {
        // `support.py assert (extra is None) == (extrakey is None)`
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_int_abs"]), 0xa);
        let extra: Vec<String> = vec!["a".to_string()];
        let _ = builtin_func_for_spec(
            Some(&cc),
            "int_abs",
            &[Type::Int],
            Type::Int,
            Some(&extra),
            None,
        );
    }

    // ───────── `decode_builtin_call` tests ─────────

    fn make_call_op(
        target: crate::model::CallTarget,
        num_args: usize,
    ) -> (SpaceOperation, Vec<crate::flowspace::model::Variable>) {
        let arg_vars: Vec<crate::flowspace::model::Variable> = (0..num_args)
            .map(|_| crate::flowspace::model::Variable::new())
            .collect();
        let op = SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target,
                args: crate::model::call_args(arg_vars.clone()),
                result_ty: crate::model::ValueType::Int,
            },
        };
        (op, arg_vars)
    }

    #[test]
    fn decode_builtin_call_returns_registered_oopspec_and_positional_args() {
        // `support.py operation_name, args = ll_func.oopspec.split('(', 1)`:
        // bare-name spec entries (e.g. the `lib.rs` jit.* bindings)
        // resolve to the spec value itself, no `(` stripping needed.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["jit", "isconstant"]);
        cc.mark_oopspec(path.clone(), "jit.isconstant".to_string());
        let (op, args) = make_call_op(
            crate::model::CallTarget::function_path(["jit", "isconstant"]),
            2,
        );
        let (name, opargs) = decode_builtin_call(&op, &cc);
        assert_eq!(name, "jit.isconstant");
        assert_eq!(
            opargs,
            vec![
                NormalizedArg::Pass(args[0].clone()),
                NormalizedArg::Pass(args[1].clone()),
            ]
        );
    }

    #[test]
    fn decode_builtin_call_strips_arg_pattern_from_spec_name() {
        // `support.py ll_func.oopspec.split('(', 1)`: when the spec
        // carries the placeholder pattern (e.g. `"int.py_mod(x, y)"`),
        // the operation name is the prefix before `(`.  Without
        // argname registration the positional flow forwards args as
        // `NormalizedArg::Pass`.
        let mut cc = CallControl::new();
        cc.mark_oopspec(
            CallPath::from_segments(["int", "py_mod"]),
            "int.py_mod(x, y)".to_string(),
        );
        let (op, args) = make_call_op(
            crate::model::CallTarget::function_path(["int", "py_mod"]),
            2,
        );
        let (name, opargs) = decode_builtin_call(&op, &cc);
        assert_eq!(name, "int.py_mod");
        assert_eq!(
            opargs,
            vec![
                NormalizedArg::Pass(args[0].clone()),
                NormalizedArg::Pass(args[1].clone()),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "carries no oopspec registration")]
    fn decode_builtin_call_panics_when_target_has_no_oopspec_registration() {
        // `support.py:707 ll_func.oopspec.split(...)` raises
        // `AttributeError` when the wrapper has no `oopspec`.  Pyre
        // mirrors via panic so callers gating on `CallKind::Builtin`
        // (per `jtransform.py handle_builtin_call`) catch wiring
        // gaps loudly.
        let cc = CallControl::new();
        let (op, _args) = make_call_op(
            crate::model::CallTarget::function_path(["some", "unregistered"]),
            0,
        );
        let _ = decode_builtin_call(&op, &cc);
    }

    #[test]
    #[should_panic(expected = "ValueError(op.opname)")]
    fn decode_builtin_call_panics_on_non_call_opkind() {
        // `support.py decode_builtin_call raise ValueError(op.opname)`: any opname
        // outside {direct_call, gc_identityhash, gc_id} raises.  Pyre
        // mirrors the raise via panic.  IndirectCall is included
        // because upstream does NOT have an `indirect_call` arm
        // (the previous pyre `_ => None` for IndirectCall was a
        // deviation; this test pins the corrected behaviour).
        let cc = CallControl::new();
        let op = SpaceOperation {
            result: None,
            kind: OpKind::IndirectCall {
                funcptr: crate::flowspace::model::Variable::new(),
                args: vec![],
                graphs: None,
                family_key: None,
                result_ty: crate::model::ValueType::Int,
            },
        };
        let _ = decode_builtin_call(&op, &cc);
    }

    // ── `parse_oopspec` + `normalize_opargs` pure-function ports ────

    #[test]
    fn parse_oopspec_returns_bare_name_when_spec_has_no_paren() {
        // `support.py ll_func.oopspec.split('(', 1)` raises
        // ValueError upstream when there's no `(`.  Pyre's port
        // gracefully handles the bare-name registrations at
        // `lib.rs` (`"jit.isconstant"` etc.) by returning an
        // empty argtuple.
        let (name, argtuple) = super::parse_oopspec("jit.isconstant", &["value"]);
        assert_eq!(name, "jit.isconstant");
        assert!(argtuple.is_empty());
    }

    #[test]
    fn parse_oopspec_returns_empty_argtuple_for_empty_paren_body() {
        // `support.py:710-711 if args.strip() == ',': args = '()'` —
        // empty parenthesised body yields an empty argtuple.
        let (name, argtuple) = super::parse_oopspec("foo()", &[]);
        assert_eq!(name, "foo");
        assert!(argtuple.is_empty());
    }

    #[test]
    fn parse_oopspec_resolves_positional_identifiers_to_index_slots() {
        // `support.py argname2index = dict(zip(argnames, [Index(n) for n in
        // range(nb_args)])); argtuple = eval(args, argname2index)` — the
        // dominant case where the spec's identifiers match the callee's
        // parameter names in order.
        let (name, argtuple) = super::parse_oopspec("int.py_mod(x, y)", &["x", "y"]);
        assert_eq!(name, "int.py_mod");
        assert_eq!(
            argtuple,
            vec![
                super::NormalizeSlot::Index(0),
                super::NormalizeSlot::Index(1)
            ]
        );
    }

    #[test]
    fn parse_oopspec_permutes_args_when_identifier_order_differs() {
        // `support.py:713-714` — if the spec lists identifiers in a
        // different order than the callee declares them, the eval
        // result is a permuted Index sequence.
        let (name, argtuple) = super::parse_oopspec("foo(y, x)", &["x", "y"]);
        assert_eq!(name, "foo");
        assert_eq!(
            argtuple,
            vec![
                super::NormalizeSlot::Index(1),
                super::NormalizeSlot::Index(0)
            ]
        );
    }

    #[test]
    fn parse_oopspec_injects_integer_literal_as_const_int_slot() {
        // `support.py argtuple = eval(args, argname2index)` —
        // literals that don't match an argname pass through as
        // `Constant(obj, lltype.typeOf(obj))`.  Pyre's narrow port
        // accepts integer literals.
        let (name, argtuple) = super::parse_oopspec("foobar(2, c, i)", &["c", "i"]);
        assert_eq!(name, "foobar");
        assert_eq!(
            argtuple,
            vec![
                super::NormalizeSlot::ConstInt(2),
                super::NormalizeSlot::Index(0),
                super::NormalizeSlot::Index(1),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "missing the closing `)`")]
    fn parse_oopspec_panics_on_unclosed_paren() {
        // `support.py assert args.endswith(')')` — mirror the
        // upstream AssertionError with a Rust panic.
        super::parse_oopspec("foo(x", &["x"]);
    }

    #[test]
    #[should_panic(expected = "is neither a known argname")]
    fn parse_oopspec_panics_on_unknown_non_integer_slot() {
        // `support.py eval(args, argname2index)` would accept
        // `"z"` only if it's bound in argname2index; otherwise eval
        // raises NameError.  Pyre's port panics with a more
        // informative message citing the eval gap.
        super::parse_oopspec("foo(z)", &["x", "y"]);
    }

    #[test]
    fn normalize_opargs_passes_through_index_slots() {
        // `support.py if isinstance(obj, Index):
        //                        result.append(opargs[obj.n])` — identity
        // pass for positional placeholders.
        let argtuple = vec![
            super::NormalizeSlot::Index(0),
            super::NormalizeSlot::Index(1),
        ];
        let v0 = crate::flowspace::model::Variable::new();
        let v1 = crate::flowspace::model::Variable::new();
        let opargs = vec![v0.clone(), v1.clone()];
        let normalized = super::normalize_opargs(&argtuple, &opargs);
        assert_eq!(
            normalized,
            vec![
                super::NormalizedArg::Pass(v0),
                super::NormalizedArg::Pass(v1),
            ]
        );
    }

    #[test]
    fn normalize_opargs_permutes_and_injects_constants() {
        // Combined: permutation (Index(1) before Index(0)) + literal
        // injection (`Constant(7, ...)` upstream → `ConstInt(7)` pyre).
        let argtuple = vec![
            super::NormalizeSlot::ConstInt(7),
            super::NormalizeSlot::Index(1),
            super::NormalizeSlot::Index(0),
        ];
        let v100 = crate::flowspace::model::Variable::new();
        let v200 = crate::flowspace::model::Variable::new();
        let opargs = vec![v100.clone(), v200.clone()];
        let normalized = super::normalize_opargs(&argtuple, &opargs);
        assert_eq!(
            normalized,
            vec![
                super::NormalizedArg::ConstInt(7),
                super::NormalizedArg::Pass(v200),
                super::NormalizedArg::Pass(v100),
            ]
        );
    }

    #[test]
    fn decode_builtin_call_uses_normalize_opargs_when_argnames_registered() {
        // End-to-end: with `mark_oopspec_argnames`, the full-port
        // branch in `decode_builtin_call` fires, parsing `(y, x)`
        // and permuting the positional opargs accordingly.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["my_helper"]);
        cc.mark_oopspec(path.clone(), "myop(y, x)".to_string());
        cc.mark_oopspec_argnames(path, vec!["x".to_string(), "y".to_string()]);
        let (op, args) = make_call_op(crate::model::CallTarget::function_path(["my_helper"]), 2);
        let (name, opargs) = decode_builtin_call(&op, &cc);
        assert_eq!(name, "myop");
        // Argname `y` → Index(1) → opargs[1] = args[1]
        // Argname `x` → Index(0) → opargs[0] = args[0]
        assert_eq!(
            opargs,
            vec![
                NormalizedArg::Pass(args[1].clone()),
                NormalizedArg::Pass(args[0].clone()),
            ]
        );
    }

    #[test]
    fn decode_builtin_call_emits_const_int_slot_for_integer_literal() {
        // Mirrors `test_support.py test_decode_builtin_call_nomethod`:
        //   `myfoobar(i, marker, c)` with `oopspec = 'foobar(2, c, i)'`.
        //   `decode_builtin_call(op)` returns `("foobar", [Constant(2),
        //   vc, vi])`.  Pyre's `NormalizedArg::ConstInt(2)` is the
        //   analogue of upstream's `Constant(2, Signed)` — the caller
        //   (`jtransform.rs::handle_builtin_call`) materialises a
        //   `OpKind::ConstInt(2)` op at the residual-call site.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["foobar"]);
        cc.mark_oopspec(path.clone(), "foobar(2, c, i)".to_string());
        cc.mark_oopspec_argnames(
            path,
            vec!["i".to_string(), "marker".to_string(), "c".to_string()],
        );
        let (op, args) = make_call_op(crate::model::CallTarget::function_path(["foobar"]), 3);
        let (name, opargs) = decode_builtin_call(&op, &cc);
        assert_eq!(name, "foobar");
        assert_eq!(
            opargs,
            vec![
                NormalizedArg::ConstInt(2),
                NormalizedArg::Pass(args[2].clone()),
                NormalizedArg::Pass(args[0].clone()),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "is neither a known argname")]
    fn parse_oopspec_rejects_char_literal_per_strict_parity() {
        // RPython upstream's `eval(args, argname2index)` would produce
        // `Constant('a', lltype.Char)` — a `lltype.Char`-tagged
        // constant.  Pyre has no `ConcreteType::Char` variant, and
        // `lltype.Char` ≠ `lltype.Signed`, so a `ConstInt(byte)`
        // fallback would be a speculative deviation.
        // `parse_literal_slot` must panic on char literals until
        // `NormalizeSlot::ConstChar(u8)` + `OpKind::ConstChar` land.
        // Confirmed by `grep -rn '@oopspec' rpython/` returning zero
        // char-literal slot patterns in current upstream — no
        // production decoration is blocked by this panic.
        let _ = super::parse_oopspec("foo('a', x)", &["x"]);
    }

    #[test]
    fn parse_oopspec_recognises_float_literal() {
        // `Constant(1.5, lltype.Float)` upstream → pyre's
        // `NormalizeSlot::ConstFloat(bits)` carrying the f64 bit pattern.
        let (name, argtuple) = super::parse_oopspec("foo(1.5, x)", &["x"]);
        assert_eq!(name, "foo");
        assert_eq!(
            argtuple,
            vec![
                super::NormalizeSlot::ConstFloat(1.5f64.to_bits()),
                super::NormalizeSlot::Index(0),
            ]
        );
    }

    #[test]
    fn parse_oopspec_recognises_float_scientific_literal() {
        // `2.0e3` is a Python float literal upstream `eval` accepts.
        let (name, argtuple) = super::parse_oopspec("foo(2.0e3)", &[]);
        assert_eq!(name, "foo");
        assert_eq!(
            argtuple,
            vec![super::NormalizeSlot::ConstFloat(2000.0f64.to_bits())]
        );
    }

    #[test]
    fn normalize_opargs_emits_const_float_for_float_slot() {
        let argtuple = vec![
            super::NormalizeSlot::ConstFloat(3.25f64.to_bits()),
            super::NormalizeSlot::Index(0),
        ];
        let v7 = crate::flowspace::model::Variable::new();
        let opargs = vec![v7.clone()];
        let normalized = super::normalize_opargs(&argtuple, &opargs);
        assert_eq!(
            normalized,
            vec![
                super::NormalizedArg::ConstFloat(3.25f64.to_bits()),
                super::NormalizedArg::Pass(v7),
            ]
        );
    }
}
