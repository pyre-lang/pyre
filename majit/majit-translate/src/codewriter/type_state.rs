//! Concrete-kind helpers — `ConcreteType` projection from
//! `LowLevelType`, `ValueType`, op-result kinds, plus
//! `apply_from_flowspace_variables` (copies rtyper-typed lltypes onto
//! the matching legacy Variables, propagating `Variable.concretetype`
//! into every alias).
//!
//! Type kinds flow through `Variable.concretetype`
//! (`rpython/flowspace/model.py Variable.__slots__ = [..., "concretetype"]`;
//! `:355 Constant.__slots__ = ["concretetype"]`) — set inline by the
//! rtyper via `RPythonTyper.setconcretetype()`
//! (`rpython/rtyper/rtyper.py v.concretetype = ...`).  Pyre
//! reproduces this through `FunctionGraph::set_concretetype_of_inline`
//! writes followed by `FunctionGraph::concretetype_of(&v)` reads
//! (which routes to the backing `Variable.concretetype` cell).  No
//! external slot table survives.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::Variable;
use crate::model::{FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// Re-export the canonical [`ConcreteType`] from [`crate::model`].
///
/// The kind enum used to live here as a side-table value type;
/// after the medium-term parity push it lives on each backing
/// `Variable.concretetype` cell carried inline on each backing
/// `Variable` referenced by the IR (mirroring upstream
/// `Variable.concretetype` line-for-line).  The alias keeps existing
/// imports working while consumers migrate to reading
/// `FunctionGraph::concretetype_of(&v)`.
pub use crate::model::ConcreteType;

/// Copy each typed Variable's `concretetype` onto the legacy graph
/// Variable it was seeded from, so subsequent
/// `FunctionGraph::concretetype_of(&v)` reads route through the
/// rtyper's `Variable.concretetype` directly.
///
/// `value_to_var` is keyed by the legacy graph Variable's object
/// identity (`legacy_var -> typed_var`).  Each legacy Variable's
/// `Rc<RefCell>` concretetype cell is shared across every reference to
/// it in the graph — `Block.inputargs`, op operands, `Link.args`,
/// `exitswitch`, `last_exception`, `last_exc_value` — so a single write
/// onto the key Variable propagates everywhere, mirroring upstream
/// `v.concretetype = T` attribute aliasing (`history.py getkind`
/// reads `v.concretetype` off the Variable).
///
/// A typed Variable whose `concretetype` is still `None` (rtyper hasn't
/// processed it yet) leaves its legacy counterpart untouched —
/// equivalent to RPython's "no `.concretetype` attribute" window before
/// `setconcretetype` runs.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn apply_from_flowspace_variables(
    value_to_var: &crate::translator::rtyper::flowspace_adapter::LegacyToTyped,
) {
    for (legacy_var, typed_var) in value_to_var.iter() {
        let Some(ct) = typed_var.concretetype() else {
            continue;
        };
        legacy_var.set_concretetype(Some(ct));
    }
}

/// Publish the Constant half of the accepted rtyper projection. Upstream
/// reads Constant.concretetype directly; the transitional MIR emitter also
/// has Variables for constant definitions and eliminated phi aliases.
/// Call only after the unchanged dual-gate comparison has accepted them.
#[expect(
    clippy::mutable_key_type,
    reason = "Variable keys use immutable identity, like RPython graph variables"
)]
pub(crate) fn apply_from_flowspace_constants(
    constants: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
    >,
) {
    for (legacy, lltype) in constants {
        legacy.set_concretetype(Some(lltype.clone()));
    }
}

/// `ValueType` → `ConcreteType` projection used by both
/// `resolve_types` (legacy graph walk) and `authoritative_result_types`
/// (post-jtransform op-result projection).
///
/// `Bool` collapses to `Signed` because RPython `BoolRepr.lowleveltype
/// = Bool` lifts to LL `Signed` for the codewriter; the legacy resolver
/// followed the same collapse and the post-jtransform projection
/// matches it.
pub(crate) fn valuetype_to_concrete(vt: &ValueType) -> ConcreteType {
    match vt {
        // `Unsigned` shares the `Signed` ConcreteType — the codewriter
        // / regalloc do not distinguish signedness (`getkind(Unsigned)
        // == 'int'`); only the rtyper picks `IntegerRepr.lowleveltype
        // = Unsigned` based on `SomeInteger.unsigned`.
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => ConcreteType::Signed,
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => ConcreteType::GcRef,
        ValueType::Float => ConcreteType::Float,
        ValueType::Void => ConcreteType::Void,
        ValueType::State
        | ValueType::Unknown
        | ValueType::Int128
        | ValueType::UInt128
        | ValueType::SingleFloat => ConcreteType::Unknown,
    }
}

/// `result_kind: char` → `ConcreteType` projection used by jtransform
/// call families (`CallElidable` / `CallResidual` / `CallMayForce` /
/// `InlineCall` / `RecursiveCall`).
pub(crate) fn kind_char_to_concrete(kind: char) -> ConcreteType {
    match kind {
        'i' => ConcreteType::Signed,
        'r' => ConcreteType::GcRef,
        'f' => ConcreteType::Float,
        'v' => ConcreteType::Void,
        _ => ConcreteType::Unknown,
    }
}

fn concrete_if_known(concrete: ConcreteType) -> Option<ConcreteType> {
    if concrete == ConcreteType::Unknown {
        None
    } else {
        Some(concrete)
    }
}

/// Per-op `ConcreteType` declared by the rewritten graph's op-result
/// fields (`result_ty` / `result_kind`).  Authoritative for op-result
/// kinds because the rewriter declares them at lowering time, so this
/// projection wins over `original` operand inferences in
/// [`merge_synth_kinds`]'s precedence chain.
pub(crate) fn authoritative_result_type_from_op(kind: &OpKind) -> Option<ConcreteType> {
    match kind {
        OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } => Some(ConcreteType::Signed),
        OpKind::ConstUInt(_) => Some(ConcreteType::Signed),
        OpKind::ConstInt128(_) | OpKind::ConstUInt128(_) => None,
        OpKind::ConstBool(_) => Some(ConcreteType::Signed),
        // `_we_are_jitted` symbolic — `Bool` concretetype folds to int
        // kind; folded to `ConstBool(true)` by `jtransform` before emit.
        OpKind::ConstSymbolic { .. } => Some(ConcreteType::Signed),
        OpKind::ConstFloat(_) => Some(ConcreteType::Float),
        OpKind::ConstStr(_) | OpKind::ConstInternedStr(_) | OpKind::ConstRefNull => {
            Some(ConcreteType::GcRef)
        }
        OpKind::Input { ty, .. } => concrete_if_known(valuetype_to_concrete(ty)),
        OpKind::FieldRead { ty, .. } | OpKind::VableFieldRead { ty, .. } => {
            concrete_if_known(valuetype_to_concrete(ty))
        }
        OpKind::ArrayRead { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. }
        // `raw_load_i/f` answers the element kind its descr names, the
        // same projection the GC array reads use.
        | OpKind::RawLoad { item_ty, .. } => {
            concrete_if_known(valuetype_to_concrete(item_ty))
        }
        OpKind::Call { result_ty, .. }
        | OpKind::IndirectCall { result_ty, .. }
        | OpKind::BinOp { result_ty, .. }
        | OpKind::UnaryOp { result_ty, .. } => concrete_if_known(valuetype_to_concrete(result_ty)),
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => {
            concrete_if_known(kind_char_to_concrete(*result_kind))
        }
        OpKind::VtableMethodPtr { .. } => Some(ConcreteType::Signed),
        // `arraylen_gc/rd>i` and `arraylen_vable/rdd>i` answer a length
        // (`jtransform.py rewrite_op_getarraysize`, `blackhole.py`
        // `bhimpl_arraylen_gc` `@arguments("cpu", "r", "d", returns="i")`).
        // Leaving this Unknown maps `getkind` to `'r'`, so a fused
        // `i < len` becomes `goto_if_not_int_lt/irL`.
        OpKind::ArrayLen { .. } | OpKind::VableArrayLen { .. } => Some(ConcreteType::Signed),
        _ => None,
    }
}

/// Walk the rewritten graph and collect every op-result that carries an
/// authoritative `ConcreteType` (per-op declaration), keyed on the backing
/// [`Variable`].  Feeds [`merge_synth_kinds`]'s `post_result` lane.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub(crate) fn authoritative_result_types(graph: &FunctionGraph) -> HashMap<Variable, ConcreteType> {
    let mut result = HashMap::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(var) = op.result.as_ref() else {
                continue;
            };
            if let Some(concrete) = authoritative_result_type_from_op(&op.kind) {
                result.insert(var.clone(), concrete);
            }
        }
    }
    result
}

// `build_value_kinds` retired — the regalloc / flatten / assemble
// pipeline now reads kinds straight off the Variable's
// `.concretetype` cell (the upstream-orthodox source).
// Per-Variable `RegKind` projections happen at the use site via
// `regalloc::perform_register_allocation`'s internal
// `concretetype_to_regkind`, matching RPython's
// `getkind(v.concretetype)` access pattern bit for bit.

/// Stamp GC `getfield`/`setfield` bases into the Ref bank before regalloc.
///
/// RPython `history.getkind`: a `Ptr` whose `TO._gckind != 'raw'` is
/// `"ref"`, so `bhimpl_getfield_gc_*` is always
/// `@arguments("cpu", "r", "d", returns="X")`.  Pyre's MIR sometimes
/// leaves a GC object pointer as `Signed` (a Rust raw pointer /
/// address-sized word).  The assembler then keys the first argcode off
/// that bank and emits the pyre-only `getfield_gc_*/id>X` form.
///
/// Walk every surviving GC field/array access (unknown owner defaults
/// to `_gckind='gc'`, matching `jtransform.py rewrite_op_getfield`'s
/// `getattr(STRUCT, '_gckind', 'gc')`) and publish `GcRef` on a
/// `Signed`/`Unknown`/`Void` base.
///
/// Raw owners (`CallControl::struct_storage_for` →
/// `is_gc_managed=false`) are left alone: a `Signed` base stays an
/// address (`getfield_raw` / `setfield_raw`), a `GcRef` base stays a
/// Ref-banked raw struct (`PyType` / `CLASSTYPE` travels through the
/// Ref bank even though its storage is not a GC object).
///
/// `ArrayRead` / `ArrayWrite` / `ArrayLen` are the GC-array family
/// (`jtransform.py` emits `getarrayitem_gc` / `arraylen_gc` only when
/// `ARRAY._gckind == 'gc'`; raw arrays go through `raw_load` /
/// `getarrayitem_raw`).  A Signed base on those ops is the same
/// mis-banked GC pointer FieldRead had.
///
/// A Signed base keeps its cell when the word is still an int at another
/// use. `make_three_lists_from_vars` stores that word in an int argument
/// list, and `emit_list_of_kind` requires the cell to still say int. A
/// later block can also read the same word as Signed: `Variable::copy`
/// gives the threaded inputarg its own concretetype cell, so stamping
/// only the field base makes `insert_renamings` emit `int_copy` from a
/// Ref register. The access reads `cast_int_to_ptr` of the word, so
/// regalloc colours a ref base and every int use keeps the Signed cell.
pub(crate) fn promote_gc_field_bases(
    graph: &mut FunctionGraph,
    callcontrol: Option<&crate::call::CallControl>,
) {
    // A `{vtable}` `method_*` slot is a function address. MIR types the
    // FieldRead as a fn-pointer Ref; leaving that cell as GcRef assembles
    // `getfield_gc_r` and `optimize_getfield` then `make_equal_to`s the
    // Ref onto the ConstInt the load produces. Stamp here, immediately
    // before the GC-base walk and SSI, so regalloc colours the slot in
    // the int bank (`jtransform.handle_getfield_typeptr` analog: the
    // slot read is a pure int load off the promoted vtable word).
    stamp_vtable_method_slots(graph);
    // Address math and the two pointer casts name an int cell. A GcRef
    // or Unknown concretetype on that cell is what emits `int_add/ii>r`,
    // `cast_ptr_to_int/r>r`, and `cast_int_to_ptr/r>r`.
    force_int_bank_cells(graph);
    let stay_int = signed_ids_that_must_stay_int(graph, callcontrol);
    let graph_name = graph.name.clone();
    let mut redirects: Vec<(usize, usize)> = Vec::new();
    for (block_index, block) in graph.blocks.iter().enumerate() {
        for (op_index, op) in block.operations.iter().enumerate() {
            let Some((base, force_gc)) = gc_access_base(&op.kind, callcontrol) else {
                continue;
            };
            if !force_gc {
                continue;
            }
            if FunctionGraph::concretetype_of(base) == ConcreteType::Signed
                && stay_int.contains(&base.id())
            {
                redirects.push((block_index, op_index));
                continue;
            }
            stamp_gc_ref_base(base, &graph_name);
        }
    }
    // Later inserts shift higher indices in the same block. Walk back so
    // each recorded index still names the access.
    for (block_index, op_index) in redirects.into_iter().rev() {
        redirect_signed_base_through_cast(graph, block_index, op_index);
    }
}

fn gc_access_base<'a>(
    kind: &'a OpKind,
    callcontrol: Option<&crate::call::CallControl>,
) -> Option<(&'a Variable, bool)> {
    match kind {
        OpKind::FieldRead { base, field, .. } | OpKind::FieldWrite { base, field, .. } => {
            Some((base, field_owner_is_gc(field, callcontrol)))
        }
        OpKind::VableFieldRead { base, .. }
        | OpKind::VableFieldWrite { base, .. }
        | OpKind::VableArrayRead { base, .. }
        | OpKind::VableArrayWrite { base, .. }
        | OpKind::VableArrayLen { base, .. } => Some((base, true)),
        // `nolength` is a raw items region (`ARRAY._gckind == 'raw'`).
        // Its base is the address integer (`getkind` → int). Promoting
        // that address to `GcRef` puts an `int_add` result in the ref
        // bank (`int_add/ii>r`), which no blackhole handler has.
        // A length-prefixed array is the GC family and stays `GcRef`.
        OpKind::ArrayRead { base, nolength, .. }
        | OpKind::ArrayWrite { base, nolength, .. }
        | OpKind::ArrayLen { base, nolength, .. } => Some((base, !nolength)),
        _ => None,
    }
}

fn gc_access_base_mut(kind: &mut OpKind) -> Option<&mut Variable> {
    match kind {
        OpKind::FieldRead { base, .. }
        | OpKind::FieldWrite { base, .. }
        | OpKind::VableFieldRead { base, .. }
        | OpKind::VableFieldWrite { base, .. }
        | OpKind::VableArrayRead { base, .. }
        | OpKind::VableArrayWrite { base, .. }
        | OpKind::VableArrayLen { base, .. }
        | OpKind::ArrayRead { base, .. }
        | OpKind::ArrayWrite { base, .. }
        | OpKind::ArrayLen { base, .. } => Some(base),
        _ => None,
    }
}

/// Variable ids sitting in an int argument list. `emit_list_of_kind`
/// asserts each of those cells is still Signed at assemble time.
fn int_argument_var_ids(graph: &FunctionGraph) -> HashSet<u64> {
    let mut ids = HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::CallResidual { args_i, .. }
                | OpKind::CallMayForce { args_i, .. }
                | OpKind::CallElidable { args_i, .. }
                | OpKind::InlineCall { args_i, .. }
                | OpKind::ConditionalCall { args_i, .. }
                | OpKind::ConditionalCallValue { args_i, .. }
                | OpKind::RecordKnownResult { args_i, .. } => {
                    extend_var_ids(&mut ids, args_i);
                }
                OpKind::RecursiveCall {
                    greens_i, reds_i, ..
                }
                | OpKind::JitMergePoint {
                    greens_i, reds_i, ..
                } => {
                    extend_var_ids(&mut ids, greens_i);
                    extend_var_ids(&mut ids, reds_i);
                }
                _ => {}
            }
        }
    }
    ids
}

fn extend_var_ids(ids: &mut HashSet<u64>, vars: &[Variable]) {
    for var in vars {
        ids.insert(var.id());
    }
}

/// Signed variable ids whose cell must not become `GcRef`.
///
/// Int-list arguments are the call shape `emit_list_of_kind` checks.
/// Every other Signed value that is not itself a GC access base is an
/// int use too. A base id that also appears as a non-base operand is
/// one of those uses: stamping it would hand the int op a Ref register.
/// The access's own base slot is not such a use. A link ties the two
/// ends of one word, so a base that flows to or from an int use keeps
/// the Signed cell as well.
fn signed_ids_that_must_stay_int(
    graph: &FunctionGraph,
    callcontrol: Option<&crate::call::CallControl>,
) -> HashSet<u64> {
    let gc_bases = signed_gc_base_ids(graph, callcontrol);
    let mut stay = int_argument_var_ids(graph);
    // An `int_add` result is an address. Promoting that cell to `GcRef`
    // because its only use is a GC field base emits `int_add/ii>r`.
    // Keep the cell Signed; the access reads `cast_int_to_ptr`.
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(result) = &op.result else {
                continue;
            };
            if FunctionGraph::concretetype_of(result) != ConcreteType::Signed {
                continue;
            }
            if integer_address_binop(&op.kind) {
                stay.insert(result.id());
            }
            // `cast_ptr_to_int` returns an int (`bhimpl_cast_ptr_to_int`).
            // A GC field use of that cell must read `cast_int_to_ptr`,
            // not retype the cast result into the ref bank.
            if let OpKind::UnaryOp { op, .. } = &op.kind
                && op == "cast_ptr_to_int"
            {
                stay.insert(result.id());
            }
        }
    }
    let mut ties: Vec<(Variable, Variable)> = Vec::new();
    for block_index in 0..graph.blocks.len() {
        match &graph.blocks[block_index].exitswitch {
            Some(crate::model::ExitSwitch::Value(cond)) => {
                stay.insert(cond.id());
            }
            Some(crate::model::ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    stay.insert(arg.id());
                }
            }
            _ => {}
        }
        for op in &graph.blocks[block_index].operations {
            if let Some(result) = &op.result {
                note_signed_non_base(result, &gc_bases, &mut stay);
            }
            // `op_variable_refs` yields the base first. That slot is the
            // access. A later occurrence, or the same id on another op,
            // is an int operand and has to keep the Signed cell.
            let mut skip_base_slot = match gc_access_base(&op.kind, callcontrol) {
                Some((base, true)) => Some(base.id()),
                _ => None,
            };
            for var in crate::inline::op_variable_refs(&op.kind) {
                if skip_base_slot == Some(var.id()) {
                    skip_base_slot = None;
                    continue;
                }
                if FunctionGraph::concretetype_of(&var) == ConcreteType::Signed {
                    stay.insert(var.id());
                }
            }
        }
        let exit_count = graph.blocks[block_index].exits.len();
        for exit_index in 0..exit_count {
            let target = graph.blocks[block_index].exits[exit_index].target;
            let arg_count = graph.blocks[block_index].exits[exit_index].args.len();
            for arg_index in 0..arg_count {
                let Some(src) = graph.blocks[block_index].exits[exit_index].args[arg_index]
                    .as_variable()
                    .cloned()
                else {
                    continue;
                };
                let Some(input) = graph.blocks[target.0].inputargs.get(arg_index).cloned() else {
                    continue;
                };
                note_signed_non_base(&src, &gc_bases, &mut stay);
                note_signed_non_base(&input, &gc_bases, &mut stay);
                ties.push((src, input));
            }
        }
    }
    let mut adj: HashMap<u64, Vec<Variable>> = HashMap::new();
    for (src, input) in ties {
        adj.entry(src.id()).or_default().push(input.clone());
        adj.entry(input.id()).or_default().push(src);
    }
    let mut queue: Vec<u64> = stay.iter().copied().collect();
    let mut index = 0;
    while index < queue.len() {
        let id = queue[index];
        index += 1;
        let Some(partners) = adj.get(&id) else {
            continue;
        };
        for partner in partners {
            if stay.contains(&partner.id()) {
                continue;
            }
            if FunctionGraph::concretetype_of(partner) != ConcreteType::Signed {
                continue;
            }
            stay.insert(partner.id());
            queue.push(partner.id());
        }
    }
    stay
}

fn signed_gc_base_ids(
    graph: &FunctionGraph,
    callcontrol: Option<&crate::call::CallControl>,
) -> HashSet<u64> {
    let mut ids = HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let Some((base, force_gc)) = gc_access_base(&op.kind, callcontrol) else {
                continue;
            };
            if force_gc && FunctionGraph::concretetype_of(base) == ConcreteType::Signed {
                ids.insert(base.id());
            }
        }
    }
    ids
}

/// A `{vtable}` `method_*` FieldRead is a function address. MIR types it
/// as a fn-pointer Ref; this pass puts the result in the int bank so
/// the assembler emits `getfield_raw_i`.
fn stamp_vtable_method_slots(graph: &mut FunctionGraph) {
    for block in &mut graph.blocks {
        for op in &mut block.operations {
            {
                let OpKind::FieldRead {
                    field, ty, pure, ..
                } = &mut op.kind
                else {
                    continue;
                };
                if !field.is_vtable_method_slot() {
                    continue;
                }
                *ty = ValueType::Int;
                *pure = true;
            }
            if let Some(result) = op.result.as_ref() {
                FunctionGraph::set_concretetype_of_inline(result, ConcreteType::Signed);
            }
        }
    }
}

/// Put address math and pointer casts back in the banks their opnames
/// declare. Only a `GcRef` or `Unknown` cell is rewritten: a `Void` cell
/// has no register, and stamping it Signed makes liveness ask the int
/// allocator for a color it never assigned.
pub(crate) fn force_int_bank_cells(graph: &FunctionGraph) {
    fn stamp_signed(var: &crate::flowspace::model::Variable) {
        let ty = FunctionGraph::concretetype_of(var);
        if ty == ConcreteType::GcRef || ty == ConcreteType::Unknown {
            FunctionGraph::set_concretetype_of_inline(var, ConcreteType::Signed);
        }
    }
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                kind if integer_address_binop(kind) => {
                    if let Some(result) = &op.result {
                        stamp_signed(result);
                    }
                }
                OpKind::UnaryOp { op: name, .. } if name == "cast_ptr_to_int" => {
                    if let Some(result) = &op.result {
                        stamp_signed(result);
                    }
                }
                OpKind::UnaryOp {
                    op: name, operand, ..
                } if name == "cast_int_to_ptr" => {
                    stamp_signed(operand);
                }
                _ => {}
            }
        }
        // `optimize_goto_if_not` names `int_is_zero` while the operand is
        // still Signed. `align_gc_link_args` can later pull that cell into
        // the ref bank because the same word is a GC phi. The fused key is
        // then `goto_if_not_int_is_zero/rL`, which has no handler
        // (`bhimpl_goto_if_not_int_is_zero` reads the int bank).
        if let Some(crate::model::ExitSwitch::Fused { opname, args }) = &block.exitswitch
            && opname.starts_with("int_")
        {
            for arg in args {
                stamp_signed(arg);
            }
        }
    }
}

fn integer_address_binop(kind: &OpKind) -> bool {
    let OpKind::BinOp { op, lhs, rhs, .. } = kind else {
        return false;
    };
    // `result_ty` can still be `Ref` while both words are already in the
    // int bank. The assembler prefixes `add` to `int_add` anyway, and a
    // GcRef result is `int_add/ii>r`.
    let int_word = |var: &crate::flowspace::model::Variable| {
        matches!(
            FunctionGraph::concretetype_of(var),
            ConcreteType::Signed | ConcreteType::Unknown
        )
    };
    int_word(lhs)
        && int_word(rhs)
        && matches!(
            op.as_str(),
            "add" | "sub" | "int_add" | "int_sub" | "mul" | "int_mul"
        )
}

fn note_signed_non_base(var: &Variable, gc_bases: &HashSet<u64>, stay: &mut HashSet<u64>) {
    if gc_bases.contains(&var.id()) {
        return;
    }
    if FunctionGraph::concretetype_of(var) == ConcreteType::Signed {
        stay.insert(var.id());
    }
}

fn redirect_signed_base_through_cast(
    graph: &mut FunctionGraph,
    block_index: usize,
    op_index: usize,
) {
    let original = gc_access_base_mut(&mut graph.blocks[block_index].operations[op_index].kind)
        .expect("redirect target is a GC field or array access")
        .clone();
    let cast_result = graph.alloc_value_var_with_type(ConcreteType::GcRef);
    *gc_access_base_mut(&mut graph.blocks[block_index].operations[op_index].kind)
        .expect("redirect target is a GC field or array access") = cast_result.clone();
    graph.blocks[block_index].operations.insert(
        op_index,
        SpaceOperation {
            result: Some(cast_result),
            kind: OpKind::UnaryOp {
                op: "cast_int_to_ptr".into(),
                operand: original,
                result_ty: ValueType::Ref(None),
            },
        },
    );
}

/// Put each `insert_renamings` argument in the inputarg's bank.
///
/// `insert_renamings` emits `%s_copy` keyed on the destination kind. A
/// GcRef source copied into a Signed inputarg is `int_copy` of a Ref
/// register, which `encode_regorconst_source` rejects. `ssa_to_ssi`
/// gives the inputarg a fresh concretetype cell, and
/// `promote_gc_field_bases` can stamp one end afterwards, so the two
/// banks diverge. `cast_ptr_to_int` / `cast_int_to_ptr` deliver the
/// destination bank.
///
/// Links that `make_link` turns into `make_return` are left alone: that
/// path colours the source, and a cast there would change the return
/// opcode. `last_exception` and `last_exc_value` are skipped the same
/// way `insert_renamings` skips them. A raising block keeps its last
/// op last, unless that op defines the cast operand.
pub(crate) fn coerce_cross_bank_links(graph: &mut FunctionGraph) {
    let mut pending: Vec<Vec<PendingCast>> = Vec::with_capacity(graph.blocks.len());
    for block_index in 0..graph.blocks.len() {
        let mut casts = Vec::new();
        let exit_count = graph.blocks[block_index].exits.len();
        for exit_index in 0..exit_count {
            if !link_reaches_insert_renamings(graph, block_index, exit_index) {
                continue;
            }
            let target = graph.blocks[block_index].exits[exit_index].target;
            let arg_count = graph.blocks[block_index].exits[exit_index].args.len();
            for arg_index in 0..arg_count {
                let arg = graph.blocks[block_index].exits[exit_index].args[arg_index].clone();
                let skipped = {
                    let link = &graph.blocks[block_index].exits[exit_index];
                    Some(&arg) == link.last_exception.as_ref()
                        || Some(&arg) == link.last_exc_value.as_ref()
                };
                if skipped {
                    continue;
                }
                let Some(src) = arg.as_variable().cloned() else {
                    continue;
                };
                let Some(input) = graph.blocks[target.0].inputargs.get(arg_index).cloned() else {
                    continue;
                };
                let Some((op_name, result_ty, result_concrete)) = cross_bank_cast(
                    FunctionGraph::concretetype_of(&src),
                    FunctionGraph::concretetype_of(&input),
                ) else {
                    continue;
                };
                casts.push(PendingCast {
                    exit_index,
                    arg_index,
                    operand: src,
                    op_name,
                    result_ty,
                    result_concrete,
                });
            }
        }
        pending.push(casts);
    }

    for (block_index, casts) in pending.into_iter().enumerate() {
        if casts.is_empty() {
            continue;
        }
        let mut built = Vec::with_capacity(casts.len());
        for (seq, cast) in casts.into_iter().enumerate() {
            let at = cast_insertion_index(&graph.blocks[block_index], &cast.operand);
            let result = graph.alloc_value_var_with_type(cast.result_concrete);
            built.push(BuiltCast {
                seq,
                at,
                exit_index: cast.exit_index,
                arg_index: cast.arg_index,
                result: result.clone(),
                op: SpaceOperation {
                    result: Some(result),
                    kind: OpKind::UnaryOp {
                        op: cast.op_name.into(),
                        operand: cast.operand,
                        result_ty: cast.result_ty,
                    },
                },
            });
        }
        for cast in &built {
            graph.blocks[block_index].exits[cast.exit_index].args[cast.arg_index] =
                LinkArg::Value(cast.result.clone());
        }
        // Higher indices first, and within one index the later cast first,
        // so each recorded index still names the same gap and the original
        // order survives.
        built.sort_by(|left, right| right.at.cmp(&left.at).then(right.seq.cmp(&left.seq)));
        for cast in built {
            graph.blocks[block_index]
                .operations
                .insert(cast.at, cast.op);
        }
    }
}

struct PendingCast {
    exit_index: usize,
    arg_index: usize,
    operand: Variable,
    op_name: &'static str,
    result_ty: ValueType,
    result_concrete: ConcreteType,
}

struct BuiltCast {
    seq: usize,
    at: usize,
    exit_index: usize,
    arg_index: usize,
    result: Variable,
    op: SpaceOperation,
}

fn cross_bank_cast(
    src: ConcreteType,
    dst: ConcreteType,
) -> Option<(&'static str, ValueType, ConcreteType)> {
    match (src, dst) {
        (ConcreteType::GcRef, ConcreteType::Signed) => {
            Some(("cast_ptr_to_int", ValueType::Int, ConcreteType::Signed))
        }
        (ConcreteType::Signed, ConcreteType::GcRef) => {
            Some(("cast_int_to_ptr", ValueType::Ref(None), ConcreteType::GcRef))
        }
        _ => None,
    }
}

/// `make_link` calls `insert_renamings` unless the target is final and
/// the link does not carry `last_exception` / `last_exc_value` in its
/// args. That other path is `make_return`.
fn link_reaches_insert_renamings(
    graph: &FunctionGraph,
    block_index: usize,
    exit_index: usize,
) -> bool {
    let target = graph.blocks[block_index].exits[exit_index].target;
    if !graph.blocks[target.0].exits.is_empty() {
        return true;
    }
    let link = &graph.blocks[block_index].exits[exit_index];
    link.last_exception
        .as_ref()
        .is_some_and(|arg| link.args.contains(arg))
        || link
            .last_exc_value
            .as_ref()
            .is_some_and(|arg| link.args.contains(arg))
}

/// Index at which a cast of `operand` still dominates its use and, in a
/// raising block, leaves the raising op last.
fn cast_insertion_index(block: &crate::model::Block, operand: &Variable) -> usize {
    let ops = &block.operations;
    let mut index = ops.len();
    if block.canraise()
        && let Some(last) = ops.last()
        && !last
            .result
            .as_ref()
            .is_some_and(|result| result.id() == operand.id())
    {
        index = ops.len() - 1;
    }
    for (op_index, op) in ops.iter().enumerate() {
        if op
            .result
            .as_ref()
            .is_some_and(|result| result.id() == operand.id())
        {
            index = index.max(op_index + 1);
        }
    }
    index
}

/// A merge mints a fresh inputarg. Promoting that phi, or `SSA_to_SSI`
/// threading one afterwards, leaves the predecessor's own variable
/// `Signed`, and `flatten.py` `insert_renamings` then copies an int into
/// a ref. The value passed in that slot is the same pointer.
///
/// One variable has one kind. Stamping the shared predecessor through the
/// GC successor leaves the other successor's phi `Signed`, and the copy
/// the other way (`GcRef` into `Signed`) aborts too. Either side being
/// `GcRef` pulls the pair, then the next pass sees the other edge.
pub(crate) fn align_gc_link_args(graph: &mut FunctionGraph) {
    // Signed/Unknown are raw-address leftovers of a GC pointer.
    // A Void returnblock inputarg of a `Result<(), PyError>` callee
    // (`return_type = "()"`) is the unit leftover / `ConstNone`:
    // promoting it to GcRef makes `FUNC.RESULT=v` disagree with cfg=r.
    // Any other Void phi — including the return of a non-void function
    // — is the same pointer merged with a unit predecessor and has to
    // leave the void bank so later ops can colour it.
    let returnblock = graph.returnblock;
    let protect_unit_return = graph.return_type.as_deref() == Some("()");
    let soft = |ty: &ConcreteType| matches!(ty, ConcreteType::Signed | ConcreteType::Unknown);
    let soft_phi = |ty: &ConcreteType, target: crate::model::BlockId| {
        soft(ty) || (*ty == ConcreteType::Void && !(protect_unit_return && target == returnblock))
    };
    loop {
        let mut changed = false;
        for block in &graph.blocks {
            for link in &block.exits {
                let Some(target) = graph.blocks.iter().find(|b| b.id == link.target) else {
                    continue;
                };
                for (arg, input) in link.args.iter().zip(target.inputargs.iter()) {
                    let Some(var) = arg.as_variable() else {
                        continue;
                    };
                    let arg_ty = FunctionGraph::concretetype_of(var);
                    let in_ty = FunctionGraph::concretetype_of(input);
                    if in_ty == ConcreteType::GcRef && soft(&arg_ty) {
                        FunctionGraph::set_concretetype_of_inline(var, ConcreteType::GcRef);
                        changed = true;
                    } else if arg_ty == ConcreteType::GcRef && soft_phi(&in_ty, link.target) {
                        FunctionGraph::set_concretetype_of_inline(input, ConcreteType::GcRef);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// `make_three_lists` ran before the final `getkind`. Move each operand
/// into the list that matches its concretetype, and flip the positional
/// `arg_classes` char so the call descr stays aligned with the lists.
pub(crate) fn rebucket_kind_lists(graph: &mut FunctionGraph) {
    for block in &mut graph.blocks {
        for op in &mut block.operations {
            match &mut op.kind {
                OpKind::CallResidual {
                    args_i,
                    args_r,
                    descriptor,
                    ..
                }
                | OpKind::CallElidable {
                    args_i,
                    args_r,
                    descriptor,
                    ..
                }
                | OpKind::CallMayForce {
                    args_i,
                    args_r,
                    descriptor,
                    ..
                }
                | OpKind::ConditionalCall {
                    args_i,
                    args_r,
                    descriptor,
                    ..
                }
                | OpKind::ConditionalCallValue {
                    args_i,
                    args_r,
                    descriptor,
                    ..
                }
                | OpKind::RecordKnownResult {
                    args_i,
                    args_r,
                    descriptor,
                    ..
                } => rebucket_args(args_i, args_r, Some(&mut descriptor.arg_classes)),
                OpKind::InlineCall {
                    args_i,
                    args_r,
                    arg_classes,
                    ..
                } => {
                    rebucket_args(args_i, args_r, Some(arg_classes));
                }
                OpKind::RecursiveCall {
                    greens_i,
                    greens_r,
                    green_classes,
                    reds_i,
                    reds_r,
                    red_classes,
                    ..
                }
                | OpKind::JitMergePoint {
                    greens_i,
                    greens_r,
                    green_classes,
                    reds_i,
                    reds_r,
                    red_classes,
                    ..
                } => {
                    rebucket_args(greens_i, greens_r, Some(green_classes));
                    rebucket_args(reds_i, reds_r, Some(red_classes));
                }
                _ => {}
            }
        }
    }
}

fn rebucket_args(
    args_i: &mut Vec<crate::flowspace::model::Variable>,
    args_r: &mut Vec<crate::flowspace::model::Variable>,
    classes: Option<&mut String>,
) {
    let has_void = args_i
        .iter()
        .chain(args_r.iter())
        .any(|var| FunctionGraph::concretetype_of(var) == ConcreteType::Void);
    let mismatched = args_i
        .iter()
        .any(|var| FunctionGraph::concretetype_of(var) == ConcreteType::GcRef)
        || args_r
            .iter()
            .any(|var| FunctionGraph::concretetype_of(var) == ConcreteType::Signed);
    if !mismatched && !has_void {
        return;
    }
    let old_i = std::mem::take(args_i);
    let old_r = std::mem::take(args_r);
    let mut ii = 0;
    let mut ri = 0;
    let mut new_classes = String::new();
    let class_src = classes.as_ref().map(|s| (*s).clone()).unwrap_or_default();
    let place = |var: crate::flowspace::model::Variable,
                 args_i: &mut Vec<crate::flowspace::model::Variable>,
                 args_r: &mut Vec<crate::flowspace::model::Variable>,
                 classes: &mut String| {
        match FunctionGraph::concretetype_of(&var) {
            ConcreteType::GcRef => {
                classes.push('r');
                args_r.push(var);
            }
            // `make_three_lists` drops Void (`getkind` `'v'`). A cell
            // that settled on Void after that split must leave the lists
            // the same way, or `emit_list_of_kind` asks regalloc for a
            // color Void never received.
            ConcreteType::Void => {}
            _ => {
                classes.push('i');
                args_i.push(var);
            }
        }
    };
    if classes.is_some() && !class_src.is_empty() {
        for c in class_src.chars() {
            match c {
                'i' if ii < old_i.len() => {
                    let var = old_i[ii].clone();
                    ii += 1;
                    place(var, args_i, args_r, &mut new_classes);
                }
                'r' if ri < old_r.len() => {
                    let var = old_r[ri].clone();
                    ri += 1;
                    place(var, args_i, args_r, &mut new_classes);
                }
                other => new_classes.push(other),
            }
        }
    }
    while ii < old_i.len() {
        let var = old_i[ii].clone();
        ii += 1;
        place(var, args_i, args_r, &mut new_classes);
    }
    while ri < old_r.len() {
        let var = old_r[ri].clone();
        ri += 1;
        place(var, args_i, args_r, &mut new_classes);
    }
    if let Some(classes) = classes
        && !class_src.is_empty()
    {
        *classes = new_classes;
    }
}

/// `goto_if_not_int_*` reads the int bank (`blackhole.py`
/// `bhhandler_goto_if_not_ii`). A fused operand whose cell is `GcRef`
/// is `cast_ptr_to_int` (`rbuiltin.py rtype_cast_ptr_to_int`) and the
/// guard reads that int.
pub(crate) fn reint_fused_int_ref_operands(graph: &mut FunctionGraph) {
    for block_index in 0..graph.blocks.len() {
        let Some(crate::model::ExitSwitch::Fused { opname, args }) =
            graph.blocks[block_index].exitswitch.clone()
        else {
            continue;
        };
        if !opname.starts_with("int_") {
            continue;
        }
        let mut new_args = args;
        let mut changed = false;
        for arg in &mut new_args {
            if FunctionGraph::concretetype_of(arg) != ConcreteType::GcRef {
                continue;
            }
            let casted = graph.alloc_value_var();
            FunctionGraph::set_concretetype_of_inline(&casted, ConcreteType::Signed);
            graph.blocks[block_index].operations.push(SpaceOperation {
                result: Some(casted.clone()),
                kind: OpKind::UnaryOp {
                    op: "cast_ptr_to_int".into(),
                    operand: arg.clone(),
                    result_ty: ValueType::Int,
                },
            });
            *arg = casted;
            changed = true;
        }
        if changed {
            graph.blocks[block_index].exitswitch = Some(crate::model::ExitSwitch::Fused {
                opname,
                args: new_args,
            });
        }
    }
}

fn stamp_gc_ref_base(base: &crate::flowspace::model::Variable, graph_name: &str) {
    match FunctionGraph::concretetype_of(base) {
        ConcreteType::GcRef => {}
        ConcreteType::Float => panic!(
            "GC field/array base {base:?} has Float concretetype — \
             history.getkind(Ptr(GC)) is 'ref' (graph {graph_name})"
        ),
        ConcreteType::Signed | ConcreteType::Unknown | ConcreteType::Void => {
            FunctionGraph::set_concretetype_of_inline(base, ConcreteType::GcRef);
        }
    }
}

pub(crate) fn field_owner_is_gc(
    field: &crate::model::FieldDescriptor,
    callcontrol: Option<&crate::call::CallControl>,
) -> bool {
    // A rustc vtable is a static array of function addresses, not a GC
    // object. `getattr(STRUCT, '_gckind', 'gc')` would default the
    // unregistered `{vtable}` owner to gc and `promote_gc_field_bases`
    // would cast the FatLen int word through `cast_int_to_ptr`, assembling
    // `getfield_gc_r` of `method_*`.
    if field.is_vtable_method_slot() {
        return false;
    }
    // The front records `Struct._gckind` on the descriptor. A Raw owner
    // stays a raw address even when this `CallControl` has no layout
    // registered for the leaf name (`getattr(..., '_gckind', 'gc')` is
    // the fallback only when the producer did not record a kind).
    if let Some(declared_gc) = field.owner_declared_gc {
        return declared_gc;
    }
    let Some(owner) = field.owner_root.as_deref() else {
        return true;
    };
    let Some(cc) = callcontrol else {
        return true;
    };
    if let Some((is_gc, _)) = cc.struct_storage_for(owner) {
        return is_gc;
    }
    match cc.declared_gckind_for(owner) {
        Some(crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw) => false,
        Some(_) => true,
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FieldDescriptor, OpKind, SpaceOperation, ValueType};

    fn push_input(
        graph: &mut FunctionGraph,
        name: &str,
        ty: ValueType,
    ) -> crate::flowspace::model::Variable {
        let var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: name.into(),
                    ty,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(graph.startblock, var.clone());
        var
    }

    #[test]
    fn reint_fused_int_lt_casts_a_ref_operand() {
        use crate::model::{ExitCase, ExitSwitch, Link};
        let mut graph = FunctionGraph::new("fused_ir_lt");
        let lhs = push_input(&mut graph, "p", ValueType::Ref(None));
        let rhs = push_input(&mut graph, "n", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&lhs, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&rhs, ConcreteType::Signed);
        let if_false = graph.create_block();
        let if_true = graph.create_block();
        graph.set_return(if_false, None);
        graph.set_return(if_true, None);
        graph.block_mut(graph.startblock).exitswitch = Some(ExitSwitch::Fused {
            opname: "int_lt".into(),
            args: vec![lhs.clone(), rhs.clone()],
        });
        graph.block_mut(graph.startblock).exits = vec![
            Link::new_mixed(vec![], if_false, Some(ExitCase::Bool(false))),
            Link::new_mixed(vec![], if_true, Some(ExitCase::Bool(true))),
        ];

        reint_fused_int_ref_operands(&mut graph);

        let ExitSwitch::Fused { opname, args } =
            graph.block(graph.startblock).exitswitch.clone().unwrap()
        else {
            panic!("fused guard");
        };
        assert_eq!(opname, "int_lt");
        assert_eq!(args[1], rhs);
        assert_ne!(args[0], lhs);
        assert_eq!(
            FunctionGraph::concretetype_of(&args[0]),
            ConcreteType::Signed
        );
        assert!(graph.block(graph.startblock).operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::UnaryOp { op, operand, .. }
                    if op == "cast_ptr_to_int" && *operand == lhs
            )
        }));
    }

    #[test]
    fn promote_gc_field_bases_lifts_a_signed_gc_owner_into_the_ref_bank() {
        let mut graph = FunctionGraph::new("int_base_getfield");
        let base = push_input(&mut graph, "obj", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));
        FunctionGraph::set_concretetype_of_inline(&base, ConcreteType::Signed);

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::GcRef,
            "a GC FieldRead base that arrived as Signed must be published as GcRef"
        );
    }

    #[test]
    fn promote_gc_field_bases_lifts_a_signed_array_read_base() {
        let mut graph = FunctionGraph::new("int_base_getarrayitem");
        let base = push_input(&mut graph, "arr", ValueType::Int);
        let index = push_input(&mut graph, "i", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayRead {
                    base: base.clone(),
                    index,
                    item_ty: ValueType::Int,
                    array_type_id: None,
                    nolength: false,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));
        FunctionGraph::set_concretetype_of_inline(&base, ConcreteType::Signed);

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::GcRef,
            "a GC ArrayRead base that arrived as Signed must be published as GcRef"
        );
    }

    #[test]
    fn promote_gc_field_bases_leaves_a_headerless_array_address_signed() {
        let (array_type_id, nolength) =
            crate::front::mir::fixed_array_index_identity(false, "&[u32]");
        assert!(nolength, "a [u32] item run has no length header");
        assert!(crate::front::typestr::nolength_from_array_type_id(
            array_type_id.as_deref()
        ));

        let mut graph = FunctionGraph::new("raw_slice_address");
        let addr = push_input(&mut graph, "addr", ValueType::Int);
        let index = push_input(&mut graph, "i", ValueType::Int);
        let value = push_input(&mut graph, "v", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&addr, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&index, ConcreteType::Signed);
        graph
            .block_mut(graph.startblock)
            .operations
            .push(crate::model::SpaceOperation {
                result: None,
                kind: OpKind::ArrayWrite {
                    base: addr.clone(),
                    index,
                    value: crate::model::LinkArg::Value(value),
                    item_ty: ValueType::Int,
                    array_type_id,
                    nolength,
                },
            });

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&addr),
            ConcreteType::Signed,
            "nolength raw slice base stays an int"
        );
    }

    #[test]
    fn promote_gc_field_bases_casts_an_int_add_used_only_as_a_field_base() {
        let mut graph = FunctionGraph::new("int_add_field_base");
        let lhs = push_input(&mut graph, "p", ValueType::Int);
        let rhs = push_input(&mut graph, "n", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&lhs, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs, ConcreteType::Signed);
        let sum = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        FunctionGraph::set_concretetype_of_inline(&sum, ConcreteType::Signed);
        let read = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: sum.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(read));

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&sum),
            ConcreteType::Signed,
            "int_add stays in the int bank"
        );
        let ops = &graph.block(graph.startblock).operations;
        assert!(ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::UnaryOp { op, operand, .. }
                if op == "cast_int_to_ptr" && operand.id() == sum.id()
        )));
    }

    #[test]
    fn promote_gc_field_bases_unstamps_a_gcref_int_add_result() {
        let mut graph = FunctionGraph::new("gcref_int_add_field_base");
        let lhs = push_input(&mut graph, "p", ValueType::Int);
        let rhs = push_input(&mut graph, "n", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&lhs, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs, ConcreteType::Signed);
        let sum = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        FunctionGraph::set_concretetype_of_inline(&sum, ConcreteType::GcRef);
        let read = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: sum.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(read));

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(FunctionGraph::concretetype_of(&sum), ConcreteType::Signed);
        let ops = &graph.block(graph.startblock).operations;
        assert!(ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::UnaryOp { op, operand, .. }
                if op == "cast_int_to_ptr" && operand.id() == sum.id()
        )));
    }

    #[test]
    fn force_int_bank_cells_keeps_a_fused_int_is_zero_operand_signed() {
        let mut graph = FunctionGraph::new("fused_int_is_zero");
        let operand = push_input(&mut graph, "n", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&operand, ConcreteType::GcRef);
        graph.block_mut(graph.startblock).exitswitch = Some(crate::model::ExitSwitch::Fused {
            opname: "int_is_zero".into(),
            args: vec![operand.clone()],
        });

        force_int_bank_cells(&graph);

        assert_eq!(
            FunctionGraph::concretetype_of(&operand),
            ConcreteType::Signed
        );
    }

    #[test]
    fn promote_gc_field_bases_lifts_an_unknown_base() {
        let mut graph = FunctionGraph::new("unknown_base_getfield");
        let base = push_input(&mut graph, "obj", ValueType::Ref(None));
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));
        assert_eq!(FunctionGraph::concretetype_of(&base), ConcreteType::Unknown);

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::GcRef,
            "the post-rewrite pass still publishes an untyped GC field base"
        );
    }

    #[test]
    fn promote_gc_field_bases_casts_a_signed_base_that_is_an_int_call_argument() {
        let mut graph = FunctionGraph::new("int_arg_and_field");
        let base = push_input(&mut graph, "obj", ValueType::Int);
        let other = push_input(&mut graph, "n", ValueType::Ref(None));
        FunctionGraph::set_concretetype_of_inline(&base, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&other, ConcreteType::GcRef);
        let read = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(read));
        graph
            .block_mut(graph.startblock)
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::InlineCall {
                    jitcode: crate::jitcode::JitCodeHandle::new(std::sync::Arc::new(
                        crate::jitcode::JitCode::new("callee"),
                    )),
                    args_i: vec![base.clone()],
                    args_r: vec![other],
                    args_f: Vec::new(),
                    result_kind: 'r',
                    arg_classes: String::new(),
                },
            });

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::Signed,
            "an int-list argument stays Signed so emit_list_of_kind still matches"
        );
        let ops = &graph.block(graph.startblock).operations;
        let cast_at = ops
            .iter()
            .position(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, operand, .. }
                        if op == "cast_int_to_ptr" && operand.id() == base.id()
                )
            })
            .expect("cast_int_to_ptr of the signed argument");
        let field_at = ops
            .iter()
            .position(|op| matches!(&op.kind, OpKind::FieldRead { .. }))
            .expect("field read");
        assert!(cast_at < field_at, "the cast dominates the field read");
        let OpKind::FieldRead {
            base: field_base, ..
        } = &ops[field_at].kind
        else {
            unreachable!("field read");
        };
        assert_ne!(field_base.id(), base.id());
        assert_eq!(
            FunctionGraph::concretetype_of(field_base),
            ConcreteType::GcRef
        );
        assert!(ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::InlineCall { args_i, .. } if args_i.iter().any(|arg| arg.id() == base.id())
        )));
    }

    #[test]
    fn promote_gc_field_bases_casts_when_a_signed_base_flows_to_an_int_successor() {
        let mut graph = FunctionGraph::new("base_flows_to_int");
        let base = push_input(&mut graph, "p", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&base, ConcreteType::Signed);
        let read = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        FunctionGraph::set_concretetype_of_inline(&read, ConcreteType::Signed);
        let (next, args) = graph.create_block_with_arg_vars(1);
        let succ = args[0].clone();
        FunctionGraph::set_concretetype_of_inline(&succ, ConcreteType::Signed);
        graph.set_return(next, Some(succ.clone()));
        graph.set_goto(graph.startblock, next, vec![base.clone()]);

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::Signed,
            "the predecessor cell stays Signed so the link copy is int-to-int"
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&succ),
            ConcreteType::Signed,
            "the successor that is not a field base stays Signed"
        );
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, operand, .. }
                    if op == "cast_int_to_ptr" && operand.id() == base.id()
            )),
            "the field access reads cast_int_to_ptr of the signed word"
        );
    }

    #[test]
    fn promote_gc_field_bases_casts_a_signed_base_used_as_an_int_operand() {
        let mut graph = FunctionGraph::new("base_and_int_add");
        let base = push_input(&mut graph, "p", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&base, ConcreteType::Signed);
        let read = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        FunctionGraph::set_concretetype_of_inline(&read, ConcreteType::Signed);
        let sum = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: base.clone(),
                    rhs: read.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        FunctionGraph::set_concretetype_of_inline(&sum, ConcreteType::Signed);
        graph.set_return(graph.startblock, Some(sum));

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::Signed,
            "an int_add operand stays Signed"
        );
        let ops = &graph.block(graph.startblock).operations;
        let cast_at = ops
            .iter()
            .position(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, operand, .. }
                        if op == "cast_int_to_ptr" && operand.id() == base.id()
                )
            })
            .expect("cast_int_to_ptr of the signed base");
        let field_at = ops
            .iter()
            .position(|op| matches!(&op.kind, OpKind::FieldRead { .. }))
            .expect("field read");
        assert!(cast_at < field_at, "the cast dominates the field read");
        let OpKind::FieldRead {
            base: field_base, ..
        } = &ops[field_at].kind
        else {
            unreachable!("field read");
        };
        assert_ne!(field_base.id(), base.id());
        assert_eq!(
            FunctionGraph::concretetype_of(field_base),
            ConcreteType::GcRef
        );
        assert!(ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::BinOp { lhs, .. } if lhs.id() == base.id()
        )));
    }

    #[test]
    fn align_gc_link_args_promotes_the_other_successor_phi() {
        let mut graph = FunctionGraph::new("gc_phi_two_successors");
        let shared = push_input(&mut graph, "p", ValueType::Int);
        let cond = push_input(&mut graph, "c", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&shared, ConcreteType::Signed);
        let (gc_bb, gc_inputs) = graph.create_block_with_arg_vars(1);
        let (other_bb, other_inputs) = graph.create_block_with_arg_vars(1);
        FunctionGraph::set_concretetype_of_inline(&gc_inputs[0], ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&other_inputs[0], ConcreteType::Signed);
        graph.set_return(gc_bb, Some(gc_inputs[0].clone()));
        graph.set_return(other_bb, Some(other_inputs[0].clone()));
        graph.set_branch(
            graph.startblock,
            cond,
            gc_bb,
            vec![shared.clone()],
            other_bb,
            vec![shared.clone()],
        );

        align_gc_link_args(&mut graph);

        assert_eq!(FunctionGraph::concretetype_of(&shared), ConcreteType::GcRef);
        assert_eq!(
            FunctionGraph::concretetype_of(&other_inputs[0]),
            ConcreteType::GcRef,
            "the other successor phi of the same pointer must leave the int bank"
        );
    }

    #[test]
    fn align_gc_link_args_leaves_a_void_return_slot() {
        let mut graph = FunctionGraph::new("void_unit_leftover");
        let shared = push_input(&mut graph, "p", ValueType::Int);
        let cond = push_input(&mut graph, "c", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&shared, ConcreteType::GcRef);
        let (gc_bb, gc_inputs) = graph.create_block_with_arg_vars(1);
        let (void_bb, void_inputs) = graph.create_block_with_arg_vars(1);
        FunctionGraph::set_concretetype_of_inline(&gc_inputs[0], ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&void_inputs[0], ConcreteType::Void);
        graph.return_type = Some("()".into());
        let ret_arg = graph.block(graph.returnblock).inputargs[0].clone();
        FunctionGraph::set_concretetype_of_inline(&ret_arg, ConcreteType::Void);
        graph.set_return(gc_bb, Some(gc_inputs[0].clone()));
        graph.set_return(void_bb, Some(void_inputs[0].clone()));
        graph.set_branch(
            graph.startblock,
            cond,
            gc_bb,
            vec![shared.clone()],
            void_bb,
            vec![shared.clone()],
        );

        align_gc_link_args(&mut graph);

        assert_eq!(
            FunctionGraph::concretetype_of(&ret_arg),
            ConcreteType::Void,
            "a Void unit leftover must not become GcRef"
        );
    }

    #[test]
    fn align_gc_link_args_promotes_a_void_intermediate_phi() {
        let mut graph = FunctionGraph::new("void_intermediate");
        let shared = push_input(&mut graph, "p", ValueType::Int);
        let cond = push_input(&mut graph, "c", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&shared, ConcreteType::GcRef);
        let (gc_bb, gc_inputs) = graph.create_block_with_arg_vars(1);
        let (void_bb, void_inputs) = graph.create_block_with_arg_vars(1);
        FunctionGraph::set_concretetype_of_inline(&gc_inputs[0], ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&void_inputs[0], ConcreteType::Void);
        let (ret_bb, ret_inputs) = graph.create_block_with_arg_vars(1);
        FunctionGraph::set_concretetype_of_inline(&ret_inputs[0], ConcreteType::GcRef);
        graph.set_return(ret_bb, Some(ret_inputs[0].clone()));
        graph.set_goto(gc_bb, ret_bb, vec![gc_inputs[0].clone()]);
        graph.set_goto(void_bb, ret_bb, vec![void_inputs[0].clone()]);
        graph.set_branch(
            graph.startblock,
            cond,
            gc_bb,
            vec![shared.clone()],
            void_bb,
            vec![shared.clone()],
        );

        align_gc_link_args(&mut graph);

        assert_eq!(
            FunctionGraph::concretetype_of(&void_inputs[0]),
            ConcreteType::GcRef,
            "a Void phi that is not the return leftover is the same pointer"
        );
    }

    #[test]
    fn rebucket_kind_lists_drops_a_void_call_arg() {
        let mut graph = FunctionGraph::new("void_in_list");
        let live = push_input(&mut graph, "p", ValueType::Ref(None));
        let unit = push_input(&mut graph, "u", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&live, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&unit, ConcreteType::Void);
        graph
            .block_mut(graph.startblock)
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::InlineCall {
                    jitcode: crate::jitcode::JitCodeHandle::new(std::sync::Arc::new(
                        crate::jitcode::JitCode::new("callee"),
                    )),
                    args_i: Vec::new(),
                    args_r: vec![live.clone(), unit.clone()],
                    args_f: Vec::new(),
                    result_kind: 'r',
                    arg_classes: "rr".into(),
                },
            });

        rebucket_kind_lists(&mut graph);

        let call = graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::InlineCall {
                    args_r,
                    args_i,
                    arg_classes,
                    ..
                } => Some((args_r.len(), args_i.len(), arg_classes.clone())),
                _ => None,
            })
            .expect("InlineCall");
        assert_eq!(call.1, 0, "Void must not fall into the int list");
        assert_eq!(call.0, 1);
        assert_eq!(call.2, "r");
    }

    #[test]
    fn stamp_vtable_method_slots_retypes_a_fnptr_fieldread() {
        let mut graph = FunctionGraph::new("vtable_slot");
        let vtable = push_input(&mut graph, "vtable", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&vtable, ConcreteType::Signed);
        let slot = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: vtable,
                    field: FieldDescriptor::new(
                        "method_length",
                        Some("dictmultiobject::DictStrategy::{vtable}".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        FunctionGraph::set_concretetype_of_inline(&slot, ConcreteType::GcRef);
        graph.set_return(graph.startblock, Some(slot.clone()));

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&slot),
            ConcreteType::Signed,
            "method_* result must be an int-bank function address"
        );
        match &graph
            .block(graph.startblock)
            .operations
            .last()
            .unwrap()
            .kind
        {
            OpKind::FieldRead {
                ty, pure, field, ..
            } => {
                assert!(field.is_vtable_method_slot());
                assert_eq!(*ty, ValueType::Int);
                assert!(*pure);
            }
            other => panic!("expected FieldRead, got {other:?}"),
        }
        assert!(
            !field_owner_is_gc(
                &FieldDescriptor::new(
                    "method_length",
                    Some("dictmultiobject::DictStrategy::{vtable}".into()),
                ),
                None
            ),
            "{{vtable}} method slots are raw function addresses"
        );
    }

    #[test]
    fn field_owner_is_gc_defaults_unknown_owners_to_gc() {
        let named = FieldDescriptor::new("x", Some("Point".into()));
        let unnamed = FieldDescriptor::new("x", None);
        assert!(
            field_owner_is_gc(&named, None),
            "jtransform.py getattr(STRUCT, '_gckind', 'gc') defaults to gc"
        );
        assert!(
            field_owner_is_gc(&unnamed, None),
            "a descriptor with no owner_root is treated as GC"
        );
    }

    #[test]
    fn arraylen_result_is_signed_like_arraylen_vable() {
        let mut graph = FunctionGraph::new("arraylen_kind");
        let base = push_input(&mut graph, "arr", ValueType::Ref(None));
        let arraylen = OpKind::ArrayLen {
            base: base.clone(),
            array_type_id: None,
            nolength: false,
        };
        let vable = OpKind::VableArrayLen {
            base,
            array_index: 0,
            item_ty: ValueType::Ref(None),
            array_itemsize: 8,
            array_is_signed: false,
        };
        assert_eq!(
            authoritative_result_type_from_op(&arraylen),
            Some(ConcreteType::Signed),
            "arraylen_gc/rd>i returns a length, never the array"
        );
        assert_eq!(
            authoritative_result_type_from_op(&vable),
            Some(ConcreteType::Signed),
            "arraylen_vable/rdd>i returns a length"
        );
    }

    fn link_to_signed_successor(
        graph: &mut FunctionGraph,
        src: &crate::flowspace::model::Variable,
    ) -> crate::flowspace::model::Variable {
        let (next, args) = graph.create_block_with_arg_vars(1);
        let input = args[0].clone();
        FunctionGraph::set_concretetype_of_inline(&input, ConcreteType::Signed);
        graph.set_return(next, Some(input.clone()));
        graph.set_goto(graph.startblock, next, vec![src.clone()]);
        input
    }

    #[test]
    fn coerce_cross_bank_links_casts_a_ref_into_a_signed_inputarg() {
        let mut graph = FunctionGraph::new("ref_to_signed_link");
        let src = push_input(&mut graph, "p", ValueType::Ref(None));
        FunctionGraph::set_concretetype_of_inline(&src, ConcreteType::GcRef);
        let input = link_to_signed_successor(&mut graph, &src);

        coerce_cross_bank_links(&mut graph);

        assert_eq!(FunctionGraph::concretetype_of(&src), ConcreteType::GcRef);
        assert_eq!(FunctionGraph::concretetype_of(&input), ConcreteType::Signed);
        let ops = &graph.block(graph.startblock).operations;
        let cast = ops
            .iter()
            .find(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, operand, .. }
                        if op == "cast_ptr_to_int" && operand.id() == src.id()
                )
            })
            .expect("cast_ptr_to_int of the ref link argument");
        let cast_result = cast.result.clone().expect("cast result");
        assert_eq!(
            FunctionGraph::concretetype_of(&cast_result),
            ConcreteType::Signed
        );
        assert_eq!(
            graph.block(graph.startblock).exits[0].args[0]
                .as_variable()
                .map(|var| var.id()),
            Some(cast_result.id())
        );
    }

    #[test]
    fn coerce_cross_bank_links_casts_a_signed_value_into_a_ref_inputarg() {
        let mut graph = FunctionGraph::new("signed_to_ref_link");
        let src = push_input(&mut graph, "n", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&src, ConcreteType::Signed);
        let (next, args) = graph.create_block_with_arg_vars(1);
        let input = args[0].clone();
        FunctionGraph::set_concretetype_of_inline(&input, ConcreteType::GcRef);
        graph.set_return(next, Some(input.clone()));
        graph.set_goto(graph.startblock, next, vec![src.clone()]);

        coerce_cross_bank_links(&mut graph);

        assert_eq!(FunctionGraph::concretetype_of(&src), ConcreteType::Signed);
        assert_eq!(FunctionGraph::concretetype_of(&input), ConcreteType::GcRef);
        let passed = graph.block(graph.startblock).exits[0].args[0]
            .as_variable()
            .expect("cast result")
            .clone();
        assert_ne!(passed.id(), src.id());
        assert_eq!(FunctionGraph::concretetype_of(&passed), ConcreteType::GcRef);
        assert!(graph.block(graph.startblock).operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::UnaryOp { op, operand, .. }
                    if op == "cast_int_to_ptr" && operand.id() == src.id()
            )
        }));
    }

    #[test]
    fn coerce_cross_bank_links_leaves_a_return_link_alone() {
        let mut graph = FunctionGraph::new("return_link");
        let src = push_input(&mut graph, "p", ValueType::Ref(None));
        FunctionGraph::set_concretetype_of_inline(&src, ConcreteType::GcRef);
        let return_var = graph.block(graph.returnblock).inputargs[0].clone();
        FunctionGraph::set_concretetype_of_inline(&return_var, ConcreteType::Signed);
        graph.set_return(graph.startblock, Some(src.clone()));

        coerce_cross_bank_links(&mut graph);

        assert_eq!(
            graph.block(graph.startblock).exits[0].args[0]
                .as_variable()
                .map(|var| var.id()),
            Some(src.id()),
            "make_return colours the source, so a return link is not cast"
        );
    }

    #[test]
    fn coerce_cross_bank_links_leaves_last_exception_args_alone() {
        let mut graph = FunctionGraph::new("exc_link");
        let src = push_input(&mut graph, "p", ValueType::Ref(None));
        FunctionGraph::set_concretetype_of_inline(&src, ConcreteType::GcRef);
        let _input = link_to_signed_successor(&mut graph, &src);
        let arg = graph.block(graph.startblock).exits[0].args[0].clone();
        graph.block_mut(graph.startblock).exits[0].last_exception = Some(arg);

        coerce_cross_bank_links(&mut graph);

        assert_eq!(
            graph.block(graph.startblock).exits[0].args[0]
                .as_variable()
                .map(|var| var.id()),
            Some(src.id())
        );
        assert!(
            graph.block(graph.startblock).operations.iter().all(
                |op| !matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "cast_ptr_to_int")
            )
        );
    }

    #[test]
    fn coerce_cross_bank_links_keeps_a_trailing_live_op_last() {
        let mut graph = FunctionGraph::new("live_last");
        let src = push_input(&mut graph, "p", ValueType::Ref(None));
        FunctionGraph::set_concretetype_of_inline(&src, ConcreteType::GcRef);
        graph
            .block_mut(graph.startblock)
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
        let _input = link_to_signed_successor(&mut graph, &src);
        graph.block_mut(graph.startblock).exitswitch =
            Some(crate::model::ExitSwitch::LastException);

        coerce_cross_bank_links(&mut graph);

        let ops = &graph.block(graph.startblock).operations;
        assert!(
            matches!(ops.last().map(|op| &op.kind), Some(OpKind::Live)),
            "the trailing live op stays last so flatten still emits catch_exception"
        );
        let cast_at = ops
            .iter()
            .position(
                |op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "cast_ptr_to_int"),
            )
            .expect("cast");
        assert_eq!(cast_at, ops.len() - 2);
    }

    #[test]
    fn coerce_cross_bank_links_follows_a_raising_op_that_defines_the_argument() {
        let mut graph = FunctionGraph::new("raise_defines");
        let produced = graph
            .push_op_var(graph.startblock, OpKind::ConstInt(1), true)
            .unwrap();
        FunctionGraph::set_concretetype_of_inline(&produced, ConcreteType::GcRef);
        let (next, args) = graph.create_block_with_arg_vars(1);
        FunctionGraph::set_concretetype_of_inline(&args[0], ConcreteType::Signed);
        graph.set_return(next, Some(args[0].clone()));
        graph.set_goto(graph.startblock, next, vec![produced.clone()]);
        graph.block_mut(graph.startblock).exitswitch =
            Some(crate::model::ExitSwitch::LastException);

        coerce_cross_bank_links(&mut graph);

        let ops = &graph.block(graph.startblock).operations;
        let produced_at = ops
            .iter()
            .position(|op| {
                op.result
                    .as_ref()
                    .is_some_and(|result| result.id() == produced.id())
            })
            .expect("producer");
        let cast_at = ops
            .iter()
            .position(
                |op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "cast_ptr_to_int"),
            )
            .expect("cast");
        assert!(cast_at > produced_at);
        assert_eq!(cast_at, ops.len() - 1);
    }
}
