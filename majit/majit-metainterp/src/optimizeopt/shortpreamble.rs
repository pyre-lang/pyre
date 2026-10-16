/// Short preamble: minimal operations to replay when entering a peeled loop
/// from a bridge rather than from the preamble.
///
/// Translated from rpython/jit/metainterp/optimizeopt/shortpreamble.py.
///
/// After loop peeling, the optimizer processes the peeled iteration (preamble)
/// and discovers facts about loop-carried values (constants, types, bounds).
/// The loop body is then optimized assuming those facts hold.
///
/// When a bridge later jumps to the loop header (Label), it doesn't have
/// the preamble's context. The "short preamble" is a minimal set of operations
/// — typically guards — that re-establish the facts the loop body depends on.
///
/// Without the short preamble, bridges would need to either:
/// 1. Re-execute the entire preamble (wasteful), or
/// 2. Be conservative and lose optimizations (slow)
///
/// # Structure
///
/// ```text
/// [preamble]               ← full first iteration, optimizer learns facts
///   Label(...)             ← loop header, short preamble stored here
///   [short preamble ops]   ← replayed when a bridge enters here
/// [optimized body]         ← relies on facts from preamble
///   Jump(...)              ← back-edge
/// ```
///
/// # Integration
///
/// The `ShortPreambleBuilder` is used during optimization. When the optimizer
/// processes the preamble and finds guards/operations that establish facts
/// the body depends on, it records them. At the Label, the builder finalizes
/// into a `ShortPreamble` that is stored alongside the compiled loop.
use indexmap::{IndexMap, IndexSet};
use majit_ir::operand::Operand;
use majit_ir::{GcRef, Op, OpCode, OpRc, OpRef};
use rustc_hash::FxBuildHasher;

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;
type FxIndexSet<K> = IndexSet<K, FxBuildHasher>;

use crate::optimizeopt::virtualstate::VirtualState;

pub type EmptyInfo = crate::optimizeopt::info::EmptyInfo;

/// A recorded preamble operation that bridges must replay.
///
/// Each entry captures an operation from the preamble that was either:
/// - A guard that the body assumes always holds
/// - A pure operation whose result the body uses as a known value
/// - A type/class check that enables downstream specialization
#[derive(Clone, Debug)]
pub struct ShortPreambleOp {
    /// The operation to replay (with args referencing label arg indices).
    pub op: Op,
    /// Which label arg indices this op's arguments map to.
    /// Maps from op arg position to label arg index.
    pub arg_mapping: Vec<(usize, usize)>,
    /// Which label arg indices this op's fail args map to.
    /// Maps from fail_arg position to label arg index.
    pub fail_arg_mapping: Vec<(usize, usize)>,
}

/// The complete short preamble for a peeled loop.
///
/// Stored alongside the Label's target token. When a bridge targets
/// this label, the short preamble ops are prepended to establish
/// the optimization context the loop body expects.
#[derive(Clone, Debug)]
pub struct ShortPreamble {
    /// Operations to prepend when entering the loop from a bridge.
    /// These are guards and setup ops with args referencing label arg indices.
    pub ops: Vec<ShortPreambleOp>,
    /// Input args of the short preamble Label.
    /// RPython stores the full short preamble as [Label(short_inputargs), ...].
    /// Stored as flat [`OpRef`] positions; an inline `OpRef::ConstPtr`
    /// ref is GC-walked directly by `walk_const_ptr_refs_mut`.
    pub inputargs: Vec<OpRef>,
    /// Extra loop-header values carried by the short preamble Jump.
    /// RPython appends `sb.used_boxes` to the loop label and jumps with
    /// `args + extra`, where `extra` is the remapped version of these boxes.
    pub used_boxes: Vec<OpRef>,
    /// Preamble producer results used by the short preamble's own trailing JUMP.
    /// RPython keeps this separate from `used_boxes`: the loop contract carries
    /// body boxes, while the short preamble JUMP reuses the corresponding
    /// preamble-produced values.
    pub jump_args: Vec<OpRef>,
    /// The exported virtual state at the loop header (from the preamble's exit).
    /// Used to check bridge compatibility and generate additional guards.
    pub exported_state: Option<VirtualState>,
    /// Constant snapshot retained only for legacy test fixtures.
    /// Production short preamble ops embed inline `Const*` OpRefs
    /// directly, matching RPython where `_map_args` passes Const boxes
    /// through unchanged. This map is therefore empty on production paths.
    pub constants: majit_ir::ConstMap<majit_ir::Const>,
    /// RPython parity: PtrInfo for each inputarg, from Phase 1 export.
    /// shortpreamble.py __init__: preamble_op.set_forwarded(info)
    /// Used by inline_short_preamble to propagate PtrInfo to jump_args
    /// so guards added by use_box are eliminated as redundant.
    pub inputarg_infos: Vec<Option<crate::optimizeopt::info::PtrInfo>>,
}

impl ShortPreamble {
    /// Create an empty short preamble (no extra operations needed).
    pub fn empty() -> Self {
        ShortPreamble {
            ops: Vec::new(),
            inputargs: Vec::new(),
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state: None,
            constants: majit_ir::ConstMap::default(),
            inputarg_infos: Vec::new(),
        }
    }

    /// Whether this short preamble has any operations to replay.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Number of operations in the short preamble.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn walk_const_ptr_refs_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        self.walk_const_ptr_holders_mut(visitor);
        if let Some(exported_state) = self.exported_state.as_mut() {
            exported_state.walk_const_ptr_refs_mut(visitor);
        }
    }

    /// Ops, input args, constants, and inputarg infos. Skips
    /// `exported_state`: that walk allocates its seen-set, and a caller
    /// that resolves table indexes cannot do it inside `trace_index`.
    pub fn walk_const_ptr_holders_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        fn visit_oprefs(refs: &mut [OpRef], visitor: &mut dyn FnMut(&mut GcRef)) {
            for r in refs.iter() {
                r.trace_const_ptr(visitor);
            }
        }

        for entry in &mut self.ops {
            entry.op.walk_const_ptr_refs_mut(visitor);
        }
        visit_oprefs(&mut self.inputargs, visitor);
        visit_oprefs(&mut self.used_boxes, visitor);
        visit_oprefs(&mut self.jump_args, visitor);
        for constant in self.constants.values() {
            if let majit_ir::Const::Ref(index) = constant {
                majit_ir::const_ptr_table::trace_index(*index, visitor);
            }
        }
        for info in self.inputarg_infos.iter_mut().flatten() {
            info.walk_const_ptr_refs_mut(visitor);
        }
    }
}

impl ShortPreamble {
    /// Count guards in the short preamble.
    pub fn num_guards(&self) -> usize {
        self.ops.iter().filter(|e| e.op.opcode.is_guard()).count()
    }

    /// Count pure ops in the short preamble.
    pub fn num_pure_ops(&self) -> usize {
        self.ops
            .iter()
            .filter(|e| e.op.opcode.is_always_pure())
            .count()
    }
}

/// Collector that extracts short preamble operations from an already-built
/// preamble trace.
///
/// This is intentionally separate from RPython's `ShortPreambleBuilder`.
/// The RPython builder consumes exported short boxes while building phase 2;
/// this collector just turns a preamble section into a `ShortPreamble`.
pub struct CollectedShortPreambleBuilder {
    /// Raw ops collected during the preamble phase (before Label).
    raw_ops: Vec<Op>,
    /// shortpreamble.py `ShortPreambleBuilder.label_args` — the
    /// OpRefs that the Label carries. Index in this list IS the label
    /// arg index. Lookup uses linear scan (label_args is small —
    /// bounded by loop-carried value count).
    label_args: Vec<OpRef>,
    /// Whether the builder is still collecting (before Label).
    active: bool,
}

impl CollectedShortPreambleBuilder {
    pub fn new() -> Self {
        CollectedShortPreambleBuilder {
            raw_ops: Vec::new(),
            label_args: Vec::new(),
            active: true,
        }
    }

    /// Set up the mapping from preamble OpRefs to label arg indices.
    ///
    /// Called when the Label is encountered. `label_args` are the OpRefs
    /// that the Label carries (= the loop-carried values from the preamble).
    pub fn set_label_args(&mut self, label_args: &[OpRef]) {
        self.label_args.clear();
        self.label_args.extend_from_slice(label_args);
        self.active = false; // Switch from preamble to body phase
    }

    /// Record a guard from the preamble that the body depends on.
    ///
    /// The guard's arguments should reference preamble OpRefs that
    /// are carried across the Label as label args.
    pub fn add_preamble_guard(&mut self, op: &Op) {
        if !self.active {
            return; // Only collect during preamble phase
        }

        // Only record guard operations
        if !op.opcode.is_guard() {
            return;
        }

        self.raw_ops.push(op.clone());
    }

    /// Record any preamble operation (guard or pure) that establishes
    /// a fact the body depends on.
    pub fn add_preamble_op(&mut self, op: &Op) {
        if !self.active {
            return;
        }

        self.raw_ops.push(op.clone());
    }

    /// Finalize the builder into a ShortPreamble.
    ///
    /// Called after the Label has been processed and the mapping is set.
    /// Computes arg mappings using the preamble-to-label-arg map.
    pub fn build(self, exported_state: Option<VirtualState>) -> ShortPreamble {
        let Self {
            raw_ops,
            label_args,
            ..
        } = self;
        let entries = raw_ops
            .into_iter()
            .map(|op| {
                let mut arg_mapping = Vec::new();
                for (arg_pos, arg_ref) in op.args_slice().iter().enumerate() {
                    if let Some(label_idx) =
                        label_args.iter().position(|a| *a == arg_ref.to_opref())
                    {
                        arg_mapping.push((arg_pos, label_idx));
                    }
                }
                let mut fail_arg_mapping = Vec::new();
                if let Some(fail_args) = op.guard_fail_args() {
                    for (fail_arg_pos, fail_arg_ref) in fail_args.iter().enumerate() {
                        if let Some(label_idx) = label_args
                            .iter()
                            .position(|a| *a == fail_arg_ref.to_opref())
                        {
                            fail_arg_mapping.push((fail_arg_pos, label_idx));
                        }
                    }
                }
                ShortPreambleOp {
                    op,
                    arg_mapping,
                    fail_arg_mapping,
                }
            })
            .collect();

        ShortPreamble {
            ops: entries,
            inputargs: label_args,
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state,
            constants: majit_ir::ConstMap::default(),
            inputarg_infos: Vec::new(),
        }
    }
}

impl Default for CollectedShortPreambleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Classification of preamble operations.
///
/// shortpreamble.py: PreambleOp, HeapOp, PureOp, LoopInvariantOp, GuardOp
/// Each type determines how the operation is replayed when a bridge enters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreambleOpKind {
    /// shortpreamble.py: PreambleOp — base class for all preamble operations.
    /// A generic preamble operation (guard or other).
    Guard,
    /// shortpreamble.py: ShortInputArg — renamed inputarg for a label slot.
    InputArg,
    /// shortpreamble.py: HeapOp — a heap read (GETFIELD_GC, GETARRAYITEM_GC)
    /// that was cached during the preamble. On bridge entry, the field/array
    /// must be re-read to populate the cache.
    Heap,
    /// shortpreamble.py: PureOp — a pure operation whose result was used
    /// in the loop body. On bridge entry, the pure op is re-computed.
    Pure,
    /// shortpreamble.py: LoopInvariantOp — a CALL_LOOPINVARIANT that was
    /// cached for the loop iteration. On bridge entry, re-execute the call.
    LoopInvariant,
}

/// Extended preamble operation with classification.
///
/// shortpreamble.py: used by ShortBoxes and ExtendedShortPreambleBuilder
/// to track which operations need replay and how.
#[derive(Clone, Debug)]
pub struct PreambleOp {
    /// The operation to replay.
    ///
    /// Carried as an [`majit_ir::OpRc`] so the exported entry, the derived
    /// `ProducedShortOp.preamble_op`, and replay-arg operands can share one
    /// object across the export/import boundary (upstream `preamble_op` is
    /// one ResOperation object, shortpreamble.py).
    pub op: majit_ir::OpRc,
    /// The original short operation before `add_op_to_short` remaps its
    /// arguments into the short-preamble namespace.  Upstream carries this as
    /// `ProducedShortOp.short_op` (for heap entries,
    /// `HeapOp.getfield_op`) alongside the replay `preamble_op`; its argument
    /// boxes are the keys used to read `ExportedState.exported_infos`.
    pub source_op: Option<majit_ir::OpRc>,
    /// `short_op.res` — the result operand this entry produces. Carried as a
    /// producer-bound / const [`Operand`] so the canonical operand travels
    /// with the struct to the export boundary (`ProducedShortOp.res`),
    /// `Rc::ptr_eq`-stable on the producer.
    pub res: majit_ir::operand::Operand,
    /// Classification of this operation.
    pub kind: PreambleOpKind,
    /// Index of the argument in the label (None if not a label arg).
    pub label_arg_idx: Option<usize>,
    /// RPython shortpreamble.py: whether this producer was assigned an
    /// invented SameAs name because another producer won the original slot.
    pub invented_name: bool,
    /// Original result box this invented name aliases, if any.
    /// Carried as an [`Operand`] so the canonical (producer-bound) operand
    /// travels with the struct instead of being re-minted positionally at
    /// each use site.
    pub same_as_source: Option<majit_ir::operand::Operand>,
}

impl PreambleOp {
    /// shortpreamble.py: add_op_to_short(sb) — per-kind logic.
    ///
    /// For HeapOp: reconstruct the getfield/getarrayitem with remapped args.
    /// For PureOp: reconstruct the pure op (promoting to CALL_PURE if call).
    /// For LoopInvariantOp: reconstruct as CALL_LOOPINVARIANT.
    pub fn add_op_to_short(
        &self,
        sb: &mut ShortBoxes,
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> Option<ProducedShortOp> {
        let mut preamble_op = match &self.kind {
            PreambleOpKind::InputArg | PreambleOpKind::Guard => (*self.op).clone(),
            PreambleOpKind::Heap => {
                // shortpreamble.py HeapOp.add_op_to_short:
                //   preamble_arg = sb.produce_arg(sop.getarg(0))
                //   if rop.is_getfield(sop.opnum):
                //       preamble_op = ResOperation(sop.getopnum(), [preamble_arg], descr=sop.getdescr())
                //   else:
                //       preamble_op = ResOperation(sop.getopnum(), [preamble_arg, sop.getarg(1)], descr=sop.getdescr())
                let preamble_arg = sb.produce_arg(ctx, &self.op.arg(0))?;
                let args: smallvec::SmallVec<[majit_ir::operand::Operand; 3]> =
                    if self.op.opcode.is_getfield() {
                        smallvec::smallvec![preamble_arg]
                    } else {
                        smallvec::smallvec![preamble_arg, self.op.arg(1)]
                    };
                self.op.copy_and_change(self.op.opcode, Some(&args), None)
            }
            PreambleOpKind::Pure => {
                // shortpreamble.py PureOp.add_op_to_short:
                //   newarg = sb.produce_arg(arg)
                // `lookup2` applies `get_box_replacement` at lookup time.
                let args = self
                    .op
                    .args_slice()
                    .iter()
                    .map(|arg| sb.produce_arg(ctx, arg))
                    .collect::<Option<smallvec::SmallVec<[majit_ir::operand::Operand; 3]>>>()?;
                let opnum = if self.op.opcode.is_call() {
                    match self.op.opcode {
                        OpCode::CallI => OpCode::CallPureI,
                        OpCode::CallR => OpCode::CallPureR,
                        OpCode::CallF => OpCode::CallPureF,
                        OpCode::CallN => OpCode::CallPureN,
                        other => other,
                    }
                } else {
                    self.op.opcode
                };
                self.op.copy_and_change(opnum, Some(&args), None)
            }
            PreambleOpKind::LoopInvariant => {
                // shortpreamble.py LoopInvariantOp.add_op_to_short:
                //   arglist = [sb.produce_arg(arg) for arg in op.getarglist()]
                //   opnum = OpHelpers.call_loopinvariant_for_descr(op.getdescr())
                //   return ProducedShortOp(self, op.copy_and_change(opnum, args=arglist))
                let args = self
                    .op
                    .args_slice()
                    .iter()
                    .map(|arg| sb.produce_arg(ctx, arg))
                    .collect::<Option<smallvec::SmallVec<[majit_ir::operand::Operand; 3]>>>()?;
                let opnum = match self.op.opcode {
                    OpCode::CallI => OpCode::CallLoopinvariantI,
                    OpCode::CallR => OpCode::CallLoopinvariantR,
                    OpCode::CallF => OpCode::CallLoopinvariantF,
                    OpCode::CallN => OpCode::CallLoopinvariantN,
                    other => other,
                };
                self.op.copy_and_change(opnum, Some(&args), None)
            }
        };
        // shortpreamble.py `copy_and_change` returns a new ResOperation, so
        // `preamble_op` is not `short_op.res`. pyre's `copy_and_change` copies
        // `pos`; mint a distinct replay result. `res` stays the body box and
        // is the only `produced_short_boxes` key. Guards later name this
        // replay op (`make_guards(preamble_op)`), not the export-time box.
        if !matches!(self.kind, PreambleOpKind::InputArg | PreambleOpKind::Guard) {
            let fresh = ctx.alloc_op_position_typed(preamble_op.result_type());
            preamble_op.pos().set(fresh);
        }
        Some(ProducedShortOp {
            kind: self.kind.clone(),
            // shortpreamble.py/85/170 `ProducedShortOp(self, ...)` —
            // short_op.res is the original result box.
            res: self.res.clone(),
            preamble_op: OpRc::new(preamble_op),
            source_op: self.source_op.clone().unwrap_or_else(|| self.op.clone()),
            invented_name: self.invented_name,
            same_as_source: self.same_as_source.clone(),
            label_arg_idx: self.label_arg_idx,
        })
    }
}

/// shortpreamble.py: ShortBoxes — tracks which values from the preamble
/// are "boxed" into the short preamble. Maps label arg indices to
/// the operations that produce them.
#[derive(Clone, Debug, Default)]
pub struct ShortBoxes {
    /// shortpreamble.py:249 self.potential_ops = OrderedDict()
    /// Keyed by the producer's result Box, compared by object identity
    /// (shortpreamble.py:259/290) — every insert/lookup resolves its
    /// position through `ctx.materialize_operand_at`, which memoizes one box
    /// per producer, so the same position yields the same object. Const
    /// results never key this map (they route to `const_short_boxes`).
    /// Values are shared so `materialize_one` can hold a candidate while
    /// it mutates `self`, the way the Python dict hands out the stored object.
    potential_ops: FxIndexMap<majit_ir::operand::Operand, std::rc::Rc<PotentialShortOp>>,
    /// shortpreamble.py:250 self.produced_short_boxes = {}
    /// (insertion order preserved by IndexMap for deterministic export.)
    /// Keyed by the result Box (`shortop.res`), compared by object
    /// identity (shortpreamble.py:317/338) — lookups resolve their
    /// position through `ctx.materialize_operand_at`, which memoizes one
    /// box per producer, so the same position yields the same object.
    produced_short_boxes: FxIndexMap<majit_ir::operand::Operand, ProducedShortOp>,
    /// shortpreamble.py: const_short_boxes
    const_short_boxes: Vec<PreambleOp>,
    /// RPython shortpreamble.py: Const boxes are directly admissible in
    /// produce_arg(). majit models constants as OpRef entries in OptContext,
    /// so we track which OpRefs correspond to constants here.
    known_constants: FxIndexSet<OpRef>,
    /// shortpreamble.py: short_inputargs
    ///
    /// shortpreamble.py `renamed = OpHelpers.inputarg_from_tp(box.type)`
    /// mints a fresh producer-less InputArg box per label slot; the stored
    /// positions ARE the short preamble's Label args (shortpreamble.py:443).
    /// Each entry is a fresh InputArg position allocated from
    /// the op-position counter (`alloc_op_position_typed`), so its identity
    /// is DISTINCT from the original label arg (`shortpreamble.py:257`
    /// `ShortInputArg(box, renamed)` — `box` and `renamed` are two objects).
    /// `short_inputargs[i]` corresponds to `label_args[i]` positionally.
    short_inputargs: Vec<OpRef>,
    /// Strong-`Rc` rooting pool for the renamed `InputArg` objects behind
    /// `short_inputargs[i]`. Index-aligned 1:1 with `short_inputargs`. Keeps
    /// each renamed `AbstractInputArg` alive so producer-shaped consumers can
    /// bind that position to `Operand::InputArg` / a bound operand view — the
    /// analog of `TraceIterator.inputargs` rooting the outer-trace inputargs
    /// (opencoder.py).
    short_inputarg_refs: Vec<majit_ir::InputArgRc>,
    /// shortpreamble.py `box = label_args[i]` — the ORIGINAL label-arg
    /// references, kept so `potential_ops`/lookups resolve a label arg by
    /// its own opref (`shortpreamble.py ShortBoxes.create_short_boxes potential_ops[box]`). pyre needs
    /// this explicitly because OpRef positions are not object identities:
    /// once `short_inputargs[i]` is a distinct renamed box, it no longer
    /// carries the original label-arg identity, so the original must be
    /// stored separately. `label_args[i]` pairs with `short_inputargs[i]`.
    label_args: Vec<OpRef>,
    /// shortpreamble.py: boxes_in_production — cycle-detection set
    /// for `materialize_one` recursion, keyed by the result Box
    /// (shortpreamble.py:314 `self.boxes_in_production[shortop.res]`).
    /// Active set is bounded by recursion depth (linear scan suffices).
    boxes_in_production: FxIndexSet<majit_ir::operand::Operand>,
    /// The number of label args.
    pub num_label_args: usize,
}

#[derive(Clone, Debug)]
enum PotentialShortOp {
    Preamble(PreambleOp),
    Compound(CompoundOp),
}

impl PotentialShortOp {
    fn add_op_to_short(
        &self,
        sb: &mut ShortBoxes,
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> Option<ProducedShortOp> {
        match self {
            PotentialShortOp::Preamble(op) => op.add_op_to_short(sb, ctx),
            PotentialShortOp::Compound(compound) => {
                let produced = compound.flatten(sb, ctx, Vec::new());
                if produced.is_empty() {
                    None
                } else {
                    let index = ShortBoxes::_pick_op_index(&produced, true);
                    let chosen = produced[index].clone();
                    // shortpreamble.py:326-330 — alias side:
                    //   opnum = OpHelpers.same_as_for_type(shortop.res.type)
                    //   new_name = ResOperation(opnum, [shortop.res])
                    //   lst[i].short_op.res = new_name
                    // The alias's type matches `compound.res.type`. PyPy
                    // Box.type is fixed at construction (resoperation.py:
                    // 719/727/739) — an untyped compound.res would be an
                    // upstream-invariant violation.
                    for (i, mut alt) in produced.into_iter().enumerate() {
                        if i == index {
                            continue;
                        }
                        let tp = compound.res.ty().unwrap_or_else(|| {
                            panic!(
                                "compound short-preamble alias source {:?} has no \
                                 variant tag; same_as_for_type requires Int/Ref/Float \
                                 (shortpreamble.py:326-330)",
                                compound.res
                            )
                        });
                        let alias = ctx.alloc_op_position_typed(tp);
                        alt.preamble_op.pos().set(alias);
                        // shortpreamble.py:329 `lst[i].short_op.res =
                        // new_name` — the alias entry's res becomes the
                        // freshly invented name.
                        alt.res = ctx.materialize_operand_at(alias);
                        alt.invented_name = true;
                        // shortpreamble.py `lst[i].short_op.res = new_name`:
                        // the alias result is the freshly invented SameAs box,
                        // not a label arg. Slot identity belongs only to
                        // `ShortInputArg`. `label_arg_idx` was stamped from the
                        // original result position, so clear it on the rebound
                        // alias (`ShortBoxes.add_op_to_short`).
                        alt.label_arg_idx = None;
                        // shortpreamble.py `ResOperation(opnum, [shortop.res])`
                        // — the alias source is the Box itself; resolve to
                        // the canonical (possibly producer-bound) box.
                        alt.same_as_source = Some(ctx.materialize_operand_at(compound.res));
                        // shortpreamble.py:333 `self.produced_short_boxes[
                        // new_name] = lst[i]` — keyed by the alias box.
                        sb.produced_short_boxes.insert(alt.res.clone(), alt.clone());
                    }
                    Some(chosen)
                }
            }
        }
    }
}

impl ShortBoxes {
    pub fn new(num_label_args: usize) -> Self {
        // shortpreamble.py create_short_boxes plants one ShortInputArg
        // per label arg, then heap/pure candidates. Size the maps to
        // that first loop so the IndexMap table is not grown per box.
        ShortBoxes {
            potential_ops: FxIndexMap::with_capacity_and_hasher(num_label_args, FxBuildHasher),
            produced_short_boxes: FxIndexMap::with_capacity_and_hasher(
                num_label_args,
                FxBuildHasher,
            ),
            const_short_boxes: Vec::new(),
            known_constants: FxIndexSet::with_hasher(FxBuildHasher),
            short_inputargs: Vec::with_capacity(num_label_args),
            short_inputarg_refs: Vec::with_capacity(num_label_args),
            label_args: Vec::with_capacity(num_label_args),
            boxes_in_production: FxIndexSet::with_hasher(FxBuildHasher),
            num_label_args,
        }
    }

    /// shortpreamble.py:254-259 — record the original label args. The
    /// matching `short_inputargs` renamed boxes are minted lazily by
    /// `add_short_input_arg` (one per `box`, in `label_args` order), so
    /// only the originals are stored here.
    pub fn with_label_args(label_args: &[OpRef]) -> Self {
        let mut boxes = Self::new(label_args.len());
        boxes.label_args = label_args.to_vec();
        boxes
    }

    /// shortpreamble.py `self.potential_ops[box]` — resolve a label
    /// arg to its slot by the ORIGINAL box opref (not the renamed
    /// `short_inputargs` box, which is a distinct identity). `potential_ops`
    /// is box-keyed, so a box duplicated across `label_args + virtuals` (a
    /// virtual field coinciding with a label arg) OVERWRITES to the LAST
    /// occurrence's `ShortInputArg`; `rposition` mirrors that overwrite,
    /// returning the live slot — consistent with `add_short_input_arg`
    /// stamping `label_arg_idx = live_slot` (the last slot). For a
    /// duplicate-free box `rposition == position`.
    pub fn lookup_label_arg(&self, opref: OpRef) -> Option<usize> {
        self.label_args.iter().rposition(|&a| a == opref)
    }

    /// RPython parity: check if opref is reachable in the short preamble.
    pub fn is_reachable(&self, opref: OpRef) -> bool {
        self.label_args.contains(&opref)
            || opref.is_constant()
            || self.potential_ops.keys().any(|k| k.to_opref() == opref)
    }

    pub fn note_known_constant(&mut self, opref: OpRef) {
        // shortpreamble.py: known_constants is a set of Const Box objects.
        // The CompoundOp alias counter is no longer ShortBoxes-internal
        // (slice β routes minting through `ctx.alloc_op_position_typed`),
        // so there is no `next_synthetic_pos` to bump past constant raws.
        self.known_constants.insert(opref);
    }

    fn add_op(&mut self, key: majit_ir::operand::Operand, pop: PotentialShortOp) {
        self.potential_ops.insert(key, std::rc::Rc::new(pop));
    }

    /// Add a pure operation as a short-box candidate.
    /// shortpreamble.py: sb.add_pure_op(op)
    pub fn add_pure_op(&mut self, ctx: &mut crate::optimizeopt::OptContext, op: &OpRc) {
        let result = op.pos().get();
        self.add_potential_op(
            ctx,
            self.lookup_label_arg(result),
            majit_ir::operand::Operand::from_bound_op(op),
            (**op).clone(),
            PreambleOpKind::Pure,
        );
    }

    /// shortpreamble.py add_heap_op(op, getfield_op)
    ///
    /// `op.pos` is the box the GETFIELD/GETARRAYITEM produces. If that box is
    /// a constant, route to `const_short_boxes` (RPython:
    /// `if isinstance(op, Const): self.const_short_boxes.append(HeapOp(op, getfield_op))`).
    /// Otherwise it joins `potential_ops` as a heap candidate.
    pub fn add_heap_op(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        res: majit_ir::operand::Operand,
        op: Op,
    ) {
        let result = op.pos().get();
        if result.is_constant() || self.known_constants.contains(&result) {
            // shortpreamble.py: const_short_boxes.append(HeapOp(...))
            let label_arg_idx = self.lookup_label_arg(result);
            self.const_short_boxes.push(PreambleOp {
                source_op: None,
                res,
                op: OpRc::new(op),
                kind: PreambleOpKind::Heap,
                label_arg_idx,
                invented_name: false,
                same_as_source: None,
            });
            return;
        }
        self.add_potential_op(
            ctx,
            self.lookup_label_arg(result),
            res,
            op,
            PreambleOpKind::Heap,
        );
    }

    /// Add a loop-invariant call as a short-box candidate.
    pub fn add_loopinvariant_op(&mut self, ctx: &mut crate::optimizeopt::OptContext, op: &OpRc) {
        let result = op.pos().get();
        self.add_potential_op(
            ctx,
            self.lookup_label_arg(result),
            majit_ir::operand::Operand::from_bound_op(op),
            (**op).clone(),
            PreambleOpKind::LoopInvariant,
        );
    }

    pub(crate) fn add_short_input_arg(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        arg: OpRef,
        arg_type: majit_ir::Type,
    ) {
        // shortpreamble.py parity: ShortInputArg's BoxType is the
        // intrinsic `box.type` (BoxInt → same_as_i, ref box → same_as_r,
        // BoxFloat → same_as_f). A `Void` reaching here is a parity
        // violation: RPython has no Void value Box / InputArgVoid class.
        if arg_type == majit_ir::Type::Void {
            panic!(
                "short preamble inputarg {arg:?} resolved to Type::Void; \
                 ShortInputArg requires an int/ref/float value box \
                 (shortpreamble.py:255-259)"
            );
        }
        // shortpreamble.py `self.short_inputargs.append(renamed)`: one
        // renamed box per call, in caller order. The production caller
        // (optimizer.rs preview loop) iterates `label_args + virtuals`
        // (unroll.py:479) in order, appending one renamed box per slot, so
        // this call's slot is `short_inputargs.len()`, captured before the
        // push below.
        let live_slot = self.short_inputargs.len();
        // shortpreamble.py `box = label_args[i]` — register the original
        // so `is_reachable` / `produce_arg` membership resolve it.
        // `with_label_args` pre-seeds `label_args`; the standalone
        // `create_short_boxes` path (empty `label_args`) appends here. A box
        // duplicated across `label_args + virtuals` (a virtual field that
        // coincides with a label arg) is registered once.
        if self.lookup_label_arg(arg).is_none() {
            self.label_args.push(arg);
        }
        // shortpreamble.py `renamed = OpHelpers.inputarg_from_tp(box.type)`
        // — a FRESH InputArg distinct from the original `box`. Its position
        // comes from the op-position counter so it is unique and accounted
        // for by `opref_high_water`; `raw()` shares the op/inputarg integer
        // space, so a fresh op position is a fresh inputarg position too.
        // Mint the `InputArg` object itself and BIND the box to it
        // (`from_bound_inputarg`), the orthodox shape of `inputarg_from_tp`'s
        // fresh `AbstractInputArg`; the strong `Rc` is rooted in
        // `short_inputarg_refs[i]` so the box stays bound (the previous
        // `new_inputarg` left `inputarg_handle = None`, minting a position-only
        // box that shed to `Operand::Box`). `to_opref` is byte-identical
        // (reads only `(type, position)`).
        let pos = ctx.alloc_op_position_typed(arg_type).raw();
        let renamed_ia = majit_ir::InputArg::from_type_rc(arg_type, pos);
        let renamed = renamed_ia.opref();
        // shortpreamble.py `self.potential_ops[box] = ShortInputArg(...)`
        // is a plain dict assignment, so for a box duplicated across the
        // combined list it OVERWRITES: the LAST slot's `ShortInputArg`
        // survives and `produce_arg` returns `short_inputargs[LAST]`, leaving
        // the FIRST slot's renamed box a dead Label arg — never produced,
        // never given info (shortpreamble.py sets info only on
        // produced boxes). pyre mirrors that by stamping `label_arg_idx =
        // live_slot` (this call's slot) on the `potential_ops` entry below;
        // the later duplicate call overwrites with its later slot, so
        // `produce_arg` returns the same LAST-slot renamed box. In the
        // duplicate-free case `live_slot == lookup_label_arg(arg)`, so the
        // rename is unchanged.
        self.short_inputargs.push(renamed);
        self.short_inputarg_refs.push(renamed_ia);
        // shortpreamble.py `ShortInputArg(box, renamed)` — `res` is the
        // original `box`; the SAME_AS replay arg is that box. Exported
        // entries carry original positions and are renamed to the matching
        // `short_inputargs` slot at import (`produced_short_boxes_from_exported_boxes`).
        // The operand's identity is the canonical `_forwarded` host Rc
        // (registered on first materialization), so every later
        // `materialize_operand_at(arg)` lookup returns the same key (ptr_eq).
        let arg_res = ctx.materialize_operand_at(arg);
        let mut same_as = Op::new(
            OpCode::same_as_for_type(arg_type),
            std::slice::from_ref(&arg_res),
        );
        // `ProducedShortOp` stores an OpRc, so the fresh renamed InputArg
        // that shortpreamble.py carries as `preamble_op` is stood in for
        // by this non-emitted SAME_AS. The stand-in stays at the ORIGINAL
        // label box: exported entries carry original positions, and the
        // rename is applied at import, where `produce_arg` maps the entry to
        // its `short_inputargs` slot through `label_arg_idx`.
        //
        // Moving it to `renamed` splits the two halves of that contract —
        // metadata stays seeded at the original while the replay consumes it
        // from `preamble_op.pos` — and an input short box then loses its
        // exported info and guards.
        same_as.pos().set(arg);
        // shortpreamble.py `self.potential_ops[box] = ShortInputArg(...)`
        // — keyed by the label-arg Box itself; `arg_res` is its canonical
        // (producer-bound) operand, shared with `res`.
        self.potential_ops.insert(
            arg_res.clone(),
            std::rc::Rc::new(PotentialShortOp::Preamble(PreambleOp {
                source_op: None,
                res: arg_res,
                op: OpRc::new(same_as),
                kind: PreambleOpKind::InputArg,
                label_arg_idx: Some(live_slot),
                invented_name: false,
                same_as_source: None,
            })),
        );
    }

    /// shortpreamble.py/294 `return ...preamble_op`: for a ShortInputArg
    /// the produced `preamble_op` IS the renamed inputarg box
    /// (shortpreamble.py `ShortInputArg(box, renamed)`). pyre stores the
    /// renamed boxes in `short_inputargs`, paired with each label/virtual slot
    /// by `label_arg_idx` (`lookup_label_arg`), so produce_arg returns that
    /// renamed box — embedding the rename into the exported short op args at
    /// export time, the same substitution the import rename pass performed
    /// (`produced_short_boxes_from_exported_boxes`, `short_inputargs[position
    /// of arg in label_args]`).
    fn renamed_short_inputarg(&self, label_arg_idx: Option<usize>) -> majit_ir::operand::Operand {
        let idx = label_arg_idx
            .expect("InputArg short box missing label_arg_idx (set by add_short_input_arg)");
        majit_ir::operand::Operand::from_bound_inputarg(&self.short_inputarg_refs[idx])
    }

    fn produce_arg(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        arg: &majit_ir::operand::Operand,
    ) -> Option<majit_ir::operand::Operand> {
        let opref = arg.to_opref();
        // shortpreamble.py:288 isinstance(op, Const) → return op.
        if opref.is_constant() {
            return Some(majit_ir::operand::Operand::from_opref(opref));
        }
        // shortpreamble.py `if op in self.produced_short_boxes` — the
        // dict membership is Box identity; an unbound position view falls
        // back to its canonical box.
        let okey = if arg.is_resop() || arg.is_inputarg() {
            arg.clone()
        } else {
            ctx.materialize_operand_at(opref)
        };
        if let Some(existing) = self.produced_short_boxes.get(&okey) {
            // shortpreamble.py:285 `return ...preamble_op` — the
            // dependency's replay op object itself, so preamble-op
            // args carry the dep replay handle.
            //
            // ShortInputArg: upstream `preamble_op` IS the renamed
            // inputarg box (shortpreamble.py `ShortInputArg(box,
            // renamed)`), so produce_arg returns the renamed
            // short_inputargs box for this slot — the export-time
            // rename. (`existing.res` is the ORIGINAL box, kept only
            // as the info-lookup key, shortpreamble.py:417.)
            if existing.kind == PreambleOpKind::InputArg {
                let label_arg_idx = existing.label_arg_idx;
                return Some(self.renamed_short_inputarg(label_arg_idx));
            }
            return Some(majit_ir::operand::Operand::from_bound_op(
                &existing.preamble_op,
            ));
        }
        if self.boxes_in_production.contains(&okey) {
            return None;
        }
        // pyre tracks iteration-known constants (body-typed OpRefs proven
        // constant for this pass) in `known_constants`; those are this
        // stage's `Const` boxes, mirroring `use_box`/`insert_dep_recursive`.
        if self.known_constants.contains(&opref) {
            return Some(okey);
        }
        if self.potential_ops.contains_key(&okey) {
            // shortpreamble.py `r = self.add_op_to_short(...);
            // return r.preamble_op`. ShortInputArg returns the RENAMED
            // short_inputargs box for the slot (shortpreamble.py
            // `ShortInputArg(box, renamed)`), same as the
            // `produced_short_boxes` hit.
            let idx = self.materialize_one(ctx, okey)?;
            let kind = self.produced_short_boxes[idx].kind.clone();
            if kind == PreambleOpKind::InputArg {
                let label_arg_idx = self.produced_short_boxes[idx].label_arg_idx;
                return Some(self.renamed_short_inputarg(label_arg_idx));
            }
            let preamble_op = self.produced_short_boxes[idx].preamble_op.clone();
            return Some(majit_ir::operand::Operand::from_bound_op(&preamble_op));
        }
        // shortpreamble.py:295-296 `else: return None`. Every label arg is
        // registered as a ShortInputArg in `potential_ops`
        // (`add_short_input_arg`, create_short_boxes:255-259), so a label arg
        // is reached through the produced/potential arms above; an opref that
        // is none of those is not produce-able.
        None
    }

    /// shortpreamble.py _pick_op_index
    ///
    /// Pick which compound-op alternative becomes the canonical short box.
    /// Heap ops are last-resort; the first non-Heap candidate wins on the
    /// initial `pick_other=True` pass. If two non-Heap candidates are seen,
    /// recurse with `pick_other=False`, which restricts the second pass to
    /// `ShortInputArg` candidates only (the highest-priority class).
    /// Falls back to index 0 when no candidate qualifies.
    fn _pick_op_index(lst: &[ProducedShortOp], pick_other: bool) -> usize {
        let mut index: Option<usize> = None;
        for (i, item) in lst.iter().enumerate() {
            let prefer = !matches!(item.kind, PreambleOpKind::Heap)
                && (pick_other || item.kind == PreambleOpKind::InputArg);
            if prefer {
                if index.is_some() {
                    debug_assert!(pick_other, "second-tier ambiguity");
                    return Self::_pick_op_index(lst, false);
                }
                index = Some(i);
            }
        }
        index.unwrap_or(0)
    }

    /// Index into `produced_short_boxes`. A hit returns that index; the
    /// stored `ProducedShortOp` is not cloned. `shortpreamble.py`
    /// `produce_arg` / `add_op_to_short` hand back the stored object.
    fn materialize_one(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        okey: majit_ir::operand::Operand,
    ) -> Option<usize> {
        // shortpreamble.py add_op_to_short — guard, cycle set,
        // and final insert all key on `shortop.res` Box identity.
        if let Some(idx) = self.produced_short_boxes.get_index_of(&okey) {
            return Some(idx);
        }
        if self.boxes_in_production.contains(&okey) {
            return None;
        }
        // shortpreamble.py add_op_to_short / produce_arg:
        // `self.potential_ops[op]` looks the entry up and leaves the
        // OrderedDict in place; the shared handle is that stored object.
        let candidate = std::rc::Rc::clone(self.potential_ops.get(&okey)?);
        self.boxes_in_production.insert(okey.clone());
        let produced = candidate.add_op_to_short(self, ctx);
        self.boxes_in_production.swap_remove(&okey);
        let produced = produced?;
        let (idx, _) = self.produced_short_boxes.insert_full(okey, produced);
        Some(idx)
    }

    /// shortpreamble.py: produced_short_boxes after add_op_to_short().
    pub fn produced_ops(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> Vec<(OpRef, ProducedShortOp)> {
        // shortpreamble.py `for shortop in self.potential_ops.values()` —
        // the stored boxes themselves, not a re-resolution of their positions.
        let keys: Vec<majit_ir::operand::Operand> = self.potential_ops.keys().cloned().collect();
        for key in keys {
            let _ = self.materialize_one(ctx, key);
        }
        self.produced_short_boxes
            .iter()
            .map(|(k, v)| (k.to_opref(), v.clone()))
            .collect()
    }

    /// shortpreamble.py:272-280 -- the `for short_op in
    /// self.const_short_boxes:` half of `create_short_boxes`. Split out from
    /// `create_short_boxes` so the production path (which calls
    /// `produced_ops` for the :263-267 half) can run it too; it must be called
    /// AFTER `produced_ops`, because `produce_arg` reads
    /// `produced_short_boxes`.
    pub fn produced_const_ops(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> Vec<ProducedShortOp> {
        let mut short_boxes = Vec::new();
        let const_pending: Vec<PreambleOp> = std::mem::take(&mut self.const_short_boxes);
        for short_op in const_pending {
            let getfield_op = &short_op.op;
            if getfield_op.num_args() == 0 {
                continue;
            }
            let struct_arg = getfield_op.arg(0);
            let Some(preamble_arg) = self.produce_arg(ctx, &struct_arg) else {
                continue;
            };
            // shortpreamble.py:277-278: copy_and_change(opnum, [preamble_arg] + args[1:])
            let mut new_args = vec![preamble_arg];
            new_args.extend_from_slice(&getfield_op.args_slice()[1..]);
            // A heap op reaches here with its descr, so this is the same
            // never-taken arm `produce_heap_field` and `produce_heap_array_item`
            // spell as `getdescr()?`; drop the short box rather than abort the
            // optimizer if that ever stops holding.
            let Some(descr) = getfield_op.getdescr() else {
                continue;
            };
            let mut new_op = Op::with_descr(getfield_op.opcode, &new_args, descr);
            // shortpreamble.py:277 `copy_and_change` produces a fresh result
            // box distinct from `short_op.res`; the latter remains the Const.
            new_op
                .pos()
                .set(ctx.alloc_op_position_typed(new_op.result_type()));
            // shortpreamble.py: ProducedShortOp(short_op, preamble_op)
            short_boxes.push(ProducedShortOp {
                res: ctx.materialize_operand_at(getfield_op.pos().get()),
                preamble_op: OpRc::new(new_op),
                source_op: short_op.source_op.unwrap_or_else(|| short_op.op.clone()),
                kind: PreambleOpKind::Heap,
                invented_name: false,
                same_as_source: None,
                label_arg_idx: None,
            });
        }
        short_boxes
    }

    /// shortpreamble.py ShortBoxes.create_short_boxes
    ///
    /// Materialize all `potential_ops` (already populated by
    /// `produce_potential_short_preamble_ops` calls on each pass) into
    /// `produced_short_boxes`, then walk `const_short_boxes`: for each
    /// constant heap read, try to produce its struct argument; if that
    /// succeeds, emit a `getfield` short op whose first arg is the
    /// produced preamble box. Constant heap reads keep the original Const as
    /// `res`; only the copied replay operation receives a fresh result box.
    ///
    /// `label_args` / `label_arg_types` populate `short_inputargs` via
    /// `add_short_input_arg` (RPython:
    /// `self.potential_ops[box] = ShortInputArg(box, renamed)`).
    pub fn create_short_boxes(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        label_args: &[OpRef],
        label_arg_types: &[majit_ir::Type],
    ) -> Vec<ProducedShortOp> {
        // shortpreamble.py:255-259: register every label arg as a
        // ShortInputArg potential op.
        for (i, &arg) in label_args.iter().enumerate() {
            // shortpreamble.py:256 reads label_args[i].type intrinsically;
            // pyre's parallel array must match label_args length.
            let arg_type = label_arg_types.get(i).copied().unwrap_or_else(|| {
                panic!(
                    "missing label_arg_types[{}] (label_args.len()={}): \
                     create_short_boxes needs the parallel type list in \
                     lockstep with label_args",
                    i,
                    label_args.len()
                )
            });
            self.add_short_input_arg(ctx, arg, arg_type);
        }

        // shortpreamble.py:261: optimizer.produce_potential_short_preamble_ops(self)
        // — caller must have invoked this on each pass before calling
        // create_short_boxes (majit threads passes externally).

        // shortpreamble.py: short_boxes = []; for shortop in potential_ops.values(): add_op_to_short
        let mut short_boxes: Vec<ProducedShortOp> = self
            .produced_ops(ctx)
            .into_iter()
            .map(|(_, op)| op)
            .collect();

        short_boxes.extend(self.produced_const_ops(ctx));
        short_boxes
    }

    /// shortpreamble.py `create_short_inputargs(label_args)`:
    ///
    /// ```python
    /// def create_short_inputargs(self, label_args):
    ///     return self.short_inputargs
    ///     # ... rest of function is dead code after this early return
    /// ```
    ///
    /// Unconditionally returns `self.short_inputargs`. The pyre fallback
    /// to `label_args.to_vec()` on empty was a TODO — empty
    /// short_inputargs is the legitimate empty-loop case in upstream.
    pub fn create_short_inputargs(&self, _label_args: &[OpRef]) -> Vec<OpRef> {
        self.short_inputargs.clone()
    }

    /// The `InputArg` rooting pool paired 1:1 with `create_short_inputargs`'s
    /// boxes. Cloned (`Rc::clone` per handle) in lock-step so the carried
    /// strong `Rc`s keep each `short_inputargs[i]` box bound downstream.
    pub fn create_short_inputarg_refs(&self) -> Vec<majit_ir::InputArgRc> {
        debug_assert_eq!(
            self.short_inputarg_refs.len(),
            self.short_inputargs.len(),
            "short_inputarg_refs must stay index-aligned with short_inputargs"
        );
        self.short_inputarg_refs.clone()
    }

    /// shortpreamble.py: add_potential_op(op, pop)
    /// Add a produced operation to the short boxes at the given position.
    pub fn add_potential_op(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        label_arg_idx: Option<usize>,
        res: majit_ir::operand::Operand,
        op: Op,
        kind: PreambleOpKind,
    ) {
        let result = op.pos().get();
        // shortpreamble.py `self.potential_ops[op]` — keyed by the
        // produced result box itself.
        let key = res;
        let pop = PreambleOp {
            source_op: None,
            res: key.clone(),
            op: OpRc::new(op),
            kind,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        };
        // shortpreamble.py add_potential_op: `CompoundOp(op, pop, prev_op)`
        // overwrites the OrderedDict value in place. `IndexMap::insert` on
        // an existing key does the same without shifting later entries.
        // The previous value moves into the CompoundOp; cloning it would
        // copy the whole `two` chain on every add to the same key.
        if let Some(slot) = self.potential_ops.get_mut(&key) {
            let prev = std::mem::replace(
                slot,
                std::rc::Rc::new(PotentialShortOp::Preamble(pop.clone())),
            );
            *slot = std::rc::Rc::new(PotentialShortOp::Compound(CompoundOp {
                res: result,
                one: pop,
                two: Box::new(std::rc::Rc::unwrap_or_clone(prev)),
            }));
        } else {
            self.add_op(key, PotentialShortOp::Preamble(pop));
        }
    }
}

/// Collector-side extended builder for extracting categorized preamble ops from
/// a peeled trace.
///
/// This is intentionally separate from RPython's active
/// `ExtendedShortPreambleBuilder`, which operates while building the short
/// preamble for phase 2 / bridge entry.
pub struct CollectedExtendedShortPreambleBuilder {
    /// Guards from the preamble.
    guards: Vec<PreambleOp>,
    /// Heap reads from the preamble.
    heap_ops: Vec<PreambleOp>,
    /// Pure operations from the preamble.
    pure_ops: Vec<PreambleOp>,
    /// Loop-invariant calls from the preamble.
    loopinvariant_ops: Vec<PreambleOp>,
    /// shortpreamble.py `ShortPreambleBuilder.label_args` — the
    /// OpRefs that the Label carries. Index in this list IS the label
    /// arg index. Lookup uses linear scan (label_args is small).
    label_args: Vec<OpRef>,
}

impl CollectedExtendedShortPreambleBuilder {
    pub fn new() -> Self {
        CollectedExtendedShortPreambleBuilder {
            guards: Vec::new(),
            heap_ops: Vec::new(),
            pure_ops: Vec::new(),
            loopinvariant_ops: Vec::new(),
            label_args: Vec::new(),
        }
    }

    /// Set the label args mapping.
    pub fn set_label_args(&mut self, label_args: &[OpRef]) {
        self.label_args.clear();
        self.label_args.extend_from_slice(label_args);
    }

    fn lookup_label_arg(&self, opref: OpRef) -> Option<usize> {
        self.label_args.iter().position(|&a| a == opref)
    }

    /// Add a guard operation.
    pub fn add_guard(&mut self, op: Op) {
        let label_arg_idx = self.lookup_label_arg(op.pos().get());
        self.guards.push(PreambleOp {
            source_op: None,
            res: majit_ir::operand::Operand::bound_from_opref(op.pos().get()),
            op: OpRc::new(op),
            kind: PreambleOpKind::Guard,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a pure operation.
    pub fn add_pure_op(&mut self, op: Op) {
        let label_arg_idx = self.lookup_label_arg(op.pos().get());
        self.pure_ops.push(PreambleOp {
            source_op: None,
            res: majit_ir::operand::Operand::bound_from_opref(op.pos().get()),
            op: OpRc::new(op),
            kind: PreambleOpKind::Pure,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a heap read.
    pub fn add_heap_op(&mut self, op: Op) {
        let label_arg_idx = self.lookup_label_arg(op.pos().get());
        self.heap_ops.push(PreambleOp {
            source_op: None,
            res: majit_ir::operand::Operand::bound_from_opref(op.pos().get()),
            op: OpRc::new(op),
            kind: PreambleOpKind::Heap,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a loop-invariant call.
    pub fn add_loopinvariant_op(&mut self, op: Op) {
        let label_arg_idx = self.lookup_label_arg(op.pos().get());
        self.loopinvariant_ops.push(PreambleOp {
            source_op: None,
            res: majit_ir::operand::Operand::bound_from_opref(op.pos().get()),
            op: OpRc::new(op),
            kind: PreambleOpKind::LoopInvariant,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Total number of recorded preamble operations.
    pub fn num_ops(&self) -> usize {
        self.guards.len() + self.heap_ops.len() + self.pure_ops.len() + self.loopinvariant_ops.len()
    }

    /// Build into a ShortPreamble, emitting operations in order:
    /// guards first, then heap reads, then pure ops, then loop-invariant.
    pub fn build(self, exported_state: Option<VirtualState>) -> ShortPreamble {
        let Self {
            guards,
            heap_ops,
            pure_ops,
            loopinvariant_ops,
            label_args,
        } = self;
        let all_ops: Vec<PreambleOp> = guards
            .into_iter()
            .chain(heap_ops)
            .chain(pure_ops)
            .chain(loopinvariant_ops)
            .collect();

        let entries = all_ops
            .into_iter()
            .map(|preamble_op| {
                let mut arg_mapping = Vec::new();
                for (arg_pos, arg_ref) in preamble_op.op.args_slice().iter().enumerate() {
                    if let Some(label_idx) =
                        label_args.iter().position(|a| *a == arg_ref.to_opref())
                    {
                        arg_mapping.push((arg_pos, label_idx));
                    }
                }
                let mut fail_arg_mapping = Vec::new();
                if let Some(fail_args) = preamble_op.op.guard_fail_args() {
                    for (fail_arg_pos, fail_arg_ref) in fail_args.iter().enumerate() {
                        if let Some(label_idx) = label_args
                            .iter()
                            .position(|a| *a == fail_arg_ref.to_opref())
                        {
                            fail_arg_mapping.push((fail_arg_pos, label_idx));
                        }
                    }
                }
                ShortPreambleOp {
                    op: (*preamble_op.op).clone(),
                    arg_mapping,
                    fail_arg_mapping,
                }
            })
            .collect();

        ShortPreamble {
            ops: entries,
            inputargs: Vec::new(),
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state,
            constants: majit_ir::ConstMap::default(),
            inputarg_infos: Vec::new(),
        }
    }
}

impl Default for CollectedExtendedShortPreambleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// shortpreamble.py: CompoundOp — a short op that is composed of
/// two sub-operations (e.g., getfield followed by getarrayitem).
#[derive(Clone, Debug)]
pub struct CompoundOp {
    /// The result OpRef of the compound operation.
    pub res: OpRef,
    /// shortpreamble.py `pop` — always a leaf (`PureOp` / `HeapOp` / …).
    one: PreambleOp,
    /// shortpreamble.py `prev_op` — the previous dict value.
    two: Box<PotentialShortOp>,
}

impl CompoundOp {
    /// shortpreamble.py: CompoundOp.flatten(sb, l)
    ///
    /// Recursively flatten a tree of CompoundOps into a list of
    /// ProducedShortOps in dependency order (children first).
    pub fn flatten(
        &self,
        sb: &mut ShortBoxes,
        ctx: &mut crate::optimizeopt::OptContext,
        mut produced: Vec<ProducedShortOp>,
    ) -> Vec<ProducedShortOp> {
        if let Some(pop) = self.one.add_op_to_short(sb, ctx) {
            produced.push(pop);
        }
        match self.two.as_ref() {
            PotentialShortOp::Compound(compound) => compound.flatten(sb, ctx, produced),
            PotentialShortOp::Preamble(op) => {
                if let Some(pop) = op.add_op_to_short(sb, ctx) {
                    produced.push(pop);
                }
                produced
            }
        }
    }
}

/// shortpreamble.py: ProducedShortOp — wraps a short op with its
/// preamble counterpart for emission during bridge compilation.
///
/// The upstream `ShortInputArg` role (shortpreamble.py) is
/// played by `PreambleOp { kind: InputArg }` entries; the separate
/// caller-less struct mirror was removed.
#[derive(Clone, Debug)]
pub struct ProducedShortOp {
    /// The short op classification.
    pub kind: PreambleOpKind,
    /// `short_op.res` (shortpreamble.py __init__/110/151/224) — the result operand
    /// this short op produces. `add_preamble_op` reads it back as the
    /// `PreambleOp.op` operand (upstream `produce_op` passes `self.res`).
    /// Always a producer-bound / const operand (`materialize_operand_at` on
    /// the preview, `res.bound_op()`-rooted exported entries per #173), never
    /// position-only — so the same operand reaches the `expand_info`
    /// boundary, `Rc::ptr_eq`-stable on the producer.
    pub res: majit_ir::operand::Operand,
    /// The preamble operation to replay.
    pub preamble_op: majit_ir::OpRc,
    /// `ProducedShortOp.short_op` in RPython.  Kept separate from
    /// `preamble_op`: heap replay remaps arg0, while exported-info lookup must
    /// use the original `HeapOp.getfield_op` arg0 box identity.
    pub source_op: majit_ir::OpRc,
    /// Whether this short op uses an invented SameAs result.
    pub invented_name: bool,
    /// Original result this invented name aliases.
    /// Carried as an [`Operand`] so the canonical producer-bound operand
    /// travels with the struct; see [`PreambleOp::same_as_source`].
    pub same_as_source: Option<majit_ir::operand::Operand>,
    /// Slot of this short box's result within the original
    /// `label_args + virtuals`, i.e. `lookup_label_arg(canonical_result)`
    /// carried over from [`PreambleOp::label_arg_idx`] across the export
    /// boundary. `None` for a result that is not a label/virtual slot. Lets
    /// the importer resolve the body-visible result slot directly instead of
    /// matching the result against a parallel originals array.
    pub label_arg_idx: Option<usize>,
}

impl ProducedShortOp {
    /// shortpreamble.py `ProducedShortOp.produce_op` + per-kind dispatch:
    /// - `PureOp.produce_op`
    /// - `HeapOp.produce_op` (getfield + getarrayitem)
    /// - `LoopInvariantOp.produce_op`
    /// - `ShortInputArg.produce_op` (no-op)
    ///
    /// Mutates `ctx` to register the imported preamble op into the
    /// optimizer's per-kind side-table (`imported_short_pure_ops` for Pure,
    /// `set_preamble_field` / `set_preamble_item` on PtrInfo for Heap,
    /// `imported_loop_invariant_results` for LoopInvariant).
    ///
    /// Returns `None` when the kind is non-emit (Heap with a non-getfield /
    /// non-getarrayitem opcode), or when a heap index or loop-invariant
    /// function pointer is not a constant int (`HeapOp.produce_op` /
    /// `LoopInvariantOp.produce_op`).
    pub fn produce_op(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
        result_map: &FxIndexMap<OpRef, OpRef>,
    ) -> Option<OpRef> {
        match self.kind {
            PreambleOpKind::Pure => self.produce_pure(ctx, result_map),
            PreambleOpKind::Heap => match self.preamble_op.opcode {
                OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                    self.produce_heap_field(ctx, optimizer, exported_infos, result_map)
                }
                OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcF => {
                    self.produce_heap_array_item(ctx, optimizer, exported_infos, result_map)
                }
                _ => None,
            },
            PreambleOpKind::LoopInvariant => self.produce_loop_invariant(ctx, result_map),
            // shortpreamble.py `ShortInputArg.produce_op` asserts
            // `not invented_name` and otherwise does nothing.
            PreambleOpKind::InputArg => {
                debug_assert!(
                    !self.invented_name,
                    "shortpreamble.py:234: ShortInputArg cannot have invented_name"
                );
                None
            }
            PreambleOpKind::Guard => None,
        }
    }

    /// The replay op of a const-result heap short box: the exported
    /// `preamble_op` with each dependency arg rebound to the replay op the
    /// imported builder holds for it (shortpreamble.py `produce_arg` returns
    /// `produced_short_boxes[op].preamble_op`).
    fn const_replay_op(&self, ctx: &crate::optimizeopt::OptContext) -> majit_ir::OpRc {
        let mut op = (*self.preamble_op).clone();
        if let Some(builder) = ctx.imported_short_preamble_builder.as_ref() {
            for i in 0..op.num_args() {
                let arg = op.arg(i);
                if arg.to_opref().is_constant() {
                    continue;
                }
                if let Some(dep) = builder.produced_short_op(&arg) {
                    op.setarg(
                        i,
                        majit_ir::operand::Operand::from_bound_op(&dep.preamble_op),
                    );
                }
            }
        }
        OpRc::new(op)
    }

    /// shortpreamble.py `PureOp.produce_op`
    fn produce_pure(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        result_map: &FxIndexMap<OpRef, OpRef>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos().get();
        // Result OpRef was fixed before ShortPreambleBuilder construction,
        // matching RPython's object identity being available before
        // `ProducedShortOp.produce_op` runs.
        let result_opref = *result_map.get(&source)?;
        let opcode = self.preamble_op.opcode;
        // shortpreamble.py:116-120 line-by-line:
        //
        //   if invented_name:
        //       op = self.orig_op.copy_and_change(self.orig_op.getopnum())
        //       op.set_forwarded(self.res)
        //   else:
        //       op = self.res
        //
        // PyPy's `self.res` for the invented arm is `new_name` — the alt
        // SAME_AS Box created by `add_op_to_short` at shortpreamble.py.
        // It is a fresh body-Box that gets emitted into the trace via
        // `add_preamble_op` and lives at its own (alt-specific) position.
        // The transient `op = orig_op.copy(...)` is the PreambleOp wrapper:
        // `op.set_forwarded(self.res)` routes body refs from the wrapper
        // straight to the alt's body slot.
        //
        // pyre's flat-OpRef analog allocates the wrapper at `source` (the
        // synthetic alias from the compound-dedup pass at
        // this file's `ShortBoxes`) and the alt body slot at `result_opref`
        // (= `result_map[source]`, the body-visible OpRef this alt's
        // imported Pure / SAME_AS will live at). Forward `source ->
        // result_opref` so body refs after `force_op_from_preamble`
        // resolve to the alt's own body identity, NOT to the canonical
        // (`same_as_source`) — collapsing both alt and canonical onto the
        // canonical's body ref erases PyPy's invented-name Box identity
        // and corrupts the SAME_AS in `extra_same_as`.
        // Non-invented re-uses self.res directly without forwarding.
        let op_source = if self.invented_name {
            let op_source = ctx
                .get_box_replacement_operand_opt(source)
                .unwrap_or_else(|| ctx.materialize_operand_at(source));
            let op_result = ctx
                .get_box_replacement_operand_opt(result_opref)
                .unwrap_or_else(|| ctx.materialize_operand_at(result_opref));
            ctx.make_equal_to(&op_source, &op_result);
            Some(op_source)
        } else {
            None
        };
        // `result_opref` is a typed synthetic alias minted by
        // `add_op_to_short` via `ctx.alloc_op_position_typed(arg_type)`
        // — its variant carries `Box.type` from the chosen producer
        // (history.py `record_same_as` parity). Downstream type
        // lookups read it directly via `OpRef::ty()` or the producing
        // SAME_AS body op's `op.type_` once it lands in
        // `new_operations`.
        // shortpreamble.py `PureOp.produce_op` hands the pure cache the op
        // object: `self.res`, or `self.orig_op.copy_and_change(...)` when
        // `invented_name`. Those args are the exporting phase's boxes.
        let args = if self.invented_name {
            self.source_op.args_slice().to_vec()
        } else {
            self.res.bound_op()?.args_slice().to_vec()
        };
        // shortpreamble.py: PureOp.produce_op routes through
        // `optpure.pure(...)` (or `extra_call_pure` for calls). majit's
        // staging list `imported_short_pure_ops` covers both; the optimizer
        // installs them in OptPure's respective caches before propagation.
        // shortpreamble.py `PreambleOp(op, preamble_op, ...)` — the
        // replay op is the SAME object ShortPreambleBuilder.__init__
        // seeded (one ResOperation per short box, threaded end to end).
        // The builder was installed by
        // `initialize_imported_short_preamble_builder_from_short_boxes`
        // just before this produce_op loop; reuse its replay Rc instead
        // of rebuilding an equal-content op. Fallback keeps the local
        // construction for harnesses that call produce_op standalone.
        let builder_pop = ctx
            .imported_short_preamble_builder
            .as_ref()
            .and_then(|b| b.produced_short_op(&self.res));
        let imported = match builder_pop {
            Some(p) => {
                debug_assert_eq!(
                    p.preamble_op.pos().get(),
                    self.preamble_op.pos().get(),
                    "builder replay pos diverged from the add_op_to_short replay op"
                );
                crate::optimizeopt::ImportedShortPureOp {
                    opcode,
                    descr: self.preamble_op.getdescr(),
                    args,
                    result: result_opref,
                    cache_args: Vec::new(),
                    pop: crate::optimizeopt::info::PreambleOp {
                        op: op_source.clone().unwrap_or_else(|| self.res.clone()),
                        invented_name: self.invented_name,
                        preamble_op: p.preamble_op.clone(),
                        same_as_source: self.same_as_source.clone(),
                    },
                }
            }
            None => crate::optimizeopt::ImportedShortPureOp::new(
                ctx,
                opcode,
                self.preamble_op.getdescr(),
                args,
                result_opref,
                source,
                self.invented_name,
                self.same_as_source.clone(),
            ),
        };
        // pure.py RecentPureOps.add stores the op. lookup2 reads
        // get_box_replacement(op.getarg(i)) later, after guard_value has
        // forwarded an arg to a Const. Keep those argument boxes.
        let mut imported = imported;
        imported.cache_args = self.source_op.getarglist().to_vec();
        ctx.imported_short_pure_ops.push(imported);
        // shortpreamble.py add_preamble_op + 437-438 extra_same_as:
        // RPython collects the SameAs op into `short_preamble_producer.extra_same_as`
        // lazily at use-box time (force_op_from_preamble path).  majit's
        // `used_imported_short_aliases()` derives the alias list directly
        // from `imported_short_preamble_builder.extra_same_as()` at the same
        // phase boundary, so an eager `imported_short_aliases.push` here
        // would be a TODO dual write.
        //
        // `PureOp.produce_op` hands the body the original result box. The
        // replay position stays on `ImportedShortPureOp.result`.
        let _ = result_opref;
        Some(source)
    }

    /// shortpreamble.py `HeapOp.produce_op` (getfield case)
    fn produce_heap_field(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
        result_map: &FxIndexMap<OpRef, OpRef>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos().get();
        let result_type = self.preamble_op.result_type();
        let descr = self.preamble_op.getdescr()?;
        // shortpreamble.py `HeapOp.produce_op` builds `g` from
        // `self.getfield_op.getarglist()` — the original heap op's receiver.
        let source_obj = self.source_op.arg(0);
        // shortpreamble.py:62-75 keeps two distinct Boxes: `self.res` is
        // body-visible, while `preamble_op` is the freshly replayed GETFIELD
        // result.
        //
        // A const-result entry arrives on the separate const channel, which
        // never registers a `result_map` slot: `produced_const_ops` minted the
        // replay position itself and `source` already names it. A non-const
        // entry must still have a phase-2 body slot. The replay result is
        // `preamble_op.pos`, not that slot.
        if !self.res.to_opref().is_constant() {
            result_map.get(&source)?;
        }
        let _ = result_type;
        // `info.py getfield` indexes `_fields` by the field's position in its
        // parent struct, so the seed has to carry the slot index every reader
        // resolves a getfield with, not the descr's own index. A pyre field
        // descr's `index()` is a tagged layout hash of offset/size/type, which
        // no slot number can equal, so keying the seed with it files the
        // `FieldEntry::Preamble` where the peeled loop's lookup never reaches.
        let descr_idx = crate::optimizeopt::heap::OptHeap::field_slot_index(&descr);
        // shortpreamble.py: if g.getarg(0) in exported_infos:
        //     setinfo_from_preamble(g.getarg(0), exported_infos[...])
        // Pass the Rc handle (unroll.py:61 identity preservation).
        if let Some(crate::optimizeopt::info::OpInfo::Ptr(rc)) = exported_infos.get(&source_obj) {
            ctx.setinfo_from_preamble(&source_obj, rc, Some(exported_infos));
        }
        let mut getfield_op = Op::new(
            OpCode::getfield_for_type(result_type),
            std::slice::from_ref(&source_obj),
        );
        getfield_op.setdescr(descr.clone());
        // shortpreamble.py:66-79 keeps the produced `preamble_op` for replay,
        // but builds `g` with the original heap operation's arguments for
        // exported-info installation and heap-cache mutation.
        let mut source_getfield_op = Op::new(
            OpCode::getfield_for_type(result_type),
            std::slice::from_ref(&source_obj),
        );
        source_getfield_op.setdescr(descr.clone());
        // The replay operation owns `result_opref`; the carried source Box
        // remains distinct. Seed info at the replay slot so
        // `take_preamble_forwarded_opinfo(preamble_op.preamble_op.pos)`
        // reads it back, matching `preamble_op.set_forwarded(info)`.
        getfield_op.pos().set(self.preamble_op.pos().get());
        // shortpreamble.py `PreambleOp(self.res, preamble_op, ...)` —
        // the stored replay is the builder's object (one ResOperation per
        // short box); see produce_pure for the threading rationale.
        let replay_rc = ctx
            .imported_short_preamble_builder
            .as_ref()
            .and_then(|b| b.produced_short_op(&self.res))
            .map(|p| {
                debug_assert_eq!(
                    p.preamble_op.pos().get(),
                    self.preamble_op.pos().get(),
                    "builder replay pos diverged from the add_op_to_short replay op"
                );
                p.preamble_op
            })
            // A const-result entry is not a builder key
            // (`ShortPreambleBuilder::new` skips const results), so its replay
            // is the exported `preamble_op`: its args are what `produce_arg`
            // returned at export, with a dependency rebound to the replay op
            // the builder holds for it.
            .unwrap_or_else(|| {
                if self.res.to_opref().is_constant() {
                    self.const_replay_op(ctx)
                } else {
                    OpRc::new(getfield_op.clone())
                }
            });
        let pop = crate::optimizeopt::info::PreambleOp {
            // shortpreamble.py `PreambleOp(self.res, preamble_op, invented_name)`:
            // the carried Box itself. An invented alias entry
            // (`lst[i].short_op.res = new_name`) still takes the Phase-2 box
            // at its replay position: carrying the Phase-1 alias box here
            // miscompiles synth/inline_subwalk_user_iterator (the peeled loop
            // never exits).
            op: if self.invented_name && !self.res.is_constant() {
                ctx.materialize_operand_at(source)
            } else {
                self.res.clone()
            },
            invented_name: self.invented_name,
            preamble_op: replay_rc,
            same_as_source: self.same_as_source.clone(),
        };
        let parent_descr = getfield_op
            .with_field_descr(|fd| fd.get_parent_descr())
            .flatten();
        if let Some(info) = ctx.get_const_info_mut_box(&source_obj, parent_descr) {
            info.set_preamble_field(descr_idx, pop.clone());
        }
        // shortpreamble.py:72-74: ensure_ptr_info_arg0 + setfield(pop)
        let pop_for_field = pop.clone();
        ctx.with_ensured_ptr_info_arg0(&source_getfield_op, |mut struct_info| {
            if let Some(mut info) = struct_info.as_mut() {
                debug_assert!(
                    !info.is_virtual(),
                    "shortpreamble.py:74: imported heap field on virtual"
                );
                info.set_preamble_field(descr_idx, pop_for_field.clone());
            }
        });
        // shortpreamble.py produce_op keeps `self.res` (the body-visible Box)
        // distinct from `preamble_op` (the replayed GETFIELD result). The
        // builder later appends the former to LABEL.used_boxes and the latter
        // to the preamble JUMP. Forwarding them here erases the replay result:
        // the LABEL then names an undefined body box and reads a stale frame
        // slot. PreambleOp bookkeeping is the only join upstream.
        let imported_base = source_obj.get_box_replacement(false);
        optimizer.register_preamble_cached_field(&imported_base, &descr);
        // see produce_pure: extra_same_as collected lazily by
        // imported_short_preamble_builder; eager push would be a dual-write.
        Some(source)
    }

    /// shortpreamble.py `HeapOp.produce_op` (getarrayitem case)
    fn produce_heap_array_item(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
        result_map: &FxIndexMap<OpRef, OpRef>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos().get();
        let result_type = self.preamble_op.result_type();
        let descr = self.preamble_op.getdescr()?;
        // shortpreamble.py `HeapOp.produce_op` builds `g` from
        // `self.getfield_op.getarglist()` and reads `g.getarg(1).getint()`.
        let source_obj = self.source_op.arg(0);
        let index = self.source_op.arg(1).const_int()?;
        // shortpreamble.py:80-85 preserves the same source/replay Box
        // distinction for GETARRAYITEM, and the same const-channel rule as
        // `produce_heap_field`: no `result_map` slot, `source` is the replay
        // position `produced_const_ops` minted. A non-const entry must still
        // have a phase-2 body slot.
        if !self.res.to_opref().is_constant() {
            result_map.get(&source)?;
        }
        let _ = result_type;
        // shortpreamble.py:68-71 applies to both getfield and
        // getarrayitem: if the base object has exported info, import it
        // before ensuring heap/array PtrInfo.
        // Pass the Rc handle (unroll.py:61 identity preservation).
        if let Some(crate::optimizeopt::info::OpInfo::Ptr(rc)) = exported_infos.get(&source_obj) {
            ctx.setinfo_from_preamble(&source_obj, rc, Some(exported_infos));
        }
        let index_const = ctx.make_constant_int(index);
        let mut getarrayitem_op = Op::new(
            OpCode::getarrayitem_for_type(result_type),
            &[source_obj.clone(), ctx.materialize_operand_at(index_const)],
        );
        getarrayitem_op.setdescr(descr.clone());
        // shortpreamble.py:80-85 mutates array info through the same original
        // `g.getarg(0)` identity while replay keeps its produced argument.
        let mut source_getarrayitem_op = Op::new(
            OpCode::getarrayitem_for_type(result_type),
            &[source_obj.clone(), self.source_op.arg(1)],
        );
        source_getarrayitem_op.setdescr(descr.clone());
        // The replay GETARRAYITEM keeps the result minted by add_op_to_short.
        getarrayitem_op.pos().set(self.preamble_op.pos().get());
        // shortpreamble.py `PreambleOp(self.res, preamble_op, ...)` —
        // stored replay is the builder's object; see produce_pure.
        let replay_rc = ctx
            .imported_short_preamble_builder
            .as_ref()
            .and_then(|b| b.produced_short_op(&self.res))
            .map(|p| {
                debug_assert_eq!(
                    p.preamble_op.pos().get(),
                    self.preamble_op.pos().get(),
                    "builder replay pos diverged from the add_op_to_short replay op"
                );
                p.preamble_op
            })
            // A const-result entry is not a builder key
            // (`ShortPreambleBuilder::new` skips const results), so its replay
            // is the exported `preamble_op`: its args are what `produce_arg`
            // returned at export, with a dependency rebound to the replay op
            // the builder holds for it.
            .unwrap_or_else(|| {
                if self.res.to_opref().is_constant() {
                    self.const_replay_op(ctx)
                } else {
                    OpRc::new(getarrayitem_op.clone())
                }
            });
        let pop = crate::optimizeopt::info::PreambleOp {
            // shortpreamble.py `PreambleOp(self.res, preamble_op, invented_name)`:
            // the carried Box itself. An invented alias entry
            // (`lst[i].short_op.res = new_name`) still takes the Phase-2 box
            // at its replay position: carrying the Phase-1 alias box here
            // miscompiles synth/inline_subwalk_user_iterator (the peeled loop
            // never exits).
            op: if self.invented_name && !self.res.is_constant() {
                ctx.materialize_operand_at(source)
            } else {
                self.res.clone()
            },
            invented_name: self.invented_name,
            preamble_op: replay_rc,
            same_as_source: self.same_as_source.clone(),
        };
        let obj_box = Some(source_obj.clone());
        if source_obj.is_constant()
            || obj_box
                .as_ref()
                .and_then(|b| ctx.get_constant_box(b))
                .is_some()
        {
            if let Some(info) = obj_box
                .as_ref()
                .and_then(|b| ctx.get_const_info_array_mut_box(b, descr.clone()))
            {
                info.set_preamble_item(index as usize, pop.clone());
            }
        } else {
            let pop_for_array = pop.clone();
            ctx.with_ensured_ptr_info_arg0(&source_getarrayitem_op, |mut array_info| {
                if let Some(mut info) = array_info.as_mut()
                    && let crate::optimizeopt::info::PtrInfo::Array(array_info) = &mut *info
                {
                    let _ = array_info.lenbound.make_gt_const(index);
                    let idx = index as usize;
                    if idx >= array_info.items.len() {
                        array_info.items.resize(
                            idx + 1,
                            crate::optimizeopt::info::FieldEntry::Value(Operand::None),
                        );
                    }
                    array_info.items[idx] =
                        crate::optimizeopt::info::FieldEntry::preamble(pop_for_array);
                }
            });
        }
        // shortpreamble.py:80-85 preserves the same source/replay distinction
        // for GETARRAYITEM. PreambleOp bookkeeping joins the two at use time.
        let imported_base = source_obj.get_box_replacement(false);
        optimizer.register_preamble_cached_arrayitem(&imported_base, &descr, index);
        // see produce_pure: extra_same_as collected lazily by
        // imported_short_preamble_builder; eager push would be a dual-write.
        Some(source)
    }

    /// shortpreamble.py `LoopInvariantOp.produce_op`
    fn produce_loop_invariant(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        result_map: &FxIndexMap<OpRef, OpRef>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos().get();
        let result_type = self.preamble_op.result_type();
        // shortpreamble.py `LoopInvariantOp.produce_op` reads
        // `self.res.getarg(0).getint()`.
        let func_ptr = self.res.bound_op()?.arg(0).const_int()?;
        // shortpreamble.py:152-159 stores `self.res` as the body-visible Box
        // and the replay call separately in PreambleOp.
        let result_opref = *result_map.get(&source)?;
        let _ = result_type;
        // `rewrite.py` `self.opt.loop_invariant_results[key] = old_op` —
        // dict-as-map semantics; pyre's Vec-backed parity overwrites the
        // entry when `func_ptr` already exists (PyPy dict behavior),
        // otherwise appends.
        if let Some(entry) = ctx
            .imported_loop_invariant_results
            .iter_mut()
            .find(|(k, _)| *k == func_ptr)
        {
            entry.1 = source;
        } else {
            ctx.imported_loop_invariant_results.push((func_ptr, source));
        }
        // see produce_pure: extra_same_as collected lazily by
        // imported_short_preamble_builder; eager push would be a dual-write.
        Some(source)
    }
}

#[derive(Clone, Debug, Default)]
struct AbstractShortPreambleBuilderState {
    short: Vec<majit_ir::OpRc>,
    short_results: FxIndexSet<OpRef>,
    used_boxes: Vec<OpRef>,
    short_preamble_jump: Vec<majit_ir::OpRc>,
    extra_same_as: Vec<Op>,
    /// shortpreamble.py `self.short_inputargs = short_inputargs` —
    /// the renamed InputArg positions; reused verbatim as the Label args
    /// (shortpreamble.py `ResOperation(rop.LABEL, self.short_inputargs[:])`).
    short_inputargs: Vec<OpRef>,
    /// Known constant OpRefs. In RPython, isinstance(box, Const) is a type
    /// check. In majit, constant OpRefs must be explicitly tracked.
    known_constants: FxIndexSet<OpRef>,
    /// Canonical dedup for `record_imported_preamble_use`, keyed by
    /// `replay_op.pos` (a stable proxy for `self.res`).  Two independent
    /// paths reach the same replay op — `record_preamble_use` and
    /// `add_preamble_op_from_pop` — and Box identity makes one Box equal
    /// one slot regardless of how it is reached, so the dedup prevents
    /// either path from pushing the same RPython Box twice into
    /// `used_boxes` / `short_preamble_jump` / `extra_same_as`.
    recorded_canonical_results: FxIndexSet<OpRef>,
}

impl AbstractShortPreambleBuilderState {
    /// shortpreamble.py: add_preamble_op
    ///
    /// RPython (4 lines):
    ///   op = preamble_op.op.get_box_replacement()
    ///   if preamble_op.invented_name:
    ///       self.extra_same_as.append(op)
    ///   self.used_boxes.append(op)
    ///   self.short_preamble_jump.append(preamble_op.preamble_op)
    ///
    /// `same_as_source`: the original OpRef this invented name aliases.
    fn record_imported_preamble_use(
        &mut self,
        op: majit_ir::operand::Operand,
        replay_op: &majit_ir::OpRc,
        needs_alias: bool,
        same_as_source: Option<majit_ir::operand::Operand>,
    ) {
        if !self
            .recorded_canonical_results
            .insert(replay_op.pos().get())
        {
            return;
        }
        if needs_alias {
            // shortpreamble.py:436-437: extra_same_as carries the resolved
            // box itself; a producer-bound box sheds to a live operand.
            //
            // Measured dead fallback: every `invented_name = true` writer
            // carries `Some(same_as_source)` — the sole production writer
            // is the compound-alias path (`alt.invented_name = true` +
            // `alt.same_as_source = Some(...)`), `add_preamble_op_from_pop`
            // passes the pop's threaded `same_as_source`, and the
            // export/re-import paths copy both fields as a pair. Fallback
            // kept as a release safety net.
            debug_assert!(
                same_as_source.is_some(),
                "needs_alias without same_as_source at {:?}",
                replay_op.pos().get()
            );
            let source = same_as_source.unwrap_or_else(|| op.clone());
            let mut same_as = Op::new(
                OpCode::same_as_for_type(replay_op.result_type()),
                std::slice::from_ref(&source),
            );
            same_as.pos().set(op.to_opref());
            self.extra_same_as.push(same_as);
        }
        self.used_boxes.push(op.to_opref());
        self.short_preamble_jump.push(replay_op.clone());
    }

    fn record_preamble_use(
        &mut self,
        result: majit_ir::operand::Operand,
        produced: &ProducedShortOp,
    ) {
        self.record_imported_preamble_use(
            result,
            &produced.preamble_op,
            produced.invented_name,
            produced.same_as_source.clone(),
        );
    }

    /// Internal: append preamble_op to short (with ovf guard).
    /// Used by add_op_to_short (recursive export-time path).
    fn append_to_short(&mut self, _result: OpRef, produced: &ProducedShortOp) -> majit_ir::OpRc {
        let canonical_result = produced.preamble_op.pos().get();
        if self.short_results.contains(&canonical_result) {
            return produced.preamble_op.clone();
        }
        let preamble_op = produced.preamble_op.clone();
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short
                .push(OpRc::new(Op::new(OpCode::GuardNoOverflow, &[])));
        }
        preamble_op
    }

    /// Non-recursive port of `AbstractShortPreambleBuilderState.use_box`: adds
    /// non-input dependencies and guards for `preamble_op`, then appends the
    /// operation and its result guards.
    /// Called by `OptUnroll.force_op_from_preamble`.
    ///
    /// Dependency args carry the dep's replay op object (produce_arg
    /// object-carry); a non-input, non-const arg whose bound op still
    /// holds the builder's `set_forwarded` marker IS a short-box replay
    /// op; append it and consume its marker with `arg.set_forwarded(None)`.
    fn use_box(
        &mut self,
        preamble_op: &majit_ir::OpRc,
        already_in_short: &FxIndexSet<OpRef>,
        arg_guards: &[OpRc],
        result_guards: &[OpRc],
    ) -> Op {
        let canonical_result = preamble_op.pos().get();
        if self.short_results.contains(&canonical_result)
            || already_in_short.contains(&canonical_result)
        {
            return (**preamble_op).clone();
        }
        // shortpreamble.py:383-396: iterate preamble_op args
        for arg in preamble_op.args_slice().iter() {
            let arg_opref = arg.to_opref();
            if self.short_results.contains(&arg_opref)
                || already_in_short.contains(&arg_opref)
                || self.short_inputargs.contains(&arg_opref)
                || self.known_constants.contains(&arg_opref)
            {
                continue;
            }
            // shortpreamble.py AbstractShortPreambleBuilder.use_box:
            // `arg.get_forwarded() is None` → pass; otherwise append the
            // arg and consume the marker.
            let Some(dep) = arg.bound_op() else { continue };
            let forwarded = dep.forwarded().borrow().clone();
            if matches!(forwarded, majit_ir::forwarding::Forwarded::None) {
                continue;
            }
            *dep.forwarded().borrow_mut() = majit_ir::forwarding::Forwarded::None;
            let dep_canonical = dep.pos().get();
            if !self.short_results.contains(&dep_canonical)
                && !already_in_short.contains(&dep_canonical)
            {
                self.short_results.insert(dep_canonical);
                self.short.push(dep.clone());
                if dep.opcode.is_ovf() {
                    self.short
                        .push(OpRc::new(Op::new(OpCode::GuardNoOverflow, &[])));
                }
            }
        }
        // shortpreamble.py:389,396: info.make_guards(arg, self.short, optimizer)
        self.short.extend(arg_guards.iter().cloned());
        // shortpreamble.py:398: self.short.append(preamble_op)
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short
                .push(OpRc::new(Op::new(OpCode::GuardNoOverflow, &[])));
        }
        // shortpreamble.py:401-402: `info = preamble_op.get_forwarded();
        // preamble_op.set_forwarded(None)` — consume the own marker so a
        // later consumer's arg walk doesn't re-append this op.
        *preamble_op.forwarded().borrow_mut() = majit_ir::forwarding::Forwarded::None;
        // shortpreamble.py:405-406: info.make_guards(preamble_op, self.short, optimizer)
        self.short.extend(result_guards.iter().cloned());
        (**preamble_op).clone()
    }
}

fn build_short_preamble_struct_from_ops(
    short_inputargs: &[OpRef],
    ops: &[majit_ir::OpRc],
    used_boxes: &[OpRef],
    jump_args: &[OpRef],
) -> ShortPreamble {
    let inputarg_idx =
        |arg: &OpRef| -> Option<usize> { short_inputargs.iter().position(|a| a == arg) };
    let entries = ops
        .iter()
        .map(|op| (**op).clone())
        .map(|op| {
            let arg_mapping = op
                .args_slice()
                .iter()
                .enumerate()
                .filter_map(|(arg_pos, arg_ref)| {
                    inputarg_idx(&arg_ref.to_opref()).map(|label_idx| (arg_pos, label_idx))
                })
                .collect();
            let fail_arg_mapping = op
                .guard_fail_args()
                .map(|fail_args| {
                    fail_args
                        .iter()
                        .enumerate()
                        .filter_map(|(fail_arg_pos, fail_arg_ref)| {
                            inputarg_idx(&fail_arg_ref.to_opref())
                                .map(|label_idx| (fail_arg_pos, label_idx))
                        })
                        .collect()
                })
                .unwrap_or_default();
            ShortPreambleOp {
                op,
                arg_mapping,
                fail_arg_mapping,
            }
        })
        .collect();
    // history.py/268/314 — `Const{Int,Float,Ptr}.value` rides inline on
    // the OpRef, so short-preamble ops embed the constant value directly in
    // `op.args` and need no parallel side table. The captured `constants`
    // map is empty along every production path; readers (`is_constant()`
    // short-circuits at every callsite) treat it as a guaranteed-empty
    // compatibility fallback retained only for legacy fixtures.
    //
    // RPython parity: `shortpreamble.py` keeps no `loop_constants` side
    // table — `arg` IS the `Const` box, so `_map_args(mapping, args)`
    // (`unroll.py:364`) passes it through unchanged.
    let constants: majit_ir::ConstMap<majit_ir::Const> = majit_ir::ConstMap::default();
    ShortPreamble {
        ops: entries,
        inputargs: short_inputargs.to_vec(),
        used_boxes: used_boxes.to_vec(),
        jump_args: jump_args.to_vec(),
        exported_state: None,
        constants,
        inputarg_infos: Vec::new(),
    }
}

/// shortpreamble.py: ShortPreambleBuilder
///
/// Builds the replayable short preamble from exported short boxes, while also
/// collecting `used_boxes`, `short_preamble_jump`, and `extra_same_as`.
#[derive(Clone, Debug)]
pub struct ShortPreambleBuilder {
    state: AbstractShortPreambleBuilderState,
    /// shortpreamble.py `self.produced_short_boxes = {}` — keyed by the
    /// short-box res Box identity (`shortop.res`), looked up by box everywhere
    /// (produce_arg/use_box/add_op_to_short). #146 re-keyed this from the
    /// flat-OpRef position (which needed a dual source/result_opref key for
    /// invented names) to the single carried res box: `self.res` at the
    /// cross-peel produce loop, `materialize_operand_at(pos)` at the single-op
    /// re-export. The carried box is invariant to the replay-position aliasing
    /// the dual key compensated for, so the two entries collapse to one. The
    /// PYRE_S8B_HARNESS census measured this lookup agreeing with the former
    /// position key on every live firing across the bench corpus.
    produced_short_boxes: FxIndexMap<majit_ir::operand::Operand, ProducedShortOp>,
}

impl ShortPreambleBuilder {
    pub fn new(
        label_args: &[OpRef],
        short_boxes: &[(majit_ir::operand::Operand, ProducedShortOp)],
        short_inputargs: &[OpRef],
    ) -> Self {
        let mut produced_short_boxes =
            FxIndexMap::with_capacity_and_hasher(short_boxes.len(), FxBuildHasher);
        for (k, v) in short_boxes {
            // shortpreamble.py: __init__ plants
            // `preamble_op.set_forwarded(info)` on every replay op. The
            // exported infos themselves are seeded through ctx position
            // slots (`set_preamble_forwarded_info`), so the on-object
            // marker is the `empty_info` sentinel: its presence tells
            // `use_box` "this operand is a short-box replay op", and
            // consuming it (`set_forwarded(None)`) is the dedup.
            // `OpInfo::EmptyInfo` never appears in exported infos
            // (mod.rs guard), so the marker is unambiguous; Op::clone
            // resets `forwarded`, so built ShortPreamble copies never
            // carry it.
            *v.preamble_op.forwarded().borrow_mut() = majit_ir::forwarding::Forwarded::Info(
                crate::optimizeopt::info::OpInfo::EmptyInfo(crate::optimizeopt::info::EmptyInfo),
            );
            // Const res boxes are ptr-unstable (minted fresh per resolution),
            // so they can never be a stable box-identity key; export already
            // filters const short boxes (`optimizer.rs`'s
            // `optimize_with_constants_and_inputs_at`) so this is inert
            // for the live path, and the single-op re-export passes the
            // memoized `materialize_operand_at(pos)` box.
            if k.to_opref().is_constant() {
                continue;
            }
            produced_short_boxes.insert(k.clone(), v.clone());
        }
        // shortpreamble.py `self.short_inputargs = short_inputargs` —
        // store the caller's renamed positions themselves. The empty
        // fallback (test paths without an exported preview list) mints
        // position refs from the label args once.
        let short_inputargs = if short_inputargs.is_empty() {
            label_args.to_vec()
        } else {
            short_inputargs.to_vec()
        };
        ShortPreambleBuilder {
            state: AbstractShortPreambleBuilderState {
                short_inputargs,
                ..AbstractShortPreambleBuilderState::default()
            },
            produced_short_boxes,
        }
    }

    pub fn note_known_constant(&mut self, opref: OpRef) {
        self.state.known_constants.insert(opref);
    }

    fn use_box_recursive(
        &mut self,
        result: &majit_ir::operand::Operand,
        visiting: &mut FxIndexSet<majit_ir::operand::Operand>,
    ) -> Option<majit_ir::OpRc> {
        let produced = self.produced_short_boxes.get(result)?.clone();
        let canonical_result = produced.preamble_op.pos().get();
        if self.state.short_results.contains(&canonical_result) {
            return Some(produced.preamble_op);
        }
        if !visiting.insert(result.clone()) {
            return None;
        }
        for arg in produced.preamble_op.args_slice().iter() {
            // shortpreamble.py:288 isinstance(arg, Const) → skip.
            // `produce_arg` stored the producer's replay op, and the map
            // is keyed by `short_op.res`, so this lookup hits only when
            // the arg box is that res. A miss leaves ordering to the
            // outer create_short_boxes walk, which emits deps first.
            if arg.to_opref().is_constant() {
                continue;
            }
            if self
                .produced_short_boxes
                .get(arg)
                .is_some_and(|p| p.kind == PreambleOpKind::InputArg)
            {
                continue;
            }
            if self.produced_short_boxes.contains_key(arg) {
                let _ = self.use_box_recursive(arg, visiting);
            }
        }
        visiting.swap_remove(result);
        Some(self.state.append_to_short(result.to_opref(), &produced))
    }

    /// Recursive `ShortPreambleBuilder.add_op_to_short`, used during
    /// export-time create_short_boxes to resolve transitive dependencies.
    pub fn add_op_to_short(&mut self, result: &majit_ir::operand::Operand) -> Option<Op> {
        self.use_box_recursive(result, &mut FxIndexSet::with_hasher(FxBuildHasher))
            .map(|op| (*op).clone())
    }

    /// Port of `AbstractShortPreambleBuilderState.use_box`.
    /// Non-recursive. Called by `OptUnroll.force_op_from_preamble`.
    ///
    /// RPython passes `preamble_op.preamble_op` directly — the replay op
    /// IS the carried object, so there is no entry-selection lookup. The
    /// pop's replay Rc is the builder's own object (threaded by the
    /// `produce_op` family), verified against the builder entry.
    pub fn use_box(
        &mut self,
        source: OpRef,
        preamble_op: &majit_ir::OpRc,
        arg_guards: &[OpRc],
        result_guards: &[OpRc],
    ) {
        // shortpreamble.py `use_box(box, preamble_op)`: `preamble_op` is the
        // replay op created by `add_op_to_short`. Its args are already
        // `produce_arg` results, and `make_guards` was called on this same
        // op, so guard args name the replay result. No second lookup.
        let _ = source;
        self.state.use_box(
            preamble_op,
            &FxIndexSet::with_hasher(FxBuildHasher),
            arg_guards,
            result_guards,
        );
    }

    /// shortpreamble.py:284-285 `op in self.produced_short_boxes`.
    ///
    /// `use_box` also looks a dependency arg up here so the producer is
    /// appended before the consumer (`AbstractShortPreambleBuilder.use_box`).
    /// The live callers reuse the
    /// builder's replay Rc during `produce_pure`/`produce_heap_field`/
    /// `produce_heap_array_item` re-export (pure.rs/shortpreamble.rs, via
    /// `OptContext.imported_short_preamble_builder`).
    ///
    /// #146: keyed by the short-box res Box identity (`shortop.res`), the
    /// carried box the produce loop holds as `self.res`. This replaced the
    /// flat-OpRef position key, whose dual source/result_opref entries
    /// compensated for invented-name replay-position aliasing; the carried box
    /// is invariant to that aliasing so the entries collapse to one. The prior
    /// "blocked" verdict (75 agree / 114 diverge) measured the WRONG box:
    /// `get_box_replacement_operand_opt(source)`, a Phase-2 re-resolution that mints a
    /// fresh non-`ptr_eq` box. The carried Phase-1 res is shared across the
    /// peel boundary (an `Rc::clone` of the exported short-box res), so the
    /// box-identity lookup hits — PYRE_S8B_HARNESS measured 82/82 agreement
    /// with the former position key on every live firing across the corpus.
    pub fn produced_short_op(&self, res: &majit_ir::operand::Operand) -> Option<ProducedShortOp> {
        self.produced_short_boxes.get(res).cloned()
    }

    /// shortpreamble.py: add_preamble_op(preamble_op)
    /// Called from optimizer.force_box when popping from potential_extra_ops.
    ///
    /// RPython unconditionally uses the carried `preamble_op` with no
    /// produced_short_boxes lookup:
    ///   op = preamble_op.op.get_box_replacement()
    ///   if preamble_op.invented_name: self.extra_same_as.append(op)
    ///   self.used_boxes.append(op)
    ///   self.short_preamble_jump.append(preamble_op.preamble_op)
    ///
    /// This is that unconditional pattern (#149 collapse): the prior
    /// produced_short_boxes map lookup is gone. `field_entry::PreambleOp` now
    /// carries the ORIGINAL box an invented-name CompoundOp alternate aliases
    /// (threaded from `ProducedShortOp.same_as_source` at the produce_* export
    /// sites), so the carried pop reproduces the former map entry's record:
    /// `op` resolves to `resolved_op`, and `invented_name` / `same_as_source`
    /// match — the SameAs emits `same_as(original)` rather than the old
    /// `same_as(op)` self-alias (op being the invented SameAs name).
    pub fn add_preamble_op_from_pop(
        &mut self,
        preamble_op: &crate::optimizeopt::info::PreambleOp,
        resolved_op: majit_ir::operand::Operand,
    ) {
        // shortpreamble.py: unconditional add_preamble_op. The carried
        // pop reproduces the builder map entry's record — `op` resolves to the
        // same `resolved_op`, and `invented_name` / `same_as_source` are
        // threaded through `field_entry::PreambleOp` — so the
        // produced_short_boxes lookup is no longer consulted here (#149).
        let replay_op = &preamble_op.preamble_op;
        // shortpreamble.py:435 `op = preamble_op.op.get_box_replacement()`:
        // for non-invented entries the resolved operand IS the preamble-defined
        // res, so the `used_boxes` label slot always has a producer at
        // label fall-through. pyre's heap import resolves through
        // `make_equal_to(source, result)` to a fresh body-visible OpRef with
        // NO preamble producer — the flat-OpRef analogue of an invented
        // name. Emit the same defining alias the invented arm uses
        // (`same_as(resolved) = carried preamble operand`) whenever the resolved
        // slot differs from the carried operand, so the extended label slot is
        // defined in the preamble exactly as upstream's box identity
        // guarantees.
        let (needs_alias, alias_source) = if preamble_op.invented_name {
            (true, preamble_op.same_as_source.clone())
        } else if resolved_op.to_opref() != preamble_op.op.to_opref() {
            (true, Some(preamble_op.op.clone()))
        } else {
            (false, None)
        };
        self.state
            .record_imported_preamble_use(resolved_op, replay_op, needs_alias, alias_source);
    }

    pub fn add_preamble_op(&mut self, result: &majit_ir::operand::Operand) -> bool {
        let Some(produced) = self.produced_short_boxes.get(result).cloned() else {
            return false;
        };
        // shortpreamble.py:435 `op = preamble_op.op.get_box_replacement()`
        // — the stored res box, not a fresh equal-positioned mint.
        let res = produced.res.clone();
        self.state.record_preamble_use(res, &produced);
        true
    }

    pub fn build_short_preamble(&self) -> Vec<Op> {
        let mut result = Vec::with_capacity(self.state.short.len() + 2);
        // shortpreamble.py:443 `ResOperation(rop.LABEL,
        // self.short_inputargs[:])` — the Label args are the stored
        // renamed-inputarg boxes themselves, not fresh mints.
        let short_inputargs_operand: Vec<majit_ir::operand::Operand> = self
            .state
            .short_inputargs
            .iter()
            .copied()
            .map(majit_ir::operand::Operand::bound_from_opref)
            .collect();
        result.push(Op::new(OpCode::Label, &short_inputargs_operand));
        result.extend(self.state.short.iter().map(|op| (**op).clone()));
        let jump_args_operand: Vec<majit_ir::operand::Operand> = self
            .state
            .short_preamble_jump
            .iter()
            .map(majit_ir::operand::Operand::from_bound_op)
            .collect();
        result.push(Op::new(OpCode::Jump, &jump_args_operand));
        result
    }

    pub fn build_short_preamble_struct(&self) -> ShortPreamble {
        let jump_args: Vec<OpRef> = self
            .state
            .short_preamble_jump
            .iter()
            .map(|op| op.pos().get())
            .collect();
        build_short_preamble_struct_from_ops(
            &self.state.short_inputargs,
            &self.state.short,
            &self.state.used_boxes,
            &jump_args,
        )
    }

    pub fn used_boxes(&self) -> &[OpRef] {
        &self.state.used_boxes
    }

    pub fn short_preamble_jump(&self) -> &[majit_ir::OpRc] {
        &self.state.short_preamble_jump
    }

    pub fn extra_same_as(&self) -> &[Op] {
        &self.state.extra_same_as
    }

    pub fn short_inputargs(&self) -> &[OpRef] {
        &self.state.short_inputargs
    }
}

/// shortpreamble.py: ExtendedShortPreambleBuilder
///
/// RPython parity: single `short` list with JUMP sentinel at end.
/// `use_box()` pops JUMP, appends deps/guards/op, re-appends JUMP.
#[derive(Clone, Debug)]
pub struct ExtendedShortPreambleBuilder {
    /// Stays `OpRef`-keyed. `setup()` populates this map (and the GC walk +
    /// constructor clone read it), but EVERY key-lookup of it
    /// (insert_dep_recursive / use_box_recursive / use_box /
    /// add_preamble_op_from_pop / add_preamble_op / produced_short_op) is dead
    /// over the full bench corpus — measured. A #146 operand re-key here is
    /// therefore unverifiable (the gate cannot exercise the silent-miss
    /// surface), like the deferred vectorizer maps.
    produced_short_boxes: FxIndexMap<OpRef, ProducedShortOp>,
    short_inputargs: Vec<OpRef>,
    /// shortpreamble.py setup: self.short = short — single ops list (base + JUMP sentinel)
    short: Vec<Op>,
    /// Tracks which OpRefs are already in `short` (for dedup).
    short_results: FxIndexSet<OpRef>,
    /// Constants tracked for RPython isinstance(arg, Const) checks.
    known_constants: FxIndexSet<OpRef>,
    extra_same_as: Vec<Op>,
    short_preamble_jump: Vec<majit_ir::OpRc>,
    base_extra_same_as: Vec<Op>,
    label_args: Vec<OpRef>,
    used_boxes: Vec<OpRef>,
    short_jump_args: Vec<OpRef>,
    pub target_token: majit_ir::DescrRef,
    /// Canonical dedup keyed by `produced.preamble_op.pos`. Mirrors
    /// `AbstractShortPreambleBuilderState.recorded_canonical_results` —
    /// `produced_short_boxes` carries dual entries (source-key plus
    /// result_opref-key) for the same RPython Box, so per-key dedup
    /// (`label_args` etc.) cannot catch a second add via the alternate
    /// key. RPython's Box identity collapses both paths to one entry.
    recorded_canonical_results: FxIndexSet<OpRef>,
}

impl ExtendedShortPreambleBuilder {
    pub fn walk_const_ptr_refs_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        fn visit_oprefs(refs: &mut [OpRef], visitor: &mut dyn FnMut(&mut GcRef)) {
            for r in refs.iter() {
                r.trace_const_ptr(visitor);
            }
        }

        fn visit_opref_set(set: &mut FxIndexSet<OpRef>, visitor: &mut dyn FnMut(&mut GcRef)) {
            let refs: Vec<OpRef> = set.iter().copied().collect();
            set.clear();
            for r in refs {
                r.trace_const_ptr(visitor);
                set.insert(r);
            }
        }

        fn visit_produced(produced: &mut ProducedShortOp, visitor: &mut dyn FnMut(&mut GcRef)) {
            produced.preamble_op.walk_const_ptr_refs_mut(visitor);
            if let Some(source) = produced.same_as_source.as_ref() {
                source.walk_const_ptr_refs(visitor);
            }
        }

        for (_, produced) in self.produced_short_boxes.iter_mut() {
            visit_produced(produced, visitor);
        }
        visit_oprefs(&mut self.short_inputargs, visitor);
        for op in &mut self.short {
            op.walk_const_ptr_refs_mut(visitor);
        }
        visit_opref_set(&mut self.known_constants, visitor);
        for op in &mut self.extra_same_as {
            op.walk_const_ptr_refs_mut(visitor);
        }
        for op in &mut self.short_preamble_jump {
            op.walk_const_ptr_refs_mut(visitor);
        }
        for op in &mut self.base_extra_same_as {
            op.walk_const_ptr_refs_mut(visitor);
        }
        visit_oprefs(&mut self.label_args, visitor);
        visit_oprefs(&mut self.used_boxes, visitor);
        visit_oprefs(&mut self.short_jump_args, visitor);
        // recorded_canonical_results is keyed by `preamble_op.pos` (op result
        // positions, never Const) — nothing to walk.
    }

    pub fn new(target_token: majit_ir::DescrRef, sb: &ShortPreambleBuilder) -> Self {
        ExtendedShortPreambleBuilder {
            // The live builder now keys `produced_short_boxes` by the short-box
            // res Box (#146); this builder keys by `preamble_op.pos` (the
            // assert in `ensure_dep_from_produced`), so re-key on copy.
            produced_short_boxes: {
                let mut m = FxIndexMap::with_capacity_and_hasher(
                    sb.produced_short_boxes.len(),
                    FxBuildHasher,
                );
                for (_, p) in sb.produced_short_boxes.iter() {
                    m.insert(p.preamble_op.pos().get(), p.clone());
                }
                m
            },
            short_inputargs: sb.short_inputargs().to_vec(),
            short: Vec::new(),
            short_results: FxIndexSet::with_hasher(FxBuildHasher),
            known_constants: FxIndexSet::with_hasher(FxBuildHasher),
            extra_same_as: sb.extra_same_as().to_vec(),
            short_preamble_jump: Vec::new(),
            base_extra_same_as: sb.extra_same_as().to_vec(),
            label_args: Vec::new(),
            used_boxes: Vec::new(),
            short_jump_args: Vec::new(),
            target_token,
            recorded_canonical_results: FxIndexSet::with_hasher(FxBuildHasher),
        }
    }

    /// shortpreamble.py: setup(jump_args, short, label_args)
    ///
    /// RPython parity: builds single `short` list from base ops + JUMP sentinel.
    /// For each base op, ensures missing deps from produced_short_boxes are
    /// inserted before the consumer (RPython use_box arg-handling parity).
    ///
    /// Upstream's `setup` (shortpreamble.py) is three field
    /// assignments and cannot fail, because the filtering happened at EXPORT:
    /// `produce_arg` returning None makes `add_op_to_short` return None and
    /// THAT ONE short_op is left out (shortpreamble.py, 311-341). The
    /// loop below reproduces that per-op drop here, where pyre first learns an
    /// arg is unresolvable — its short boxes are exported against one
    /// label-arg set and replayed against another.
    ///
    /// Returns `false` only for the one case the per-op drop cannot express: a
    /// dropped op whose result the short preamble's own JUMP carries, which
    /// would leave `inline_short_preamble`'s `_map_args` (unroll.py) with
    /// an unmapped key. The caller then falls back to `jump_to_preamble`.
    pub fn setup(
        &mut self,
        short_preamble: &ShortPreamble,
        label_args: &[OpRef],
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> bool {
        // shortpreamble.py ExtendedShortPreambleBuilder.setup stores the
        // lists and leaves short ops referencing the renamed short_inputargs.
        // `inputargs_set` is that same list (`short[0].getarglist()`).
        // Build single short list with inline dep resolution.
        let inputargs_set: FxIndexSet<OpRef> = short_preamble.inputargs.iter().copied().collect();
        let constants_set: FxIndexSet<u32> = short_preamble.constants.keys().copied().collect();
        self.short.clear();
        self.short_results.clear();
        for entry in &short_preamble.ops {
            let op = entry.op.clone();
            // RPython use_box arg loop: insert missing deps before this op.
            // Recursive: deps of deps are also inserted (transitive closure).
            //
            // shortpreamble.py `PureOp.add_op_to_short` (and the
            // HeapOp/LoopInvariantOp siblings):
            //
            //     for arg in op.getarglist():
            //         newarg = sb.produce_arg(arg)
            //         if newarg is None:
            //             return None
            //
            // and `add_op_to_short` (:311-341) then leaves THAT ONE op out of
            // `produced_short_boxes`. Every other op still goes into the short
            // preamble, and an op that depended on the dropped one drops in
            // turn because its own `produce_arg` now misses — the transitive
            // closure this loop reproduces through `short_results`.
            //
            // That filtering is an EXPORT-time rule, and it is the only drop
            // upstream has. By the time upstream reaches the inline it is done:
            // `ExtendedShortPreambleBuilder.setup` (:460-463) is three field
            // assignments and cannot fail, and `inline_short_preamble`
            // (unroll.py:372-410) then replays EVERY op in `short` — guards
            // included, through their own `sop.is_guard()` arm — with
            // `_map_args` (unroll.py) raising KeyError rather than
            // dropping anything. So there is no upstream counterpart for a drop
            // HERE, at inline time, against a body already optimized under the
            // exported facts.
            //
            // Pyre needs one anyway: its short boxes are exported against one
            // label-arg set and replayed against another (a retrace runs no
            // Phase 1 and imports the partial trace's exported state), so an arg
            // that was produce-able at export can be unresolvable here. The drop
            // is therefore bounded to the only case where it changes nothing the
            // target body can observe — a PURE producer whose result the short
            // preamble's own JUMP does not carry. Anything else declines the
            // whole inline, which is what this loop used to do unconditionally,
            // so `jump_to_existing_trace` falls back to `jump_to_preamble`
            // (unroll.py:154/167).
            let resolvable = op.args_slice().iter().all(|arg| {
                self.insert_dep_recursive(arg.to_opref(), &inputargs_set, &constants_set)
            });
            if !resolvable {
                // A GUARD carries the target body's assumption, not a value.
                // `short_preamble.ops` holds `is_guard() || is_always_pure()`
                // ops (see the inclusion rule where this list is built), and
                // upstream replays the guards explicitly at
                // `unroll.py:405-409`, restamping each with a
                // `ResumeAtPositionDescr`. Dropping one lets the JUMP enter a
                // body that was optimized under that guard's fact without ever
                // checking it — a miscompile, not a lost optimization. A guard
                // has no result box, so it can never appear in `jump_args` and
                // the arity test below cannot catch it.
                //
                // An OVF PRODUCER carries a guard positionally rather than in
                // its own arg list, so the `is_guard()` arm cannot see it.
                // Upstream never separates the two: `use_box`
                // (shortpreamble.py) SYNTHESIZES the
                // `GUARD_NO_OVERFLOW` immediately after appending an
                // `is_ovf()` op, so an orphan cannot exist. Pyre stores them
                // as two entries — `extract_short_preamble` admits an
                // `is_guard_overflow()` entry only as the paired successor of
                // an included `is_ovf()` producer, and pushes the producer
                // first — and that successor is ARGLESS, so it is vacuously
                // resolvable and survives its producer's drop. What the
                // orphan then does is worse than being lost:
                // `optimize_GUARD_NO_OVERFLOW` (intbounds.py) reads
                // `last_emitted_operation`, and either kills the guard
                // outright, or — if an unrelated ovf op happens to precede it
                // — mis-attaches and registers that op's inverse pure
                // operations (intbounds.py) as facts. Either way the
                // body's assumptions enter unchecked, which is the same class
                // the `is_guard()` arm above closes.
                //
                // A PRODUCER whose result the short preamble's own JUMP carries
                // has no per-op answer either: `short_preamble.ops` and
                // `short_preamble.jump_args` are filled in lockstep, leaving it
                // unmapped is what `_map_args` (unroll.py) raises
                // KeyError on, and shortening the JUMP instead would disagree
                // with the already-compiled target LABEL's arity.
                // A USED_BOX PRODUCER is carried by the target LABEL, not by
                // the short preamble's own JUMP, so the `jump_args` test above
                // does not see it. `used_boxes` is appended to the LABEL's
                // arglist verbatim (`jump_to_existing_trace` builds
                // `current_label_args` as `label_args + used_boxes`). Dropping
                // the producer while a later short op still reads its box
                // leaves that reader bound to a different loop-carried value
                // than the producer computed: for `t += c[0] + c[-1]` the
                // LABEL slot holding `c[0]`'s unboxed intval receives the
                // accumulator, and the invariant is clobbered once per
                // iteration.
                let decline = if op.opcode.is_guard() {
                    Some("guard")
                } else if op.opcode.is_ovf() {
                    Some("ovf producer")
                } else if short_preamble.jump_args.contains(&op.pos().get()) {
                    Some("jump-arg producer")
                } else if short_preamble.used_boxes.contains(&op.pos().get()) {
                    Some("used-box producer")
                } else {
                    None
                };
                if let Some(reason) = decline {
                    if crate::optimizeopt::majit_log_enabled() {
                        eprintln!(
                            "[jit] short_preamble setup: dropping inline (unresolved arg in \
                             {reason} pos={:?} opcode={:?})",
                            op.pos().get(),
                            op.opcode
                        );
                    }
                    self.short.clear();
                    self.short_results.clear();
                    self.label_args = label_args.to_vec();
                    return false;
                }
                if crate::optimizeopt::majit_log_enabled() {
                    eprintln!(
                        "[jit] short_preamble setup: dropping op pos={:?} opcode={:?} (unresolved arg)",
                        op.pos().get(),
                        op.opcode
                    );
                }
                continue;
            }
            self.short_results.insert(op.pos().get());
            self.short.push(op);
        }
        // JUMP sentinel at end (RPython: short[-1] is always JUMP)
        let mut jump_args = Vec::with_capacity(short_preamble.jump_args.len());
        let mut jump_args_operand = Vec::with_capacity(short_preamble.jump_args.len());
        for arg in &short_preamble.jump_args {
            // Keep the stored short-jump position. That position is the replay
            // key `inline_short_preamble` registers (`mapping[sp_op.pos] =
            // new_ref`) or a seeded short-inputarg — `short_jump_args =
            // short[-1].getarglist()` (unroll.py `inline_short_preamble`)
            // references the same boxes as the short-op results.
            // Applying `get_box_replacement` to the stored key would diverge
            // that identity: under the tagged-int flip a loop-carried raw-Int
            // jump arg (e.g. IntOp(36)) box-forwards to its Phase-2 SameAsI
            // alias (IntOp(247)), which is neither a replay key nor a seeded
            // inputarg, so `_map_args` (unroll.rs force loop) then reads an
            // unmapped key and panics. Forwarding is applied later, AFTER the
            // mapping lookup, via `get_replacement_opref` — the
            // `get_box_replacement(mapping[box])` order. Keep the resolved
            // operand only for the emitted JUMP sentinel's producer binding.
            let operand = ctx
                .get_box_replacement_operand_opt(*arg)
                .unwrap_or_else(|| ctx.materialize_operand_at(*arg));
            jump_args.push(*arg);
            jump_args_operand.push(operand);
        }
        self.short.push(Op::new(OpCode::Jump, &jump_args_operand));
        // Reset state
        self.extra_same_as = self.base_extra_same_as.clone();
        self.short_preamble_jump.clear();
        self.label_args = label_args.to_vec();
        self.used_boxes = short_preamble.used_boxes.clone();
        self.short_jump_args = jump_args;
        true
    }

    /// Recursively insert a dep (and its transitive deps) into self.short.
    /// Used by setup() to ensure all args of base ops are satisfied.
    /// Dep ops keep the stored renamed short_inputargs operands.
    ///
    /// Returns `true` if `arg` was resolved (or was already known); `false`
    /// if it was an unresolvable Phase 1 reference. This IS RPython's
    /// `produce_arg → None` (shortpreamble.py): the consuming op is
    /// left out of the short preamble, so it can never leave a dangling arg
    /// for `inline_short_preamble`'s `_map_args` (unroll.py) to trip on.
    fn insert_dep_recursive(
        &mut self,
        arg: OpRef,
        inputargs_set: &FxIndexSet<OpRef>,
        constants_set: &FxIndexSet<u32>,
    ) -> bool {
        // history.py/268/314 inline-Const variants short-circuit
        // before `arg.raw()` (which panics on inline) — covered by
        // `arg.is_constant()` below.
        if arg.is_constant() || arg.is_none() {
            return true;
        }
        if self.short_results.contains(&arg)
            || self.known_constants.contains(&arg)
            || constants_set.contains(&arg.raw())
        {
            return true;
        }
        // shortpreamble.py:284-285 — `op in self.produced_short_boxes`,
        // keyed by Box identity. Every entry is keyed by its own
        // `preamble_op.pos` (produced_short_boxes_from_exported_boxes), so the
        // map key equals the producer position a dep arg references — the
        // direct lookup is exact, matching RPython's single Box-identity
        // lookup. The former `preamble_op.pos → key` reverse index was a
        // vestige of an earlier export that keyed by something other than the
        // replay pos; it can never fire now (key == pos by construction).
        let dep = self.produced_short_boxes.get(&arg).cloned();
        // A produced short box is a short-op result. It can also sit on the
        // loop LABEL as a `used_box` (`unroll.py` `finalize_short_preamble`
        // appends `sb.used_boxes` to `label_op`). That label slot is not a
        // short inputarg: `inline_short_preamble` maps only
        // `short_inputargs` onto `args + virtuals` (`unroll.py`
        // `inline_short_preamble`). Treating the used box as already
        // satisfied skips its producer, and `_map_args` then misses the
        // consumer's arg. `short_jump_args` (the short preamble JUMP) is a
        // different list of the same length as `used_boxes`; it is not a
        // tail of `jump_args`, so the producer has to be replayed.
        let produced_replay = dep
            .as_ref()
            .is_some_and(|dep| dep.kind != PreambleOpKind::InputArg);
        if !produced_replay && inputargs_set.contains(&arg) {
            return true;
        }
        let Some(dep) = dep else {
            // Tripwire: a reverse `preamble_op.pos == arg` entry must not exist
            // when the direct lookup misses — that would mean some insert path
            // broke the `key == preamble_op.pos` invariant the deletion relies on.
            debug_assert!(
                !self
                    .produced_short_boxes
                    .iter()
                    .any(|(_, prod)| prod.preamble_op.pos().get() == arg),
                "produced_short_boxes key != preamble_op.pos at {arg:?}: \
                 direct lookup missed but a pos-keyed entry exists"
            );
            return false;
        };
        let dep_pos = dep.preamble_op.pos().get();
        if self.short_results.contains(&dep_pos) {
            return true;
        }
        let dep_op = (*dep.preamble_op).clone();
        // Recurse into dep's own args first (transitive). If any sub-dep
        // can't be resolved, bail out — the dep cannot be safely emitted.
        let dep_op_args = dep_op.getarglist_copy();
        for dep_arg in dep_op_args.iter() {
            let dep_arg = dep_arg.to_opref();
            if !self.insert_dep_recursive(dep_arg, inputargs_set, constants_set) {
                return false;
            }
        }
        self.short_results.insert(dep_pos);
        self.short.push(dep_op);
        if dep.preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        true
    }

    fn use_box_recursive(&mut self, result: OpRef, visiting: &mut FxIndexSet<OpRef>) -> Option<Op> {
        let produced = self.produced_short_boxes.get(&result)?.clone();
        let canonical_result = produced.preamble_op.pos().get();
        if self.short_results.contains(&canonical_result) {
            return Some((*produced.preamble_op).clone());
        }
        if !visiting.insert(result) {
            return None;
        }
        for arg in produced.preamble_op.args_slice().iter() {
            let arg = arg.to_opref();
            // shortpreamble.py:288 isinstance(arg, Const) → skip
            if arg.is_constant() {
                continue;
            }
            if self.produced_short_boxes.iter().any(|(k, _)| *k == arg) {
                let _ = self.use_box_recursive(arg, visiting);
            }
        }
        visiting.swap_remove(&result);
        // Append to self.short directly
        let preamble_op = (*produced.preamble_op).clone();
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        Some(preamble_op)
    }

    /// shortpreamble.py: add_preamble_op(preamble_op)
    ///
    /// RPython unconditionally appends to label_args and jump_args:
    ///   op = preamble_op.op.get_box_replacement()
    ///   self.label_args.append(op)
    ///   self.jump_args.append(preamble_op.preamble_op)
    /// shortpreamble.py: add_preamble_op(preamble_op)
    ///
    /// Extended version: label_args/jump_args instead of used_boxes.
    ///   op = preamble_op.op.get_box_replacement()
    ///   if preamble_op.invented_name:
    ///       self.extra_same_as.append(op)
    ///   self.label_args.append(op)
    ///   self.jump_args.append(preamble_op.preamble_op)
    pub fn add_preamble_op_from_pop(
        &mut self,
        preamble_op: &crate::optimizeopt::info::PreambleOp,
        resolved_op: majit_ir::operand::Operand,
    ) {
        let resolved_key = resolved_op.to_opref();
        let lookup_key = if self
            .produced_short_boxes
            .iter()
            .any(|(k, _)| *k == resolved_key)
        {
            resolved_key
        } else {
            preamble_op.op.to_opref()
        };
        if let Some(produced) = self.produced_short_boxes.get(&lookup_key).cloned() {
            self.add_tracked_preamble_op(resolved_op, &produced);
        } else {
            // shortpreamble.py add_preamble_op: same pattern via replay_op.
            let replay_op = &preamble_op.preamble_op;
            if !self
                .recorded_canonical_results
                .insert(replay_op.pos().get())
            {
                return;
            }
            let op = resolved_key;
            // shortpreamble.py:436-437 plus the fresh-slot case: a
            // non-invented pop whose resolved slot differs from the carried
            // preamble box has no preamble producer for the label slot (see
            // ShortPreambleBuilder::add_preamble_op_from_pop); emit the same
            // defining alias with the carried box as source.
            let alias_source = if preamble_op.invented_name {
                // shortpreamble.py:436-437: alias the carried original
                // (same_as_source), matching `add_tracked_preamble_op`; the
                // resolved operand is the release fallback when none was
                // threaded.
                debug_assert!(
                    preamble_op.same_as_source.is_some(),
                    "invented_name without same_as_source at {:?}",
                    replay_op.pos().get()
                );
                Some(
                    preamble_op
                        .same_as_source
                        .clone()
                        .unwrap_or_else(|| resolved_op.clone()),
                )
            } else if resolved_key != preamble_op.op.to_opref() {
                Some(preamble_op.op.clone())
            } else {
                None
            };
            if let Some(source) = alias_source {
                let mut same_as =
                    Op::new(OpCode::same_as_for_type(replay_op.result_type()), &[source]);
                same_as.pos().set(op);
                self.extra_same_as.push(same_as);
            }
            self.label_args.push(resolved_key);
            // `ExtendedShortPreambleBuilder.add_preamble_op` appends to the
            // target LABEL's own `label_args`. The flattened `ShortPreamble`
            // carries that LABEL extra tail as `used_boxes`, so it grows with
            // `short_jump_args`: the bridge close keeps `inline_short_preamble`'s
            // extra args only up to `used_boxes.len()` (constants dropped, as
            // the assembled LABEL drops them), and a missing entry leaves the
            // bridge JUMP one arg short of the LABEL `label_args` built.
            self.used_boxes.push(resolved_key);
            // The flattened struct only needs the replay position; the replay
            // op itself is rooted by short_preamble_jump.
            self.short_jump_args.push(replay_op.pos().get());
            self.short_preamble_jump.push(replay_op.clone());
        }
    }

    /// shortpreamble.py: add_preamble_op (internal)
    pub fn add_tracked_preamble_op(
        &mut self,
        result: majit_ir::operand::Operand,
        produced: &ProducedShortOp,
    ) {
        let current_result = produced.preamble_op.pos().get();
        if !self.recorded_canonical_results.insert(current_result) {
            return;
        }
        if produced.invented_name {
            // shortpreamble.py:436-437: the resolved box itself is the
            // SameAs source; a producer-bound box sheds to a live operand.
            let source = produced
                .same_as_source
                .clone()
                .unwrap_or_else(|| result.clone());
            let mut op = Op::new(
                OpCode::same_as_for_type(produced.preamble_op.result_type()),
                std::slice::from_ref(&source),
            );
            op.pos().set(current_result);
            self.extra_same_as.push(op);
        }
        self.label_args.push(result.to_opref());
        self.used_boxes.push(produced.preamble_op.pos().get());
        self.short_jump_args.push(produced.preamble_op.pos().get());
        self.short_preamble_jump.push(produced.preamble_op.clone());
    }

    pub fn add_preamble_op(&mut self, result: OpRef) -> bool {
        let Some(produced) = self.produced_short_boxes.get(&result).cloned() else {
            return false;
        };
        // shortpreamble.py:466 `op = preamble_op.op.get_box_replacement()`
        // — the stored res box, not a fresh equal-positioned mint.
        let res = produced.res.clone();
        self.add_tracked_preamble_op(res, &produced);
        true
    }

    /// shortpreamble.py: add_op_to_short — recursive, export-time.
    pub fn add_op_to_short(&mut self, result: OpRef) -> Option<Op> {
        self.use_box_recursive(result, &mut FxIndexSet::with_hasher(FxBuildHasher))
    }

    /// Port of `ExtendedShortPreambleBuilder.use_box`: pop `JUMP`, add its
    /// dependencies, and append `JUMP` again.
    /// Called by `OptUnroll.force_op_from_preamble`.
    ///
    /// RPython passes `preamble_op.preamble_op` directly — the pop's
    /// replay Rc is the carried object (threaded by the produce_op
    /// family); the debug probe checks it against the builder entry.
    pub fn use_box(
        &mut self,
        source: OpRef,
        preamble_op: &majit_ir::OpRc,
        arg_guards: &[OpRc],
        result_guards: &[OpRc],
    ) {
        #[cfg(debug_assertions)]
        if let Some(produced) = self.produced_short_boxes.get(&source) {
            debug_assert!(
                OpRc::ptr_eq(&produced.preamble_op, preamble_op),
                "ext use_box pop replay diverged from builder entry at {source:?}"
            );
        }
        #[cfg(not(debug_assertions))]
        let _ = source;
        let preamble_op = (**preamble_op).clone();
        let canonical = preamble_op.pos().get();
        // shortpreamble.py:479: jump_op = self.short.pop()
        let jump_op = self.short.pop();
        // shortpreamble.py: AbstractShortPreambleBuilder.use_box(...)
        if !self.short_results.contains(&canonical) {
            // Add deps for each arg. produced_short_boxes is keyed by each
            // entry's own `preamble_op.pos`, so a dep arg (a producer position)
            // hits its entry directly — RPython single Box-identity lookup. The
            // former `preamble_op.pos → key` reverse index can never fire
            // (key == pos by construction) and was removed.
            for arg in preamble_op.args_slice().iter() {
                let arg = arg.to_opref();
                if self.short_results.contains(&arg)
                    || self.short_inputargs.contains(&arg)
                    || self.known_constants.contains(&arg)
                {
                    continue;
                }
                let dep = self.produced_short_boxes.get(&arg);
                if let Some(dep) = dep {
                    let dep_pos = dep.preamble_op.pos().get();
                    if !self.short_results.contains(&dep_pos) {
                        self.short_results.insert(dep_pos);
                        self.short.push((*dep.preamble_op).clone());
                        if dep.preamble_op.opcode.is_ovf() {
                            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
                        }
                    }
                }
            }
            self.short
                .extend(arg_guards.iter().map(|op| (**op).clone()));
            self.short_results.insert(canonical);
            self.short.push(preamble_op.clone());
            if preamble_op.opcode.is_ovf() {
                self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
            }
            self.short
                .extend(result_guards.iter().map(|op| (**op).clone()));
        }
        // shortpreamble.py:481: self.short.append(jump_op)
        if let Some(jump) = jump_op {
            self.short.push(jump);
        }
    }

    pub fn produced_short_op(&self, result: OpRef) -> Option<ProducedShortOp> {
        self.produced_short_boxes.get(&result).cloned()
    }

    pub fn short_inputargs(&self) -> &[OpRef] {
        &self.short_inputargs
    }

    pub fn build_short_preamble_struct(&self) -> ShortPreamble {
        // short[..len-1] excludes the JUMP sentinel
        let ops: Vec<majit_ir::OpRc> = self.short[..self.short_ops_len()]
            .iter()
            .map(|op| OpRc::new(op.clone()))
            .collect();
        build_short_preamble_struct_from_ops(
            &self.short_inputargs,
            &ops,
            &self.used_boxes,
            &self.short_jump_args,
        )
    }

    pub fn extra_same_as(&self) -> &[Op] {
        &self.extra_same_as
    }

    pub fn label_args(&self) -> &[OpRef] {
        &self.label_args
    }

    pub fn jump_args(&self) -> &[OpRef] {
        &self.short_jump_args
    }

    /// short ops length excluding JUMP sentinel.
    pub fn short_ops_len(&self) -> usize {
        self.short.len().saturating_sub(1)
    }

    pub fn short_op(&self, index: usize) -> Option<&Op> {
        if index < self.short_ops_len() {
            self.short.get(index)
        } else {
            None
        }
    }
}

/// shortpreamble.py: build short preamble from optimizer state.
/// Called after preamble optimization is complete.
/// Collects guards + pure ops from the optimized preamble and
/// maps them to label arg indices.
#[cfg(test)]
fn build_from_preamble_and_label(
    preamble_ops: &[Op],
    label_args: &[OpRef],
    exported_state: Option<VirtualState>,
) -> ShortPreamble {
    let mut builder = CollectedShortPreambleBuilder::new();
    let mut included_ovf_positions = FxIndexSet::with_hasher(FxBuildHasher);
    // Record all preamble ops
    for (idx, op) in preamble_ops.iter().enumerate() {
        if op.opcode.is_guard() {
            if op.opcode.is_guard_overflow()
                && idx > 0
                && preamble_ops[idx - 1].opcode.is_ovf()
                && included_ovf_positions.insert(preamble_ops[idx - 1].pos().get())
            {
                builder.add_preamble_op(&preamble_ops[idx - 1]);
            }
            builder.add_preamble_guard(op);
        } else if op.opcode.is_always_pure() {
            builder.add_preamble_op(op);
        }
    }
    // Set label args to create the mapping
    builder.set_label_args(label_args);
    builder.build(exported_state)
}

/// Extract guards AND pure ops from a peeled trace's preamble section.
///
/// Given a peeled trace (output of OptUnroll), identifies the preamble
/// section (before the Label) and collects all guard + pure operations
/// as short preamble entries.
///
/// This is a simpler alternative to integrating the builder with the
/// optimizer — it works on already-peeled traces.
pub(crate) fn extract_short_preamble(peeled_ops: &[Op]) -> ShortPreamble {
    // Find the Label position
    let label_pos = peeled_ops.iter().position(|op| op.opcode == OpCode::Label);

    let label_pos = match label_pos {
        Some(pos) => pos,
        None => return ShortPreamble::empty(), // No label = no peeling happened
    };

    let label_args = peeled_ops[label_pos].getarglist_copy();
    let label_arg_idx =
        |arg: &OpRef| -> Option<usize> { label_args.iter().position(|a| a.to_opref() == *arg) };

    // shortpreamble.py: Collect guards AND pure operations from the preamble.
    // Guards must be replayed so the body's assumptions hold.
    // Pure ops whose results are used as label args must also be replayed
    // (e.g., GETFIELD from preamble that feeds into loop body).
    let mut entries = Vec::new();
    let mut included_positions = FxIndexSet::with_hasher(FxBuildHasher);
    for (idx, op) in peeled_ops[..label_pos].iter().enumerate() {
        let mut included_overflow_producer = false;
        if op.opcode.is_guard_overflow() && idx > 0 {
            let ovf_op = &peeled_ops[idx - 1];
            if ovf_op.opcode.is_ovf() && included_positions.insert(ovf_op.pos().get()) {
                let ovf_arg_mapping: Vec<(usize, usize)> = ovf_op
                    .args_slice()
                    .iter()
                    .enumerate()
                    .filter_map(|(pos, arg)| label_arg_idx(&arg.to_opref()).map(|idx| (pos, idx)))
                    .collect();
                let ovf_fail_arg_mapping: Vec<(usize, usize)> = ovf_op
                    .guard_fail_args()
                    .map(|fail_args| {
                        fail_args
                            .iter()
                            .enumerate()
                            .filter_map(|(pos, arg)| {
                                label_arg_idx(&arg.to_opref()).map(|idx| (pos, idx))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if !ovf_arg_mapping.is_empty() || !ovf_fail_arg_mapping.is_empty() {
                    entries.push(ShortPreambleOp {
                        op: ovf_op.clone(),
                        arg_mapping: ovf_arg_mapping,
                        fail_arg_mapping: ovf_fail_arg_mapping,
                    });
                    included_overflow_producer = true;
                } else {
                    included_positions.swap_remove(&ovf_op.pos().get());
                }
            }
        }
        if op.opcode.is_guard_overflow() && !included_overflow_producer {
            continue;
        }
        let include = op.opcode.is_guard() || op.opcode.is_always_pure();
        if !include {
            continue;
        }

        let arg_mapping: Vec<(usize, usize)> = op
            .args_slice()
            .iter()
            .enumerate()
            .filter_map(|(pos, arg)| label_arg_idx(&arg.to_opref()).map(|idx| (pos, idx)))
            .collect();
        let fail_arg_mapping: Vec<(usize, usize)> = op
            .guard_fail_args()
            .map(|fail_args| {
                fail_args
                    .iter()
                    .enumerate()
                    .filter_map(|(pos, arg)| label_arg_idx(&arg.to_opref()).map(|idx| (pos, idx)))
                    .collect()
            })
            .unwrap_or_default();

        // Only include ops that reference label args
        if (!arg_mapping.is_empty() || !fail_arg_mapping.is_empty())
            && included_positions.insert(op.pos().get())
        {
            entries.push(ShortPreambleOp {
                op: op.clone(),
                arg_mapping,
                fail_arg_mapping,
            });
        }
    }

    ShortPreamble {
        ops: entries,
        inputargs: Vec::new(),
        used_boxes: Vec::new(),
        jump_args: Vec::new(),
        exported_state: None,
        constants: majit_ir::ConstMap::default(),
        inputarg_infos: Vec::new(),
    }
}

/// `unroll.py ExportedState.short_boxes` shape: per-OpRef
/// `ProducedShortOp` records derived from `PreviewShortState::short_boxes`.
///
/// The label-arg → short-inputarg rename happens at EXPORT time inside
/// `produce_arg` (shortpreamble.py/294): every short op the import
/// path emits (Pure / Heap / LoopInvariant) already carries the renamed
/// `short_inputargs` box in its args. The only exported entries that still
/// reference an original label box are the `ShortInputArg` `SameAs*`
/// stand-ins, and those are never emitted (`produce_op` returns None for
/// `InputArg`, shortpreamble.py) nor read by any import consumer:
/// `result_map` and the `produce_op` loop skip `InputArg`, the builder
/// init skips entries with no `result_map` slot, and `use_box_recursive`'s
/// arg recursion is rename-invariant for the `SameAs` self-reference. So
/// no import-time arg rewrite is needed — the produced view is a plain
/// filter + transform of `exported_short_boxes`.
///
/// OVF guards are filtered out: the guard entry depends on the
/// preceding `Int*Ovf` op and is re-emitted by the builder through
/// `append_to_short`'s `is_ovf` branch, so the standalone guard must
/// not appear in the produced map.
pub(crate) fn produced_short_boxes_from_exported_boxes(
    exported_short_boxes: &[PreambleOp],
) -> Vec<(OpRef, ProducedShortOp)> {
    exported_short_boxes
        .iter()
        .filter(|entry| !entry.op.opcode.is_guard_overflow())
        .map(|entry| {
            // Fresh clone per entry so the builder's on-object forwarded
            // marker (`ShortPreambleBuilder::new`) mutates an isolated Rc.
            let preamble_op = (*entry.op).clone();
            (
                preamble_op.pos().get(),
                ProducedShortOp {
                    kind: entry.kind.clone(),
                    // shortpreamble.py __init__/110 short_op.res — the exported
                    // entry carries the Phase-1 res operand across the
                    // boundary; it already holds the live producer / const
                    // operand (#173 roots the producer, never position-only).
                    res: entry.res.clone(),
                    preamble_op: OpRc::new(preamble_op),
                    source_op: entry.source_op.clone().unwrap_or_else(|| entry.op.clone()),
                    invented_name: entry.invented_name,
                    same_as_source: entry.same_as_source.clone(),
                    label_arg_idx: entry.label_arg_idx,
                },
            )
        })
        .collect()
}

/// `unroll.py:497-504` build path: drive `ShortPreambleBuilder` from
/// pre-derived `produced_short_boxes` (RPython `ExportedState.short_boxes`).
///
/// Separated from
/// `build_short_preamble_from_exported_boxes` so callers that already
/// hold the `Vec<(OpRef, ProducedShortOp)>` (e.g. `ExportedState.produced_short_boxes`
/// in `unroll.rs`) can invoke the builder directly without
/// re-running the rename + filter pass.
pub(crate) fn build_short_preamble_from_produced_boxes(
    label_args: &[OpRef],
    short_inputargs: &[OpRef],
    produced: &[(OpRef, ProducedShortOp)],
) -> ShortPreamble {
    // #146: the builder map keys by the entry res Box (`p.res`, the carried
    // Phase-1 short-box result). The input slice is keyed by `preamble_op.pos`;
    // re-key to res for the builder map and the use_box driving loop (const res
    // is ptr-unstable and never a key — export filters it).
    //
    // Correctness here rests on the OUTER loop below: it calls add_op_to_short
    // on EVERY entry in input order, which is dependency order (RPython
    // create_short_boxes appends deps before consumers). `use_box_recursive`'s
    // dep pre-append is a best-effort ordering aid whose box-identity dep
    // lookup may miss (an op's dep arg need not Rc::ptr_eq the dependency's res
    // box), but a miss only drops the redundant recursive append — the outer
    // loop still emits every entry in valid def-before-use order. This path is
    // a corpus-dead fallback (rebuilt preamble is Some in the measured corpus).
    let entries: Vec<(majit_ir::operand::Operand, ProducedShortOp)> = produced
        .iter()
        .filter(|(_, p)| !p.res.to_opref().is_constant())
        .map(|(_, p)| (p.res.clone(), p.clone()))
        .collect();
    let mut builder = ShortPreambleBuilder::new(label_args, &entries, short_inputargs);
    // history.py/268/314 — inline-Const variants carry the value on
    // the OpRef; `is_constant()` returns true intrinsically and the
    // `is_reachable` / `add_heap_op` checks short-circuit before
    // consulting `known_constants`. There are no constant pool entries
    // to seed into `known_constants` and the `loop_constants`
    // parameter has been retired.
    for (res, _) in &entries {
        let _ = builder.add_op_to_short(res);
        let _ = builder.add_preamble_op(res);
    }
    builder.build_short_preamble_struct()
}

#[cfg(test)]
fn build_short_preamble_from_exported_boxes(
    label_args: &[OpRef],
    short_inputargs: &[OpRef],
    exported_short_boxes: &[PreambleOp],
) -> ShortPreamble {
    let produced = produced_short_boxes_from_exported_boxes(exported_short_boxes);
    build_short_preamble_from_produced_boxes(label_args, short_inputargs, &produced)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::test_support::rooted_resop_operand;
    use majit_ir::operand::Operand;
    use majit_ir::{Op, OpCode, OpRc, OpRef, Type};

    /// oparser-faithful op-arg minter: a rooted resop producer shed to its
    /// `Operand::Op` face. Op-arg slices take `Operand`.
    fn rop(ty: Type, pos: u32) -> Operand {
        rooted_resop_operand(ty, pos)
    }

    /// The context's canonical box for `pos`, so every use of one position
    /// is the same box.
    fn crop(ctx: &mut crate::optimizeopt::OptContext, ty: Type, pos: u32) -> Operand {
        ctx.materialize_operand_at(OpRef::op_typed(pos, ty))
    }

    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos()
                .set(OpRef::op_typed(base + i as u32, op.result_type()));
        }
    }

    /// Direct port of RPython test_short.py::TestShortBoxes::test_pure_ops.
    #[test]
    fn test_rpython_pure_ops_contract() {
        let mut ctx =
            crate::optimizeopt::OptContext::with_inputarg_types(16, &[Type::Int, Type::Int]);
        let i0 = OpRef::input_arg_int(0);
        let i1 = OpRef::input_arg_int(1);
        let mut add = Op::new(
            OpCode::IntAdd,
            &[
                ctx.materialize_operand_at(i0),
                ctx.materialize_operand_at(i1),
            ],
        );
        add.pos().set(OpRef::int_op(2));

        let mut sb = ShortBoxes::with_label_args(&[i0, i1]);
        sb.add_pure_op(&mut ctx, &OpRc::new(add));
        let short_boxes = sb.create_short_boxes(&mut ctx, &[i0, i1], &[Type::Int, Type::Int]);

        assert_eq!(short_boxes.len(), 3);
        let short_inputargs = sb.create_short_inputargs(&[i0, i1]);
        let pure = short_boxes
            .iter()
            .find(|produced| produced.kind == PreambleOpKind::Pure)
            .expect("reachable IntAdd must be exported");
        assert_eq!(
            pure.preamble_op
                .args_slice()
                .iter()
                .map(|arg| arg.to_opref())
                .collect::<Vec<_>>(),
            short_inputargs
        );
    }

    /// Direct port of
    /// RPython test_short.py::TestShortBoxes::test_pure_ops_does_not_work.
    #[test]
    fn test_rpython_pure_ops_missing_dependency_is_not_exported() {
        let mut ctx =
            crate::optimizeopt::OptContext::with_inputarg_types(16, &[Type::Int, Type::Int]);
        let i0 = OpRef::input_arg_int(0);
        let i1 = OpRef::input_arg_int(1);
        let mut add = Op::new(
            OpCode::IntAdd,
            &[
                ctx.materialize_operand_at(i0),
                ctx.materialize_operand_at(i1),
            ],
        );
        add.pos().set(OpRef::int_op(2));

        let mut sb = ShortBoxes::with_label_args(&[i0]);
        sb.add_pure_op(&mut ctx, &OpRc::new(add));
        let short_boxes = sb.create_short_boxes(&mut ctx, &[i0], &[Type::Int]);

        assert_eq!(short_boxes.len(), 1);
        assert_eq!(short_boxes[0].kind, PreambleOpKind::InputArg);
    }

    #[test]
    fn test_extract_from_peeled_trace() {
        // Simulate a peeled trace:
        // 0: guard_true(v100)        ← preamble guard on loop-carried value
        // 1: int_add(v100, v101)     ← preamble computation
        // 2: Label(v100, v101)       ← loop header
        // 3: guard_true(v100)        ← body guard (same)
        // 4: int_add(v100, v101)     ← body computation
        // 5: Jump(v4, v101)          ← back-edge
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]),
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Label, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]),
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Jump, &[rop(Type::Int, 4), rop(Type::Int, 101)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // Should have captured the preamble guard AND the pure IntAdd
        assert_eq!(sp.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::GuardTrue);
        assert_eq!(sp.ops[1].op.opcode, OpCode::IntAdd);

        // The guard's arg v100 maps to label arg index 0
        assert_eq!(sp.ops[0].arg_mapping.len(), 1);
        assert_eq!(sp.ops[0].arg_mapping[0], (0, 0)); // arg position 0 → label arg 0
    }

    #[test]
    fn test_extract_no_label() {
        // No label = no peeling happened
        let ops = vec![
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Finish, &[rop(Type::Int, 0)]),
        ];

        let sp = extract_short_preamble(&ops);
        assert!(sp.is_empty());
    }

    #[test]
    fn test_extract_overflow_guard_includes_preceding_ovf_op() {
        let mut ops = vec![
            Op::new(
                OpCode::IntMulOvf,
                &[rop(Type::Int, 100), rop(Type::Int, 100)],
            ),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Label, &[rop(Type::Int, 100)]),
            Op::new(OpCode::Jump, &[rop(Type::Int, 100)]),
        ];
        assign_positions(&mut ops, 0);
        ops[1].setfailargs(vec![rooted_resop_operand(Type::Int, 100)].into());
        let sp = extract_short_preamble(&ops);

        assert_eq!(sp.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntMulOvf);
        assert_eq!(sp.ops[1].op.opcode, OpCode::GuardNoOverflow);
    }

    #[test]
    fn test_extract_overflow_guard_without_replayable_ovf_is_skipped() {
        let mut ops = vec![
            Op::new(
                OpCode::IntMulOvf,
                &[rop(Type::Int, 200), rop(Type::Int, 200)],
            ),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Label, &[rop(Type::Int, 100)]),
            Op::new(OpCode::Jump, &[rop(Type::Int, 100)]),
        ];
        assign_positions(&mut ops, 0);
        ops[1].setfailargs(vec![rooted_resop_operand(Type::Int, 100)].into());
        let sp = extract_short_preamble(&ops);

        assert!(sp.is_empty());
    }

    #[test]
    fn test_extract_skips_non_label_guards() {
        // Guards that don't reference label args should not be included
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 0)]), // refs temporary, not label arg
            Op::new(OpCode::Label, &[rop(Type::Int, 100)]),   // only v100 is a label arg
            Op::new(OpCode::Jump, &[rop(Type::Int, 100)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // The guard refs v0 (the IntAdd result), which is NOT in the label args.
        // But IntAdd refs v100 which IS a label arg → IntAdd IS extracted.
        // The guard on v0 is NOT extracted (v0 is not a label arg).
        assert_eq!(sp.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_builder_collects_guards() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // Simulate preamble processing
        let guard1 = Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]);
        let guard2 = Op::new(
            OpCode::GuardClass,
            &[rop(Type::Int, 101), rop(Type::Int, 200)],
        );
        let non_guard = Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]);

        builder.add_preamble_guard(&guard1);
        builder.add_preamble_guard(&guard2);
        builder.add_preamble_guard(&non_guard); // should be ignored (not a guard)

        // Set label args (preamble phase ends)
        builder.set_label_args(&[OpRef::int_op(100), OpRef::int_op(101)]);

        // After label, no more collection
        let guard3 = Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]);
        builder.add_preamble_guard(&guard3); // should be ignored

        let sp = builder.build(None);
        assert_eq!(sp.len(), 2); // Only the two guards from the preamble
    }

    #[test]
    fn test_builder_maps_args_to_label_indices() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // Preamble has guard on v100 and v101
        let guard = Op::new(
            OpCode::GuardValue,
            &[rop(Type::Int, 100), rop(Type::Int, 200)],
        );
        builder.add_preamble_guard(&guard);

        // Label carries v100 as arg 0 and v101 as arg 1
        builder.set_label_args(&[OpRef::int_op(100), OpRef::int_op(101)]);

        let sp = builder.build(None);
        assert_eq!(sp.ops[0].arg_mapping.len(), 1); // v100 → label arg 0
        assert_eq!(sp.ops[0].arg_mapping[0], (0, 0));
        // v200 is not a label arg, so it's not in the mapping
    }

    #[test]
    fn test_builder_add_preamble_op_any_type() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // add_preamble_op accepts any op type (not just guards)
        let pure_op = Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]);
        builder.add_preamble_op(&pure_op);

        builder.set_label_args(&[OpRef::int_op(100), OpRef::int_op(101)]);

        let sp = builder.build(None);
        assert_eq!(sp.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_extract_multiple_guards() {
        // Multiple guards in the preamble
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]),
            Op::new(OpCode::GuardNonnull, &[rop(Type::Int, 101)]),
            Op::new(
                OpCode::GuardClass,
                &[rop(Type::Int, 100), rop(Type::Int, 200)],
            ),
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Label, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Jump, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // All three guards + the pure IntAdd reference label args
        assert_eq!(sp.len(), 4);
        assert_eq!(sp.ops[0].op.opcode, OpCode::GuardTrue);
        assert_eq!(sp.ops[1].op.opcode, OpCode::GuardNonnull);
        assert_eq!(sp.ops[2].op.opcode, OpCode::GuardClass);
        assert_eq!(sp.ops[3].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_build_short_preamble_from_exported_boxes_uses_exported_order() {
        // shortpreamble.py:285/294: the label-arg → short-inputarg rename
        // happens at EXPORT time (produce_arg), so the exported short ops
        // already carry the renamed `short_inputargs` boxes (10/11) in place
        // of the original label args (0/1). The op-result positions (7/8) are
        // not inputargs and are unchanged.
        let exported = vec![
            PreambleOp {
                source_op: None,
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[rop(Type::Int, 10), rop(Type::Int, 11)]);
                    op.pos().set(OpRef::int_op(7));
                    OpRc::new(op)
                },
                res: rooted_resop_operand(Type::Int, 7),
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
            PreambleOp {
                source_op: None,
                op: {
                    let mut op = Op::new(OpCode::IntSub, &[rop(Type::Int, 7), rop(Type::Int, 11)]);
                    op.pos().set(OpRef::int_op(8));
                    OpRc::new(op)
                },
                res: rooted_resop_operand(Type::Int, 8),
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
        ];

        let sp = build_short_preamble_from_exported_boxes(
            &[OpRef::int_op(0), OpRef::int_op(1)],
            &[OpRef::int_op(10), OpRef::int_op(11)],
            &exported,
        );
        assert_eq!(sp.ops.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(sp.ops[1].op.opcode, OpCode::IntSub);
        assert_eq!(sp.ops[1].arg_mapping, vec![(1, 1)]);
        assert_eq!(sp.inputargs, vec![OpRef::int_op(10), OpRef::int_op(11)]);
    }

    #[test]
    fn test_build_short_preamble_from_exported_boxes_skips_standalone_overflow_guards() {
        let label_args = vec![OpRef::int_op(10), OpRef::int_op(11)];
        let short_inputargs = vec![OpRef::int_op(100), OpRef::int_op(101)];

        let mut ovf = Op::new(OpCode::IntAddOvf, &[rop(Type::Int, 10), rop(Type::Int, 11)]);
        ovf.pos().set(OpRef::int_op(20));
        let guard = Op::new(OpCode::GuardNoOverflow, &[]);

        let exported = vec![
            PreambleOp {
                source_op: None,
                op: OpRc::new(ovf),
                res: rooted_resop_operand(Type::Int, 20),
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
            PreambleOp {
                source_op: None,
                op: OpRc::new(guard),
                res: majit_ir::operand::Operand::None,
                kind: PreambleOpKind::Guard,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
        ];

        let sp = build_short_preamble_from_exported_boxes(&label_args, &short_inputargs, &exported);
        let opcodes: Vec<OpCode> = sp.ops.iter().map(|entry| entry.op.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::IntAddOvf, OpCode::GuardNoOverflow]);
    }

    #[test]
    fn test_rpython_short_preamble_builder_add_op_to_short_recurses_dependencies() {
        // #146: the builder map keys by the short-box res Box identity, and a
        // dependency is found when an op's arg Box Rc::ptr_eq's the dep's res
        // (RPython box-identity `if arg in produced_short_boxes`). Express the
        // B-depends-on-A edge by binding B's IntMul arg(0) to A's producer op
        // via `from_bound_op` — the memoized result box is ptr_eq to A's `res`,
        // so the dependency lookup hits and `Op::new` round-trips the same box
        // through `getarglist()` (vs a position-only `from_opref`, which would
        // mint a non-`ptr_eq` arg box).
        let in0 = rooted_resop_operand(Type::Int, 0);
        let in1 = rooted_resop_operand(Type::Int, 1);
        let producer7 = {
            let mut op = Op::new(OpCode::IntAdd, &[in0.clone(), in1.clone()]);
            op.pos().set(OpRef::int_op(7));
            OpRc::new(op)
        };
        let res7 = Operand::from_bound_op(&producer7);
        let producer8 = {
            let mut op = Op::new(OpCode::IntMul, &[res7.clone(), in1.clone()]);
            op.pos().set(OpRef::int_op(8));
            OpRc::new(op)
        };
        let res8 = Operand::from_bound_op(&producer8);
        let produced = vec![
            (
                res7.clone(),
                ProducedShortOp {
                    kind: PreambleOpKind::Pure,
                    res: res7.clone(),
                    preamble_op: producer7.clone(),
                    source_op: producer7,
                    invented_name: false,
                    same_as_source: None,
                    label_arg_idx: None,
                },
            ),
            (
                res8.clone(),
                ProducedShortOp {
                    kind: PreambleOpKind::Pure,
                    res: res8.clone(),
                    preamble_op: producer8.clone(),
                    source_op: producer8,
                    invented_name: false,
                    same_as_source: None,
                    label_arg_idx: None,
                },
            ),
        ];
        let mut builder = ShortPreambleBuilder::new(
            &[OpRef::int_op(0), OpRef::int_op(1)],
            &produced,
            &[OpRef::int_op(0), OpRef::int_op(1)],
        );

        let used = builder.add_op_to_short(&res8).unwrap();
        assert!(builder.add_preamble_op(&res7));
        assert!(builder.add_preamble_op(&res8));
        assert_eq!(used.opcode, OpCode::IntMul);
        let short = builder.build_short_preamble();
        assert_eq!(short[1].opcode, OpCode::IntAdd);
        assert_eq!(short[2].opcode, OpCode::IntMul);
        assert_eq!(builder.used_boxes(), &[OpRef::int_op(7), OpRef::int_op(8)]);
    }

    #[test]
    fn test_build_from_preamble_and_label() {
        let mut preamble = vec![
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]),
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
        ];
        assign_positions(&mut preamble, 0);

        let label_args = &[OpRef::int_op(100), OpRef::int_op(101)];
        let sp = build_from_preamble_and_label(&preamble, label_args, None);

        // Guard + pure IntAdd
        assert_eq!(sp.len(), 2);
    }

    #[test]
    fn test_extended_builder() {
        let mut builder = CollectedExtendedShortPreambleBuilder::new();
        builder.set_label_args(&[OpRef::int_op(100), OpRef::int_op(101)]);
        builder.add_guard(Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]));
        builder.add_pure_op(Op::new(
            OpCode::IntAdd,
            &[rop(Type::Int, 100), rop(Type::Int, 101)],
        ));
        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rop(Type::Int, 100)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos().set(OpRef::int_op(102));
        builder.add_heap_op(heap);
        builder.add_loopinvariant_op(Op::new(OpCode::CallI, &[rop(Type::Int, 100)]));
        assert_eq!(builder.num_ops(), 4);
    }

    #[test]
    fn test_short_boxes() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb =
            ShortBoxes::with_label_args(&[OpRef::int_op(10), OpRef::int_op(11), OpRef::int_op(12)]);
        assert_eq!(sb.num_label_args, 3);
        // Production seeds a renamed ShortInputArg for every label arg
        // (optimizer.rs preview loop); the pure/heap deps resolve through them.
        for arg in [OpRef::int_op(10), OpRef::int_op(11), OpRef::int_op(12)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }
        let mut pure = Op::new(
            OpCode::IntAdd,
            &[
                crop(&mut __ctx, Type::Int, 10),
                crop(&mut __ctx, Type::Int, 11),
            ],
        );
        pure.pos().set(OpRef::int_op(20));
        sb.add_pure_op(&mut __ctx, &OpRc::new(pure));
        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[crop(&mut __ctx, Type::Int, 10)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos().set(OpRef::int_op(21));
        {
            let res = __ctx.materialize_operand_at(heap.pos().get());
            sb.add_heap_op(&mut __ctx, res, heap);
        }
        let produced = sb.produced_ops(&mut __ctx);
        // 3 ShortInputArgs (one per label arg) + the pure and heap ops.
        let non_input: Vec<_> = produced
            .iter()
            .filter(|(_, p)| p.kind != PreambleOpKind::InputArg)
            .collect();
        assert_eq!(non_input.len(), 2);
    }

    #[test]
    fn test_short_boxes_reject_unknown_nonconstant_dependency() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(10)]);
        sb.add_short_input_arg(&mut __ctx, OpRef::int_op(10), majit_ir::Type::Int);
        let mut pure = Op::new(OpCode::IntAdd, &[rop(Type::Int, 10), rop(Type::Int, 999)]);
        pure.pos().set(OpRef::int_op(20));
        sb.add_pure_op(&mut __ctx, &OpRc::new(pure));

        let produced = sb.produced_ops(&mut __ctx);
        // The label arg OpRef::int_op(10) itself is produced (as ShortInputArg),
        // but the pure op depending on unknown OpRef::int_op(999) is rejected.
        assert!(
            !produced.iter().any(|(r, _)| *r == OpRef::int_op(20)),
            "pure op with unknown dependency should be rejected"
        );
    }

    #[test]
    fn test_short_boxes_accept_known_constant_dependency() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(10)]);
        sb.add_short_input_arg(&mut __ctx, OpRef::int_op(10), majit_ir::Type::Int);
        sb.note_known_constant(OpRef::int_op(999));
        let mut pure = Op::new(
            OpCode::IntAdd,
            &[
                crop(&mut __ctx, Type::Int, 10),
                crop(&mut __ctx, Type::Int, 999),
            ],
        );
        pure.pos().set(OpRef::int_op(20));
        sb.add_pure_op(&mut __ctx, &OpRc::new(pure));

        let produced = sb.produced_ops(&mut __ctx);
        let renamed10 = sb.create_short_inputargs(&[OpRef::int_op(10)])[0];
        // 1 ShortInputArg (label arg 10) + the accepted pure op.
        let non_input: Vec<_> = produced
            .iter()
            .filter(|(_, p)| p.kind != PreambleOpKind::InputArg)
            .collect();
        assert_eq!(non_input.len(), 1);
        let pure = produced
            .iter()
            .find(|(result, _)| *result == OpRef::int_op(20))
            .expect("missing produced pure op");
        // The label-arg dep is renamed to its short_inputargs box; the known
        // constant 999 passes through unchanged.
        assert_eq!(
            pure.1
                .preamble_op
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![renamed10, OpRef::int_op(999)]
        );
    }

    #[test]
    fn test_short_boxes_compound_prefers_non_heap_and_emits_invented_alias() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        // Only the op DEPENDENCIES (30, 31) are label args; the compound result
        // (pos 10) is an op result, not a label arg.
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(30), OpRef::int_op(31)]);
        for arg in [OpRef::int_op(30), OpRef::int_op(31)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[crop(&mut __ctx, Type::Int, 30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos().set(OpRef::int_op(10));
        {
            let res = __ctx.materialize_operand_at(heap.pos().get());
            sb.add_potential_op(&mut __ctx, None, res, heap, PreambleOpKind::Heap);
        }

        let mut pure = Op::new(
            OpCode::IntAdd,
            &[
                crop(&mut __ctx, Type::Int, 30),
                crop(&mut __ctx, Type::Int, 31),
            ],
        );
        pure.pos().set(OpRef::int_op(10));
        // Second add wraps the first value in CompoundOp (shortpreamble.py
        // add_potential_op). produced_ops must flatten both after the
        // wrap moves `prev` instead of cloning it.
        {
            let res = __ctx.materialize_operand_at(pure.pos().get());
            sb.add_potential_op(&mut __ctx, None, res, pure, PreambleOpKind::Pure);
        }

        let produced = sb.produced_ops(&mut __ctx);
        // 2 ShortInputArgs (30, 31) + the compound@10 (Pure chosen + Heap alias).
        let non_input_count = produced
            .iter()
            .filter(|(_, p)| p.kind != PreambleOpKind::InputArg)
            .count();
        assert_eq!(non_input_count, 2);

        let chosen = produced
            .iter()
            .find(|(result, p)| *result == OpRef::int_op(10) && p.kind != PreambleOpKind::InputArg)
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::Pure);
        assert!(!chosen.1.invented_name);

        let alias = produced
            .iter()
            .find(|(result, p)| *result != OpRef::int_op(10) && p.kind != PreambleOpKind::InputArg)
            .unwrap();
        assert_eq!(alias.1.kind, PreambleOpKind::Heap);
        assert!(alias.1.invented_name);
        // shortpreamble.py HeapOp keeps `getfield_op` beside the remapped
        // `preamble_op`: exported_infos is keyed by the original receiver
        // box, while the replay must read the renamed short-inputarg box.
        assert_eq!(alias.1.source_op.arg(0).to_opref(), OpRef::int_op(30));
        assert_ne!(
            alias.1.preamble_op.arg(0).to_opref(),
            alias.1.source_op.arg(0).to_opref()
        );
        assert_eq!(
            alias.1.same_as_source.as_ref().map(|b| b.to_opref()),
            Some(OpRef::int_op(10))
        );
    }

    fn heap_oprc_strong_count(op: &PotentialShortOp) -> Option<usize> {
        match op {
            PotentialShortOp::Preamble(p) if p.kind == PreambleOpKind::Heap => {
                Some(OpRc::strong_count(&p.op))
            }
            PotentialShortOp::Compound(c) => (c.one.kind == PreambleOpKind::Heap)
                .then(|| OpRc::strong_count(&c.one.op))
                .or_else(|| heap_oprc_strong_count(&c.two)),
            _ => None,
        }
    }

    #[test]
    fn add_potential_op_moves_prev_into_compound_without_cloning_oprc() {
        // shortpreamble.py add_potential_op: `CompoundOp(op, pop, prev_op)`
        // stores the previous dict value. Cloning PreambleOp clones its
        // OpRc and bumps the strong count; a move leaves the count alone.
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(30)]);
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(30), majit_ir::Type::Int);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rop(Type::Int, 30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos().set(OpRef::int_op(10));
        {
            let res = ctx.materialize_operand_at(heap.pos().get());
            sb.add_potential_op(&mut ctx, None, res, heap, PreambleOpKind::Heap);
        }
        let count_before = sb
            .potential_ops
            .values()
            .find_map(|pop| heap_oprc_strong_count(pop))
            .expect("heap producer is in potential_ops");

        let mut pure = Op::new(OpCode::IntAdd, &[rop(Type::Int, 30), rop(Type::Int, 30)]);
        pure.pos().set(OpRef::int_op(10));
        {
            let res = ctx.materialize_operand_at(pure.pos().get());
            sb.add_potential_op(&mut ctx, None, res, pure, PreambleOpKind::Pure);
        }
        let count_after = sb
            .potential_ops
            .values()
            .find_map(|pop| heap_oprc_strong_count(pop))
            .expect("heap producer is inside the CompoundOp");
        assert_eq!(
            count_before, count_after,
            "add_potential_op must move prev into CompoundOp; cloning PreambleOp bumps OpRc"
        );
    }

    #[test]
    fn test_short_boxes_nested_compound_emits_multiple_invented_aliases() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        // Only the op DEPENDENCIES (30, 31) are label args; the compound result
        // (pos 20) is an op result, not a label arg.
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(30), OpRef::int_op(31)]);
        for arg in [OpRef::int_op(30), OpRef::int_op(31)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[crop(&mut __ctx, Type::Int, 30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos().set(OpRef::int_op(20));
        {
            let res = __ctx.materialize_operand_at(heap.pos().get());
            sb.add_potential_op(&mut __ctx, None, res, heap, PreambleOpKind::Heap);
        }

        let mut loopinv = Op::new(OpCode::CallI, &[crop(&mut __ctx, Type::Int, 30)]);
        loopinv.pos().set(OpRef::int_op(20));
        {
            let res = __ctx.materialize_operand_at(loopinv.pos().get());
            sb.add_potential_op(
                &mut __ctx,
                None,
                res,
                loopinv,
                PreambleOpKind::LoopInvariant,
            );
        }

        let mut pure = Op::new(
            OpCode::IntAdd,
            &[
                crop(&mut __ctx, Type::Int, 30),
                crop(&mut __ctx, Type::Int, 31),
            ],
        );
        pure.pos().set(OpRef::int_op(20));
        {
            let res = __ctx.materialize_operand_at(pure.pos().get());
            sb.add_potential_op(&mut __ctx, None, res, pure, PreambleOpKind::Pure);
        }

        let produced = sb.produced_ops(&mut __ctx);
        // 2 ShortInputArgs (30, 31) + the compound@20 (Pure chosen + 2 aliases).
        let non_input_count = produced
            .iter()
            .filter(|(_, p)| p.kind != PreambleOpKind::InputArg)
            .count();
        assert_eq!(non_input_count, 3);

        let chosen = produced
            .iter()
            .find(|(result, p)| *result == OpRef::int_op(20) && p.kind != PreambleOpKind::InputArg)
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::Pure);
        assert!(!chosen.1.invented_name);

        let aliases: Vec<_> = produced
            .iter()
            .filter(|(result, p)| {
                *result != OpRef::int_op(20) && p.kind != PreambleOpKind::InputArg
            })
            .collect();
        assert_eq!(aliases.len(), 2);
        assert!(aliases.iter().all(|(_, produced)| produced.invented_name));
        assert!(aliases.iter().all(|(_, produced)| {
            produced.same_as_source.as_ref().map(|b| b.to_opref()) == Some(OpRef::int_op(20))
        }));
    }

    #[test]
    fn test_rpython_create_short_boxes_prefers_short_inputarg_over_heap_result() {
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(10), OpRef::int_op(30)]);
        // pos 10 is both a label arg and a heap result (the case under test);
        // seed both label args (the heap dep 30 must be a ShortInputArg too).
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(10), majit_ir::Type::Int);
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(30), majit_ir::Type::Int);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[crop(&mut ctx, Type::Int, 30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos().set(OpRef::int_op(10));
        {
            let res = ctx.materialize_operand_at(heap.pos().get());
            sb.add_heap_op(&mut ctx, res, heap);
        }

        let produced = sb.produced_ops(&mut ctx);

        // The compound at pos 10 prefers the ShortInputArg; the Heap becomes an
        // invented alias. (Label arg 30 is also produced as a ShortInputArg.)
        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef::int_op(10))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::InputArg);
        assert!(!chosen.1.invented_name);

        let alias = produced
            .iter()
            .find(|(_, p)| p.kind == PreambleOpKind::Heap)
            .unwrap();
        assert!(alias.1.invented_name);
        assert_eq!(
            alias.1.same_as_source.as_ref().map(|b| b.to_opref()),
            Some(OpRef::int_op(10))
        );
    }

    /// shortpreamble.py:272-280 -- const heap reads survive the production
    /// split and keep the const result distinct from the replay operation.
    #[test]
    fn const_heap_short_box_survives_production() {
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(16, &[Type::Ref]);
        let struct_arg = OpRef::input_arg_ref(0);
        let constant = OpRef::const_int(7);
        let mut sb = ShortBoxes::with_label_args(&[struct_arg]);
        sb.add_short_input_arg(&mut ctx, struct_arg, Type::Ref);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[ctx.materialize_operand_at(struct_arg)],
            majit_ir::make_field_descr(0, 8, Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos().set(constant);
        {
            let res = ctx.materialize_operand_at(heap.pos().get());
            sb.add_heap_op(&mut ctx, res, heap);
        }

        let produced = sb.produced_ops(&mut ctx);
        let produced_const = sb.produced_const_ops(&mut ctx);

        assert!(
            produced
                .iter()
                .all(|(_, produced)| produced.res.to_opref() != constant),
            "produced_ops must not carry the const result"
        );
        assert_eq!(produced_const.len(), 1);
        assert_eq!(produced_const[0].kind, PreambleOpKind::Heap);
        assert_eq!(produced_const[0].res.to_opref(), constant);
        assert_ne!(produced_const[0].preamble_op.pos().get(), constant);
    }

    #[test]
    fn descrless_const_heap_short_box_is_skipped_not_panicked() {
        // The sibling above builds the heap op through `Op::with_descr`, so the
        // field descr the const channel needs is always there. A heap short box
        // can reach this point without one, and the channel has to drop it
        // rather than unwrap: without a descr there is nothing to re-emit the
        // load from, and a produced entry would name a field the import cannot
        // resolve.
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(16, &[Type::Ref]);
        let struct_arg = OpRef::input_arg_ref(0);
        let constant = OpRef::const_int(7);
        let mut sb = ShortBoxes::with_label_args(&[struct_arg]);
        sb.add_short_input_arg(&mut ctx, struct_arg, Type::Ref);

        let heap = Op::new(
            OpCode::GetfieldGcI,
            &[ctx.materialize_operand_at(struct_arg)],
        );
        heap.pos().set(constant);
        {
            let res = ctx.materialize_operand_at(heap.pos().get());
            sb.add_heap_op(&mut ctx, res, heap);
        }

        let produced = sb.produced_ops(&mut ctx);
        let produced_const = sb.produced_const_ops(&mut ctx);

        assert!(
            produced
                .iter()
                .all(|(_, produced)| produced.res.to_opref() != constant),
            "produced_ops must not carry the const result"
        );
        assert!(
            produced_const.is_empty(),
            "a descr-less const heap short box is skipped, not produced"
        );
    }

    #[test]
    fn test_compound_pure_loser_to_short_inputarg_clears_label_arg_idx() {
        // shortpreamble.py `ShortBoxes.add_op_to_short` — when a Pure
        // alternative loses the compound tie to the ShortInputArg, its result
        // is rebound to a fresh SameAs box (`lst[i].short_op.res = new_name`),
        // so the invented alias is not a label arg. Slot identity belongs only
        // to `ShortInputArg`, so `label_arg_idx` is cleared on the alias.
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb =
            ShortBoxes::with_label_args(&[OpRef::int_op(10), OpRef::int_op(30), OpRef::int_op(31)]);
        // A pure op whose result box is label arg 10, depending on the other
        // two label args (avoids a self-referential in-production cycle).
        let pure = Op::new(
            OpCode::IntAdd,
            &[crop(&mut ctx, Type::Int, 30), crop(&mut ctx, Type::Int, 31)],
        );
        pure.pos().set(OpRef::int_op(10));
        let pure = OpRc::new(pure);
        ctx.emit_rc(pure.clone());
        // pos 10 is both a label arg (slot 0) and a pure result (the case under
        // test); seed all three label args (the pure deps 30/31 are
        // ShortInputargs too).
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(10), majit_ir::Type::Int);
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(30), majit_ir::Type::Int);
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(31), majit_ir::Type::Int);

        sb.add_pure_op(&mut ctx, &pure);

        let produced = sb.produced_ops(&mut ctx);

        // The compound at pos 10 prefers the ShortInputArg; the Pure becomes an
        // invented alias.
        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef::int_op(10))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::InputArg);
        assert!(!chosen.1.invented_name);
        // The winning ShortInputArg keeps its label slot (box 10 = slot 0).
        assert_eq!(chosen.1.label_arg_idx, Some(0));

        let alias = produced
            .iter()
            .find(|(_, p)| p.kind == PreambleOpKind::Pure)
            .unwrap();
        assert!(alias.1.invented_name);
        assert_eq!(
            alias.1.same_as_source.as_ref().map(|b| b.to_opref()),
            Some(OpRef::int_op(10))
        );
        // THE REGRESSION LOCK: the invented alias must NOT carry the original
        // label slot. Before the fix this was Some(0) and the import collapsed
        // the alias onto the loop-carried input.
        assert_eq!(alias.1.label_arg_idx, None);
    }

    #[test]
    fn test_short_inputargs_have_distinct_renamed_identity() {
        // shortpreamble.py `renamed = OpHelpers.inputarg_from_tp(box.type)`
        // — each short inputarg is a FRESH InputArg distinct from the original
        // label arg `box`. pyre's old position-identity model reused the label
        // arg's OpRef; this test locks the distinct-renamed-box parity.
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        let label_args = [OpRef::int_op(10), OpRef::ref_op(11)];
        let mut sb = ShortBoxes::with_label_args(&label_args);
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(10), majit_ir::Type::Int);
        sb.add_short_input_arg(&mut ctx, OpRef::ref_op(11), majit_ir::Type::Ref);

        let si = sb.create_short_inputargs(&label_args);
        assert_eq!(si.len(), 2);

        // (A) DISTINCT identity: the renamed box is not the original label arg.
        assert_ne!(si[0], OpRef::int_op(10));
        assert_ne!(si[1], OpRef::ref_op(11));

        // (B) the renamed boxes are InputArg-kind of the matching type
        // (inputarg_from_tp(box.type)).
        assert!(matches!(si[0], OpRef::InputArgInt(_)));
        assert!(matches!(si[1], OpRef::InputArgRef(_)));
        assert_eq!(si[0].ty(), Some(majit_ir::Type::Int));
        assert_eq!(si[1].ty(), Some(majit_ir::Type::Ref));

        // (C) lookups still resolve a label arg by its ORIGINAL opref
        // (shortpreamble.py:259 potential_ops[box]).
        assert_eq!(sb.lookup_label_arg(OpRef::int_op(10)), Some(0));
        assert_eq!(sb.lookup_label_arg(OpRef::ref_op(11)), Some(1));
        assert!(sb.is_reachable(OpRef::int_op(10)));
    }

    #[test]
    fn test_duplicate_slot_keeps_last_renamed_inputarg() {
        // shortpreamble.py:255-259 — when a box appears at TWO slots of the
        // combined `label_args + virtuals` (a virtual field coinciding with a
        // label arg; empirically RefOp(50)/RefOp(174) in synth/tuple_unpacking),
        // the `potential_ops[box] = ShortInputArg(box, renamed)` dict assignment
        // OVERWRITES, so the LAST slot's renamed inputarg is the live one and
        // `produce_arg` returns `short_inputargs[LAST]`; the FIRST slot's renamed
        // box is the dead Label arg. pyre stamps `label_arg_idx = live_slot` (the
        // current call's slot) so the later (overwriting) call records the LAST
        // slot, matching upstream. Before the fix `lookup_label_arg`'s
        // first-occurrence made produce_arg return short_inputargs[FIRST].
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        // The SAME box at both slots: the production caller (optimizer.rs preview
        // loop) iterates label_args+virtuals, so a coinciding box reaches
        // add_short_input_arg once per slot.
        let label_args = [OpRef::ref_op(50), OpRef::ref_op(50)];
        let mut sb = ShortBoxes::with_label_args(&label_args);
        sb.add_short_input_arg(&mut ctx, OpRef::ref_op(50), majit_ir::Type::Ref);
        sb.add_short_input_arg(&mut ctx, OpRef::ref_op(50), majit_ir::Type::Ref);

        // Two FRESH distinct renamed boxes, one per slot (shortpreamble.py:258).
        let si = sb.create_short_inputargs(&label_args);
        assert_eq!(si.len(), 2);
        assert_ne!(si[0], si[1]);

        // The duplicate collapses to ONE produced InputArg entry, keyed at the
        // LAST slot (label_arg_idx == Some(1)), not the first (Some(0)).
        let produced = sb.produced_ops(&mut ctx);
        let inputarg_entries: Vec<_> = produced
            .iter()
            .filter(|(r, p)| *r == OpRef::ref_op(50) && p.kind == PreambleOpKind::InputArg)
            .collect();
        assert_eq!(inputarg_entries.len(), 1);
        // THE REGRESSION LOCK: the overwriting (later) call must record the LAST
        // slot, matching upstream's dict-overwrite. Some(0) before the fix.
        assert_eq!(inputarg_entries[0].1.label_arg_idx, Some(1));

        // produce_arg returns the LAST slot's renamed box.
        let box50 = ctx.materialize_operand_at(OpRef::ref_op(50));
        let produced_arg = sb.produce_arg(&mut ctx, &box50).unwrap();
        assert_eq!(produced_arg.to_opref(), si[1]);
        assert_ne!(produced_arg.to_opref(), si[0]);

        // `lookup_label_arg` mirrors `potential_ops[box]`'s last-wins overwrite,
        // resolving the duplicated box to the LAST/live slot. `rposition`; was
        // `Some(0)` (first) before the fix.
        assert_eq!(sb.lookup_label_arg(OpRef::ref_op(50)), Some(1));
    }

    #[test]
    fn test_rpython_short_preamble_builder_add_op_to_short_builds_label_short_and_jump() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        // The ovf result (pos 10) is an op result; its deps (30, 31) are the
        // label args and become renamed ShortInputargs that the Label carries.
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(30), OpRef::int_op(31)]);
        for arg in [OpRef::int_op(30), OpRef::int_op(31)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }

        let mut ovf = Op::new(
            OpCode::IntAddOvf,
            &[
                crop(&mut __ctx, Type::Int, 30),
                crop(&mut __ctx, Type::Int, 31),
            ],
        );
        ovf.pos().set(OpRef::int_op(10));
        {
            let res = __ctx.materialize_operand_at(ovf.pos().get());
            sb.add_potential_op(&mut __ctx, None, res, ovf, PreambleOpKind::Pure);
        }

        let produced = sb.produced_ops(&mut __ctx);
        let short_inputargs = sb.create_short_inputargs(&[OpRef::int_op(30), OpRef::int_op(31)]);
        let label_arg_oprefs: Vec<OpRef> = short_inputargs.clone();
        // #146: the builder map keys by the entry res Box; re-key the
        // produced_ops list (keyed by `preamble_op.pos`) to res for new() and
        // look up by the res box of the int_op(10) entry.
        let entries: Vec<(majit_ir::operand::Operand, ProducedShortOp)> = produced
            .iter()
            .map(|(_, p)| (p.res.clone(), p.clone()))
            .collect();
        let res10 = produced
            .iter()
            .find(|(r, _)| *r == OpRef::int_op(10))
            .unwrap()
            .1
            .res
            .clone();
        let mut builder = ShortPreambleBuilder::new(&label_arg_oprefs, &entries, &short_inputargs);
        let used = builder.add_op_to_short(&res10).unwrap();
        assert!(builder.add_preamble_op(&res10));
        assert_eq!(used.opcode, OpCode::IntAddOvf);
        assert_eq!(builder.used_boxes(), &[OpRef::int_op(10)]);

        let short = builder.build_short_preamble();
        assert_eq!(short.len(), 4);
        assert_eq!(short[0].opcode, OpCode::Label);
        assert_eq!(short[1].opcode, OpCode::IntAddOvf);
        assert_eq!(short[2].opcode, OpCode::GuardNoOverflow);
        assert_eq!(short[3].opcode, OpCode::Jump);
        // shortpreamble.py `short_preamble_jump` carries `preamble_op`,
        // the replay result, not `used_boxes` (the body box int_op(10)).
        assert_ne!(used.pos().get(), OpRef::int_op(10));
        assert_eq!(
            short[3]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![used.pos().get()]
        );
    }

    #[test]
    fn test_rpython_short_preamble_builder_tracks_extra_same_as() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        // The compound result (pos 20) is an op result; its deps (30, 31) are
        // the label args / renamed ShortInputargs.
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(30), OpRef::int_op(31)]);
        for arg in [OpRef::int_op(30), OpRef::int_op(31)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[crop(&mut __ctx, Type::Int, 30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos().set(OpRef::int_op(20));
        {
            let res = __ctx.materialize_operand_at(heap.pos().get());
            sb.add_potential_op(&mut __ctx, None, res, heap, PreambleOpKind::Heap);
        }

        let mut pure = Op::new(
            OpCode::IntAdd,
            &[
                crop(&mut __ctx, Type::Int, 30),
                crop(&mut __ctx, Type::Int, 31),
            ],
        );
        pure.pos().set(OpRef::int_op(20));
        {
            let res = __ctx.materialize_operand_at(pure.pos().get());
            sb.add_potential_op(&mut __ctx, None, res, pure, PreambleOpKind::Pure);
        }

        let produced = sb.produced_ops(&mut __ctx);
        let (alias_result, alias_res) = produced
            .iter()
            .find(|(result, pop)| *result != OpRef::int_op(20) && pop.invented_name)
            .map(|(result, pop)| (*result, pop.res.clone()))
            .unwrap();

        // #146: re-key the produced_ops list to res for new() + look up the
        // invented-name alias entry by its res box.
        let entries: Vec<(majit_ir::operand::Operand, ProducedShortOp)> = produced
            .iter()
            .map(|(_, p)| (p.res.clone(), p.clone()))
            .collect();
        let mut builder =
            ShortPreambleBuilder::new(&[OpRef::int_op(20)], &entries, &[OpRef::int_op(20)]);
        assert!(builder.add_preamble_op(&alias_res));
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos().get(), alias_result);
        assert_eq!(
            extra[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(20)]
        );
    }

    #[test]
    fn test_short_preamble_builder_fallback_keeps_invented_name_alias_identity() {
        let mut builder = ShortPreambleBuilder::new(&[OpRef::int_op(7)], &[], &[OpRef::int_op(7)]);
        let mut replay_op = Op::new(OpCode::GetfieldGcI, &[rop(Type::Int, 30)]);
        replay_op.pos().set(OpRef::int_op(14));
        let pop = crate::optimizeopt::info::PreambleOp {
            op: rooted_resop_operand(Type::Int, 14),
            invented_name: true,
            preamble_op: OpRc::new(replay_op),
            // Imported invented-name pop carries the original it aliases;
            // the else arm reads it to emit `same_as(source)`.
            same_as_source: Some(rooted_resop_operand(Type::Int, 14)),
        };

        builder.add_preamble_op_from_pop(&pop, rooted_resop_operand(Type::Int, 41));

        assert_eq!(builder.used_boxes(), &[OpRef::int_op(41)]);
        assert_eq!(builder.short_preamble_jump().len(), 1);
        assert_eq!(
            builder.short_preamble_jump()[0].pos().get(),
            OpRef::int_op(14)
        );
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos().get(), OpRef::int_op(41));
        assert_eq!(
            extra[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(14)]
        );
    }

    #[test]
    fn test_extended_short_preamble_builder_fallback_keeps_invented_name_alias_identity() {
        let sb = ShortPreambleBuilder::new(&[OpRef::int_op(7)], &[], &[OpRef::int_op(7)]);
        let mut builder = ExtendedShortPreambleBuilder::new(
            crate::history::TargetToken::new_loop(0).as_jump_target_descr(),
            &sb,
        );
        let mut replay_op = Op::new(OpCode::GetfieldGcI, &[rop(Type::Int, 30)]);
        replay_op.pos().set(OpRef::int_op(14));
        let pop = crate::optimizeopt::info::PreambleOp {
            op: rooted_resop_operand(Type::Int, 14),
            invented_name: true,
            preamble_op: OpRc::new(replay_op),
            // Imported invented-name pop carries the original it aliases;
            // the else arm reads it to emit `same_as(source)`.
            same_as_source: Some(rooted_resop_operand(Type::Int, 14)),
        };

        builder.add_preamble_op_from_pop(&pop, rooted_resop_operand(Type::Int, 41));

        assert_eq!(builder.label_args(), &[OpRef::int_op(41)]);
        assert_eq!(builder.jump_args(), &[OpRef::int_op(14)]);
        let short = builder.build_short_preamble_struct();
        assert_eq!(short.used_boxes, vec![OpRef::int_op(41)]);
        assert_eq!(short.jump_args, vec![OpRef::int_op(14)]);
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos().get(), OpRef::int_op(41));
        assert_eq!(
            extra[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(14)]
        );
    }

    #[test]
    fn extended_builder_keeps_renamed_short_inputargs() {
        let renamed = OpRef::input_arg_typed(51, Type::Int);
        let original = OpRef::input_arg_typed(8, Type::Int);
        let jump_arg = OpRef::int_op(80);
        let sb = ShortPreambleBuilder::new(&[original], &[], &[renamed]);
        let mut builder = ExtendedShortPreambleBuilder::new(
            crate::history::TargetToken::new_loop(0).as_jump_target_descr(),
            &sb,
        );
        let mut add = Op::new(
            OpCode::IntAdd,
            &[
                Operand::bound_from_opref(renamed),
                Operand::from_opref(OpRef::const_int(1)),
            ],
        );
        add.pos().set(OpRef::int_op(13));
        let initial = ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: add,
                arg_mapping: vec![(0, 0)],
                fail_arg_mapping: Vec::new(),
            }],
            inputargs: vec![renamed],
            used_boxes: Vec::new(),
            jump_args: vec![renamed],
            exported_state: None,
            constants: majit_ir::ConstMap::default(),
            inputarg_infos: Vec::new(),
        };
        let mut ctx = crate::optimizeopt::OptContext::new(128);

        // shortpreamble.py ExtendedShortPreambleBuilder.setup stores the op
        // unchanged; the renamed short_inputargs stay the operands.
        assert!(builder.setup(&initial, &[original], &mut ctx));
        assert_eq!(builder.label_args(), &[original]);
        assert_eq!(builder.short_op(0).unwrap().arg(0).to_opref(), renamed);
        let rebuilt = builder.build_short_preamble_struct();
        assert_eq!(rebuilt.inputargs, vec![renamed]);
        assert_eq!(rebuilt.ops[0].op.arg(0).to_opref(), renamed);

        // unroll.py inline_short_preamble seeds only
        // mapping[short_inputargs[i]] = jump_args[i].
        let jump_arg_box = ctx.materialize_operand_at(jump_arg);
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let extra = crate::optimizeopt::unroll::OptUnroll::inline_short_preamble(
            &[jump_arg_box.clone()],
            &[jump_arg_box],
            &rebuilt,
            &mut optimizer,
            &mut ctx,
        );
        assert!(
            ctx.take_invalid_loop().is_none(),
            "leg (1) maps the renamed short inputarg"
        );
        assert_eq!(
            extra.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![jump_arg]
        );
        let replayed = ctx
            .new_operations
            .iter()
            .find(|op| op.opcode == OpCode::IntAdd)
            .expect("replayed int_add");
        assert_eq!(replayed.arg(0).to_opref(), jump_arg);
    }

    /// `GetfieldGcR` of a `CALL_PURE` result that is also a loop-label
    /// `used_box`. `jump_args` is only `args + virtuals` (`unroll.py`
    /// `inline_short_preamble`); `used_boxes` are not a tail of that list.
    /// `setup` must replay the producer so `_map_args` finds the getfield arg.
    #[test]
    fn setup_replays_call_pure_before_getfield_of_used_box() {
        use crate::optimizeopt::optimizer::Optimizer;
        use crate::optimizeopt::unroll::OptUnroll;
        use majit_ir::descr::make_call_descr;
        use majit_ir::{EffectInfo, ExtraEffect, OopSpecIndex, Value};

        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(32, &[Type::Int]);
        let i0 = OpRef::input_arg_int(0);
        let mut call = Op::new(
            OpCode::CallR,
            &[
                Operand::const_from_value(Value::Int(0x1000)),
                ctx.materialize_operand_at(i0),
                Operand::const_from_value(Value::Int(1)),
            ],
        );
        call.setdescr(make_call_descr(
            vec![Type::Int, Type::Int, Type::Int],
            Type::Ref,
            EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None),
        ));
        call.pos().set(OpRef::ref_op(72));
        let mut getfield = Op::with_descr(
            OpCode::GetfieldGcR,
            &[ctx.materialize_operand_at(OpRef::ref_op(72))],
            majit_ir::make_field_descr(0, 8, Type::Ref, majit_ir::ArrayFlag::Pointer),
        );
        getfield.pos().set(OpRef::ref_op(77));

        let mut sb = ShortBoxes::with_label_args(&[i0]);
        sb.add_short_input_arg(&mut ctx, i0, Type::Int);
        {
            let res = ctx.materialize_operand_at(call.pos().get());
            sb.add_potential_op(&mut ctx, None, res, call, PreambleOpKind::Pure);
        }
        {
            let res = ctx.materialize_operand_at(getfield.pos().get());
            sb.add_potential_op(&mut ctx, None, res, getfield, PreambleOpKind::Heap);
        }
        let produced = sb.produced_ops(&mut ctx);
        let call_prod = produced
            .iter()
            .find(|(_, pop)| pop.preamble_op.opcode == OpCode::CallPureR)
            .expect("PureOp.add_op_to_short records CALL_PURE_R")
            .1
            .clone();
        let getfield_prod = produced
            .iter()
            .find(|(_, pop)| pop.preamble_op.opcode == OpCode::GetfieldGcR)
            .expect("heap getfield")
            .1
            .clone();
        let call_pos = call_prod.preamble_op.pos().get();
        assert_ne!(
            call_pos,
            call_prod.res.to_opref(),
            "replay op is not the export-time call box"
        );
        assert_ne!(
            getfield_prod.preamble_op.pos().get(),
            getfield_prod.res.to_opref(),
            "replay getfield is not the export-time box"
        );
        assert_eq!(getfield_prod.preamble_op.arg(0).to_opref(), call_pos);
        let short_inputargs = sb.create_short_inputargs(&[i0]);
        let built = build_short_preamble_from_produced_boxes(&[i0], &short_inputargs, &produced);
        let built_ops: Vec<OpCode> = built.ops.iter().map(|entry| entry.op.opcode).collect();
        let call_at = built_ops
            .iter()
            .position(|op| *op == OpCode::CallPureR)
            .expect("short preamble replays CALL_PURE_R");
        let getfield_at = built_ops
            .iter()
            .position(|op| *op == OpCode::GetfieldGcR)
            .expect("short preamble replays the getfield");
        assert!(
            call_at < getfield_at,
            "use_box appends the producer before GetfieldGcR, got {built_ops:?}"
        );

        let entries: Vec<(Operand, ProducedShortOp)> = produced
            .iter()
            .map(|(_, pop)| (pop.res.clone(), pop.clone()))
            .collect();
        let builder = ShortPreambleBuilder::new(&[i0], &entries, &short_inputargs);
        let mut ext = ExtendedShortPreambleBuilder::new(
            crate::history::TargetToken::new_loop(1).as_jump_target_descr(),
            &builder,
        );
        // Serialized short ops omit the producer. The loop label carries it
        // as a used_box, which is not a short inputarg.
        let stored = ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: (*getfield_prod.preamble_op).clone(),
                arg_mapping: Vec::new(),
                fail_arg_mapping: Vec::new(),
            }],
            inputargs: short_inputargs.clone(),
            used_boxes: vec![call_pos],
            jump_args: vec![getfield_prod.preamble_op.pos().get()],
            exported_state: None,
            constants: majit_ir::ConstMap::default(),
            inputarg_infos: Vec::new(),
        };
        let label_args = vec![i0, call_pos];
        assert!(
            ext.setup(&stored, &label_args, &mut ctx),
            "producer is in produced_short_boxes, so the getfield stays"
        );
        let replayed: Vec<OpCode> = (0..ext.short_ops_len())
            .map(|i| ext.short_op(i).unwrap().opcode)
            .collect();
        assert_eq!(replayed, vec![OpCode::CallPureR, OpCode::GetfieldGcR]);

        ctx.activate_short_preamble_producer(ext);
        let jump_arg = OpRef::int_op(3);
        let jump_arg_box = ctx.materialize_operand_at(jump_arg);
        let mut optimizer = Optimizer::new();
        let extra = OptUnroll::inline_short_preamble(
            &[jump_arg_box.clone()],
            &[jump_arg_box],
            &stored,
            &mut optimizer,
            &mut ctx,
        );
        assert!(
            ctx.take_invalid_loop().is_none(),
            "GetfieldGcR(call result) must map through the replayed CALL_PURE_R"
        );
        assert_eq!(extra.len(), 1);
    }
}
