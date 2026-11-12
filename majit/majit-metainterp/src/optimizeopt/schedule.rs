//! Instruction scheduling and pack management for vectorization.
//!
//! Mirrors RPython's `schedule.py`: pack groups, accumulation tracking,
//! guard analysis, and vector scheduling state.

use indexmap::{IndexMap, IndexSet};
use majit_ir::operand::Operand;
use majit_ir::{Op, OpCode, OpRc, OpRef, Type};

use crate::jitexc::{NotAProfitableLoop, NotAVectorizeableLoop};
use crate::optimizeopt::vector::CostModel;

/// Resolve an `OpRef` to a producer-bound `Operand` against the supplied
/// producer buffers. A hit binds to the canonical producer `OpRc`
/// (`from_bound_op` → `Operand::Op`, no mint); a miss (inputarg / external /
/// constant) falls back to the position-only / const `from_opref` box.
fn bound_boxref_in(
    r: OpRef,
    buffers: &[&[OpRc]],
    renamer: &mut super::renamer::Renamer,
) -> Operand {
    if r.is_constant() || r.is_none() {
        return Operand::from_opref(r);
    }
    if let Some(rc) = buffers
        .iter()
        .flat_map(|b| b.iter())
        .find(|p| p.pos().get() == r)
    {
        return Operand::from_bound_op(rc);
    }
    // No producer in the supplied buffers (e.g. a loop inputarg): bind to a
    // renamer-rooted producer box carrying the same `pos`, never a
    // position-only `Operand::Box`.
    renamer.bound_box(r)
}

/// schedule.py+: A pack is a set of n isomorphic operations that can
/// execute as a single SIMD instruction.
#[derive(Clone, Debug)]
pub struct Pack {
    /// The scalar opcode of the group members.
    pub scalar_opcode: OpCode,
    /// The vector opcode to replace them with.
    pub vector_opcode: OpCode,
    /// Indices into the DepGraph nodes.
    pub members: Vec<usize>,
    /// schedule.py: whether this pack tracks an accumulation (reduction).
    pub is_accumulating: bool,
    /// schedule.py: accumulation argument position (-1 = none).
    pub position: i32,
    /// schedule.py: AccumPack.operator — '+' for INT_ADD, 'g' for guard, None for normal.
    pub operator: Option<char>,
}

/// vector.py: Accumulation pack — tracks reduction operations
/// (e.g., `sum += array[i]`) that can be vectorized with horizontal
/// reduction instructions.
#[derive(Clone, Debug)]
/// schedule.py: AccumPack — accumulation (reduction) pack.
pub struct AccumPack {
    /// The scalar opcode of the accumulation (e.g., IntAdd, FloatAdd).
    pub scalar_opcode: OpCode,
    /// schedule.py: operator ('+' for INT_ADD, etc.)
    pub operator: char,
    /// schedule.py: position — which arg of the op is the accumulator.
    pub position: usize,
    /// The initial accumulator value OpRef.
    pub init_value: OpRef,
    /// Indices of the accumulation operations in the loop body.
    pub members: Vec<usize>,
    /// Whether this is a float accumulation.
    pub is_float: bool,
}

/// Accumulation info stored in the accumulation map.
/// schedule.py: `state.accumulation[arg] = pack`
///
/// schedule.py keys `accumulation` by the failarg box and `getleftmostseed`
/// returns a box object; pyre shapes `seed` as a flat `OpRef` and keys
/// `accumulation` by `OpRef`. This is intentional, not a box-identity gap: the
/// seed is always the loop-carried label arg fed through `accumulate_prepare`
/// (schedule.py) — a `FlowValue::Variable` register, never a `Const` —
/// so identity-by-position and identity-by-box coincide (the Const ptr-
/// instability that forces #115 to keep const-namespace tables OpRef-keyed
/// never applies here). The consumer rebinds the producer at emit time
/// (vector.rs `pre_emit_guard_accum`). Box-shaping `seed` is a cosmetic
/// #169/#175 follow-up, not a correctness fix.
#[derive(Clone, Debug)]
pub struct AccumEntry {
    /// schedule.py: getleftmostseed() — first member's arg at `position`.
    pub seed: OpRef,
    /// schedule.py: operator character ('+' etc.)
    pub operator: char,
    /// The original scalar opcode (preserves int/float distinction).
    pub accum_opcode: OpCode,
}

/// vector.py: Guard analysis result — determines which guards can be
/// moved to the loop header (hoisted) to expose more vectorization.
#[derive(Clone, Debug)]
pub struct GuardAnalysis {
    /// Guards that can be hoisted to the loop header.
    pub hoistable: Vec<usize>,
    /// Guards that must remain in the loop body.
    pub body_guards: Vec<usize>,
}

impl GuardAnalysis {
    /// Analyze guards in a loop body for hoistability.
    /// vector.py: analyze_guards()
    /// A guard is hoistable if its arguments are loop-invariant
    /// (not produced by any op in the loop body).
    pub fn analyze(ops: &[Op]) -> Self {
        let mut body_results: indexmap::IndexSet<OpRef> = indexmap::IndexSet::new();
        for op in ops {
            if !op.pos().get().is_none() {
                body_results.insert(op.pos().get());
            }
        }

        let mut hoistable = Vec::new();
        let mut body_guards = Vec::new();

        for (i, op) in ops.iter().enumerate() {
            if !op.opcode.is_guard() {
                continue;
            }
            let all_invariant = op
                .getarglist()
                .iter()
                .all(|arg| !body_results.contains(&arg.to_opref()));
            if all_invariant {
                hoistable.push(i);
            } else {
                body_guards.push(i);
            }
        }

        GuardAnalysis {
            hoistable,
            body_guards,
        }
    }
}

/// schedule.py: State for vector-aware instruction scheduling.
/// Tracks which scalar ops have been mapped to vector ops, handles
/// pack/unpack/expand operations, and manages the output op list.
pub struct VecScheduleState {
    /// Map from scalar OpRef → (index_in_vector, vector OpRef).
    pub box_to_vbox: indexmap::IndexMap<OpRef, (usize, OpRef)>,
    /// Output operations (vector + remaining scalar).
    ///
    /// `Vec<OpRc>` (not `Vec<Op>`): each emitted op is the canonical producer
    /// box for its value, so `create_vec_op` binds a new vector op's args to
    /// the producer `Rc` already in this list (`Operand::from_bound_op`)
    /// rather than minting a position-only `Operand::Box`.
    pub oplist: Vec<OpRc>,
    /// Renamer for SSA fixup during vectorization.
    pub renamer: super::renamer::Renamer,
    /// Cost model for profitability analysis.
    pub costmodel: CostModel,
    /// schedule.py: expanded_map — tracks expanded scalars.
    pub expanded_map: indexmap::IndexMap<OpRef, Vec<(OpRef, i32)>>,
    /// schedule.py: inputargs of the loop label.
    pub inputargs: indexmap::IndexMap<OpRef, ()>,
    /// schedule.py: invariant_vector_vars — vector ops created by expand()
    /// for loop-invariant scalars (constants and inputargs). Populated in
    /// expand() (schedule.py), called from prepare_arguments().
    pub invariant_vector_vars: indexmap::IndexSet<OpRef>,
    /// schedule.py: invariant_oplist — ops to emit before the loop.
    /// `Vec<OpRc>` for the same producer-identity reason as `oplist`.
    pub invariant_oplist: Vec<OpRc>,
    /// schedule.py: accumulation info.
    pub accumulation: indexmap::IndexMap<OpRef, AccumEntry>,
    /// Next OpRef counter for newly created vector ops.
    next_pos: u32,
    /// `schedule.py forwarded_vecinfo(op)` scratch, keyed by full `OpRef`
    /// (InputArg and op namespaces never collide).
    ///
    /// PyPy carries this scheduling scratch in `op._forwarded`
    /// (`vector.py setup_vectorization`) and re-propagates it across its
    /// SINGLE clone path, `copy_resop` (`vector.py`), which COPIES the
    /// already-resolved `VectorizationInfo` — INT_SIGNEXT's bytesize is the
    /// dynamic value of `arg1`, resolved ONCE at setup time
    /// (`resoperation.py`) and thereafter only copied, never recomputed.
    /// pyre cannot store it on `op._forwarded` yet because it has no `copy_resop`
    /// analog: `Op::clone` resets `forwarded` to `None` (resoperation.rs) while
    /// preserving `pos`, and the scheduler reads vecinfo off CLONED ops — the
    /// dependency graph clones every op into its nodes (dependency.rs) and the
    /// unroll/schedule paths clone `loop_.operations`. A clone-reset `forwarded`
    /// would drop the stamp with nothing to re-attach it. The `pos`-keyed cache
    /// is clone-stable (`OpRef`/`pos` survives a clone) and stands in until a
    /// `copy_resop`-equivalent exists.
    ///
    /// `Op.vecinfo` is the SEPARATE permanent carrier: the
    /// `resoperation.py VecOperationNew` datatype/bytesize/signed/count
    /// that survives `copy_and_change`, cleared for non-vector ops by
    /// `vector.py teardown_vectorization`.
    vecinfo_cache: indexmap::IndexMap<OpRef, majit_ir::VectorizationInfo>,
}

impl VecScheduleState {
    pub fn new(start_pos: u32) -> Self {
        VecScheduleState {
            box_to_vbox: indexmap::IndexMap::new(),
            oplist: Vec::new(),
            renamer: super::renamer::Renamer::new(),
            costmodel: CostModel::new(),
            expanded_map: indexmap::IndexMap::new(),
            inputargs: indexmap::IndexMap::new(),
            invariant_vector_vars: indexmap::IndexSet::new(),
            invariant_oplist: Vec::new(),
            accumulation: indexmap::IndexMap::new(),
            next_pos: start_pos,
            vecinfo_cache: indexmap::IndexMap::new(),
        }
    }

    /// vector.py `op.set_forwarded(VectorizationInfo(op))` for one op:
    /// the per-op body that `VectorLoop::setup_vectorization` iterates.
    /// PyPy stores the vecinfo on `op._forwarded`; pyre keeps it in this
    /// scheduler's pos-keyed store.
    ///
    /// `constant_of` reads `arg1.value` for the
    /// `resoperation.py VectorizationInfo.__init__` INT_SIGNEXT branch —
    /// PyPy reads the value off the `ConstInt` instance because
    /// `_args[1]` IS the const.
    pub(crate) fn set_op_forwarded_vecinfo(
        &mut self,
        op: &Op,
        constant_of: &dyn Fn(&Operand) -> Option<i64>,
    ) {
        let info = if op.opcode == OpCode::IntSignext {
            self.int_signext_vecinfo(op, constant_of)
        } else {
            self.vectorization_info_for_op(op)
        };
        self.set_forwarded_vecinfo(op.pos().get(), info);
    }

    /// vector.py teardown_vectorization `op.set_forwarded(None)` for one op: drops the
    /// stored vecinfo, the per-op body of `VectorLoop::teardown_vectorization`.
    pub(crate) fn clear_op_forwarded_vecinfo(&mut self, opref: OpRef) {
        self.vecinfo_cache.swap_remove(&opref);
    }

    /// `resoperation.py VectorizationInfo.__init__` INT_SIGNEXT
    /// branch.  PyPy reads `op.getarg(1).value` off the `ConstInt` object
    /// directly, after `assert isinstance(arg1, history.ConstInt)`. An
    /// unresolvable `arg1` is a malformed INT_SIGNEXT, so this fails fast
    /// (mirroring the `assert`) rather than silently degrading to a
    /// generic vecinfo.
    fn int_signext_vecinfo(
        &self,
        op: &Op,
        constant_of: &dyn Fn(&Operand) -> Option<i64>,
    ) -> majit_ir::VectorizationInfo {
        // resoperation.py `assert isinstance(arg1, history.ConstInt)`.
        let bytesize = constant_of(&op.arg(1)).expect(
            "INT_SIGNEXT arg1 must resolve to a ConstInt (resoperation.py VectorizationInfo.__init__)",
        );
        assert!(
            (i8::MIN as i64..=i8::MAX as i64).contains(&bytesize),
            "INT_SIGNEXT byte size {bytesize} out of VectorizationInfo range"
        );
        let mut info = majit_ir::VectorizationInfo::new();
        info.setinfo('i', bytesize as i8, true);
        info
    }

    fn get_forwarded_vecinfo(&self, opref: OpRef) -> Option<majit_ir::VectorizationInfo> {
        if opref.is_none() || opref.is_constant() {
            return None;
        }
        self.vecinfo_cache.get(&opref).cloned()
    }

    fn set_forwarded_vecinfo(&mut self, opref: OpRef, info: majit_ir::VectorizationInfo) {
        if opref.is_none() || opref.is_constant() {
            return;
        }
        self.vecinfo_cache.insert(opref, info);
    }

    /// schedule.py `forwarded_vecinfo(op)`.
    pub fn forwarded_vecinfo(&mut self, op: &Op) -> majit_ir::VectorizationInfo {
        let opref = op.pos().get();
        if let Some(info) = self.get_forwarded_vecinfo(opref) {
            return info;
        }
        let info = self.vectorization_info_for_op(op);
        if !opref.is_constant() {
            self.set_forwarded_vecinfo(opref, info.clone());
        }
        info
    }

    pub fn forwarded_vecinfo_for_ref(
        &mut self,
        opref: OpRef,
        ops: &[OpRc],
    ) -> majit_ir::VectorizationInfo {
        if let Some(info) = self.get_forwarded_vecinfo(opref) {
            return info;
        }
        if let Some(op) = ops
            .iter()
            .chain(self.oplist.iter())
            .chain(self.invariant_oplist.iter())
            .find(|op| op.pos().get() == opref)
            .cloned()
        {
            return self.forwarded_vecinfo(&op);
        }
        let tp = opref.ty().unwrap_or(Type::Int);
        let info = majit_ir::VectorizationInfo::from_type(tp);
        if !opref.is_constant() {
            self.set_forwarded_vecinfo(opref, info.clone());
        }
        info
    }

    /// resoperation.py `VectorizationInfo(op)` for ResOps.
    /// Const/InputArg are handled by the cache miss path in
    /// `forwarded_vecinfo_for_ref` (`VectorizationInfo.from_type`).
    ///
    /// Mirrors PyPy `resoperation.py VectorizationInfo.__init__`.
    /// The `INT_SIGNEXT` branch (`:181-186`) is handled in
    /// `int_signext_vecinfo` at setup time because it needs the
    /// caller-supplied const-pool resolver; once the forwarded vecinfo
    /// cache is stamped, every later lookup hits the cache and never
    /// re-enters this method for INT_SIGNEXT.  A bare miss-path call on
    /// INT_SIGNEXT (e.g. a synthesised op that bypassed setup) falls
    /// through to the typecast branch, which returns `None` for it, and
    /// then to the source-operand pass-through; `setup_vectorization`
    /// stamps every loop INT_SIGNEXT up front, so this residual path is
    /// not reached for real loop bodies.
    fn vectorization_info_for_op(&self, op: &Op) -> majit_ir::VectorizationInfo {
        // resoperation.py:170-180 primitive_array_access branch.
        if op.opcode.is_primitive_array_access_opcode()
            && let Some(descr) = op.getdescr()
            && let Some(arr) = majit_ir::descr::descr_arc_as_array_descr(descr)
            && arr.is_array_of_primitives()
        {
            let datatype = match op.result_type() {
                Type::Int => 'i',
                Type::Float => 'f',
                Type::Ref => 'r',
                Type::Void => 'v',
            };
            let bytesize = arr.item_size() as i8;
            let signed = arr.is_item_signed();
            let mut info = majit_ir::VectorizationInfo::new();
            info.setinfo(datatype, bytesize, signed);
            return info;
        }

        // resoperation.py is_typecast branch (INT_SIGNEXT static
        // gating returns None per `cast_to_bytesize_static`; the dynamic
        // INT_SIGNEXT bytesize is stamped in `int_signext_vecinfo` at
        // setup time and read back through the forwarded vecinfo cache).
        if op.opcode.is_typecast()
            && let Some(bytesize) = op.opcode.cast_to_bytesize_static()
        {
            let (_ft, tt) = op.opcode.cast_types();
            let mut info = majit_ir::VectorizationInfo::new();
            info.setinfo(tt, bytesize as i8, true);
            return info;
        }

        // resoperation.py:192-209 else branch:
        //   type = op.type
        //   signed = type == 'i'
        //   bytesize = -1
        //   if op.numargs() > 0:
        //       i = 0
        //       arg = op.getarg(i)
        //       while arg.is_constant() and i+1 < op.numargs():
        //           i += 1
        //           arg = op.getarg(i)
        //       if not arg.is_constant():
        //           vecinfo = arg.get_forwarded()
        //           if vecinfo is not None and isinstance(vecinfo, VectorizationInfo):
        //               if vecinfo.datatype != '\x00' and vecinfo.bytesize != -1:
        //                   type     = vecinfo.datatype
        //                   signed   = vecinfo.signed
        //                   bytesize = vecinfo.bytesize
        //   if rop.returns_bool_result(op.opnum):
        //       type = 'i'
        //   self.setinfo(type, bytesize, signed)
        let mut tp = op.result_type();
        let mut datatype = match tp {
            Type::Int => 'i',
            Type::Float => 'f',
            Type::Ref => 'r',
            Type::Void => 'v',
        };
        let mut signed = datatype == 'i';
        let mut bytesize: i8 = -1;

        let n = op.num_args();
        if n > 0 {
            let mut i = 0usize;
            let mut arg = op.arg(i);
            while arg.is_constant() && i + 1 < n {
                i += 1;
                arg = op.arg(i);
            }
            if !arg.is_constant()
                && let Some(vinfo) = self.get_forwarded_vecinfo(arg.to_opref())
                && vinfo.datatype != '\x00'
                && vinfo.bytesize != -1
            {
                datatype = vinfo.datatype;
                tp = match datatype {
                    'i' => Type::Int,
                    'f' => Type::Float,
                    'r' => Type::Ref,
                    _ => tp,
                };
                signed = vinfo.signed;
                bytesize = vinfo.bytesize;
            }
        }

        if op.opcode.returns_bool() {
            datatype = 'i';
            tp = Type::Int;
        }
        let _ = tp;

        let mut info = majit_ir::VectorizationInfo::new();
        info.setinfo(datatype, bytesize, signed);
        info
    }

    /// Allocate a fresh typed OpRef for a newly created vector op. The
    /// caller supplies the op's result type (`opcode.result_type()`) so
    /// the returned OpRef carries the proper `Int/Float/Ref/Void`
    /// variant tag.
    pub fn alloc_op_pos(&mut self, tp: Type) -> OpRef {
        let pos = OpRef::op_typed(self.next_pos, tp);
        self.next_pos += 1;
        pos
    }

    /// Resolve an arg `OpRef` to a producer-bound `Operand`.
    ///
    /// vector.py's renamer maps box→box and carries the box objects through
    /// the scheduler. pyre's args are integer positions, so to recover the
    /// producer box we look the position up among the ops already emitted
    /// into `oplist` / `invariant_oplist` (SSA guarantees a producer is
    /// emitted before its consumers). A hit binds to that exact producer
    /// `Rc` (`Operand::from_bound_op` → `Operand::Op`, no mint); a constant
    /// sheds to `Operand::Const`. A miss for a ResOp/InputArg position
    /// (an inputarg, or a scalar not yet emitted as a vector op) is bound
    /// to a renamer-rooted producer box carrying the same `pos`
    /// (`Renamer::bound_box`), so no position-only `Operand::Box` is minted.
    pub fn bound_arg_boxref(&mut self, r: OpRef) -> Operand {
        if r.is_constant() || r.is_none() {
            return Operand::from_opref(r);
        }
        if let Some(rc) = self
            .oplist
            .iter()
            .chain(self.invariant_oplist.iter())
            .find(|op| op.pos().get() == r)
        {
            return Operand::from_bound_op(rc);
        }
        self.renamer.bound_box(r)
    }

    /// Re-bind an op's args to their producer boxes after the renamer has
    /// rewritten them to new positions.
    ///
    /// The renamer (`renamer.rs`) maps old→new positions and, lacking a
    /// producer view, sets each rewritten arg via `from_opref` (a
    /// position-only box). This pass resolves every non-const arg against the
    /// supplied producer buffers and rebinds the hits to the canonical
    /// producer `OpRc` (`from_bound_op`), so no position-only `Operand::Box`
    /// survives for an arg whose producer is in a buffer. Args with no buffer
    /// producer (inputargs / constants) are left as the renamer set them.
    /// Uses `Op::setarg` interior mutability (`&self`).
    fn rebind_op_args_in(op: &Op, buffers: &[&[OpRc]]) {
        for i in 0..op.num_args() {
            let r = op.arg(i).to_opref();
            if r.is_constant() || r.is_none() {
                continue;
            }
            if let Some(rc) = buffers
                .iter()
                .flat_map(|b| b.iter())
                .find(|p| p.pos().get() == r)
            {
                op.setarg(i, Operand::from_bound_op(rc));
            }
        }
        // Also rebind any guard fail_args carried position-only.
        if let Some(mut rebound) = op.getfailargs() {
            let mut changed = false;
            for slot in rebound.iter_mut() {
                let r = slot.to_opref();
                if r.is_constant() || r.is_none() {
                    continue;
                }
                if let Some(rc) = buffers
                    .iter()
                    .flat_map(|b| b.iter())
                    .find(|p| p.pos().get() == r)
                {
                    *slot = Operand::from_bound_op(rc);
                    changed = true;
                }
            }
            if changed {
                op.setfailargs(rebound);
            }
        }
    }

    /// Re-bind against the scheduler's own emitted buffers
    /// (`oplist` / `invariant_oplist`).
    pub fn rebind_op_args(&self, op: &Op) {
        Self::rebind_op_args_in(op, &[&self.oplist, &self.invariant_oplist]);
    }

    /// resoperation.py (VecOperationNew): Create a vector op with
    /// proper VectorizationInfo. All vector helper functions should use this
    /// instead of raw Op::new + register_vec_type.
    pub fn create_vec_op(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        datatype: char,
        bytesize: i32,
        signed: bool,
        count: usize,
    ) -> Op {
        let ba: Vec<Operand> = args.iter().map(|a| self.bound_arg_boxref(*a)).collect();
        let op = Op::new(opcode, &ba);
        op.pos().set(self.alloc_op_pos(opcode.result_type()));
        let mut vinfo = majit_ir::VectorizationInfo::new();
        vinfo.setinfo(datatype, bytesize as i8, signed);
        vinfo.count = count as i16;
        // resoperation.py VecOperationNew.__init__ stores the
        // datatype/bytesize/signed/count on the op object itself; copy_and_change
        // (resoperation.py) propagates them. Cache the same payload
        // on `Op.vecinfo` so the vector shape survives schedule-state teardown.
        op.set_vecinfo(vinfo.clone());
        self.set_forwarded_vecinfo(op.pos().get(), vinfo);
        op
    }

    /// Check if an OpRef refers to a float-type vector op.  The OpRef
    /// variant itself carries the result-type tag (`opcode.result_type()`
    /// at `alloc_op_pos`), mirroring PyPy's `opclasses[opnum].type == 'f'`
    /// gating in `resoperation.py` — no side-table needed.
    pub fn is_float_vector(&self, opref: OpRef) -> bool {
        opref.ty() == Some(Type::Float)
    }

    /// schedule.py: setvector_of_box — record that scalar_op
    /// is at index `idx` in the vector `vecop`.
    pub fn setvector_of_box(&mut self, scalar_op: OpRef, idx: usize, vecop: OpRef) {
        self.box_to_vbox.insert(scalar_op, (idx, vecop));
    }

    /// schedule.py: getvector_of_box — look up which vector
    /// op contains the scalar op.
    pub fn getvector_of_box(&self, scalar_op: OpRef) -> Option<(usize, OpRef)> {
        self.box_to_vbox.get(&scalar_op).copied()
    }

    /// schedule.py: append to output.
    ///
    /// Wraps the finished op into the canonical producer `OpRc` as it enters
    /// `oplist`, so a later `create_vec_op` whose args reference this op's
    /// position binds directly to this `Rc` (`Operand::from_bound_op`) rather
    /// than minting a position-only `Operand::Box`.
    pub fn append_to_oplist(&mut self, op: Op) {
        self.oplist.push(OpRc::new(op));
    }

    /// schedule.py: remember_args_in_vector — after creating a new
    /// vector from assemble/position, update box_to_vbox for each scalar arg
    /// so future lookups find the correct vector box.
    pub fn remember_args_in_vector(
        &mut self,
        pack: &Pack,
        index: usize,
        vecbox: OpRef,
        ops: &[OpRc],
    ) {
        for (i, &member_idx) in pack.members.iter().enumerate() {
            let op = &ops[member_idx];
            if index >= op.num_args() {
                break;
            }
            let arg = op.arg(index).to_opref();
            // schedule.py:
            //   vecinfo = forwarded_vecinfo(arg)
            //   if i >= vecinfo.count: break
            //   self.setvector_of_box(arg, i, box)
            let vecinfo = self.forwarded_vecinfo_for_ref(arg, ops);
            if (i as i16) >= vecinfo.count {
                break;
            }
            self.setvector_of_box(arg, i, vecbox);
        }
    }

    /// schedule.py: VecScheduleState.post_schedule.
    ///
    /// RPython reads `self.graph.loop`; majit's VecScheduleState holds no
    /// graph/loop field, so the loop is a parameter. `seen` is
    /// threaded in because majit's `ensure_args_unpacked` takes it as an
    /// explicit param (vector.rs); RPython keeps the equivalent state on self.
    pub fn post_schedule(
        &mut self,
        loop_: &mut crate::optimizeopt::vector::VectorLoop,
        seen: &mut indexmap::IndexSet<OpRef>,
    ) {
        // schedule.py → base SchedulerState.post_schedule (schedule.py),
        // inlined here. schedule.py resolve_delayed is omitted: majit has
        // no `delayed` list (ILP scheduling is done up front via
        // schedule_operations), so the base reduces to rename-jump + move-oplist.
        self.renamer.rename(&mut loop_.jump); // schedule.py

        // schedule.py
        crate::optimizeopt::vector::ensure_args_unpacked(self, &mut loop_.jump, seen);
        // Rebind the renamed / unpacked jump args to their producers (still in
        // the scheduler's `oplist` / `invariant_oplist` before the moves below).
        self.rebind_op_args(&loop_.jump);

        // schedule.py: loop.operations = self.oplist. In PyPy line 116 runs in
        // the base post_schedule (before 765) but ALIASES self.oplist, so any
        // VecUnpack ops that ensure_args_unpacked (765) appends to self.oplist are
        // visible in loop.operations. Rust's mem::take MOVES (no aliasing), so the
        // take must run AFTER ensure_args_unpacked to capture those unpack ops —
        // otherwise the finalized jump would reference OpRefs with no defining op.
        loop_.operations = std::mem::take(&mut self.oplist);

        // schedule.py
        loop_.prefix = std::mem::take(&mut self.invariant_oplist);

        // schedule.py: if len(invariant_vector_vars) + len(invariant_oplist) > 0.
        // We read `loop_.prefix.len()` because invariant_oplist was just moved into
        // it; RPython aliases the same list object so the length is unchanged.
        if !self.invariant_vector_vars.is_empty() || !loop_.prefix.is_empty() {
            // schedule.py: prefix_label.
            //   args = loop.label.getarglist_copy() + self.invariant_vector_vars
            let mut args = loop_.label.getarglist();
            // invariant_vector_vars is a IndexSet (insertion-ordered re-export of
            // vecmap_rs::VecSet), so iterating reproduces RPython's list-append
            // order. RPython's list may hold dups but expand() only appends fresh
            // boxes, so IndexSet's de-dup is a no-op here. Each invariant var is a
            // vector op now living in `loop_.prefix`; bind to that producer box.
            // Collected first so the per-var bind can take `&mut self.renamer`
            // (the bound-box synthesis for an inputarg-position miss).
            let inv_vars: Vec<OpRef> = self.invariant_vector_vars.iter().copied().collect();
            for r in &inv_vars {
                args.push(bound_boxref_in(
                    *r,
                    &[&loop_.prefix, &loop_.operations],
                    &mut self.renamer,
                ));
            }
            // schedule.py: opnum = loop.label.getopnum();
            //   op = loop.label.copy_and_change(opnum, args).
            // The opcode ("opnum") is unchanged → loop_.label.opcode; descr None
            // means "keep self.descr"; copy_and_change preserves the result `pos`.
            let args_ops: Vec<Operand> = args.to_vec();
            let mut prefix_label =
                loop_
                    .label
                    .copy_and_change(loop_.label.opcode, Some(args_ops.as_slice()), None);
            self.renamer.rename(&mut prefix_label); // schedule.py
            // The producers now live in `loop_.operations` / `loop_.prefix`
            // (oplist/invariant_oplist were moved above); rebind against them.
            Self::rebind_op_args_in(&prefix_label, &[&loop_.operations, &loop_.prefix]);
            loop_.prefix_label = Some(prefix_label); // schedule.py

            // schedule.py: jump.
            let mut args = loop_.jump.getarglist();
            for r in &inv_vars {
                args.push(bound_boxref_in(
                    *r,
                    &[&loop_.prefix, &loop_.operations],
                    &mut self.renamer,
                ));
            }
            let args_ops2: Vec<Operand> = args.to_vec();
            let mut new_jump =
                loop_
                    .jump
                    .copy_and_change(loop_.jump.opcode, Some(args_ops2.as_slice()), None);
            self.renamer.rename(&mut new_jump);
            Self::rebind_op_args_in(&new_jump, &[&loop_.operations, &loop_.prefix]);
            loop_.jump = new_jump;
        }
    }

    // ── schedule.py: expand / find_expanded ──

    /// schedule.py: record that `args` were expanded into `vecop`.
    pub fn record_expansion(&mut self, args: &[OpRef], vecop: OpRef) {
        let mut index: i32 = 0;
        if args.len() == 1 {
            index = -1; // schedule.py: broadcast marker
        }
        for arg in args {
            self.expanded_map
                .entry(*arg)
                .or_default()
                .push((vecop, index));
            index += 1;
        }
    }

    /// schedule.py: find_expanded — look up if `args` have already
    /// been expanded into a vector op.
    pub fn find_expanded(&self, args: &[OpRef]) -> Option<OpRef> {
        if args.len() == 1 {
            // schedule.py: single arg → look for broadcast (index == -1)
            let candidates = self.expanded_map.get(&args[0])?;
            for &(vecop, index) in candidates {
                if index == -1 {
                    return Some(vecop);
                }
            }
            return None;
        }
        // schedule.py: multi-arg → intersect candidates at correct positions.
        // For each arg position i, collect vecops that expanded arg at index i.
        // A vecop is valid only if it appears at every position — intersect.
        let mut possible: indexmap::IndexMap<OpRef, bool> = indexmap::IndexMap::new();
        for (i, arg) in args.iter().enumerate() {
            let expansions = self.expanded_map.get(arg)?;
            // schedule.py: filter by index match AND possible.get(vecop, True)
            let candidates: Vec<OpRef> = expansions
                .iter()
                .filter(|&&(vecop, idx)| {
                    idx == i as i32 && possible.get(&vecop).copied().unwrap_or(true)
                })
                .map(|&(vecop, _)| vecop)
                .collect();
            // schedule.py: invalidate vecops NOT in this position's candidates
            for (k, v) in possible.iter_mut() {
                if !candidates.contains(k) {
                    *v = false;
                }
            }
            // schedule.py: mark surviving candidates as valid
            for vecop in candidates {
                if !possible.contains_key(&vecop) {
                    possible.insert(vecop, true);
                }
            }
            if possible.is_empty() {
                return None;
            }
        }
        possible.iter().find(|(_, v)| **v).map(|(k, _)| *k)
    }
}

/// Combined failure mode for `optimize_vector` / `run_optimization`,
/// mirroring vector.py's two `except` arms. Callers convert this
/// back to a "no-vectorize-this-time" decision and replay the original
/// loop ops; the distinction is kept so future passes (e.g. logging or
/// GuardStrengthenOpt) can react differently.
#[derive(Debug)]
pub enum VectorizeError {
    NotVectorizeable,
    NotProfitable,
}

impl From<NotAVectorizeableLoop> for VectorizeError {
    fn from(_: NotAVectorizeableLoop) -> Self {
        VectorizeError::NotVectorizeable
    }
}

impl From<NotAProfitableLoop> for VectorizeError {
    fn from(_: NotAProfitableLoop) -> Self {
        VectorizeError::NotProfitable
    }
}

/// schedule.py: check_if_pack_supported — validate pack constraints.
pub fn check_if_pack_supported(
    state: &mut VecScheduleState,
    pack: &Pack,
    ops: &[OpRc],
) -> Result<(), NotAProfitableLoop> {
    let first_op = &ops[pack.members[0]];
    // schedule.py: INT_MUL with bytesize 8 or 1 is not profitable
    if first_op.opcode == OpCode::IntMul {
        let vi = state.forwarded_vecinfo(first_op);
        let insize = vi.getbytesize();
        if insize == 8 || insize == 1 {
            return Err(NotAProfitableLoop);
        }
    }
    Ok(())
}

/// schedule.py: unpack_from_vector — extract a scalar from a vector box.
/// Creates a VecUnpack op with the correct type (I or F) based on the
/// vector box's datatype. Mirrors OpHelpers.create_vec_unpack(var.type, ...).
pub fn unpack_from_vector(
    state: &mut VecScheduleState,
    vec_ref: OpRef,
    index: usize,
    count: usize,
) -> OpRef {
    assert!(count > 0);
    let index_const = OpRef::const_int(index as i64);
    let count_const = OpRef::const_int(count as i64);
    let is_float = state.is_float_vector(vec_ref);
    let unpack_opcode = if is_float {
        OpCode::VecUnpackF
    } else {
        OpCode::VecUnpackI
    };
    // schedule.py: forwarded_vecinfo(arg).bytesize/signed
    let (datatype, bytesize, signed) = get_vec_info(state, vec_ref, &[]);
    let unpack_op = state.create_vec_op(
        unpack_opcode,
        &[vec_ref, index_const, count_const],
        datatype,
        bytesize,
        signed,
        count,
    );
    let result = unpack_op.pos().get();
    // schedule.py: costmodel.record_vector_unpack
    state.costmodel.record_vector_unpack(is_float, index, count);
    state.append_to_oplist(unpack_op);
    result
}

/// schedule.py: prepare_fail_arguments — process guard failargs
/// for vectorized guard ops, unpacking vector boxes to scalar.
pub fn prepare_fail_arguments(
    state: &mut VecScheduleState,
    pack: &Pack,
    ops: &[OpRc],
    vecop: &mut Op,
) {
    let first_op = &ops[pack.members[0]];
    if !first_op.opcode.is_guard() {
        return;
    }
    if let Some(fail_args) = first_op.guard_fail_args() {
        let mut new_fail_args: majit_ir::resoperation::OpArgVec =
            fail_args.iter().cloned().collect();
        for slot in new_fail_args.iter_mut() {
            let arg = slot.to_opref();
            // schedule.py: look up if arg is in a vector box
            let (_pos, newarg) = state.getvector_of_box(arg).unwrap_or((0, arg));
            if newarg != arg {
                // schedule.py: vector box → unpack at position 0
                let unpacked = unpack_from_vector(state, newarg, 0, 1);
                *slot = state.bound_arg_boxref(unpacked);
            }
            // else newarg == arg: keep the original bound Operand
            // (schedule.py newarg = arg; args[i] = newarg).
        }
        vecop.setfailargs(new_fail_args);
    }
}

/// schedule.py: prepare_arguments — transform scalar args to vector args.
///
/// RPython gates this on cpu.vector_ext.get_operation_restriction(left).
/// majit uses Cranelift which has no per-op restriction object, so we apply
/// the same logic for all args: reuse, assemble, position, crop, or expand.
pub fn prepare_arguments(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut [OpRef],
    ops: &[OpRc],
) {
    for i in 0..args.len() {
        let arg = args[i];
        // schedule.py: check if arg is in a vector box
        if let Some((pos, vecop)) = state.getvector_of_box(arg) {
            // schedule.py: case 1a — reuse existing vector
            args[i] = vecop;
            // schedule.py: case 1c — scattered values
            assemble_scattered_values(state, pack, args, i, ops);
            // schedule.py: case 1d — wrong position in vector
            position_values(state, pack, args, i, pos, ops);
            // schedule.py: case 1b — size mismatch (crop)
            crop_vector(state, pack, args, i, ops);
        } else {
            // schedule.py: case 2 — not in a vector, expand
            expand(state, pack, args, arg, i, ops);
        }
    }
}

/// schedule.py: assemble_scattered_values — when pack members'
/// args at `index` reside in different vector boxes, gather them into one.
pub fn assemble_scattered_values(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut [OpRef],
    index: usize,
    ops: &[OpRc],
) {
    // schedule.py: collect each member's arg at this index
    let mut args_at_index: Vec<OpRef> = pack
        .members
        .iter()
        .map(|&m| {
            let op = &ops[m];
            if index < op.num_args() {
                op.arg(index).to_opref()
            } else {
                args[index]
            }
        })
        .collect();
    // schedule.py: first one is already assigned
    args_at_index[0] = args[index];

    // schedule.py: check which vector boxes these args reside in
    let mut vectors: Vec<(usize, OpRef)> = Vec::new();
    for &a in &args_at_index {
        if let Some((pos, vecop)) = state.getvector_of_box(a)
            && (vectors.is_empty() || vectors.last().map(|v| v.1) != Some(vecop))
        {
            vectors.push((pos, vecop));
        }
    }

    // schedule.py: if scattered across >1 vector, gather
    if vectors.len() > 1 {
        args[index] = gather(state, &vectors, pack.members.len(), ops);
        // schedule.py: remember_args_in_vector
        state.remember_args_in_vector(pack, index, args[index], ops);
    }
}

/// schedule.py: gather — combine multiple vector fragments into one.
/// Uses each fragment's actual lane count (vecinfo.count / newvecinfo.count)
/// to determine insertion position and guard against overfill.
pub fn gather(
    state: &mut VecScheduleState,
    vectors: &[(usize, OpRef)],
    count: usize,
    ops: &[OpRc],
) -> OpRef {
    let (_, mut arg) = vectors[0];
    let mut i = 1;
    while i < vectors.len() {
        let (newarg_pos, newarg) = vectors[i];
        // schedule.py: get actual lane counts from vecinfo
        let arg_count = get_vec_count(state, arg, ops);
        let newarg_count = get_vec_count(state, newarg, ops);
        // schedule.py: guard: combined count must fit in target
        if arg_count + newarg_count <= count {
            // schedule.py: pack newarg into arg at arg's current count
            arg = pack_into_vector(state, arg, arg_count, newarg, newarg_pos, newarg_count, ops);
        }
        i += 1;
    }
    arg
}

/// Get the lane count of a vector OpRef. Falls back to 1.
fn get_vec_count(state: &mut VecScheduleState, opref: OpRef, ops: &[OpRc]) -> usize {
    state.forwarded_vecinfo_for_ref(opref, ops).count.max(1) as usize
}

/// Get (datatype, bytesize, signed) from a vector OpRef's vecinfo.
/// Falls back to is_float_vector registry + default 8-byte.
fn get_vec_info(state: &mut VecScheduleState, opref: OpRef, ops: &[OpRc]) -> (char, i32, bool) {
    let vi = state.forwarded_vecinfo_for_ref(opref, ops);
    if vi.datatype != '\0' {
        return (vi.datatype, vi.bytesize as i32, vi.signed);
    }
    let is_float = state.is_float_vector(opref);
    if is_float {
        ('f', 8, false)
    } else {
        ('i', 8, true)
    }
}

/// schedule.py: pack_into_vector — insert `src` at position `tidx`
/// in `tgt`, producing a wider vector.
///
/// `tgt = [1,2,3,4,_,_,_,_]`, `src = [5,6,_,_]`
/// `result = [1,2,3,4,5,6,_,_]` (tidx=4, scount=2)
pub fn pack_into_vector(
    state: &mut VecScheduleState,
    tgt: OpRef,
    tidx: usize,
    src: OpRef,
    sidx: usize,
    scount: usize,
    ops: &[OpRc],
) -> OpRef {
    // schedule.py: assert sidx == 0
    debug_assert!(sidx == 0, "pack_into_vector: sidx must be 0, got {}", sidx);
    let is_float = state.is_float_vector(tgt);
    let pack_opcode = if is_float {
        OpCode::VecPackF
    } else {
        OpCode::VecPackI
    };
    let tidx_const = OpRef::const_int(tidx as i64);
    let scount_const = OpRef::const_int(scount as i64);
    // schedule.py: forwarded_vecinfo(tgt).bytesize/signed, newcount
    let tgt_count = get_vec_count(state, tgt, ops);
    let newcount = tgt_count + scount;
    let (datatype, bytesize, signed) = get_vec_info(state, tgt, ops);
    let vecop = state.create_vec_op(
        pack_opcode,
        &[tgt, src, tidx_const, scount_const],
        datatype,
        bytesize,
        signed,
        newcount,
    );
    let result = vecop.pos().get();
    state.append_to_oplist(vecop);
    // schedule.py: record cost
    state.costmodel.record_vector_pack(is_float, 0, scount);
    result
}

/// schedule.py: position_values — if an arg is at position != 0
/// in its vector box but needs to be at position 0, unpack it.
pub fn position_values(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut [OpRef],
    index: usize,
    position: usize,
    ops: &[OpRc],
) {
    // schedule.py: position != 0 → unpack to reposition
    if position != 0 {
        let arg = args[index];
        // schedule.py: count = restrict.max_input_count(vecinfo.count)
        // Without oprestrict, default to 1 (extract single element)
        let count = 1;
        args[index] = unpack_from_vector(state, arg, position, count);
        // schedule.py: remember_args_in_vector
        state.remember_args_in_vector(pack, index, args[index], ops);
    }
}

/// schedule.py: crop_vector — if the vector's element size doesn't
/// match what the operation requires, insert VEC_INT_SIGNEXT.
pub fn crop_vector(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut [OpRef],
    index: usize,
    ops: &[OpRc],
) {
    let arg = args[index];
    let first_op = &ops[pack.members[0]];
    // schedule.py: check if bytesize needs conversion
    // Determine the vector's current element size and the op's expected size
    let arg_bytesize = get_op_bytesize_for_ref(state, arg, ops);
    let op_bytesize = state.forwarded_vecinfo(&ops[pack.members[0]]).getbytesize() as i32;
    if arg_bytesize > 0 && op_bytesize > 0 && arg_bytesize != op_bytesize {
        // schedule.py: integer type → VEC_INT_SIGNEXT
        if first_op.opcode.result_type() != majit_ir::Type::Float {
            let newsize_const = OpRef::const_int(op_bytesize as i64);
            let vec_count = get_vec_count(state, arg, ops);
            // schedule.py: VecOperationNew with proper vecinfo
            let signext_op = state.create_vec_op(
                OpCode::VecIntSignext,
                &[arg, newsize_const],
                'i',
                op_bytesize,
                true, // signed
                vec_count,
            );
            let result = signext_op.pos().get();
            state.append_to_oplist(signext_op);
            // schedule.py: record cost
            state
                .costmodel
                .record_cast_int(arg_bytesize as usize, op_bytesize as usize, vec_count);
            args[index] = result;
        }
    }
}

/// Helper: get bytesize for an OpRef that may be a vector op created during scheduling.
fn get_op_bytesize_for_ref(state: &mut VecScheduleState, opref: OpRef, ops: &[OpRc]) -> i32 {
    state.forwarded_vecinfo_for_ref(opref, ops).getbytesize() as i32
}

/// schedule.py: expand — broadcast or gather a scalar into a vector box.
///
/// Two cases:
///   1. All pack members use the same arg at position `index` → VecExpand (broadcast)
///   2. Different args per member → Vec + VecPack (gather)
///
/// Loop-invariant expansions (constants and inputargs) go to invariant_oplist
/// and are tracked in invariant_vector_vars.
pub fn expand(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut [OpRef],
    arg: OpRef,
    index: usize,
    ops: &[OpRc],
) {
    // schedule.py: choose target list (invariant vs inline)
    let is_invariant = arg.is_constant() || state.inputargs.contains_key(&arg);

    // schedule.py: check if all pack members have the same arg at `index`
    let all_same = pack.members.iter().all(|&m| {
        let op = &ops[m];
        index < op.num_args() && op.arg(index).to_opref() == arg
    });

    // datatype is `arg.type` per PyPy `OpHelpers.create_vec_expand`
    // (resoperation.py) — opcode dispatch + the resulting
    // vecinfo.datatype both come from the arg being expanded.
    let datatype = match arg.ty().unwrap_or(Type::Void) {
        Type::Int => 'i',
        Type::Float => 'f',
        Type::Ref => 'r',
        Type::Void => 'v',
    };
    let is_float = datatype == 'f';
    let numops = pack.members.len();

    if all_same {
        // schedule.py: VecExpand (broadcast)
        if let Some(existing) = state.find_expanded(&[arg]) {
            args[index] = existing;
            return;
        }
        // schedule.py: bytesize/signed come from the left-most
        // pack op's vecinfo, NOT from `arg` — the pack's element width is
        // the authoritative shape for the broadcast destination.
        //   left = pack.leftmost()
        //   vecinfo = forwarded_vecinfo(left)
        //   vecop = OpHelpers.create_vec_expand(arg, vecinfo.bytesize,
        //                                       vecinfo.signed, pack.numops())
        let left_op = &ops[pack.members[0]];
        let left_info = state.forwarded_vecinfo(left_op);
        let bytesize = left_info.bytesize as i32;
        let signed = left_info.signed;
        let expand_opcode = if is_float {
            OpCode::VecExpandF
        } else {
            OpCode::VecExpandI
        };
        let vecop = state.create_vec_op(expand_opcode, &[arg], datatype, bytesize, signed, numops);
        let vecop_pos = vecop.pos().get();
        if is_invariant {
            state.invariant_oplist.push(OpRc::new(vecop));
            state.invariant_vector_vars.insert(vecop_pos);
        } else {
            state.append_to_oplist(vecop);
        }
        state.record_expansion(&[arg], vecop_pos);
        args[index] = vecop_pos;
        return;
    }

    // schedule.py: arg_vecinfo = forwarded_vecinfo(arg)
    //   vecop = OpHelpers.create_vec(arg.type, arg_vecinfo.bytesize,
    //                                arg_vecinfo.signed, pack.opnum())
    // Only the heterogeneous (VecPack/gather) branch uses arg's vecinfo —
    // each pack member contributes its own scalar to the vector, so the
    // arg's element width is the shape source.
    let arg_info = state.forwarded_vecinfo_for_ref(arg, ops);
    let bytesize = arg_info.bytesize as i32;
    let signed = arg_info.signed;

    // schedule.py: VecPack (gather) — heterogeneous args
    let expandargs: Vec<OpRef> = pack
        .members
        .iter()
        .map(|&m| {
            let op = &ops[m];
            if index < op.num_args() {
                op.arg(index).to_opref()
            } else {
                arg
            }
        })
        .collect();

    if let Some(existing) = state.find_expanded(&expandargs) {
        args[index] = existing;
        return;
    }

    // schedule.py: create_vec(datatype, bytesize, signed, count)
    let vec_create_opcode = if is_float { OpCode::VecF } else { OpCode::VecI };
    let vec_create =
        state.create_vec_op(vec_create_opcode, &[], datatype, bytesize, signed, numops);
    let mut current_vec = vec_create.pos().get();
    if is_invariant {
        state.invariant_oplist.push(OpRc::new(vec_create));
    } else {
        state.append_to_oplist(vec_create);
    }

    // schedule.py: pack each member's arg into the vector
    let pack_opcode = if is_float {
        OpCode::VecPackF
    } else {
        OpCode::VecPackI
    };
    for (i, &member_arg) in expandargs.iter().enumerate() {
        let i_const = OpRef::const_int(i as i64);
        let one_const = OpRef::const_int(1);
        // schedule.py: create_vec_pack(type, args, bytesize, signed, count+1)
        let pack_op = state.create_vec_op(
            pack_opcode,
            &[current_vec, member_arg, i_const, one_const],
            datatype,
            bytesize,
            signed,
            i + 2, // schedule.py: vecinfo.count+1 (grows by 1 each iteration)
        );
        current_vec = pack_op.pos().get();
        state.costmodel.record_vector_pack(is_float, 0, 1);
        if is_invariant {
            state.invariant_oplist.push(OpRc::new(pack_op));
        } else {
            state.append_to_oplist(pack_op);
        }
    }

    state.record_expansion(&expandargs, current_vec);
    if is_invariant {
        state.invariant_vector_vars.insert(current_vec);
    }
    args[index] = current_vec;
}

/// schedule.py: Turn a pack of scalar ops into a single vector op.
pub fn turn_into_vector(state: &mut VecScheduleState, pack: &Pack, ops: &[OpRc]) {
    if pack.members.is_empty() {
        return;
    }
    // schedule.py: check_if_pack_supported
    if check_if_pack_supported(state, pack, ops).is_err() {
        return;
    }
    let count = pack.members.len();
    let first_op = &ops[pack.members[0]];

    // schedule.py: costmodel.record_pack_savings
    state.costmodel.record_pack_savings(pack, count);

    let Some(vec_opcode) = first_op.opcode.to_vector() else {
        return; // not vectorizable
    };

    // schedule.py: build args list + prepare_arguments
    let mut args: Vec<OpRef> = first_op.getarglist().iter().map(|a| a.to_opref()).collect();
    prepare_arguments(state, pack, &mut args, ops);

    // schedule.py: VecOperation(left.vector, args, left, pack.numops())
    // resoperation.py: copy datatype/bytesize/signed from baseop's vecinfo
    let vi = state.forwarded_vecinfo(first_op);
    let (mut datatype, mut bytesize, signed) = (vi.datatype, vi.bytesize, vi.signed);
    // resoperation.py VecOperation typecast override.
    //   if baseop.is_typecast():
    //       ft, tt = baseop.cast_types()
    //       datatype = tt
    //       bytesize = baseop.cast_to_bytesize()
    // INT_SIGNEXT is excluded by the static-bytesize gate (see
    // TODO in `vectorization_info_for_op`): the dynamic
    // arg1.value path needs const-pool threading.
    if first_op.opcode.is_typecast()
        && let Some(bs) = first_op.opcode.cast_to_bytesize_static()
    {
        let (_ft, tt) = first_op.opcode.cast_types();
        datatype = tt;
        bytesize = bs as i8;
    }
    let mut vecop =
        state.create_vec_op(vec_opcode, &args, datatype, bytesize as i32, signed, count);
    if let Some(d) = first_op.getdescr() {
        vecop.setdescr(d);
    }

    let vecop_pos = vecop.pos().get();
    // schedule.py: map scalar ops to vector positions
    for (i, &member_idx) in pack.members.iter().enumerate() {
        let op = &ops[member_idx];
        if op.opcode.result_type() == majit_ir::Type::Void {
            continue; // schedule.py: skip void ops
        }
        let scalar_pos = op.pos().get();
        if !scalar_pos.is_none() {
            state.setvector_of_box(scalar_pos, i, vecop_pos);
            // schedule.py: only rename for accumulating packs
            if pack.is_accumulating && !op.opcode.is_guard() {
                state.renamer.start_renaming(scalar_pos, vecop_pos);
            }
        }
    }

    // schedule.py: handle guard failargs
    if first_op.opcode.is_guard() {
        prepare_fail_arguments(state, pack, ops, &mut vecop);
    }

    state.append_to_oplist(vecop);
    assert!(count >= 1); // schedule.py
}
