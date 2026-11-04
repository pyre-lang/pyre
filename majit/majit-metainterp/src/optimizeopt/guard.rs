//! Guard strengthening for the vector optimizer.
//!
//! Ports `guard.py`'s `Guard` + `GuardStrengthenOpt`: the integer-comparison
//! guard implication analysis (`transitive_imply` / `emit_varops`) that removes
//! redundant array-bound-check guards during vectorization. Reached only from
//! the vectorizer (`vector.rs` `run_optimization`), matching `guard.py`'s note
//! that this strengthening is used only by the vec optimizer.
//!
//! Guard folding on ordinary traces (GUARD_TRUE/FALSE/VALUE/CLASS/NONNULL/
//! ISNULL) lives in `OptRewrite` (`rewrite.rs`) and `OptIntBounds`
//! (`intbounds.rs`), folding via box-attached `getptrinfo`/`getintbound` +
//! `make_constant` — `rewrite.py` `optimize_GUARD_*`.
use indexmap::IndexMap;
use majit_ir::{Op, OpCode, OpRc, OpRef};

use crate::optimizeopt::OptContext;
use crate::optimizeopt::dependency::IndexVar;

/// guard.py: Guard — wraps a guard op with its comparison op for
/// implication analysis (vector optimizer).
///
/// Stores the full `Op` for both the guard and its comparison, matching
/// RPython's `_attrs_ = ('index', 'op', 'cmp_op', 'rhs', 'lhs')`.
/// This preserves descr/fail_args for inhert_attributes and emit_operations.
#[derive(Clone, Debug)]
pub struct Guard {
    /// guard.py:20 — position in the operations list.
    pub index: usize,
    /// guard.py:20 — the guard op (GuardTrue/GuardFalse), with descr + fail_args.
    pub op: Op,
    /// guard.py:20 — the comparison op (IntLt/IntLe/IntGt/IntGe).
    pub cmp_op: Op,
    /// guard.py:27-29 — left-hand side IndexVar.
    pub lhs: IndexVar,
    /// guard.py:31-33 — right-hand side IndexVar.
    pub rhs: IndexVar,
}

impl Guard {
    /// guard.py: Guard.__init__(index, op, cmp_op, index_vars)
    pub fn new(
        index: usize,
        guard_op: Op,
        cmp_op: Op,
        index_vars: &indexmap::IndexMap<OpRef, IndexVar>,
    ) -> Self {
        let lhs_arg = cmp_op.arg(0).to_opref();
        let lhs = index_vars
            .get(&lhs_arg)
            .cloned()
            .unwrap_or_else(|| IndexVar::new(lhs_arg));
        let rhs_arg = if cmp_op.num_args() > 1 {
            cmp_op.arg(1).to_opref()
        } else {
            OpRef::NONE
        };
        let rhs = index_vars
            .get(&rhs_arg)
            .cloned()
            .unwrap_or_else(|| IndexVar::new(rhs_arg));
        Guard {
            index,
            op: guard_op,
            cmp_op,
            lhs,
            rhs,
        }
    }

    /// guard.py: Guard.of(boolarg, operations, index, index_vars)
    pub fn of(
        index: usize,
        guard_op: &Op,
        cmp_op: &Op,
        index_vars: &indexmap::IndexMap<OpRef, IndexVar>,
    ) -> Option<Self> {
        if !guard_op.opcode.is_guard() {
            return None;
        }
        match cmp_op.opcode {
            OpCode::IntLt | OpCode::IntLe | OpCode::IntGt | OpCode::IntGe => {}
            _ => return None,
        }
        Some(Self::new(
            index,
            guard_op.clone(),
            cmp_op.clone(),
            index_vars,
        ))
    }

    /// guard.py: setindex / setoperation / setcmp
    pub fn setindex(&mut self, index: usize) {
        self.index = index;
    }

    pub fn setoperation(&mut self, op: Op) {
        self.op = op;
    }

    pub fn setcmp(&mut self, cmp: Op) {
        self.cmp_op = cmp;
    }

    /// guard.py: get_compare_opnum
    pub fn get_compare_opnum(&self) -> OpCode {
        if self.op.opcode == OpCode::GuardTrue {
            self.cmp_op.opcode
        } else {
            // guard_false inversion (cmp_op.boolinverse)
            match self.cmp_op.opcode {
                OpCode::IntLt => OpCode::IntGe,
                OpCode::IntLe => OpCode::IntGt,
                OpCode::IntGt => OpCode::IntLe,
                OpCode::IntGe => OpCode::IntLt,
                OpCode::IntEq => OpCode::IntNe,
                OpCode::IntNe => OpCode::IntEq,
                other => other,
            }
        }
    }

    /// guard.py: implies(other)
    pub fn implies(&self, other: &Guard, _opt: Option<&OptContext>) -> bool {
        if self.op.opcode != other.op.opcode {
            return false;
        }
        if self.getleftkey() != other.getleftkey() {
            return false;
        }
        let (lhs_valid, lc) = self.lhs.compare(&other.lhs);
        if !lhs_valid {
            return false;
        }
        let (rhs_valid, rc) = self.rhs.compare(&other.rhs);
        if !rhs_valid {
            return false;
        }
        let opnum = self.get_compare_opnum();
        match opnum {
            OpCode::IntLe | OpCode::IntLt => lc >= 0 && rc <= 0,
            OpCode::IntGe | OpCode::IntGt => lc <= 0 && rc >= 0,
            _ => false,
        }
    }

    /// guard.py: getleftkey / getrightkey
    pub fn getleftkey(&self) -> OpRef {
        self.lhs.getvariable()
    }

    pub fn getrightkey(&self) -> OpRef {
        self.rhs.getvariable()
    }

    /// guard.py: emit_varops(opt, var, old_arg)
    fn emit_varops(
        var: &IndexVar,
        old_arg: OpRef,
        new_ops: &mut Vec<Op>,
        renamer: &mut indexmap::IndexMap<OpRef, OpRef>,
        next_const_pos: &mut u32,
        const_values: &mut indexmap::IndexMap<OpRef, i64>,
    ) -> OpRef {
        if var.is_identity() {
            return var.var;
        }
        // history.py ConstInt.value carried inline on the Box. The
        // const_values side-table is preserved for legacy callers that
        // still walk `OpRef → raw u32 → i64`; inline-Const variants make
        // it redundant for new consumers.
        let ncp = next_const_pos;
        let cv = const_values;
        let ops = var.get_operations(|value| {
            let cref = OpRef::const_int(value);
            *ncp += 1;
            cv.insert(cref, value);
            cref
        });
        let mut last = var.var;
        for op in ops {
            last = op.pos().get();
            new_ops.push(op);
        }
        // guard.py:131: opt.renamer.start_renaming(old_arg, box)
        if !last.is_constant() {
            renamer.insert(old_arg, last);
        }
        last
    }

    /// guard.py: transitive_imply(other, opt, loop)
    ///
    /// Emit a transitive guard that eliminates a loop guard.
    /// `label_args` = `loop.label.getarglist_copy()`.
    /// Emits compare + guard into `new_ops`. Returns the guard op.
    pub fn transitive_imply(
        &self,
        other: &Guard,
        label_args: &[OpRef],
        new_ops: &mut Vec<Op>,
        renamer: &mut indexmap::IndexMap<OpRef, OpRef>,
        next_const_pos: &mut u32,
        const_values: &mut indexmap::IndexMap<OpRef, i64>,
    ) -> Option<Op> {
        if self.op.opcode != other.op.opcode {
            return None;
        }
        if self.getleftkey() != other.getleftkey() {
            return None;
        }
        if !self.rhs.is_identity() {
            return None;
        }
        // guard.py:83
        let opnum = Self::transitive_cmpop(self.cmp_op.opcode);
        // guard.py: emit_varops
        let box_rhs = Self::emit_varops(
            &self.rhs,
            self.cmp_op.arg(1).to_opref(),
            new_ops,
            renamer,
            next_const_pos,
            const_values,
        );
        let other_rhs = Self::emit_varops(
            &other.rhs,
            other.cmp_op.arg(1).to_opref(),
            new_ops,
            renamer,
            next_const_pos,
            const_values,
        );
        // #175: GuardStrengthenOpt carries `Op` VALUES, not `OpRc` —
        // `operations_as_ops` (vector.rs) deep-clones each producer and
        // `emit_varops` returns a flat `OpRef`, so no real producer `Rc` is
        // reachable here to carry the way guard.py threads `box_rhs`.
        // `bound_from_opref` mints a synthetic producer whose `to_opref()` is
        // byte-identical (`majit-ir/src/operand.rs`); every reader in this pass keys by
        // `OpRef` (renamer/strongest_guards/rename_op), so the synthetic is
        // positionally equivalent. Carrying the live producer is deferred to
        // #175 (bind dormant vectorizer guard args to producers).
        // guard.py: compare = ResOperation(opnum, [box_rhs, other_rhs])
        let compare = Op::new(
            opnum,
            &[
                majit_ir::operand::Operand::bound_from_opref(box_rhs),
                majit_ir::operand::Operand::bound_from_opref(other_rhs),
            ],
        );
        new_ops.push(compare.clone());
        // guard.py:89-91:
        //   descr = CompileLoopVersionDescr()
        //   descr.copy_all_attributes_from(self.op.getdescr())
        //   descr.rd_vector_info = None
        //
        // Always create a NEW CompileLoopVersionDescr (loop_version()=true),
        // then copy resume attributes from the source guard's descr.
        // compile.py copy_all_attributes_from copies:
        //   rd_consts, rd_pendingfields, rd_virtuals, rd_numb
        // In pyre these live on the FailDescr (compile.py:855 `_attrs_`);
        // make_compile_loop_version_descr_from reference-shares each
        // Arc<[T]> slot from the donor onto the fresh descr.
        let fresh_descr = crate::compile::make_compile_loop_version_descr_from(&self.op);
        let mut guard_op = Op::new(
            self.op.opcode,
            &[majit_ir::operand::Operand::bound_from_opref(
                compare.pos().get(),
            )],
        );
        guard_op.setdescr(fresh_descr);
        // guard.py:94: guard.setfailargs(loop.label.getarglist_copy())
        // #175: `label_args` was reduced to `&[OpRef]` at the vectorizer
        // boundary (the label op's `getarglist()` Operands were `.to_opref()`'d),
        // so the live producers guard.py copies are not reachable here; re-bind
        // positionally. Carrying the label operands end-to-end is deferred to
        // #175 (the failargs channel re-narrows to OpRef downstream via
        // rename_failargs, so a partial Operand carry would be cosmetic).
        guard_op.setfailargs(
            label_args
                .iter()
                .map(|a| majit_ir::operand::Operand::bound_from_opref(*a))
                .collect(),
        );
        // copy_all_attributes_from parity: compile.py copies
        // rd_consts / rd_pendingfields / rd_virtuals / rd_numb.  In
        // pyre these live on the FailDescr (compile.py:855 `_attrs_`)
        // and `make_compile_loop_version_descr_from` (called above)
        // reference-shares the donor's payload onto the fresh descr,
        // so `guard_op.descr.fail_descr().rd_*()` returns the donor's
        // values automatically.
        match self.op.get_fail_arg_types() {
            Some(types) => guard_op.set_fail_arg_types(types.to_vec()),
            None => guard_op.clear_fail_arg_types(),
        }
        guard_op.set_rd_resume_position(self.op.rd_resume_position());
        // guard.py: opt.emit_operation(guard)
        new_ops.push(guard_op.clone());
        Some(guard_op)
    }

    /// guard.py: transitive_cmpop(opnum)
    pub fn transitive_cmpop(opnum: OpCode) -> OpCode {
        match opnum {
            OpCode::IntLt => OpCode::IntLe,
            OpCode::IntGt => OpCode::IntGe,
            other => other,
        }
    }

    /// guard.py: inhert_attributes(other)
    ///
    /// Copy index and resume attributes from `other` onto self's existing
    /// descr, preserving `myop.descr` identity (fail_index / status /
    /// subtype tag).  Mirrors RPython's:
    ///   descr = myop.getdescr()
    ///   descr.copy_all_attributes_from(other.op.getdescr())
    ///   `myop.setfailargs(otherop.guard_fail_args()[:])`
    /// where `descr` is the strengthened guard's *own* ResumeGuardDescr.
    pub fn inhert_attributes(&mut self, other: &Guard) {
        // guard.py:118
        self.index = other.index;
        // guard.py:120-121: descr.copy_all_attributes_from(other.op.getdescr())
        // compile.py ResumeGuardDescr.copy_all_attributes_from:
        //     other = other.get_resumestorage()
        //     assert isinstance(other, ResumeGuardDescr)
        //     self.rd_consts = other.rd_consts
        //     self.rd_pendingfields = other.rd_pendingfields
        //     self.rd_virtuals = other.rd_virtuals
        //     self.rd_numb = other.rd_numb
        // i.e. the strengthened guard keeps its own descr identity; only
        // the resume payload is copied from the donor (chasing through
        // ResumeGuardCopiedDescr.prev via get_resumestorage()).
        let my_descr = self
            .op
            .getdescr()
            .expect("guard.py:120 myop.getdescr() must exist");
        let donor_descr = other
            .op
            .getdescr()
            .expect("guard.py:121 otherop.getdescr() must exist");
        // compile.py:861-872 / 840-842: in-place copy preserving
        // myop.descr identity (`fail_index` / status / subtype tag).
        crate::compile::copy_all_attributes_from(&my_descr, &donor_descr);
        self.op
            .set_rd_resume_position(other.op.rd_resume_position());
        // guard.py inhert_attributes: myop.setfailargs(otherop.getfailargs()[:])
        match other.op.guard_fail_args() {
            Some(fa) => self.op.setfailargs(fa.iter().cloned().collect()),
            None => self.op.clearfailargs(),
        }
        match other.op.get_fail_arg_types() {
            Some(types) => self.op.set_fail_arg_types(types.to_vec()),
            None => self.op.clear_fail_arg_types(),
        }
    }

    /// guard.py: emit_operations(opt)
    ///
    /// Re-emit the guard: materialize lhs/rhs via emit_varops,
    /// create fresh cmp + guard, emit them.
    pub fn emit_operations(
        &mut self,
        new_ops: &mut Vec<Op>,
        renamer: &mut indexmap::IndexMap<OpRef, OpRef>,
        next_const_pos: &mut u32,
        const_values: &mut indexmap::IndexMap<OpRef, i64>,
    ) {
        // guard.py: lhs/rhs via emit_varops
        let lhs = Self::emit_varops(
            &self.lhs,
            self.cmp_op.arg(0).to_opref(),
            new_ops,
            renamer,
            next_const_pos,
            const_values,
        );
        let rhs = Self::emit_varops(
            &self.rhs,
            self.cmp_op.arg(1).to_opref(),
            new_ops,
            renamer,
            next_const_pos,
            const_values,
        );
        // guard.py:138-140: cmp_op = ResOperation(opnum, [lhs, rhs])
        let cmp_op = Op::new(
            self.cmp_op.opcode,
            &[
                majit_ir::operand::Operand::bound_from_opref(lhs),
                majit_ir::operand::Operand::bound_from_opref(rhs),
            ],
        );
        new_ops.push(cmp_op.clone());
        // guard.py:142-144: guard = ResOperation(opnum, [cmp_op], descr)
        let mut guard = Op::new(
            self.op.opcode,
            &[majit_ir::operand::Operand::bound_from_opref(
                cmp_op.pos().get(),
            )],
        );
        if let Some(d) = self.op.getdescr() {
            guard.setdescr(d);
        }
        // guard.py emit_operations: guard.setfailargs(self.op.getfailargs()[:])
        match self.op.guard_fail_args() {
            Some(fa) => guard.setfailargs(fa.iter().cloned().collect()),
            None => guard.clearfailargs(),
        }
        match self.op.get_fail_arg_types() {
            Some(types) => guard.set_fail_arg_types(types.to_vec()),
            None => guard.clear_fail_arg_types(),
        }
        guard.set_rd_resume_position(self.op.rd_resume_position());
        // compile.py _attrs_ on descr; Arc-clone above shares them.
        new_ops.push(guard.clone());
        // guard.py:145-147
        self.setindex(new_ops.len() - 1);
        self.setoperation(guard);
        self.setcmp(cmp_op);
    }

    /// guard.py: set_to_none(info, loop)
    pub fn set_to_none(&self, ops: &mut [Option<Op>]) {
        ops[self.index] = None;
        if self.index > 0 {
            // guard.py:154: if operations[self.index-1] is self.cmp_op
            if let Some(ref prev) = ops[self.index - 1]
                && prev.pos().get() == self.cmp_op.pos().get()
            {
                ops[self.index - 1] = None;
            }
        }
    }
}

/// guard.py: GuardStrengthenOpt (vector optimizer guard pass).
///
/// RPython name: `GuardStrengthenOpt`. Collects guard information from
/// a complete loop, determines implication/strengthening, re-emits with
/// proper descr/fail_args, and optionally eliminates array bound checks.
pub struct GuardStrengthenOpt {
    /// guard.py:168
    pub index_vars: indexmap::IndexMap<OpRef, IndexVar>,
    /// guard.py:169
    _newoperations: Vec<Op>,
    /// guard.py:170
    pub strength_reduced: usize,
    /// guard.py:171
    pub strongest_guards: indexmap::IndexMap<OpRef, Vec<Guard>>,
    /// guard.py:172
    guards: indexmap::IndexMap<usize, Option<Guard>>,
    /// renamer.py: Renamer — maps old OpRef → new OpRef for renamed vars.
    renamer: indexmap::IndexMap<OpRef, OpRef>,
    /// Zero-based counter for constant-namespace OpRef allocation.
    next_const_pos: u32,
    /// Materialized constant values: OpRef → i64.
    /// RPython uses ConstInt boxes inline; majit stores const values here.
    pub const_values: indexmap::IndexMap<OpRef, i64>,
}

impl GuardStrengthenOpt {
    /// guard.py __init__
    pub fn new(index_vars: indexmap::IndexMap<OpRef, IndexVar>) -> Self {
        GuardStrengthenOpt {
            index_vars,
            _newoperations: Vec::new(),
            strength_reduced: 0,
            strongest_guards: indexmap::IndexMap::new(),
            guards: indexmap::IndexMap::new(),
            renamer: indexmap::IndexMap::new(),
            next_const_pos: 0,
            const_values: indexmap::IndexMap::new(),
        }
    }

    /// guard.py: collect_guard_information(loop)
    pub fn collect_guard_information(&mut self, ops: &[Op]) {
        for (i, op) in ops.iter().enumerate() {
            if !op.opcode.is_guard() {
                continue;
            }
            if op.opcode != OpCode::GuardTrue && op.opcode != OpCode::GuardFalse {
                continue;
            }
            // guard.py: Guard.of(op.getarg(0), operations, i, self.index_vars)
            let bool_arg = op.arg(0).to_opref();
            let cmp_op = ops.iter().rfind(|o| o.pos().get() == bool_arg);
            if let Some(cmp) = cmp_op
                && let Some(guard) = Guard::of(i, op, cmp, &self.index_vars)
            {
                let lk = guard.getleftkey();
                let rk = guard.getrightkey();
                self.record_guard(lk, guard.clone());
                self.record_guard(rk, guard);
            }
        }
    }

    /// guard.py: record_guard(key, guard)
    fn record_guard(&mut self, key: OpRef, guard: Guard) {
        if key.is_none() {
            return;
        }
        // guard.py — in-place mutation of `self.strongest_guards[key]`.
        // Split-borrow on disjoint fields lets us hold `&mut others`
        // (a slice into strongest_guards) and `&mut self.guards`
        // simultaneously, matching PyPy's two-map update order.
        let strongest_guards = &mut self.strongest_guards;
        let guards = &mut self.guards;
        let others = strongest_guards.entry(key).or_default();
        if !others.is_empty() {
            let mut replaced = false;
            for other in others.iter_mut() {
                if guard.implies(other, None) {
                    // guard.py:204-210: strengthened
                    let old = other.clone();
                    guards.insert(guard.index, None); // mark new as 'do not emit'
                    let mut new_guard = guard.clone();
                    new_guard.inhert_attributes(&old);
                    guards.insert(old.index, Some(new_guard.clone()));
                    *other = new_guard;
                    replaced = true;
                } else if other.implies(&guard, None) {
                    // guard.py:211-215: implied
                    guards.insert(guard.index, None);
                    replaced = true;
                }
            }
            if !replaced {
                others.push(guard);
            }
        } else {
            others.push(guard);
        }
    }

    /// guard.py: eliminate_guards(loop)
    pub fn eliminate_guards(&mut self, ops: &[Op]) -> Vec<Op> {
        // guard.py:222: self.renamer = Renamer()
        self.renamer = indexmap::IndexMap::new();
        self._newoperations = Vec::with_capacity(ops.len());

        // Take guards out of self to satisfy borrow checker.
        let mut guards = std::mem::take(&mut self.guards);
        let index_vars = self.index_vars.clone();

        for (i, op) in ops.iter().enumerate() {
            if op.opcode.is_guard() {
                if let Some(replacement) = guards.get_mut(&i) {
                    self.strength_reduced += 1;
                    match replacement {
                        None => {
                            // guard.py:233: implied → skip
                            continue;
                        }
                        Some(guard) => {
                            // guard.py: guard.emit_operations(self)
                            guard.emit_operations(
                                &mut self._newoperations,
                                &mut self.renamer,
                                &mut self.next_const_pos,
                                &mut self.const_values,
                            );
                            continue;
                        }
                    }
                } else {
                    // guard.py: self.emit_operation(op)
                    let mut renamed = op.clone();
                    self.rename_op(&mut renamed);
                    self._newoperations.push(renamed);
                    continue;
                }
            }
            // guard.py: non-void index_var → emit_operations + rename
            if op.opcode.result_type() != majit_ir::Type::Void
                && let Some(index_var) = index_vars.get(&op.pos().get())
                && !index_var.is_identity()
            {
                let ncp = &mut self.next_const_pos;
                let cv = &mut self.const_values;
                let result = index_var.emit_operations(&mut self._newoperations, |value| {
                    let cref = OpRef::const_int(value);
                    *ncp += 1;
                    cv.insert(cref, value);
                    cref
                });
                self.renamer.insert(op.pos().get(), result);
                continue;
            }
            // guard.py: self.emit_operation(op)
            let mut renamed = op.clone();
            self.rename_op(&mut renamed);
            self._newoperations.push(renamed);
        }
        self.guards = guards;
        self._newoperations.clone()
    }

    /// guard.py: propagate_all_forward(info, loop, user_code)
    ///
    /// `version_info`: optional LoopVersionInfo for loop-version tracking.
    /// `label_args`: label arglist for transitive guard fail_args.
    /// `user_code`: if true, run eliminate_array_bound_checks.
    ///
    /// Returns `(ops, const_values)`. `const_values` maps constant OpRefs
    /// (allocated by IndexVar materialization) to their i64 values.
    /// The caller must register these in the trace's constant pool.
    /// guard.py: propagate_all_forward(info, loop, user_code)
    pub fn propagate_all_forward(
        &mut self,
        ops: &[Op],
        info: &mut super::version::LoopVersionInfo,
        label_args: &[OpRef],
        user_code: bool,
    ) -> (Vec<Op>, indexmap::IndexMap<OpRef, i64>) {
        self.collect_guard_information(ops);
        let mut result = self.eliminate_guards(ops);

        // guard.py:257-266: track loop-version guards.
        assert!(
            info.versions.len() == 1,
            "guard.py:257 assert len(info.versions) == 1"
        );
        let version = info.versions[0].clone();
        for op in &result {
            if !op.opcode.is_guard() {
                continue;
            }
            let __descr_arc_descr = op.getdescr();
            if let Some(descr) = __descr_arc_descr.as_ref()
                && let Some(fd) = descr.as_fail_descr()
                && fd.loop_version()
            {
                // `version.py:32-36 leads_to[descr]` keys by descr
                // object identity (`compute_unique_id(faildescr)` /
                // `descr_identity`).  Pyre uses `Descr::index()` (the
                // globally unique `alloc_fail_index()` value).
                info.track(descr.index(), version.clone());
            }
        }

        // guard.py:268-269
        if user_code {
            let prefix = self.eliminate_array_bound_checks(&mut result, label_args, info);
            if !prefix.is_empty() {
                let mut combined = prefix;
                combined.append(&mut result);
                result = combined;
            }
        }

        (result, std::mem::take(&mut self.const_values))
    }

    /// renamer.py: rename(op) — apply renamer map to op args.
    fn rename_op(&self, op: &mut Op) {
        for i in 0..op.num_args() {
            let arg = op.arg(i).to_opref();
            if let Some(&replacement) = self.renamer.get(&arg) {
                // `renamer.py` carries box objects; the replacement is a
                // producer position, so bind a synthetic producer carrying
                // the same `pos` rather than minting a position-only operand.
                op.setarg(i, majit_ir::operand::Operand::bound_from_opref(replacement));
            }
        }
    }

    /// guard.py: emit_operation(op)
    pub fn emit_operation(&mut self, mut op: Op) {
        self.rename_op(&mut op);
        self._newoperations.push(op);
    }

    /// guard.py: operation_position()
    pub fn operation_position(&self) -> usize {
        self._newoperations.len()
    }
}

/// Helper: track a guard in version_info without borrow conflicts.
fn info_track_guard(
    info: &mut super::version::LoopVersionInfo,
    fail_index: u32,
    version: super::version::LoopVersion,
) {
    info.track(fail_index, version);
}

impl GuardStrengthenOpt {
    /// guard.py: eliminate_array_bound_checks(info, loop)
    ///
    /// `version_info`: LoopVersionInfo for tracking.
    /// `label_args` = `loop.label.getarglist_copy()`.
    /// Mutates `ops` in place (nullifies removed guards then compacts).
    /// Returns prefix ops to prepend to the loop.
    pub fn eliminate_array_bound_checks(
        &mut self,
        ops: &mut Vec<Op>,
        label_args: &[OpRef],
        version_info: &mut super::version::LoopVersionInfo,
    ) -> Vec<Op> {
        // guard.py:280
        version_info.mark();
        // guard.py:281
        let mut version: Option<super::version::LoopVersion> = None;
        // guard.py:282
        self._newoperations = Vec::new();

        // guard.py:283-299
        let mut opt_ops: Vec<Option<Op>> = ops.drain(..).map(Some).collect();
        let guards_snapshot: indexmap::IndexMap<OpRef, Vec<Guard>> = self.strongest_guards.clone();
        for guards in guards_snapshot.values() {
            if guards.len() <= 1 {
                continue;
            }
            let one = &guards[0];
            for other in &guards[1..] {
                // guard.py:291
                let transitive_guard = one.transitive_imply(
                    other,
                    label_args,
                    &mut self._newoperations,
                    &mut self.renamer,
                    &mut self.next_const_pos,
                    &mut self.const_values,
                );
                if let Some(tg) = transitive_guard {
                    // guard.py:293-294: version = info.snapshot(loop)
                    if version.is_none() {
                        let flat_ops: Vec<Op> = opt_ops.iter().filter_map(|o| o.clone()).collect();
                        version = Some(version_info.snapshot(&flat_ops, label_args));
                    }
                    // guard.py:295: info.remove(other.op.getdescr())
                    // RPython: unconditional call. version.py:38-42 asserts
                    // descr is in leads_to — if not, it's a programming
                    // error.  A guard is identified by its descr object
                    // (`compute_unique_id(faildescr)` / `descr_identity`);
                    // pyre keys `leads_to` by `Descr::index()`.  Fail loud
                    // on missing/non-FailDescr
                    // descr instead of silently desynchronizing
                    // `version_info` (stale `leads_to` entry).
                    let other_descr = other
                        .op
                        .getdescr()
                        .expect("guard.py:295 other.op.getdescr() must exist");
                    assert!(
                        other_descr.as_fail_descr().is_some(),
                        "guard.py:295 other.op.getdescr() must be a FailDescr"
                    );
                    version_info.remove(other_descr.index());
                    // guard.py: other.set_to_none(info, loop)
                    other.set_to_none(&mut opt_ops);
                    // guard.py:297-299: info.track(transitive_guard, descr, version)
                    let tg_descr = tg
                        .getdescr()
                        .expect("guard.py:297 transitive_guard.descr must exist");
                    assert!(
                        tg_descr.as_fail_descr().is_some(),
                        "guard.py:297 transitive_guard.descr must be a FailDescr"
                    );
                    info_track_guard(
                        version_info,
                        tg_descr.index(),
                        version.as_ref().unwrap().clone(),
                    );
                }
            }
        }
        // guard.py:300
        version_info.clear();
        // guard.py:303: loop.operations = [op for op in loop.operations if op]
        *ops = opt_ops.into_iter().flatten().collect();

        // guard.py:302: loop.prefix = self._newoperations + loop.prefix
        std::mem::take(&mut self._newoperations)
    }
}

impl Default for GuardStrengthenOpt {
    fn default() -> Self {
        Self::new(indexmap::IndexMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::operand::Operand;

    /// Attach a fresh `ResumeGuardDescr` to every guard op that lacks one,
    /// mirroring RPython's invariant (`optimizer.py:691 assert
    /// isinstance(last_descr, compile.ResumeGuardDescr)`) that every guard
    /// reaching the optimizer has a head-of-chain ResumeGuardDescr — this lets
    /// `_copy_resume_data_from` share via `make_resume_guard_copied_descr(prev)`
    /// without panicking on a missing donor. Positions are assigned at the
    /// bound-DAG build sites where each producer `OpRc`'s result box is taken.
    fn seed_guard_descrs(op: &majit_ir::OpRc) {
        if op.opcode.is_guard() && !op.has_descr() {
            op.setdescr(crate::compile::make_resume_guard_descr_typed(Vec::new()));
        }
    }

    #[test]
    fn test_overflow_guards_preserved_in_full_pipeline() {
        use crate::history::test_support::rooted_inputarg_operand;
        use majit_ir::{OpRc, Type};

        // oparser-faithful bound DAG (`rpython/jit/tool/oparser.py`): header
        // inputs i0/i1/i2 are bound `InputArg` boxes; each producer op is a
        // live `OpRc` whose result box (`from_bound_op`) is the consumer arg,
        // so every op-arg sheds to `Operand::{InputArg,Op}` rather than the
        // position-only `Operand::Box`. The producers are threaded through
        // `optimize_with_constants_and_inputs_oprc` so they are the canonical
        // ops the OptContext indexes.
        let i0 = rooted_inputarg_operand(Type::Int, 0);
        let i1 = rooted_inputarg_operand(Type::Int, 1);
        let i2 = rooted_inputarg_operand(Type::Int, 2);

        // Producer ops carry their result positions (base 100) so they do not
        // collide with the inputarg slots `[0, num_inputs)`.
        let guard_true = OpRc::new(Op::new(OpCode::GuardTrue, std::slice::from_ref(&i1)));
        let sub = OpRc::new(Op::new(OpCode::IntSubOvf, &[i0.clone(), i2.clone()]));
        let guard_ovf1 = OpRc::new(Op::new(OpCode::GuardNoOverflow, &[]));
        let mul = OpRc::new(Op::new(OpCode::IntMulOvf, &[i2.clone(), i1.clone()]));
        let guard_ovf2 = OpRc::new(Op::new(OpCode::GuardNoOverflow, &[]));

        // Sequential positions from base 100 + a fresh ResumeGuardDescr on
        // every guard. Set positions before binding the producer result boxes
        // so `from_bound_op` reads the final pos.
        let producers: [&OpRc; 5] = [&guard_true, &sub, &guard_ovf1, &mul, &guard_ovf2];
        for (i, op) in producers.iter().enumerate() {
            op.pos()
                .set(OpRef::op_typed(100 + i as u32, op.result_type()));
            seed_guard_descrs(op);
        }

        let sub_box = Operand::from_bound_op(&sub);
        let mul_box = Operand::from_bound_op(&mul);
        let jump = OpRc::new(Op::new(
            OpCode::Jump,
            &[sub_box.clone(), sub_box.clone(), mul_box],
        ));
        jump.pos().set(OpRef::op_typed(105, jump.result_type()));

        let ops: Vec<OpRc> = vec![guard_true, sub, guard_ovf1, mul, guard_ovf2, jump];

        let mut opt = Optimizer::default_pipeline();
        // Overflow guards and IntMulOvf work on Int-typed args — override
        // the test default so the renamed inputargs are minted Int, not Ref.
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&vec![majit_ir::Type::Int; 1024]);
        let scratch: Vec<Op> = ops.iter().map(|op| (**op).clone()).collect();
        let (seeded, snapshots) = super::super::seed_empty_guard_snapshots(&scratch);
        for (op, seed) in ops.iter().zip(seeded.iter()) {
            op.set_rd_resume_position(seed.rd_resume_position());
        }
        opt.snapshot_boxes = snapshots;
        let result: Vec<Op> = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut majit_ir::ConstMap::default(), 1024)
            .expect("test: unexpected InvalidLoop")
            .into_iter()
            .map(|rc| (*rc).clone())
            .collect();
        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNoOverflow)
            .count();

        // With intbounds postprocess_GUARD_TRUE, i1 becomes known
        // constant 1 after GuardTrue(i1). IntMulOvf(x, 1) cannot
        // overflow → second GuardNoOverflow is removed. This matches
        // RPython intbounds.py _postprocess_guard_true_false_value.
        assert_eq!(
            guard_count, 1,
            "first overflow guard survives; second removed (mul by constant 1 cannot overflow)"
        );
    }

    #[test]
    fn test_guard_value_to_guard_true() {
        // GUARD_VALUE(v, 1) on a bool-bounded v → GUARD_TRUE(v)
        // (_maybe_replace_guard_value, optimizer.py). The [0,1]
        // bound on v must come from a producer the intbounds pass analyzed
        // (here `int_gt`), not from the guard's own make_constant:
        // postprocess_GUARD_VALUE (rewrite.py) runs make_constant
        // after the guard is emitted, so at emit time the bound is only
        // whatever the producer established.
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::intbounds::OptIntBounds::new()));
        opt.add_pass(Box::new(crate::optimizeopt::rewrite::OptRewrite::new()));
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&[majit_ir::Type::Int; 2]);

        use crate::history::test_support::rooted_inputarg_operand;
        use majit_ir::{OpRc, Type, Value};

        // oparser-faithful bound DAG: i0/i1 are header `InputArg` boxes; the
        // `int_gt` producer is a live `OpRc` whose result box (`from_bound_op`)
        // feeds `guard_value`; the guarded constant `1` is an inline `ConstInt`
        // box (sheds to `Operand::Const`). Every op-arg sheds to
        // `Operand::{InputArg,Op,Const}`, never the position-only `Operand::Box`.
        let i0 = rooted_inputarg_operand(Type::Int, 0);
        let i1 = rooted_inputarg_operand(Type::Int, 1);
        // v = (i0 > i1): intbounds bounds the comparison result to [0,1].
        let int_gt = OpRc::new(Op::new(OpCode::IntGt, &[i0, i1]));
        int_gt.pos().set(OpRef::int_op(100));
        let v = Operand::from_bound_op(&int_gt);
        // guard_value(v, 1)
        let guard_value = OpRc::new(Op::new(
            OpCode::GuardValue,
            &[v, Operand::const_from_value(Value::Int(1))],
        ));
        guard_value.pos().set(OpRef::void_op(0));

        let ops: Vec<OpRc> = vec![int_gt, guard_value];
        let scratch: Vec<Op> = ops.iter().map(|op| (**op).clone()).collect();
        let (seeded, snapshots) = super::super::seed_empty_guard_snapshots(&scratch);
        for (op, seed) in ops.iter().zip(seeded.iter()) {
            op.set_rd_resume_position(seed.rd_resume_position());
        }
        opt.snapshot_boxes = snapshots;
        let result: Vec<Op> = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut majit_ir::ConstMap::default(), 2)
            .expect("test: unexpected InvalidLoop")
            .into_iter()
            .map(|rc| (*rc).clone())
            .collect();

        // GUARD_VALUE should be replaced with GUARD_TRUE
        assert!(
            result.iter().any(|o| o.opcode == OpCode::GuardTrue),
            "GUARD_VALUE(v, 1) should become GUARD_TRUE(v)"
        );
        assert!(
            !result.iter().any(|o| o.opcode == OpCode::GuardValue),
            "GUARD_VALUE should be gone"
        );
    }
}
