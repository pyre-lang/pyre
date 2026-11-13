//! SSA renamer for vectorization loop unrolling.
//!
//! Mirrors RPython's `optimizeopt/renamer.py`.
//! Used during loop unrolling to rename OpRefs from one iteration to the next.

use majit_ir::resoperation::{Op, OpCode, OpRc};
use majit_ir::{InputArg, InputArgRc, OpRef, Type};

use indexmap::IndexMap;
use majit_ir::operand::Operand;

/// renamer.py: Renamer — maps old OpRefs to new OpRefs during unrolling.
///
/// RPython `rename_map` maps box→box (`renamer.py`): the value IS the
/// renamed-to box OBJECT, and `rename` does `op.setarg(i, rename_map.get(arg,
/// arg))` — a BOUND box. pyre's `Op.args` carry `Operand`, which sheds a bound
/// box (`Operand::Op` / `Operand::InputArg`) but mints a position-only
/// `Operand::Box` for an unbound `from_opref` box. So the map value here is the
/// renamed position's BOUND producer box; `rename` re-installs it directly,
/// never re-deriving a position-only box. `to_opref()` of every renamed arg /
/// failarg is byte-identical to the old position-keyed map (same renamed
/// `OpRef`), so the only observable change is the box KIND (bound vs
/// position-only).
pub struct Renamer {
    rename_map: IndexMap<OpRef, Operand>,
    /// Producer `Rc`s minted by [`Renamer::bound_box`] to back the bound map
    /// values. The bound operand holds only a `Weak<Op>` / `Weak<InputArg>`,
    /// so its producer must be rooted for the upgrade to stay live. The
    /// vectorizer's op buffers are `Vec<Op>` (value, not `Rc`), so no live
    /// producer `Rc` is reachable for a renamed-to ResOp position; the renamer
    /// synthesises one carrying the exact renamed `pos` (immutable once the
    /// rename is registered — unroll / clone set the copied op's `pos` once and
    /// never re-number it), so `from_bound_op(&rc).to_opref()` equals the
    /// renamed `OpRef` byte-for-byte. Identity (`Rc::ptr_eq`) is irrelevant: the
    /// dormant vectorizer keys every map by `OpRef`, never by box identity.
    producer_roots: Vec<Box<dyn std::any::Any>>,
}

impl Default for Renamer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renamer {
    pub fn new() -> Self {
        Renamer {
            rename_map: IndexMap::new(),
            producer_roots: Vec::new(),
        }
    }

    fn lookup(&self, opref: OpRef) -> Option<Operand> {
        self.rename_map.get(&opref).cloned()
    }

    /// Like [`Renamer::rename_box`] but returns the BOUND map value (not just
    /// its `OpRef`) on a hit, so callers that re-install an op arg / failarg can
    /// carry the bound box directly instead of re-deriving a position-only box.
    /// `None` on a miss (the caller leaves the existing operand untouched).
    pub fn lookup_box(&self, opref: OpRef) -> Option<Operand> {
        self.lookup(opref)
    }

    /// renamer.py: rename_box — look up the renamed OpRef.
    /// Returns the original if no mapping exists. The map value is now a bound
    /// box, so read its `to_opref()` — byte-identical to the old position-keyed
    /// value.
    pub fn rename_box(&self, opref: OpRef) -> OpRef {
        self.lookup(opref).map(|b| b.to_opref()).unwrap_or(opref)
    }

    /// Materialise a BOUND `Operand` for `r` whose `to_opref()` equals `r`,
    /// rooting a synthetic producer so the bound box's `Weak` stays live.
    ///
    /// Const / None positions shed to `Operand::Const` / none through
    /// `Operand::from_opref` (no `Operand::Box` mint). ResOp / InputArg positions
    /// bind to a freshly-minted, rooted producer `Rc` carrying the same `pos`,
    /// so they shed to `Operand::Op` / `Operand::InputArg`. This is the
    /// production analogue of the test fixtures' `rooted_resop_operand` /
    /// `rooted_inputarg_operand`: a real producer `Rc` is unavailable because the
    /// vectorizer's buffers hold `Op` values, not `OpRc`.
    pub fn bound_box(&mut self, r: OpRef) -> Operand {
        if r.is_none() || r.is_constant() {
            return Operand::from_opref(r);
        }
        let ty = r.ty().unwrap_or(Type::Void);
        let pos = r.raw();
        match r {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                let ia = InputArgRc::new(InputArg::from_type(ty, pos));
                let b = Operand::from_bound_inputarg(&ia);
                self.producer_roots.push(Box::new(ia));
                b
            }
            _ => {
                let opcode = match ty {
                    Type::Int => OpCode::SameAsI,
                    Type::Float => OpCode::SameAsF,
                    Type::Ref => OpCode::SameAsR,
                    Type::Void => OpCode::Jump,
                };
                let op: OpRc = OpRc::new(Op::new(opcode, &[]));
                op.pos().set(OpRef::op_typed(pos, ty));
                let b = Operand::from_bound_op(&op);
                self.producer_roots.push(Box::new(op));
                b
            }
        }
    }

    /// renamer.py: start_renaming — register a mapping from `var` to
    /// `tovar`.
    ///
    /// The renamed-to position is bound to a rooted producer box (see
    /// [`Renamer::bound_box`]) so `rename` re-installs a BOUND operand. The
    /// `tovar.is_constant()` skip is kept (renamer.py): a constant in the
    /// rename target is never installed (constants are not allowed in failargs).
    pub fn start_renaming(&mut self, var: OpRef, tovar: OpRef) {
        // renamer.py: don't rename constants.
        if tovar.is_constant() {
            return;
        }
        let bound = self.bound_box(tovar);
        self.rename_map.insert(var, bound);
    }

    /// renamer.py: rename — apply renaming to all args and fail_args of an op.
    pub fn rename(&self, op: &mut Op) -> bool {
        // renamer.py:21-23:
        //   for i, arg in enumerate(op.getarglist()):
        //       arg = self.rename_map.get(arg, arg)
        //       op.setarg(i, arg)
        // A hit re-installs the BOUND map value directly (no position-only
        // re-derivation); a miss leaves the existing operand untouched.
        for i in 0..op.num_args() {
            let arg = op.arg(i);
            if let Some(renamed) = self.rename_map.get(&arg.to_opref()) {
                op.setarg(i, renamed.clone());
            }
        }

        if op.opcode.is_guard() {
            // renamer.py: TODO op.rd_snapshot = self.rename_rd_snapshot(...)
            // renamer.py: failargs = self.rename_failargs(op, clone=True)
            // renamer.py: `self.rename_map.get(arg, arg)` — a missed
            // lookup keeps the SAME box object, so only hits are rewritten.
            if let Some(fail_args) = op.fail_args_mut() {
                for arg in fail_args.iter_mut() {
                    if let Some(renamed) = self.lookup(arg.to_opref()) {
                        *arg = renamed;
                    }
                }
            }
        }

        true
    }

    /// renamer.py: rename_failargs — rename a slice of fail_args.
    pub fn rename_failargs(&self, fail_args: &[OpRef]) -> Vec<OpRef> {
        fail_args.iter().map(|arg| self.rename_box(*arg)).collect()
    }

    /// renamer.py: rename_rd_snapshot — recursively rename snapshot boxes.
    /// In RPython, snapshots are nested MIFrame structures. In majit, resume data
    /// uses rd_numb (compact varint encoding), so this is a no-op for now,
    /// matching RPython's own TODO comment at renamer.py.
    pub fn rename_rd_snapshot(&self, _rd_numb: &Option<Vec<u8>>) -> Option<Vec<u8>> {
        // RPython: TODO op.rd_snapshot = self.rename_rd_snapshot(op.rd_snapshot, clone=True)
        // Not yet implemented in RPython's vector optimizer either.
        None
    }
}
