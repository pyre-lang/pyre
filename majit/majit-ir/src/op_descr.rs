//! `ResOpWithDescr.{getdescr,setdescr,cleardescr}` accessors plus
//! per-trait `with_*_descr` shortcuts and `rd_*` Arc-returning
//! resolvers, separated from `resoperation.rs` so the build-script
//! source analyzer in `pyre-jit-trace/build.rs` (which reads
//! `resoperation.rs` for the `RdVirtualInfo` enum declarations) does
//! not need to lower the closure-bearing accessor surface.
//!
//! `Op.descr` is now `RefCell<Option<DescrRef>>` so the optimizer can
//! stamp a `ResumeGuardDescr` onto a shared `Op` (reached through
//! `Rc<Op>`) the same way RPython's `op.setdescr(d)` writes on a shared
//! Python object. `setdescr` / `cleardescr` therefore take `&self`.

use std::rc::Rc;
use std::sync::Arc;

use crate::descr::{
    ArrayDescr, CallDescr, Descr, DescrRef, FailDescr, FieldDescr, InteriorFieldDescr,
    LoopTargetDescr, LoopTokenDescr, SizeDescr,
};
use crate::resoperation::{GuardPendingFieldEntry, Op, RdVirtualInfo};

impl Op {
    /// `resoperation.py AbstractResOpOrInputArg.getdescr` + `:462
    /// ResOpWithDescr.getdescr` parity. Returns an owned `Arc` clone of
    /// the `DescrRef` so callers can chain `.as_ref()`, `.expect()`, or
    /// pattern-match without holding a `RefCell` borrow across the call.
    pub fn getdescr(&self) -> Option<DescrRef> {
        self.descr.borrow()
    }

    /// `resoperation.py ResOpWithDescr.setdescr` parity — overwrites
    /// the descr slot.  Takes `&self` (interior mutability) so callers
    /// holding a shared `Op` (e.g., through `Rc<Op>`) can stamp a fresh
    /// descr the same way RPython's `op.setdescr(d)` writes on a shared
    /// Python object.
    pub fn setdescr(&self, descr: DescrRef) {
        self.descr.set_descr(Some(descr));
    }

    /// `resoperation.py ResOpWithDescr.cleardescr` parity — clears
    /// the descr slot.
    pub fn cleardescr(&self) {
        self.descr.set_descr(None);
    }

    // `has_descr` lives in `resoperation.rs` so the build-script
    // source analyzer can resolve the bool return type when callers
    // appear in `!op.has_descr()` patterns inside that file.

    /// `resoperation.py VectorizationInfo` slot accessor —
    /// returns an owned clone of the per-op vector metadata installed
    /// by the vectorizer.
    pub fn get_vecinfo(&self) -> Option<crate::resoperation::VectorizationInfo> {
        self.vecinfo_slot()
    }

    /// Overwrite the per-op vector metadata slot.  Takes `&self` —
    /// interior mutability through `RefCell` matches RPython's
    /// `op._vector_info = …` write on a shared object.
    pub fn set_vecinfo(&self, vecinfo: crate::resoperation::VectorizationInfo) {
        self.set_vecinfo_slot(vecinfo);
    }

    /// Clear the per-op vector metadata slot.
    pub fn clear_vecinfo(&self) {
        self.clear_vecinfo_slot();
    }

    /// True iff the per-op vector metadata slot is populated.
    pub fn has_vecinfo(&self) -> bool {
        self.vecinfo_slot().is_some()
    }

    /// Project the descr (if any) through a closure operating on a
    /// `&dyn Descr`. `f` may freely return owned values derived from
    /// borrowed projections (`as_field_descr`, `as_array_descr`, etc.).
    /// Holds the descr `Ref` for the duration of `f`; the inner trait
    /// object lives behind `Arc`, so callers can safely keep an owned
    /// clone if they need to outlive the borrow.
    pub fn project_descr<R>(&self, f: impl FnOnce(&dyn Descr) -> R) -> Option<R> {
        self.descr.borrow().as_ref().map(|d| f(&**d))
    }

    /// `as_field_descr` shortcut.
    pub fn with_field_descr<R>(&self, f: impl FnOnce(&dyn FieldDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_field_descr().map(f)).flatten()
    }

    /// `as_array_descr` shortcut.
    pub fn with_array_descr<R>(&self, f: impl FnOnce(&dyn ArrayDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_array_descr().map(f)).flatten()
    }

    /// `as_call_descr` shortcut.
    pub fn with_call_descr<R>(&self, f: impl FnOnce(&dyn CallDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_call_descr().map(f)).flatten()
    }

    /// `as_loop_target_descr` shortcut.
    pub fn with_loop_target_descr<R>(
        &self,
        f: impl FnOnce(&dyn LoopTargetDescr) -> R,
    ) -> Option<R> {
        self.project_descr(|d| d.as_loop_target_descr().map(f))
            .flatten()
    }

    /// `as_size_descr` shortcut.
    pub fn with_size_descr<R>(&self, f: impl FnOnce(&dyn SizeDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_size_descr().map(f)).flatten()
    }

    /// `as_fail_descr` shortcut.
    pub fn with_fail_descr<R>(&self, f: impl FnOnce(&dyn FailDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_fail_descr().map(f)).flatten()
    }

    /// `as_loop_token_descr` shortcut.
    pub fn with_loop_token_descr<R>(&self, f: impl FnOnce(&dyn LoopTokenDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_loop_token_descr().map(f))
            .flatten()
    }

    /// `as_interior_field_descr` shortcut.
    pub fn with_interior_field_descr<R>(
        &self,
        f: impl FnOnce(&dyn InteriorFieldDescr) -> R,
    ) -> Option<R> {
        self.project_descr(|d| d.as_interior_field_descr().map(f))
            .flatten()
    }

    /// `compile.py ResumeGuardCopiedDescr.get_resumestorage(): return prev`
    /// parity. Reads `rd_numb` from `op.descr` — `ResumeGuardCopiedDescr`
    /// chases `prev` automatically.  Returns `NumberingRef` so the slice
    /// stays valid once the borrow on `op.descr` drops.
    pub fn resolved_rd_numb(&self) -> Option<crate::NumberingRef> {
        self.descr.borrow().as_ref()?.as_fail_descr()?.rd_numb_arc()
    }

    /// Same as `resolved_rd_numb` but for the `rd_consts` const pool.
    pub fn resolved_rd_consts(&self) -> Option<Arc<crate::SharedConstPool>> {
        self.descr
            .borrow()
            .as_ref()?
            .as_fail_descr()?
            .rd_consts_arc()
    }

    /// Same as `resolved_rd_numb` but for the `rd_virtuals` table.
    pub fn resolved_rd_virtuals(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.descr
            .borrow()
            .as_ref()?
            .as_fail_descr()?
            .rd_virtuals_arc()
    }

    /// Same as `resolved_rd_numb` but for the `rd_pendingfields` table.
    pub fn resolved_rd_pendingfields(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.descr
            .borrow()
            .as_ref()?
            .as_fail_descr()?
            .rd_pendingfields_arc()
    }

    /// Owned snapshot of `GuardResOp.getfailargs` (`self._fail_args[:]`).
    /// The live list is [`Op::guard_fail_args`]; use that on hot reads.
    /// A `SmallVec<[; 4]>` clone stays inline for the common four-or-fewer
    /// failarg list (same capacity as `OpArgVec` / `setfailargs`).
    pub fn getfailargs(&self) -> Option<crate::resoperation::OpArgVec> {
        self.guard_fail_args()
            .map(|fa| fa.iter().cloned().collect())
    }

    /// `resoperation.py GuardResOp.getfailargs_copy` parity.
    /// Returns an owned `Vec` copy of the fail_args slot — equivalent
    /// to RPython's `self._fail_args[:]`, which raises `TypeError:
    /// 'NoneType' object is not subscriptable` when `_fail_args` is
    /// `None`.  Mirror that fail-loud behaviour here so a missing-
    /// fail-args bug surfaces at the call site rather than returning
    /// a silently-empty vector.
    pub fn getfailargs_copy(&self) -> Vec<crate::operand::Operand> {
        let borrow = self.guard_fail_args();
        let fa = borrow.as_ref().unwrap_or_else(|| {
            panic!(
                "getfailargs_copy on op with fail_args=None — RPython \
                 `self._fail_args[:]` raises TypeError; pyre matches the \
                 fail-loud shape (resoperation.py:492)"
            )
        });
        fa.to_vec()
    }

    /// `resoperation.py GuardResOp.setfailargs` parity — overwrite
    /// the fail_args slot.  Takes `&self` (interior mutability) so the
    /// optimizer can stamp fail_args onto a shared `Op` reached through
    /// `Rc<Op>`.
    /// Rewrite each live fail-arg in place. Avoids a `SmallVec` clone
    /// when the caller only remaps boxes already stored on the guard.
    pub fn map_failargs_in_place(&self, mut f: impl FnMut(&mut crate::operand::Operand)) {
        if let Some(g) = self.try_guard_extra_mut()
            && let Some(fa) = g.fail_args_mut()
        {
            for arg in fa.iter_mut() {
                f(arg);
            }
        }
    }

    pub fn setfailargs(&self, fail_args: crate::resoperation::OpArgVec) {
        // `GuardResOp._fail_args` holds the producer operands themselves
        // (resoperation.py `setfailargs`), exactly like `Op.args`: a bound
        // fail-arg is its `Operand::Op`/`Operand::InputArg` producer, a
        // constant is `Operand::Const`. An unbound position-only fail-arg
        // is a contract violation (every writer binds its producer) —
        // `Operand::from_opref` panics for it at the call site.
        //
        // Four inline slots match `OpArgVec`: a 4-failarg list stays on
        // the stack here and becomes one `Vec` heap, not a `[; 3]` spill
        // plus a second collect.
        self.ensure_guard_extra().set_fail_args(fail_args);
    }

    /// Share the source guard's `_fail_args` list (same slab slot).
    /// Avoids a second 4-/6-operand heap when stamp/copy already has
    /// the replacements on `src`.
    pub fn copy_failargs_shared(&self, src: &Self) {
        match src.try_guard_extra() {
            Some(g) => self.ensure_guard_extra().share_fail_args_from(g),
            None => self.clearfailargs(),
        }
    }

    /// In-place mutable view of the fail_args slot.  Lets callers iterate
    /// the slice mutably (`fa.iter_mut()`, `fa[i] = …`) without going
    /// through a clone/setfailargs round-trip.  Returns `None` when the
    /// slot is empty.  Routes through [`DescrSlot::extra_mut`] so a
    /// shared `Rc<Op>` can rewrite failargs on the same ResOperation.
    pub fn fail_args_mut(&self) -> Option<&mut [crate::operand::Operand]> {
        match self.descr.extra_mut()? {
            crate::resoperation::OpKindExtra::Guard(g) => g.fail_args_mut(),
            crate::resoperation::OpKindExtra::VectorGuard(vg) => vg.guard.fail_args_mut(),
            crate::resoperation::OpKindExtra::Vector(_) => None,
        }
    }

    /// Clear the fail_args slot.  PyPy has no separate `clearfailargs`
    /// method; the pattern is `op.setfailargs(None)` in RPython, but
    /// pyre's signature distinguishes the two paths (set vs clear) for
    /// clarity.
    pub fn clearfailargs(&self) {
        if let Some(g) = self.try_guard_extra_mut() {
            g.clear_fail_args();
        }
    }

    /// True iff the fail_args slot is populated.
    pub fn has_failargs(&self) -> bool {
        self.guard_fail_args().is_some()
    }

    /// Per-failarg type vector accessor.  Pyre's `fail_arg_types` slot
    /// caches the types the optimizer assigned to each `fail_arg` (the
    /// `compile.py:855 _attrs_` set lives on the descr, but the
    /// per-op view is kept here for backend dispatch convenience).
    /// Returns an owned `Vec` clone now that the slot is `RefCell`-
    /// wrapped; callers no longer hold a borrow across other `Op`
    /// accesses.
    pub fn get_fail_arg_types(&self) -> Option<Vec<crate::value::Type>> {
        self.with_fail_arg_types(|t| t.to_vec())
    }

    /// history.py Box.type lives on the box. Walk the descr / extra slice
    /// in place — `get_fail_arg_types` clones a `Vec<Type>` per call.
    pub fn with_fail_arg_types<R>(&self, f: impl FnOnce(&[crate::value::Type]) -> R) -> Option<R> {
        self.try_guard_extra()
            .and_then(|g| g.fail_arg_types().map(f))
    }

    /// Owned-clone variant — RPython would write `fail_arg_types[:]`.
    pub fn get_fail_arg_types_copy(&self) -> Vec<crate::value::Type> {
        self.get_fail_arg_types().unwrap_or_default()
    }

    /// Overwrite the per-failarg type vector.  Takes `&self`
    /// (interior mutability through `RefCell`) so shared `Op` instances
    /// can be re-stamped without `&mut`.
    pub fn set_fail_arg_types(&self, types: Vec<crate::value::Type>) {
        self.ensure_guard_extra().set_fail_arg_types(&types);
    }

    /// Clear the per-failarg type vector.
    pub fn clear_fail_arg_types(&self) {
        if let Some(g) = self.try_guard_extra_mut() {
            g.clear_fail_arg_types();
        }
    }

    /// True iff the per-failarg type vector slot is populated.
    pub fn has_fail_arg_types(&self) -> bool {
        self.get_fail_arg_types().is_some()
    }

    /// `resoperation.py AbstractResOp.getarglist` parity — the stored
    /// [`Operand`](crate::operand::Operand) args.  Subclass mixins
    /// (`UnaryOp`, `BinaryOp`, ..., `N_aryOp`) implement this differently;
    /// pyre collapses them into a single SmallVec slot.
    ///
    /// Returns an owned `SmallVec` (clones of the `Rc`-cheap operands)
    /// rather than borrowing the slot, dropping the `RefCell` borrow at
    /// the call boundary so a caller may freely `setarg` afterwards.
    pub fn getarglist(&self) -> crate::resoperation::OpArgVec {
        self.args.clone_vec(self.arg_len_value())
    }

    /// `resoperation.py AbstractResOp.getarglist` parity for a reader
    /// that only walks the arguments: upstream returns `self._args` itself,
    /// and only `getarglist_copy` (`N_aryOp`) returns `self._args[:]`.
    /// [`getarglist`](Self::getarglist) has to copy so the `RefCell` borrow
    /// does not outlive the call, which charges every read an `Operand` clone
    /// per argument and a heap spill past the inline three; this hands the
    /// stored slot to `f` instead.
    ///
    /// `f` must not call `setarg` / `initarglist` on the same op — the borrow
    /// is live for its body, which is the invariant the copying accessor buys
    /// its callers.
    pub fn with_arglist<R>(&self, f: impl FnOnce(&[crate::operand::Operand]) -> R) -> R {
        f(self.args_slice())
    }

    /// `resoperation.py AbstractResOp.getarglist_copy` parity —
    /// `N_aryOp.getarglist_copy` returns `self._args[:]`; pyre returns
    /// an owned `SmallVec` of the stored operands.
    pub fn getarglist_copy(&self) -> crate::resoperation::OpArgVec {
        self.args.clone_vec(self.arg_len_value())
    }

    /// `resoperation.py AbstractResOp.initarglist` parity — bulk
    /// store the operand list.  In RPython this is "supposed to be
    /// called only just after the ResOp has been created"
    /// (resoperation.py); pyre matches both the name and intent.
    ///
    /// One existing post-creation caller in upstream:
    /// `unroll.py:301 label_op.initarglist(label_op.getarglist() +
    ///  sb.used_boxes)` extends the LABEL arg list when finishing the
    /// peel pass.  pyre's matching call lives in `unroll.rs` and
    /// rebuilds the SmallVec rather than pushing onto `args`.
    pub fn initarglist(&self, args: impl IntoIterator<Item = crate::operand::Operand>) {
        let new_len = self
            .args
            .replace(self.arg_len_value(), args.into_iter().collect());
        self.set_arg_len_value(new_len);
    }

    /// `resoperation.py AbstractResOp.setarg` parity — position-wise
    /// in-place arg mutation.  Subclass mixins index `_arg0/_arg1/...`
    /// or `_args[i]`; pyre indexes the SmallVec directly.
    pub fn setarg(&self, i: usize, arg: crate::operand::Operand) {
        self.args_slice_mut()[i] = arg;
    }
}
