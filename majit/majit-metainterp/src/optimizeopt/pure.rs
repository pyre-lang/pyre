use std::cell::RefCell;
use std::rc::Rc;

use indexmap::IndexMap;
use majit_ir::operand::Operand;
/// Pure operation optimization (Common Subexpression Elimination).
///
/// Translated from rpython/jit/metainterp/optimizeopt/pure.py.
///
/// When the same pure operation is seen again with the same arguments,
/// the cached result is returned instead of recomputing.
use majit_ir::{GcRef, Op, OpCode, OpRc, OpRef, Value};

use crate::optimizeopt::info::{PreambleOp, PtrInfoExt};
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// pure.py: extra_call_pure entry.
/// RPython stores AbstractResOp (or PreambleOp) directly in the list.
/// isinstance(old_op, PreambleOp) check → force_op_from_preamble → replace.
#[derive(Clone, Debug)]
enum ExtraCallPureEntry {
    Direct { key: PureOpKey, result: OpRef },
    Preamble { key: PureOpKey, pop: PreambleOp },
}

/// Key for looking up a previously computed pure operation.
///
/// RecentPureOps keeps native argument boxes, as pure.py does, so later
/// forwarding writes are visible without recreating their identities.
/// The existing imported-call adapter still uses positional arguments.
///
/// RPython's optimizeopt/pure.py includes the descriptor in pure-op identity
/// checks for operations like GETFIELD_GC_PURE_*; otherwise distinct immutable
/// fields on the same object can be incorrectly CSE'd together.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PureOpKey<A = OpRef> {
    opcode: OpCode,
    args: smallvec::SmallVec<[A; 2]>,
    descr_identity: Option<usize>,
}

impl PureOpKey<Operand> {
    fn from_operand_op(op: &Op) -> Self {
        Self {
            opcode: op.opcode,
            args: op.with_arglist(|args| args.iter().cloned().collect()),
            descr_identity: op.getdescr().as_ref().map(majit_ir::descr::descr_identity),
        }
    }
}

/// pure.py: known_result_call_pure entry.
/// RPython stores the full RECORD_KNOWN_RESULT op and compares by descr +
/// _same_args(known_op, query_op, 1, start_index). We pre-extract the
/// fields to avoid storing a dummy PureOpKey with an opcode.
#[derive(Clone, Debug)]
struct KnownResultEntry {
    descr_identity: Option<usize>,
    /// args[1..] from the RECORD_KNOWN_RESULT op (the call arguments).
    args: Vec<OpRef>,
    /// arg(0) from the RECORD_KNOWN_RESULT op (the known result).
    result: OpRef,
}

/// Ring payload. RPython `RecentPureOps.lst` holds the `AbstractResOp`
/// itself, which may be a `PreambleOp`; lookup then `force_preamble_op`
/// replaces that slot in place.
#[derive(Clone, Debug)]
enum PureRingValue {
    Direct(Operand),
    Preamble {
        pop: PreambleOp,
        forced: Option<Operand>,
    },
}

impl PureRingValue {
    fn as_direct(&self) -> Option<Operand> {
        match self {
            Self::Direct(result) => Some(result.clone()),
            Self::Preamble { forced, .. } => forced.clone(),
        }
    }
}

/// pure.py RecentPureOps — fixed-size ring buffer with linear scan.
///
/// RPython uses a flat array of Op references, scanned linearly on lookup.
/// At limit=16 (pureop_historylength), linear scan beats HashMap because:
/// - No hashing overhead or `Vec<OpRef>` allocation per lookup
/// - Cache-friendly sequential memory access
/// - Typical hit is within first few entries
pub struct RecentPureOps {
    /// Ring buffer of (key, result) pairs. None = empty slot.
    lst: Vec<Option<(PureOpKey<Operand>, PureRingValue)>>,
    next_index: usize,
}

impl RecentPureOps {
    fn new(limit: usize) -> Self {
        RecentPureOps {
            lst: vec![None; limit],
            next_index: 0,
        }
    }

    /// pure.py — add(op): record a pure operation result.
    fn insert(&mut self, key: PureOpKey<Operand>, result: Operand) {
        self.insert_value(key, PureRingValue::Direct(result));
    }

    /// shortpreamble.py `opt.pure(opnum, PreambleOp(...))` — the same ring
    /// RPython `getrecentops` writes, not a side list.
    fn insert_preamble(&mut self, key: PureOpKey<Operand>, pop: PreambleOp) {
        self.insert_value(key, PureRingValue::Preamble { pop, forced: None });
    }

    fn insert_value(&mut self, key: PureOpKey<Operand>, value: PureRingValue) {
        self.lst[self.next_index] = Some((key, value));
        self.next_index += 1;
        if self.next_index >= self.lst.len() {
            self.next_index = 0;
        }
    }

    fn clear(&mut self) {
        for slot in &mut self.lst {
            *slot = None;
        }
        self.next_index = 0;
    }

    /// pure.py lookup(self, optimizer, op, commutative=False).
    ///
    /// Dispatches to `lookup1` / `lookup2` by arg count; any other
    /// numargs hits the upstream `assert False` because OptPure only
    /// runs against ALWAYS_PURE ops with 1 or 2 args (`LOAD_EFFECTIVE_ADDRESS`
    /// has 4 args but is emitted by `rewrite.py` after OptPure).
    ///
    /// Matching uses producer identity after forwarding, or constant value
    /// equality (`history.py Const.same_box`).

    /// pure.py `lookup` + `force_preamble_op`: a PreambleOp slot is
    /// replaced in place with the forced body op.
    fn lookup_forcing(
        &mut self,
        op: &Op,
        ctx: &mut OptContext,
        commutative: bool,
    ) -> Option<(OpCode, Operand)> {
        let index = self.find_index(op, ctx, commutative)?;
        self.force_at(index, ctx)
    }

    fn find_index(&self, op: &Op, ctx: &OptContext, commutative: bool) -> Option<usize> {
        let descr_identity = op.getdescr().as_ref().map(majit_ir::descr::descr_identity);
        op.with_arglist(|args| {
            let arg0 = args
                .first()
                .and_then(|arg| ctx.resolve_operand_operand_opt(arg));
            let arg1 = args
                .get(1)
                .and_then(|arg| ctx.resolve_operand_operand_opt(arg));
            let same_box =
                |(raw, query): (OpRef, Option<&Operand>),
                 (stored, resolved): (OpRef, Option<&Operand>)| match (
                    query, resolved,
                ) {
                    (Some(query), Some(stored)) => query.same_box(stored),
                    _ => ctx.same_box(raw, stored),
                };
            match args.len() {
                1 => {
                    let query = (args[0].to_opref(), arg0.as_ref());
                    self.lst.iter().position(|entry| {
                        let Some((k, _)) = entry else {
                            return false;
                        };
                        k.args.len() == 1
                            && k.descr_identity == descr_identity
                            && k.args[0].with_box_replacement(|stored| {
                                same_box(query, (k.args[0].to_opref(), Some(stored)))
                            })
                    })
                }
                2 => {
                    let q0 = (args[0].to_opref(), arg0.as_ref());
                    let q1 = (args[1].to_opref(), arg1.as_ref());
                    self.lst.iter().position(|entry| {
                        let Some((k, _)) = entry else {
                            return false;
                        };
                        if k.args.len() != 2 || k.descr_identity != descr_identity {
                            return false;
                        }
                        let match_fwd = |query: (OpRef, Option<&Operand>), stored: &Operand| {
                            same_box(query, (stored.to_opref(), Some(stored)))
                        };
                        k.args[0].with_box_replacement(|s0| {
                            k.args[1].with_box_replacement(|s1| {
                                (match_fwd(q0, s0) && match_fwd(q1, s1))
                                    || (commutative && match_fwd(q1, s0) && match_fwd(q0, s1))
                            })
                        })
                    })
                }
                _ => panic!(
                    "RecentPureOps::find_index: numargs must be 1 or 2, got {}",
                    args.len()
                ),
            }
        })
    }

    fn force_at(&mut self, index: usize, ctx: &mut OptContext) -> Option<(OpCode, Operand)> {
        let (opcode, pop) = {
            let (key, value) = self.lst[index].as_ref()?;
            match value {
                PureRingValue::Direct(result) => return Some((key.opcode, result.clone())),
                PureRingValue::Preamble {
                    pop,
                    forced: Some(result),
                } => return Some((key.opcode, result.clone())),
                PureRingValue::Preamble { pop, forced: None } => (key.opcode, pop.clone()),
            }
        };
        let forced = ctx.force_op_from_preamble_op(&pop);
        if let Some((_, value)) = self.lst[index].as_mut() {
            *value = PureRingValue::Direct(forced.clone());
        }
        Some((opcode, forced))
    }

    /// pure.py lookup1(opt, box0, descr).
    ///
    /// RPython: `box0.same_box(get_box_replacement(op.getarg(0)))`.
    /// `same_box` is identity for non-constants, value equality for constants.
    /// `Q` can carry an already-resolved query; the callback resolves stored args.
    fn lookup1<Q: Copy>(
        &self,
        arg0: Q,
        descr_identity: Option<usize>,
        same_box: impl Fn(Q, &Operand) -> bool,
    ) -> Option<(OpCode, Operand)> {
        for entry in &self.lst {
            let Some((k, result)) = entry else { break };
            if k.args.len() != 1 {
                continue;
            }
            if k.descr_identity != descr_identity {
                continue;
            }
            // pure.py — box0.same_box(get_box_replacement(op.getarg(0)))
            if same_box(arg0, &k.args[0]) {
                return result.as_direct().map(|r| (k.opcode, r));
            }
        }
        None
    }

    /// pure.py lookup2(opt, box0, box1, descr, commutative).
    ///
    /// The callback compares a query against a forwarded stored argument.
    fn lookup2<Q: Copy, S>(
        &self,
        arg0: Q,
        arg1: Q,
        descr_identity: Option<usize>,
        commutative: bool,
        resolve: impl Fn(&Operand) -> S,
        same_box: impl Fn(Q, &S) -> bool,
    ) -> Option<(OpCode, Operand)> {
        for entry in &self.lst {
            let Some((k, result)) = entry else { break };
            if k.args.len() != 2 {
                continue;
            }
            if k.descr_identity != descr_identity {
                continue;
            }
            // pure.py RecentPureOps.lookup2 resolves each stored operand once,
            // then compares both orders against those same terminal boxes.
            let stored0 = resolve(&k.args[0]);
            let stored1 = resolve(&k.args[1]);
            if (same_box(arg0, &stored0) && same_box(arg1, &stored1))
                || (commutative && same_box(arg1, &stored0) && same_box(arg0, &stored1))
            {
                return result.as_direct().map(|r| (k.opcode, r));
            }
        }
        None
    }
}

pub(crate) type SharedPureOps = Rc<RefCell<RecentPureOpTable>>;

/// `OptPure.get_pure_result` against a ring published into `OptContext`.
///
/// `Optimization.get_pure_result` reads `optimizer.optpure`; the pass list
/// cannot name that pass, so `OptPure` publishes this handle and
/// `OptRewrite` probes it. Lookup is the same `RecentPureOps.lookup` the
/// pass uses for CSE, including the in-place preamble force.
pub(crate) fn shared_get_pure_result(
    table: &SharedPureOps,
    op: &Op,
    ctx: &mut OptContext,
) -> Option<Operand> {
    let commutative = OptPure::is_commutative(op.opcode);
    let found = table.borrow_mut().lookup_forcing(op, ctx, commutative);
    found
        .filter(|(_, result)| OptPure::matches_result_type(op, result, ctx))
        .map(|(_, result)| result)
}

pub(crate) struct RecentPureOpTable {
    buckets: Vec<Option<RecentPureOps>>,
    history_length: usize,
}

impl RecentPureOpTable {
    fn new(limit: usize) -> Self {
        let bucket_count = Self::bucket_count();
        let buckets = std::iter::repeat_with(|| None).take(bucket_count).collect();
        RecentPureOpTable {
            buckets,
            history_length: limit,
        }
    }

    fn bucket_count() -> usize {
        OpCode::LoadEffectiveAddress as usize - OpCode::IntAdd as usize + 1
    }

    fn bucket_index(opcode: OpCode) -> Option<usize> {
        if opcode.is_ovf() {
            return Some(opcode as usize - OpCode::IntAddOvf as usize);
        }
        match opcode {
            _ if opcode.is_always_pure() => Some(opcode as usize - OpCode::IntAdd as usize),
            _ => None,
        }
    }

    fn bucket(&self, opcode: OpCode) -> Option<&RecentPureOps> {
        let idx = Self::bucket_index(opcode)?;
        self.buckets.get(idx)?.as_ref()
    }

    fn bucket_mut(&mut self, opcode: OpCode) -> Option<&mut RecentPureOps> {
        let idx = Self::bucket_index(opcode)?;
        if self.buckets[idx].is_none() {
            self.buckets[idx] = Some(RecentPureOps::new(self.history_length));
        }
        self.buckets[idx].as_mut()
    }

    fn lookup_forcing(
        &mut self,
        op: &Op,
        ctx: &mut OptContext,
        commutative: bool,
    ) -> Option<(OpCode, Operand)> {
        self.bucket_mut(op.opcode)?
            .lookup_forcing(op, ctx, commutative)
    }

    fn insert(&mut self, key: PureOpKey<Operand>, result: Operand) {
        if let Some(bucket) = self.bucket_mut(key.opcode) {
            bucket.insert(key, result);
        }
    }

    fn insert_preamble(&mut self, key: PureOpKey<Operand>, pop: PreambleOp) {
        if let Some(bucket) = self.bucket_mut(key.opcode) {
            bucket.insert_preamble(key, pop);
        }
    }

    fn lookup1(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        descr_identity: Option<usize>,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        self.bucket(opcode)
            .and_then(|bucket| {
                bucket.lookup1(arg0, descr_identity, |query, stored| {
                    same_box(query, stored.to_opref())
                })
            })
            .map(|(_, result)| result.to_opref())
    }

    fn lookup2(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        arg1: OpRef,
        descr_identity: Option<usize>,
        commutative: bool,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        self.bucket(opcode)
            .and_then(|bucket| {
                bucket.lookup2(
                    arg0,
                    arg1,
                    descr_identity,
                    commutative,
                    |arg| arg.to_opref(),
                    |query, stored| same_box(query, *stored),
                )
            })
            .map(|(_, result)| result.to_opref())
    }

    #[allow(dead_code)] // pure.py OptPure.history length
    fn history_length(&self) -> usize {
        self.history_length
    }

    /// Drop ring entries, keep the bucket `Vec` and each ring's allocation.
    /// `pure.py OptPure.__init__` builds `_pure_operations` once; `setup`
    /// does not reallocate it.
    fn clear(&mut self) {
        for ops in self.buckets.iter_mut().flatten() {
            ops.clear();
        }
    }
}

/// The OptPure optimization pass.
///
/// pure.py: OptPure class.
/// For pure operations (is_always_pure), checks if the same operation was
/// computed before. If yes, replaces the current op with the cached result
/// (CSE). If no, records the operation for future lookups.
///
/// Also handles:
/// - CALL_PURE -> CALL demotion when arguments aren't all constant.
/// - OVF operation postponement (INT_ADD_OVF etc. are deferred until GUARD_NO_OVERFLOW).
/// - GUARD_NO_EXCEPTION removal after eliminated CALL_PURE.
/// - RECORD_KNOWN_RESULT for pre-recorded call_pure results.
pub struct OptPure {
    /// The per-opnum recent-ops ring. Shared with `OptRewrite` through
    /// `OptContext::pure_ops` (one `Rc`, not a copy). `setup` clears the
    /// ring in place; `set_pureop_historylength` replaces it.
    cache: SharedPureOps,
    /// Postponed OVF operation: INT_ADD_OVF, INT_SUB_OVF, INT_MUL_OVF.
    /// pure.py: postponed_op — deferred until GUARD_NO_OVERFLOW is seen.
    /// `self.postponed_op = op` keeps the op object itself, so a later
    /// `make_equal_to` / `make_constant` on it and its eventual emission
    /// name the same box.
    postponed_op: Option<OpRc>,
    /// Indices into new_operations of emitted CALL_PURE ops.
    /// pure.py: call_pure_positions — tracked for short preamble generation.
    call_pure_positions: Vec<usize>,
    /// One-shot arm for `CallPureOptimizationResult.callback()`
    /// (pure.py). Set when `optimize_call_pure` demotes a CALL_PURE and
    /// returns `Replace`; consumed in `propagate_postprocess` to append the
    /// op's final `new_operations` index AFTER emission. RPython attaches the
    /// callback to the specific result object; pyre dispatches postprocess
    /// per pass, so a per-op flag stands in for that per-result binding.
    pending_call_pure_position: bool,
    /// Whether the last emitted operation was removed (for GUARD_NO_EXCEPTION elimination).
    /// pure.py: last_emitted_operation is REMOVED check.
    last_emitted_was_removed: bool,
    /// Pre-recorded CALL_PURE results from RECORD_KNOWN_RESULT.
    /// pure.py: known_result_call_pure — stores the full RECORD_KNOWN_RESULT op.
    /// RPython lookup: descr + _same_args(known_op, op, 1, start_index).
    /// We store (descr_identity, args_from_1, result) — no opcode comparison.
    known_result_call_pure: Vec<KnownResultEntry>,
    /// pure.py: extra_call_pure — CALL_PURE results from the previous
    /// loop iteration and preamble import. May contain PreambleOp entries
    /// (RPython isinstance check → force_op_from_preamble → replace in-place).
    extra_call_pure: Vec<ExtraCallPureEntry>,
    /// optimizer.py: call_pure_results passed into propagate_all_forward.
    /// RPython keys are lists of constant boxes (value-based equality).
    /// Keys are the constant Values that _can_optimize_call_pure builds.
    call_pure_results: crate::optimizeopt::util::ArgsDict,
}

impl OptPure {
    pub fn new() -> Self {
        OptPure {
            cache: Rc::new(RefCell::new(RecentPureOpTable::new(
                crate::jit::PARAMETERS.pureop_historylength as usize,
            ))),
            postponed_op: None,
            call_pure_positions: Vec::new(),
            pending_call_pure_position: false,
            last_emitted_was_removed: false,
            known_result_call_pure: Vec::new(),
            extra_call_pure: Vec::new(),
            call_pure_results: crate::optimizeopt::util::ArgsDict::default(),
        }
    }

    /// pure.py: inject extra_call_pure from optimizer.call_pure_results.
    /// Called before optimization starts to seed cross-iteration CSE data.
    pub fn set_extra_call_pure(&mut self, results: Vec<(Vec<OpRef>, OpRef)>) {
        self.extra_call_pure = results
            .into_iter()
            .map(|(args, result)| {
                let key = PureOpKey {
                    opcode: OpCode::CallPureI,
                    args: smallvec::SmallVec::from_vec(args),
                    descr_identity: None,
                };
                ExtraCallPureEntry::Direct { key, result }
            })
            .collect();
    }

    /// Whether this opcode is commutative (order of args doesn't matter).
    fn is_commutative(opcode: OpCode) -> bool {
        matches!(
            opcode,
            OpCode::IntAdd
                | OpCode::IntAddOvf
                | OpCode::IntMul
                | OpCode::IntMulOvf
                | OpCode::IntAnd
                | OpCode::IntOr
                | OpCode::IntXor
        )
    }

    /// PyPy RecentPureOps stores AbstractResOp / Const boxes directly, so a
    /// reused result necessarily carries the same `box.type` as the query op.
    fn matches_result_type(op: &Op, result: &Operand, ctx: &OptContext) -> bool {
        ctx.resolve_operand_operand(result).type_() == op.result_type()
    }

    /// Try to find a cached result for this operation, considering commutativity.
    ///
    /// pure.py `RecentPureOps.lookup` dispatches to `lookup1` /
    /// `lookup2` so that arg comparison goes through `box.same_box`.
    /// Resolve forwarding before comparing producer identity or constant value.
    fn lookup_pure(&mut self, op: &Op, ctx: &mut OptContext) -> Option<(OpCode, Operand)> {
        // pure.py — `commutative` is forwarded into `lookup2`,
        // which checks both `(arg0, arg1)` and `(arg1, arg0)` orderings.
        // A PreambleOp hit is forced and the ring slot replaced, as
        // RecentPureOps.force_preamble_op does.
        let commutative = Self::is_commutative(op.opcode);
        let found = self.cache.borrow_mut().lookup_forcing(op, ctx, commutative);
        found.filter(|(_, result)| Self::matches_result_type(op, result, ctx))
    }

    /// Record a pure operation in the CSE cache.
    /// pure.py: pure(opnum, op)
    pub fn pure(&mut self, op: &Op, op_rc: &majit_ir::OpRc) {
        let key = PureOpKey::from_operand_op(op);
        self.cache
            .borrow_mut()
            .insert(key, Operand::from_bound_op(op_rc));
    }

    /// Point `ctx` at this pass's ring. A history-length reset replaces the
    /// `Rc`, so a stale handle is overwritten rather than aliased.
    fn publish_pure_ops(&self, ctx: &mut OptContext) {
        let same = ctx
            .pure_ops
            .as_ref()
            .is_some_and(|existing| Rc::ptr_eq(existing, &self.cache));
        if !same {
            ctx.pure_ops = Some(Rc::clone(&self.cache));
        }
    }

    /// Record a pure operation with explicit args.
    /// `pure_from_args(opnum, args, op, descr=None)`: each key arg is
    /// `get_box_replacement(arg)`. The cache still stores the forwarded
    /// box (`op` in the pure ring).
    pub fn pure_from_args(
        &mut self,
        opcode: OpCode,
        args: &[Operand],
        result: &Operand,
        ctx: &mut OptContext,
    ) {
        let key = PureOpKey {
            opcode,
            args: args
                .iter()
                .map(|arg| {
                    let replaced = arg.get_box_replacement(false);
                    #[cfg(debug_assertions)]
                    ctx.debug_assert_operand_matches_positional(
                        &replaced,
                        arg.to_opref(),
                        "pure_from_args",
                    );
                    replaced
                })
                .collect(),
            descr_identity: None,
        };
        self.publish_pure_ops(ctx);
        self.cache.borrow_mut().insert(key, result.clone());
    }

    /// `pure_from_args1(opnum, arg0, op)`.
    pub fn pure_from_args1(
        &mut self,
        opcode: OpCode,
        arg0: &Operand,
        result: &Operand,
        ctx: &mut OptContext,
    ) {
        self.pure_from_args(opcode, std::slice::from_ref(arg0), result, ctx);
    }

    /// `pure_from_args(opnum, [arg0], op, descr=...)`. `descr_identity`
    /// keeps distinct descrs from colliding on the same (opcode, arg0)
    /// pair (ARRAYLEN_GC across array descrs).
    pub fn pure_from_args1_with_descr(
        &mut self,
        opcode: OpCode,
        arg0: &Operand,
        result: &Operand,
        descr: majit_ir::DescrRef,
        ctx: &mut OptContext,
    ) {
        let replaced = arg0.get_box_replacement(false);
        #[cfg(debug_assertions)]
        ctx.debug_assert_operand_matches_positional(&replaced, arg0.to_opref(), "pure_from_args1");
        let key = PureOpKey {
            opcode,
            args: smallvec::smallvec![replaced],
            descr_identity: Some(majit_ir::descr::descr_identity(&descr)),
        };
        self.publish_pure_ops(ctx);
        self.cache.borrow_mut().insert(key, result.clone());
    }

    /// `pure_from_args2(opnum, arg0, arg1, op)`.
    pub fn pure_from_args2(
        &mut self,
        opcode: OpCode,
        arg0: &Operand,
        arg1: &Operand,
        result: &Operand,
        ctx: &mut OptContext,
    ) {
        self.pure_from_args(opcode, &[arg0.clone(), arg1.clone()], result, ctx);
    }

    /// Look up a previously recorded pure operation result.
    /// pure.py: get_pure_result(op)
    pub fn get_pure_result(&mut self, op: &Op, ctx: &mut OptContext) -> Option<Operand> {
        self.lookup_pure(op, ctx).map(|(_, result)| result)
    }

    #[cfg(test)]
    fn recent_ops_has_preamble(&self, op: &Op, ctx: &OptContext) -> bool {
        let commutative = Self::is_commutative(op.opcode);
        let cache = self.cache.borrow();
        let Some(bucket) = cache.bucket(op.opcode) else {
            return false;
        };
        let Some(index) = bucket.find_index(op, ctx, commutative) else {
            return false;
        };
        matches!(
            bucket.lst.get(index).and_then(|entry| entry.as_ref()),
            Some((_, PureRingValue::Preamble { .. }))
        )
    }

    /// pure.py lookup1(opt, box0, descr).
    ///
    /// `same_box(a, b)`: should apply get_box_replacement to `b` and then
    /// compare — identity for ops, value equality for constants.
    pub fn lookup1(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        descr_identity: Option<usize>,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        self.cache
            .borrow()
            .lookup1(opcode, arg0, descr_identity, same_box)
    }

    /// pure.py lookup2(opt, box0, box1, descr, commutative).
    pub fn lookup2(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        arg1: OpRef,
        descr_identity: Option<usize>,
        commutative: bool,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        self.cache
            .borrow()
            .lookup2(opcode, arg0, arg1, descr_identity, commutative, same_box)
    }

    /// Get the positions of emitted CALL_PURE ops (for short preamble generation).
    /// pure.py: call_pure_positions
    pub fn call_pure_positions(&self) -> &[usize] {
        &self.call_pure_positions
    }

    /// pure.py — check known_result_call_pure for a matching call.
    ///
    /// RPython iterates known_result_call_pure and compares:
    ///   `op.getdescr() is not known_result_op.getdescr()` → descr check
    ///   `self._same_args(known_result_op, op, 1, start_index)` → args check
    /// No opcode comparison.
    fn lookup_known_result(&self, op: &Op, start_index: usize, ctx: &OptContext) -> Option<OpRef> {
        let op_descr_identity = op.getdescr().as_ref().map(majit_ir::descr::descr_identity);
        for entry in &self.known_result_call_pure {
            if entry.descr_identity != op_descr_identity {
                continue;
            }
            // _same_args(known_op, op, 1, start_index):
            // entry.args is already known_op.args[1..], so compare from 0.
            let op_args: Vec<OpRef> = op.args_slice().iter().map(|a| a.to_opref()).collect();
            if Self::_same_args(&entry.args, &op_args, 0, start_index, ctx) {
                return Some(entry.result);
            }
        }
        None
    }

    fn demote_call_pure(&self, op: &Op) -> Op {
        let mut new_op = op.clone();
        new_op.opcode = OpCode::call_for_type(op.result_type());
        new_op
    }

    fn call_pure_can_raise(op: &Op) -> bool {
        op.with_call_descr(|cd| cd.get_extra_info().check_can_raise(true))
            .unwrap_or(true)
    }

    /// `pure.py` `opt.pure(opnum, PreambleOp(op, ...))` with `op`'s own
    /// argument boxes. `RecentPureOps.lookup2` applies
    /// `get_box_replacement` when the peeled loop looks the op up.
    fn pure_preamble_boxes(
        &mut self,
        opcode: OpCode,
        args: Vec<Operand>,
        descr_identity: Option<usize>,
        pop: PreambleOp,
    ) {
        let key = PureOpKey {
            opcode,
            args: args.into_iter().collect(),
            descr_identity,
        };
        self.cache.borrow_mut().insert_preamble(key, pop);
    }

    /// Store PreambleOp in OptPure for always-pure ops.
    /// shortpreamble.py `PureOp.produce_op`: `opt.pure(op.getopnum(), PreambleOp(...))`.
    /// Key args are each operand's `get_box_replacement(false)`.
    pub fn pure_preamble(
        &mut self,
        opcode: OpCode,
        args: Vec<Operand>,
        descr_identity: Option<usize>,
        pop: PreambleOp,
        ctx: &mut OptContext,
    ) {
        let key = PureOpKey {
            opcode,
            args: args
                .iter()
                .map(|arg| arg.get_box_replacement(false))
                .collect(),
            descr_identity,
        };
        self.publish_pure_ops(ctx);
        self.cache.borrow_mut().insert_preamble(key, pop);
    }

    /// Store PreambleOp in extra_call_pure for CALL_PURE preamble imports.
    /// RPython shortpreamble.py: optpure.extra_call_pure.append(PreambleOp(...))
    pub fn extra_call_pure_preamble(
        &mut self,
        opcode: OpCode,
        args: Vec<OpRef>,
        descr_identity: Option<usize>,
        pop: PreambleOp,
    ) {
        let key = PureOpKey {
            opcode,
            args: smallvec::SmallVec::from_vec(args),
            descr_identity,
        };
        self.extra_call_pure
            .push(ExtraCallPureEntry::Preamble { key, pop });
    }

    /// pure.py _can_reuse_oldop
    ///
    /// OVF guard pairing requires that an overflow-tracking op only reuse
    /// the result of another overflow-tracking op of the same opnum (since
    /// a regular `INT_ADD` may have overflowed). The non-OVF case is
    /// always safe.
    fn _can_reuse_oldop(oldop_opcode: OpCode, op_opcode: OpCode, ovf: bool) -> bool {
        if ovf {
            return oldop_opcode == op_opcode;
        }
        true
    }

    /// Hand a released `postponed_op` to the passes AFTER OptPure.
    ///
    /// pure.py releases the postponed OVF op as the
    /// `DefaultOptimizationResult.op`, so `send_extra_operation`
    /// (optimizer.py) carries it to `opt.next_optimization` —
    /// OptEarlyForce and then OptHeap — exactly like any other op OptPure
    /// passes on. Emitting it straight into `new_operations` would skip
    /// OptHeap, whose own `emit()` override (heap.py) postpones
    /// comparisons and flushes the pending one before the next op. Skipping
    /// it strands a preceding comparison in `OptHeap::postponed_op`, which
    /// then flushes BETWEEN the OVF op and its GUARD_NO_OVERFLOW — breaking
    /// the adjacency `aarch64/assembler.py _walk_operations`
    /// asserts and the backends rely on to read the overflow flags.
    fn emit_postponed_downstream(&mut self, postponed: OpRc, ctx: &mut OptContext) {
        ctx.emit_extra_rc(ctx.current_pass_idx, postponed);
    }

    /// pure.py _same_args
    ///
    /// Compare two argument lists with optional skip-prefixes on each side
    /// (used by COND_CALL_VALUE so its leading condition slot is not
    /// matched against a CALL_PURE's first real argument).
    fn _same_args(
        op1_args: &[OpRef],
        op2_args: &[OpRef],
        start_index1: usize,
        start_index2: usize,
        ctx: &OptContext,
    ) -> bool {
        if op1_args.len() - start_index1 != op2_args.len() - start_index2 {
            return false;
        }
        for (j, i) in (start_index2..).zip(start_index1..op1_args.len()) {
            // pure.py — same_box(op1_args[i], op2_args[j])
            // applies `get_box_replacement` to both sides, then dispatches
            // identity vs Const value equality (`history.py`).
            if !ctx.same_box(op1_args[i], op2_args[j]) {
                return false;
            }
        }
        true
    }

    /// pure.py optimize_call_pure_old
    ///
    /// Try to fuse `op` with a previously emitted `old_op` (either an
    /// inline call_pure recorded in `call_pure_positions` or an entry from
    /// `extra_call_pure`). Returns true when a match is found and the
    /// caller should mark the op REMOVED.
    fn optimize_call_pure_old(
        op: &Op,
        old_op_opcode: OpCode,
        old_op_args: &[OpRef],
        old_op_descr_identity: Option<usize>,
        op_descr_identity: Option<usize>,
        start_index: usize,
        ctx: &OptContext,
    ) -> bool {
        // pure.py: descr identity check.
        if op_descr_identity != old_op_descr_identity {
            return false;
        }
        // RPython relies on each CALL_PURE having a unique descriptor that
        // already encodes the return type, so a separate result-type check
        // is unnecessary upstream. majit allows tests with `descr = None`,
        // so we keep the implicit invariant explicit here: never CSE across
        // different return types.
        if op.opcode.result_type() != old_op_opcode.result_type() {
            return false;
        }
        // pure.py: old_start_index = OpHelpers.is_cond_call_value(old_op.opnum)
        let old_start_index = if old_op_opcode.is_cond_call_value() {
            1
        } else {
            0
        };
        // pure.py: self._same_args(old_op, op, old_start_index, start_index)
        let op_args: Vec<OpRef> = op.args_slice().iter().map(|a| a.to_opref()).collect();
        Self::_same_args(old_op_args, &op_args, old_start_index, start_index, ctx)
    }
}

impl Default for OptPure {
    fn default() -> Self {
        Self::new()
    }
}

impl OptPure {
    fn force_box(&mut self, op: &Operand, ctx: &mut OptContext) -> Operand {
        // optimizer.py force_box: `op = get_box_replacement(op)`, then force
        // the virtual behind it.
        match ctx.resolve_operand_operand_opt(op) {
            Some(resolved) if ctx.is_virtual(&resolved) => {
                let mut info = ctx.take_ptr_info(&resolved).unwrap();
                let forced = info.force_box(&resolved, ctx);
                ctx.resolve_operand_operand(&forced)
            }
            Some(resolved) => resolved,
            None => op.clone(),
        }
    }

    /// optimizer.py _can_optimize_call_pure.
    ///
    /// RPython: for each arg, `get_constant_box(arg)` returns the constant
    /// value (ConstInt/ConstPtr/ConstFloat), then uses those values as the
    /// lookup key in call_pure_results. Value-based equality, not identity.
    fn lookup_call_pure_result(
        &mut self,
        op: &Op,
        start_index: usize,
        ctx: &mut OptContext,
    ) -> Option<Value> {
        let mut arg_consts = Vec::with_capacity(op.num_args().saturating_sub(start_index));
        for i in start_index..op.num_args() {
            let forced = self.force_box(&op.arg(i), ctx);
            let const_value = ctx.get_constant_box(&forced)?;
            arg_consts.push(const_value);
        }
        self.call_pure_results.get(&arg_consts)
    }
}

impl Optimization for OptPure {
    fn set_pureop_historylength(&mut self, limit: usize) {
        self.cache = Rc::new(RefCell::new(RecentPureOpTable::new(limit)));
    }

    fn propagate_forward(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        self.publish_pure_ops(ctx);
        // optimizer.py: pure_from_args1 parity — consume pending registrations
        // from rewrite pass (CAST_*, CONVERT_* reverse-pure relationships)
        // and virtualize pass (ARRAYLEN_GC with array descr keying per
        // virtualize.py:220).
        for index in 0..ctx.pending_pure_from_args.len() {
            let (opcode, arg0, result, descr) = ctx.pending_pure_from_args[index].clone();
            match descr {
                Some(d) => self.pure_from_args1_with_descr(opcode, &arg0, &result, d, ctx),
                None => self.pure_from_args1(opcode, &arg0, &result, ctx),
            }
        }
        ctx.pending_pure_from_args.clear();
        // INSTANCE_PTR_EQ/NE swapped-args reverse relations.
        for index in 0..ctx.pending_pure_from_args2.len() {
            let (opcode, arg0, arg1, result) = ctx.pending_pure_from_args2[index].clone();
            self.pure_from_args2(opcode, &arg0, &arg1, &result, ctx);
        }
        ctx.pending_pure_from_args2.clear();

        // Don't reset for GUARD_NO_EXCEPTION — it needs the previous state.
        if op.opcode != OpCode::GuardNoException {
            self.last_emitted_was_removed = false;
        }

        // Clear any position arm the previous op left un-consumed, so the
        // one-shot cannot leak onto this op.
        //
        // The arm is armed in `optimize_call_pure` and consumed in
        // `propagate_postprocess`. Every ordinary exit now consumes it: the op
        // is collected under its demoted `CallI`/`CallR` opcode, and emission,
        // a later pass's removal and a restart all run the collected callbacks
        // (`send_extra_operation`, optimizer.py, ends the walk on a
        // removal but not the callbacks). What is left un-consumed is the
        // error exit — a pass reporting `InvalidLoop`, which abandons the
        // trace — so this reset is defensive rather than load-bearing.
        //
        // Consuming it on a removal is what upstream does too: the callback
        // rides on the result object `emit_result` built (pure.py), and
        // that object stays in `opt_results` whatever a later pass decides. It
        // then appends `len(_newoperations) - 1`, which for a removed call
        // names a different op — an exposure both implementations carry.
        // Unreachable today: the only removal of a call below this pass is
        // OptHeap's DICT_LOOKUP dedup, and a dict lookup is registered as a
        // plain residual call, so it never arrives here demoted from a pure one.
        self.pending_call_pure_position = false;

        // pure.py: OVF operation postponement.
        // INT_ADD_OVF, INT_SUB_OVF, INT_MUL_OVF are deferred until we see
        // GUARD_NO_OVERFLOW, so we can try CSE on the OVF op + guard pair.
        if op.opcode.is_ovf() {
            self.postponed_op = Some(op_rc.clone());
            return OptimizationResult::Remove;
        }

        // Handle the postponed OVF op when we see GUARD_NO_OVERFLOW.
        if let Some(postponed) = self.postponed_op.take() {
            let postponed_box = Operand::from_bound_op(&postponed);
            if op.opcode == OpCode::GuardNoOverflow {
                // pure.py — only call `constant_fold` when every
                // arg has resolved to a `Const*` via `get_constant_box`
                // (the `for...else:` gate).  Without this pre-check,
                // `constant_fold` would panic on the now-strict
                // `expect("arg must be Const")` since postponed OVF
                // ops can have non-const args (e.g. `IntMulOvf(p, 1)`
                // where p is an inputarg).
                let all_args_const = (0..postponed.num_args()).all(|i| {
                    ctx.get_constant_box(&postponed.arg(i).get_box_replacement(false))
                        .is_some()
                });
                if all_args_const && let Some(Value::Int(folded)) = ctx.constant_fold(&postponed) {
                    ctx.make_constant_box(&postponed_box, Value::Int(folded));
                    self.last_emitted_was_removed = true;
                    return OptimizationResult::Remove; // guard also removed
                }

                // pure.py + 162-171 _can_reuse_oldop:
                // The lookup may surface a non-OVF op of the same shape
                // (e.g. INT_ADD vs INT_ADD_OVF). _can_reuse_oldop accepts
                // it only when the cached opnum matches our OVF opnum.
                if let Some((old_opcode, cached_ref)) = self.lookup_pure(&postponed, ctx)
                    && Self::_can_reuse_oldop(old_opcode, postponed.opcode, true)
                {
                    let b_old = postponed_box.clone();
                    let b_cached = ctx.resolve_operand_operand(&cached_ref);
                    ctx.make_equal_to(&b_old, &b_cached);
                    self.last_emitted_was_removed = true;
                    return OptimizationResult::Remove; // guard also removed
                }

                // Resolve the args to their forwarded terminals before the op
                // is recorded in `self.cache` under its arg identities
                // (optimizer.py force_box loop).
                let key = PureOpKey::from_operand_op(&postponed);
                for i in 0..postponed.num_args() {
                    let forced = self.force_box(&postponed.arg(i), ctx);
                    postponed.setarg(i, forced);
                }
                // Record and emit both the OVF op and the guard.
                self.cache.borrow_mut().insert(key, postponed_box.clone());
                // pure.py:321-322 walks `_newoperations` for is_ovf followed
                // by GUARD_NO_OVERFLOW; do not keep a parallel candidate list.
                self.emit_postponed_downstream(postponed, ctx);
                return OptimizationResult::PassOn; // guard passes through
            } else {
                // Not a GUARD_NO_OVERFLOW: emit the postponed op now.
                for i in 0..postponed.num_args() {
                    let forced = self.force_box(&postponed.arg(i), ctx);
                    postponed.setarg(i, forced);
                }
                self.emit_postponed_downstream(postponed, ctx);
            }
        }

        // pure.py: GUARD_NO_EXCEPTION — remove if last emitted was removed
        // (CALL_PURE was constant-folded or CSE'd away).
        if op.opcode == OpCode::GuardNoException {
            if self.last_emitted_was_removed {
                return OptimizationResult::Remove;
            }
            return OptimizationResult::PassOn;
        }

        // pure.py: RECORD_KNOWN_RESULT — record for later CALL_PURE lookup.
        // pure.py: `self.known_result_call_pure.append(op)`
        // Lookup compares descr + _same_args(known_op, query_op, 1, start_index).
        if op.opcode == OpCode::RecordKnownResult {
            if op.num_args() >= 2 {
                self.known_result_call_pure.push(KnownResultEntry {
                    descr_identity: op.getdescr().as_ref().map(majit_ir::descr::descr_identity),
                    args: op.args_slice()[1..].iter().map(|a| a.to_opref()).collect(),
                    result: op.arg(0).to_opref(),
                });
            }
            return OptimizationResult::Remove;
        }

        if op.opcode.is_always_pure() {
            // pure.py:
            //     for i in range(op.numargs()):
            //         if self.get_constant_box(op.getarg(i)) is None:
            //             break
            //     else:
            //         # all constant arguments: constant-fold away
            //         resbox = self.optimizer.constant_fold(op)
            //         self.optimizer.make_constant(op, resbox)
            //         return
            let all_args_const = (0..op.num_args()).all(|i| {
                ctx.get_constant_box(&op.arg(i).get_box_replacement(false))
                    .is_some()
            });
            if all_args_const {
                // Upstream `pure.py for...else:` calls
                // `optimizer.constant_fold(op)` unconditionally and
                // feeds the result straight into `make_constant`.
                // Pyre's `constant_fold` now narrows to two
                // documented `None` paths only:
                //   1. `protect_speculative_operation`'s
                //      `supports_guard_gc_type == false` gate on
                //      memory-reading folds (mod.rs).  Upstream's
                //      comment at `optimizer.py` skips
                //      unrolling in that mode; pyre's
                //      `constant_fold` runs outside unroll too, so
                //      the fold itself must decline.
                //   2. Helper-internal narrow skips on `IntAddOvf` /
                //      `IntSubOvf` / `IntMulOvf` overflow, shift
                //      count outside `0..64`, `IntFloorDiv` /
                //      `IntMod` divide-by-zero, and CAST_FLOAT_TO_INT
                //      non-finite (`execute_binary_int_const` /
                //      `execute_cast_const` return `Ok(None)`).
                //      Upstream `do_int_add_ovf` would `assert
                //      metainterp is not None` on overflow with
                //      `metainterp=None` (executor.py:287); pyre
                //      prefers the softer skip so the op stays in
                //      the trace and the runtime guard fires.
                // Every caller-invariant violation (missing box,
                // descr, wrong Value variant) panics inside
                // `constant_fold` / `protect_speculative_operation`,
                // matching upstream's `AttributeError`.  A genuine
                // speculative-fold failure (ill-typed heap access) is
                // recorded as a deferred `InvalidLoop` signal on the
                // context; `constant_fold` then returns `None` and the
                // driver aborts the trace at its next barrier per
                // `unroll.py:119-123`.
                if let Some(folded_value) = ctx.constant_fold(op) {
                    ctx.make_constant_box(&Operand::from_bound_op(op_rc), folded_value);
                    self.last_emitted_was_removed = true;
                    return OptimizationResult::Remove;
                }
            }

            // CSE: exact same operation already computed?
            if let Some((_, cached_ref)) = self.lookup_pure(op, ctx) {
                let b_old = Operand::from_bound_op(op_rc);
                let b_cached = ctx.resolve_operand_operand(&cached_ref);
                ctx.make_equal_to(&b_old, &b_cached);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            return OptimizationResult::PassOn;
        }

        // pure.py optimize_call_pure(op, start_index)
        // pure.py: optimize_CALL_PURE_I/R/F/N → start_index=0
        // pure.py: optimize_COND_CALL_VALUE_I/R → start_index=1
        if op.opcode.is_call_pure() || op.opcode.is_cond_call_value() {
            let start_index: usize = if op.opcode.is_cond_call_value() { 1 } else { 0 };
            let op_descr_identity = op.getdescr().as_ref().map(majit_ir::descr::descr_identity);

            // pure.py: _can_optimize_call_pure(op, start_index=1).
            if let Some(value) = self.lookup_call_pure_result(op, start_index, ctx) {
                let b = Operand::from_bound_op(op_rc);
                ctx.make_constant_box(&b, value);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            // pure.py: iterate call_pure_positions, try
            // optimize_call_pure_old with adjusted start_index.
            for &pos in &self.call_pure_positions {
                if let Some(old_op) = ctx.new_operations.get(pos) {
                    let old_descr_identity = old_op
                        .getdescr()
                        .as_ref()
                        .map(majit_ir::descr::descr_identity);
                    let old_op_args: Vec<OpRef> =
                        old_op.args_slice().iter().map(|a| a.to_opref()).collect();
                    if Self::optimize_call_pure_old(
                        op,
                        old_op.opcode,
                        &old_op_args,
                        old_descr_identity,
                        op_descr_identity,
                        start_index,
                        ctx,
                    ) {
                        let cached_src = old_op.pos().get();
                        let b_old = Operand::from_bound_op(op_rc);
                        let b_cached = ctx.get_box_replacement_operand(cached_src);
                        ctx.make_equal_to(&b_old, &b_cached);
                        self.last_emitted_was_removed = true;
                        return OptimizationResult::Remove;
                    }
                }
            }
            // pure.py: iterate extra_call_pure entries.
            //   if isinstance(old_op, PreambleOp):
            //       old_op = self.optimizer.force_op_from_preamble(old_op)
            //       self.extra_call_pure[i] = old_op
            // Force runs after the match so use_box / potential_extra_ops /
            // forwarded-info side effects are tied to the actual fuse, and
            // the entry is rewritten so subsequent matches reuse the forced
            // result without re-forcing.
            let mut matched: Option<(usize, Option<PreambleOp>)> = None;
            for (i, entry) in self.extra_call_pure.iter().enumerate() {
                let (entry_opcode, entry_args, entry_descr_identity, pop) = match entry {
                    ExtraCallPureEntry::Direct { key, .. } => {
                        (key.opcode, &key.args, key.descr_identity, None)
                    }
                    ExtraCallPureEntry::Preamble { key, pop } => {
                        (key.opcode, &key.args, key.descr_identity, Some(pop.clone()))
                    }
                };
                if Self::optimize_call_pure_old(
                    op,
                    entry_opcode,
                    entry_args,
                    entry_descr_identity,
                    op_descr_identity,
                    start_index,
                    ctx,
                ) {
                    matched = Some((i, pop));
                    break;
                }
            }
            if let Some((i, pop)) = matched {
                let entry_result = if let Some(pop) = pop {
                    let forced = ctx.force_op_from_preamble_op(&pop).to_opref();
                    let key = match &self.extra_call_pure[i] {
                        ExtraCallPureEntry::Preamble { key, .. } => key.clone(),
                        _ => unreachable!("matched index must still hold the Preamble entry"),
                    };
                    self.extra_call_pure[i] = ExtraCallPureEntry::Direct {
                        key,
                        result: forced,
                    };
                    forced
                } else {
                    match &self.extra_call_pure[i] {
                        ExtraCallPureEntry::Direct { result, .. } => *result,
                        _ => unreachable!("non-preamble matched index must be Direct"),
                    }
                };
                let b_old = Operand::from_bound_op(op_rc);
                let b_cached = ctx.get_box_replacement_operand(entry_result);
                ctx.make_equal_to(&b_old, &b_cached);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }
            // pure.py: known_result_call_pure.
            if let Some(result_ref) = self.lookup_known_result(op, start_index, ctx) {
                let b_old = Operand::from_bound_op(op_rc);
                let b_result = ctx.get_box_replacement_operand(result_ref);
                ctx.make_equal_to(&b_old, &b_result);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            // pure.py: `CallPureOptimizationResult.callback()` appends
            // `len(_newoperations) - 1` AFTER `emit_result` has routed the op
            // through the remaining optimizations. Recording it here (before
            // the `Replace` routes through earlyforce/heap) is wrong: OptHeap
            // flushes a postponed comparison op ahead of this call, so the
            // pre-emit `len` would index that flushed op, not the call. Arm
            // the one-shot instead and record the true index in
            // `propagate_postprocess`, mirroring the callback's post-emit fire.
            self.pending_call_pure_position = true;
            // pure.py route the emitted op through `self.emit`, i.e.
            // the passes downstream of OptPure (earlyforce, heap). Returning
            // `Replace` continues the op through those remaining passes, so
            // OptHeap.emit() still runs — flushing any postponed comparison op
            // ahead of this call. `Emit` would append the op terminally and
            // skip OptHeap, stranding a postponed comparison whose result this
            // call consumes past its own definition (regalloc def-after-use).
            if start_index == 0 {
                // pure.py: replace CALL_PURE with CALL.
                let new_op = self.demote_call_pure(op);
                return OptimizationResult::Replace(new_op);
            } else {
                // pure.py: COND_CALL_VALUE is NOT demoted.
                return OptimizationResult::Replace(op.clone());
            }
        }

        OptimizationResult::PassOn
    }

    fn setup(&mut self) {
        self.cache.borrow_mut().clear();
        self.postponed_op = None;
        self.call_pure_positions.clear();
        self.pending_call_pure_position = false;
        self.last_emitted_was_removed = false;
        self.known_result_call_pure.clear();
        // Note: extra_call_pure is NOT cleared on setup — it persists
        // across optimization runs (set by set_extra_call_pure before opt).
    }

    fn reset_between_compiles(&mut self) {
        self.setup();
        self.extra_call_pure.clear();
    }

    fn set_call_pure_results(&mut self, results: &crate::optimizeopt::util::ArgsDict) {
        self.call_pure_results = results.clone();
    }

    fn name(&self) -> &'static str {
        "pure"
    }

    /// pure.py `CallPureOptimizationResult` — the demoted CALL / un-demoted
    /// COND_CALL_VALUE carry a post-emit callback that records the call's
    /// position. `DefaultOptimizationResult._callback` also publishes boolean
    /// bounds even when OptHeap postpones the operation's final emission.
    fn have_postprocess_op(&self, opcode: OpCode) -> bool {
        opcode.is_real_call()
            || opcode.is_cond_call_value()
            || opcode.returns_bool()
            || opcode.is_always_pure()
    }

    /// pure.py `CallPureOptimizationResult.callback`:
    ///
    /// ```text
    /// self.opt.call_pure_positions.append(
    ///     len(self.opt.optimizer._newoperations) - 1)
    /// ```
    ///
    /// The callback fires after `emit_result` has routed the op through the
    /// remaining optimizations (earlyforce, heap) and it has landed in
    /// `_newoperations`, so `len - 1` is the call's final index — past any
    /// postponed op OptHeap flushed ahead of it. The one-shot armed in
    /// `optimize_call_pure` selects exactly the op this callback belongs to.
    fn propagate_postprocess(&mut self, op: &Op, op_rc: &majit_ir::OpRc, ctx: &mut OptContext) {
        self.publish_pure_ops(ctx);
        // pure.py DefaultOptimizationResult._callback: consumers need this
        // fact before a postponed comparison reaches Optimizer.emit.
        if op.opcode.returns_bool() {
            let result = Operand::from_bound_op(op_rc);
            ctx.with_intbound_mut(&result, |bound| bound.make_bool());
        }
        // pure.py DefaultOptimizationResult._callback: `recentops.add(op)`
        // records the optimized op itself. Looking it up in `new_operations`
        // missed when emit reminted the result pos, so the ring stayed empty
        // and identical IntEq never CSEd.
        if op.opcode.is_always_pure() {
            self.pure(op, op_rc);
        }
        if self.pending_call_pure_position
            && (op.opcode.is_real_call() || op.opcode.is_cond_call_value())
        {
            self.pending_call_pure_position = false;
            self.call_pure_positions
                .push(ctx.new_operations.len().saturating_sub(1));
        }
    }

    /// shortpreamble.py: PureOp.produce_op stores PreambleOp in
    /// optpure. In RPython, produce_op accesses opt.optimizer.optpure directly.
    /// In majit, import_short_preamble_ops stores in ctx.imported_short_pure_ops,
    /// then this method transfers them into OptPure's preamble caches.
    fn install_preamble_pure_ops(&mut self, ctx: &mut OptContext) {
        self.publish_pure_ops(ctx);
        let imported = ctx.imported_short_pure_ops.clone();
        for entry in &imported {
            let descr_identity = entry.descr.as_ref().map(majit_ir::descr::descr_identity);
            let pop = entry.pop.clone();
            if entry.opcode.is_call_pure() || entry.opcode.is_call() {
                // shortpreamble.py `PureOp.produce_op`: extra_call_pure keeps
                // positional OpRefs resolved from the exporting-phase boxes.
                let resolved_args: Vec<OpRef> = entry
                    .args
                    .iter()
                    .map(|arg| ctx.resolve_operand_operand(arg).to_opref())
                    .collect();
                self.extra_call_pure_preamble(entry.opcode, resolved_args, descr_identity, pop);
            } else if !entry.cache_args.is_empty() {
                // pure.py RecentPureOps.lookup2 compares
                // get_box_replacement of the cached args at lookup time.
                // Keep the preamble boxes; rematerializing them here drops
                // the guard_value forwarding.
                self.pure_preamble_boxes(
                    entry.opcode,
                    entry.cache_args.clone(),
                    descr_identity,
                    pop,
                );
            } else {
                // shortpreamble.py `PureOp.produce_op`: opt.pure(opnum, PreambleOp)
                self.pure_preamble(entry.opcode, entry.args.clone(), descr_identity, pop, ctx);
            }
        }
    }

    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        ctx: &mut OptContext,
    ) {
        // pure.py produce_potential_short_preamble_ops: walk
        // optimizer._newoperations, then call_pure_positions.
        let n = ctx.new_operations.len();
        for i in 0..n {
            let opcode = ctx.new_operations[i].opcode;
            let hoist = opcode.is_always_pure()
                || (opcode.is_ovf()
                    && ctx
                        .new_operations
                        .get(i + 1)
                        .is_some_and(|next| next.opcode == OpCode::GuardNoOverflow));
            if hoist {
                let op = ctx.new_operations[i].clone();
                sb.add_pure_op(ctx, &op);
            }
        }
        for &i in &self.call_pure_positions {
            let Some(op_rc) = ctx.new_operations.get(i).cloned() else {
                continue;
            };
            // don't move call_pure_with_exception in the short preamble...
            // issue #2015
            //
            // Also, don't move cond_call_value in the short preamble.
            // The issue there is that it's usually pointless to try to
            // because the 'value' argument is typically not a loop
            // invariant, and would really need to be in order to end up
            // in the short preamble.  Maybe the code works anyway in the
            // other rare case, but better safe than sorry and don't try.
            if !Self::call_pure_can_raise(op_rc.as_ref()) {
                debug_assert!(op_rc.opcode.is_call());
                if !op_rc.opcode.is_cond_call_value() {
                    sb.add_pure_op(ctx, &op_rc);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initialize_imported_short_pure_builder(
        ctx: &mut OptContext,
        preamble_op: Op,
        label_arg_idx: Option<usize>,
    ) {
        let source = preamble_op.pos().get();
        let short_inputargs: Vec<OpRef> = match label_arg_idx {
            Some(idx) => (0..=idx as u32).map(OpRef::int_op).collect(),
            None => vec![OpRef::int_op(0)],
        };
        let res = ctx.materialize_operand_at(source);
        ctx.initialize_imported_short_preamble_builder(
            &short_inputargs,
            &short_inputargs,
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: OpRc::new(preamble_op),
                source_op: None,
                res,
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx,
                invented_name: false,
                same_as_source: None,
            }],
        );
        // Keep the source result available to use_box() exactly like the
        // imported short preamble path does after unroll import.
        if source != OpRef::NONE {
            // PreambleOp.op carries the Box itself (shortpreamble.py).
            let source_box = ctx.materialize_operand_at(source);
            // Production threads the builder's replay Rc into the pop
            // (produce_op family); mirror that here so use_box receives
            // the same object the builder entry carries. The builder keys by
            // its entry res (`materialize_operand_at(pos)`); `source_box` is the
            // memoized box for the same position, so the lookup hits.
            let replay = ctx
                .imported_short_preamble_builder
                .as_ref()
                .and_then(|b| b.produced_short_op(&source_box.clone()))
                .map(|p| p.preamble_op)
                .unwrap_or_else(|| {
                    let mut same_as = Op::new(OpCode::SameAsI, std::slice::from_ref(&source_box));
                    same_as.pos().set(source);
                    OpRc::new(same_as)
                });
            let pop = crate::optimizeopt::info::PreambleOp {
                op: source_box.clone(),
                invented_name: false,
                preamble_op: replay,
                // Non-invented imported pure re-export: no SameAs alias.
                same_as_source: None,
            };
            ctx.set_potential_extra_op(source_box, pop);
        }
    }
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::OpRc;
    use majit_ir::Type;
    use majit_ir::descr::make_field_descr_full;
    /// Helper: assign sequential positions to ops.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            // Type-tag op.pos via the result-type-aware factory so the
            // OpRef variant carries `Box.type` (history.py:220 /
            // resoperation.py:1693 parity). Argument OpRefs in these
            // fixtures must use the matching typed factory at the same
            // raw N to satisfy variant-aware Eq against `op.pos`.
            op.pos().set(OpRef::op_typed(i as u32, op.result_type()));
        }
    }

    /// One argument of an [`OpSpec`] in the oparser-faithful bound DAG
    /// (`rpython/jit/tool/oparser.py`). Each variant resolves to a *bound*
    /// `Operand` at [`build_trace`] time so the arg sheds to
    /// `Operand::{InputArg,Op,Const}` — never the position-only
    /// `Operand::Box`.
    #[derive(Clone)]
    enum Arg {
        /// Header input var (oparser `[i0]`): a rooted `InputArg`-bound
        /// operand, sheds to `Operand::InputArg`.
        In(u32),
        /// Earlier producer's result, referenced by the producing op's
        /// index in the spec slice (`from_bound_op`, sheds to `Operand::Op`).
        Prod(usize),
        /// Const literal arg (`const_from_value`, sheds to `Operand::Const`).
        Const(Value),
    }

    /// A producer op (oparser `iN = opcode(args)`): its args name earlier
    /// producers / header inputs / literals, never a position-only box.
    #[derive(Clone)]
    struct OpSpec {
        opcode: OpCode,
        args: Vec<Arg>,
        descr: Option<majit_ir::DescrRef>,
    }

    fn op_spec(opcode: OpCode, args: &[Arg]) -> OpSpec {
        OpSpec {
            opcode,
            args: args.to_vec(),
            descr: None,
        }
    }

    fn op_spec_descr(opcode: OpCode, args: &[Arg], descr: majit_ir::DescrRef) -> OpSpec {
        OpSpec {
            opcode,
            args: args.to_vec(),
            descr: Some(descr),
        }
    }

    /// Turn a producer-spec slice into a bound `OpRc` graph plus the header
    /// input types, oparser-faithful (`self.vars[name] = resop;
    /// args.append(self.vars[arg])`): each consumer arg references the
    /// producing op's bound result box (`from_bound_op`) or a rooted
    /// `InputArg` box (`Arg::In`) or a `Const` literal, so every arg sheds to
    /// `Operand::Op` / `Operand::InputArg` / `Operand::Const` at construction.
    /// Inputs are positioned `[0, num_inputs)`; producers get sequential
    /// positions starting at `num_inputs`. Drive the result through
    /// `optimize_with_constants_and_inputs_oprc` so these OpRcs are the
    /// canonical producers `find_producer_op` resolves to.
    fn build_trace(num_inputs: u32, specs: &[OpSpec]) -> (Vec<OpRc>, Vec<Type>) {
        let mut input_types = vec![Type::Int; num_inputs as usize];
        let mut ops: Vec<OpRc> = Vec::new();
        // Lazily bind a header input box, recording its type from first use.
        let mut input_boxes: Vec<Option<Operand>> = vec![None; num_inputs as usize];
        for (i, spec) in specs.iter().enumerate() {
            let arg_ops: Vec<Operand> = spec
                .args
                .iter()
                .map(|a| match a {
                    Arg::In(idx) => {
                        let slot = *idx as usize;
                        input_boxes[slot]
                            .get_or_insert_with(|| {
                                crate::history::test_support::rooted_inputarg_operand(
                                    Type::Int,
                                    *idx,
                                )
                            })
                            .clone()
                    }
                    Arg::Prod(pos) => Operand::from_bound_op(&ops[*pos]),
                    Arg::Const(v) => Operand::const_from_value(*v),
                })
                .collect();
            let op = match &spec.descr {
                Some(d) => OpRc::new(Op::with_descr(spec.opcode, &arg_ops, d.clone())),
                None => OpRc::new(Op::new(spec.opcode, &arg_ops)),
            };
            op.pos().set(OpRef::op_typed(
                num_inputs + i as u32,
                spec.opcode.result_type(),
            ));
            ops.push(op);
        }
        let _ = &mut input_types;
        (ops, input_types)
    }

    /// Build a bound trace from `specs` and run `OptPure` (plus any
    /// `extra_passes`, applied first) over it via the OpRc entry. Mirrors the
    /// raw `Optimizer::new(); add_pass(OptPure); optimize_with_constants_and
    /// _inputs` shape the position-only fixtures used, but with no
    /// `Operand::Box` minted.
    fn run_pure(
        num_inputs: u32,
        specs: &[OpSpec],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        extra_passes: &[fn() -> Box<dyn crate::optimizeopt::Optimization>],
        seed_guard_snapshots: bool,
    ) -> Vec<Op> {
        let (producers, inputs) = build_trace(num_inputs, specs);
        let mut opt = Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        for make in extra_passes {
            opt.add_pass(make());
        }
        opt.add_pass(Box::new(OptPure::new()));
        // `producers` is held to the end of the call so every `from_bound_op`
        // arg box's `Weak<Op>` upgrade stays live (forwarding.rs). The
        // guard-snapshot path wraps the seeded `Vec<Op>` in fresh Rcs whose
        // args still point at `producers`, so those originals must outlive the
        // run.
        let run_ops: Vec<OpRc> = if seed_guard_snapshots {
            let flat: Vec<Op> = producers.iter().map(|o| (**o).clone()).collect();
            let (seeded, snapshots) = super::super::seed_empty_guard_snapshots(&flat);
            opt.snapshot_boxes = snapshots;
            seeded.into_iter().map(OpRc::new).collect()
        } else {
            producers.clone()
        };
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&run_ops, constants, num_inputs as usize)
            .expect("test: unexpected InvalidLoop")
            .into_iter()
            .map(|rc| (*rc).clone())
            .collect();
        drop(producers);
        result
    }

    #[test]
    fn two_int_eq_of_the_same_input_and_const_cse() {
        let result = run_pure(
            1,
            &[
                op_spec(OpCode::IntEq, &[Arg::In(0), Arg::Const(Value::Int(97))]),
                op_spec(OpCode::IntEq, &[Arg::In(0), Arg::Const(Value::Int(97))]),
                op_spec(OpCode::IntEq, &[Arg::In(0), Arg::Const(Value::Int(98))]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );
        let eqs: Vec<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::IntEq)
            .collect();
        assert_eq!(
            eqs.len(),
            2,
            "IntEq(i0, 97) must CSE; IntEq(i0, 98) stays. got {result:?}"
        );
    }

    #[test]
    fn pure_history_keeps_the_configured_number_of_distinct_emitted_ops() {
        let limit = crate::jit::PARAMETERS.pureop_historylength as u32;
        let mut specs: Vec<_> = (0..limit)
            .map(|i| op_spec(OpCode::IntNeg, &[Arg::In(i)]))
            .collect();
        specs.push(op_spec(OpCode::IntNeg, &[Arg::In(0)]));
        let result = run_pure(
            limit,
            &specs,
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );
        assert_eq!(result.len(), limit as usize);
    }

    #[test]
    fn downstream_remove_does_not_unrecord_a_pure_op() {
        struct RemoveFirst(bool);
        impl Optimization for RemoveFirst {
            fn name(&self) -> &'static str {
                "remove-first"
            }
            fn propagate_forward(
                &mut self,
                _: &Op,
                _: &OpRc,
                _: &mut OptContext,
            ) -> OptimizationResult {
                if std::mem::replace(&mut self.0, true) {
                    OptimizationResult::PassOn
                } else {
                    OptimizationResult::Remove
                }
            }
        }
        let (ops, types) = build_trace(
            1,
            &[
                op_spec(OpCode::IntNeg, &[Arg::In(0)]),
                op_spec(OpCode::IntNeg, &[Arg::In(0)]),
            ],
        );
        let mut optimizer = Optimizer::new();
        optimizer.add_pass(Box::new(OptPure::new()));
        optimizer.add_pass(Box::new(RemoveFirst(false)));
        let result = optimizer
            .optimize_with_constants_and_inputs_oprc(
                &ops,
                &mut majit_ir::ConstMap::default(),
                types.len(),
            )
            .unwrap();
        // pure.py `recentops.add(op)` records the optimized op even when a
        // later pyre pass Removes it (RPython emits from OptPure before later
        // passes run). The second IntNeg CSEs away, so nothing is emitted.
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn ring_lookup_rejects_a_forced_preamble_result_in_another_bank() {
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Int, Type::Ref]);
        let arg = ctx.materialize_operand_at(OpRef::input_arg_typed(0, Type::Int));
        let wrong = ctx.materialize_operand_at(OpRef::input_arg_typed(1, Type::Ref));
        let query = OpRc::new(Op::new(OpCode::IntNeg, &[arg]));
        let pop = PreambleOp {
            op: wrong.clone(),
            invented_name: false,
            preamble_op: query.clone(),
            same_as_source: None,
        };
        let mut pass = OptPure::new();
        pass.cache
            .borrow_mut()
            .insert_preamble(PureOpKey::from_operand_op(&query), pop);
        {
            let mut cache = pass.cache.borrow_mut();
            let bucket = cache.buckets.iter_mut().flatten().next().unwrap();
            if let Some((_, PureRingValue::Preamble { forced, .. })) = &mut bucket.lst[0] {
                *forced = Some(wrong.clone());
            }
        }
        assert_eq!(pass.lookup_pure(&query, &mut ctx), None);
    }

    #[test]
    fn test_cse_int_add() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i0, i1)  <- should be eliminated, replaced by i2
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        // Only the first IntAdd should remain.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_cse_different_args_not_eliminated() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i0, i2)  <- different args, should NOT be eliminated
        let result = run_pure(
            3,
            &[
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(2)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_commutative() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i1, i0)  <- commutative, should be eliminated
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntAdd, &[Arg::In(1), Arg::In(0)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cse_non_commutative() {
        // i2 = int_sub(i0, i1)
        // i3 = int_sub(i1, i0)  <- NOT commutative, should NOT be eliminated
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntSub, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntSub, &[Arg::In(1), Arg::In(0)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_multiple_opcodes() {
        // i2 = int_add(i0, i1)
        // i3 = int_mul(i0, i1)  <- different opcode, should NOT be eliminated
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntMul, &[Arg::In(0), Arg::In(1)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_three_duplicates() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i0, i1)  <- eliminated
        // i4 = int_add(i0, i1)  <- eliminated
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_call_pure_demoted_to_call() {
        // call_pure_i(args...) -> should become call_i(args...)
        let result = run_pure(
            2,
            &[op_spec(OpCode::CallPureI, &[Arg::In(0), Arg::In(1)])],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallI);
        // Demotion preserves the operand identities — the two header inputs.
        assert_eq!(
            result[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)]
        );
    }

    #[test]
    fn test_call_pure_r_demoted() {
        // CallPureR / CallR carry RPython `RefOp.type = 'r'` parity
        // (resoperation.py): the result is a Ref-typed Box, and
        // the function-pointer arg is also Ref-typed.
        let mut b = crate::history::test_support::TraceBuilder::new();
        let funcptr = b.input(Type::Ref, 0);
        b.op(OpCode::CallPureR, &[funcptr]);
        let (ops, inputs) = b.build();

        let mut opt = Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt
            .optimize_with_constants_and_inputs_oprc(
                &ops,
                &mut majit_ir::ConstMap::default(),
                inputs.len(),
            )
            .expect("test: unexpected InvalidLoop");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallR);
        assert_eq!(result[0].pos().get().ty(), Some(Type::Ref));
    }

    #[test]
    fn test_non_pure_op_passes_through() {
        // setfield_gc is not pure, should pass through unchanged.
        // struct is a Ref input, value an Int input.
        let mut b = crate::history::test_support::TraceBuilder::new();
        let struct_box = b.input(Type::Ref, 0);
        let value = b.input(Type::Int, 1);
        b.op(OpCode::SetfieldGc, &[struct_box, value]);
        let (ops, inputs) = b.build();

        let mut opt = Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt
            .optimize_with_constants_and_inputs_oprc(
                &ops,
                &mut majit_ir::ConstMap::default(),
                inputs.len(),
            )
            .expect("test: unexpected InvalidLoop");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
    }

    /// Two `getarrayitem_gc_pure_r(arr, const_i, descr)` operations
    /// against the SAME (immutable) array, index, and descr are CSE'd by
    /// OptPure — the second is folded to the first (`make_equal_to`).  This
    /// is the producer-side invariant the canonical-tuple `t[i]` arm relies
    /// on: purity is carried by the `GetarrayitemGcPureR` OPCODE (the descr
    /// stays the shared non-pure singleton), and OptPure never invalidates a
    /// pure load on an intervening write.
    #[test]
    fn test_cse_getarrayitem_gc_pure_r() {
        let arr_descr = majit_ir::descr::make_array_descr(8, 8, Type::Ref);
        let result = run_pure(
            1,
            &[
                op_spec_descr(
                    OpCode::GetarrayitemGcPureR,
                    &[Arg::In(0), Arg::Const(Value::Int(0))],
                    arr_descr.clone(),
                ),
                op_spec_descr(
                    OpCode::GetarrayitemGcPureR,
                    &[Arg::In(0), Arg::Const(Value::Int(0))],
                    arr_descr.clone(),
                ),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        // Only the first pure load survives; the second is CSE'd away.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GetarrayitemGcPureR);
    }

    /// Negative control for the pure CSE above: two pure loads at DIFFERENT
    /// indices are NOT folded together (the index discriminates the slot).
    #[test]
    fn test_cse_getarrayitem_gc_pure_r_distinct_index_not_folded() {
        let arr_descr = majit_ir::descr::make_array_descr(8, 8, Type::Ref);
        let result = run_pure(
            1,
            &[
                op_spec_descr(
                    OpCode::GetarrayitemGcPureR,
                    &[Arg::In(0), Arg::Const(Value::Int(0))],
                    arr_descr.clone(),
                ),
                op_spec_descr(
                    OpCode::GetarrayitemGcPureR,
                    &[Arg::In(0), Arg::Const(Value::Int(1))],
                    arr_descr.clone(),
                ),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_unary_ops() {
        // i1 = int_neg(i0)
        // i2 = int_neg(i0)  <- should be eliminated
        let result = run_pure(
            1,
            &[
                op_spec(OpCode::IntNeg, &[Arg::In(0)]),
                op_spec(OpCode::IntNeg, &[Arg::In(0)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cse_float_ops() {
        // f2 = float_add(f0, f1)
        // f3 = float_add(f0, f1)  <- should be eliminated
        let mut b = crate::history::test_support::TraceBuilder::new();
        let f0 = b.input(Type::Float, 0);
        let f1 = b.input(Type::Float, 1);
        b.op(OpCode::FloatAdd, &[f0.clone(), f1.clone()]);
        b.op(OpCode::FloatAdd, &[f0, f1]);
        let (ops, inputs) = b.build();

        let mut opt = Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt
            .optimize_with_constants_and_inputs_oprc(
                &ops,
                &mut majit_ir::ConstMap::default(),
                inputs.len(),
            )
            .expect("test: unexpected InvalidLoop");

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cache_eviction() {
        // Force a tiny cache so eviction behavior is deterministic even if
        // the production default changes. Each op's operands (i, i+100) are
        // distinct header inputs; the final op re-reads inputs 0 and 100.
        let mut specs = Vec::new();
        for i in 0..17u32 {
            specs.push(op_spec(OpCode::IntAdd, &[Arg::In(i), Arg::In(i + 100)]));
        }
        // Re-insert op #0: same args as specs[0]
        specs.push(op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(100)]));
        let num_inputs = 117u32;
        let (ops, inputs) = build_trace(num_inputs, &specs);

        let mut opt = Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptPure {
            cache: Rc::new(RefCell::new(RecentPureOpTable::new(16))),
            postponed_op: None,
            call_pure_positions: Vec::new(),
            pending_call_pure_position: false,
            last_emitted_was_removed: false,
            known_result_call_pure: Vec::new(),
            extra_call_pure: Vec::new(),
            call_pure_results: crate::optimizeopt::util::ArgsDict::default(),
        }));
        let result = opt
            .optimize_with_constants_and_inputs_oprc(
                &ops,
                &mut majit_ir::ConstMap::default(),
                num_inputs as usize,
            )
            .expect("test: unexpected InvalidLoop");

        // All 17 unique ops should be emitted, plus the re-inserted one
        // (since the first was evicted from the LRU cache of size 16).
        assert_eq!(result.len(), 18);
    }

    #[test]
    fn test_cse_with_forwarding() {
        // Test that CSE works correctly when OpRef forwarding is involved.
        let mut ctx = OptContext::new(10);
        let mut pass = OptPure::new();

        // Production binds the input operands a, b before the int_add is
        // processed; bind bound InputArg boxes here and reuse them across both
        // ops so same_box resolves both ops' args to one shared identity.
        let a = crate::history::test_support::rooted_inputarg_operand(Type::Int, 0);
        let b = crate::history::test_support::rooted_inputarg_operand(Type::Int, 1);

        // Simulate: op0 = int_add(a, b)
        let op0 = Op::new(OpCode::IntAdd, &[a.clone(), b.clone()]);
        let mut op0 = op0;
        op0.pos().set(OpRef::int_op(2));
        let result0 = pass.propagate_forward(&op0, &OpRc::new(op0.clone()), &mut ctx);
        assert!(matches!(result0, OptimizationResult::PassOn));
        let op0_rc = OpRc::new(op0.clone());
        ctx.push_new_operation(op0_rc.clone());
        pass.propagate_postprocess(&op0, &op0_rc, &mut ctx);

        // Simulate: op1 = int_add(a, b) with same args
        let op1 = Op::new(OpCode::IntAdd, &[a.clone(), b.clone()]);
        let mut op1 = op1;
        op1.pos().set(OpRef::int_op(3));
        let result1 = pass.propagate_forward(&op1, &OpRc::new(op1.clone()), &mut ctx);
        assert!(matches!(result1, OptimizationResult::Remove));
    }

    #[test]
    fn pure_history_does_not_merge_distinct_unregistered_producers() {
        let mut ctx = OptContext::with_num_inputs(16, 0);
        let mut pass = OptPure::new();
        let make_arg = || {
            let op = OpRc::new(Op::new(
                OpCode::SameAsI,
                &[Operand::const_from_value(Value::Int(7))],
            ));
            op.pos().set(OpRef::int_op(2));
            Operand::from_bound_op(&op)
        };
        let first = make_arg();
        let different = make_arg();
        let op = OpRc::new(Op::new(OpCode::IntNeg, &[first]));
        op.pos().set(OpRef::int_op(4));
        pass.pure(&op, &op);
        let query = Op::new(OpCode::IntNeg, &[different]);
        assert_eq!(pass.get_pure_result(&query, &mut ctx), None);
    }

    #[test]
    fn pure_history_retains_argument_boxes_and_observes_forwarding() {
        let mut ctx = OptContext::with_num_inputs(16, 0);
        let mut pass = OptPure::new();
        let a = ctx.materialize_operand_at(OpRef::int_op(2));
        let b = ctx.materialize_operand_at(OpRef::int_op(3));
        let op = OpRc::new(Op::new(OpCode::IntAdd, &[a.clone(), b.clone()]));
        op.pos().set(OpRef::int_op(4));
        pass.pure(&op, &op);
        let stored = {
            let cache = pass.cache.borrow();
            cache.bucket(OpCode::IntAdd).unwrap().lst[0]
                .as_ref()
                .unwrap()
                .0
                .args[0]
                .clone()
        };
        assert!(OpRc::ptr_eq(
            &stored.bound_op().unwrap(),
            &a.bound_op().unwrap(),
        ));
        ctx.make_constant_box(&a, Value::Int(7));
        let query = Op::new(
            OpCode::IntAdd,
            &[Operand::const_from_value(Value::Int(7)), b],
        );
        assert_eq!(
            pass.get_pure_result(&query, &mut ctx).map(|r| r.to_opref()),
            Some(OpRef::int_op(4))
        );
    }

    #[test]
    fn commutative_lookup_resolves_each_stored_argument_once() {
        let mut recent = RecentPureOps::new(4);
        recent.insert(
            PureOpKey {
                opcode: OpCode::IntAdd,
                args: smallvec::smallvec![
                    Operand::const_from_value(Value::Int(2)),
                    Operand::const_from_value(Value::Int(1))
                ],
                descr_identity: None,
            },
            {
                let result = OpRc::new(Op::new(
                    OpCode::SameAsI,
                    &[Operand::const_from_value(Value::Int(3))],
                ));
                result.pos().set(OpRef::int_op(3));
                Operand::from_bound_op(&result)
            },
        );
        let resolved = std::cell::Cell::new(0);
        let result = recent.lookup2(
            OpRef::const_int(1),
            OpRef::const_int(2),
            None,
            true,
            |arg| {
                resolved.set(resolved.get() + 1);
                arg.to_opref()
            },
            |query, stored| query == *stored,
        );
        assert_eq!(
            result.map(|(opcode, r)| (opcode, r.to_opref())),
            Some((OpCode::IntAdd, OpRef::int_op(3)))
        );
        assert_eq!(resolved.get(), 2);
    }

    #[test]
    fn pure_keys_keep_unary_and_binary_arguments_inline() {
        for arity in [1, 2, 8] {
            let args: Vec<_> = (0..arity)
                .map(|i| Operand::const_from_value(Value::Int(i)))
                .collect();
            let op = Op::new(OpCode::CallPureI, &args);
            let key = PureOpKey::from_operand_op(&op);
            assert_eq!(key.args.len(), arity as usize);
            assert_eq!(key.args.spilled(), arity > 2);
            for (i, arg) in key.args.iter().enumerate() {
                assert_eq!(arg.to_opref(), OpRef::const_int(i as i64));
            }
        }
    }

    #[test]
    fn test_pure_op_key_equality() {
        let key1 = PureOpKey {
            opcode: OpCode::IntAdd,
            args: smallvec::smallvec![OpRef::int_op(0), OpRef::int_op(1)],
            descr_identity: None,
        };
        let key2 = PureOpKey {
            opcode: OpCode::IntAdd,
            args: smallvec::smallvec![OpRef::int_op(0), OpRef::int_op(1)],
            descr_identity: None,
        };
        let key3 = PureOpKey {
            opcode: OpCode::IntAdd,
            args: smallvec::smallvec![OpRef::int_op(1), OpRef::int_op(0)],
            descr_identity: None,
        };
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_commutative_xor() {
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntXor, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntXor, &[Arg::In(1), Arg::In(0)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_commutative_int_and() {
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntAnd, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntAnd, &[Arg::In(1), Arg::In(0)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_comparison_cse() {
        // i2 = int_lt(i0, i1)
        // i3 = int_lt(i0, i1)  <- should be eliminated
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntLt, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::IntLt, &[Arg::In(0), Arg::In(1)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_call_pure_f_n_demoted() {
        // CallPureF / CallF carry RPython `FloatOp.type = 'f'` parity
        // (resoperation.py). CallPureN / CallN are void-result —
        // `AbstractResOp.type = 'v'` (resoperation.py) — and pyre
        // mints them as `OpRef::VoidOp(pos)` whose `ty()` is
        // `Some(Type::Void)`.
        let mut b = crate::history::test_support::TraceBuilder::new();
        let f0 = b.input(Type::Float, 0);
        b.op(OpCode::CallPureF, std::slice::from_ref(&f0));
        b.op(OpCode::CallPureN, &[f0]);
        let (ops, inputs) = b.build();

        let mut opt = Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt
            .optimize_with_constants_and_inputs_oprc(
                &ops,
                &mut majit_ir::ConstMap::default(),
                inputs.len(),
            )
            .expect("test: unexpected InvalidLoop");

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallF);
        assert_eq!(result[0].pos().get().ty(), Some(Type::Float));
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[1].pos().get().ty(), Some(Type::Void));
    }

    #[test]
    fn test_mixed_pure_and_non_pure() {
        // Mix of pure and non-pure operations, only duplicated pure ops get CSE'd.
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]), // pure, kept
                op_spec(OpCode::SetfieldGc, &[Arg::In(0), Arg::In(1)]), // not pure, kept
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]), // pure dup, eliminated
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
    }

    fn rewrite_pass() -> Box<dyn crate::optimizeopt::Optimization> {
        Box::new(crate::optimizeopt::rewrite::OptRewrite::new())
    }

    #[test]
    fn test_call_loopinvariant_cse() {
        // Two identical CALL_LOOPINVARIANT_I calls → second eliminated.
        // The func pointer (arg0) is a known Const; arg1 a regular input.
        let func = Arg::Const(Value::Int(0xCAFE));
        let result = run_pure(
            1,
            &[
                op_spec(OpCode::CallLoopinvariantI, &[func.clone(), Arg::In(0)]),
                op_spec(OpCode::CallLoopinvariantI, &[func, Arg::In(0)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[rewrite_pass],
            false,
        );

        // Only the first call should remain, demoted to CallI.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallI);
    }

    #[test]
    fn test_call_loopinvariant_different_args() {
        // CALL_LOOPINVARIANT_I with different args → both kept. The func
        // pointer is a plain input here (not a known constant), so OptRewrite
        // only demotes to CallI without loop-invariant CSE.
        let result = run_pure(
            3,
            &[
                op_spec(OpCode::CallLoopinvariantI, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::CallLoopinvariantI, &[Arg::In(0), Arg::In(2)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[rewrite_pass],
            false,
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallI);
        assert_eq!(result[1].opcode, OpCode::CallI);
    }

    #[test]
    fn test_call_loopinvariant_all_types() {
        for (loopinv_op, expected_op) in [
            (OpCode::CallLoopinvariantI, OpCode::CallI),
            (OpCode::CallLoopinvariantR, OpCode::CallR),
            (OpCode::CallLoopinvariantF, OpCode::CallF),
            (OpCode::CallLoopinvariantN, OpCode::CallN),
        ] {
            let result = run_pure(
                1,
                &[op_spec(loopinv_op, &[Arg::In(0)])],
                &mut majit_ir::ConstMap::default(),
                &[rewrite_pass],
                false,
            );

            assert_eq!(result.len(), 1);
            assert_eq!(result[0].opcode, expected_op);
        }
    }

    #[test]
    fn test_call_loopinvariant_no_eviction() {
        // Unlike pure CSE (LRU limit 16), loop-invariant cache has no eviction.
        // Create 20 unique calls, then re-check the first one. Each func
        // pointer is a distinct known Const (required for OptRewrite CSE);
        // they all share one non-const input arg.
        let mut specs = Vec::new();
        for i in 0..20u32 {
            specs.push(op_spec(
                OpCode::CallLoopinvariantI,
                &[Arg::Const(Value::Int((i + 100) as i64)), Arg::In(0)],
            ));
        }
        // Re-insert call #0: same args as specs[0]
        specs.push(op_spec(
            OpCode::CallLoopinvariantI,
            &[Arg::Const(Value::Int(100)), Arg::In(0)],
        ));
        let result = run_pure(
            1,
            &specs,
            &mut majit_ir::ConstMap::default(),
            &[rewrite_pass],
            false,
        );

        // 20 unique calls + the duplicate (#0) should be eliminated → 20 total
        assert_eq!(result.len(), 20);
    }

    #[test]
    fn test_call_loopinvariant_mixed_with_pure() {
        // Loop-invariant and pure CSE should coexist. The loop-invariant call's
        // func pointer is a known Const; its other arg and the pure-op operands
        // are inputs.
        let func = Arg::Const(Value::Int(0xBEEF));
        let result = run_pure(
            3,
            &[
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]), // pure
                op_spec(OpCode::CallLoopinvariantI, &[func.clone(), Arg::In(2)]), // loopinvariant
                op_spec(OpCode::IntAdd, &[Arg::In(0), Arg::In(1)]), // pure dup → removed
                op_spec(OpCode::CallLoopinvariantI, &[func, Arg::In(2)]), // loopinv dup → removed
            ],
            &mut majit_ir::ConstMap::default(),
            &[rewrite_pass],
            false,
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::CallI);
    }

    #[test]
    fn test_constant_fold_int_add() {
        // IntAdd(const(3), const(4)) → eliminated, result = const(7).
        // Finish consumes the result so the fold is observed.
        let result = run_pure(
            0,
            &[
                op_spec(
                    OpCode::IntAdd,
                    &[Arg::Const(Value::Int(3)), Arg::Const(Value::Int(4))],
                ),
                // Use the result in a finish to prevent dead code elimination
                op_spec(OpCode::Finish, &[Arg::Prod(0)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        // IntAdd should be folded away (only Finish remains)
        assert_eq!(result.len(), 1, "IntAdd(3,4) should be constant-folded");
        assert_eq!(result[0].opcode, OpCode::Finish);
    }

    #[test]
    fn test_constant_fold_int_lt() {
        // IntLt(const(3), const(5)) → const(1) (true)
        let result = run_pure(
            0,
            &[
                op_spec(
                    OpCode::IntLt,
                    &[Arg::Const(Value::Int(3)), Arg::Const(Value::Int(5))],
                ),
                op_spec(OpCode::Finish, &[Arg::Prod(0)]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        assert_eq!(result.len(), 1, "IntLt(3,5) should be constant-folded");
    }

    #[test]
    fn test_ovf_postponement_cse() {
        // INT_ADD_OVF(a, b) + GUARD_NO_OVERFLOW
        // then same INT_ADD_OVF(a, b) + GUARD_NO_OVERFLOW → CSE'd away
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::IntAddOvf, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::GuardNoOverflow, &[]),
                op_spec(OpCode::IntAddOvf, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::GuardNoOverflow, &[]),
                op_spec(OpCode::Finish, &[]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            true,
        );

        // First pair stays, second pair CSE'd → 3 ops total
        let ovf_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::IntAddOvf)
            .count();
        assert_eq!(ovf_count, 1, "duplicate OVF should be CSE'd");
    }

    #[test]
    fn test_ovf_constant_fold() {
        // INT_ADD_OVF(const(3), const(4)) + GUARD_NO_OVERFLOW → both removed
        let result = run_pure(
            0,
            &[
                op_spec(
                    OpCode::IntAddOvf,
                    &[Arg::Const(Value::Int(3)), Arg::Const(Value::Int(4))],
                ),
                op_spec(OpCode::GuardNoOverflow, &[]),
                op_spec(OpCode::Finish, &[]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            true,
        );

        // Both OVF and guard should be folded away
        let ovf_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::IntAddOvf)
            .count();
        assert_eq!(ovf_count, 0, "OVF(3,4) should be constant-folded");
    }

    /// The all-constant pair that actually overflows. `constant_fold` panics
    /// on `Err(NoConstExecutor)` — upstream's `NotImplementedError` — and the
    /// `all_args_const` pre-check above the call does not filter this case, so
    /// the executor must answer `Ok(None)` here and not fall through to
    /// `Err`. The op and its guard then stay, and the runtime guard fires.
    #[test]
    fn test_ovf_constant_fold_declines_on_real_overflow() {
        let result = run_pure(
            0,
            &[
                op_spec(
                    OpCode::IntAddOvf,
                    &[Arg::Const(Value::Int(i64::MAX)), Arg::Const(Value::Int(1))],
                ),
                op_spec(OpCode::GuardNoOverflow, &[]),
                op_spec(OpCode::Finish, &[]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            true,
        );

        let opcodes: Vec<OpCode> = result.iter().map(|o| o.opcode).collect();
        let ovf_at = opcodes
            .iter()
            .position(|opcode| *opcode == OpCode::IntAddOvf)
            .unwrap_or_else(|| panic!("the OVF op must survive the fold, got {opcodes:?}"));
        assert_eq!(
            opcodes.get(ovf_at + 1),
            Some(&OpCode::GuardNoOverflow),
            "the guard that catches the overflow must still follow it, got {opcodes:?}"
        );
    }

    fn run_immutable_getfield_cse(opcode: OpCode, index: u32, field_type: Type) -> Vec<Op> {
        let flag = match field_type {
            Type::Int => majit_ir::ArrayFlag::Signed,
            Type::Float => majit_ir::ArrayFlag::Float,
            Type::Ref => majit_ir::ArrayFlag::Unsigned,
            _ => unreachable!("getfield CSE test requires a typed field"),
        };
        let group = majit_ir::descr::make_simple_descr_group(
            index,
            16,
            1,
            0,
            &[majit_ir::descr::SimpleFieldDescrSpec {
                is_class_word: Some(false),
                index,
                field_key: format!("CseField{index}"),
                name: format!("CseField{index}"),
                offset: 0,
                field_size: 8,
                field_type,
                is_immutable: true,
                is_quasi_immutable: false,
                flag,
                virtualizable: false,
                index_in_parent: 0,
            }],
        );
        let descr = group.field_descrs[0].clone() as majit_ir::DescrRef;
        let base = crate::history::test_support::rooted_inputarg_operand(Type::Ref, 100);
        let mut ops = vec![
            Op::with_descr(opcode, std::slice::from_ref(&base), descr.clone()),
            Op::with_descr(opcode, &[base], descr),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::heap::OptHeap::new()));
        opt.trace_inputargs = OpRef::inputarg_refs(&vec![Type::Ref; 1024]);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024)
    }

    #[test]
    fn test_immutable_getfield_gc_i_cse_uses_heap_cache() {
        let result = run_immutable_getfield_cse(OpCode::GetfieldGcI, 1, Type::Int);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    #[test]
    fn test_immutable_getfield_gc_f_cse_uses_heap_cache() {
        let result = run_immutable_getfield_cse(OpCode::GetfieldGcF, 2, Type::Float);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcF);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    #[test]
    fn test_immutable_getfield_gc_r_cse_uses_heap_cache() {
        let result = run_immutable_getfield_cse(OpCode::GetfieldGcR, 3, Type::Ref);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    #[test]
    #[should_panic(expected = "must be a gcref")]
    fn test_constant_fold_getfield_gc_does_not_treat_int_constant_as_gc_pointer() {
        // Upstream `optimizer.py protect_speculative_operation`
        // derefs `op.getarg(0)` via `getref_base()` — only `ConstPtr`
        // supports that. RPython's type system makes a `GETFIELD_GC_I`
        // with `ConstInt` arg0 unconstructible. Pyre's
        // strict-orthodoxy port panics on the variant mismatch instead
        // of returning `None`; this test pins that behavior.
        let descr = make_field_descr_full(4, 0, 8, Type::Int, true);
        let mut ctx = OptContext::with_num_inputs(4, 0);
        let arg_box = ctx.materialize_operand_at(OpRef::int_op(10));
        ctx.make_constant_box(&arg_box, Value::Int(2));
        let mut op = Op::with_descr(OpCode::GetfieldGcI, std::slice::from_ref(&arg_box), descr);
        op.pos().set(OpRef::int_op(0));

        // Resolve forwarded args (mirrors propagate_from_pass_range) so the op
        // carries the canonical const box the pass reads via get_constant_box.
        let resolved = ctx
            .resolve_operand_operand_opt(&op.arg(0))
            .expect("constant arg resolves");
        op.setarg(0, resolved);

        let _ = ctx.constant_fold(&op);
    }

    #[test]
    fn test_guard_no_exception_after_removed_call_pure() {
        // CALL_PURE_I(same args) × 2 → second removed → GUARD_NO_EXCEPTION after removed
        let result = run_pure(
            2,
            &[
                op_spec(OpCode::CallPureI, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::GuardNoException, &[]),
                op_spec(OpCode::CallPureI, &[Arg::In(0), Arg::In(1)]),
                op_spec(OpCode::GuardNoException, &[]), // should be removed
                op_spec(OpCode::Finish, &[]),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            true,
        );

        // Second CALL_PURE → removed (CSE), its GUARD_NO_EXCEPTION → removed
        let gne_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNoException)
            .count();
        assert_eq!(
            gne_count, 1,
            "GUARD_NO_EXCEPTION after removed CALL_PURE should be eliminated"
        );
    }

    /// pure.py / :72-74 same_box semantics for constant args.
    /// history.py / :251 — `same_box(a, b) == same_constant(a, b)`
    /// for Const subclasses, so cache hits are value-equality. With
    /// inline `ConstInt.value`, two `make_constant_int(5)` calls return
    /// the same `OpRef::ConstInt(5)` and the cache hit is by
    /// OpRef equality (which is now value equality).
    #[test]
    fn lookup_pure_matches_same_value_constants_across_slots() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::new(0);

        let c5_a = ctx.make_constant_int(5);
        let c5_b = ctx.make_constant_int(5);
        assert_eq!(
            c5_a, c5_b,
            "two ConstInt(5) boxes are same_constant-equal (history.py:251)"
        );
        assert_eq!(ctx.get_constant(c5_a), ctx.get_constant(c5_b));

        // Cache `IntAdd(c5_a, x)` and look up `IntAdd(c5_b, x)`. `x` is the
        // canonical box materialized in ctx; the constants are Const operands.
        let x = OpRef::int_op(7);
        let x_box = ctx.materialize_operand_at(x);
        let x8_box = ctx.materialize_operand_at(OpRef::int_op(8));
        let c5 = Operand::const_from_value(Value::Int(5));
        let result42 = ctx.materialize_operand_at(OpRef::int_op(42));
        pass.pure_from_args2(OpCode::IntAdd, &c5, &x_box, &result42, &mut ctx);

        let mut q = Op::new(
            OpCode::IntAdd,
            &[Operand::const_from_value(Value::Int(5)), x_box.clone()],
        );
        q.pos().set(OpRef::int_op(99));
        assert_eq!(
            pass.get_pure_result(&q, &mut ctx).map(|r| r.to_opref()),
            Some(OpRef::int_op(42)),
            "lookup_pure must use same_box for constant args (history.py:204)"
        );

        // A non-constant slot mismatch must still miss.
        let mut q_miss = Op::new(
            OpCode::IntAdd,
            &[Operand::const_from_value(Value::Int(5)), x8_box.clone()],
        );
        q_miss.pos().set(OpRef::int_op(100));
        assert_eq!(pass.get_pure_result(&q_miss, &mut ctx), None);
    }

    /// pure.py applies get_box_replacement to the lookup-side op args
    /// before comparing them with stored pure op args.
    #[test]
    fn lookup_pure_replaces_query_args_before_matching() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::new(0);

        let query_arg = OpRef::int_op(7);
        let canonical_arg = OpRef::int_op(8);
        let other_arg = OpRef::int_op(9);
        let result = OpRef::int_op(42);
        let b_query = ctx.materialize_operand_at(query_arg);
        let b_canonical = ctx.materialize_operand_at(canonical_arg);
        ctx.make_equal_to(&b_query.clone(), &b_canonical.clone());
        let b_other = ctx.materialize_operand_at(other_arg);

        let result_box = ctx.materialize_operand_at(result);
        pass.pure_from_args2(
            OpCode::IntAdd,
            &b_canonical,
            &b_other,
            &result_box,
            &mut ctx,
        );

        // Query op carries the canonical ctx boxes (query_arg forwards to
        // canonical_arg via make_equal_to) — no position-only mint.
        let mut q = Op::new(OpCode::IntAdd, &[b_query.clone(), b_other.clone()]);
        q.pos().set(OpRef::int_op(99));
        assert_eq!(
            pass.get_pure_result(&q, &mut ctx).map(|r| r.to_opref()),
            Some(result),
            "lookup_pure must apply get_box_replacement to query args"
        );
    }

    #[test]
    fn test_pure_and_pure_from_args() {
        let mut pass = OptPure::new();

        let mut ctx = OptContext::new(0);
        // Bind the operand positions canonically so same_box resolves the
        // looked-up op's args to the same boxes the cache recorded.
        let b10 = ctx.materialize_operand_at(OpRef::int_op(10));
        let b20 = ctx.materialize_operand_at(OpRef::int_op(20));
        let b30 = ctx.materialize_operand_at(OpRef::int_op(30));
        let b40 = ctx.materialize_operand_at(OpRef::int_op(40));

        // Manually record a pure operation via the API
        let op = OpRc::new(Op::new(OpCode::IntAdd, &[b10.clone(), b20.clone()]));
        op.pos().set(OpRef::int_op(0));
        pass.pure(&op, &op);

        // Should find it via get_pure_result
        let lookup_op = Op::new(OpCode::IntAdd, &[b10.clone(), b20.clone()]);
        assert!(pass.get_pure_result(&lookup_op, &mut ctx).is_some());

        // pure_from_args
        let b5 = ctx.materialize_operand_at(OpRef::int_op(5));
        pass.pure_from_args(OpCode::IntMul, &[b30.clone(), b40.clone()], &b5, &mut ctx);
        let mut lookup_mul = Op::new(OpCode::IntMul, &[b30.clone(), b40.clone()]);
        lookup_mul.pos().set(OpRef::int_op(99));
        assert!(pass.get_pure_result(&lookup_mul, &mut ctx).is_some());
    }

    #[test]
    fn test_extra_call_pure() {
        let mut pass = OptPure::new();

        // Inject extra_call_pure from a previous loop iteration
        let args = vec![OpRef::int_op(100), OpRef::int_op(101)];
        pass.set_extra_call_pure(vec![(args.clone(), OpRef::int_op(50))]);

        // extra_call_pure entries are searched via optimize_call_pure_old
        // in the CALL_PURE handler (pure.py), not via
        // lookup_known_result (which only searches known_result_call_pure).
        let key = super::PureOpKey {
            opcode: OpCode::CallPureI,
            args: smallvec::SmallVec::from_vec(args),
            descr_identity: None,
        };
        // Verify entries exist in extra_call_pure
        assert_eq!(pass.extra_call_pure.len(), 1);
        match &pass.extra_call_pure[0] {
            super::ExtraCallPureEntry::Direct { key: k, result } => {
                assert_eq!(k, &key);
                assert_eq!(*result, OpRef::int_op(50));
            }
            _ => panic!("expected Direct entry"),
        }
    }

    #[test]
    fn test_known_result_call_pure_lookup() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(4, 0);
        // Bind the matched call args canonically so same_box resolves them.
        let b100 = ctx.materialize_operand_at(OpRef::int_op(100));
        let b101 = ctx.materialize_operand_at(OpRef::int_op(101));
        let b999 = ctx.materialize_operand_at(OpRef::int_op(999));

        // pure.py: self.known_result_call_pure.append(op)
        pass.known_result_call_pure.push(super::KnownResultEntry {
            descr_identity: None,
            args: vec![OpRef::int_op(100), OpRef::int_op(101)],
            result: OpRef::int_op(50),
        });

        // CALL_PURE lookup: start_index=0, descr matches (both None), args match
        let op = Op::new(OpCode::CallPureI, &[b100.clone(), b101.clone()]);
        assert_eq!(
            pass.lookup_known_result(&op, 0, &ctx),
            Some(OpRef::int_op(50))
        );

        // COND_CALL_VALUE lookup: start_index=1, skip arg(0)
        let cond_op = Op::new(
            OpCode::CondCallValueI,
            &[b999.clone(), b100.clone(), b101.clone()],
        );
        assert_eq!(
            pass.lookup_known_result(&cond_op, 1, &ctx),
            Some(OpRef::int_op(50))
        );

        // Args mismatch → None
        let bad_args = Op::new(OpCode::CallPureI, &[b100.clone(), b999.clone()]);
        assert_eq!(pass.lookup_known_result(&bad_args, 0, &ctx), None);
    }

    #[test]
    fn test_imported_short_pure_is_not_a_produce_side_table() {
        // pure.py produce_potential_short_preamble_ops walks
        // `_newoperations` only. An imported PreambleOp seeds RecentPureOps
        // (shortpreamble.py opt.pure) and is not re-exported as a short box
        // unless this compilation emits it again.
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        let const_opref = OpRef::const_int(7);
        let const_box = ctx.materialize_operand_at(const_opref);
        ctx.seed_constant(&const_box.clone(), majit_ir::Value::Int(7));
        let box0 = ctx.materialize_operand_at(OpRef::int_op(0));
        let imported = crate::optimizeopt::ImportedShortPureOp::new(
            &mut ctx,
            OpCode::IntAdd,
            None,
            vec![box0, Operand::from_opref(const_opref)],
            OpRef::int_op(2),
            OpRef::int_op(2),
            false,
            None,
        );
        ctx.imported_short_pure_ops.push(imported);

        pass.setup();
        pass.install_preamble_pure_ops(&mut ctx);

        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::int_op(0),
            OpRef::int_op(1),
        ]);
        for arg in [OpRef::int_op(0), OpRef::int_op(1)] {
            sb.add_short_input_arg(&mut ctx, arg, majit_ir::Type::Int);
        }
        pass.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let collected = sb.produced_ops(&mut ctx);
        let pure: Vec<_> = collected
            .iter()
            .filter(|(_, p)| {
                matches!(
                    p.kind,
                    crate::optimizeopt::shortpreamble::PreambleOpKind::Pure
                )
            })
            .collect();
        assert!(pure.is_empty());
    }

    #[test]
    fn imported_preamble_pure_is_found_through_the_recent_ops_ring() {
        // shortpreamble.py opt.pure(opnum, PreambleOp) writes the same
        // RecentPureOps ring body CSE reads. A later IntAdd with those
        // args must force the preamble slot rather than miss it.
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        let const_opref = OpRef::const_int(7);
        let const_box = ctx.materialize_operand_at(const_opref);
        ctx.seed_constant(&const_box.clone(), majit_ir::Value::Int(7));
        let arg0 = ctx.materialize_operand_at(OpRef::int_op(0));
        let imported = crate::optimizeopt::ImportedShortPureOp::new(
            &mut ctx,
            OpCode::IntAdd,
            None,
            vec![arg0.clone(), Operand::from_opref(const_opref)],
            OpRef::int_op(2),
            OpRef::int_op(2),
            false,
            None,
        );
        ctx.imported_short_pure_ops.push(imported);
        pass.setup();
        pass.install_preamble_pure_ops(&mut ctx);
        let query = Op::new(OpCode::IntAdd, &[arg0, const_box]);
        assert!(
            pass.recent_ops_has_preamble(&query, &ctx),
            "imported IntAdd must land in RecentPureOps"
        );
    }

    #[test]
    fn test_imported_short_call_pure_result_replays_into_pure_cache() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(8, 0);
        // Bind the non-const call arg position so same_box resolves the
        // dispatched op's arg to the same box the imported op recorded.
        let arg0 = ctx.materialize_operand_at(OpRef::int_op(0));
        let const_opref = OpRef::const_int(0x1234);
        let call_descr = majit_ir::descr::make_call_descr_full(
            77,
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            true,
            8,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::ElidableCannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        let mut imported = crate::optimizeopt::ImportedShortPureOp::new(
            &mut ctx,
            OpCode::CallPureI,
            Some(call_descr.clone()),
            vec![Operand::from_opref(const_opref), arg0.clone()],
            OpRef::int_op(1),
            OpRef::int_op(1),
            false,
            None,
        );
        initialize_imported_short_pure_builder(
            &mut ctx,
            (*imported.pop.preamble_op).clone(),
            Some(1),
        );
        // Production threads the builder's replay Rc into the pop
        // (produce_pure); mirror it so use_box sees one object. The
        // builder keys by the entry res box (`materialize_operand_at(pos)`); the
        // memoized box for the same position hits.
        let src1 = ctx.materialize_operand_at(OpRef::int_op(1));
        if let Some(p) = ctx
            .imported_short_preamble_builder
            .as_ref()
            .and_then(|b| b.produced_short_op(&src1.clone()))
        {
            imported.pop.preamble_op = p.preamble_op;
        }
        ctx.imported_short_pure_ops.push(imported);

        pass.setup();
        pass.install_preamble_pure_ops(&mut ctx);

        let mut op = Op::new(
            OpCode::CallPureI,
            &[Operand::const_from_value(Value::Int(0x1234)), arg0.clone()],
        );
        op.pos().set(OpRef::int_op(2));
        op.setdescr(call_descr);
        // Register the dispatched op as the producer at its position so the
        // collapsed `from_bound_op(op_rc)` resolves to the same box the test
        // reads back via `get_box_replacement_operand_opt(pos)` (mirrors production
        // input-op binding).
        let op_rc = OpRc::new(op.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));
        let result = pass.propagate_forward(&op, &op_rc, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_operand_opt(OpRef::int_op(2))
                .map(|o| o.to_opref()),
            Some(OpRef::int_op(1))
        );
    }

    #[test]
    fn test_short_preamble_collects_pure_op_candidate() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(4, 0);
        pass.setup();

        let a0 = ctx.materialize_operand_at(OpRef::int_op(0));
        let a1 = ctx.materialize_operand_at(OpRef::int_op(1));
        let mut op = Op::new(OpCode::IntAdd, &[a0.clone(), a1.clone()]);
        op.pos().set(OpRef::int_op(2));
        let result = pass.propagate_forward(&op, &OpRc::new(op.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));

        // Only `_newoperations` (pure.py:317) makes the producer a candidate.
        let op_rc = OpRc::new(op.clone());
        ctx.push_new_operation(op_rc.clone());
        pass.propagate_postprocess(&op, &op_rc, &mut ctx);

        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::int_op(0),
            OpRef::int_op(1),
            OpRef::int_op(2),
        ]);
        // Seed the renamed ShortInputargs for the op's deps (0, 1); the op
        // result (pos 2) is not a ShortInputArg.
        for arg in [OpRef::int_op(0), OpRef::int_op(1)] {
            sb.add_short_input_arg(&mut ctx, arg, majit_ir::Type::Int);
        }
        pass.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let collected = sb.produced_ops(&mut ctx);
        let pure: Vec<_> = collected
            .iter()
            .filter(|(_, p)| {
                matches!(
                    p.kind,
                    crate::optimizeopt::shortpreamble::PreambleOpKind::Pure
                )
            })
            .collect();
        assert_eq!(pure.len(), 1);
        assert_eq!(pure[0].1.preamble_op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_short_preamble_collects_non_raising_call_pure_candidate() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        pass.setup();

        let a100 = ctx.materialize_operand_at(OpRef::int_op(100));
        let a0 = ctx.materialize_operand_at(OpRef::int_op(0));
        let a1 = ctx.materialize_operand_at(OpRef::int_op(1));
        let mut op = Op::new(OpCode::CallPureI, &[a100.clone(), a0.clone(), a1.clone()]);
        op.pos().set(OpRef::int_op(2));
        op.setdescr(majit_ir::descr::make_call_descr(
            vec![
                majit_ir::Type::Int,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
            ],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::ElidableCannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        ));
        let result = pass.propagate_forward(&op, &OpRc::new(op.clone()), &mut ctx);
        // The demote routes the CALL through the remaining passes (Replace),
        // mirroring RPython's `self.emit(newop)`, so OptHeap still processes it.
        let emitted = match result {
            OptimizationResult::Replace(emitted) => {
                assert_eq!(emitted.opcode, OpCode::CallI);
                emitted
            }
            other => panic!("expected demoted call routed via Replace, got {other:?}"),
        };
        // pure.py reads the demoted CALL out of `_newoperations`
        // at `call_pure_positions`.
        let emitted_rc = OpRc::new(emitted.clone());
        ctx.push_new_operation(emitted_rc.clone());
        pass.propagate_postprocess(&emitted, &emitted_rc, &mut ctx);

        // Deps are the call args (100, 0, 1); the call result (pos 2) is not a
        // label arg.
        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::int_op(0),
            OpRef::int_op(1),
            OpRef::int_op(100),
        ]);
        for arg in [OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(100)] {
            sb.add_short_input_arg(&mut ctx, arg, majit_ir::Type::Int);
        }
        pass.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let collected = sb.produced_ops(&mut ctx);
        let pure: Vec<_> = collected
            .iter()
            .filter(|(_, p)| {
                matches!(
                    p.kind,
                    crate::optimizeopt::shortpreamble::PreambleOpKind::Pure
                )
            })
            .collect();
        assert_eq!(pure.len(), 1);
        assert_eq!(pure[0].1.preamble_op.opcode, OpCode::CallPureI);
    }

    #[test]
    fn test_short_preamble_collects_loopinvariant_candidate() {
        let mut rewrite = crate::optimizeopt::rewrite::OptRewrite::new();
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        // func pointer arg must be a known constant for OptRewrite tracking
        let func_box = ctx.materialize_operand_at(OpRef::int_op(100));
        ctx.seed_constant(&func_box.clone(), majit_ir::Value::Int(0xCAFE));
        rewrite.setup();
        pass.setup();

        let a0 = ctx.materialize_operand_at(OpRef::int_op(0));
        let mut op = Op::new(OpCode::CallLoopinvariantI, &[func_box.clone(), a0.clone()]);
        op.pos().set(OpRef::int_op(2));
        op.setdescr(majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::ElidableCannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        ));
        // optimizer.py:651-652 setarg loop parity: canonicalize args the way
        // propagate_from_pass_range does before propagate_forward.
        for i in 0..op.num_args() {
            op.setarg(
                i,
                ctx.resolve_operand_operand_opt(&op.arg(i))
                    .unwrap_or_else(|| op.arg(i).clone()),
            );
        }
        // OptRewrite demotes CallLoopinvariantI → CallI
        let rewrite_result = rewrite.propagate_forward(&op, &OpRc::new(op.clone()), &mut ctx);
        let demoted = match rewrite_result {
            OptimizationResult::Emit(emitted) => emitted,
            other => panic!("expected OptRewrite to emit demoted call, got {other:?}"),
        };
        assert_eq!(demoted.opcode, OpCode::CallI);
        // The driver runs OptRewrite's postprocess on the emitted call.
        let demoted_rc = OpRc::new(demoted.clone());
        rewrite.propagate_postprocess(&demoted, &demoted_rc, &mut ctx);
        // OptPure sees the demoted CallI
        let result = pass.propagate_forward(&demoted, &demoted_rc, &mut ctx);
        match result {
            OptimizationResult::Emit(emitted) => assert_eq!(emitted.opcode, OpCode::CallI),
            OptimizationResult::PassOn => {} // PassOn is also acceptable
            other => panic!("expected emitted or pass-on from OptPure, got {other:?}"),
        }

        // OptRewrite tracks loopinvariant for short preamble collection.
        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::int_op(0),
            OpRef::int_op(2),
            OpRef::int_op(100),
        ]);
        // The func arg 100 is a known constant (produce_arg returns it
        // unrenamed); only the real input 0 gets a renamed ShortInputArg.
        sb.note_known_constant(OpRef::int_op(100));
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(0), majit_ir::Type::Int);
        rewrite.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let collected = sb.produced_ops(&mut ctx);
        let loopinv: Vec<_> = collected
            .iter()
            .filter(|(_, p)| {
                matches!(
                    p.kind,
                    crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant
                )
            })
            .collect();
        assert_eq!(loopinv.len(), 1);
        assert_eq!(loopinv[0].1.preamble_op.opcode, OpCode::CallLoopinvariantI);
    }

    #[test]
    fn test_lookup1_lookup2() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(64, 0);

        // Record via pure_from_args
        let a10 = ctx.materialize_operand_at(OpRef::int_op(10));
        let a20 = ctx.materialize_operand_at(OpRef::int_op(20));
        let a30 = ctx.materialize_operand_at(OpRef::int_op(30));
        pass.pure_from_args(OpCode::IntAdd, &[a10, a20], &a30, &mut ctx);

        // same_box: identity comparison (no constants, no forwarding)
        let sb = |a: OpRef, b: OpRef| a == b;
        // lookup2 should find it
        assert!(
            pass.lookup2(
                OpCode::IntAdd,
                OpRef::int_op(10),
                OpRef::int_op(20),
                None,
                false,
                sb
            )
            .is_some()
        );
        // lookup2 with commutative should find swapped
        assert!(
            pass.lookup2(
                OpCode::IntAdd,
                OpRef::int_op(20),
                OpRef::int_op(10),
                None,
                true,
                sb
            )
            .is_some()
        );
        // Non-commutative swapped should NOT find it
        assert!(
            pass.lookup2(
                OpCode::IntAdd,
                OpRef::int_op(20),
                OpRef::int_op(10),
                None,
                false,
                sb
            )
            .is_none()
        );

        // lookup1 for a unary op
        let a10b = ctx.materialize_operand_at(OpRef::int_op(10));
        let a40 = ctx.materialize_operand_at(OpRef::int_op(40));
        pass.pure_from_args(OpCode::IntNeg, &[a10b], &a40, &mut ctx);
        assert!(
            pass.lookup1(OpCode::IntNeg, OpRef::int_op(10), None, sb)
                .is_some()
        );
        assert!(
            pass.lookup1(OpCode::IntNeg, OpRef::int_op(99), None, sb)
                .is_none()
        );
    }

    #[test]
    fn test_cond_call_value_cse() {
        // COND_CALL_VALUE_I(cond, func, arg) → CSE using args[1..]
        // A second COND_CALL_VALUE_I with same func+arg should reuse result.
        // The two conds differ (inputs 0/1); func and arg are shared inputs.
        let result = run_pure(
            4,
            &[
                op_spec(
                    OpCode::CondCallValueI,
                    &[Arg::In(0), Arg::In(2), Arg::In(3)],
                ),
                op_spec(
                    OpCode::CondCallValueI,
                    &[Arg::In(1), Arg::In(2), Arg::In(3)],
                ),
            ],
            &mut majit_ir::ConstMap::default(),
            &[],
            false,
        );

        // First COND_CALL_VALUE emitted, second removed by CSE
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CondCallValueI);
    }

    #[test]
    fn test_cond_call_value_uses_call_pure_results_starting_at_arg1() {
        // cond is a header input; func and arg are Const literals.
        let (ops, inputs) = build_trace(
            1,
            &[op_spec(
                OpCode::CondCallValueI,
                &[
                    Arg::In(0),
                    Arg::Const(Value::Int(0xCAFE)),
                    Arg::Const(Value::Int(7)),
                ],
            )],
        );
        let mut opt = Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.record_call_pure_result(vec![Value::Int(0xCAFE), Value::Int(7)], Value::Int(42));
        opt.add_pass(Box::new(OptPure::new()));

        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, inputs.len())
            .expect("test: unexpected InvalidLoop");

        assert!(result.is_empty());
        assert_eq!(
            ops[0].forwarded().borrow().const_value(),
            Some(majit_ir::Value::Int(42))
        );
    }

    /// REF analog of `test_cond_call_value_uses_call_pure_results_*` /
    /// the Int `test_call_pure_results`: a `CALL_PURE_R` whose all-const
    /// args (funcptr + Ref) hit a seeded `call_pure_results` entry must
    /// be removed by OptPure and its result made a Ref constant.
    ///
    /// pure.py `optimize_CALL_PURE_R` → start_index=0 (the
    /// funcptr is part of the lookup key).  `_can_optimize_call_pure`
    /// (optimizer.py) reads `get_constant_box(arg)` per arg →
    /// `Value::Ref(ConstPtr)` for the Ref arg — value equality, so the
    /// 2nd identical call collapses to the 1st's stored result.
    ///
    /// resoperation.py `RefOp.type = 'r'`: the folded constant must
    /// be `Value::Ref` (history.py ConstPtr), NOT `Value::Int`
    /// (history.py ConstInt) of the same numeric value.
    #[test]
    fn test_call_pure_r_results_folds_second_identical_call() {
        let result_ptr = GcRef(0xDEAD_BEEF);

        // op0's all-const args (funcptr 0xCAFE + Ref GcRef(0x4242)) hit the
        // seeded cache and fold to a Ref constant. A second residual CallPureR
        // (distinct key, no cache hit) consumes op0's result via a bound
        // `from_bound_op` reference (Prod(0)), so the fold is observed where it
        // matters in production: the folded Ref reaches its consumer as an
        // inline `ConstPtr` operand (later rewritten to `LoadFromGcTable` by the
        // GC rewrite). compilation-panic: a Ref constant is NEVER exported as a raw `GcRef`
        // into the backend constant pool — that pool has no GC root walker, so
        // refs live only in the GC-traced gc_table. (The Int analog DOES export
        // to the pool; ints carry no GC concern.)
        let (ops, inputs) = build_trace(
            0,
            &[
                op_spec(
                    OpCode::CallPureR,
                    &[
                        Arg::Const(Value::Int(0xCAFE)),
                        Arg::Const(Value::Ref(GcRef(0x4242))),
                    ],
                ),
                // arg(1) binds op0's result box (Prod(0) → RefOp(0)); after the
                // fold it resolves to the inline ConstPtr of the cached result.
                op_spec(
                    OpCode::CallPureR,
                    &[Arg::Const(Value::Int(0xF00D)), Arg::Prod(0)],
                ),
            ],
        );
        // `build_trace` types op.pos via result_type() → CallPureR gives a
        // Ref-typed position OpRef.
        assert_eq!(
            ops[0].pos().get().ty(),
            Some(Type::Ref),
            "CallPureR position must be Ref-typed (resoperation.py:638)"
        );

        let mut opt = Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        // Seed the cross-call result keyed on [funcptr, ref_arg] (Ref
        // value), as record_result_of_call_pure does at trace time.
        opt.record_call_pure_result(
            vec![Value::Int(0xCAFE), Value::Ref(GcRef(0x4242))],
            Value::Ref(result_ptr),
        );
        opt.add_pass(Box::new(OptPure::new()));

        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, inputs.len())
            .expect("test: unexpected InvalidLoop");

        // op0 collapses to the cached constant (removed); op1 is the residual
        // call that consumes it.
        assert_eq!(
            result.len(),
            1,
            "op0 folds via call_pure_results; op1 residual remains; got {result:?}"
        );
        // A pure call with no cache hit lowers to a residual `CallR`.
        let consumer = &result[0];
        assert_eq!(consumer.opcode, OpCode::CallR);
        // The folded constant reaches the consumer as ConstPtr(result_ptr)
        // (history.py), NOT a ConstInt of the same numeric value
        // (resoperation.py `RefOp.type = 'r'`).
        assert_eq!(
            consumer.arg(1).to_opref(),
            OpRef::const_ptr(result_ptr),
            "CallPureR fold must yield ConstPtr(result_ptr), got {:?}",
            consumer.arg(1).to_opref()
        );
        assert!(
            !matches!(consumer.arg(1).to_opref(), OpRef::ConstInt(_)),
            "folded CallPureR constant aliased to ConstInt — ConstPtr/ConstInt distinction lost"
        );
        // compilation-panic: no raw GcRef is exported into the backend constant pool.
        assert!(
            constants.values().all(|v| !matches!(v, Value::Ref(_))),
            "backend constant pool must not retain a raw GcRef (use the gc_table); got {constants:?}"
        );
    }
}
