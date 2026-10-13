/// Heap optimization pass: caches field/array reads and eliminates redundant loads/stores.
///
/// Translated from rpython/jit/metainterp/optimizeopt/heap.py.
///
/// Optimizations performed:
/// - Read-after-write elimination: SETFIELD then GETFIELD on same obj/field -> use cached value
/// - Write-after-write elimination: two SETFIELDs on same obj/field -> only keep the last
/// - Read-after-read elimination: two GETFIELDs on same obj/field -> reuse first result
/// - Same for array items (SETARRAYITEM_GC / GETARRAYITEM_GC) with constant index
/// - Cache invalidation on calls and side-effecting operations
/// - Lazy set emission: SETFIELD_GC is delayed until a guard or side-effecting op forces it
/// - GUARD_NOT_INVALIDATED deduplication
use bit_set::BitSet;

#[inline(always)]
fn use_untranslated_heap_ordering() -> bool {
    cfg!(debug_assertions)
}

#[inline]
#[allow(dead_code)] // heap.py descr sort before translation
fn sort_descr_entries_untranslated<T>(entries: &mut [(u32, DescrRef, T)]) {
    if use_untranslated_heap_ordering() {
        entries.sort_by_key(|b| std::cmp::Reverse(b.1.repr()));
    }
}

/// [`sort_descr_entries_untranslated`] for a caller that permutes an index
/// list instead of the entries themselves.
#[inline]
fn sort_descr_entry_indices_untranslated<T>(entries: &[(u32, DescrRef, T)], order: &mut [usize]) {
    if use_untranslated_heap_ordering() {
        order.sort_by(|&a, &b| entries[b].1.repr().cmp(&entries[a].1.repr()));
    }
}

#[inline]
fn sort_descr_item_refs_untranslated<T>(entries: &mut [(&DescrRef, T)]) {
    if use_untranslated_heap_ordering() {
        entries.sort_by_key(|b| std::cmp::Reverse(b.0.repr()));
    }
}

#[inline]
fn sort_array_index_entries_untranslated<T>(entries: &mut [(i64, T)]) {
    if use_untranslated_heap_ordering() {
        entries.sort_by_key(|(index, _)| *index);
    }
}

use indexmap::{IndexMap, IndexSet};
use majit_ir::{
    DescrRef, IndexMapExt, OopSpecIndex, Op, OpCode, OpRc, OpRef, Value, descr::descr_identity,
};
use std::sync::Arc;

/// `descr_identity` as a local integer: the `Arc<dyn Descr>` data pointer.
/// A linear `cached_fields` scan compares this word; it must not go through
/// a `dyn Descr` method per cached entry.
#[inline(always)]
fn descr_ptr(descr: &DescrRef) -> usize {
    Arc::as_ptr(descr) as *const () as usize
}

use crate::optimizeopt::info::PtrInfoExt;
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};
use majit_ir::operand::Operand;

fn same_ptr_info(ctx: &OptContext, left: &Operand, right: &Operand) -> bool {
    let left_box = ctx.resolve_operand_operand(left);
    let right_box = ctx.resolve_operand_operand(right);
    match (
        ctx.getptrinfo_handle(&left_box),
        ctx.getptrinfo_handle(&right_box),
    ) {
        // heap.py possible_aliasing_two_infos: opinfo1.same_info(opinfo2).
        (Some(left_info), Some(right_info)) => left_info.same_info(&right_info),
        // Both normal callers install PtrInfo via ensure_ptr_info_arg0 before
        // reaching this path. Keep synthetic tests conservative without adding
        // a parallel store.
        _ => left_box.same_box(&right_box),
    }
}

/// heap.py AbstractCachedEntry._cannot_alias_via_constants
///
/// Two constant pointers are two addresses fixed at trace time, so unless they
/// are the same constant they name different objects. Lives beside
/// [`same_ptr_info`] rather than on `CachedField` / `ArrayCachedItem` because,
/// like its upstream counterpart on the base class, it reads nothing about
/// what is cached and so is shared by fields and array items.
///
/// Upstream asks only whether both sides are `ConstPtrInfo`, since
/// `same_info` has already returned MUST_ALIAS for the equal case by then;
/// comparing the two constants here answers the same question without
/// depending on that ordering.
fn cannot_alias_via_constants(ctx: &mut OptContext, left: &Operand, right: &Operand) -> bool {
    fn const_ref(ctx: &mut OptContext, op: &Operand) -> Option<majit_ir::GcRef> {
        match ctx.resolve_operand_operand(op).const_value() {
            Some(Value::Ref(gcref)) => Some(gcref),
            _ => None,
        }
    }
    let left = const_ref(ctx, left);
    let right = const_ref(ctx, right);
    matches!((left, right), (Some(a), Some(b)) if a != b)
}

#[inline]
fn make_nonnull_box(ctx: &mut OptContext, arg: &Operand) {
    if let Some(box_ref) = ctx.resolve_operand_operand_opt(arg) {
        ctx.make_nonnull(&box_ref);
    }
}

/// util.py args_dict() / args_eq(): same_box semantics — identity
/// for non-Const boxes, value-equality for Const subclasses (history.py).
/// Two distinct Const slots holding the same value must hash and compare
/// equal so consecutive dict lookups with key `5` (encoded via different
/// const slots) hit the cache.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum DictArgKey {
    Const(Value),
    Op(Operand),
}

impl DictArgKey {
    fn from_arg(arg: &Operand, ctx: &OptContext) -> Self {
        // One chain walk to the terminal box; a constant terminal keys by
        // value, anything else by its resolved producer identity.
        match ctx.get_box_replacement_operand_opt(arg.to_opref()) {
            Some(b) => match b.const_value() {
                Some(value) => DictArgKey::Const(value),
                None => DictArgKey::Op(b),
            },
            // Defensive: an arg with no resolved terminal has no canonical
            // identity. Key by the arg's own producer identity — two distinct
            // unresolved args do not dedup, a missed cache hit and never a
            // correctness issue (the lookup is simply re-emitted). Production
            // dict-lookup args are recorded CALL operands and are always bound.
            None => {
                debug_assert!(
                    false,
                    "cached_dict_reads: unbound dict-lookup arg {arg:?} — bind-at-alloc invariant"
                );
                DictArgKey::Op(arg.clone())
            }
        }
    }
}

/// Cache key for a field access: (struct OpRef, field descriptor identity).
///
/// RPython `heap.py` keys `cached_fields` and green/quasi field caches by the
/// descriptor object itself.  `FieldDescr.get_index()` is a different domain:
/// it indexes `PtrInfo._fields`.
#[allow(dead_code)]
type FieldKey = (OpRef, usize);

/// Rust representation of `heap.AbstractCachedEntry` and `CachedField`.
///
/// PyPy uses Python inheritance to share `do_setfield`,
/// `force_lazy_set`, `getfield_from_cache`, `possible_aliasing` and
/// `possible_aliasing_two_infos` between `CachedField` and
/// `ArrayCachedItem`. The Rust port keeps the same per-method
/// signatures as inherent methods on each struct (so call sites read
/// like the PyPy source). Shared bodies are kept in
/// `abstract_cached_entry::*` free helpers and invoked from each
/// inherent method.
///
/// Rust-specific naming notes that mirror PyPy's contract:
/// - PyPy `cached_infos: [PtrInfo]` is replaced by `cached_structs:
///   Vec<OpRef>` because Rust's borrow checker forbids holding
///   parallel `&mut PtrInfo` references; the PtrInfo itself is read
///   on-demand from `box._forwarded` / `ctx.const_infos[gcref]`.
/// - PyPy `descr` parameters are carried as descriptor references for
///   field-cache identity, with a separate `field_idx` / `descr_idx`
///   (u32) only where the RPython source indexes `PtrInfo` slots or
///   EffectInfo bitsets.
struct CachedField {
    /// `AbstractCachedEntry.cached_structs`: struct boxes with a cached value
    /// for this descr. Replaces RPython's parallel `cached_infos`;
    /// the PtrInfo itself is read on-demand from
    /// `ctx.get_ptr_info(opref)` / `ctx.get_const_info(opref)`.
    cached_structs: Vec<Operand>,
    /// `AbstractCachedEntry._lazy_set`: at most one pending `SetfieldGc` per descr.
    /// Stores the pending ResOperation object (`_lazy_set = op`); the struct
    /// base is `op.getarg(0)`, resolved on demand by the consumers.
    lazy_set: Option<majit_ir::OpRc>,
}

impl CachedField {
    fn new() -> Self {
        CachedField {
            cached_structs: Vec::new(),
            lazy_set: None,
        }
    }

    /// heap.py AbstractCachedEntry.register_info(structop, info)
    ///
    /// Tracks `struct_box` so subsequent `invalidate(descr)` knows
    /// to clear `opinfo._fields[descr_idx]`. RPython appends to both
    /// `cached_structs` and `cached_infos`; the Rust port skips
    /// `cached_infos` and reads PtrInfo on-demand.
    fn register_info(&mut self, struct_box: &Operand) {
        self.cached_structs.push(struct_box.clone());
    }

    /// heap.py AbstractCachedEntry.possible_aliasing
    ///
    /// `not info.getptrinfo(self._lazy_set.getarg(0)).same_info(opinfo)`.
    fn possible_aliasing(&self, struct_box: &Operand, ctx: &OptContext) -> bool {
        match &self.lazy_set {
            Some(lazy_op) => !same_ptr_info(ctx, &lazy_op.arg(0), struct_box),
            None => false,
        }
    }

    /// heap.py CachedField._get_rhs_from_set_op
    fn _get_rhs_from_set_op(op: &Op) -> Operand {
        op.arg(1)
    }

    /// heap.py CachedField.invalidate(descr)
    ///
    /// PyPy iterates `cached_infos` and writes
    /// `opinfo._fields[descr.get_index()] = None`. The Rust port walks
    /// `cached_structs`, resolves each opref through `box._forwarded`
    /// OR `ctx.const_infos` (the latter mirrors `info.py make_guards
    /// ConstPtrInfo._get_info` which routes constant bases through
    /// `optheap.const_infos[gcref]`), and calls
    /// `info.clear_field(descr_idx)`.
    /// heap.py `CachedField.invalidate(descr)` line-by-line:
    ///
    /// ```python
    /// def invalidate(self, descr):
    ///     if descr.is_always_pure():
    ///         return
    ///     for opinfo in self.cached_infos:
    ///         assert isinstance(opinfo, info.AbstractStructPtrInfo)
    ///         opinfo._fields[descr.get_index()] = None
    ///     self.cached_infos = []
    /// ```
    ///
    /// The `descr.is_always_pure()` short-circuit is performed inside
    /// the method, not the caller — symmetric with upstream. The slot
    /// index is the `field_idx` stored on the `cached_fields` entry
    /// (computed once at insert via `field_slot_index`). Callers no
    /// longer need to gate manually.
    fn invalidate(&mut self, descr: &DescrRef, field_idx: u32, ctx: &mut OptContext) {
        // `CachedField.invalidate`: `if descr.is_always_pure(): return`.
        // Upstream's word covers a quasi-immutable field too — `is_pure` is
        // `_immutable_field(name) != False` and `IR_QUASIIMMUTABLE` is an
        // instance, not `False` — so a `?` field's cache survives a call and
        // the loop keeps one read, with `guard_not_invalidated` carrying the
        // per-window check. pyre's `is_always_pure` answers false for a `?`
        // field to withhold the const-fold that same word licenses, so the
        // surviving-a-call half is spelled here instead. Without it every
        // residual call forces a re-read; over the scored bench dir that is 2
        // of the 5 `ExecutionContext.w_tracefunc` getfield+guard pairs left in
        // a peeled body. The other three read a different struct box each
        // time, so no cache can reach them: upstream gets one box per loop from
        // `ThreadLocalReference(ExecutionContext, loop_invariant=True)`.
        if descr.is_always_pure() || descr.is_quasi_immutable() {
            return;
        }
        let descr_idx = field_idx;
        for obj in &self.cached_structs {
            // One chain walk: an unresolved position has no PtrInfo and no
            // const_infos slot (the box-native resolver yields None there).
            if let Some(b_op) = ctx.resolve_operand_operand_opt(obj) {
                ctx.with_ptr_info_mut(&b_op, |info| info.clear_field(descr_idx));
                // Clear existing const_infos slot if present; do NOT create.
                if let Some(info) = ctx.get_const_info_mut_if_exists_box(&b_op) {
                    info.clear_field(descr_idx);
                }
            }
        }
        self.cached_structs.clear();
    }

    /// heap.py CachedField._getfield(opinfo, descr, optheap, true_force=True)
    ///
    /// Returns the raw FieldEntry from _fields — caller checks Preamble vs Value.
    /// RPython: `res = opinfo.getfield(descr, optheap)`
    ///          `if isinstance(res, PreambleOp): ...`
    fn _getfield(
        &self,
        struct_box: &Operand,
        descr: &DescrRef,
        field_idx: u32,
        ctx: &mut OptContext,
    ) -> Option<crate::optimizeopt::info::FieldEntry> {
        // info.py:212-214: return self._fields[fielddescr.get_index()]
        let struct_box = ctx.resolve_operand_operand(struct_box);
        if let Some(info) = struct_box.ptr_info()
            && let Some(entry) = info.getfield(field_idx)
        {
            return Some(entry);
        }
        // info.py ConstPtrInfo.getfield → _get_info(parent_descr, optheap)
        let parent_descr = descr.as_field_descr().and_then(|fd| fd.get_parent_descr());
        if let Some(info) = ctx.get_const_info_mut_box(&struct_box, parent_descr)
            && let Some(entry) = info.getfield(field_idx)
        {
            return Some(entry);
        }
        None
    }

    /// heap.py AbstractCachedEntry.getfield_from_cache
    /// heap.py AbstractCachedEntry.getfield_from_cache
    #[allow(dead_code)]
    fn getfield_from_cache(
        &self,
        struct_box: &Operand,
        descr: &DescrRef,
        field_idx: u32,
        ctx: &mut OptContext,
    ) -> Option<crate::optimizeopt::info::FieldEntry> {
        if let Some(lazy_op) = &self.lazy_set
            && same_ptr_info(ctx, &lazy_op.arg(0), struct_box)
        {
            let rhs = Self::_get_rhs_from_set_op(lazy_op);
            return Some(crate::optimizeopt::info::FieldEntry::Value(rhs.clone()));
        }
        let struct_box = ctx.resolve_operand_operand(struct_box);
        self._getfield(&struct_box, descr, field_idx, ctx)
    }

    /// heap.py CachedField._cannot_alias_via_classes_or_lengths
    fn _cannot_alias_via_classes_or_lengths(
        box1: &Operand,
        box2: &Operand,
        ctx: &mut OptContext,
    ) -> bool {
        // info.py get_known_class. PyPy: opinfo1.get_known_class(cpu)
        // / opinfo2.get_known_class(cpu); CANNOT_ALIAS iff both are
        // known and not the same constant.
        let b1 = ctx.resolve_operand_operand(box1);
        let b2 = ctx.resolve_operand_operand(box2);
        let class1 = ctx
            .getptrinfo_handle(&b1)
            .and_then(|h| h.borrow().get_known_class(ctx.cpu.as_ref()));
        let class2 = ctx
            .getptrinfo_handle(&b2)
            .and_then(|h| h.borrow().get_known_class(ctx.cpu.as_ref()));
        matches!((class1, class2), (Some(c1), Some(c2)) if c1 != c2)
    }

    /// heap.py CachedField._cannot_alias_via_content
    fn _cannot_alias_via_content(box1: &Operand, box2: &Operand, ctx: &mut OptContext) -> bool {
        // heap.py CachedField._cannot_alias_via_content: both must be
        // AbstractStructPtrInfo.
        let b1 = ctx.resolve_operand_operand(box1);
        let b2 = ctx.resolve_operand_operand(box2);
        let (Some(info1), Some(info2)) = (b1.ptr_info(), b2.ptr_info()) else {
            return false;
        };
        // heap.py:211-216: all_items() may be None
        let f1 = info1.all_items();
        let f2 = info2.all_items();
        if f1.is_empty() || f2.is_empty() {
            return false;
        }
        // heap.py:217-225: shared field with two different constants
        // → CANNOT_ALIAS. RPython iterates positionally; the Rust port
        // matches by field_idx (equivalent for the same descriptor layout).
        for (idx1, e1) in &f1 {
            for (idx2, e2) in &f2 {
                if idx1 != idx2 {
                    continue;
                }
                // Only compare concrete values; Preamble entries have no
                // known constant value. One chain walk per entry to its
                // terminal box, then compare constants
                // (heap.py:221-224 get_box_replacement + same_constant).
                if let (Some(v1), Some(v2)) = (e1.as_value(), e2.as_value()) {
                    let c1 = ctx.resolve_operand_operand(&v1).const_value();
                    let c2 = ctx.resolve_operand_operand(&v2).const_value();
                    if matches!((c1, c2), (Some(a), Some(b)) if a != b) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// heap.py CachedField.put_field_back_to_info
    #[allow(dead_code)]
    fn put_field_back_to_info(&mut self, op: &Op, ctx: &mut OptContext) {
        // info.py opinfo.setfield(descr, struct, op, optheap, cf=self)
        // PyPy: `setfield(..., cf=cf)` calls `cf.register_info(struct, self)`
        // (info.py). The Rust port performs both halves here.
        let descr_idx = op
            .getdescr()
            .as_ref()
            .map(OptHeap::field_slot_index)
            .unwrap_or(0);
        let arg = ctx.resolve_operand_operand(&op.arg(1));
        let struct_box = ctx.resolve_operand_operand(&op.arg(0));
        self.register_info(&struct_box);
        ctx.structinfo_setfield(op, descr_idx, &arg);
    }

    /// heap.py AbstractCachedEntry.produce_potential_short_preamble_ops
    ///
    /// Iterates `cached_structs` and emits a getfield op for each
    /// cached entry that still has a non-None `opinfo._fields[descr_idx]`.
    /// PyPy's method calls `info.produce_short_preamble_ops(...)` on
    /// each cached_info, which itself emits a `GETFIELD_GC` /
    /// `GETARRAYITEM_GC` to the short preamble; the Rust port still
    /// inlines the emission here even though
    /// [`crate::optimizeopt::info::produce_short_preamble_ops`] is available.
    /// The inline path stays for callers that lack the `OptContext` plumbing
    /// the info helper expects.
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        descr: &DescrRef,
        descr_idx: u32,
        ctx: &mut OptContext,
    ) {
        debug_assert!(self.lazy_set.is_none());
        for cached in &self.cached_structs {
            // One chain walk; the position view falls back to the stored
            // key when no terminal box resolves (resolve_operand_operand_opt).
            let structbox_box = ctx.resolve_operand_operand_opt(cached);
            let structbox = structbox_box
                .as_ref()
                .map_or(cached.to_opref(), |b| b.to_opref());
            if structbox.is_none() {
                continue;
            }
            let cached_val = match structbox_box
                .as_ref()
                .and_then(Operand::ptr_info)
                .and_then(|info| info.getfield(descr_idx))
                .map(|entry| entry.as_seen_operand())
                .or_else(|| {
                    let parent_descr = descr.as_field_descr().and_then(|fd| fd.get_parent_descr());
                    structbox_box
                        .as_ref()
                        .and_then(|b| ctx.get_const_info_mut_box(b, parent_descr))
                        .and_then(|info| info.getfield(descr_idx))
                        .map(|entry| entry.as_seen_operand())
                }) {
                Some(v) if !v.is_none() => v,
                _ => continue,
            };
            let cached_box = ctx.resolve_operand_operand(&cached_val);
            let cached_val = cached_box.to_opref();
            if cached_val.is_none() {
                continue;
            }
            let opcode = descr
                .as_field_descr()
                .map(|fd| OpCode::getfield_for_type(fd.field_type()))
                .unwrap_or(OpCode::GetfieldGcI);
            // heap.py `structbox = optimizer.get_box_replacement(...)`.
            let sarg = structbox_box
                .clone()
                .unwrap_or_else(|| ctx.materialize_operand_at(structbox));
            let mut op = Op::with_descr(opcode, &[sarg], descr.clone());
            op.pos().set(cached_val);
            sb.add_heap_op(ctx, cached_box, op);
        }
    }
}

/// Cache key for an array item access: (array OpRef, descriptor index, constant array index).
type ArrayItemKey = (OpRef, usize, i64);

/// heap.py ArrayCachedItem(AbstractCachedEntry)
struct ArrayCachedItem {
    /// heap.py:229-230 self.index — constant array index this entry
    /// is keyed by. RPython stores it as part of `ArrayCachedItem.__init__`.
    index: i64,
    /// heap.py:39 cached_structs — array boxes whose `_items[index]`
    /// slot holds a cached value. Replaces RPython's `cached_infos`.
    cached_structs: Vec<Operand>,
    /// heap.py:40 _lazy_set — at most one pending SetarrayitemGc.
    /// Stores the pending ResOperation object (`_lazy_set = op`); the array
    /// base is `op.getarg(0)`, resolved on demand by the consumers.
    lazy_set: Option<majit_ir::OpRc>,
}

impl ArrayCachedItem {
    fn new(index: i64) -> Self {
        // heap.py:230 assert index >= 0; self.index = index
        debug_assert!(index >= 0);
        ArrayCachedItem {
            index,
            cached_structs: Vec::new(),
            lazy_set: None,
        }
    }

    /// heap.py AbstractCachedEntry.register_info(structop, info)
    fn register_info(&mut self, array_box: &Operand) {
        self.cached_structs.push(array_box.clone());
    }

    /// heap.py AbstractCachedEntry.possible_aliasing
    ///
    /// `not info.getptrinfo(self._lazy_set.getarg(0)).same_info(opinfo)`.
    fn possible_aliasing(&self, array_box: &Operand, ctx: &OptContext) -> bool {
        match &self.lazy_set {
            Some(lazy_op) => !same_ptr_info(ctx, &lazy_op.arg(0), array_box),
            None => false,
        }
    }

    /// heap.py ArrayCachedItem._get_rhs_from_set_op
    fn _get_rhs_from_set_op(op: &Op) -> Operand {
        op.arg(2)
    }

    /// heap.py ArrayCachedItem._cannot_alias_via_classes_or_lengths
    fn _cannot_alias_via_classes_or_lengths(
        box1: &Operand,
        box2: &Operand,
        ctx: &mut OptContext,
    ) -> bool {
        use crate::optimizeopt::info::PtrInfo;
        // heap.py ArrayCachedItem._cannot_alias_via_classes_or_lengths:
        // both must be ArrayPtrInfo with known_ne lenbounds.
        let b1 = ctx.resolve_operand_operand(box1);
        let b2 = ctx.resolve_operand_operand(box2);
        let info1 = b1.ptr_info();
        let info2 = b2.ptr_info();
        match (info1.as_deref(), info2.as_deref()) {
            (Some(PtrInfo::Array(a1)), Some(PtrInfo::Array(a2))) => {
                a1.lenbound.known_ne(&a2.lenbound)
            }
            _ => false,
        }
    }

    /// heap.py ArrayCachedItem._cannot_alias_via_content
    fn _cannot_alias_via_content(box1: &Operand, box2: &Operand, ctx: &mut OptContext) -> bool {
        use crate::optimizeopt::info::PtrInfo;
        // heap.py ArrayCachedItem._cannot_alias_via_content:
        // isinstance(opinfo, ArrayPtrInfo). info.py all_items() returns
        // _items (the dense list, None slots included). The comparisons
        // only read forwarding, so retain the info borrows instead of
        // copying both arrays before reading their items.
        let b1 = ctx.resolve_operand_operand(box1);
        let b2 = ctx.resolve_operand_operand(box2);
        let (Some(info1), Some(info2)) = (b1.ptr_info(), b2.ptr_info()) else {
            return false;
        };
        let (PtrInfo::Array(a1), PtrInfo::Array(a2)) = (&*info1, &*info2) else {
            return false;
        };
        let (items1, items2) = (&a1.items, &a2.items);
        // heap.py:288-298: slot-by-slot comparison preserving index alignment.
        // None/Preamble slots are kept at their original positions.
        let len = items1.len().min(items2.len());
        for i in 0..len {
            // heap.py:289-292: value = get_box_replacement(content[index]);
            // if value is None: continue — one chain walk per slot to its
            // terminal box.
            let (Some(r1), Some(r2)) = (items1[i].as_value(), items2[i].as_value()) else {
                continue;
            };
            if r1.is_none() || r2.is_none() {
                continue;
            }
            // heap.py:293-294: if not value.is_constant(): continue
            let c1 = ctx.resolve_operand_operand(&r1).const_value();
            let c2 = ctx.resolve_operand_operand(&r2).const_value();
            let (Some(c1), Some(c2)) = (c1, c2) else {
                continue;
            };
            // heap.py: if not value1.same_constant(value2): return CANNOT_ALIAS
            if c1 != c2 {
                return true;
            }
        }
        false
    }

    /// heap.py ArrayCachedItem.invalidate(descr)
    ///
    /// PyPy iterates `cached_infos` and writes
    /// `opinfo._items[self.index] = None`. The Rust port walks
    /// `cached_structs` and routes through `box._forwarded` /
    /// `ctx.const_infos`. The `self.parent.clear_varindex()` half is
    /// performed by the caller (`ArrayCacheSubMap::invalidate_index`)
    /// because Rust forbids the back-pointer.
    fn invalidate(&mut self, ctx: &mut OptContext) {
        let index = self.index as usize;
        for obj in &self.cached_structs {
            // One chain walk: an unresolved position has no PtrInfo and no
            // const_infos slot (the box-native resolver yields None there).
            if let Some(b_op) = ctx.resolve_operand_operand_opt(obj) {
                ctx.with_ptr_info_mut(&b_op, |info| info.clear_item(index));
                // info.py ConstPtrInfo._get_array_info — only clear
                // an existing ArrayPtrInfo slot; do NOT create on miss.
                if let Some(info) = ctx.get_const_info_mut_if_exists_box(&b_op) {
                    info.clear_item(index);
                }
            }
        }
        self.cached_structs.clear();
    }

    /// heap.py ArrayCachedItem._getfield(opinfo, descr, optheap)
    ///
    /// Takes `descr` so the constant-base path can route through
    /// `ConstPtrInfo._get_array_info(descr, optheap)` (info.py)
    /// which creates an `ArrayPtrInfo` on miss.
    /// heap.py ArrayCachedItem._getfield — returns raw FieldEntry.
    fn _getfield(
        &self,
        array_opref: OpRef,
        descr: &DescrRef,
        ctx: &mut OptContext,
    ) -> Option<crate::optimizeopt::info::FieldEntry> {
        if self.index < 0 {
            return None;
        }
        let idx = self.index as usize;
        let array_box = ctx.get_box_replacement_operand_opt(array_opref);
        if let Some(info) = array_box.as_ref().and_then(Operand::ptr_info)
            && let Some(entry) = info.getitem(idx)
        {
            return Some(entry);
        }
        // info.py ConstPtrInfo.getitem → _get_array_info(descr, optheap)
        if let Some(info) = array_box
            .as_ref()
            .and_then(|b| ctx.get_const_info_array_mut_box(b, descr.clone()))
            && let Some(entry) = info.getitem(idx)
        {
            return Some(entry);
        }
        None
    }

    /// heap.py AbstractCachedEntry.getfield_from_cache
    #[allow(dead_code)]
    fn getfield_from_cache(
        &self,
        array_box: &Operand,
        descr: &DescrRef,
        ctx: &mut OptContext,
    ) -> Option<crate::optimizeopt::info::FieldEntry> {
        if let Some(lazy_op) = &self.lazy_set
            && same_ptr_info(ctx, &lazy_op.arg(0), array_box)
        {
            let rhs = Self::_get_rhs_from_set_op(lazy_op);
            return Some(crate::optimizeopt::info::FieldEntry::Value(rhs.clone()));
        }
        let array_box = ctx.resolve_operand_operand(array_box);
        self._getfield(array_box.to_opref(), descr, ctx)
    }

    /// heap.py ArrayCachedItem.put_field_back_to_info
    #[allow(dead_code)]
    fn put_field_back_to_info(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg = ctx.resolve_operand_operand(&op.arg(2));
        let struct_box = ctx.resolve_operand_operand(&op.arg(0));
        self.register_info(&struct_box);
        ctx.arrayinfo_setitem(op, self.index as usize, &arg);
    }

    /// heap.py AbstractCachedEntry.produce_potential_short_preamble_ops
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        descr: &DescrRef,
        ctx: &mut OptContext,
    ) {
        debug_assert!(self.lazy_set.is_none());
        for cached in &self.cached_structs {
            // One chain walk; the position view falls back to the stored
            // key when no terminal box resolves (resolve_operand_operand_opt).
            let arraybox_box = ctx.resolve_operand_operand_opt(cached);
            let arraybox = arraybox_box
                .as_ref()
                .map_or(cached.to_opref(), |b| b.to_opref());
            if arraybox.is_none() {
                continue;
            }
            let cached_val = match arraybox_box
                .as_ref()
                .and_then(Operand::ptr_info)
                .and_then(|info| info.getitem(self.index as usize))
                .map(|entry| entry.as_seen_operand())
                .or_else(|| {
                    arraybox_box
                        .as_ref()
                        .and_then(|b| ctx.get_const_info_array_mut_box(b, descr.clone()))
                        .and_then(|info| info.getitem(self.index as usize))
                        .map(|entry| entry.as_seen_operand())
                }) {
                Some(v) if !v.is_none() => v,
                _ => continue,
            };
            let cached_box = ctx.resolve_operand_operand(&cached_val);
            let cached_val = cached_box.to_opref();
            if cached_val.is_none() {
                continue;
            }
            // compile.py ResOperation(... [arrayop, ConstInt(index)] ...)
            let opcode = descr
                .as_array_descr()
                .map(|array_descr| OpCode::getarrayitem_for_type(array_descr.item_type()))
                .unwrap_or(OpCode::GetarrayitemGcI);
            let arraybox_b = arraybox_box
                .clone()
                .unwrap_or_else(|| ctx.materialize_operand_at(arraybox));
            let idx_b = Operand::const_(majit_ir::Const::Int(self.index));
            let mut op = Op::with_descr(opcode, &[arraybox_b, idx_b], descr.clone());
            op.pos().set(cached_val);
            sb.add_heap_op(ctx, cached_box, op);
        }
    }
}

/// heap.py ArrayCacheSubMap
///
/// Per-arraydescr container holding both constant-index entries and a
/// variable-index triples list. Mirrors RPython's `ArrayCacheSubMap`
/// 1:1.
struct ArrayCacheSubMap {
    /// heap.py: const_indexes = {} (int -> ArrayCachedItem)
    const_indexes: crate::FxIndexMap<i64, ArrayCachedItem>,
    /// heap.py:305-306: cached_varindex_triples = None
    /// List of (arrayinfo, indexbox, resbox). RPython keys arrayinfo by the
    /// info object's identity; the info lives on the array's resolved box, so
    /// that box stands for it.
    cached_varindex_triples: Option<Vec<(Operand, Operand, Operand)>>,
}

impl ArrayCacheSubMap {
    fn new() -> Self {
        ArrayCacheSubMap {
            const_indexes: crate::FxIndexMap::default(),
            cached_varindex_triples: None,
        }
    }

    fn const_get(&self, index: i64) -> Option<&ArrayCachedItem> {
        self.const_indexes.get(&index)
    }

    fn const_get_mut(&mut self, index: i64) -> Option<&mut ArrayCachedItem> {
        self.const_indexes.get_mut(&index)
    }

    fn const_keys(&self) -> Vec<i64> {
        self.const_indexes.keys().copied().collect()
    }

    fn const_get_or_new(&mut self, index: i64) -> &mut ArrayCachedItem {
        self.const_indexes
            .entry_or_insert_with(index, || ArrayCachedItem::new(index))
    }

    /// heap.py clear_varindex
    fn clear_varindex(&mut self) {
        self.cached_varindex_triples = None;
    }

    /// heap.py cache_varindex_read
    fn cache_varindex_read(&mut self, arrayinfo: Operand, indexbox: Operand, resbox: Operand) {
        let entry = (arrayinfo, indexbox, resbox);
        if self.cached_varindex_triples.is_none() {
            self.cached_varindex_triples = Some(vec![entry]);
            return;
        }
        self.cached_varindex_triples.as_mut().unwrap().push(entry);
    }

    /// heap.py cache_varindex_write
    fn cache_varindex_write(&mut self, arrayinfo: Operand, indexbox: Operand, resbox: Operand) {
        self.cached_varindex_triples = Some(vec![(arrayinfo, indexbox, resbox)]);
    }

    /// heap.py lookup_cached
    fn lookup_cached(
        &self,
        arrayinfo: &Operand,
        indexbox: &Operand,
        ctx: &OptContext,
    ) -> Option<Operand> {
        if let Some(triples) = &self.cached_varindex_triples {
            for (cached_arrayinfo, cached_index, cached_result) in triples {
                // heap.py lookup_cached: cached_arrayinfo is arrayinfo
                //   and get_box_replacement(cached_index) is indexbox
                if cached_arrayinfo.same_box(arrayinfo)
                    && ctx.resolve_operand_operand(cached_index).same_box(indexbox)
                {
                    return Some(ctx.resolve_operand_operand(cached_result));
                }
            }
        }
        None
    }

    /// heap.py ArrayCachedItem.invalidate (parent step inlined)
    ///
    /// Clears the cached entries at `index` (also clearing
    /// `arrayinfo._items[index]` for each cached_struct via
    /// `cai.invalidate(ctx)`) AND calls `self.parent.clear_varindex()`.
    /// The parent step is inlined here because Rust forbids the
    /// back-pointer that PyPy uses on `ArrayCachedItem.parent`.
    fn invalidate_index(&mut self, index: i64, ctx: &mut OptContext) {
        if let Some(cai) = self.const_get_mut(index) {
            cai.invalidate(ctx);
        }
        self.clear_varindex();
    }
}

/// Heap optimization pass.
///
/// Caches field and array item values to eliminate redundant loads, and delays
/// store emission (lazy sets) to enable write-after-write elimination.
///
/// Green field optimization: immutable field caches survive cache invalidation
/// by calls and side-effecting operations. When an immutable field is read from
/// a constant object, the result is also a constant (green field folding).
///
/// Aliasing analysis: objects allocated during the trace (NEW, NEW_WITH_VTABLE,
/// NEW_ARRAY, etc.) cannot alias each other or pre-existing objects. Their field
/// caches survive writes to other objects. Objects that haven't escaped (not
/// passed to calls or stored into the heap) keep their caches across calls.
pub struct OptHeap {
    /// heap.py `self.optimizer.metainterp_sd.virtualref_info`, published by
    /// Optimizer together with the other metainterp-static pass inputs.
    vrefinfo: crate::virtualref::VirtualRefInfo,
    /// Per-descr field cache: field_idx → CachedField.
    /// RPython heap.py: cached_fields OrderedDict keyed by descr.
    cached_fields: Vec<(u32, DescrRef, CachedField)>,
    /// heap.py:332: cached_arrayitems OrderedDict keyed by array descr.
    cached_arrayitems: Vec<(usize, DescrRef, ArrayCacheSubMap)>,
    /// Whether we've already emitted a GUARD_NOT_INVALIDATED.
    seen_guard_not_invalidated: bool,
    /// Postponed operation: held back until the next GUARD_NO_EXCEPTION.
    /// RPython heap.py: `postponed_op` — delays emission of operations
    /// that may raise (CALL_MAY_FORCE, comparison ops) until we see
    /// a GUARD_NO_EXCEPTION, ensuring correct exception semantics.
    postponed_op: Option<Op>,
    // ── Aliasing analysis state — RPython: PtrInfo flags ──
    seen_allocation: BitSet,
    /// heapcache.py `_check_flag(box, HF_IS_UNESCAPED)` — the set of
    /// unescaped (freshly-allocated, not-yet-escaped) boxes. RPython stores the
    /// flag on the box (`box._heapc_flags`); pyre keeps an OptHeap-owned set
    /// keyed by operand identity (`Rc::ptr_eq`) — box identity, not the retired
    /// `opref.raw()` slot index. OptHeap ownership preserves `setup()`'s per-run
    /// reset, which a per-box flag on a shared `Box` could not bulk-clear.
    unescaped: crate::FxIndexSet<Operand>,
    /// heapcache.py:209/298-307/453-455 `box._heapc_deps` — per-Box
    /// dependency list. RPython attaches `_heapc_deps: list | None`
    /// as an attribute on the `RefFrontendOp` Box object itself;
    /// pyre keeps a side-table keyed by operand identity for the same effect.
    /// When an unescaped value is stored into an unescaped container,
    /// the value is recorded as a dependency of the container instead
    /// of being immediately escaped. When the container escapes later,
    /// all its dependencies are transitively escaped.
    heapc_deps: crate::FxIndexMap<Operand, Vec<Operand>>,

    /// heap.py:27 Optimization.last_emitted_operation is REMOVED.
    /// Set to true when `_optimize_CALL_DICT_LOOKUP` folds a lookup;
    /// read by `optimize_GUARD_NO_EXCEPTION` to suppress the trailing guard.
    last_emitted_removed: bool,
    /// heap.py OptHeap.cached_dict_reads: cached_dict_reads — descr_identity(extradescrs[0]) → { [dict,key] → result box }.
    /// Consecutive dict lookups on the same dict+key are deduplicated.
    /// Inner key uses `DictArgKey` so Const args compare by value
    /// (util.py args_dict / args_eq via history.py same_box).
    cached_dict_reads: crate::FxIndexMap<usize, crate::FxIndexMap<[DictArgKey; 2], Operand>>,
    /// heap.py:560: corresponding_array_descrs — maps extradescrs[1] (entries
    /// array descr) → extradescrs[0] dict identity.
    ///
    /// PyPy stores the *array descr object* directly and resolves
    /// `arraydescr.ei_index` inside `effectinfo.check_write_descr_array(arraydescr)`
    /// at invalidation time (`effectinfo.py`).  Pyre keeps the
    /// `DescrRef` alive on the value side so the same lazy resolution can
    /// happen via `descr.get_ei_index()` at `force_from_effectinfo`.
    /// The map key is the descriptor object's identity; `descr.index()` is
    /// not globally unique across descriptor objects.
    corresponding_array_descrs: crate::FxIndexMap<usize, (DescrRef, usize)>,
    /// Fields known to be quasi-immutable: (obj box, field_idx) -> cached value
    /// OpRef. Keyed by the object's `Operand` identity (heap keys structs by box,
    /// not by the retired `opref.raw()` slot). Populated by QUASIIMMUT_FIELD,
    /// consumed by subsequent GETFIELD_GC_*. Survives calls (guarded by
    /// GUARD_NOT_INVALIDATED).
    quasi_immut_cache: crate::FxIndexMap<(Operand, usize), OpRef>,
}

impl OptHeap {
    pub fn new() -> Self {
        OptHeap {
            vrefinfo: crate::virtualref::VirtualRefInfo::new(),
            cached_fields: Vec::new(),
            cached_arrayitems: Vec::new(),
            seen_guard_not_invalidated: false,
            postponed_op: None,
            seen_allocation: BitSet::new(),
            unescaped: crate::FxIndexSet::default(),
            heapc_deps: crate::FxIndexMap::default(),
            last_emitted_removed: false,
            cached_dict_reads: crate::FxIndexMap::default(),
            corresponding_array_descrs: crate::FxIndexMap::default(),
            quasi_immut_cache: crate::FxIndexMap::default(),
        }
    }

    fn cached_field_pos_for_descr(&self, descr: &DescrRef) -> Option<usize> {
        self.cached_field_pos(descr_ptr(descr))
    }

    fn cached_field_pos(&self, identity: usize) -> Option<usize> {
        self.cached_fields
            .iter()
            .position(|(_, cached_descr, _)| descr_ptr(cached_descr) == identity)
    }

    fn cached_array_pos_for_descr(&self, descr: &DescrRef) -> Option<usize> {
        self.cached_array_pos_for_index(descr_ptr(descr))
    }

    fn cached_array_pos_for_index(&self, descr_idx: usize) -> Option<usize> {
        self.cached_arrayitems
            .iter()
            .position(|(idx, _, _)| *idx == descr_idx)
    }

    /// RPython field descriptor identity for heap cache keys.
    ///
    /// This is the Rust equivalent of using `descr` as a Python dict key in
    /// `heap.py:392-397`.
    fn field_cache_identity(descr: &DescrRef) -> usize {
        descr_ptr(descr)
    }

    /// `quasiimmut.py QuasiImmutDescr.is_still_valid_for`, the check
    /// `heap.py:818-819` turns into `InvalidLoop('quasi immutable field changed
    /// during tracing')`.
    ///
    /// ```text
    ///  def is_still_valid_for(self, structconst):
    ///      assert self.struct
    ///      if self.struct != structconst.getref_base():
    ///          return False
    ///      qmut = get_current_qmut_instance(cpu, self.struct,
    ///                                       self.mutatefielddescr)
    ///      if qmut is not self.qmut:
    ///          return False
    ///      else:
    ///          currentbox = self.get_current_constant_fieldvalue()
    ///          assert self.constantfieldbox.same_constant(currentbox)
    ///          return True
    /// ```
    ///
    /// `struct_ptr` is `structconst.getref_base()`, already resolved by the
    /// caller's `heap.py:809 is_constant()` gate.
    ///
    /// The identity test asks the recorded instance whether it is still the
    /// field's own. A fresh instance is only minted into a field the sweep
    /// has nulled, so `is_current` is `qmut is not self.qmut`.
    ///
    /// The value comparison is upstream's closing assert
    /// (`same_constant`). Translated PyPy does not run it; a mismatch
    /// after identity matches is not `InvalidLoop`.
    fn quasiimmut_field_still_valid(
        _op: &Op,
        obj: OpRef,
        struct_ptr: u64,
        qmutdescr: &majit_ir::QuasiImmutDescr,
        ctx: &mut OptContext,
    ) -> bool {
        // `quasiimmut.py is_still_valid_for` compares the recorded gcref
        // to the box now in the optimizer. `PyObject.w_class?` uses one
        // process-global watcher (every instance shares the mutate
        // field), so the recorded struct_ptr is whichever object the
        // tracer happened to read — not the owner of the QuasiImmut.
        // A later replacement of that box (force of a virtual int)
        // must not look like the field changed. The watcher identity
        // (`is_current`) still revokes on `__class__` assignment.
        let shared_w_class_watcher = qmutdescr
            .fielddescr()
            .as_field_descr()
            .is_some_and(|f| f.is_w_class());
        if struct_ptr != qmutdescr.struct_ptr() {
            if !shared_w_class_watcher {
                return false;
            }
            // Same watcher, different object: accept only when this
            // object's class still matches the folded constant. Otherwise
            // a polymorphic replacement would keep A's class on B.
            let Some(constantfieldbox) = qmutdescr.constantfieldbox() else {
                return false;
            };
            let Some(currentbox) = ctx
                .get_runtime_field(obj, qmutdescr.fielddescr())
                .and_then(|r| r.inline_const_to_value())
            else {
                return false;
            };
            if currentbox != constantfieldbox {
                return false;
            }
        }
        // quasiimmut.py `is_still_valid_for`: `qmut is not self.qmut`
        // is unconditional False. A later wrapint New+setfield that
        // revokes the process-global watcher must InvalidLoop, not be
        // rescued by a live w_class compare.
        if !qmutdescr.qmut().is_current() {
            return false;
        }
        if let (Some(constantfieldbox), Some(currentbox)) = (
            qmutdescr.constantfieldbox(),
            ctx.get_runtime_field(obj, qmutdescr.fielddescr())
                .and_then(|r| r.inline_const_to_value()),
        ) {
            debug_assert_eq!(
                currentbox, constantfieldbox,
                "quasiimmut.py same_constant after qmut identity"
            );
        }
        true
    }

    /// Slot space for the object header words, above every position a parent's
    /// field list can hand out. Keyed by offset so `ob_type` and `w_class` stay
    /// apart. A `PtrInfo`'s field list is an association list, not a dense
    /// array, so a sparse high slot costs one entry and nothing else.
    const HEADER_FIELD_SLOT_BASE: u32 = 0x8000_0000;

    /// Slot base for a field descriptor the parent's slot list does not place
    /// (see [`Self::field_slot_index`]).  Disjoint from both the dense
    /// `index_in_parent` space and [`Self::HEADER_FIELD_SLOT_BASE`].
    const UNSLOTTED_FIELD_SLOT_BASE: u32 = 0x4000_0000;

    /// Header-band slot for a descriptor already narrowed to `FieldDescr`.
    /// Kept beside `field_slot_index` so allocation metadata and heap ops use
    /// one namespace even when Rust cannot upcast `Arc<dyn FieldDescr>` back
    /// to the owning `DescrRef` trait object.
    pub(crate) fn header_field_slot(field_descr: &dyn majit_ir::FieldDescr) -> u32 {
        debug_assert!(field_descr.is_typeptr() || field_descr.is_w_class());
        Self::HEADER_FIELD_SLOT_BASE + field_descr.offset() as u32
    }

    /// Compute the `PtrInfo._fields` slot for a field descriptor.
    ///
    /// RPython uses `descr.get_index()` only for `info._fields[index]`
    /// (`info.py`).  In majit this is `FieldDescr::index_in_parent`
    /// when a parent SizeDescr is available; older/simple descriptors fall
    /// back to their `Descr::index()`.
    ///
    /// "A parent is available" is not the same question as "this number is a
    /// position in that parent's list".  A header word carries no position at
    /// all — the positional census skips it by name (`heaptracker.py:64-66`),
    /// so the mint hands it the placeholder `0` and documents that consumers
    /// resolve it through `FieldDescr::is_w_class()` instead.  A parent that
    /// publishes an empty `all_fielddescrs` has no positions to hand out
    /// either.  Both reach here as slot `0`, and slot `0` is a REAL slot: the
    /// one the parent's first enumerated field occupies.  A `Ref` header read
    /// then answers out of the entry an `Int` field of the same object filled,
    /// which is what `make_equal_to`'s `Box.type` invariant fires on.
    ///
    /// So use the position only where the parent's list actually holds this
    /// field at it.  A field that list does not place carries no position, and
    /// the descr's own `index()` is not one either: every `VirtualizableInfo`
    /// field descriptor is minted with index `0`, so answering with it puts
    /// the vable's array-pointer read on the same slot as the vable token
    /// store, and the read comes back with the token.  Key those by offset in
    /// a band of their own instead.
    ///
    /// The header words are answered before all of that, in a band of their
    /// own.  They resolve through no field list at all -- `OptVirtualize`
    /// folds `is_typeptr` from `known_class` and `is_w_class` from the
    /// virtual's class identity -- and an offset alone would not separate a
    /// header word from an unplaced field naming the same word, so the two
    /// bands stay apart.
    pub(crate) fn field_slot_index(descr: &DescrRef) -> u32 {
        let Some(field_descr) = descr.as_field_descr() else {
            return descr.index();
        };
        if field_descr.is_typeptr() || field_descr.is_w_class() {
            return Self::header_field_slot(field_descr);
        }
        // One walk of the parent list. `parent_list_slot` already searched it;
        // re-fetching the row and cloning the `DescrRef` just to confirm the
        // same `slot_holds_field` answer is the same work twice.
        if let Some(parent) = field_descr.get_parent_descr()
            && let Some(size) = parent.as_size_descr()
        {
            let fields = size.all_fielddescrs();
            let minted = field_descr.index_in_parent();
            if fields.get(minted).is_some_and(|slot| {
                crate::optimizeopt::virtualize::slot_holds_field(slot.as_ref(), field_descr)
            }) {
                return minted as u32;
            }
            if let Some(i) = fields.iter().position(|slot| {
                crate::optimizeopt::virtualize::slot_holds_field(slot.as_ref(), field_descr)
            }) {
                return i as u32;
            }
        }
        // The parent places this field nowhere, so there is no slot number
        // for it — and `descr.index()` is not one either.  Slot numbers are
        // `index_in_parent` values, small and dense; a descriptor minted
        // without a parent slot would land on top of whatever the parent
        // really holds at that slot, and a read through one descriptor
        // would then answer with the value stored through the other.
        // Offset is the identity such a field does carry, so key it by
        // that: descriptors naming the same word alias, descriptors naming
        // different words never do.
        Self::UNSLOTTED_FIELD_SLOT_BASE + field_descr.offset() as u32
    }

    /// `effectinfo.py:529-532` `bitstrr = [descr.ei_index for descr in
    /// getattr(ei, '_readonly_descrs_' + key)]` — the bit position in
    /// each EI's `bitstring_*` is the descr's `ei_index`, set by
    /// `compute_bitstrings` (`effectinfo.py descr.ei_index = …`).
    /// `effectinfo.py descr.ei_index = sys.maxint` is the sentinel
    /// for descrs absent from any EI's raw set;
    /// `bitstring.py bitcheck if byte_number >= len(bitstring)` then makes
    /// `bitcheck` return false out of range. Pyre matches by returning
    /// `u32::MAX`, whose `byte_number = u32::MAX >> 3` is far past any
    /// realistic bitstring length so `bitcheck` shorts to false the
    /// same way.
    fn field_effect_index(descr: &DescrRef) -> u32 {
        descr.get_ei_index()
    }

    /// Same as [`field_effect_index`] for the array namespace
    /// (`effectinfo.py add_array` writes `ei_index` onto the
    /// array descr; `compute_bitstrings` re-stamps in-place per
    /// `effectinfo.py descr.ei_index = …`).
    #[allow(dead_code)]
    fn array_effect_index(descr: &DescrRef) -> u32 {
        descr.get_ei_index()
    }

    /// heapcache.py `_escape_box`: escape a box and transitively
    /// escape all its dependencies stored in `box._heapc_deps`.
    /// Const boxes (history.py) are globally-scoped values, never
    /// tracked in `unescaped` bitset.
    fn escape_box(&mut self, box_: &Operand) {
        if box_.is_constant() {
            return;
        }
        self.unescaped.swap_remove(box_);
        if let Some(deps) = self.heapc_deps.swap_remove(box_) {
            for dep in deps {
                self.escape_box(&dep);
            }
        }
    }

    /// heapcache.py `is_unescaped(box)` — `_check_flag(box,
    /// HF_IS_UNESCAPED)`. Const boxes are never `RefFrontendOp` so the
    /// flag is never set (history.py:213); `None` sentinels likewise.
    fn is_unescaped(&self, box_: &Operand) -> bool {
        if box_.is_none() || box_.is_constant() {
            return false;
        }
        self.unescaped.contains(box_)
    }

    /// heapcache.py `_escape_from_write`: when storing a value
    /// into a container, append to `_get_deps(box)` if both are
    /// unescaped; otherwise escape the value immediately.
    fn escape_from_write(&mut self, ctx: &OptContext, container: &Operand, value: &Operand) {
        // heapcache.py:224-229: if both box and fieldbox are unescaped,
        // record the dependency; otherwise escape the fieldbox (the value).
        // `is_unescaped`/`escape_box` are no-ops for Const operands, so a
        // constant value never escapes the container and a constant
        // container still escapes a non-constant value.
        //
        // heapcache operates on box objects; resolve both operands through
        // `get_box_replacement` so set membership is by box identity.
        if value.is_none() {
            return;
        }
        let value_box = ctx.resolve_operand_operand(value);
        let container_box = ctx.resolve_operand_operand(container);
        if self.is_unescaped(&container_box) && self.is_unescaped(&value_box) {
            self.heapc_deps
                .entry(container_box)
                .or_default()
                .push(value_box);
        } else {
            self.escape_box(&value_box);
        }
    }

    /// Build the field cache key from a GETFIELD or SETFIELD op.
    ///
    /// For GETFIELD_GC_I/R/F: args = [obj], descr = field descriptor.
    /// For SETFIELD_GC: args = [obj, value], descr = field descriptor.
    #[allow(dead_code)]
    fn field_key(op: &Op) -> Option<FieldKey> {
        let descr = op.getdescr()?;
        let obj = op.arg(0).to_opref();
        Some((obj, descr_ptr(&descr)))
    }

    /// heap.py arrayitem_cache: constant-index array cache key.
    /// Canonicalizes array and index through get_box_replacement.
    fn arrayitem_key(op: &Op, ctx: &mut OptContext) -> Option<ArrayItemKey> {
        let index_val = ctx
            .resolve_operand_operand_opt(&op.arg(1))
            .and_then(|b| ctx.get_constant_int_box(&b))?;
        // `heap.py` `arrayitem_cache` keys a constant index. A negative
        // index is not a cache slot; the read stays on the variable-index path.
        if index_val < 0 {
            return None;
        }
        let descr = op.getdescr()?;
        let array = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
        Some((array, descr_ptr(&descr), index_val))
    }

    /// Register a struct opref in the per-descr CachedField.
    ///
    fn cache_field(&mut self, struct_box: &Operand, descr: &DescrRef) {
        self.field_cache(descr).register_info(struct_box);
    }

    /// heap.py: field_cache (read-only borrow variant).
    #[allow(dead_code)]
    fn get_cached_field(&self, descr: &DescrRef) -> Option<&CachedField> {
        self.cached_field_pos_for_descr(descr)
            .map(|pos| &self.cached_fields[pos].2)
    }

    /// heap.py field_cache — get or create CachedField for a descr.
    fn field_cache(&mut self, descr: &DescrRef) -> &mut CachedField {
        let pos = self.ensure_cached_field(descr, None);
        &mut self.cached_fields[pos].2
    }

    /// Position of `descr` in `cached_fields`, inserting on miss.
    /// `field_idx` is computed only when the entry is created.
    fn ensure_cached_field(&mut self, descr: &DescrRef, field_idx: Option<u32>) -> usize {
        match self.cached_field_pos_for_descr(descr) {
            Some(pos) => pos,
            None => {
                let field_idx = field_idx.unwrap_or_else(|| Self::field_slot_index(descr));
                self.cached_fields
                    .push((field_idx, descr.clone(), CachedField::new()));
                self.cached_fields.len() - 1
            }
        }
    }

    #[allow(dead_code)]
    fn get_cached_field_mut(&mut self, descr: &DescrRef) -> Option<&mut CachedField> {
        let pos = self.cached_field_pos_for_descr(descr)?;
        Some(&mut self.cached_fields[pos].2)
    }

    /// heap.py arrayitem_submap(descr, create_if_nonexistant=True)
    ///
    /// `descr_idx` is the descriptor object's identity. PyPy's
    /// `cached_arrayitems[descr]` (`heap.py:399`) uses the descriptor object
    /// itself and reserves `descr.get_ei_index()` for EffectInfo bitstrings.
    fn arrayitem_submap(&mut self, descr: &DescrRef) -> &mut ArrayCacheSubMap {
        let descr_idx = descr_ptr(descr);
        let pos = match self.cached_array_pos_for_index(descr_idx) {
            Some(pos) => pos,
            None => {
                self.cached_arrayitems
                    .push((descr_idx, descr.clone(), ArrayCacheSubMap::new()));
                self.cached_arrayitems.len() - 1
            }
        };
        &mut self.cached_arrayitems[pos].2
    }

    /// heap.py arrayitem_cache(descr, index)
    /// → submap[descr].const_indexes[index] (or insert).
    fn arrayitem_cache(&mut self, descr: &DescrRef, index: i64) -> &mut ArrayCachedItem {
        self.arrayitem_submap(descr).const_get_or_new(index)
    }

    fn get_cached_array_submap(&self, descr_idx: usize) -> Option<&ArrayCacheSubMap> {
        self.cached_array_pos_for_index(descr_idx)
            .map(|pos| &self.cached_arrayitems[pos].2)
    }

    fn get_cached_array_submap_mut(&mut self, descr_idx: usize) -> Option<&mut ArrayCacheSubMap> {
        let pos = self.cached_array_pos_for_index(descr_idx)?;
        Some(&mut self.cached_arrayitems[pos].2)
    }

    #[allow(dead_code)]
    fn get_cached_array_descr(&self, descr_idx: usize) -> Option<DescrRef> {
        self.cached_array_pos_for_index(descr_idx)
            .map(|pos| self.cached_arrayitems[pos].1.clone())
    }

    fn invalidate_arrayitem_cache(&mut self, descr_idx: usize, index: i64, ctx: &mut OptContext) {
        if let Some(submap) = self.get_cached_array_submap_mut(descr_idx)
            && let Some(cai) = submap.const_get_mut(index)
        {
            cai.invalidate(ctx);
        }
    }

    fn cache_arrayitem(
        &mut self,
        array_box: &Operand,
        descr_idx: usize,
        index: i64,
        descr: Option<&DescrRef>,
    ) {
        let Some(descr) = descr else {
            if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                submap.const_get_or_new(index).register_info(array_box);
            }
            return;
        };
        let cai = self.arrayitem_cache(descr, index);
        cai.register_info(array_box);
    }

    /// heap.py: force_lazy_set — emit lazy setfields.
    /// If any lazy setfield argument references the postponed_op,
    /// emit the postponed_op first (RPython heap.py exact logic).
    /// Emit a lazy setfield after resolving forwarding and forcing a virtual
    /// rhs if needed.
    ///
    /// RPython heap.py: force_lazy_set → emit_extra(op, emit=False).
    ///
    /// force_lazy_set (heap.py) emits unconditionally; the
    /// virtual-rhs skip belongs only to force_lazy_sets_for_guard
    /// (heap.py:610-639), which routes those ops to rd_pendingfields
    /// before they ever reach this fn. A virtual rhs is left virtual
    /// here: force_lazy_set never forces its rhs, and the queued op
    /// carries it to `Optimizer::emit_operation`, whose force_box loop
    /// (optimizer.py) is the single force point that appends
    /// the materialization directly ahead of the store.
    fn emit_lazy_setfield(op: &majit_ir::OpRc, ctx: &mut OptContext) {
        // Resolve forwarding and route after heap
        // optimizer.py:651-652 setarg loop parity.
        for i in 0..op.num_args() {
            op.setarg(i, ctx.resolve_operand_operand(&op.arg(i)));
        }
        // heap.py: emit_extra(op, emit=False) → next_optimization.
        // `_lazy_set` already holds this ResOperation; do not cls() again.
        ctx.emit_extra_rc(ctx.current_pass_idx, op.clone());
    }

    /// heap.py: force_lazy_set → emit_extra(op, emit=False)
    ///
    /// For each CachedField with a pending lazy set:
    /// 1. invalidate(descr) — clear conflicting cache entries
    /// 2. emit_extra(op, emit=False) — route through passes AFTER heap
    /// 3. put_field_back_to_info — restore this specific cache entry
    ///
    /// `heap_pass_idx`: this pass's own index. RPython uses
    /// `self.next_optimization` which always starts AFTER heap.
    fn force_all_lazy_setfields(&mut self, _heap_pass_idx: usize, ctx: &mut OptContext) {
        // heap.py force_all_lazy_sets (field half) line-by-line:
        //
        //     items = self.cached_fields.items()
        //     if not we_are_translated():
        //         items.sort(...)
        //     for descr, cf in items:
        //         cf.force_lazy_set(self, descr)
        //
        // Pyre collects the sorted descrs first to avoid mutably
        // borrowing `cached_fields` and `self` simultaneously inside
        // the loop body.
        let mut order: Vec<usize> = self
            .cached_fields
            .iter()
            .enumerate()
            .filter_map(|(i, (_, _, cf))| cf.lazy_set.is_some().then_some(i))
            .collect();
        // we_are_translated() == False sort path for test stability.
        order.sort_by_key(|&i| descr_ptr(&self.cached_fields[i].1));
        for i in order {
            self.force_lazy_set_field_at(i, true, ctx);
        }
    }

    fn force_all_lazy_setarrayitems(&mut self, _heap_pass_idx: usize, ctx: &mut OptContext) {
        // heap.py force_all_lazy_sets (array half) line-by-line:
        //
        //     for descr, submap in self.cached_arrayitems.iteritems():
        //         self.force_lazy_setarrayitem_submap(submap, can_cache=True)
        //
        // force_lazy_setarrayitem_submap (heap.py) iterates
        // submap.const_indexes and calls cf.force_lazy_set per cai.
        let entries: Vec<(usize, DescrRef, Vec<i64>)> = self
            .cached_arrayitems
            .iter()
            .map(|(idx, descr, submap)| {
                let mut indexes: Vec<i64> = submap.const_keys();
                indexes.sort();
                (*idx, descr.clone(), indexes)
            })
            .collect();
        for (descr_idx, descr, indexes) in entries {
            for index in indexes {
                // heap.py cf.force_lazy_set(self, None, can_cache=True)
                self.force_lazy_set_array(descr_idx, index, true, ctx);
            }
        }
    }

    /// Force all pending lazy stores (both fields and array items).
    /// `heap_pass_idx`: this pass's own pipeline index for emit routing.
    fn force_all_lazy_sets(&mut self, heap_pass_idx: usize, ctx: &mut OptContext) {
        self.force_all_lazy_setfields(heap_pass_idx, ctx);
        self.force_all_lazy_setarrayitems(heap_pass_idx, ctx);
    }

    /// heap.py force_lazy_setarrayitem(arraydescr, indexb=None, can_cache=True)
    ///
    /// Selectively force lazy array stores for a specific array descriptor.
    /// Only entries whose const index is within `indexb` (if provided) are forced.
    /// RPython uses this for variable-index GETARRAYITEM/SETARRAYITEM to avoid
    /// forcing ALL lazy stores — only those that could alias the variable index.
    fn force_lazy_setarrayitem(
        &mut self,
        descr: &DescrRef,
        indexb: Option<&crate::optimizeopt::intutils::IntBound>,
        can_cache: bool,
        ctx: &mut OptContext,
    ) {
        let pos = match self.cached_array_pos_for_descr(descr) {
            Some(pos) => pos,
            None => return,
        };
        let descr_idx = self.cached_arrayitems[pos].0;
        let indexes: Vec<i64> = self.cached_arrayitems[pos]
            .2
            .const_indexes
            .keys()
            .copied()
            .filter(|&idx| indexb.is_none_or(|b| b.contains(idx)))
            .collect();
        for index in indexes {
            self.force_lazy_set_array(descr_idx, index, can_cache, ctx);
        }
    }

    /// heap.py `emitting_operation`'s guard case, run at heap's OWN
    /// `emit()` point (heap.py, which precedes `emit_postponed_op()` on
    /// heap.py) instead of at the deferred `emitting_operation` callback
    /// that `Optimizer::emit_operation` fires at final emission.
    ///
    /// The order matters for every guard that arrives with a postponed op
    /// pending: GUARD_NOT_FORCED after its CALL_MAY_FORCE, GUARD_[NO_]OVERFLOW
    /// after its INT_*_OVF, GUARD_TRUE / GUARD_FALSE after the comparison they
    /// test. A lazy SETFIELD_GC forced late lands between the pair, and the
    /// backend reads the second of the two out of the first
    /// (`aarch64/assembler.py:1187-1193`).
    ///
    /// The virtual-valued sets `force_lazy_sets_for_guard` hands back are put
    /// straight back into the caches, so the later `emitting_operation` call
    /// re-collects them into `ctx.pending_for_guard` and the guard's
    /// `rd_pendingfields` is unchanged.
    fn force_lazy_sets_for_guard_early(&mut self, ctx: &mut OptContext) {
        let pending_virtual = self.force_lazy_sets_for_guard(ctx.current_pass_idx, ctx);
        for pending_op in pending_virtual {
            let descr = pending_op.getdescr().unwrap();
            if pending_op.opcode == OpCode::SetarrayitemGc {
                // A non-constant index has no `arrayitem_cache` slot to hold
                // it, so it cannot be re-collected — emit it here.
                match ctx.get_constant_int_box(&pending_op.arg(1).get_box_replacement(false)) {
                    Some(index) => self.arrayitem_cache(&descr, index).lazy_set = Some(pending_op),
                    None => {
                        ctx.emit((*pending_op).clone());
                    }
                }
            } else {
                self.field_cache(&descr).lazy_set = Some(pending_op);
            }
        }
    }

    /// heap.py force_lazy_sets_for_guard()
    ///
    /// Returns pendingfields: SetfieldGc/SetarrayitemGc ops where the stored
    /// VALUE is virtual. These go into rd_pendingfields on the guard's resume
    /// data (emitting_operation stores them in ctx.pending_for_guard →
    /// optimizer.rs encodes as op.rd_pendingfields).
    /// Non-virtual lazy sets are emitted (forced) immediately.
    fn force_lazy_sets_for_guard(
        &mut self,
        self_pass_idx: usize,
        ctx: &mut OptContext,
    ) -> Vec<majit_ir::OpRc> {
        let mut pendingfields = Vec::new();

        // heap.py:610-621: iterate cached fields
        // Collect lazy sets WITHOUT consuming them: a virtual-valued
        // lazy set stays cached (heap.py:618-620 `continue` does not
        // clear `_lazy_set`), so every later guard re-collects it into
        // pendingfields and a later call/flush boundary still emits it.
        // Only force_lazy_set — the non-virtual arm — clears it.
        // heap.py:611-612 `op = cf._lazy_set; if op is None: continue` — skip
        // before touching the entry, so a guard costs one `Option` probe per
        // cached field instead of a `DescrRef` clone per cached field. Filtering
        // ahead of the sort is equivalent: the sort is stable and filtering
        // preserves relative order.
        let mut order: Vec<usize> = self
            .cached_fields
            .iter()
            .enumerate()
            .filter_map(|(i, (_, _, cf))| cf.lazy_set.is_some().then_some(i))
            .collect();
        sort_descr_entry_indices_untranslated(&self.cached_fields, &mut order);
        for i in order {
            let Some(op) = self.cached_fields[i].2.lazy_set.clone() else {
                continue;
            };
            // heap.py:617-618: val = op.getarg(1); if is_virtual(val)
            let is_virtual = ctx.is_virtual(&op.arg(1).get_box_replacement(false));
            if is_virtual {
                // heap.py:618-619: virtual value → pendingfields
                pendingfields.push(op);
                continue;
            }
            // heap.py: cf.force_lazy_set(self, descr) →
            // _lazy_set = None, invalidate, emit_extra(op, emit=False),
            // then put_field_back_to_info restores the cache.
            // optimizer.py setarg loop parity.
            for arg_i in 0..op.num_args() {
                op.setarg(arg_i, ctx.resolve_operand_operand(&op.arg(arg_i)));
            }
            let final_value = op.arg(1);
            let field_idx = self.cached_fields[i].0;
            // heap.py: invalidate(descr) — purity self-gate
            // inside CachedField::invalidate (heap.py parity).
            {
                let (idx, descr, cf) = &mut self.cached_fields[i];
                cf.lazy_set = None;
                cf.invalidate(descr, *idx, ctx);
            }
            // heap.py put_field_back_to_info needs the lazy_set Op
            // AFTER it's been emitted by emit_extra. Clone it so the
            // structinfo write goes through `structinfo_setfield` (which
            // also handles the constant arg0 → const_infos route).
            let put_back_op = op.clone();
            // emit_extra(op, emit=False): route through passes after heap.
            // RPython: self.next_optimization — always starts AFTER heap,
            // regardless of which pass emitted the guard that triggered this.
            ctx.emit_extra_rc(self_pass_idx, op);
            // heap.py: put_field_back_to_info — restore cache + PtrInfo.
            // Struct base = op.getarg(0) (args already resolved above).
            let obj_box = ctx.resolve_operand_operand(&put_back_op.arg(0));
            self.cached_fields[i].2.register_info(&obj_box);
            ctx.structinfo_setfield(&put_back_op, field_idx, &final_value);
        }

        // heap.py:622-636: iterate cached array items
        //   for descr, submap in self.cached_arrayitems.iteritems():
        //       for index, cf in submap.const_indexes.iteritems():
        let array_entries: Vec<(usize, i64, majit_ir::OpRc)> = self
            .cached_arrayitems
            .iter_mut()
            .flat_map(|(descr_idx, _, submap)| {
                submap
                    .const_indexes
                    .iter_mut()
                    .filter_map(move |(index, cai)| {
                        cai.lazy_set.clone().map(|op| (*descr_idx, *index, op))
                    })
            })
            .collect();
        for (descr_idx, index, op) in array_entries {
            // heap.py:631-633: assert container not virtual; check value virtual
            let is_virtual = ctx.is_virtual(&op.arg(2).get_box_replacement(false));
            if is_virtual {
                // heap.py:634: pendingfields.append(op)
                pendingfields.push(op);
                continue;
            }

            // heap.py cf.force_lazy_set(...): consume the lazy set
            // (the non-virtual arm is what clears it).
            if let Some(cai) = self
                .get_cached_array_submap_mut(descr_idx)
                .and_then(|s| s.const_get_mut(index))
            {
                cai.lazy_set = None;
            }
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..op.num_args() {
                op.setarg(i, ctx.resolve_operand_operand(&op.arg(i)));
            }
            let final_value = op.arg(2);
            let array_ref = op.arg(0);
            let descr = op.getdescr();
            let put_back_op = op.clone();
            // emit_extra(op, emit=False): route through passes after heap.
            ctx.emit_extra_rc(self_pass_idx, op);
            self.cache_arrayitem(&array_ref, descr_idx, index, descr.as_ref());
            // info.py: ArrayPtrInfo.setitem — keep PtrInfo in sync.
            ctx.arrayinfo_setitem(&put_back_op, index as usize, &final_value);
        }

        pendingfields
    }

    /// Invalidate caches on calls and other side-effecting operations.
    ///
    /// `OptHeap.force_from_effectinfo` invalidates non-pure field and array
    /// caches. Caches that survive are:
    /// - Immutable (green) field caches: values never change.
    /// - Unescaped object caches: calls cannot access objects that have not
    ///   been passed to a call or stored into the heap.
    ///
    /// `CachedField.invalidate` clears `opinfo._fields[idx]` for every cached
    /// info before clearing `cached_infos` and `cached_structs`. The Rust port routes that
    /// PtrInfo cleanup through `invalidate_with_ctx` so the per-pass
    /// "single source of truth" stays in sync after a clean.
    fn clean_caches(&mut self, ctx: &mut OptContext) {
        // `OptHeap.clean_caches` sorts descriptors only when untranslated, so walk the
        // cache in place through a permuted index list. Materializing the
        // entries instead cost a `DescrRef` clone per cached field on every
        // residual call, which is where clean_caches runs.
        let mut order: Vec<usize> = (0..self.cached_fields.len()).collect();
        sort_descr_entry_indices_untranslated(&self.cached_fields, &mut order);
        for i in order {
            let (field_idx, descr, cf) = &mut self.cached_fields[i];
            // `CachedField.invalidate` applies its own purity gate. The stored
            // slot index is `descr.get_index()` from insert time.
            cf.invalidate(descr, *field_idx, ctx);
        }
        // heap.py:386-389:
        //   for descr, submap in self.cached_arrayitems.iteritems():
        //       if not descr.is_always_pure():
        //           for index, cf in submap.const_indexes.iteritems():
        //               cf.invalidate(None)
        //
        // RPython's `cf.invalidate(None)` clears `cached_infos` items AND
        // calls `self.parent.clear_varindex()` (heap.py). The Rust port
        // walks `cached_arrayitems` directly so each `cai.invalidate(ctx)`
        // can drop the matching `arrayinfo._items[index]` slot through
        // `ctx.get_ptr_info_mut` / `ctx.get_const_info_mut`.
        let descr_entries: Vec<(usize, bool)> = self
            .cached_arrayitems
            .iter()
            .map(|(idx, descr, _)| (*idx, descr.is_always_pure()))
            .collect();
        for (descr_idx, is_pure) in descr_entries {
            if is_pure {
                continue;
            }
            let indexes: Vec<i64> = match self.get_cached_array_submap(descr_idx) {
                Some(submap) => submap.const_keys(),
                None => continue,
            };
            for index in indexes {
                if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                    if let Some(cai) = submap.const_get_mut(index) {
                        cai.invalidate(ctx);
                    }
                    // heap.py self.parent.clear_varindex()
                    submap.clear_varindex();
                }
            }
        }
        // heap.py:390: self.cached_dict_reads.clear()
        self.cached_dict_reads.clear();
    }

    /// Extract OopSpecIndex from a call op's descriptor, if available.
    fn get_oopspecindex(op: &Op) -> OopSpecIndex {
        op.with_call_descr(|cd| cd.get_extra_info().oopspecindex)
            .unwrap_or(OopSpecIndex::None)
    }

    /// heap.py _optimize_CALL_DICT_LOOKUP.
    ///
    /// Cache consecutive dict lookup calls on the same dict+key.
    /// FLAG_LOOKUP (0): always cache and reuse.
    /// FLAG_STORE  (1): don't cache new; reuse only if cached value ≥ 0.
    /// FLAG_DELETE (2+): never cache, never reuse.
    fn _optimize_call_dict_lookup(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> bool {
        const FLAG_LOOKUP: i64 = 0;
        const FLAG_STORE: i64 = 1;

        // heap.py:497-500: flag_value = self.getintbound(op.getarg(4))
        //   if not flag_value.is_constant(): return False
        //   flag = flag_value.get_constant_int()
        if op.num_args() < 5 {
            return false;
        }
        let flag = match ctx.get_constant_int_or_bound_box(&op.arg(4).get_box_replacement(false)) {
            Some(v) => v,
            None => return false,
        };
        if flag != FLAG_LOOKUP && flag != FLAG_STORE {
            return false;
        }

        // heap.py:504: descrs = op.getdescr().get_extra_info().extradescrs
        let extradescrs = op
            .with_call_descr(|cd| cd.get_extra_info().extradescrs.clone())
            .flatten();
        let descrs = match extradescrs {
            Some(ref d) if d.len() >= 2 => d,
            _ => return false,
        };
        let descr1_id = descr_identity(&descrs[0]);
        let descr2 = descrs[1].clone();

        // heap.py:506-511 try/except KeyError:
        //   try:
        //       d = self.cached_dict_reads[descr1]
        //   except KeyError:
        //       d = self.cached_dict_reads[descr1] = args_dict()
        //       self.corresponding_array_descrs[descrs[1]] = descr1
        // The corresponding_array_descrs registration fires ONLY on the
        // first encounter of `descr1` (the KeyError arm); repeated lookups
        // skip it. Mirror this — `or_default()` would unconditionally
        // re-register and diverge from RPython.
        if !self.cached_dict_reads.contains_key(&descr1_id) {
            self.cached_dict_reads
                .insert(descr1_id, crate::FxIndexMap::default());
            self.corresponding_array_descrs
                .insert(descr_identity(&descr2), (descr2, descr1_id));
        }
        let d = self
            .cached_dict_reads
            .get_mut(&descr1_id)
            .expect("just inserted above when absent");

        // heap.py:513-514: key = [get_box_replacement(arg1), get_box_replacement(arg2)]
        // util.py/127 args_dict() compares args via same_box: identity for
        // non-Const, value-equality for Const (history.py). Encode each
        // arg through DictArgKey so two ConstInt slots with the same value
        // hash and compare equal.
        let key = [
            DictArgKey::from_arg(&op.arg(1), ctx),
            DictArgKey::from_arg(&op.arg(2), ctx),
        ];

        if let Some(res_v) = d.get(&key).cloned() {
            // heap.py:523-525: flag != FLAG_LOOKUP → self.getintbound(res_v).known_ge_const(0)
            if flag != FLAG_LOOKUP {
                let known_ge_zero = ctx
                    .resolve_operand_operand_opt(&res_v)
                    .and_then(|b| ctx.peek_intbound_box(&b))
                    .is_some_and(|b| b.known_ge_const(0));
                if !known_ge_zero {
                    return false;
                }
            }
            // heap.py:525-527: make_equal_to + last_emitted_operation = REMOVED
            let b_old = Operand::from_bound_op(op_rc);
            let b_res = ctx.resolve_operand_operand(&res_v);
            ctx.make_equal_to(&b_old, &b_res);
            self.last_emitted_removed = true;
            return true;
        }

        // heap.py:517-518: no hit — cache if FLAG_LOOKUP
        if flag == FLAG_LOOKUP {
            d.insert(key, Operand::from_bound_op(op_rc));
        }
        false
    }

    /// heap.py self.emit(op) → emitting_operation for is_call(op).
    /// Generic residual-call emit path: mark args escaped, then route
    /// through force_from_effectinfo / clean_caches / invalidate_for_escaped
    /// per the same descr-aware policy as the catch-all `_` arm.
    fn emit_residual_call(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let escaped_owners = self.call_argument_owner_closure(op, ctx);
        self.mark_escaped_varargs(op, ctx);
        // STRUCTURAL ADAPTATION: RPython heap.py relies purely on EffectInfo
        // here. Removing this pyre-specific direct-argument flush currently
        // breaks synthetic correctness (`comprehensions.py`, and then wider
        // dynasm/cranelift wrong-output failures). Keep it until the caller
        // materialization path is ported enough that direct call arguments
        // cannot observe stale lazy stores.
        self.force_call_argument_lazy_sets(&escaped_owners, ctx.current_pass_idx, ctx);
        // heapcache.py clear_caches_varargs.
        // Plain residual calls preserve cache entries for unescaped
        // allocations. Calls with explicit EffectInfo keep the more
        // precise heap.py force_from_effectinfo path.
        if !op.has_descr() || Self::call_has_random_effects(op) {
            self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
            self.clean_caches(ctx);
        } else {
            // heap.py force_from_effectinfo — selective cache
            // invalidation using EffectInfo bitstrings.
            self.force_from_effectinfo(op, ctx);
        }
        if Self::call_can_invalidate(op) {
            self.seen_guard_not_invalidated = false;
        }
        OptimizationResult::PassOn
    }

    /// Mark call arguments as escaped.
    /// heapcache.py mark_escaped_varargs parity.
    /// ARRAYCOPY/ARRAYMOVE with constant indices and known array descriptor
    /// do NOT escape arguments.
    fn mark_escaped_varargs(&mut self, op: &Op, ctx: &mut OptContext) {
        let oopspec = Self::get_oopspecindex(op);
        // heapcache.py:277 `descr.get_extra_info().single_write_descr_array
        // is not None`. The check predicates on the dedicated
        // `single_write_descr_array` field, NOT on the `bitstring_write*`
        // bit-count — `effectinfo.py:201-206 set_single_write_descr_array`
        // populates the field in addition to flipping the array bit, and
        // heapcache reads back the field directly.
        let has_single_write_descr = op
            .with_call_descr(|cd| cd.get_extra_info().single_write_descr_array.is_some())
            .unwrap_or(false);
        // heapcache.py:274-276 tests `isinstance(argboxes[3], ConstInt)` on the
        // raw box: it runs in the tracing-layer heapcache where operands are not
        // yet forwarded, so a constant index already IS a ConstInt. pyre fuses
        // this check into the optimizer layer (heap.rs), where an operand may be
        // a bound op forwarding to a const, so the same question — is the index a
        // known constant — must resolve the forwarding via get_box_replacement
        // before reading const_int.
        if oopspec == OopSpecIndex::Arraycopy
            && has_single_write_descr
            && op.num_args() >= 6
            && op.arg(3).get_box_replacement(false).const_int().is_some()
            && op.arg(4).get_box_replacement(false).const_int().is_some()
            && op.arg(5).get_box_replacement(false).const_int().is_some()
        {
            return;
        }
        if oopspec == OopSpecIndex::Arraymove
            && has_single_write_descr
            && op.num_args() >= 5
            && op.arg(2).get_box_replacement(false).const_int().is_some()
            && op.arg(3).get_box_replacement(false).const_int().is_some()
            && op.arg(4).get_box_replacement(false).const_int().is_some()
        {
            return;
        }
        for i in 0..op.num_args() {
            if let Some(arg_box) = ctx.resolve_operand_operand_opt(&op.arg(i)) {
                self.escape_box(&arg_box);
            }
        }
    }

    fn call_argument_owner_closure(&self, op: &Op, ctx: &OptContext) -> Vec<OpRef> {
        let mut owners = Vec::new();
        let mut stack: Vec<Operand> = op
            .args_slice()
            .iter()
            .map(|arg| arg.get_box_replacement(false))
            .collect();
        while let Some(owner_op) = stack.pop() {
            // heap.py:173-174 resolves the owner through `get_box_replacement`
            // at every use. A dependency recorded in `heapc_deps` carries the
            // box that was canonical at escape time; its producer can be
            // forwarded afterwards (`make_equal_to`), so re-resolve here before
            // the dedup and the owner record, mirroring the canonical seed args.
            // The recorded owner list is still `OpRef` positions (`contains`).
            let owner_box = ctx.resolve_operand_operand(&owner_op);
            let owner = owner_box.to_opref();
            if owners.contains(&owner) {
                continue;
            }
            owners.push(owner);
            if let Some(deps) = self.heapc_deps.get(&owner_box) {
                stack.extend(deps.iter().cloned());
            }
        }
        owners
    }

    fn emit_postponed_if_referenced(
        &mut self,
        op: &Op,
        heap_pass_idx: usize,
        ctx: &mut OptContext,
    ) {
        let needs_postponed = self.postponed_op.as_ref().is_some_and(|postponed| {
            op.args_slice()
                .iter()
                .any(|arg| arg.to_opref() == postponed.pos().get())
        });
        if needs_postponed && let Some(p) = self.postponed_op.take() {
            ctx.emit_extra(heap_pass_idx, p);
        }
    }

    /// Flush lazy stores for objects observable by a residual call: the
    /// objects that escape as direct arguments (and their transitive
    /// dependency closure), plus any trace-allocated object that has already
    /// escaped. EffectInfo bitstrings describe global heap effects, but a
    /// callee can still read fields from objects it receives explicitly or
    /// reaches through the escaped heap.
    ///
    /// Keep this selective: a lazy store on an object that is still UNESCAPED,
    /// or one this trace never allocated (e.g. an inputarg carrying
    /// loop-invariant state), must stay pending until the regular guard/JUMP
    /// flush. A trace-allocated object that has escaped, however, is now
    /// globally reachable — a residual call can observe its fields, so its
    /// pending store must be materialized here (the role
    /// `force_from_effectinfo` fills in RPython once the EffectInfo path is
    /// fully ported). Storing such a value into an already-escaped container
    /// (e.g. an int box into a forced array) escapes it via
    /// `escape_from_write` WITHOUT recording it in the container's dependency
    /// list, so the argument closure alone does not reach it.
    ///
    /// Do not remove as a cosmetic PyPy-parity cleanup. The direct parity
    /// change was tested with `python3 pyre/check.py --synthetic-only` and
    /// caused correctness failures.
    fn force_call_argument_lazy_sets(
        &mut self,
        escaped_owners: &[OpRef],
        heap_pass_idx: usize,
        ctx: &mut OptContext,
    ) {
        // Owners whose pending lazy stores this residual call can observe:
        // the escaped-argument closure plus any trace-allocated object that
        // has already escaped. Built before the mutable cache walk below so
        // the `is_unescaped`/`seen_allocation` queries do not clash with the
        // `cached_fields`/`cached_arrayitems` borrow.
        let mut flush_owners: Vec<OpRef> = escaped_owners.to_vec();
        {
            let mut consider = |owner_arg: Operand| {
                let owner_box = ctx.resolve_operand_operand(&owner_arg);
                let owner = owner_box.to_opref();
                if flush_owners.contains(&owner) {
                    return;
                }
                // `seen_allocation` is keyed by real SSA-op positions.  A
                // forwarded lazy-store owner may instead be an inline Const
                // (`ConstPtr` after a raising/non-raising path merge); like
                // RPython's Const box it cannot be a trace allocation, and
                // it has no raw position to query.
                let allocated_here =
                    !owner.is_constant() && self.seen_allocation.contains(owner.raw() as usize);
                let escaped = !self.is_unescaped(&owner_box);
                if allocated_here && escaped {
                    flush_owners.push(owner);
                }
            };
            for (_field_idx, _descr, cf) in self.cached_fields.iter() {
                if let Some(lazy_op) = cf.lazy_set.as_ref() {
                    consider(lazy_op.arg(0));
                }
            }
            for (_descr_idx, _, submap) in self.cached_arrayitems.iter() {
                for (_index, cai) in submap.const_indexes.iter() {
                    if let Some(lazy_op) = cai.lazy_set.as_ref() {
                        consider(lazy_op.arg(0));
                    }
                }
            }
        }
        let escaped_owners: &[OpRef] = &flush_owners;

        // Same `_lazy_set is None` skip as force_lazy_sets_for_guard: only the
        // entries that carry a lazy set can reach the arm below, so nothing
        // else needs its `DescrRef` cloned.
        let mut order: Vec<usize> = self
            .cached_fields
            .iter()
            .enumerate()
            .filter_map(|(i, (_, _, cf))| cf.lazy_set.is_some().then_some(i))
            .collect();
        sort_descr_entry_indices_untranslated(&self.cached_fields, &mut order);
        let pending_fields: Vec<(usize, u32, Operand, majit_ir::OpRc)> = order
            .into_iter()
            .filter_map(|i| {
                let lazy_op = self.cached_fields[i].2.lazy_set.as_ref()?;
                let owner_arg = lazy_op.arg(0);
                let owner = ctx.resolve_operand_operand(&owner_arg).to_opref();
                if escaped_owners.contains(&owner) {
                    let field_idx = self.cached_fields[i].0;
                    self.cached_fields[i]
                        .2
                        .lazy_set
                        .take()
                        .map(|op| (i, field_idx, owner_arg, op))
                } else {
                    None
                }
            })
            .collect();

        for (pos, field_idx, obj, pending_op) in pending_fields {
            // heap.py invalidate(descr) — purity self-gate
            // inside the method.
            {
                let (idx, descr, cf) = &mut self.cached_fields[pos];
                cf.invalidate(descr, *idx, ctx);
            }
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..pending_op.num_args() {
                pending_op.setarg(i, ctx.resolve_operand_operand(&pending_op.arg(i)));
            }
            self.emit_postponed_if_referenced(&pending_op, heap_pass_idx, ctx);
            let final_value = pending_op.arg(1);
            let put_back_op = pending_op.clone();
            ctx.emit_extra_rc(heap_pass_idx, pending_op);
            let obj_box = ctx.resolve_operand_operand(&obj);
            self.cached_fields[pos].2.register_info(&obj_box);
            ctx.structinfo_setfield(&put_back_op, field_idx, &final_value);
        }

        let mut pending_arrays = Vec::new();
        for (descr_idx, _, submap) in &mut self.cached_arrayitems {
            let mut index_entries: Vec<_> = submap
                .const_indexes
                .iter_mut()
                .map(|(index, cai)| (*index, cai))
                .collect();
            sort_array_index_entries_untranslated(&mut index_entries);
            for (index, cai) in index_entries {
                if cai.lazy_set.as_ref().is_some_and(|op| {
                    let owner = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
                    escaped_owners.contains(&owner)
                }) && let Some(op) = cai.lazy_set.take()
                {
                    let owner = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
                    pending_arrays.push((*descr_idx, index, owner, op));
                }
            }
        }
        for (descr_idx, index, _obj, pending_op) in pending_arrays {
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..pending_op.num_args() {
                pending_op.setarg(i, ctx.resolve_operand_operand(&pending_op.arg(i)));
            }
            self.invalidate_arrayitem_cache(descr_idx, index, ctx);
            self.emit_postponed_if_referenced(&pending_op, heap_pass_idx, ctx);
            let final_value = pending_op.arg(2);
            let array_ref = pending_op.arg(0);
            let descr = pending_op.getdescr();
            let put_back_op = pending_op.clone();
            ctx.emit_extra_rc(heap_pass_idx, pending_op);
            self.cache_arrayitem(&array_ref, descr_idx, index, descr.as_ref());
            ctx.arrayinfo_setitem(&put_back_op, index as usize, &final_value);
        }
    }

    /// heap.py: check if a call has random effects (EffectInfo).
    /// Calls with HAS_RANDOM_EFFECTS invalidate all caches.
    /// Calls without it only invalidate non-immutable/non-unescaped entries.
    fn call_has_random_effects(op: &Op) -> bool {
        op.with_call_descr(|cd| cd.get_extra_info().has_random_effects())
            .unwrap_or(true) // conservative: assume random effects if unknown
    }

    /// heap.py: check if a call can invalidate quasi-immutable fields.
    fn call_can_invalidate(op: &Op) -> bool {
        op.with_call_descr(|cd| cd.get_extra_info().check_can_invalidate())
            .unwrap_or(true)
    }

    /// heap.py: check if a call forces virtual/virtualizable objects.
    #[allow(dead_code)]
    fn call_forces_virtual(op: &Op) -> bool {
        op.with_call_descr(|cd| cd.get_extra_info().check_forces_virtual_or_virtualizable())
            .unwrap_or(false)
    }

    /// heap.py: force_from_effectinfo(effectinfo)
    ///
    /// Selective cache invalidation based on EffectInfo bitstrings.
    /// Instead of invalidating all caches, only force/invalidate
    /// fields and arrays that the call may read or write.
    fn force_from_effectinfo(&mut self, op: &Op, ctx: &mut OptContext) {
        // heapcache.py mark_escaped_varargs: escape call arguments first
        self.mark_escaped_varargs(op, ctx);

        let __descr_arc_ei = op.getdescr();
        let ei = match __descr_arc_ei.as_ref().and_then(|d| d.as_call_descr()) {
            Some(cd) => cd.get_extra_info().clone(),
            None => {
                self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                self.clean_caches(ctx);
                return;
            }
        };

        // RPython effectinfo.py: zero bitstrings mean the call touches NO
        // tracked heap fields (e.g., I/O). Only fall back to conservative
        // invalidation for calls with ForcesVirtual/RandomEffects.
        // Note: has_random_effects() calls are filtered BEFORE reaching this
        // function (emitting_operation line 460: !has_random_effects → return).
        // No special handling needed here.

        // heap.py force_from_effectinfo field/array loops.
        // Each descr is checked for readonly and write effects.
        // cf.force_lazy_set(optheap, descr, can_cache) is the core:
        //   can_cache=True  (readonly): invalidate → emit → put_field_back_to_info
        //   can_cache=False (write):    invalidate → emit → return (no put_back)
        //                               if no lazy_set: just invalidate
        // heap.py: invalidate checks descr.is_always_pure()

        // heap.py:542-552: for fielddescr, cf in self.cached_fields.items()
        let n_fields = self.cached_fields.len();
        for i in 0..n_fields {
            let effect_idx = Self::field_effect_index(&self.cached_fields[i].1);
            if ei.check_readonly_descr_field(effect_idx) {
                // heap.py cf.force_lazy_set(self, fielddescr)
                // [can_cache=True].
                self.force_lazy_set_field_at(i, true, ctx);
            }
            // Raw-set fallback for the macro / `JitDriver` path where
            // `compute_bitstrings` never runs (so `effect_idx` is the unset
            // u32::MAX sentinel and the bitstring lookup always misses):
            // invalidate when the call's concrete `_write_descrs_fields` names
            // this exact field descr by pointer identity.  Gated on
            // `!compute_bitstrings_has_run()` so the translated interpreter's
            // bitstring path is unchanged.
            //
            // A descr whose `ei_index` is still the sentinel is NOT special-
            // cased here, matching `heap.py` `force_from_effectinfo`. Upstream leaves the
            // sentinel on any descr that no EffectInfo's raw set names —
            // `effectinfo.py` stamps `sys.maxint` on every descr in
            // `all_descrs` and only `:524-526` renumbers the ones that appear
            // in some `_readonly_descrs_*` / `_write_descrs_*`, which
            // `test_effectinfo.py` pins with `f3descr`. A sentinel
            // descr therefore bitchecks false upstream too. That is sound
            // because of the OTHER half of the contract: a call whose write
            // set was never computed is `EF_RANDOM_EFFECTS`
            // (`graphanalyze.py top_result()` →
            // `effectinfo.py`) and never reaches this function at all
            // — `heap.py` routes it to `clean_caches`. pyre upholds the
            // same contract at `call_descr.rs default_effect_info()`; keeping
            // a descr-side guard here instead would be papering over any call
            // that still asserts an empty write set it did not earn.
            let writes_field = ei.check_write_descr_field(effect_idx)
                || (!majit_ir::effectinfo::compute_bitstrings_has_run()
                    && ei.writes_field_descr_by_identity(&self.cached_fields[i].1));
            if writes_field {
                // heap.py cf.force_lazy_set(self, fielddescr,
                //                                   can_cache=False).
                let is_pure = self.cached_fields[i].1.is_always_pure();
                let did = descr_ptr(&self.cached_fields[i].1);
                self.force_lazy_set_field_at(i, false, ctx);
                // heap.py:547-552 del self.cached_dict_reads[fielddescr]
                if !is_pure {
                    self.cached_dict_reads.swap_remove(&did);
                }
            }
        }

        // heap.py:554-558: for arraydescr, submap in self.cached_arrayitems.items()
        // Bitstring bit position resolves through `descr.get_ei_index()`
        // directly; `effectinfo.py:526 descr.ei_index = …` stamps the
        // slot in-place at `compute_bitstrings` time, and
        // `effectinfo.py descr.ei_index = sys.maxint` is the
        // sentinel for descrs absent from any EI's raw set.
        let n_arrays = self.cached_arrayitems.len();
        for i in 0..n_arrays {
            let descr_idx = self.cached_arrayitems[i].0;
            let effect_idx = self.cached_arrayitems[i].1.get_ei_index();
            let read = ei.check_readonly_descr_array(effect_idx);
            // heap.py:556 — no sentinel special case: a descr whose
            // `ei_index` is still the sentinel bitchecks false upstream too,
            // and a call with no computed write set is `EF_RANDOM_EFFECTS`
            // and never reaches here (spelled out in the `cached_fields`
            // loop above).
            //
            // Raw-set fallback, the array twin of the field loop's: on the
            // macro / `JitDriver` path `compute_bitstrings` never runs, so
            // `effect_idx` is the unset sentinel and the bitstring lookup
            // always misses. Invalidate when the call's concrete
            // `_write_descrs_arrays` names this exact array descr by pointer
            // identity. Gated on `!compute_bitstrings_has_run()` so the
            // translated interpreter's bitstring path is unchanged.
            //
            // The soundness argument is the field loop's verbatim: a call
            // whose write set was never computed is `EF_RANDOM_EFFECTS` and
            // is routed to `clean_caches` before reaching this function, so a
            // sentinel descr bitchecking false here is correct.
            let write = ei.check_write_descr_array(effect_idx)
                || (!majit_ir::effectinfo::compute_bitstrings_has_run()
                    && ei.writes_array_descr_by_shape(&self.cached_arrayitems[i].1));
            if !read && !write {
                continue;
            }
            let indexes: Vec<i64> = match self.get_cached_array_submap(descr_idx) {
                Some(submap) => submap.const_keys(),
                None => continue,
            };
            for index in indexes {
                if read {
                    // heap.py force_lazy_setarrayitem_submap(submap)
                    // [can_cache=True] → cf.force_lazy_set per index.
                    self.force_lazy_set_array(descr_idx, index, true, ctx);
                }
                if write {
                    // heap.py force_lazy_setarrayitem_submap(submap,
                    // can_cache=False) → cf.force_lazy_set per index.
                    self.force_lazy_set_array(descr_idx, index, false, ctx);
                }
            }
            if write {
                // heap.py force_lazy_setarrayitem_submap explicit
                // `if not can_cache: submap.clear_varindex()` — pyre
                // already calls invalidate_index per cai inside
                // force_lazy_set_array, which clears varindex via
                // submap.invalidate_index → submap.clear_varindex; this
                // outer clear_varindex is a defensive sweep for the
                // entire submap matching the upstream's batched call.
                if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                    submap.clear_varindex();
                }
            }
        }

        // heap.py: invalidate cached_dict_reads via corresponding_array_descrs.
        // PyPy `effectinfo.check_write_descr_array(arraydescr)` reads
        // `arraydescr.ei_index` (`effectinfo.py`); pyre's lift
        // resolves `descr.get_ei_index()` directly per
        // `effectinfo.py descr.ei_index = …` in-place stamp.
        let array_ids_to_clear: Vec<usize> = self
            .corresponding_array_descrs
            .iter()
            .filter_map(|(_, (arr_descr, dict_id))| {
                let effect_idx = arr_descr.get_ei_index();
                if ei.check_write_descr_array(effect_idx) {
                    Some(*dict_id)
                } else {
                    None
                }
            })
            .collect();
        for dict_id in array_ids_to_clear {
            self.cached_dict_reads.swap_remove(&dict_id);
        }

        // heap.py OptHeap.force_from_effectinfo:
        //     if effectinfo.check_forces_virtual_or_virtualizable():
        //         vrefinfo = self.optimizer.metainterp_sd.virtualref_info
        //         self.force_lazy_set(vrefinfo.descr_forced)
        // Only the `forced` field is a GC pointer that must become visible to
        // the forcing residual; the token and virtual fields are not flushed.
        if ei.check_forces_virtual_or_virtualizable() {
            let descr_forced = self.vrefinfo.descr_forced.clone();
            self.force_lazy_set_field(&descr_forced, true, ctx);
        }
    }

    // ── Handlers for specific opcodes ──

    fn optimize_getfield(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let Some(descr) = op.getdescr() else {
            return OptimizationResult::PassOn;
        };

        // heap.py `optimize_GETFIELD_GC_I`: constant_fold only when
        // `descr.is_always_pure()` and arg0 is a constant box.
        // Quasi-immutability is recorded by `optimize_QUASIIMMUT_FIELD`,
        // not by widening this purity test.
        if descr.is_always_pure()
            && ctx
                .get_constant_box(&op.arg(0).get_box_replacement(false))
                .is_some()
            && let Some(value) = ctx.constant_fold(op)
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, value);
            return OptimizationResult::Remove;
        }

        let field_idx = Self::field_slot_index(&descr);
        let descr_id = descr_ptr(&descr);
        let raw_obj = op.arg(0).to_opref();
        // heap.py:645
        //     structinfo = self.ensure_ptr_info_arg0(op)
        //
        // PyPy passes `structinfo` directly to `cf.getfield_from_cache`,
        // which uses Python object identity. The Rust port's
        // `cached_fields` map is keyed by `OpRef` instead, so we resolve
        // arg0 once and then call `ensure_ptr_info_arg0` purely for its
        // side-effect of installing a `PtrInfo` slot on `box._forwarded`.
        // Subsequent passes (intbounds, virtualstate) and the local
        // `setfield` mutation point all read/write that slot via the
        // canonical OpRef.
        let obj_box = ctx.resolve_operand_operand(&op.arg(0));
        let _ = ctx.ensure_ptr_info_arg0(op);
        // heap.py `cf = self.field_cache(descr)` — get-or-CREATE
        // (heap.py:392-397), so the aliasing check and the cache consult below
        // run even for a descr this pass has not touched yet.
        //
        // That matters at the head of the peeled loop, where every
        // `cached_fields` entry starts empty while the imported short box's
        // `PreambleOp` already sits on the STRUCT's ptr_info (`_getfield` reads
        // `info.getfield(field_idx)`). Looking the cache up read-only skipped
        // the `FieldEntry::Preamble` arm for every first-touch field, so
        // `force_op_from_preamble_op` — and with it `use_box`, which appends the
        // re-executed op to the short preamble — never ran: `used_boxes` and
        // the short jump args stayed empty and the peeled LABEL never grew the
        // hoisted field slots.
        let pos = self.ensure_cached_field(&descr, Some(field_idx));
        let mut force_lazy = false;
        let mut must_alias_cached: Option<Operand> = None;
        let mut cached_entry: Option<crate::optimizeopt::info::FieldEntry> = None;
        {
            let cf = &self.cached_fields[pos].2;
            if let Some(lazy_op) = &cf.lazy_set {
                let lazy_struct = lazy_op.arg(0);
                // heap.py possible_aliasing_two_infos:
                // opinfo1.same_info(opinfo2) → MUST_ALIAS.
                if same_ptr_info(ctx, &lazy_struct, &obj_box) {
                    // MUST_ALIAS: lazy_set targets the same struct → return rhs
                    must_alias_cached = Some(lazy_op.arg(1));
                } else {
                    // heap.py possible_aliasing_two_infos:
                    //     if opinfo1.same_info(opinfo2): return MUST_ALIAS
                    //     if self._cannot_alias_via_constants(...): return CANNOT_ALIAS
                    //     if cf._cannot_alias_via_classes_or_lengths(...): return CANNOT_ALIAS
                    //     if cf._cannot_alias_via_content(...): return CANNOT_ALIAS
                    //     return UNKNOWN_ALIAS
                    let cannot_alias = cannot_alias_via_constants(ctx, &lazy_struct, &obj_box)
                        || CachedField::_cannot_alias_via_classes_or_lengths(
                            &lazy_struct,
                            &obj_box,
                            ctx,
                        )
                        || CachedField::_cannot_alias_via_content(&lazy_struct, &obj_box, ctx);
                    if !cannot_alias {
                        // UNKNOWN_ALIAS → force_lazy_set, return None (cache miss)
                        force_lazy = true;
                    }
                    // CANNOT_ALIAS: fall through to _getfield below (heap.py)
                }
            }
            // heap.py:117-120: always check cache entries after alias analysis.
            // RPython falls through here even when lazy_set exists (CANNOT_ALIAS).
            if must_alias_cached.is_none()
                && !force_lazy
                && let Some(entry) = cf._getfield(&obj_box, &descr, field_idx, ctx)
            {
                cached_entry = Some(entry);
            }
        }
        if let Some(cached) = must_alias_cached {
            let b_old = Operand::from_bound_op(op_rc);
            let b_cached = ctx.resolve_operand_operand(&cached);
            ctx.make_equal_to(&b_old, &b_cached);
            return OptimizationResult::Remove;
        }
        if let Some(entry) = cached_entry {
            // heap.py:182-186: isinstance(res, PreambleOp)
            match entry.kind() {
                crate::optimizeopt::info::FieldEntryKind::Preamble(pop) => {
                    // heap.py:185-186:
                    //     res = optheap.optimizer.force_op_from_preamble(res)
                    //     opinfo.setfield(descr, None, res, optheap=optheap)
                    // Force first (use_box / potential_extra_ops side
                    // effects) and store the returned `preamble_op.op`,
                    // then walk forwarding for the body replace.
                    let cached = ctx.force_op_from_preamble_op(pop);
                    ctx.structinfo_setfield(op, field_idx, &cached);
                    let obj_box = ctx.resolve_operand_operand(&obj_box);
                    self.cached_fields[pos].2.register_info(&obj_box);
                    let b_old = Operand::from_bound_op(op_rc);
                    let b_cached = ctx.resolve_operand_operand(&cached);
                    ctx.make_equal_to(&b_old, &b_cached);
                    return OptimizationResult::Remove;
                }
                crate::optimizeopt::info::FieldEntryKind::Value(cached) => {
                    if !cached.is_none() {
                        // heap.py `CachedField._getfield` asserts that an
                        // AbstractStructPtrInfo supplying a cached value
                        // is present in `cached_infos`.  Allocation-owned
                        // header facts start on PtrInfo before this
                        // per-descr cache sees them, so enlist the object
                        // now; later calls/writes can then invalidate the
                        // mutable public class word normally.
                        let obj_box = ctx.resolve_operand_operand(&obj_box);
                        self.cached_fields[pos].2.register_info(&obj_box);
                        let b_old = Operand::from_bound_op(op_rc);
                        let b_cached = ctx.resolve_operand_operand(&cached);
                        ctx.make_equal_to(&b_old, &b_cached);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        // heap.py: UNKNOWN_ALIAS → force lazy_set and return cache miss
        // heap.py: force_lazy_set(can_cache=True) — reads don't destroy cache,
        // so put_field_back_to_info restores the lazy value into the cache.
        // (Contrast with write-descr force in force_from_effectinfo which uses
        // can_cache=False and does NOT restore the value.)
        if force_lazy {
            let lazy_data = self.cached_fields[pos].2.lazy_set.take();
            if let Some(mut lazy_op) = lazy_data {
                // heap.py invalidate(descr) — purity self-gate
                // inside the method.
                {
                    let (idx, d, cf) = &mut self.cached_fields[pos];
                    cf.invalidate(d, *idx, ctx);
                }
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos().get();
                    if lazy_op.args_slice().iter().any(|a| a.to_opref() == ppos)
                        && let Some(p) = self.postponed_op.take()
                    {
                        ctx.emit_extra(ctx.current_pass_idx, p);
                    }
                }
                Self::emit_lazy_setfield(&lazy_op, ctx);
                // can_cache=True: put_field_back_to_info
                let final_value = lazy_op.arg(1);
                let lazy_obj_box = ctx.resolve_operand_operand(&lazy_op.arg(0));
                self.cached_fields[pos].2.register_info(&lazy_obj_box);
                // heap.py (force_lazy_set → put_field_back_to_info):
                //     opinfo.setfield(...) on the structinfo of lazy_obj.
                // Routes constants through `const_infos` per
                // `info.py ConstPtrInfo.setfield`.
                ctx.structinfo_setfield(&lazy_op, field_idx, &final_value);
            }
            // Cache miss — fall through to emit the getfield
        }

        // Virtualizable fields are loop-variant; skip caching/import.
        let is_vable_field = descr.is_virtualizable();

        // RPython parity: PreambleOp detection is now inline in _getfield →
        // FieldEntry::Preamble (heap.py). The getfield_from_cache path
        // above handles both Value and Preamble entries uniformly.

        // Check read cache (after import).
        if let Some(entry) = self.cached_fields[pos]
            .2
            ._getfield(&obj_box, &descr, field_idx, ctx)
        {
            match entry.kind() {
                crate::optimizeopt::info::FieldEntryKind::Preamble(pop) => {
                    // heap.py:185-186: force the preamble op, then write the
                    // forced value back into the struct info.
                    let cached = ctx.force_op_from_preamble_op(pop);
                    ctx.structinfo_setfield(op, field_idx, &cached);
                    let obj_box = ctx.resolve_operand_operand(&obj_box);
                    self.cached_fields[pos].2.register_info(&obj_box);
                    let b_old = Operand::from_bound_op(op_rc);
                    let b_cached = ctx.resolve_operand_operand(&cached);
                    ctx.make_equal_to(&b_old, &b_cached);
                    return OptimizationResult::Remove;
                }
                crate::optimizeopt::info::FieldEntryKind::Value(cached) => {
                    if !cached.is_none() {
                        let obj_box = ctx.resolve_operand_operand(&obj_box);
                        self.cached_fields[pos].2.register_info(&obj_box);
                        let b_old = Operand::from_bound_op(op_rc);
                        let b_cached = ctx.resolve_operand_operand(&cached);
                        ctx.make_equal_to(&b_old, &b_cached);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        // Check quasi-immutable cache: if this field was marked by
        // QUASIIMMUT_FIELD, the value is stable (guarded by GUARD_NOT_INVALIDATED).
        // Keyed by the object's canonical box identity.
        {
            let qi_obj = ctx.resolve_operand_operand(&obj_box);
            let qi_key = (qi_obj, descr_id);
            if let Some(qi_cached) = self.quasi_immut_cache.get(&qi_key).copied() {
                if !qi_cached.is_none() {
                    // Subsequent read: reuse the cached value.
                    let b_old = Operand::from_bound_op(op_rc);
                    let b_qi = ctx.get_box_replacement_operand(qi_cached);
                    ctx.make_equal_to(&b_old, &b_qi);
                    return OptimizationResult::Remove;
                }
                // First read after QUASIIMMUT_FIELD: emit the load, then cache
                // the result so it survives calls (unlike normal mutable fields).
                self.quasi_immut_cache.insert(qi_key, op.pos().get());
                make_nonnull_box(ctx, &op.arg(0));
                let obj_box = ctx.resolve_operand_operand(&obj_box);
                self.cached_fields[pos].2.register_info(&obj_box);
                ctx.structinfo_setfield(op, field_idx, &Operand::from_bound_op(op_rc));
                return OptimizationResult::PassOn;
            }
        }

        // Cache miss: emit the load and cache the result.
        // heap.py postprocess_GETFIELD_GC_I:
        //     structinfo = self.ensure_ptr_info_arg0(op)
        //     structinfo.setfield(descr, op.getarg(0), op, ...)
        // heap.py optimize_GETFIELD_GC_I default path also marks the base:
        //     self.make_nonnull(op.getarg(0))
        make_nonnull_box(ctx, &op.arg(0));
        let obj_box = ctx.resolve_operand_operand(&obj_box);
        self.cached_fields[pos].2.register_info(&obj_box);
        // heap.py postprocess_GETFIELD_GC_I: structinfo.setfield(descr, op)
        //
        // PyPy info.py routes ConstPtrInfo.setfield through
        // optheap.const_infos via `_get_info(parent_descr, optheap)`, so
        // a constant struct base ALSO gets its field cached. The Rust
        // port mirrors that via `OptContext::structinfo_setfield`,
        // which dispatches by `arg0.is_constant()` to either
        // `const_infos[gcref]` (constant) or
        // `ensure_ptr_info_arg0(op).as_mut()` (regular).
        if !is_vable_field {
            ctx.structinfo_setfield(op, field_idx, &Operand::from_bound_op(op_rc));
        }
        // Virtualizable Ref fields (linked list head) need a null guard.
        let is_vable_ref = is_vable_field && matches!(op.opcode, OpCode::GetfieldGcR);
        if is_vable_ref {
            ctx.emit(op.clone());
            let cmp_pos = ctx.alloc_op_position_typed(OpCode::IntNe.result_type());
            let cmp_arg0 = Operand::from_bound_op(op_rc);
            let cmp_arg1 = Operand::const_(majit_ir::Const::Int(0));
            let mut cmp_op = Op::new(OpCode::IntNe, &[cmp_arg0, cmp_arg1]);
            cmp_op.pos().set(cmp_pos);
            let cmp_rc = OpRc::new(cmp_op);
            ctx.emit_rc(cmp_rc.clone());
            // unroll.py:409 parity: synthetic guards inherit
            // rd_resume_position from patchguardop (the optimizer's
            // running GUARD_FUTURE_CONDITION). Without this, the guard
            // arrives at store_final_boxes_in_guard with -1 and would
            // be silently dropped under the patchguardop-only fallback.
            let guard_arg = Operand::from_bound_op(&cmp_rc);
            let guard_op = Op::new(OpCode::GuardTrue, &[guard_arg]);
            if let Some(ref patch) = ctx.patchguardop {
                guard_op.set_rd_resume_position(patch.rd_resume_position());
            }
            ctx.emit(guard_op);
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }

    fn optimize_setfield(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let Some(descr) = op.getdescr() else {
            return OptimizationResult::PassOn;
        };
        let raw_obj = op.arg(0).to_opref();
        // heap.py:78 ensure_ptr_info_arg0 — install structinfo as a
        // side effect; canonical OpRef for the cache key.
        let obj_box = ctx.resolve_operand_operand(&op.arg(0));
        let obj = obj_box.to_opref();
        let _ = ctx.ensure_ptr_info_arg0(op);
        // heapcache.py _escape_from_write — pyre-specific
        // escape tracking outside the do_setfield contract.
        self.escape_from_write(ctx, &obj_box, &op.arg(1));
        // heap.py do_setfield line-by-line.
        self.do_setfield_field(op, op_rc, &descr, &obj_box, ctx)
    }

    /// heap.py `AbstractCachedEntry.do_setfield(optheap, op)`
    /// line-by-line. Lives on `OptHeap` rather than on `CachedField`
    /// because of the Rust borrow restriction: the upstream method
    /// calls `self.force_lazy_set(optheap, descr)` which would require
    /// holding `&mut cf` (cf lives inside `optheap.cached_fields`) and
    /// `&mut optheap` simultaneously. The body is identical to
    /// upstream's; the receiver is the parent container.
    ///
    /// ```python
    /// def do_setfield(self, optheap, op):
    ///     structinfo = optheap.ensure_ptr_info_arg0(op)
    ///     arg1 = get_box_replacement(self._get_rhs_from_set_op(op))
    ///     if self.possible_aliasing(structinfo):
    ///         self.force_lazy_set(optheap, op.getdescr())
    ///         assert not self.possible_aliasing(structinfo)
    ///     cached_field = self._getfield(structinfo, op.getdescr(), optheap, False)
    ///     if cached_field is not None:
    ///         cached_field = cached_field.get_box_replacement()
    ///     if not cached_field or not cached_field.same_box(arg1):
    ///         # common case: store the 'op' as lazy_set
    ///         self._lazy_set = op
    ///     else:
    ///         # cancel out — value already there
    ///         self._getfield(structinfo, op.getdescr(), optheap)
    ///         self._lazy_set = None
    /// ```
    ///
    /// The pyre `_getfield` returns `Option<FieldEntry>`; the
    /// `FieldEntry::Preamble` branch is handled by emitting the
    /// preamble op + registering the info before the cancel — that's
    /// the pyre-specific Preamble plumbing parity (info.py:259-272
    /// `produce_short_preamble_ops`).
    fn do_setfield_field(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        descr: &DescrRef,
        obj_box: &Operand,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // heap.py arg1 = get_box_replacement(self._get_rhs_from_set_op(op))
        let arg1 = ctx.resolve_operand_operand(&op.arg(1));
        let field_idx = Self::field_slot_index(descr);
        let pos = self.ensure_cached_field(descr, Some(field_idx));
        // heap.py if self.possible_aliasing(structinfo):
        //                  self.force_lazy_set(optheap, op.getdescr())
        let needs_force = self.cached_fields[pos].2.possible_aliasing(obj_box, ctx);
        if needs_force {
            // heap.py force_lazy_set(self, optheap, descr)
            // [can_cache=True]. Lifted into `force_lazy_set_field` per
            // upstream's CachedField method.
            self.force_lazy_set_field_at(pos, true, ctx);
        }
        // heap.py cached_field = self._getfield(structinfo, op.getdescr(),
        //                                          optheap, False)
        // heap.py:86-87 if cached_field is not None:
        //                   cached_field = cached_field.get_box_replacement()
        // heap.py:88-101 lazy_set vs cancel dispatch (with pyre's
        // Preamble/Value FieldEntry split).
        if let Some(entry) = self.cached_fields[pos]
            .2
            ._getfield(obj_box, descr, field_idx, ctx)
        {
            match entry.kind() {
                crate::optimizeopt::info::FieldEntryKind::Preamble(pop) => {
                    let cached_seen = ctx.resolve_operand_operand(&pop.op);
                    // heap.py:88 not cached_field.same_box(arg1)
                    if cached_seen.same_box(&arg1) {
                        let cached = ctx.force_op_from_preamble_op(pop);
                        ctx.structinfo_setfield(op, field_idx, &cached);
                        let obj_box = ctx.resolve_operand_operand(obj_box);
                        self.cached_fields[pos].2.register_info(&obj_box);
                        // heap.py:100 self._lazy_set = None
                        self.cached_fields[pos].2.lazy_set = None;
                        return OptimizationResult::Remove;
                    }
                }
                crate::optimizeopt::info::FieldEntryKind::Value(cached) => {
                    // heap.py:85-88 `if cached_field is not None:
                    // cached_field = cached_field.get_box_replacement()` then
                    // `if not cached_field or not cached_field.same_box(arg1)`
                    // — a cleared slot stores None and counts as not cached.
                    if !cached.is_none() && ctx.resolve_operand_operand(&cached).same_box(&arg1) {
                        // heap.py:100 self._lazy_set = None
                        self.cached_fields[pos].2.lazy_set = None;
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        // heap.py:89-91 common case: self._lazy_set = op
        self.cached_fields[pos].2.lazy_set = Some(op_rc.clone());
        OptimizationResult::Remove
    }

    /// heap.py `AbstractCachedEntry.force_lazy_set(optheap,
    /// descr, can_cache=True)` array-side line-by-line. Body identical
    /// to `force_lazy_set_field` modulo:
    /// - cai entry locator (submap by descr_idx, const_index by index).
    /// - cai.invalidate() also invalidates parent.clear_varindex via
    ///   submap.invalidate_index (heap.py parent.clear_varindex()
    ///   inside ArrayCachedItem.invalidate; pyre lifts this to the
    ///   caller-side per the `ArrayCachedItem::invalidate` doc).
    /// - `_get_rhs_from_set_op` uses op.arg(2) at the put_back step.
    fn force_lazy_set_array(
        &mut self,
        descr_idx: usize,
        const_index: i64,
        can_cache: bool,
        ctx: &mut OptContext,
    ) {
        // heap.py:123 op = self._lazy_set
        let lazy_data = self
            .get_cached_array_submap_mut(descr_idx)
            .and_then(|s| s.const_get_mut(const_index))
            .and_then(|cai| cai.lazy_set.take());
        match lazy_data {
            Some(mut lazy_op) => {
                // heap.py self.invalidate(descr) — cai.invalidate +
                // parent.clear_varindex (lifted into submap.invalidate_index).
                if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                    submap.invalidate_index(const_index, ctx);
                }
                // heap.py:128 self._lazy_set = None — already done via take().
                // heap.py emit postponed_op if referenced.
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos().get();
                    if lazy_op.args_slice().iter().any(|a| a.to_opref() == ppos)
                        && let Some(p) = self.postponed_op.take()
                    {
                        ctx.emit_extra(ctx.current_pass_idx, p);
                    }
                }
                // heap.py optheap.emit_extra(op, emit=False)
                let put_back_op = lazy_op.clone();
                Self::emit_lazy_setfield(&lazy_op, ctx);
                // heap.py:136-137 if not can_cache: return
                if !can_cache {
                    return;
                }
                // heap.py put_field_back_to_info — array-side write.
                let final_value = lazy_op.arg(2);
                let lazy_descr = put_back_op.getdescr();
                let lazy_obj_box = ctx.resolve_operand_operand(&put_back_op.arg(0));
                self.cache_arrayitem(&lazy_obj_box, descr_idx, const_index, lazy_descr.as_ref());
                ctx.arrayinfo_setitem(&put_back_op, const_index as usize, &final_value);
            }
            None => {
                // heap.py elif not can_cache: self.invalidate(descr)
                if !can_cache && let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                    submap.invalidate_index(const_index, ctx);
                }
            }
        }
    }

    /// heap.py `AbstractCachedEntry.do_setfield` array-side
    /// line-by-line port. Body identical to `do_setfield_field` modulo
    /// `_get_rhs_from_set_op` (arg2 vs arg1) and the entry locator
    /// (ArrayCachedItem in submap keyed by const_index vs CachedField
    /// keyed by descr). Lives on OptHeap for the same Rust borrow
    /// reason — cai is owned by `optheap.cached_arrayitems[descr_idx]
    /// .const_indexes[const_index]`.
    fn do_setfield_array(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        descr: &DescrRef,
        array: OpRef,
        array_box: &Operand,
        descr_idx: usize,
        const_index: i64,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // heap.py arg1 = get_box_replacement(self._get_rhs_from_set_op(op))
        let arg1 = ctx.resolve_operand_operand(&op.arg(2));
        // heap.py if self.possible_aliasing(structinfo):
        //                  self.force_lazy_set(optheap, op.getdescr())
        let needs_force = self
            .get_cached_array_submap(descr_idx)
            .and_then(|s| s.const_get(const_index))
            .is_some_and(|cai| cai.possible_aliasing(array_box, ctx));
        if needs_force {
            // heap.py force_lazy_set(self, optheap, descr) [can_cache=True]
            // extracted as force_lazy_set_array per AbstractCachedEntry parity.
            self.force_lazy_set_array(descr_idx, const_index, true, ctx);
        }
        // heap.py cached_field = self._getfield(structinfo, descr, optheap, False)
        // heap.py:86-87 get_box_replacement
        // heap.py:88-101 lazy_set vs cancel dispatch (pyre's
        // Preamble/Value FieldEntry split).
        if let Some(entry) = self
            .arrayitem_cache(descr, const_index)
            ._getfield(array, descr, ctx)
        {
            match entry.kind() {
                crate::optimizeopt::info::FieldEntryKind::Preamble(pop) => {
                    let cached_seen = ctx.resolve_operand_operand(&pop.op);
                    // heap.py:88 not cached_field.same_box(arg1)
                    if cached_seen.same_box(&arg1) {
                        let cached = ctx.force_op_from_preamble_op(pop);
                        let array_box = ctx.resolve_operand_operand(array_box);
                        self.arrayitem_cache(descr, const_index)
                            .register_info(&array_box);
                        ctx.arrayinfo_setitem(op, const_index as usize, &cached);
                        // heap.py:100 self._lazy_set = None
                        self.arrayitem_cache(descr, const_index).lazy_set = None;
                        return OptimizationResult::Remove;
                    }
                }
                crate::optimizeopt::info::FieldEntryKind::Value(cached) => {
                    // heap.py:85-88 `if cached_field is not None:
                    // cached_field = cached_field.get_box_replacement()` then
                    // `if not cached_field or not cached_field.same_box(arg1)`
                    // — a cleared slot stores None and counts as not cached.
                    if !cached.is_none() && ctx.resolve_operand_operand(&cached).same_box(&arg1) {
                        // heap.py:100 self._lazy_set = None
                        self.arrayitem_cache(descr, const_index).lazy_set = None;
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        // heap.py:89-91 common case: self._lazy_set = op
        let cai = self.arrayitem_cache(descr, const_index);
        cai.lazy_set = Some(op_rc.clone());
        OptimizationResult::Remove
    }

    /// heap.py `AbstractCachedEntry.force_lazy_set(optheap,
    /// descr, can_cache=True)` line-by-line for the CachedField path.
    /// Lives on `OptHeap` rather than `CachedField` due to the Rust
    /// borrow restriction (cf is owned by `optheap.cached_fields`, so
    /// `&mut cf` and `&mut optheap` can't coexist — see do_setfield_field).
    ///
    /// ```python
    /// def force_lazy_set(self, optheap, descr, can_cache=True):
    ///     op = self._lazy_set
    ///     if op is not None:
    ///         self.invalidate(descr)
    ///         self._lazy_set = None
    ///         if optheap.postponed_op:
    ///             for a in op.getarglist():
    ///                 if a is optheap.postponed_op:
    ///                     optheap.emit_postponed_op()
    ///                     break
    ///         optheap.emit_extra(op, emit=False)
    ///         if not can_cache:
    ///             return
    ///         opinfo = optheap.ensure_ptr_info_arg0(op)
    ///         self.put_field_back_to_info(op, opinfo, optheap)
    ///     elif not can_cache:
    ///         self.invalidate(descr)
    /// ```
    ///
    /// `put_field_back_to_info` is heap.py; pyre's
    /// `cache_field` + `ctx.structinfo_setfield` is the line-by-line
    /// inline port.
    fn force_lazy_set_field(&mut self, descr: &DescrRef, can_cache: bool, ctx: &mut OptContext) {
        let Some(pos) = self.cached_field_pos_for_descr(descr) else {
            return;
        };
        self.force_lazy_set_field_at(pos, can_cache, ctx);
    }

    fn force_lazy_set_field_at(&mut self, pos: usize, can_cache: bool, ctx: &mut OptContext) {
        // heap.py:123 op = self._lazy_set
        let lazy_data = self.cached_fields[pos].2.lazy_set.take();
        match lazy_data {
            Some(mut lazy_op) => {
                let field_idx = self.cached_fields[pos].0;
                // heap.py self.invalidate(descr) — purity self-gate
                // inside CachedField::invalidate.
                {
                    let (idx, descr, cf) = &mut self.cached_fields[pos];
                    cf.invalidate(descr, *idx, ctx);
                }
                // heap.py:128 self._lazy_set = None — already done via
                // `take()` above.

                // heap.py emit postponed_op if referenced.
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos().get();
                    if lazy_op.args_slice().iter().any(|a| a.to_opref() == ppos)
                        && let Some(p) = self.postponed_op.take()
                    {
                        ctx.emit_extra(ctx.current_pass_idx, p);
                    }
                }
                // heap.py optheap.emit_extra(op, emit=False)
                let put_back_op = lazy_op.clone();
                Self::emit_lazy_setfield(&lazy_op, ctx);
                // heap.py:136-137 if not can_cache: return
                if !can_cache {
                    return;
                }
                // heap.py put_field_back_to_info(op, opinfo, optheap)
                let final_value = lazy_op.arg(1);
                let lazy_obj_box = ctx.resolve_operand_operand(&put_back_op.arg(0));
                self.cached_fields[pos].2.register_info(&lazy_obj_box);
                ctx.structinfo_setfield(&put_back_op, field_idx, &final_value);
            }
            None => {
                // heap.py elif not can_cache: self.invalidate(descr)
                if !can_cache {
                    let (idx, descr, cf) = &mut self.cached_fields[pos];
                    cf.invalidate(descr, *idx, ctx);
                }
            }
        }
    }

    fn optimize_getarrayitem(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        self.optimize_getarrayitem_inner(op, op_rc, ctx, false)
    }

    /// `heap.py` `OptHeap.optimize_GETARRAYITEM_GC_PURE_I`.
    ///
    /// Constant-index cache lookup only. A variable index forces lazy
    /// setarrayitems and emits. No postprocess: a miss does not
    /// `arrayinfo_setitem` or `cache_varindex_read`.
    fn optimize_getarrayitem_gc_pure(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        self.optimize_getarrayitem_inner(op, op_rc, ctx, true)
    }

    fn optimize_getarrayitem_inner(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
        pure: bool,
    ) -> OptimizationResult {
        // Install ArrayPtrInfo via ensure_ptr_info_arg0 (return value
        // unused — we re-borrow further down via a fresh call so the
        // intermediate cache mutations can take &mut ctx without
        // tripping the borrow checker).
        let _ = ctx.ensure_ptr_info_arg0(op);
        let array_ref = ctx.resolve_operand_operand(&op.arg(0));

        // Try constant-index cache first.
        if let Some(key) = Self::arrayitem_key(op, ctx) {
            let (array, descr_idx, const_index) = key;
            let descr = op.getdescr().unwrap();
            if pure && const_index >= 0 {
                // `optimize_GETARRAYITEM_GC_PURE_I` tightens the length
                // before the constant-index cache lookup.
                ctx.with_ensured_ptr_info_arg0(op, |mut arrayinfo| {
                    if let Some(mut bound) = arrayinfo.getlenbound(None) {
                        let _ = bound.make_gt_const(const_index);
                        if let Some(mut handle) = arrayinfo.as_mut()
                            && let crate::optimizeopt::info::PtrInfo::Array(a) = &mut *handle
                        {
                            a.lenbound = bound;
                        }
                    }
                });
            }
            // heap.py getfield_from_cache — 3-way aliasing check.
            // PyPy's shared AbstractCachedEntry method on ArrayCachedItem
            // calls possible_aliasing_two_infos which can force_lazy_set
            // on UNKNOWN_ALIAS. The Rust port inlines this at the call
            // site because force_lazy_set needs &mut OptHeap + &mut OptContext.
            let mut force_lazy_arr = false;
            if let Some(cai) = self
                .get_cached_array_submap(descr_idx)
                .and_then(|s| s.const_get(const_index))
            {
                if let Some(lazy_op) = &cai.lazy_set {
                    let lazy_struct = lazy_op.arg(0);
                    // heap.py possible_aliasing_two_infos:
                    // opinfo1.same_info(opinfo2) → MUST_ALIAS.
                    if same_ptr_info(ctx, &lazy_struct, &array_ref) {
                        // MUST_ALIAS: lazy_set targets the same array → return rhs
                        let cached = lazy_op.arg(2);
                        let b_old = Operand::from_bound_op(op_rc);
                        let b_cached = ctx.resolve_operand_operand(&cached);
                        ctx.make_equal_to(&b_old, &b_cached);
                        return OptimizationResult::Remove;
                    }
                    // heap.py possible_aliasing_two_infos
                    let cannot_alias = cannot_alias_via_constants(ctx, &lazy_struct, &array_ref)
                        || ArrayCachedItem::_cannot_alias_via_classes_or_lengths(
                            &lazy_struct,
                            &array_ref,
                            ctx,
                        )
                        || ArrayCachedItem::_cannot_alias_via_content(
                            &lazy_struct,
                            &array_ref,
                            ctx,
                        );
                    if !cannot_alias {
                        // UNKNOWN_ALIAS → force_lazy_set
                        force_lazy_arr = true;
                    }
                    // CANNOT_ALIAS: fall through to _getfield
                }
                if !force_lazy_arr
                    && let Some(entry) = cai._getfield(array, &op.getdescr().unwrap(), ctx)
                {
                    match entry.kind() {
                        crate::optimizeopt::info::FieldEntryKind::Preamble(pop) => {
                            // heap.py ArrayCachedItem._getfield:
                            //   res = optheap.optimizer.force_op_from_preamble(res)
                            //   opinfo.setitem(descr, index, None, res, optheap=optheap)
                            let cached = ctx.force_op_from_preamble_op(pop);
                            let array_box = ctx.resolve_operand_operand(&array_ref);
                            self.arrayitem_cache(&descr, const_index)
                                .register_info(&array_box);
                            ctx.arrayinfo_setitem(op, const_index as usize, &cached);
                            let b_old = Operand::from_bound_op(op_rc);
                            let b_cached = ctx.resolve_operand_operand(&cached);
                            ctx.make_equal_to(&b_old, &b_cached);
                            return OptimizationResult::Remove;
                        }
                        crate::optimizeopt::info::FieldEntryKind::Value(cached) => {
                            if !cached.is_none() {
                                let b_old = Operand::from_bound_op(op_rc);
                                let b_cached = ctx.resolve_operand_operand(&cached);
                                ctx.make_equal_to(&b_old, &b_cached);
                                return OptimizationResult::Remove;
                            }
                        }
                    }
                }
            }
            // heap.py: UNKNOWN_ALIAS → force lazy_set (can_cache=True)
            if force_lazy_arr {
                let lazy_data = self
                    .get_cached_array_submap_mut(descr_idx)
                    .and_then(|s| s.const_get_mut(const_index))
                    .and_then(|cai| cai.lazy_set.take());
                if let Some(mut lazy_op) = lazy_data {
                    if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
                        submap.invalidate_index(const_index, ctx);
                    }
                    if let Some(ref postponed) = self.postponed_op {
                        let ppos = postponed.pos().get();
                        if lazy_op.args_slice().iter().any(|a| a.to_opref() == ppos)
                            && let Some(p) = self.postponed_op.take()
                        {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                    Self::emit_lazy_setfield(&lazy_op, ctx);
                    // can_cache=True: put_field_back_to_info
                    let final_value = lazy_op.arg(2);
                    let descr = lazy_op.getdescr();
                    let lazy_obj_box = ctx.resolve_operand_operand(&lazy_op.arg(0));
                    self.cache_arrayitem(&lazy_obj_box, descr_idx, const_index, descr.as_ref());
                    ctx.arrayinfo_setitem(&lazy_op, const_index as usize, &final_value);
                }
                // Cache miss — fall through to emit the getarrayitem
            }
            // Consume the imported short arrayitem: remove it so that if a later
            // setarrayitem/call invalidates cached_arrayitems, the stale preamble
            // value cannot re-populate the cache on a subsequent getarrayitem.
            let array_box = Some(ctx.resolve_operand_operand(&array_ref));
            let pop = array_box
                .as_ref()
                .and_then(|b| {
                    ctx.with_ptr_info_mut(b, |info| info.take_preamble_item(const_index as usize))
                })
                .flatten()
                .or_else(|| {
                    array_box
                        .as_ref()
                        .and_then(|b| ctx.get_const_info_mut_if_exists_box(b))
                        .and_then(|info| info.take_preamble_item(const_index as usize))
                });
            if let Some(pop) = pop {
                // heap.py:243-249: force the preamble op, then write the
                // forced value back into the array info.
                let cached = ctx.force_op_from_preamble_op(&pop);
                let array_box = ctx.resolve_operand_operand(&array_ref);
                self.arrayitem_cache(&descr, const_index)
                    .register_info(&array_box);
                ctx.arrayinfo_setitem(op, const_index as usize, &cached);
                let b_old = Operand::from_bound_op(op_rc);
                let b_cached = ctx.resolve_operand_operand(&cached);
                ctx.make_equal_to(&b_old, &b_cached);
                return OptimizationResult::Remove;
            }
            if let Some(cai) = self
                .get_cached_array_submap(descr_idx)
                .and_then(|s| s.const_get(const_index))
                && let Some(entry) = cai._getfield(array, &op.getdescr().unwrap(), ctx)
            {
                match entry.kind() {
                    crate::optimizeopt::info::FieldEntryKind::Preamble(pop) => {
                        // heap.py:243-249: force the preamble op, then write
                        // the forced value back into the array info.
                        let cached = ctx.force_op_from_preamble_op(pop);
                        let array_box = ctx.resolve_operand_operand(&array_ref);
                        self.arrayitem_cache(&descr, const_index)
                            .register_info(&array_box);
                        ctx.arrayinfo_setitem(op, const_index as usize, &cached);
                        let b_old = Operand::from_bound_op(op_rc);
                        let b_cached = ctx.resolve_operand_operand(&cached);
                        ctx.make_equal_to(&b_old, &b_cached);
                        return OptimizationResult::Remove;
                    }
                    crate::optimizeopt::info::FieldEntryKind::Value(cached) => {
                        if !cached.is_none() {
                            let b_old = Operand::from_bound_op(op_rc);
                            let b_cached = ctx.resolve_operand_operand(&cached);
                            ctx.make_equal_to(&b_old, &b_cached);
                            return OptimizationResult::Remove;
                        }
                    }
                }
            }
            if pure {
                // `optimize_GETARRAYITEM_GC_PURE_I` has no postprocess.
                if const_index >= 0 {
                    ctx.with_ensured_ptr_info_arg0(op, |mut arrayinfo| {
                        if let Some(mut bound) = arrayinfo.getlenbound(None) {
                            let _ = bound.make_gt_const(const_index);
                            if let Some(mut handle) = arrayinfo.as_mut()
                                && let crate::optimizeopt::info::PtrInfo::Array(a) = &mut *handle
                            {
                                a.lenbound = bound;
                            }
                        }
                    });
                }
                make_nonnull_box(ctx, &op.arg(0));
                return OptimizationResult::PassOn;
            }
            let array_box = ctx.resolve_operand_operand(&array_ref);
            self.cache_arrayitem(&array_box, descr_idx, const_index, op.getdescr().as_ref());
            // heap.py:676-681:
            //     arrayinfo = self.ensure_ptr_info_arg0(op)
            //     ...
            //     arrayinfo.getlenbound(None).make_gt_const(index)
            //
            // PyPy then `arrayinfo.setitem(...)` records the cached element.
            // The Rust port:
            //   1) `make_nonnull(op.getarg(0))` (heap.py:701) on the box itself
            //   2) for non-constant arg0, narrow the lenbound on the
            //      Forwarded::Info(ArrayPtrInfo) slot via `ensure_ptr_info_arg0`
            //   3) `arrayinfo.setitem(...)` via `arrayinfo_setitem` which
            //      routes constant arg0 through `_get_array_info` /
            //      `const_infos[gcref]` and regular arg0 through
            //      `ensure_ptr_info_arg0(op).as_mut().setitem(...)`.
            if const_index >= 0 {
                ctx.with_ensured_ptr_info_arg0(op, |mut arrayinfo| {
                    if let Some(mut bound) = arrayinfo.getlenbound(None) {
                        let _ = bound.make_gt_const(const_index);
                        if let Some(mut handle) = arrayinfo.as_mut()
                            && let crate::optimizeopt::info::PtrInfo::Array(a) = &mut *handle
                        {
                            a.lenbound = bound;
                        }
                    }
                });
            }
            // heap.py:703 `make_nonnull(op.getarg(0))` runs in the
            // fallthrough default of `optimize_GETARRAYITEM_GC_I`. PyPy's
            // constant-index branch only short-circuits on a cache hit;
            // when it falls through to record the new value (matching
            // pyre's `arrayinfo_setitem` below), `make_nonnull` still
            // fires.
            make_nonnull_box(ctx, &op.arg(0));
            ctx.arrayinfo_setitem(op, const_index as usize, &Operand::from_bound_op(op_rc));
            return OptimizationResult::PassOn;
        }

        // heap.py:690-701: variable-index GETARRAYITEM_GC path.
        //   self.force_lazy_setarrayitem(op.getdescr(), self.getintbound(op.getarg(1)))
        //   submap = self.arrayitem_submap(op.getdescr(), create_if_nonexistant=False)
        //   cached_result = submap.lookup_cached(arrayinfo, indexop)
        //
        // `optimize_GETARRAYITEM_GC_PURE_I` forces the lazy stores and emits.
        // It does not `lookup_cached` or `cache_varindex_read`.
        if pure {
            if let Some(descr) = op.getdescr() {
                let indexb = {
                    let b = ctx.resolve_operand_operand(&op.arg(1));
                    ctx.getintbound_handle(&b).borrow().clone()
                };
                self.force_lazy_setarrayitem(&descr, Some(&indexb), true, ctx);
            }
            make_nonnull_box(ctx, &op.arg(0));
            return OptimizationResult::PassOn;
        }
        if let Some(descr) = op.getdescr() {
            // heap.py:692-693: force lazy stores for this descr within the index bound
            let indexb = {
                let b = ctx.resolve_operand_operand(&op.arg(1));
                ctx.getintbound_handle(&b).borrow().clone()
            };
            self.force_lazy_setarrayitem(&descr, Some(&indexb), true, ctx);

            let descr_idx = descr_ptr(&descr);
            let arrayinfo = array_ref;
            let indexbox = ctx.resolve_operand_operand(&op.arg(1));
            if let Some(submap) = self.get_cached_array_submap(descr_idx)
                && let Some(cached) = submap.lookup_cached(&arrayinfo, &indexbox, ctx)
            {
                let b_old = Operand::from_bound_op(op_rc);
                ctx.make_equal_to(&b_old, &cached);
                return OptimizationResult::Remove;
            }
            self.arrayitem_submap(&descr).cache_varindex_read(
                arrayinfo,
                indexbox,
                Operand::from_bound_op(op_rc),
            );
        }

        // heap.py line 701: make_nonnull(op.getarg(0)) (optimizer.py).
        make_nonnull_box(ctx, &op.arg(0));
        OptimizationResult::PassOn
    }

    fn optimize_setarrayitem(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // heapcache.py _escape_from_write parity:
        self.escape_from_write(ctx, &op.arg(0), &op.arg(2));

        let key = match Self::arrayitem_key(op, ctx) {
            Some(k) => k,
            None => {
                // heap.py:762-767: variable index SETARRAYITEM_GC
                //   self.force_lazy_setarrayitem(op.getdescr(), indexb, can_cache=False)
                //   submap.cache_varindex_write(arrayinfo, ...)
                //   return self.emit(op)
                if let Some(descr) = op.getdescr() {
                    let indexb = {
                        let b = ctx.resolve_operand_operand(&op.arg(1));
                        ctx.getintbound_handle(&b).borrow().clone()
                    };
                    self.force_lazy_setarrayitem(&descr, Some(&indexb), false, ctx);
                    let arrayinfo = ctx.resolve_operand_operand(&op.arg(0));
                    let indexbox = ctx.resolve_operand_operand(&op.arg(1));
                    let resbox = ctx.resolve_operand_operand(&op.arg(2));
                    self.arrayitem_submap(&descr)
                        .cache_varindex_write(arrayinfo, indexbox, resbox);
                }
                return OptimizationResult::PassOn;
            }
        };

        let (array, descr_idx, const_index) = key;
        let array_box = ctx.resolve_operand_operand(&op.arg(0));
        let descr = op.getdescr().unwrap();
        // heap.py ArrayCachedItem.do_setfield (shared body via
        // AbstractCachedEntry).
        let result = self.do_setfield_array(
            op,
            op_rc,
            &descr,
            array,
            &array_box,
            descr_idx,
            const_index,
            ctx,
        );
        // heap.py `submap.clear_varindex()` AFTER do_setfield (called
        // at the optimize_SETARRAYITEM_GC site — outside do_setfield).
        if let Some(submap) = self.get_cached_array_submap_mut(descr_idx) {
            submap.clear_varindex();
        }
        result
    }

    /// `heap.py OptHeap.emitting_operation`: operations with no effect on the
    /// GC heap caches.
    ///
    /// Consulted at TWO sites here against upstream's one.  Upstream reaches the
    /// list only from `emitting_operation`, via `OptHeap.emit` and
    /// `dispatch_opt`'s `default=OptHeap.emit` (`heap.py`).  Here
    /// `emitting_operation` is a whole-optimizer callback that fires at final
    /// emission, after this pass has already dispatched — `propagate_forward`
    /// dispatches on its first line — so without the `handle_side_effects`
    /// consultation an exempt opcode would clean the caches on the way in and
    /// the later callback would then spare caches that were already gone.  The
    /// cost of missing it is pessimisation, not wrong code, which is why
    /// measurement moved nothing.
    ///
    /// Four entries — `SetfieldGc`, `SetfieldRaw`, `SetarrayitemGc`,
    /// `SetarrayitemRaw` — have their own dispatch arm and so never reach the
    /// `handle_side_effects` test; they still reach the `emitting_operation`
    /// one, which is why they stay listed.  Upstream lists its two for the same
    /// reason and marks them `# handled specially` in `emitting_operation`.
    ///
    /// Of the other eleven, ten cannot arrive on this tree today and one —
    /// `LeavePortalFrame` — is only *not known* to arrive; the reasons differ
    /// enough to be worth recording separately:
    ///   - `DebugMergePoint` has no producer at all; every occurrence is a
    ///     declaration or a consumer.
    ///   - `EnterPortalFrame` needs either a `newframe` carrying a greenkey —
    ///     both non-test callers pass `None` — or a runtime answering
    ///     `portal_jitcode`, which the production runtime leaves `None`.
    ///     `LeavePortalFrame` is NOT symmetric: it also rides `popframe`'s
    ///     `jitdriver_sd` gate, and `call.py grab_initial_jitcodes` does stamp
    ///     one, so that arm is
    ///     not known to be dead.  Do not read it as unreachable without
    ///     measuring.
    ///   - `setinteriorfield_raw` has no producer and no codewriter `OpKind`.
    ///   - `jit_debug` IS in the assembled jitcode alphabet, minted from the
    ///     `jit.debug` oopspec and wired in the blackhole.  It is absent only
    ///     because nothing calls it.
    ///   - `RawStore`, `CheckMemoryError`, `Strsetitem`, `Unicodesetitem`,
    ///     `Copystrcontent` and `Copyunicodecontent` are minted INSIDE the
    ///     optimizer — the first two by the virtual-raw-buffer force arm in
    ///     `optimizeopt/info.rs`, the rest by the force and copy paths in
    ///     `optimizeopt/vstring.rs` — and `emit_for_force` (or the equivalent
    ///     local closure in `info.rs`) routes them downstream into this pass,
    ///     which sits after `string` and `earlyforce` in `ENABLE_ALL_OPTS`.  So
    ///     the jitcode alphabet does not bound these six.
    ///
    /// The string half has carriers: `rstr.py ll_strconcat`
    /// (`pyre_object::lowlevel_string::jit_ll_strconcat`, `OS_STR_CONCAT`) and
    /// `_ll_stringslice` (`stroruni.slice`), both over `rstr.STR` payloads, so
    /// `opt_call_stroruni_*` and the vstring force paths are reachable.  The
    /// raw-buffer half has one,
    /// `_cffi_backend::cdataobj::raw_malloc_varsize_char`, so a build that
    /// includes that module can reach the virtual raw buffer; wasm32, which
    /// excludes it, cannot.
    fn has_no_heap_cache_effect(opcode: OpCode) -> bool {
        matches!(
            opcode,
            OpCode::SetfieldGc
                | OpCode::SetfieldRaw
                | OpCode::SetarrayitemGc
                | OpCode::SetarrayitemRaw
                | OpCode::SetinteriorfieldRaw
                | OpCode::RawStore
                | OpCode::Strsetitem
                | OpCode::Unicodesetitem
                | OpCode::DebugMergePoint
                | OpCode::JitDebug
                | OpCode::EnterPortalFrame
                | OpCode::LeavePortalFrame
                | OpCode::Copystrcontent
                | OpCode::Copyunicodecontent
                | OpCode::CheckMemoryError
                // `rewrite.py gen_write_barrier` only creates this op after
                // the optimizer has run, so `emitting_operation` has no arm
                // for it.  The walker records it ahead of that (the list
                // barrier), and it touches a header flag and the remembered
                // set -- no field or item any cache here describes.  Falling
                // through to the generic side-effect arm would force every
                // lazy set and drop every cache once per `list.append`.
                | OpCode::CondCallGcWb
                | OpCode::CondCallGcWbArray
        )
    }

    /// Handle operations that may have side effects.
    /// Forces lazy sets and invalidates caches as needed.
    /// Tracks allocations for aliasing analysis.
    fn handle_side_effects(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let opcode = op.opcode;

        if Self::has_no_heap_cache_effect(opcode) {
            return OptimizationResult::PassOn;
        }

        // Track allocations for aliasing analysis.
        // Allocated objects are always non-null.
        if opcode.is_malloc() {
            self.seen_allocation.insert(op.pos().get().raw() as usize);
            if let Some(new_box) = ctx.get_box_replacement_operand_opt(op.pos().get()) {
                self.unescaped.insert(new_box);
            }
            return OptimizationResult::PassOn;
        }

        if opcode.is_guard() {
            // Every guard heap keeps is downstreamed; `propagate_forward`
            // then runs heap.py for it — the lazy-set flush, then the
            // postponed op. Neither step belongs here.
            return OptimizationResult::PassOn;
        }

        // Final operations (Jump, Finish): force everything.
        if opcode.is_final() {
            self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
            return OptimizationResult::PassOn;
        }

        // Calls: mark arguments as escaped, force lazy sets, and invalidate.
        if opcode.is_call() {
            let oopspec = Self::get_oopspecindex(op);
            match oopspec {
                // heap.py:472-475: DICT_LOOKUP caching — consecutive dict
                // lookups on the same dict with the same key can be
                // deduplicated. On a miss the call falls through to
                // self.emit(op) → emitting_operation → force_from_effectinfo,
                // identical to a non-DICT_LOOKUP residual call.
                OopSpecIndex::DictLookup => {
                    if self._optimize_call_dict_lookup(op, op_rc, ctx) {
                        return OptimizationResult::Remove;
                    }
                    return self.emit_residual_call(op, ctx);
                }
                // heap.py:466-477: only DICT_LOOKUP gets special handling.
                // ARRAYCOPY/ARRAYMOVE optimization belongs in rewrite.py
                // (rewrite.py:596-688), not the heap pass. The heap pass
                // sees them as regular calls through force_from_effectinfo.
                _ => return self.emit_residual_call(op, ctx),
            }
        }

        // Other side-effecting ops: force and invalidate.
        if !opcode.has_no_side_effect() && !opcode.is_ovf() {
            self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
            self.clean_caches(ctx);
            return OptimizationResult::PassOn;
        }

        // Pure / no-side-effect / overflow ops: pass through.
        OptimizationResult::PassOn
    }

    fn dispatch_propagate(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        match op.opcode {
            // ── Field reads ──
            OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                self.optimize_getfield(op, op_rc, ctx)
            }

            // ── Raw field reads/writes ──
            // Keep these conservative. The standard heap.py cache/postprocess
            // logic applies to GC field descriptors, while raw field traffic is
            // used by compatibility paths that intentionally reload state from
            // memory instead of carrying it through loop args.
            OpCode::GetfieldRawI | OpCode::GetfieldRawR | OpCode::GetfieldRawF => {
                OptimizationResult::PassOn
            }
            OpCode::SetfieldRaw => OptimizationResult::PassOn,

            // ── Field writes ──
            OpCode::SetfieldGc => self.optimize_setfield(op, op_rc, ctx),

            // ── Array item reads ──
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcF => {
                self.optimize_getarrayitem(op, op_rc, ctx)
            }
            // `heap.py` `optimize_GETARRAYITEM_GC_PURE_{I,R,F}`.
            OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF => self.optimize_getarrayitem_gc_pure(op, op_rc, ctx),

            // ── Raw array item reads/writes ──
            // Same rationale as the `GetfieldRaw*` / `SetfieldRaw` arms:
            // keep exact ordering and
            // dynamic indices visible until we have RPython-style virtualizable
            // handling for these buffers.
            OpCode::GetarrayitemRawI | OpCode::GetarrayitemRawR | OpCode::GetarrayitemRawF => {
                OptimizationResult::PassOn
            }

            // ── Array item writes ──
            OpCode::SetarrayitemGc => self.optimize_setarrayitem(op, op_rc, ctx),
            OpCode::SetarrayitemRaw => OptimizationResult::PassOn,

            // ── Interior field reads ──
            // info.py:682: "heapcache does not work for interiorfields"
            // RPython has no optimize_GETINTERIORFIELD_GC handler — falls
            // through to optimize_default (just emit). GETINTERIORFIELD has
            // no side effect so emitting_operation returns early (heap.py).
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF => OptimizationResult::PassOn,
            // SETINTERIORFIELD_GC: NOT matched here — falls through to
            // handle_side_effects (the `_` arm). RPython heap.py:463-464:
            // SETINTERIORFIELD_GC is NOT in the emitting_operation exclusion
            // list, so it triggers force_all_lazy_sets + clean_caches.

            // ARRAYLEN_GC / STRLEN / UNICODELEN: pure ops handled by OptPure
            // (resoperation.py:947-1056 _ALWAYS_PURE_FIRST..LAST). Heap CSE
            // would shadow OptPure's `_pure_operations[opnum]` table.

            // ── heap.py: Allocation tracking ──
            OpCode::New | OpCode::NewWithVtable | OpCode::NewArray | OpCode::NewArrayClear => {
                self.seen_allocation.insert(op.pos().get().raw() as usize);
                if let Some(new_box) = ctx.get_box_replacement_operand_opt(op.pos().get()) {
                    self.unescaped.insert(new_box);
                }
                OptimizationResult::PassOn
            }

            // RPython heap.py: CALL_ASSEMBLER — force all lazy sets before
            // the call. The callee reads from the allocated objects passed
            // in the args array; any pending SetfieldGc must be flushed to
            // memory before execution transfers to the callee.
            //
            // Unlike force_all_lazy_setfields (which mirrors RPython's
            // emit_extra(emit=False) and drops non-virtualizable ops),
            // CALL_ASSEMBLER REQUIRES the SetfieldGc ops to reach the
            // compiled code so that forced-virtual objects have their
            // fields initialized before the callee reads them.
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                // heap.py:454-455: call_assembler always resets
                // _seen_guard_not_invalidated (can call arbitrary code).
                self.seen_guard_not_invalidated = false;
                self.mark_escaped_varargs(op, ctx);
                // heap.py: force_all_lazy_sets + clean_caches.
                self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                self.clean_caches(ctx);
                OptimizationResult::PassOn
            }

            // ── heap.py: CALL_MAY_FORCE — postpone until GUARD_NOT_FORCED ──
            // These calls may force virtualizable objects, so we defer emission
            // until the guard arrives, ensuring correct exception semantics.
            OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[opt-heap] postpone {:?} pos={:?} descr={:?}",
                        op.opcode,
                        op.pos().get(),
                        op.descr
                    );
                }
                // RPython emitting_operation: calls go through
                // force_from_effectinfo (selective) or clean_caches,
                // NOT force_all_lazy. force_all_lazy is only in flush().
                let escaped_owners = self.call_argument_owner_closure(op, ctx);
                self.mark_escaped_varargs(op, ctx);
                // See `emit_residual_call`: this pyre-specific flush is
                // required for current synthetic correctness.
                self.force_call_argument_lazy_sets(&escaped_owners, ctx.current_pass_idx, ctx);
                // Postpone the call — it will be emitted when GUARD_NOT_FORCED arrives.
                self.postponed_op = Some(op.clone());
                if Self::call_has_random_effects(op) {
                    self.clean_caches(ctx);
                } else {
                    self.force_from_effectinfo(op, ctx);
                }
                if Self::call_can_invalidate(op) {
                    self.seen_guard_not_invalidated = false;
                }
                OptimizationResult::Remove
            }

            // GUARD_NOT_FORCED (with its postponed CALL_MAY_FORCE) and
            // GUARD_[NO_]OVERFLOW (with its postponed INT_*_OVF) need no arms of
            // their own: `propagate_forward` runs heap.py — the guard's
            // `force_lazy_sets_for_guard`, then the postponed op — for every
            // guard, so both pairs stay adjacent. heap.py likewise has no
            // `optimize_GUARD_NOT_FORCED` and no `optimize_GUARD_NO_OVERFLOW`;
            // its only guard-named methods are the two below.

            // heap.py has no optimize_COND_CALL. COND_CALL is is_call
            // (`resoperation.py` `OpHelpers.is_call`), so `OptHeap.emit` →
            // `emitting_operation` routes it through `force_from_effectinfo`
            // when `!has_random_effects`. A dedicated always-flush arm
            // residualized still-virtual inlined-frame `JitVirtualRef`s
            // (`virtualize.py` `optimize_VIRTUAL_REF`).
            OpCode::CondCallN => OptimizationResult::PassOn,

            // heap.py optimize_GUARD_NO_EXCEPTION / optimize_GUARD_EXCEPTION.
            // When _optimize_CALL_DICT_LOOKUP folds a lookup, it sets
            // last_emitted_removed; the trailing GUARD_NO_EXCEPTION is dead.
            OpCode::GuardNoException | OpCode::GuardException => {
                if self.last_emitted_removed {
                    self.last_emitted_removed = false;
                    OptimizationResult::Remove
                } else {
                    OptimizationResult::PassOn
                }
            }

            // heap.py optimize_GUARD_NOT_INVALIDATED
            OpCode::GuardNotInvalidated => {
                if self.seen_guard_not_invalidated {
                    OptimizationResult::Remove
                } else {
                    self.seen_guard_not_invalidated = true;
                    OptimizationResult::PassOn
                }
            }

            // Quasi-immutable field: treat as read + guard_not_invalidated.
            // The QUASIIMMUT_FIELD op marks a field that rarely changes.
            // The optimizer replaces the field read with the cached value and
            // emits GUARD_NOT_INVALIDATED to ensure validity.
            OpCode::QuasiimmutField => {
                // RPython optimize_QUASIIMMUT_FIELD (heap.py):
                // Does NOT create a new GUARD_NOT_INVALIDATED — the tracer
                // already emitted one via generate_guard (pyjitpl.py).
                // Records quasi_immutable_deps for invalidation tracking.
                let obj_arg = op.arg(0);
                let obj = obj_arg.to_opref();
                let descr = op.getdescr();
                // heap.py `optimize_QUASIIMMUT_FIELD`:
                //     structvalue = self.ensure_ptr_info_arg0(op)
                //     if not structvalue.is_constant():
                //         return
                // `get_box_replacement_operand_opt` + `get_constant_ptr_box`
                // skipped the dep block when the ConstPtr lived on the
                // replacement chain that `ensure_ptr_info_arg0` already
                // answers (`optimizer.py` `ensure_ptr_info_arg0`).
                let structvalue = ctx.ensure_ptr_info_arg0(op);
                let struct_ptr = if structvalue.is_constant() {
                    match structvalue {
                        crate::optimizeopt::info::EnsuredPtrInfo::Constant { gcref, .. } => {
                            gcref.0 as u64
                        }
                        _ => 0,
                    }
                } else {
                    0
                };
                if struct_ptr != 0 {
                    // heap.py:812-814 `assert isinstance(qmutdescr,
                    // QuasiImmutDescr)`, answered as a verdict.  A marker whose
                    // recording resolved no instance has nobody to register the
                    // finished loop with, and a loop that folded the field with
                    // no watcher behind it would never be revoked.
                    let Some(qmutdescr) = descr.as_ref().and_then(|d| d.as_quasi_immut_descr())
                    else {
                        return OptimizationResult::InvalidLoop(
                            "quasi immutable field marker carries no QuasiImmut instance",
                        );
                    };
                    // heap.py:818-819 — the field can have changed between the
                    // tracer reading it and this pass running.  Abandon the loop
                    // when it has; the traced value is baked in as a constant
                    // and nothing downstream will re-prove it.
                    if !Self::quasiimmut_field_still_valid(op, obj, struct_ptr, qmutdescr, ctx) {
                        return OptimizationResult::InvalidLoop(
                            "quasi immutable field changed during tracing",
                        );
                    }
                    // heap.py:821-823 `self.optimizer.quasi_immutable_deps[
                    // qmutdescr.qmut] = None`.
                    ctx.add_quasi_immutable_dep(qmutdescr.qmut().clone());
                }
                if let Some(descr) = descr.as_ref() {
                    let obj_box = ctx.resolve_operand_operand(&obj_arg);
                    // The cache is keyed on the field, so it reads through the
                    // per-read `QuasiImmutDescr` to the registry singleton it
                    // wraps — a fresh wrapper per read would key each entry to
                    // itself and never hit.
                    let key = Self::field_cache_identity(
                        descr
                            .as_quasi_immut_descr()
                            .map_or(descr, |q| q.fielddescr()),
                    );
                    self.quasi_immut_cache.insert((obj_box, key), OpRef::NONE);
                }
                OptimizationResult::Remove
            }

            // ── heap.py: RAW_LOAD / RAW_STORE — virtualize.py handles ──
            //
            // PyPy heap.py does NOT cache RAW_LOAD/RAW_STORE. Raw
            // pointer arithmetic over `VirtualRawBuffer` /
            // `VirtualRawSlice` is handled by virtualize.py optimize_RAW_LOAD_I.
            // RAW_STORE is also listed in `emitting_operation`'s "no
            // effect on GC struct" list (heap.py:442). The shared exemption in
            // handle_side_effects makes this fallthrough match
            // `dispatch_opt(default=OptHeap.emit)` at heap.py.

            // ── GC_LOAD / GC_LOAD_INDEXED: generic memory loads ──
            // These could read from any field/array slot, so force all
            // pending lazy writes to ensure correct values.
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::GcLoadIndexedI
            | OpCode::GcLoadIndexedR
            | OpCode::GcLoadIndexedF => {
                self.force_all_lazy_setfields(ctx.current_pass_idx, ctx);
                self.force_all_lazy_setarrayitems(ctx.current_pass_idx, ctx);
                OptimizationResult::PassOn
            }

            // ── Everything else: check for side effects ──
            _ => self.handle_side_effects(op, op_rc, ctx),
        }
    }
}

impl Default for OptHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptHeap {
    fn propagate_forward(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let result = self.dispatch_propagate(op, op_rc, ctx);
        // heap.py `emit()` override parity, in upstream's order:
        //
        //     self.emitting_operation(op)     # heap.py
        //     self.emit_postponed_op()        # heap.py
        //     ... postpone comparison / call_may_force / ovf, else pass on
        //
        // Both of the first two fire for EVERY op that reaches `heap.emit()` —
        // any op this pass downstreams, not only the ones it returns as `Emit`.
        // PassOn / Replace / Restart hand the op to the next pass (eventually
        // `Optimizer::emit_operation`), so they reach `emit()` too. Gating on
        // `Emit` alone held a postponed comparison past a following non-`Emit`
        // consumer (e.g. a COND_CALL whose condition is the comparison),
        // emitting the comparison after its use. A removed op never reaches
        // `emit()`, so Remove / InvalidLoop must run neither step.
        let downstreamed = matches!(
            result,
            OptimizationResult::Emit(_)
                | OptimizationResult::PassOn
                | OptimizationResult::Replace(_)
                | OptimizationResult::Restart(_)
        );
        if downstreamed {
            // heap.py `emitting_operation(op)`, whose guard case is
            // heap.py: ONE `rop.is_guard(op.opnum)` test covering every
            // guard opcode, running `force_lazy_sets_for_guard`.
            //
            // This is about ORDER, not coverage. The force already ran for every
            // guard — `Optimizer::emit_operation` fires `emitting_operation` for
            // all passes at final emission — but it ran too LATE, after the
            // postponed op had been released. Running it here, at heap's own
            // emit point, puts the forced SETFIELD_GC in FRONT of the op
            // released on the next line instead of between that op and its
            // guard. Both consumers want the adjacency: an `INT_*_OVF`
            // publishes its answer in the condition flags, and there is nothing
            // to re-test if it is lost (`aarch64/assembler.py:1191-1193`
            // asserts it); a comparison is merely fused into its guard by
            // `next_op_can_accept_cc` (`aarch64/assembler.py`) and
            // falls back to materialise-and-re-test when it is not, so for that
            // one the cost is an extra `tst`, not a wrong answer.
            if op.opcode.is_guard() {
                self.force_lazy_sets_for_guard_early(ctx);
            }
            // heap.py `emit_postponed_op()`.
            if let Some(postponed) = self.postponed_op.take() {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[opt-heap] emit postponed {:?} pos={:?} before {:?} pos={:?}",
                        postponed.opcode,
                        postponed.pos().get(),
                        op.opcode,
                        op.pos().get()
                    );
                }
                ctx.emit_extra(ctx.current_pass_idx, postponed);
            }
        }
        // heap.py:419-423 — postpone comparison / ovf ops. CALL_MAY_FORCE is
        // postponed in its own arm, which returns Remove before reaching here.
        let postpone = match &result {
            OptimizationResult::Emit(emit_op)
                if emit_op.opcode.is_comparison() || emit_op.opcode.is_ovf() =>
            {
                Some(emit_op.clone())
            }
            OptimizationResult::PassOn if op.opcode.is_comparison() || op.opcode.is_ovf() => {
                Some(op.clone())
            }
            _ => None,
        };
        if let Some(emit_op) = postpone {
            self.postponed_op = Some(emit_op);
            // optimizer.py emit — postponed ops do NOT call
            // Optimization.emit, so line 86's `last_emitted_operation = op`
            // does NOT fire. Leave `last_emitted_removed` intact so a
            // GUARD_NO_EXCEPTION trailing a folded DICT_LOOKUP that came
            // before this comparison still observes REMOVED.
            return OptimizationResult::Remove;
        }
        // optimizer.py — `Optimization.emit(op)` and
        // `Optimization.emit_result(opt_result)` BOTH set
        // `self.last_emitted_operation = op` before returning. The
        // PASS_OP_ON path (line 87) reaches that assignment too, so
        // every dispatch result that is downstreamed (Emit non-postponed,
        // PassOn, Replace, Restart) overwrites the REMOVED sentinel.
        // Remove / InvalidLoop / postponed-Emit do not.
        match &result {
            OptimizationResult::Emit(_)
            | OptimizationResult::PassOn
            | OptimizationResult::Replace(_)
            | OptimizationResult::Restart(_) => {
                self.last_emitted_removed = false;
            }
            OptimizationResult::Remove | OptimizationResult::InvalidLoop(_) => {}
        }
        result
    }

    fn setup(&mut self) {
        self.cached_fields.clear();
        self.cached_arrayitems.clear();
        self.seen_guard_not_invalidated = false;
        self.postponed_op = None;
        self.seen_allocation.make_empty();
        self.unescaped.clear();
        self.heapc_deps.clear();
        self.last_emitted_removed = false;
        self.cached_dict_reads.clear();
        self.corresponding_array_descrs.clear();
        self.quasi_immut_cache.clear();
    }

    fn flush(&mut self, ctx: &mut OptContext) {
        // heap.py flush():
        //   self.cached_dict_reads.clear()
        //   self.corresponding_array_descrs.clear()
        //   self.force_all_lazy_sets()
        //   self.emit_postponed_op()
        self.cached_dict_reads.clear();
        self.corresponding_array_descrs.clear();
        self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
        if let Some(postponed) = self.postponed_op.take() {
            ctx.emit_extra(ctx.current_pass_idx, postponed);
        }
    }

    /// RPython heap.py: emitting_operation(op)
    /// Called for EVERY op about to be emitted, regardless of which pass emits it.
    /// This is how the heap optimizer forces lazy sets before guards even when
    /// the guard was emitted by an earlier pass (e.g., IntBounds).
    fn emitting_operation(&mut self, op: &Op, ctx: &mut OptContext, self_pass_idx: usize) {
        // heap.py: emitting_operation(op)
        //
        // RPython calls emitting_operation in heap pass context.
        // Save/restore current_pass_idx so internal methods
        // (force_from_effectinfo, etc.) use the correct heap index.
        let saved_pass_idx = ctx.current_pass_idx;
        ctx.current_pass_idx = self_pass_idx;

        // RPython early returns for side-effect-free operations:
        if op.opcode.has_no_side_effect() {
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        if op.opcode.is_ovf() {
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        // heap.py:432-434: guards → force lazy sets for guard
        if op.opcode.is_guard() {
            let pending_virtual = self.force_lazy_sets_for_guard(self_pass_idx, ctx);
            // heap.py:433: self.optimizer.pendingfields = pendingfields
            ctx.pending_for_guard = pending_virtual
                .into_iter()
                .map(|op| (*op).clone())
                .collect();
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        if Self::has_no_heap_cache_effect(op.opcode) {
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        // heap.py:453-463: calls → handle effects
        if op.opcode.is_call() {
            if op.opcode.is_call_assembler() {
                self.seen_guard_not_invalidated = false;
            } else {
                if Self::call_can_invalidate(op) {
                    self.seen_guard_not_invalidated = false;
                }
                if !op.has_descr() {
                    // RPython: all calls have descriptors. Conservative
                    // fallback for pyre calls that lack an EffectInfo.
                    self.force_all_lazy_sets(self_pass_idx, ctx);
                    self.clean_caches(ctx);
                    ctx.current_pass_idx = saved_pass_idx;
                    return;
                }
                if !Self::call_has_random_effects(op) {
                    self.force_from_effectinfo(op, ctx);
                    ctx.current_pass_idx = saved_pass_idx;
                    return;
                }
            }
        }
        // heap.py:464: everything else → force all lazy sets + clean caches
        self.force_all_lazy_sets(self_pass_idx, ctx);
        self.clean_caches(ctx);
        ctx.current_pass_idx = saved_pass_idx;
    }

    /// heap.py OptHeap.produce_potential_short_preamble_ops(sb)
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        ctx: &mut OptContext,
    ) {
        // heap.py:370-372:
        //     for descr in descrkeys:
        //         d = self.cached_fields[descr]
        //         d.produce_potential_short_preamble_ops(self.optimizer, sb, descr)
        let mut field_entries: Vec<_> = self
            .cached_fields
            .iter()
            .map(|(field_idx, descr, cf)| (descr, (*field_idx, cf)))
            .collect();
        sort_descr_item_refs_untranslated(&mut field_entries);
        for (descr, (field_idx, cf)) in field_entries {
            cf.produce_potential_short_preamble_ops(sb, descr, field_idx, ctx);
        }
        // heap.py:374-377:
        //     for descr, submap in self.cached_arrayitems.items():
        //         for index, d in submap.const_indexes.items():
        //             d.produce_potential_short_preamble_ops(self.optimizer, sb, descr, index)
        for (_, descr, submap) in &self.cached_arrayitems {
            for (_, cai) in &submap.const_indexes {
                cai.produce_potential_short_preamble_ops(sb, descr, ctx);
            }
        }
    }

    fn register_preamble_cached_field(&mut self, struct_box: &Operand, descr: &DescrRef) {
        // shortpreamble.py:72-75:
        // `opinfo.setfield(..., optheap=optheap, cf=cf)` registers the
        // imported PtrInfo immediately, before the body first reads it.
        self.cache_field(struct_box, descr);
    }

    fn register_preamble_cached_arrayitem(
        &mut self,
        array_box: &Operand,
        descr: &DescrRef,
        index: i64,
    ) {
        // shortpreamble.py:84-85, symmetric ArrayCachedItem registration.
        self.arrayitem_cache(descr, index).register_info(array_box);
    }

    fn name(&self) -> &'static str {
        "heap"
    }

    fn set_vrefinfo(&mut self, vrefinfo: crate::virtualref::VirtualRefInfo) {
        self.vrefinfo = vrefinfo;
    }

    /// heap.py OptHeap.serialize_optheap — emit struct field triples.
    /// The `available_boxes` filter (heap.py:836,845) is applied later, in
    /// `bridgeopt::serialize_optimizer_knowledge`, once the live-box set is
    /// known; this raw export accepts every cached field.
    fn export_cached_fields(&self, ctx: &mut OptContext) -> Vec<(OpRef, DescrRef, OpRef)> {
        let mut result = Vec::new();
        // heap.py:827-846: for descr, cf in cached_fields.iteritems():
        for (field_idx, descr, cf) in &self.cached_fields {
            // heap.py:830-831: if cf._lazy_set: continue
            if cf.lazy_set.is_some() {
                continue;
            }
            // heap.py:828: if descr.get_descr_index() == -1: continue
            if descr.get_descr_index() == -1 {
                continue;
            }
            // heap.py:833-834:
            //     parent_descr = descr.get_parent_descr()
            //     if not parent_descr.is_object(): continue
            let parent = descr.as_field_descr().and_then(|fd| fd.get_parent_descr());
            let is_object = parent
                .as_ref()
                .and_then(|pd| pd.as_size_descr())
                .is_some_and(|sd| sd.is_object());
            if !is_object {
                continue;
            }
            // heap.py:835-846: for i, box1 in enumerate(cf.cached_structs)
            for obj in &cf.cached_structs {
                if obj.is_none() {
                    continue;
                }
                // heap.py:838-839: structinfo = cf.cached_infos[i]
                //                  box2 = structinfo.getfield(descr)
                let resolved_box = ctx.resolve_operand_operand_opt(obj);
                let Some(entry) = resolved_box
                    .as_ref()
                    .and_then(Operand::ptr_info)
                    .and_then(|info| info.getfield(*field_idx))
                    .or_else(|| {
                        resolved_box
                            .as_ref()
                            .and_then(|b| ctx.get_const_info_mut_box(b, parent.clone()))
                            .and_then(|info| info.getfield(*field_idx))
                    })
                else {
                    continue;
                };
                // heap.py serialize_optheap then does
                // `box2 = box2.get_box_replacement()` and keeps the triple
                // only when `box2.is_constant() or box2 in available_boxes`.
                // An unforced `PreambleOp` is an `AbstractResOp`
                // (`shortpreamble.py PreambleOp`):
                // `AbstractValue.get_box_replacement` returns the
                // `PreambleOp` itself, `AbstractValue.is_constant` is
                // false, and the wrapper is not a livebox, so the triple
                // is dropped. `FieldEntry::as_seen_opref` exposes
                // `PreambleOp.op` (often a Const) instead, which would
                // pass that filter and let a later bridge inherit a heap
                // fact the short preamble never re-established.
                if entry.is_preamble() {
                    continue;
                }
                let seen = entry.as_seen_operand();
                // heap.py:842-843: if box2 is None: continue (cleared slot)
                if seen.is_none() {
                    continue;
                }
                // heap.py:844: box2 = box2.get_box_replacement(). The export
                // triple still stores the terminal position.
                let val = ctx.resolve_operand_operand(&seen).to_opref();
                // heap.py:845 `box2 in available_boxes` is enforced later in
                // bridgeopt; here the resolved field is exported unconditionally.
                result.push((obj.to_opref(), descr.clone(), val));
            }
        }
        result
    }

    /// heap.py OptHeap.deserialize_optheap (struct half)
    fn import_cached_fields(&mut self, entries: &[(OpRef, DescrRef, OpRef)], ctx: &mut OptContext) {
        use crate::optimizeopt::info::PtrInfo;
        for (box1, descr, box2) in entries {
            if box1.is_none() || box2.is_none() {
                continue;
            }
            let field_idx = Self::field_slot_index(descr);
            let resolved = ctx.get_replacement_opref(*box1);
            // heap.py:872-873: parent_descr = descr.get_parent_descr()
            //                  assert parent_descr.is_object()
            let parent_descr = descr.as_field_descr().and_then(|fd| fd.get_parent_descr());
            debug_assert!(
                parent_descr
                    .as_ref()
                    .and_then(|pd| pd.as_size_descr())
                    .is_some_and(|sd| sd.is_object()),
                "deserialize_optheap: parent_descr must be is_object()"
            );
            // heap.py:874-881:
            //     if box1.is_constant():
            //         structinfo = info.ConstPtrInfo(box1)
            //     else:
            //         structinfo = box1.get_forwarded()
            //         if not isinstance(structinfo, info.AbstractVirtualPtrInfo):
            //             structinfo = info.InstancePtrInfo(parent_descr)
            //             structinfo.init_fields(parent_descr, descr.get_index())
            //             box1.set_forwarded(structinfo)
            // The `isinstance` test is against `AbstractVirtualPtrInfo`, which
            // a NON-virtual `InstancePtrInfo` also satisfies (info.py ->
            // :175 -> :124), so an info that is merely class-known keeps its
            // class. Reading it as `is_virtual` instead replaces it with a
            // fresh `Instance(known_class: None)` and discards the class bit
            // `make_constant_class` just decoded from the parent's rd_numb.
            let resolved_has_info = ctx
                .get_box_replacement_operand_opt(*box1)
                .as_ref()
                .and_then(Operand::ptr_info)
                .is_some_and(|info| info.is_abstract_virtual_ptr_info());
            let needs_install = ctx
                .get_box_replacement_operand_opt(resolved)
                .and_then(|cb| cb.const_value())
                .is_none()
                && !resolved_has_info;
            if needs_install {
                // info.py InstancePtrInfo + init_fields
                if let Some(b) = ctx.get_box_replacement_operand_opt(resolved) {
                    ctx.set_ptr_info(&b, PtrInfo::instance(parent_descr.clone(), None));
                }
            }
            // heap.py: cf = self.field_cache(&descr)
            //                  structinfo.setfield(descr, box1, box2, optheap, cf=cf)
            let box1_op = ctx.materialize_operand_at(*box1);
            self.cache_field(&box1_op, descr);
            let resolved_box = ctx.get_box_replacement_operand_opt(resolved);
            if resolved_box
                .as_ref()
                .and_then(|cb| cb.const_value())
                .is_some()
            {
                let box2 = ctx.materialize_operand_at(*box2);
                if let Some(info) = resolved_box
                    .as_ref()
                    .and_then(|cb| ctx.get_const_info_mut_box(cb, parent_descr.clone()))
                {
                    info.setfield(field_idx, box2);
                }
            } else {
                let box2 = ctx.materialize_operand_at(*box2);
                if let Some(b) = resolved_box.as_ref() {
                    ctx.with_ptr_info_mut(b, |info| info.setfield(field_idx, box2.clone()));
                }
            }
        }
    }

    /// heap.py serialize_optheap (array half) — emit array-item quads.
    /// The `available_boxes` filter (heap.py:855,866) is applied later, in
    /// `bridgeopt::serialize_optimizer_knowledge`; this raw export accepts all.
    fn export_cached_arrayitems(&self, ctx: &mut OptContext) -> Vec<(OpRef, i64, DescrRef, OpRef)> {
        let mut result = Vec::new();
        for (_, descr, submap) in &self.cached_arrayitems {
            // heap.py:849: if descr.get_descr_index() == -1: continue
            if descr.get_descr_index() == -1 {
                continue;
            }
            for (&index, cai) in submap.const_indexes.iter() {
                // heap.py:852: if cf._lazy_set: continue
                if cai.lazy_set.is_some() {
                    continue;
                }
                for obj in &cai.cached_structs {
                    if obj.is_none() {
                        continue;
                    }
                    // heap.py:858: if index >= 2**15: continue
                    if index >= (1 << 15) {
                        continue;
                    }
                    let resolved_box = ctx.resolve_operand_operand_opt(obj);
                    // heap.py:860: box2 = arrayinfo.getitem(descr, index)
                    let Some(entry) = resolved_box
                        .as_ref()
                        .and_then(Operand::ptr_info)
                        .and_then(|info| info.getitem(index as usize))
                        .or_else(|| {
                            resolved_box
                                .as_ref()
                                .and_then(|b| ctx.get_const_info_array_mut_box(b, descr.clone()))
                                .and_then(|info| info.getitem(index as usize))
                        })
                    else {
                        continue;
                    };
                    // Same `PreambleOp` filter as the struct half:
                    // `serialize_optheap` keeps a triple only when the
                    // replacement is a Const or a livebox, and an unforced
                    // `PreambleOp` is neither.
                    if entry.is_preamble() {
                        continue;
                    }
                    let seen = entry.as_seen_operand();
                    // heap.py:863-864: if box2 is None: continue (cleared slot)
                    if seen.is_none() {
                        continue;
                    }
                    // heap.py:865: box2 = box2.get_box_replacement(). The export
                    // quad still stores the terminal position.
                    let val = ctx.resolve_operand_operand(&seen).to_opref();
                    // heap.py:866 `box2 in available_boxes` is enforced later in
                    // bridgeopt; here the resolved item is exported unconditionally.
                    result.push((obj.to_opref(), index, descr.clone(), val));
                }
            }
        }
        result
    }

    /// heap.py deserialize_optheap (array half)
    fn import_cached_arrayitems(
        &mut self,
        entries: &[(OpRef, i64, DescrRef, OpRef)],
        ctx: &mut OptContext,
    ) {
        use crate::optimizeopt::info::PtrInfo;
        for (box1, index, descr, box2) in entries {
            if box1.is_none() || box2.is_none() {
                continue;
            }
            let resolved = ctx.get_replacement_opref(*box1);
            // heap.py:886-892:
            //     if box1.is_constant(): arrayinfo = info.ConstPtrInfo(box1)
            //     else:
            //         arrayinfo = box1.get_forwarded()
            //         if not isinstance(arrayinfo, info.AbstractVirtualPtrInfo):
            //             arrayinfo = info.ArrayPtrInfo(descr)
            //             box1.set_forwarded(arrayinfo)
            // Same `AbstractVirtualPtrInfo` test as the struct half above: an
            // existing non-virtual `ArrayPtrInfo` satisfies it (info.py ->
            // :124) and must be kept, along with the length bound it carries.
            let resolved_has_info = ctx
                .get_box_replacement_operand_opt(*box1)
                .as_ref()
                .and_then(Operand::ptr_info)
                .is_some_and(|info| info.is_abstract_virtual_ptr_info());
            let needs_install = ctx
                .get_box_replacement_operand_opt(resolved)
                .and_then(|cb| cb.const_value())
                .is_none()
                && !resolved_has_info;
            if needs_install && let Some(b) = ctx.get_box_replacement_operand_opt(resolved) {
                ctx.set_ptr_info(
                    &b,
                    PtrInfo::array(
                        descr.clone(),
                        crate::optimizeopt::intutils::IntBound::nonnegative(),
                    ),
                );
            }
            // heap.py: cf = self.arrayitem_cache(descr, index)
            //                  arrayinfo.setitem(descr, index, box1, box2, optheap, cf=cf)
            let box1_op = ctx.materialize_operand_at(*box1);
            let cai = self.arrayitem_cache(descr, *index);
            cai.register_info(&box1_op);
            let resolved_box = ctx.get_box_replacement_operand_opt(resolved);
            if resolved_box
                .as_ref()
                .and_then(|cb| cb.const_value())
                .is_some()
            {
                // info.py ConstPtrInfo.setitem → _get_array_info
                let box2 = ctx.materialize_operand_at(*box2);
                if let Some(info) = resolved_box
                    .as_ref()
                    .and_then(|b| ctx.get_const_info_array_mut_box(b, descr.clone()))
                {
                    info.setitem(*index as usize, box2);
                }
            } else {
                let idx = *index as usize;
                let box2 = ctx.materialize_operand_at(*box2);
                if let Some(b) = &resolved_box {
                    ctx.with_ptr_info_mut(b, |info| info.setitem(idx, box2.clone()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Upstream parity anchor:
    //! `rpython/jit/metainterp/test/test_heapcache.py` and
    //! `rpython/jit/metainterp/test/test_tracingopts.py`
    //! (`test_heapcache_interiorfields`, `test_heapcache_from_constant`, ...).
    //!
    //! Imported-short-field, arraycopy-range, and byte-array cases below cover
    //! Rust-specific optimizer-state boundaries that upstream mostly exercises
    //! indirectly through larger integration tests.

    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use majit_ir::{
        CallDescr, Descr, DescrRef, EffectInfo, ExtraEffect, FieldDescr, OopSpecIndex, Op, OpCode,
        OpRc, OpRef, SizeDescr, Type, bitstring,
    };

    use crate::optimizeopt::info::PtrInfo;
    use crate::optimizeopt::optimizer::Optimizer;
    use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

    use majit_ir::operand::Operand;

    use super::OptHeap;
    use crate::history::test_support::{rooted_inputarg_operand, rooted_resop_operand};

    /// oparser-faithful drop-in for `Operand::from_opref(o)` at op-arg /
    /// fail-arg sites where `o` is a bound-at-runtime `OpRef`. Constants shed
    /// to `Operand::Const` (no producer), so they keep the bare `from_opref`
    /// path; `InputArg*` / `{Int,Float,Ref}Op` position-only refs bind to a
    /// rooted synthetic producer (`Operand::InputArg` / `Operand::Op`) so they
    /// never mint the position-only `Operand::Box`. `to_opref()` is identical
    /// to `from_opref(o)` either way, preserving every position-keyed
    /// assertion / cache key / trace-inputarg auto-seed.
    fn bound_arg(o: OpRef) -> Operand {
        match o {
            OpRef::InputArgInt(i) => rooted_inputarg_operand(Type::Int, i),
            OpRef::InputArgFloat(i) => rooted_inputarg_operand(Type::Float, i),
            OpRef::InputArgRef(i) => rooted_inputarg_operand(Type::Ref, i),
            OpRef::IntOp(p) => rooted_resop_operand(Type::Int, p),
            OpRef::FloatOp(p) => rooted_resop_operand(Type::Float, p),
            OpRef::RefOp(p) => rooted_resop_operand(Type::Ref, p),
            OpRef::VoidOp(p) => rooted_resop_operand(Type::Void, p),
            // Const* (shed to Operand::Const), None, TempVar: no producer to
            // bind — the bare from_opref path does not mint.
            _ => Operand::from_opref(o),
        }
    }

    /// Test SizeDescr that pretends to wrap a struct with `is_object()` matching
    /// the constructor arg. Mirrors the PyPy `optimizer.py:480` dispatch test
    /// for `parent_descr.is_object()`.
    #[derive(Debug)]
    struct TestSizeDescr {
        index: u32,
        is_object: bool,
    }

    impl Descr for TestSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            self.index
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn is_object(&self) -> bool {
            self.is_object
        }
    }

    /// Single shared parent SizeDescr for all test FieldDescrs. The exact
    /// instance doesn't matter — `ensure_ptr_info_arg0` only reads
    /// `is_object()`. We use a Struct (is_object=false) so the field branch
    /// constructs `PtrInfo::Struct` (the matchless case in
    /// `put_field_back_to_info`, which writes through `structinfo_setfield`
    /// without matching on the variant).
    fn test_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0000,
            is_object: false,
        })
    }

    fn test_object_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0001,
            is_object: true,
        })
    }

    /// Minimal descriptor for tests, identified by its index. Implements
    /// `FieldDescr` with a synthetic Struct parent so the optimizer's
    /// `ensure_ptr_info_arg0` field branch can dispatch correctly.
    ///
    /// `ei_index` mirrors PyPy production field descrs (`history.py:498
    /// FieldDescr.ei_index`); default `u32::MAX` matches PyPy
    /// `effectinfo.py descr.ei_index = sys.maxint` for descrs absent
    /// from any EI's raw set. Tests that need a specific `ei_index`
    /// (i.e., the descr appears in some EI's `_*_descrs_*` raw set) call
    /// `descr.set_ei_index(N)` before constructing the EI, mirroring
    /// `effectinfo.py:526 descr.ei_index = mapping.setdefault(...)`.
    #[derive(Debug)]
    struct TestDescr {
        index: u32,
        ei_index: AtomicU32,
    }

    impl Descr for TestDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn get_ei_index(&self) -> u32 {
            self.ei_index.load(Ordering::Relaxed)
        }
        fn set_ei_index(&self, idx: u32) {
            self.ei_index.store(idx, Ordering::Relaxed);
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_parent_descr())
        }
        // The declared `offset()` already places this descr at slot `index` of
        // `test_parent_descr()`; `index_in_parent` must agree, because
        // `info.py getfield _fields[fielddescr.get_index()]` indexes the
        // struct's PtrInfo by it. Leaving the trait default (0) made every
        // fixture descr claim slot 0 of the SAME parent, so two distinct
        // descrs aliased one PtrInfo field slot — a layout no real struct can
        // have.
        fn index_in_parent(&self) -> usize {
            self.index as usize
        }
        fn offset(&self) -> usize {
            self.index as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
    }

    #[derive(Debug)]
    struct ObjectTestDescr {
        index: u32,
        ei_index: AtomicU32,
    }

    impl Descr for ObjectTestDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn get_descr_index(&self) -> i32 {
            self.index as i32
        }
        fn get_ei_index(&self) -> u32 {
            self.ei_index.load(Ordering::Relaxed)
        }
        fn set_ei_index(&self, idx: u32) {
            self.ei_index.store(idx, Ordering::Relaxed);
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for ObjectTestDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_object_parent_descr())
        }
        fn offset(&self) -> usize {
            self.index as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
    }

    /// Descriptor for immutable (green) fields. `is_always_pure()` returns true.
    #[derive(Debug)]
    struct ImmutableDescr {
        index: u32,
        ei_index: AtomicU32,
    }

    impl Descr for ImmutableDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn get_ei_index(&self) -> u32 {
            self.ei_index.load(Ordering::Relaxed)
        }
        fn set_ei_index(&self, idx: u32) {
            self.ei_index.store(idx, Ordering::Relaxed);
        }

        fn is_always_pure(&self) -> bool {
            true
        }

        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for ImmutableDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_parent_descr())
        }
        // See TestDescr::index_in_parent.
        fn index_in_parent(&self) -> usize {
            self.index as usize
        }
        fn offset(&self) -> usize {
            self.index as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
        fn is_immutable(&self) -> bool {
            true
        }
    }

    fn descr(idx: u32) -> DescrRef {
        Arc::new(TestDescr {
            index: idx,
            ei_index: AtomicU32::new(u32::MAX),
        })
    }

    fn immutable_descr(idx: u32) -> DescrRef {
        Arc::new(ImmutableDescr {
            index: idx,
            ei_index: AtomicU32::new(u32::MAX),
        })
    }

    fn object_descr(idx: u32) -> DescrRef {
        Arc::new(ObjectTestDescr {
            index: idx,
            ei_index: AtomicU32::new(u32::MAX),
        })
    }

    /// A parent that publishes the positional list its fields' `index_in_parent`
    /// numbers are positions in. `field_slot_index` reads the slot off that
    /// list, so a parent without one hands every field the same fallback and
    /// the tests below would stop exercising the slot they name.
    #[derive(Debug)]
    struct ListedSizeDescr {
        rows: Vec<Arc<dyn FieldDescr>>,
    }

    impl Descr for ListedSizeDescr {
        fn index(&self) -> u32 {
            0xFFFF_0000
        }
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for ListedSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            0xFFFF_0000
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn is_object(&self) -> bool {
            false
        }
        fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
            &self.rows
        }
    }

    /// Eight `Ref` slots at `i * 8`, the shape `ParentIndexedDescr` describes.
    fn listed_parent_descr() -> DescrRef {
        Arc::new(ListedSizeDescr {
            rows: (0..8u32)
                .map(|parent_idx| {
                    Arc::new(ParentIndexedDescr { parent_idx }) as Arc<dyn FieldDescr>
                })
                .collect(),
        })
    }

    #[derive(Debug)]
    struct ParentIndexedDescr {
        parent_idx: u32,
    }

    impl Descr for ParentIndexedDescr {
        fn index(&self) -> u32 {
            0
        }

        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for ParentIndexedDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(listed_parent_descr())
        }

        fn offset(&self) -> usize {
            self.parent_idx as usize * 8
        }

        fn field_size(&self) -> usize {
            8
        }

        fn field_type(&self) -> Type {
            Type::Ref
        }

        fn index_in_parent(&self) -> usize {
            self.parent_idx as usize
        }
    }

    /// The header words are outside the positional census: `heaptracker.py:64-66`
    /// skips `typeptr` by name, so the mint has no position to give them and
    /// hands over the placeholder `0`. Slot `0` is the first slot the parent
    /// DOES enumerate, so answering with the placeholder makes a `Ref` header
    /// read resolve into whatever bank that field last stored — the shape
    /// `make_equal_to`'s `Box.type` invariant fires on.
    #[derive(Debug)]
    struct HeaderWordDescr;

    impl Descr for HeaderWordDescr {
        fn index(&self) -> u32 {
            // The disjoint key an unnumbered field is minted with.
            0x1000_0049
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for HeaderWordDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(listed_parent_descr())
        }
        fn field_name(&self) -> &str {
            "pyobject::PyObject.w_class"
        }
        fn offset(&self) -> usize {
            4
        }
        fn field_size(&self) -> usize {
            4
        }
        fn field_type(&self) -> Type {
            Type::Ref
        }
        fn index_in_parent(&self) -> usize {
            0
        }
    }

    #[test]
    fn a_header_word_does_not_share_the_first_enumerated_slot() {
        let header: DescrRef = Arc::new(HeaderWordDescr);
        let first_enumerated: DescrRef = Arc::new(ParentIndexedDescr { parent_idx: 0 });

        assert_eq!(header.as_field_descr().unwrap().index_in_parent(), 0);
        assert_eq!(
            first_enumerated.as_field_descr().unwrap().index_in_parent(),
            0
        );
        assert_ne!(
            OptHeap::field_slot_index(&header),
            OptHeap::field_slot_index(&first_enumerated),
        );
    }

    #[test]
    fn field_cache_is_keyed_by_descr_identity_not_field_slot() {
        let descr_a: DescrRef = Arc::new(ParentIndexedDescr { parent_idx: 1 });
        let descr_b: DescrRef = Arc::new(ParentIndexedDescr { parent_idx: 1 });

        assert_eq!(OptHeap::field_slot_index(&descr_a), 1);
        assert_eq!(OptHeap::field_slot_index(&descr_b), 1);
        assert_ne!(
            OptHeap::field_cache_identity(&descr_a),
            OptHeap::field_cache_identity(&descr_b)
        );

        let mut heap = OptHeap::new();
        heap.cache_field(&rooted_resop_operand(Type::Int, 0), &descr_a);
        heap.cache_field(&rooted_resop_operand(Type::Int, 0), &descr_b);

        assert_eq!(heap.cached_fields.len(), 2);
    }

    /// heap.py `serialize_optheap` drops an unforced `PreambleOp`:
    /// `AbstractValue.get_box_replacement` returns the wrapper, which is
    /// neither a Const nor a livebox. Exporting `PreambleOp.op` (a Const)
    /// instead lets a bridge inherit a heap fact the short preamble never
    /// replayed.
    #[test]
    fn export_cached_fields_skips_unforced_preamble_op() {
        use crate::optimizeopt::info::PreambleOp;
        use majit_ir::GcRef;

        let descr = object_descr(3);
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::new(256);
        let obj = rooted_resop_operand(Type::Ref, 10);
        ctx.seed_boxes_canonical(std::slice::from_ref(&obj));
        ctx.set_ptr_info(&obj, PtrInfo::instance(None, None));
        let const_code = OpRef::const_ptr(GcRef(0xaaa));
        let mut replay = Op::with_descr(
            OpCode::GetfieldGcR,
            std::slice::from_ref(&obj),
            descr.clone(),
        );
        replay.pos().set(OpRef::ref_op(11));
        ctx.with_ptr_info_mut(&obj, |info| {
            info.set_preamble_field(
                OptHeap::field_slot_index(&descr),
                PreambleOp {
                    op: Operand::from_opref(const_code),
                    invented_name: false,
                    preamble_op: OpRc::new(replay),
                    same_as_source: None,
                },
            );
        })
        .unwrap();
        heap.cache_field(&obj, &descr);

        let exported = heap.export_cached_fields(&mut ctx);
        assert!(
            exported.is_empty(),
            "unforced PreambleOp must not serialize as its Const payload, got {exported:?}"
        );

        ctx.with_ptr_info_mut(&obj, |info| {
            info.setfield(
                OptHeap::field_slot_index(&descr),
                Operand::from_opref(const_code),
            );
        })
        .unwrap();
        let exported = heap.export_cached_fields(&mut ctx);
        assert_eq!(
            exported.len(),
            1,
            "a forced Const field value is serializable"
        );
        assert_eq!(exported[0].2, const_code);
    }

    fn initialize_imported_short_heap_field(
        heap: &mut OptHeap,
        ctx: &mut OptContext,
        object: &Operand,
        descr: &DescrRef,
        source: OpRef,
        resolved: OpRef,
        opcode: OpCode,
    ) {
        use crate::optimizeopt::info::{PreambleOp, PtrInfo};

        let object_ref = object.to_opref();
        ctx.seed_boxes_canonical(std::slice::from_ref(object));
        let mut preamble_op = Op::with_descr(opcode, &[object.clone()], descr.clone());
        preamble_op.pos().set(source);
        ctx.initialize_imported_short_preamble_builder(
            &[object_ref, resolved],
            &[object_ref, resolved],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: OpRc::new(preamble_op.clone()),
                source_op: None,
                res: bound_arg(source),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            }],
        );
        let object_box = object.clone();
        ctx.set_ptr_info(&object_box, PtrInfo::instance(None, None));
        ctx.with_ptr_info_mut(&object_box, |info| {
            info.set_preamble_field(
                OptHeap::field_slot_index(descr),
                PreambleOp {
                    op: bound_arg(source),
                    invented_name: false,
                    preamble_op: OpRc::new(preamble_op),
                    same_as_source: None,
                },
            );
        })
        .unwrap();
        let _ = resolved;
        heap.import_cached_fields(&[(object_ref, descr.clone(), resolved)], ctx);
    }

    /// Call descriptor with default EffectInfo (non-random, non-elidable).
    /// Test helper for "residual call with unknown heap effects". Mirrors
    /// PyPy `effectinfo.MOST_GENERAL` (`effectinfo.py`):
    /// `extraeffect=EF_RANDOM_EFFECTS`, all six raw sets `None`, all six
    /// bitstrings `None`, `can_invalidate=True`. Production sites and
    /// tests use this whenever no analyzer info is available — invalidation
    /// flows through `emit_residual_call` / `handle_side_effects`'s
    /// `call_has_random_effects == true → clean_caches` branch, mirroring
    /// `heap.py:551`'s top-level
    /// random-effects branch. The previous saturated-bitstring fallback
    /// (`CanRaise + raw=Some(empty) + bitstring=Some(0xff;8)`) was a
    /// pyre-only shape PyPy never produces — `effectinfo_from_writeanalyze`
    /// (`effectinfo.py`) force-promotes top_set inputs to RandomEffects
    /// before constructing the EI.
    fn plain_call_descr(idx: u32) -> DescrRef {
        Arc::new(majit_ir::SimpleCallDescr::new(
            idx,
            vec![],
            majit_ir::Type::Void,
            false,
            0,
            EffectInfo::MOST_GENERAL,
        ))
    }

    /// Helper: assign sequential positions to ops.  Also attaches a
    /// fresh `ResumeGuardDescr` to every guard op that lacks one, mirroring
    /// RPython's `optimizer.py:691 assert isinstance(last_descr,
    /// compile.ResumeGuardDescr)` invariant — every guard reaching the
    /// optimizer carries a head-of-chain descr, so `_copy_resume_data_from`
    /// can share via `make_resume_guard_copied_descr(prev)` without panicking
    /// on a missing donor.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            // Type-tag op.pos via the result-type-aware factory
            // so the OpRef variant carries `Box.type` (history.py:220 +
            // resoperation.py:1693) at priority 0 in `opref_type`. The
            // intrinsic `op.type_` field set at `Op::new` is the
            // authoritative source, surfaced via `Op::result_type`.
            op.pos().set(OpRef::op_typed(i as u32, op.result_type()));
            if op.opcode.is_guard() && !op.has_descr() {
                op.setdescr(crate::compile::make_resume_guard_descr_typed(Vec::new()));
            }
        }
    }

    /// Run a single OptHeap pass over the given ops.
    ///
    /// Uses num_inputs=1024 so that high-numbered OpRef values used as
    /// input arguments in tests (e.g. OpRef::int_op(100), OpRef::int_op(500)) are treated
    /// as valid defined positions by the optimizer's undefined-ref filter.
    ///
    /// In production every recorded Box carries its intrinsic type via
    /// `trace_inputargs`, so the preamble exporter can recover a
    /// renamed inputarg's type without guessing. Unit-test inputs are
    /// anonymous stand-ins, so we seed Ref for every slot — the only
    /// use of the type in these tests is to mint typed renamed inputarg
    /// OpRefs, and Ref keeps heap/aliasing tests on the same path
    /// RPython exercises for pointer Boxes.
    fn run_heap_opt(ops: &mut [Op]) -> Vec<Op> {
        run_heap_opt_typed(ops, &[])
    }

    /// Like `run_heap_opt`, but declares specific OpRef slots as Int-typed.
    /// Use for tests whose anonymous high-numbered Boxes are bound to
    /// int-typed fields (setfield_gc value, int_ args, etc.). Without this
    /// the `trace_inputargs = vec![Ref; 1024]` default types every
    /// anonymous slot as Ref and the heap cache's MUST_ALIAS replacement
    /// `make_equal_to(getfield_gc_i.pos:Int, cached_value:Ref)` trips the
    /// `make_equal_to` cross-type assertion introduced for the Box.type
    /// invariant.
    fn run_heap_opt_typed(ops: &mut [Op], int_slots: &[u32]) -> Vec<Op> {
        assign_positions(ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptHeap::new()));
        let mut types = vec![Type::Ref; 1024];
        for &idx in int_slots {
            types[idx as usize] = Type::Int;
        }
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&types);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(ops);
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024)
    }

    // ── Test 1: SETFIELD then GETFIELD → read from cache ──

    #[test]
    fn test_setfield_then_getfield_cached() {
        // setfield_gc(p0, i1, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d0)   <- should be eliminated, replaced by i1
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Int, 101),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            // Terminate trace so lazy set is forced.
            Op::new(OpCode::Jump, &[]),
        ];
        // OpRef::input_arg_int(101) is the Int value stored into the Int-typed field; the
        // cache replays it as the GetfieldGcI result.
        let result = run_heap_opt_typed(&mut ops, &[101]);

        // force_all_lazy_setfields emits the lazy SetfieldGc before Jump.
        // GetfieldGcI is eliminated (replaced by cached i1). SetfieldGc + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    #[test]
    fn test_imported_short_cached_fields_replays_into_heap() {
        let d = object_descr(55);
        let mut heap = OptHeap::new();
        // `inputarg_from_tp(arg.type)` per opencoder.py:259 — base is a Ref
        // pointer, cached_value is the Int field result of GetfieldGcI.
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref, Type::Int]);
        let p0 = OpRef::input_arg_typed(0, Type::Ref);
        let p0_box = bound_arg(p0);
        let p1 = OpRef::input_arg_typed(1, Type::Int);
        let cached_op_key = OpRef::int_op(100);
        initialize_imported_short_heap_field(
            &mut heap,
            &mut ctx,
            &p0_box,
            &d,
            cached_op_key,
            p1,
            OpCode::GetfieldGcI,
        );

        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[p0_box.clone()], d);
        op.pos().set(pos2);

        let op_rc = OpRc::new(op.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));
        let result = heap.optimize_getfield(&op, &op_rc, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_replacement_opref(pos2), p1);
    }

    /// shortpreamble.py `HeapOp.add_op_to_short` builds its replay op on
    /// `sb.produce_arg(sop.getarg(0))`, so a heap short box whose receiver is
    /// another short box reads that dependency's replay op.  Binding the
    /// receiver to the body box instead makes `use_box` append the body op,
    /// whose own args the short preamble never produces.
    #[test]
    fn test_imported_heap_short_box_receiver_is_the_dependency_replay_op() {
        use crate::optimizeopt::shortpreamble::{PreambleOpKind, ProducedShortOp};

        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let p0 = OpRef::input_arg_typed(0, Type::Ref);
        let p0_box = bound_arg(p0);
        ctx.seed_boxes_canonical(std::slice::from_ref(&p0_box));
        let outer_pos = OpRef::ref_op(100);
        let inner_pos = OpRef::ref_op(101);
        let outer_box = ctx.materialize_operand_at(outer_pos);
        let inner_box = ctx.materialize_operand_at(inner_pos);
        let heap_entry = |op: Op, res: &Operand| ProducedShortOp {
            kind: PreambleOpKind::Heap,
            res: res.clone(),
            source_op: OpRc::new(op.clone()),
            preamble_op: OpRc::new(op),
            invented_name: false,
            same_as_source: None,
            label_arg_idx: None,
        };
        let mut outer = Op::with_descr(OpCode::GetfieldGcR, &[p0_box], object_descr(1));
        outer.pos().set(outer_pos);
        let mut inner = Op::with_descr(OpCode::GetfieldGcR, &[outer_box.clone()], object_descr(2));
        inner.pos().set(inner_pos);
        let short_boxes = vec![
            (outer_pos, heap_entry(outer, &outer_box)),
            (inner_pos, heap_entry(inner, &inner_box)),
        ];
        let mut result_map = indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        for &(source, _) in &short_boxes {
            result_map.insert(source, ctx.alloc_op_position_typed(Type::Ref));
        }

        assert!(
            ctx.initialize_imported_short_preamble_builder_from_short_boxes(
                &[p0],
                &[p0],
                &short_boxes,
                &result_map,
                &indexmap::IndexMap::new(),
            )
        );
        let builder = ctx.imported_short_preamble_builder.as_ref().unwrap();
        let outer_replay = builder.produced_short_op(&outer_box).unwrap().preamble_op;
        let inner_replay = builder.produced_short_op(&inner_box).unwrap().preamble_op;
        let receiver = inner_replay.arg(0).bound_op().unwrap();
        assert!(OpRc::ptr_eq(&receiver, &outer_replay));
    }

    /// After consuming an imported short field, a cache invalidation followed
    /// by another getfield must emit the actual load (not reuse the stale
    /// preamble value).  This prevents null-pointer crashes when the
    /// preamble's cached value (e.g. a linked-list head) is no longer valid
    /// after a call/setfield that empties the container.
    #[test]
    fn test_imported_short_field_not_reused_after_invalidation() {
        let d_head = object_descr(10); // head field
        let mut heap = OptHeap::new();
        // `inputarg_from_tp(arg.type)` per opencoder.py:259 — base and
        // cached_value are both Ref (GetfieldGcR result type).
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref, Type::Ref]);
        let p0 = OpRef::input_arg_typed(0, Type::Ref);
        let p0_box = bound_arg(p0);
        let p1 = OpRef::input_arg_typed(1, Type::Ref);
        let cached_op_key = OpRef::int_op(100);
        initialize_imported_short_heap_field(
            &mut heap,
            &mut ctx,
            &p0_box,
            &d_head,
            cached_op_key,
            p1,
            OpCode::GetfieldGcR,
        );

        // First getfield on head: consumes the import, caches the value.
        let pos2 = ctx.reserve_pos_typed(Type::Ref);
        let mut op1 = Op::with_descr(OpCode::GetfieldGcR, &[p0_box.clone()], d_head.clone());
        op1.pos().set(pos2);
        let op1_rc = OpRc::new(op1.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op1_rc));
        let result1 = heap.optimize_getfield(&op1, &op1_rc, &mut ctx);
        assert!(matches!(result1, OptimizationResult::Remove));
        assert_eq!(ctx.get_replacement_opref(pos2), p1);

        // A call invalidates all mutable field caches.
        heap.clean_caches(&mut ctx);

        // Second getfield on head after invalidation: must NOT return the
        // stale preamble value.  The import was consumed, so it should emit.
        let pos3 = ctx.reserve_pos_typed(Type::Ref);
        let mut op2 = Op::with_descr(OpCode::GetfieldGcR, &[p0_box], d_head.clone());
        op2.pos().set(pos3);
        let result2 = heap.optimize_getfield(&op2, &OpRc::new(op2.clone()), &mut ctx);
        assert!(
            matches!(result2, OptimizationResult::PassOn),
            "getfield after invalidation must emit, not reuse stale import"
        );
    }

    #[test]
    #[should_panic(expected = "must be a gcref")]
    fn test_getfield_does_not_deref_arbitrary_int_constant_base() {
        // `optimizer.py protect_speculative_operation` derefs
        // `op.getarg(0)` via `getref_base()` — upstream `ConstInt`
        // does not expose that method and would `AttributeError`.
        // RPython's type-typed `AbstractValue` makes this state
        // unrepresentable at construction time: a `GETFIELD_GC_I`
        // whose first arg is an `Int` cannot exist.  Pyre's flat
        // `Value` enum allows the misuse syntactically, but per
        // strict-orthodoxy parity the optimizer panics instead of
        // emitting a defensive fallback.
        let d = immutable_descr(77);
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Int]);
        // The receiver is a ConstInt. A SameAs stand-in made constant by
        // position would be a different object from the arg the op carries.
        let const_base = Operand::const_from_value(majit_ir::Value::Int(1));

        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[const_base], d);
        op.pos().set(pos1);
        op.setarg(
            0,
            ctx.resolve_operand_operand_opt(&op.arg(0))
                .expect("constant receiver resolves to an operand"),
        );

        let _ = heap.optimize_getfield(&op, &OpRc::new(op.clone()), &mut ctx);
    }

    /// Same shape as `test_getfield_read_after_read`, but through a descr that
    /// owns a parent SizeDescr and sits at a non-zero slot. The
    /// parentless `TestDescr` used by the sibling test takes
    /// `field_slot_index`'s `Descr::index()` fallback, so it never exercises
    /// the `index_in_parent` slot that `ensure_ptr_info_arg0` sizes the
    /// `StructPtrInfo` for at `optimizer.py init_fields`.
    #[test]
    fn getfield_read_after_read_folds_through_a_parented_descr() {
        let d: DescrRef = Arc::new(ParentIndexedDescr { parent_idx: 7 });
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcR,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcR,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(
            result.len(),
            2,
            "second read of the same parented field must reuse the first: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
        assert_eq!(result[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    /// The frame-enter shape `inline_helper` traces: read `ec` off the frame,
    /// read `ec.topframeref`, read `ec` again, then store through the first
    /// `ec` and read back through the second. Both `ec` reads name the same
    /// receiver and the same descr, so the second must fold; once it does, the
    /// store is visible to the read-back and the pair can cancel.
    #[test]
    fn getfield_folds_across_an_intervening_read_of_its_own_result() {
        let d_ec: DescrRef = Arc::new(ParentIndexedDescr { parent_idx: 7 });
        let d_tf: DescrRef = Arc::new(ParentIndexedDescr { parent_idx: 0 });
        let p0 = rooted_inputarg_operand(Type::Ref, 100);
        let stored = rooted_inputarg_operand(Type::Ref, 101);
        // `assign_positions` numbers ops by index, so op 0's result is RefOp(0).
        let ec1 = bound_arg(OpRef::ref_op(0));
        let ec2 = bound_arg(OpRef::ref_op(2));
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcR, &[p0.clone()], d_ec.clone()),
            Op::with_descr(OpCode::GetfieldGcR, &[ec1.clone()], d_tf.clone()),
            Op::with_descr(OpCode::GetfieldGcR, &[p0.clone()], d_ec.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[ec1, stored], d_tf.clone()),
            Op::with_descr(OpCode::GetfieldGcR, &[ec2], d_tf),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);
        let opcodes: Vec<_> = result.iter().map(|o| o.opcode).collect();

        let ec_reads = result
            .iter()
            .filter(|o| {
                o.opcode == OpCode::GetfieldGcR
                    && o.getdescr().is_some_and(|dd| {
                        OptHeap::field_cache_identity(&dd) == OptHeap::field_cache_identity(&d_ec)
                    })
            })
            .count();
        assert_eq!(ec_reads, 1, "the second `ec` read must fold: {opcodes:?}");
    }

    /// Drive two identical `getfield_gc_r`s at a receiver that already carries
    /// a given `PtrInfo`, and report whether the second folded.
    fn second_getfield_folds_with_receiver_info(seed: crate::optimizeopt::info::PtrInfo) -> bool {
        let d: DescrRef = Arc::new(ParentIndexedDescr { parent_idx: 7 });
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let p0_box = bound_arg(OpRef::input_arg_typed(0, Type::Ref));
        ctx.seed_boxes_canonical(std::slice::from_ref(&p0_box));
        ctx.set_ptr_info(&p0_box, seed);

        let mut folded = false;
        for _ in 0..2 {
            let pos = ctx.reserve_pos_typed(Type::Ref);
            let mut op = Op::with_descr(OpCode::GetfieldGcR, &[p0_box.clone()], d.clone());
            op.pos().set(pos);
            let op_rc = OpRc::new(op.clone());
            ctx.bind_input_resops(std::slice::from_ref(&op_rc));
            folded = matches!(
                heap.optimize_getfield(&op, &op_rc, &mut ctx),
                OptimizationResult::Remove
            );
        }
        folded
    }

    /// An ordinary heap field read off a *virtualizable* receiver must still be
    /// cached. Upstream has no `VirtualizablePtrInfo` — `info.py`'s hierarchy
    /// stops at `InstancePtrInfo` / `StructPtrInfo` — so a virtualizable frame
    /// carries a plain `InstancePtrInfo` and `optimizer.py init_fields`
    /// gives every declared slot a home. pyre's extra `PtrInfo::Virtualizable`
    /// variant has no `setfield` / `getfield` arm (`ptr_info.rs` falls through
    /// to `_ => {}` / `_ => None`), and `ensure_ptr_info_arg0` early-returns on
    /// it instead of upgrading, so the write is dropped and every later read of
    /// the same field misses.
    ///
    /// Inline-frame helpers exercise the same general shape for ordinary heap
    /// fields read from a virtualizable receiver; those reads must still fold
    /// exactly as they do on a plain `InstancePtrInfo` receiver.
    #[test]
    fn getfield_off_a_virtualizable_receiver_is_still_cached() {
        use crate::optimizeopt::info::{PtrInfo, VirtualizableFieldState};

        assert!(
            second_getfield_folds_with_receiver_info(PtrInfo::instance(None, None)),
            "control: a non-virtualizable receiver must fold the second read"
        );
        assert!(
            second_getfield_folds_with_receiver_info(PtrInfo::Virtualizable(
                VirtualizableFieldState {
                    fields: Vec::new(),
                    field_descrs: Vec::new(),
                    arrays: Vec::new(),
                    heap_fields: Vec::new(),
                    last_guard_pos: -1,
                }
            )),
            "an ordinary field read off a virtualizable receiver must be cached too"
        );
    }

    // ── Test 2: Two GETFIELDs on same object/field → second eliminated ──

    #[test]
    fn test_getfield_read_after_read() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d0)   <- eliminated, reuse i1
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only the first GETFIELD + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 3: SETFIELD then SETFIELD → first eliminated (write-after-write) ──

    #[test]
    fn test_setfield_write_after_write() {
        // setfield_gc(p0, i1, descr=d0)
        // setfield_gc(p0, i2, descr=d0)   <- first is dead
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 101),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 102),
                ],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // First SetfieldGc is dead (overwritten). Second is emitted as lazy set before Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 4: SETFIELD then CALL then GETFIELD → cache invalidated ──

    #[test]
    fn test_setfield_call_invalidates_cache() {
        // setfield_gc(p0, i1, descr=d0)
        // call_n(...)
        // i2 = getfield_gc_i(p0, descr=d0)   <- cache invalidated by call, must emit
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 101),
                ],
                d.clone(),
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // force_all_lazy at call emits SetfieldGc + invalidates caches.
        // SetfieldGc + CALL + GETFIELD (re-emitted, cache was invalidated) + Jump.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 5: SETFIELD on different objects → both cached independently ──

    #[test]
    fn test_setfield_different_objects() {
        // setfield_gc(p0, i1, descr=d0)
        // setfield_gc(p1, i2, descr=d0)  <- possible_aliasing: forces first lazy_set
        // i3 = getfield_gc_i(p0, descr=d0)   <- cached from forced set (i1)
        // i4 = getfield_gc_i(p1, descr=d0)   <- cached from second lazy_set (i2)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Int, 101),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 200),
                    rooted_inputarg_operand(Type::Int, 201),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt_typed(&mut ops, &[101, 201]);

        // RPython CachedField per-descr with aliasing analysis:
        // - setfield(p1) forces lazy_set(p0) → emit SETFIELD(p0), put_back p0
        // - invalidate_for_write(p1) removes p0 (input args can alias)
        // - getfield(p0): lazy_set is p1, UNKNOWN_ALIAS → force lazy_set(p1)
        //   → emit SETFIELD(p1), put_back p1. p0 entry gone → cache miss → emit GETFIELD(p0)
        // - getfield(p1): entry p1=i2 from put_back → cache hit → remove
        // Result: SETFIELD(p0) + SETFIELD(p1) + GETFIELD(p0) + Jump.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
        assert_eq!(result[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    #[test]
    fn test_setfield_on_distinct_array_loads_forces_prior_lazy_set() {
        // Same shape as `test_setfield_different_objects`, but the two structs
        // arrive as loads out of one array instead of as separate inputargs:
        //
        //   p2 = getarrayitem_gc_r(p0, 7)
        //   p3 = getarrayitem_gc_r(p0, 16)
        //   setfield_gc(p2, i1, descr=d1)   <- lazy
        //   setfield_gc(p3, i2, descr=d1)   <- heap.py possible_aliasing:
        //                                      p2 != p3, so p2's lazy set must
        //                                      be forced before this one takes
        //                                      the descr's single `_lazy_set`
        //                                      slot
        //   guard_true(i_cond)              <- heap.py:608-637 forces p3's
        //
        // Dropping the p2 store instead loses a write the trace recorded.
        let d_arr = descr(0);
        let d_field = descr(1);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcR,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    Operand::const_from_value(majit_ir::Value::Int(7)),
                ],
                d_arr.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcR,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    Operand::const_from_value(majit_ir::Value::Int(16)),
                ],
                d_arr.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 0),
                    rooted_inputarg_operand(Type::Int, 101),
                ],
                d_field.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 1),
                    rooted_inputarg_operand(Type::Int, 201),
                ],
                d_field.clone(),
            ),
            Op::new(
                OpCode::GuardTrue,
                &[rooted_inputarg_operand(Type::Int, 200)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt_typed(&mut ops, &[101, 200, 201]);

        let setfields: Vec<&Op> = result
            .iter()
            .filter(|o| o.opcode == OpCode::SetfieldGc)
            .collect();
        assert_eq!(
            setfields.len(),
            2,
            "both stores must survive; emitted = {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>(),
        );
        let guard_at = result
            .iter()
            .position(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard must be emitted");
        let last_setfield_at = result
            .iter()
            .rposition(|o| o.opcode == OpCode::SetfieldGc)
            .unwrap();
        assert!(
            last_setfield_at < guard_at,
            "both stores must be emitted before the guard they precede",
        );
    }

    /// heap.py `emit()` runs `emitting_operation(op)` — for a guard,
    /// `force_lazy_sets_for_guard` — BEFORE `emit_postponed_op()`. That order is
    /// what keeps a postponed op adjacent to the guard that reads it:
    ///
    ///   setfield_gc(p0, i1, descr=d)   <- lazy
    ///   i3 = int_gt(i2, 10)            <- heap.py:420-423 postpones comparisons
    ///   guard_true(i3)
    ///
    /// must emit `[setfield_gc][int_gt][guard_true]`. Emitting the forced set
    /// between the comparison and its guard costs the fusion that
    /// `aarch64/assembler.py` performs via `next_op_can_accept_cc`.
    #[test]
    fn test_lazy_set_flushes_before_a_postponed_comparison_not_between_it_and_its_guard() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Int, 101),
                ],
                d.clone(),
            ),
            Op::new(
                OpCode::IntGt,
                &[
                    rooted_inputarg_operand(Type::Int, 102),
                    Operand::const_from_value(majit_ir::Value::Int(10)),
                ],
            ),
            Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 1)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt_typed(&mut ops, &[101, 102]);

        let emitted: Vec<OpCode> = result.iter().map(|o| o.opcode).collect();

        // Pin the whole sequence, not just the relative order. The store has a
        // non-virtual target (an inputarg pointer, so OptVirtualize never claims
        // it and it does reach `cached_fields`) and a non-virtual stored value
        // (an inputarg int, so heap.py:618-620 `is_virtual(op.getarg(1))` is
        // false and `force_lazy_set` EMITS it rather than parking it in
        // `pendingfields`). Both are required for this to be a real control:
        // with a virtual on either side no SETFIELD_GC is emitted at all and the
        // test would pass under either ordering.
        assert_eq!(
            emitted,
            vec![
                OpCode::SetfieldGc,
                OpCode::IntGt,
                OpCode::GuardTrue,
                OpCode::Jump,
            ],
            "heap.py:417-418: the guard's forced lazy set is emitted BEFORE the \
             postponed comparison, leaving the comparison adjacent to the guard \
             that tests it (aarch64/assembler.py:1187-1190 next_op_can_accept_cc)",
        );
    }

    /// A `SizeDescr` for a `NEW` whose result stays virtual: `OptVirtualize`
    /// indexes `VirtualInfo._fields` by `field_slot_index(descr)` and asserts
    /// the slot is inside `all_fielddescrs()`, so the bare `TestSizeDescr`
    /// (empty list) cannot host a field.
    #[derive(Debug)]
    struct TestVirtualSizeDescr {
        index: u32,
        all_fielddescrs: Vec<Arc<dyn FieldDescr>>,
    }

    impl Descr for TestVirtualSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestVirtualSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            self.index
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn is_object(&self) -> bool {
            false
        }
        fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
            &self.all_fielddescrs
        }
    }

    fn virtual_size_descr(index: u32, fields: u32) -> DescrRef {
        Arc::new(TestVirtualSizeDescr {
            index,
            all_fielddescrs: (0..fields)
                .map(|i| {
                    Arc::new(TestDescr {
                        index: i,
                        ei_index: AtomicU32::new(u32::MAX),
                    }) as Arc<dyn FieldDescr>
                })
                .collect(),
        })
    }

    /// Like `run_heap_opt_typed`, but the whole `__init__.py:15-22` chain, so
    /// `OptVirtualize` is present and a `NEW` result can still be virtual when
    /// the heap pass reaches the guard.
    fn run_default_pipeline_typed(ops: &mut [Op], int_slots: &[u32]) -> Vec<Op> {
        assign_positions(ops);
        let mut opt = Optimizer::default_pipeline();
        let mut types = vec![Type::Ref; 1024];
        for &idx in int_slots {
            types[idx as usize] = Type::Int;
        }
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&types);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(ops);
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024)
    }

    #[test]
    fn test_guard_forces_size_store_paired_with_virtual_head_store() {
        // heap.py:617-621 splits the two lazy sets a linked-list push leaves
        // pending, by whether the stored VALUE is virtual:
        //
        //   p2 = new(descr=size)
        //   setfield_gc(p2, i_val, descr=d_value)
        //   setfield_gc(p0, p2, descr=d_head)   <- value is still VIRTUAL, so
        //                                          heap.py:618-619 routes it to
        //                                          pendingfields and leaves the
        //                                          lazy set in place
        //   i1 = getfield_gc_i(p0, descr=d_size)
        //   i2 = int_add(i1, 1)
        //   setfield_gc(p0, i2, descr=d_size)   <- non-virtual, so heap.py:621
        //                                          FORCES it at the guard
        //   guard_true(i_cond)
        //
        // Deopt reconstructs `head` from the guard's pendingfields, so the
        // emitted heap only has to carry the non-virtual half; this pins that
        // half being emitted BEFORE the guard rather than left lazy past it.
        let sd = virtual_size_descr(1, 4);
        // `d_value` is stored INTO the virtual, so its slot has to be one of
        // `sd`'s own fields: `optimize_setfield_gc` indexes
        // `VirtualStruct.fields` by the field descr's slot and
        // `virtualize.rs` asserts it stays inside `all_fielddescrs()`. A
        // `TestDescr`'s slot reads back as `offset() / field_size()` — its
        // index — wherever the descr is reconciled rather than used raw, so
        // `descr(10)` addresses field 10 of a 4-field struct. `d_head` /
        // `d_size` are stored into the non-virtual `p0` and carry no such
        // bound.
        let d_value = descr(0);
        let d_head = descr(11);
        let d_size = descr(12);
        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 0),
                    rooted_inputarg_operand(Type::Int, 101),
                ],
                d_value.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_resop_operand(Type::Ref, 0),
                ],
                d_head.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d_size.clone(),
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 3),
                    Operand::const_from_value(majit_ir::Value::Int(1)),
                ],
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_resop_operand(Type::Int, 4),
                ],
                d_size.clone(),
            ),
            Op::new(
                OpCode::GuardTrue,
                &[rooted_inputarg_operand(Type::Int, 200)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_default_pipeline_typed(&mut ops, &[101, 200]);

        let guard_at = result
            .iter()
            .position(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard must be emitted");
        let size_store_at = result.iter().position(|o| {
            o.opcode == OpCode::SetfieldGc
                && o.getdescr().as_ref().is_some_and(|d| {
                    majit_ir::descr::descr_identity(d) == majit_ir::descr::descr_identity(&d_size)
                })
        });
        assert!(
            size_store_at.is_some_and(|at| at < guard_at),
            "the size store must be forced before the guard; emitted = {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>(),
        );
    }

    // ── Test 6: Array items: SETARRAYITEM then GETARRAYITEM → cached ──

    #[test]
    fn test_setarrayitem_then_getarrayitem_cached() {
        // setarrayitem_gc(p0, i_idx, i_val, descr=d0)
        // i2 = getarrayitem_gc_i(p0, i_idx, descr=d0)   <- eliminated
        // We need i_idx to be a known constant.
        let d = descr(0);
        let arr = rooted_resop_operand(Type::Int, 100);
        let val = rooted_resop_operand(Type::Int, 101);
        let idx = Operand::const_from_value(majit_ir::Value::Int(3));
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[arr.clone(), idx.clone(), val.clone()],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[arr.clone(), idx.clone()],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.seed_boxes_canonical(&[arr, val]);

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // force_all_lazy at Jump drops lazy setarrayitems. Only Jump remains.
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::Jump]);
    }

    /// `heap.py` `optimize_GETARRAYITEM_GC_PURE_I`: a constant-index miss
    /// does not `arrayinfo_setitem`, so a second pure read is not a hit.
    #[test]
    fn test_getarrayitem_gc_pure_constant_miss_is_not_cached() {
        let d = descr(0);
        let idx = OpRef::int_op(50);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcPureI,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, idx.raw()),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcPureI,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, idx.raw()),
                ],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);
        let mut ctx = OptContext::new(ops.len());
        let b = ctx.materialize_operand_at(idx);
        // The index argument is the box the constant is attached to.
        for op in &mut ops[..2] {
            op.setarg(1, b.clone());
        }
        ctx.make_constant_box(&b, majit_ir::Value::Int(3));
        let mut pass = OptHeap::new();
        pass.setup();
        for op in &ops {
            let mut resolved = op.clone();
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            match pass.propagate_forward(&resolved, &OpRc::new(resolved.clone()), &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop(_) => panic!("unexpected InvalidLoop"),
            }
        }
        let gets = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcPureI)
            .count();
        assert_eq!(gets, 2, "a pure constant miss must not cache the read");
    }

    /// A negative constant index is not an `arrayitem_cache` key
    /// (`arrayitem_key` returns `None`), so the read stays on the
    /// variable-index path and is emitted.
    #[test]
    fn arrayitem_key_rejects_a_negative_constant_index() {
        let d = descr(0);
        let idx = OpRef::int_op(50);
        let mut op = Op::with_descr(
            OpCode::GetarrayitemGcPureI,
            &[
                rooted_resop_operand(Type::Int, 100),
                rooted_resop_operand(Type::Int, idx.raw()),
            ],
            d,
        );
        let mut ctx = OptContext::new(1);
        let b = ctx.materialize_operand_at(idx);
        // The index argument is the box the constant is attached to.
        op.setarg(1, b.clone());
        ctx.make_constant_box(&b, majit_ir::Value::Int(-1));
        assert!(OptHeap::arrayitem_key(&op, &mut ctx).is_none());
    }

    /// A variable index forces lazy setarrayitems and emits. It does not
    /// consult `lookup_cached`, so a prior variable write does not remove
    /// the pure read.
    #[test]
    fn test_getarrayitem_gc_pure_variable_index_is_not_cached() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 51),
                    rooted_resop_operand(Type::Int, 101),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcPureI,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 51),
                ],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);
        let mut ctx = OptContext::new(ops.len());
        let mut pass = OptHeap::new();
        pass.setup();
        for op in &ops {
            let mut resolved = op.clone();
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            match pass.propagate_forward(&resolved, &OpRc::new(resolved.clone()), &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop(_) => panic!("unexpected InvalidLoop"),
            }
        }
        assert!(
            ctx.new_operations
                .iter()
                .any(|o| o.opcode == OpCode::GetarrayitemGcPureI),
            "variable-index pure read must be emitted, not folded through the varindex cache"
        );
    }

    /// #171/#11 Approach C invariant: NON-pure `getarrayitem_gc_r` is
    /// invalidated by an intervening `setarrayitem_gc` at the same index, so
    /// a later read MUST observe the WRITTEN value — it is NOT folded to the
    /// earlier read.  This is exactly why list element loads stay non-pure:
    ///
    ///   r1 = getarrayitem_gc_r(p0, i_idx, descr=d0)
    ///   setarrayitem_gc(p0, i_idx, r_new, descr=d0)   <- invalidates the read
    ///   r2 = getarrayitem_gc_r(p0, i_idx, descr=d0)   <- reads r_new, NOT r1
    ///
    /// (A pure load would CSE `r2` to `r1` across the write — the
    /// read-after-write miscompile `pure_listload_raw.py` guards against.)
    #[test]
    fn test_getarrayitem_non_pure_invalidated_by_setarrayitem() {
        let d = descr(0);
        let arr = rooted_resop_operand(Type::Ref, 100);
        let new_val_box = rooted_resop_operand(Type::Ref, 102);
        let idx = Operand::const_from_value(majit_ir::Value::Int(3));
        let mut ops = vec![
            // r1 = getarrayitem_gc_r(p0, i_idx)
            Op::with_descr(
                OpCode::GetarrayitemGcR,
                &[arr.clone(), idx.clone()],
                d.clone(),
            ),
            // setarrayitem_gc(p0, i_idx, r_new)
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[arr.clone(), idx.clone(), new_val_box.clone()],
                d.clone(),
            ),
            // r2 = getarrayitem_gc_r(p0, i_idx)
            Op::with_descr(
                OpCode::GetarrayitemGcR,
                &[arr.clone(), idx.clone()],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);
        let r1_pos = ops[0].pos().get();
        let r2_pos = ops[2].pos().get();

        let mut ctx = OptContext::new(ops.len());
        ctx.seed_boxes_canonical(&[arr, new_val_box.clone()]);

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // The intervening `setarrayitem_gc` invalidated the cached read, so the
        // post-write read r2 is NOT folded to the pre-write read r1.  A pure
        // load would have CSE'd r2 → r1 across the write (the read-after-write
        // miscompile `pure_listload_raw.py` guards against), so this `!=` is
        // the decisive non-pure property that keeps list element loads correct.
        let r2_repl = ctx.get_replacement_opref(r2_pos);
        let _ = new_val_box;
        assert_ne!(
            r2_repl, r1_pos,
            "non-pure getarrayitem must NOT be CSE'd to the pre-write read"
        );
        // A live `GetarrayitemGcR` survives the optimization (the write did not
        // let the optimizer collapse the read away to the stale pre-write box).
        assert!(
            ctx.new_operations
                .iter()
                .any(|o| o.opcode == OpCode::GetarrayitemGcR),
            "a non-pure read must survive across the write"
        );
    }

    /// `array_deopt_resume`: `buf[i%4] = newv; total += buf[i%4]` uses
    /// `GetarrayitemGcI` with a non-const index.  The post-write load
    /// must observe `newv`, not CSE to the pre-write raw.
    #[test]
    fn test_getarrayitem_gc_i_varindex_sees_setarrayitem() {
        let d = descr(0);
        let arr = rooted_resop_operand(Type::Ref, 100);
        let idx = rooted_resop_operand(Type::Int, 50);
        let new_val_box = rooted_resop_operand(Type::Int, 102);
        let new_val = new_val_box.to_opref();
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[arr.clone(), idx.clone()],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[arr.clone(), idx.clone(), new_val_box.clone()],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[arr.clone(), idx.clone()],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);
        let r1_pos = ops[0].pos().get();
        let r2_pos = ops[2].pos().get();

        let mut ctx = OptContext::new(ops.len());
        ctx.seed_boxes_canonical(&[arr, idx, new_val_box]);

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let r2_repl = ctx.get_replacement_opref(r2_pos);
        assert_ne!(
            r2_repl, r1_pos,
            "post-write GetarrayitemGcI must not CSE to the pre-write raw"
        );
        // heap.py cache_varindex_write can replace the load with `new_val`;
        // a surviving GetarrayitemGcI is also correct if SETARRAYITEM emitted.
        assert!(
            r2_repl == new_val
                || ctx
                    .new_operations
                    .iter()
                    .any(|o| o.opcode == OpCode::GetarrayitemGcI),
            "post-write load must be the stored raw or a live GetarrayitemGcI"
        );
    }

    #[test]
    fn test_getarrayitem_postprocess_updates_ptr_info() {
        let d = descr(0);
        let idx = OpRef::int_op(50);
        let array_box = rooted_resop_operand(Type::Ref, 100);
        let index_box = rooted_resop_operand(Type::Int, idx.raw());
        let mut op = Op::with_descr(
            OpCode::GetarrayitemGcI,
            &[array_box.clone(), index_box.clone()],
            d.clone(),
        );
        op.pos().set(OpRef::int_op(200));

        let mut ctx = OptContext::new(256);
        ctx.make_constant_box(&index_box, majit_ir::Value::Int(3));
        ctx.set_ptr_info(&array_box, PtrInfo::virtual_array(d, 8, false));

        let mut pass = OptHeap::new();
        pass.setup();

        let result = pass.propagate_forward(&op, &OpRc::new(op.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
        let arr_box = array_box.get_box_replacement(false);
        assert_eq!(
            ctx.peek_ptr_info(&arr_box)
                .and_then(|info| info.getitem(3))
                .and_then(|e| e.as_opref()),
            Some(OpRef::int_op(200))
        );
    }

    #[test]
    fn test_setarrayitem_postprocess_updates_ptr_info() {
        // heap.py do_setfield common case: only sets `_lazy_set = op`.
        // The ArrayPtrInfo._items[index] write is deferred to
        // force_lazy_set -> put_field_back_to_info; until then a
        // subsequent getarrayitem on the same (array, index) reads
        // the value back via getfield_from_cache's _lazy_set check.
        let d = descr(0);
        let arr = rooted_resop_operand(Type::Int, 100);
        let rhs = rooted_resop_operand(Type::Int, 101);
        let idx = Operand::const_from_value(majit_ir::Value::Int(3));
        let mut op = Op::with_descr(
            OpCode::SetarrayitemGc,
            &[arr.clone(), idx, rhs.clone()],
            d.clone(),
        );

        let mut ctx = OptContext::new(256);
        ctx.seed_boxes_canonical(&[arr.clone(), rhs]);
        ctx.set_ptr_info(&arr, PtrInfo::virtual_array(d.clone(), 8, false));

        let mut pass = OptHeap::new();
        pass.setup();

        let result = pass.propagate_forward(&op, &OpRc::new(op.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        // _lazy_set holds the pending op; PtrInfo is NOT yet written.
        let cai = pass
            .get_cached_array_submap(majit_ir::descr::descr_identity(&d))
            .and_then(|s| s.const_get(3))
            .expect("ArrayCachedItem must exist");
        assert!(
            cai.lazy_set.is_some(),
            "do_setfield should have stored _lazy_set"
        );
        // After flush() the lazy set is forced and PtrInfo._items[3]
        // becomes the rhs value via put_field_back_to_info.
        pass.flush(&mut ctx);
        let arr_box = ctx
            .get_box_replacement_operand_opt(OpRef::int_op(100))
            .expect("array box");
        assert_eq!(
            ctx.peek_ptr_info(&arr_box)
                .and_then(|info| info.getitem(3))
                .and_then(|e| e.as_opref()),
            Some(OpRef::int_op(101))
        );
    }

    // ── Test 7: Guard forces lazy sets ──

    #[test]
    fn test_guard_forces_lazy_setfield() {
        // setfield_gc(p0, i1, descr=d0)     <- lazy, not emitted yet
        // guard_true(i_cond)                <- forces the lazy set
        // i2 = getfield_gc_i(p0, descr=d0) <- still cached (guards don't invalidate)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Int, 101),
                ],
                d.clone(),
            ),
            Op::new(
                OpCode::GuardTrue,
                &[rooted_inputarg_operand(Type::Int, 200)],
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        // resoperation.py InputArgInt — guard_true expects an int
        // condition box (truthiness via INT_IS_TRUE / GUARD_TRUE family).
        let result = run_heap_opt_typed(&mut ops, &[101, 200]);

        // SETFIELD (forced by guard) + GUARD_TRUE + Jump.
        // GETFIELD is eliminated (cache survives guards).
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 8: GUARD_NOT_INVALIDATED deduplication ──

    #[test]
    fn test_guard_not_invalidated_dedup() {
        let mut ops = vec![
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only one GUARD_NOT_INVALIDATED + Jump.
        let gni_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNotInvalidated)
            .count();
        assert_eq!(gni_count, 1);
        assert_eq!(result.last().unwrap().opcode, OpCode::Jump);
    }

    // ── Test 9: Different field descriptors are independent ──

    #[test]
    fn test_different_field_descriptors() {
        // setfield_gc(p0, i1, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d1)   <- different descriptor, NOT cached
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 101),
                ],
                d0,
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d1,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // SetfieldGc(d0) emitted as lazy set + GETFIELD(d1, different descriptor) + Jump.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 10: SETFIELD_RAW does not affect GC caches ──

    #[test]
    fn test_setfield_raw_no_effect_on_gc_cache() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // setfield_raw(p1, i2, descr=d1)     <- RAW, no effect on GC caches
        // i3 = getfield_gc_i(p0, descr=d0)   <- still cached from first read
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldRaw,
                &[
                    rooted_inputarg_operand(Type::Ref, 200),
                    rooted_inputarg_operand(Type::Ref, 201),
                ],
                d1,
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + SETFIELD_RAW + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::SetfieldRaw);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_raw_store_no_effect_on_gc_cache() {
        // heap.py: RAW_STORE writes raw memory and must not invalidate an
        // unrelated GC-field cache. Unlike SETFIELD_RAW it reaches the default
        // OptHeap dispatch, which is the catch-all path this test pins.
        let d0 = descr(0);
        let raw_descr = descr(1);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::RawStore,
                &[
                    rooted_inputarg_operand(Type::Ref, 200),
                    rooted_inputarg_operand(Type::Int, 201),
                    rooted_inputarg_operand(Type::Int, 202),
                ],
                raw_descr,
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt_typed(&mut ops, &[201, 202]);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::RawStore);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 11: Writing same value is redundant ──

    #[test]
    fn test_setfield_same_value_redundant() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // setfield_gc(p0, i1, descr=d0)   <- writing back the same value, redundant
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            // assign_positions stamps GETFIELD's pos as IntOp(0); the
            // setfield's value arg must match that variant for the
            // store-load elision to recognize identity.
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_resop_operand(Type::Int, 0),
                ],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + Jump only. SETFIELD removed (writing same value).
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 12: Pure/overflow ops don't invalidate ──

    #[test]
    fn test_pure_ops_dont_invalidate() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // i2 = int_add(i1, i1)              <- pure, no invalidation
        // i3 = getfield_gc_i(p0, descr=d0)  <- still cached
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 0),
                ],
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + INT_ADD + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 13: Ref and float field variants ──

    #[test]
    fn test_getfield_ref_cached() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcR,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcR,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    #[test]
    fn test_getfield_float_cached() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcF,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcF,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcF);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 14: Array write-after-write ──

    #[test]
    fn test_setarrayitem_write_after_write() {
        let d = descr(0);
        let arr = rooted_resop_operand(Type::Int, 100);
        let val1 = rooted_resop_operand(Type::Int, 101);
        let val2 = rooted_resop_operand(Type::Int, 102);
        let idx = Operand::const_from_value(majit_ir::Value::Int(5));
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[arr.clone(), idx.clone(), val1.clone()],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[arr.clone(), idx.clone(), val2.clone()],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.seed_boxes_canonical(&[arr, val1, val2]);

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // force_all_lazy at Jump drops lazy setarrayitems. Only Jump remains.
        let result_opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(result_opcodes, vec![OpCode::Jump]);
    }

    // ── Test 15: Overflow ops don't invalidate caches ──

    #[test]
    fn test_overflow_ops_dont_invalidate() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(
                OpCode::IntAddOvf,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 0),
                ],
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + INT_ADD_OVF + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::IntAddOvf);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 16: Multiple fields on same object ──

    #[test]
    fn test_multiple_fields_same_object() {
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Int, 101),
                ],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Int, 102),
                ],
                d1.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d1.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt_typed(&mut ops, &[101, 102]);

        // Both GETFIELDs eliminated (cached). Both lazy SetfieldGc emitted before Jump.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Green field optimization tests ──

    // ── Test 17: Immutable field cache survives call invalidation ──

    #[test]
    fn test_immutable_field_survives_call() {
        // i1 = getfield_gc_i(p0, descr=immutable_d0)
        // call_n(...)                              <- invalidates mutable caches
        // i2 = getfield_gc_i(p0, descr=immutable_d0) <- still cached (immutable)
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + CALL + Jump. Second GETFIELD eliminated (immutable survives call).
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 18: Mutable field cache is still invalidated by call ──

    #[test]
    fn test_mutable_field_invalidated_by_call() {
        // i1 = getfield_gc_i(p0, descr=mutable_d0)
        // call_n(...)
        // i2 = getfield_gc_i(p0, descr=mutable_d0) <- re-emitted (mutable, invalidated)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + CALL + GETFIELD (re-emitted) + Jump.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 19: Mixed immutable and mutable fields: only mutable invalidated ──

    #[test]
    fn test_mixed_immutable_mutable_fields() {
        // i1 = getfield_gc_i(p0, descr=immut_d0)
        // i2 = getfield_gc_i(p0, descr=mut_d1)
        // call_n(...)
        // i3 = getfield_gc_i(p0, descr=immut_d0)  <- cached (immutable survives)
        // i4 = getfield_gc_i(p0, descr=mut_d1)    <- re-emitted (mutable invalidated)
        let d_immut = immutable_descr(0);
        let d_mut = descr(1);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d_immut.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d_mut.clone(),
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d_immut.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d_mut.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD(immut) + GETFIELD(mut) + CALL + GETFIELD(mut, re-emitted) + Jump.
        // GETFIELD(immut) after call is eliminated.
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI); // immutable, first read
        assert_eq!(result[1].opcode, OpCode::GetfieldGcI); // mutable, first read
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[3].opcode, OpCode::GetfieldGcI); // mutable, re-emitted
        assert_eq!(result[4].opcode, OpCode::Jump);
    }

    // ── Test 20: Immutable field from non-constant object still gets read-cache ──

    #[test]
    fn test_immutable_field_read_cache_no_constant() {
        // Even without a constant source object, immutable fields benefit from
        // read-after-read caching that survives side effects.
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only first GETFIELD + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 21: Immutable Ref and Float field variants survive call ──

    #[test]
    fn test_immutable_field_ref_survives_call() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcR,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::with_descr(
                OpCode::GetfieldGcR,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_immutable_field_float_survives_call() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcF,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::with_descr(
                OpCode::GetfieldGcF,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcF);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_short_preamble_ref_field_preserves_getfield_opcode() {
        let descr =
            majit_ir::make_field_descr(55, 8, majit_ir::Type::Ref, majit_ir::ArrayFlag::Pointer);
        let mut pass = OptHeap::new();
        // history.py:182 PtrInfo applies to ref-typed boxes; the field
        // descr is Type::Ref so the field source is ref-typed too.
        let pos100 = rooted_resop_operand(Type::Ref, 100);
        let val101 = rooted_resop_operand(Type::Ref, 101);
        pass.cache_field(&pos100, &descr);

        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::ref_op(100),
            OpRef::ref_op(101),
        ]);
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        // Register the objects the cache already names, before
        // add_short_input_arg materializes those positions.
        ctx.seed_boxes_canonical(&[pos100.clone(), val101.clone()]);
        sb.add_short_input_arg(&mut ctx, OpRef::ref_op(100), majit_ir::Type::Ref);
        sb.add_short_input_arg(&mut ctx, OpRef::ref_op(101), majit_ir::Type::Ref);
        use crate::optimizeopt::info::PtrInfo;
        ctx.set_ptr_info(&pos100, PtrInfo::instance(None, None));
        // Seed the slot the reader will consult: `field_slot_index`, not the
        // descriptor's own key.  A descriptor with no parent carries no slot
        // number, so the two differ.
        let slot = OptHeap::field_slot_index(&descr);
        ctx.with_ptr_info_mut(&pos100, |info| {
            info.setfield(slot, val101.clone());
        })
        .unwrap();
        pass.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let produced = sb.produced_ops(&mut ctx);

        // Filter to heap-produced ops (exclude SameAs* from add_short_input_arg).
        let heap_ops: Vec<_> = produced
            .iter()
            .filter(|(_, p)| {
                !matches!(
                    p.preamble_op.opcode,
                    OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
                )
            })
            .collect();
        assert_eq!(heap_ops.len(), 1);
        assert_eq!(heap_ops[0].1.preamble_op.opcode, OpCode::GetfieldGcR);
    }

    // ── Test 22: Immutable field survives multiple calls ──

    #[test]
    fn test_immutable_field_survives_multiple_calls() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 201)]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + CALL + CALL + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 23: Different objects with same immutable descr are independent ──

    #[test]
    fn test_immutable_field_different_objects() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                d.clone(),
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 300)]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ), // cached
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                d.clone(),
            ), // cached
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Both initial GETFIELDs + CALL + Jump. Both post-call GETFIELDs eliminated.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Aliasing analysis tests ──

    // ── Test 24: Two NEW objects don't alias — write to one preserves cache of the other ──

    // ── Test 25: Unknown-origin object write invalidates other unknown caches ──

    #[test]
    fn test_unknown_object_write_invalidates() {
        // p0 = InputRef(100), p1 = InputRef(200)  — both unknown origin
        // i1 = getfield_gc_i(p0, descr=d0)
        // setfield_gc(p1, i20, descr=d0)   <- p1 is unknown, might alias p0
        // i2 = getfield_gc_i(p0, descr=d0) <- must re-emit (might have been clobbered)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 200),
                    rooted_inputarg_operand(Type::Ref, 20),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + SETFIELD + GETFIELD (re-emitted) + Jump.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "second GETFIELD must be re-emitted for unknown-origin objects"
        );
    }

    // ── Test 26: Unescaped allocation's cache survives call ──

    #[test]
    fn test_unescaped_invalidated_by_call_write() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // call_n(some_func)               <- p0 NOT passed to call
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit
        //
        // heap.py: check_write_descr_field → force_lazy_set(can_cache=False)
        // PyPy does a full invalidate on write, regardless of escape status.
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 0),
                    rooted_inputarg_operand(Type::Ref, 10),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                plain_call_descr(100),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_resop_operand(Type::Ref, 0)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // heap.py: force_lazy_set(can_cache=False) invalidates ALL entries.
        // GETFIELD must be re-emitted after call that writes the field.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count,
            1,
            "GETFIELD must be re-emitted after call writes field (full invalidate), got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Test 27: Escaped allocation's cache is invalidated by call ──

    #[test]
    fn test_escaped_invalidated_by_call() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // call_n(p0)                       <- p0 is passed to call, escapes
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit (call might have modified p0)
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 0),
                    rooted_inputarg_operand(Type::Ref, 10),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[rooted_resop_operand(Type::Ref, 0)],
                plain_call_descr(100),
            ), // pass p0 to call
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_resop_operand(Type::Ref, 0)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // NEW + SETFIELD + CALL + GETFIELD (re-emitted) + Jump.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "GETFIELD must be re-emitted after escape via call"
        );
    }

    // ── Test 28: SetfieldGc marks stored value as escaped ──

    #[test]
    fn test_setfield_marks_escape() {
        // p0 = new()       <- unescaped
        // p1 = new()       <- unescaped
        // setfield_gc(p0, i10, descr=d0)
        // setfield_gc(p1, p0, descr=d1)   <- p0 is stored into p1's field, p0 escapes
        // call_n(p1)                       <- p1 escapes; p0 already escaped via setfield
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit (p0 escaped)
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::new(OpCode::New, &[]), // pos=1 -> p1
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 0),
                    rooted_inputarg_operand(Type::Ref, 10),
                ],
                d0.clone(),
            ), // p0.f0 = i10
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 1),
                    rooted_resop_operand(Type::Ref, 0),
                ],
                d1.clone(),
            ), // p1.f1 = p0 (p0 escapes)
            Op::with_descr(
                OpCode::CallN,
                &[rooted_resop_operand(Type::Ref, 1)],
                plain_call_descr(100),
            ), // call(p1) (p1 escapes)
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_resop_operand(Type::Ref, 0)],
                d0.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // p0 escaped via setfield, so its cache is invalidated by the call.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "GETFIELD must be re-emitted after p0 escaped via setfield"
        );
    }

    // ── Test 29: Seen-allocation cache survives write from unknown-origin object ──

    // ── Test 30: Different field descriptors are not affected by aliasing ──

    #[test]
    fn test_aliasing_different_fields_independent() {
        // Even with unknown-origin objects, writes to field d0 don't
        // invalidate caches for field d1.
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d1.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 200),
                    rooted_inputarg_operand(Type::Ref, 20),
                ],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d1.clone(),
            ), // different field
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD(d1) + SETFIELD(d0) + Jump. Second GETFIELD eliminated.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(get_count, 1, "write to d0 should not invalidate d1 cache");
    }

    // ── Test 31: Unescaped array cache survives call ──

    #[test]
    fn test_unescaped_array_invalidated_by_call_write() {
        // p0 = new_array(5)
        // setarrayitem_gc(p0, idx, i10, descr=d0)
        // call_n(some_func)                <- p0 not passed
        // i1 = getarrayitem_gc_i(p0, idx, descr=d0) <- must re-emit
        //
        // heap.py: check_write_descr_array → force_lazy_setarrayitem_submap(can_cache=False)
        // PyPy does a full invalidate on write, regardless of escape status.
        let d = descr(0);
        let len = rooted_resop_operand(Type::Int, 5);
        let val = rooted_resop_operand(Type::Int, 10);
        let func = rooted_resop_operand(Type::Int, 200);
        let idx = Operand::const_from_value(majit_ir::Value::Int(3));
        let new_array = OpRc::new(Op::new(OpCode::NewArray, &[len.clone()]));
        new_array.pos().set(OpRef::ref_op(0));
        let arr = Operand::from_bound_op(&new_array);
        let mut ops = vec![
            (*new_array).clone(),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[arr.clone(), idx.clone(), val.clone()],
                d.clone(),
            ),
            Op::with_descr(OpCode::CallN, &[func.clone()], plain_call_descr(100)),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[arr.clone(), idx.clone()],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);
        // assign_positions rewrites op 0; keep the shared array producer.
        ops[0] = (*new_array).clone();

        let mut ctx = OptContext::new(ops.len());
        ctx.seed_boxes_canonical(&[len, val, func]);
        // The array operand is the NewArray producer. Register it as the
        // live synthetic so emit forwards that same object to the emitted op.
        ctx.bind_input_resops(std::slice::from_ref(&new_array));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        let get_count = opcodes
            .iter()
            .filter(|&&o| o == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "GETARRAYITEM must be re-emitted after call writes array (full invalidate), got: {opcodes:?}"
        );
    }

    // ── Test 32: Multiple calls — unescaped object stays cached ──

    #[test]
    fn test_unescaped_invalidated_by_multiple_calls() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // call_n(f1)
        // call_n(f2)
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit
        //
        // heap.py: full invalidate on write call, regardless of escape status.
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 0),
                    rooted_inputarg_operand(Type::Ref, 10),
                ],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                plain_call_descr(100),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[rooted_inputarg_operand(Type::Ref, 201)],
                plain_call_descr(101),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_resop_operand(Type::Ref, 0)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count,
            1,
            "GETFIELD must be re-emitted after calls that write field (full invalidate), got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Nullity tracking tests ──

    // ── Test 37: GuardNonnull after allocation is removed ──

    #[test]
    fn test_guard_nonnull_after_allocation() {
        // p0 = new()
        // guard_nonnull(p0)   <- redundant, allocation is always non-null
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::new(OpCode::GuardNonnull, &[rooted_resop_operand(Type::Ref, 0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after allocation should be removed"
        );
    }

    // ── Test 38: GuardNonnull after GuardNonnull is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_nonnull() {
        // guard_nonnull(p0)
        // guard_nonnull(p0)   <- redundant
        let mut ops = vec![
            Op::new(
                OpCode::GuardNonnull,
                &[rooted_inputarg_operand(Type::Ref, 100)],
            ),
            Op::new(
                OpCode::GuardNonnull,
                &[rooted_inputarg_operand(Type::Ref, 100)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(nonnull_count, 2, "second guard_nonnull should be removed");
    }

    // ── Test 39: GuardNonnull after GuardClass is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_class() {
        // guard_class(p0, cls)  <- implies non-null
        // guard_nonnull(p0)     <- redundant
        let mut ops = vec![
            Op::new(
                OpCode::GuardClass,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 101),
                ],
            ),
            Op::new(
                OpCode::GuardNonnull,
                &[rooted_inputarg_operand(Type::Ref, 100)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after guard_class should be removed"
        );
    }

    // ── Test 40: GuardNonnull on unknown input arg is kept ──

    #[test]
    fn test_guard_nonnull_unknown_not_removed() {
        // guard_nonnull(p0)  <- first time seeing p0, must keep
        let mut ops = vec![
            Op::new(
                OpCode::GuardNonnull,
                &[rooted_inputarg_operand(Type::Ref, 100)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(nonnull_count, 1, "guard_nonnull on unknown should be kept");
    }

    // ── Test 41: Nonnull from allocation survives call ──

    #[test]
    fn test_known_nonnull_survives_call_for_allocation() {
        // p0 = new()
        // call_n(some_func)    <- invalidates caches, but not allocation nonnull
        // guard_nonnull(p0)    <- still redundant (allocation is always non-null)
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::new(OpCode::GuardNonnull, &[rooted_resop_operand(Type::Ref, 0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after allocation should be removed even after call"
        );
    }

    // ── Test 42: Nonnull from guard does NOT survive call ──

    #[test]
    fn test_known_nonnull_from_guard_invalidated_by_call() {
        // guard_nonnull(p0)
        // call_n(some_func)   <- invalidates guard-derived nonnull
        // guard_nonnull(p0)   <- must re-emit
        let mut ops = vec![
            Op::new(
                OpCode::GuardNonnull,
                &[rooted_inputarg_operand(Type::Ref, 100)],
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::new(
                OpCode::GuardNonnull,
                &[rooted_inputarg_operand(Type::Ref, 100)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 2,
            "guard_nonnull after call should be re-emitted for non-allocation values"
        );
    }

    // ── Test 43: GuardNonnull after GuardNonnullClass is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_nonnull_class() {
        // guard_nonnull_class(p0, cls) <- implies non-null
        // guard_nonnull(p0)            <- redundant
        let mut ops = vec![
            Op::new(
                OpCode::GuardNonnullClass,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 101),
                ],
            ),
            Op::new(
                OpCode::GuardNonnull,
                &[rooted_inputarg_operand(Type::Ref, 100)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after guard_nonnull_class should be removed"
        );
    }

    // ── Test 44: GuardNonnull after GuardValue is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_value() {
        // guard_value(p0, c) <- implies non-null
        // guard_nonnull(p0)  <- redundant
        let mut ops = vec![
            Op::new(
                OpCode::GuardValue,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 101),
                ],
            ),
            Op::new(
                OpCode::GuardNonnull,
                &[rooted_inputarg_operand(Type::Ref, 100)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after guard_value should be removed"
        );
    }

    // ── Test 45: GuardNonnull after NewWithVtable is removed ──

    #[test]
    fn test_guard_nonnull_after_new_with_vtable() {
        let mut ops = vec![
            Op::new(OpCode::NewWithVtable, &[]),
            Op::new(OpCode::GuardNonnull, &[rooted_resop_operand(Type::Ref, 0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after new_with_vtable should be removed"
        );
    }

    // ── Test 46: GuardNonnull after NewArray is removed ──

    #[test]
    fn test_guard_nonnull_after_new_array() {
        let mut ops = vec![
            Op::new(OpCode::NewArray, &[rooted_inputarg_operand(Type::Ref, 5)]),
            Op::new(OpCode::GuardNonnull, &[rooted_resop_operand(Type::Ref, 0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after new_array should be removed"
        );
    }

    // ── Call descriptor with OopSpecIndex for arraycopy tests ──

    /// Call descriptor with configurable EffectInfo for testing.
    #[derive(Debug)]
    struct TestCallDescr {
        idx: u32,
        effect: EffectInfo,
    }

    impl Descr for TestCallDescr {
        fn index(&self) -> u32 {
            self.idx
        }
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[majit_ir::Type] {
            &[]
        }
        fn result_type(&self) -> majit_ir::Type {
            majit_ir::Type::Void
        }
        fn result_size(&self) -> usize {
            0
        }
        fn get_extra_info(&self) -> &EffectInfo {
            &self.effect
        }
    }

    fn call_descr(idx: u32, effect: EffectInfo) -> DescrRef {
        Arc::new(TestCallDescr { idx, effect })
    }

    #[test]
    fn test_call_may_force_uses_effectinfo_to_keep_unaffected_cached_fields() {
        let d0 = descr(0);
        let d1 = descr(1);
        let call_d = call_descr(
            70,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                // `effectinfo.py:131 _write_descrs_fields = frozenset({descr1})`
                // — Arc-identity raw set carrying d1, paired with the
                // legacy `descr.index()`-keyed bitstring so the fixture
                // works both pre- and post-`compute_bitstrings`.
                _write_descrs_fields: Some(vec![d1]),
                write_descrs_fields: Some(bitstring::make_bitstring(&[1])),
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::CallMayForceN,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                call_d,
            ),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "CallMayForce with unrelated write bit should preserve cached GETFIELD"
        );
    }

    #[test]
    fn test_forcing_call_flushes_virtualref_forced_field_before_call() {
        // heap.py force_from_effectinfo flushes exactly
        // `vrefinfo.descr_forced` even though the descriptor is not named in
        // the call's ordinary readonly/write sets.
        let vrefinfo = crate::virtualref::VirtualRefInfo::new();
        let forced_descr = vrefinfo.descr_forced.clone();
        let call_d = call_descr(
            75,
            EffectInfo {
                extraeffect: ExtraEffect::ForcesVirtualOrVirtualizable,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 101),
                ],
                forced_descr.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                call_d,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptHeap::new()));
        opt.set_vrefinfo(vrefinfo);
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&vec![Type::Ref; 1024]);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;

        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);

        let set_pos = result
            .iter()
            .position(|op| {
                op.opcode == OpCode::SetfieldGc
                    && op
                        .getdescr()
                        .is_some_and(|descr| Arc::ptr_eq(&descr, &forced_descr))
            })
            .expect("forcing call must flush vref.forced");
        let call_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::CallN)
            .expect("forcing call must remain in the trace");
        assert!(
            set_pos < call_pos,
            "vref.forced must be visible to the call"
        );
    }

    #[test]
    fn test_cond_call_without_random_effects_keeps_unrelated_lazy_set() {
        // heap.py has no optimize_COND_CALL. COND_CALL is is_call
        // (`resoperation.py` `OpHelpers.is_call`); `emitting_operation`
        // uses `force_from_effectinfo` when `!has_random_effects`
        // (`effectinfo.py` `EF_CAN_RAISE`). An unrelated lazy SETFIELD_GC
        // stays lazy across the call and is flushed at Jump (`flush` →
        // `force_all_lazy_sets`).
        let field = descr(0);
        let call_d = call_descr(
            81,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Int, 20),
                ],
                field,
            ),
            Op::with_descr(
                OpCode::CondCallN,
                &[
                    rooted_inputarg_operand(Type::Int, 1),
                    rooted_inputarg_operand(Type::Ref, 200),
                ],
                call_d,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt_typed(&mut ops, &[1, 20]);
        let set_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::SetfieldGc)
            .expect("lazy set must still flush");
        let call_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::CondCallN)
            .expect("COND_CALL remains");
        assert!(
            set_pos > call_pos,
            "unrelated SETFIELD_GC must stay lazy across COND_CALL"
        );
    }

    #[test]
    fn test_unanalyzed_residual_flushes_lazy_sets_and_drops_field_cache() {
        // effectinfo.py `effectinfo_from_writeanalyze`: an unknown write
        // analysis is `EF_RANDOM_EFFECTS` (`EffectInfo.MOST_GENERAL`,
        // `has_random_effects`). heap.py `OptHeap.emitting_operation`
        // then skips `force_from_effectinfo` and runs
        // `force_all_lazy_sets` + `clean_caches`, so a lazy
        // `setfield_gc` / `setarrayitem_gc` is emitted before the call
        // and a field read cached before it is not reused after it.
        // `default_effect_info` is the descr
        // `residual_effect_info_for_target` stamps on an unanalyzed residual.
        let field = descr(0);
        let array = descr(1);
        let call_d = call_descr(80, crate::call_descr::default_effect_info());
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                field.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Int, 101),
                ],
                field.clone(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 110),
                    Operand::const_from_value(majit_ir::Value::Int(0)),
                    rooted_inputarg_operand(Type::Int, 111),
                ],
                array,
            ),
            Op::with_descr(
                OpCode::CallN,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                call_d,
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                field,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt_typed(&mut ops, &[101, 111]);

        let call_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::CallN)
            .expect("unanalyzed residual call must survive");
        let setfield_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::SetfieldGc)
            .expect("lazy setfield_gc must be emitted");
        let setarray_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::SetarrayitemGc)
            .expect("lazy setarrayitem_gc must be emitted");
        assert!(
            setfield_pos < call_pos,
            "setfield_gc must be emitted before the residual call; got {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
        assert!(
            setarray_pos < call_pos,
            "setarrayitem_gc must be emitted before the residual call; got {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
        let gets_after = result
            .iter()
            .skip(call_pos)
            .filter(|op| op.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            gets_after,
            1,
            "a field read cached before an unanalyzed residual must not be reused after it; got {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_call_may_force_uses_effectinfo_to_invalidate_written_cached_fields() {
        let d0 = descr(0);
        // PyPy `effectinfo.py:526 mapping.setdefault(...)` would assign
        // `d0.ei_index = 0` once compute_bitstrings has run, since d0 is
        // the only descr in `_write_descrs_fields`. Tests skip
        // compute_bitstrings, so we set it manually so the bitstring at
        // bit 0 actually corresponds to d0's encoding.
        d0.set_ei_index(0);
        let call_d = call_descr(
            71,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                _write_descrs_fields: Some(vec![d0.clone()]),
                write_descrs_fields: Some(bitstring::make_bitstring(&[0])),
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0.clone(),
            ),
            Op::with_descr(
                OpCode::CallMayForceN,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                call_d,
            ),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "CallMayForce with matching write bit must invalidate cached GETFIELD"
        );
    }

    #[test]
    fn test_call_may_force_resets_guard_not_invalidated_when_call_can_invalidate() {
        let call_d = call_descr(
            72,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                can_invalidate: true,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::with_descr(
                OpCode::CallMayForceN,
                &[rooted_inputarg_operand(Type::Ref, 200)],
                call_d,
            ),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNotInvalidated)
            .count();
        assert_eq!(
            guard_count, 2,
            "CallMayForce that can invalidate must keep the later GuardNotInvalidated"
        );
    }

    #[test]
    fn test_call_may_force_keeps_unaffected_variable_index_array_cache() {
        let d0 = descr(0);
        let d1 = descr(1);
        let arr = rooted_resop_operand(Type::Int, 100);
        let idx_box = rooted_resop_operand(Type::Int, 50);
        let func = rooted_resop_operand(Type::Int, 200);
        let call_d = call_descr(
            73,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                _write_descrs_arrays: Some(vec![d1]),
                write_descrs_arrays: Some(bitstring::make_bitstring(&[1])),
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[arr.clone(), idx_box.clone()],
                d0.clone(),
            ),
            Op::with_descr(OpCode::CallMayForceN, &[func.clone()], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetarrayitemGcI, &[arr.clone(), idx_box.clone()], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let mut ctx = OptContext::new(ops.len() + 64);
        assign_positions(&mut ops);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        ctx.snapshot_boxes = snapshots;
        let mut pass = OptHeap::new();
        pass.setup();
        // The variable index is the operand the ops carry, so getintbound
        // installs its IntBound on that same object.
        ctx.seed_boxes_canonical(&[arr, idx_box, func]);

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "CallMayForce with unrelated array write bit should preserve variable-index cache"
        );
    }

    #[test]
    fn test_call_may_force_invalidates_written_variable_index_array_cache() {
        let d0 = descr(0);
        // d0 is the only descr in `_write_descrs_arrays`, so PyPy
        // `effectinfo.py:526` would assign `d0.ei_index = 0`. Tests skip
        // compute_bitstrings, so we set it manually.
        d0.set_ei_index(0);
        let arr = rooted_resop_operand(Type::Int, 100);
        let idx_box = rooted_resop_operand(Type::Int, 50);
        let func = rooted_resop_operand(Type::Int, 200);
        let call_d = call_descr(
            74,
            EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                _write_descrs_arrays: Some(vec![d0.clone()]),
                write_descrs_arrays: Some(bitstring::make_bitstring(&[0])),
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[arr.clone(), idx_box.clone()],
                d0.clone(),
            ),
            Op::with_descr(OpCode::CallMayForceN, &[func.clone()], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetarrayitemGcI, &[arr.clone(), idx_box.clone()], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let mut ctx = OptContext::new(ops.len() + 64);
        assign_positions(&mut ops);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        ctx.snapshot_boxes = snapshots;
        let mut pass = OptHeap::new();
        pass.setup();
        ctx.seed_boxes_canonical(&[arr, idx_box, func]);

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "CallMayForce with matching array write bit must invalidate variable-index cache"
        );
    }

    // ARRAYCOPY tests removed — RPython heap.py has no ARRAYCOPY special case.
    // ARRAYCOPY optimization belongs in rewrite.py (rewrite.py).

    // ── Test 50: GC_LOAD forces lazy setfields ──

    #[test]
    fn test_gc_load_forces_lazy_setfields() {
        // setfield_gc(p0, i1, descr=d0)   <- lazy, not emitted yet
        // i2 = gc_load_i(p1, offset, size) <- generic load, forces all lazy writes
        // The SETFIELD must be emitted before the GC_LOAD.
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 101),
                ],
                d.clone(),
            ),
            Op::new(
                OpCode::GcLoadI,
                &[
                    rooted_inputarg_operand(Type::Ref, 200),
                    rooted_inputarg_operand(Type::Ref, 8),
                    rooted_inputarg_operand(Type::Ref, 4),
                ],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // force_all_lazy_setfields at GcLoadI emits lazy SetfieldGc. SetfieldGc + GcLoadI + Jump.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::GcLoadI);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 51: GC_LOAD marks base as nonnull ──

    #[test]
    fn test_gc_load_marks_nonnull() {
        // i1 = gc_load_i(p0, offset, size)  <- dereferences p0, so p0 is nonnull
        // guard_nonnull(p0)                  <- redundant
        let mut ops = vec![
            Op::new(
                OpCode::GcLoadI,
                &[
                    rooted_inputarg_operand(Type::Ref, 100),
                    rooted_inputarg_operand(Type::Ref, 8),
                    rooted_inputarg_operand(Type::Ref, 4),
                ],
            ),
            Op::new(
                OpCode::GuardNonnull,
                &[rooted_inputarg_operand(Type::Ref, 100)],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 1,
            "guard_nonnull after gc_load should be removed"
        );
    }

    // ── Test 52: QUASIIMMUT_FIELD on field 0 doesn't affect field 1 ──

    #[test]
    fn test_quasiimmut_field_different_field_not_cached() {
        // quasiimmut_field(p0, descr=d0)      <- marks field 0 as quasi-immut
        // i1 = getfield_gc_i(p0, descr=d1)   <- different field, NOT quasi-immut
        // call_n(some_func)
        // i2 = getfield_gc_i(p0, descr=d1)   <- must re-emit (d1 is mutable)
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(
                OpCode::QuasiimmutField,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d0,
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d1.clone(),
            ),
            Op::new(OpCode::CallN, &[rooted_inputarg_operand(Type::Ref, 200)]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_inputarg_operand(Type::Ref, 100)],
                d1.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GUARD_NOT_INVALIDATED + GETFIELD(d1) + CALL + GETFIELD(d1, re-emitted) + Jump.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "quasi-immut on field 0 should not affect field 1"
        );
    }

    // ── Test 53: Bytearray-as-array heap cache verification ──
    //
    // RPython treats bytearray as regular arrays with item_size=1.
    // Verify the heap cache works correctly with byte-sized array items.

    /// Array descriptor with item_size=1 (byte array).
    #[derive(Debug)]
    struct ByteArrayDescr(u32);

    impl Descr for ByteArrayDescr {
        fn index(&self) -> u32 {
            self.0
        }
        fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
            Some(self)
        }
    }

    impl majit_ir::ArrayDescr for ByteArrayDescr {
        fn base_size(&self) -> usize {
            8 // typical GC header
        }
        fn item_size(&self) -> usize {
            1 // byte-sized items
        }
        fn type_id(&self) -> u32 {
            0
        }
        fn item_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
    }

    fn byte_array_descr(idx: u32) -> DescrRef {
        Arc::new(ByteArrayDescr(idx))
    }

    #[test]
    fn test_bytearray_setitem_then_getitem_cached() {
        // setarrayitem_gc(p0, idx, val, descr=byte_array)
        // i2 = getarrayitem_gc_i(p0, idx, descr=byte_array)  <- eliminated
        let d = byte_array_descr(50);
        let arr = rooted_resop_operand(Type::Int, 100);
        let val = rooted_resop_operand(Type::Int, 101);
        let idx = Operand::const_from_value(majit_ir::Value::Int(5));
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[arr.clone(), idx.clone(), val.clone()],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[arr.clone(), idx.clone()],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.seed_boxes_canonical(&[arr, val]);

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // force_all_lazy at Jump drops lazy setarrayitems. Only Jump remains.
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::Jump],
            "byte-array getitem should be cached after setitem; lazy set dropped at Jump"
        );
    }

    #[test]
    fn test_bytearray_different_indices_not_cached() {
        // setarrayitem_gc(p0, idx=5, val, descr=byte_array)
        // i2 = getarrayitem_gc_i(p0, idx=6, descr=byte_array)  <- NOT cached (different index)
        let d = byte_array_descr(50);
        let arr = rooted_resop_operand(Type::Int, 100);
        let val = rooted_resop_operand(Type::Int, 101);
        let idx5 = Operand::const_from_value(majit_ir::Value::Int(5));
        let idx6 = Operand::const_from_value(majit_ir::Value::Int(6));
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[arr.clone(), idx5, val.clone()],
                d.clone(),
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[arr.clone(), idx6], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.seed_boxes_canonical(&[arr, val]);

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // GETARRAYITEM must be emitted (not cached — different index).
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert!(
            opcodes.contains(&OpCode::GetarrayitemGcI),
            "different byte-array index should not use cache: {:?}",
            opcodes
        );
    }

    #[test]
    fn test_bytearray_read_after_read_cached() {
        // i1 = getarrayitem_gc_i(p0, idx=3, descr=byte_array)
        // i2 = getarrayitem_gc_i(p0, idx=3, descr=byte_array)  <- eliminated (same read)
        let d = byte_array_descr(50);
        let idx = OpRef::int_op(60);
        // GETARRAYITEM_GC's receiver is a GC reference Box. Keeping it Int
        // made the fixture depend on positional re-wrapping to recover a
        // different Ref-shaped box instead of exercising the real cache path.
        let array_box = rooted_resop_operand(Type::Ref, 100);
        let index_box = rooted_resop_operand(Type::Int, idx.raw());
        let mut ops = vec![
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[array_box.clone(), index_box.clone()],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[array_box.clone(), index_box.clone()],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        // The optimizer's real input-op seed makes position lookup resolve to
        // this same receiver Box. Model that producer edge explicitly; a bare
        // synthetic ResOp is otherwise absent from the context's OpRef store.
        let positional_array = ctx.materialize_operand_at(array_box.to_opref());
        ctx.make_equal_to(&positional_array, &array_box);
        ctx.make_constant_box(&index_box, majit_ir::Value::Int(3));

        let mut pass = OptHeap::new();
        pass.setup();

        for (op_index, op) in ops.iter().enumerate() {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                let arg = resolved.arg(i);
                let rb = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let __ar = arg.to_opref();
                        if __ar.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(__ar).get_box_replacement(false)
                        }
                    }
                };
                resolved.setarg(i, rb);
            }
            let op_rc = OpRc::new(resolved.clone());
            match pass.propagate_forward(&resolved, &op_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op_rc);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
            if op_index == 0 {
                assert!(
                    ctx.peek_ptr_info(&array_box)
                        .and_then(|info| info.getitem(3))
                        .is_some(),
                    "first byte-array read must populate the receiver's item cache"
                );
            }
        }

        // Only one GETARRAYITEM + Jump: the second read is eliminated.
        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(get_count, 1, "byte-array read-after-read should be cached");
    }

    /// `resoperation.py` lists `ARRAYLEN_GC` inside the
    /// `_ALWAYS_PURE_FIRST.._ALWAYS_PURE_LAST` band; CSE of always-pure
    /// ops is `optimizeopt/pure.py`'s `_pure_operations[opnum]`
    /// table, not heap.py.  This test wires `OptPure` to confirm the
    /// dedup path; running with `OptHeap` alone would (correctly) leave
    /// both reads in place — heap.py has no parallel cache.
    #[test]
    fn test_arraylen_caching_via_optpure() {
        let d = descr(42);
        let array = rooted_inputarg_operand(Type::Ref, 100);
        let mut ops = vec![
            {
                let mut op = Op::new(OpCode::ArraylenGc, &[array.clone()]);
                op.setdescr(d.clone());
                op
            },
            {
                let mut op = Op::new(OpCode::ArraylenGc, &[array]);
                op.setdescr(d);
                op
            },
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::pure::OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);
        let len_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::ArraylenGc)
            .count();
        assert_eq!(
            len_count, 1,
            "duplicate ARRAYLEN_GC should be cached by OptPure"
        );
    }

    /// heap.py AbstractCachedEntry._cannot_alias_via_constants — two
    /// different constant pointers name two different objects, while a
    /// non-constant operand on either side leaves the question open.
    #[test]
    fn test_cannot_alias_via_constants() {
        use crate::optimizeopt::OptContext;

        let mut ctx = OptContext::with_inputarg_types(64, &[Type::Ref]);
        let first_ref = ctx.make_constant_ref(majit_ir::GcRef(0x1000));
        let second_ref = ctx.make_constant_ref(majit_ir::GcRef(0x2000));
        let first = ctx.materialize_operand_at(first_ref);
        let second = ctx.materialize_operand_at(second_ref);
        let unknown = ctx.materialize_operand_at(OpRef::input_arg_typed(0, Type::Ref));

        assert!(
            super::cannot_alias_via_constants(&mut ctx, &first, &second),
            "two different constant pointers cannot alias"
        );
        assert!(
            !super::cannot_alias_via_constants(&mut ctx, &first, &first),
            "the same constant pointer is the same object, not a disproof"
        );
        assert!(
            !super::cannot_alias_via_constants(&mut ctx, &first, &unknown),
            "a non-constant operand leaves aliasing unknown"
        );
    }

    /// heap.py ArrayCachedItem._cannot_alias_via_content — two
    /// arrays with the same descr + length but differing constant items
    /// must be reported as unable to alias.
    #[test]
    fn test_cannot_alias_via_content_different_constants() {
        use crate::optimizeopt::OptContext;
        use crate::optimizeopt::info::{ArrayPtrInfo, FieldEntry};
        use crate::optimizeopt::intutils::IntBound;

        // `inputarg_from_tp(arg.type)` per opencoder.py:259 — both arrays are
        // Ref-typed inputargs.
        let mut ctx = OptContext::with_inputarg_types(64, &[Type::Ref, Type::Ref]);

        let arr_descr = descr(50);
        let op1 = OpRef::input_arg_typed(0, Type::Ref);
        let op2 = OpRef::input_arg_typed(1, Type::Ref);

        let const_10 = ctx.make_constant_int(10);
        let const_20 = ctx.make_constant_int(20);
        let const_30 = ctx.make_constant_int(30);

        let op1_box = ctx.materialize_operand_at(op1);
        let op2_box = ctx.materialize_operand_at(op2);
        ctx.set_ptr_info(
            &op1_box,
            PtrInfo::Array(ArrayPtrInfo {
                descr: arr_descr.clone(),
                lenbound: IntBound::from_constant(2),
                items: vec![
                    FieldEntry::Value(bound_arg(const_10)),
                    FieldEntry::Value(bound_arg(const_20)),
                ],
                last_guard_pos: -1,
            }),
        );
        ctx.set_ptr_info(
            &op2_box,
            PtrInfo::Array(ArrayPtrInfo {
                descr: arr_descr,
                lenbound: IntBound::from_constant(2),
                items: vec![
                    FieldEntry::Value(bound_arg(const_10)),
                    FieldEntry::Value(bound_arg(const_30)),
                ],
                last_guard_pos: -1,
            }),
        );

        assert!(
            super::ArrayCachedItem::_cannot_alias_via_content(&op1_box, &op2_box, &mut ctx),
            "arrays with different constant at index 1 cannot alias"
        );
    }

    fn dict_lookup_descr(idx: u32, extra0: DescrRef, extra1: DescrRef) -> DescrRef {
        Arc::new(TestCallDescr {
            idx,
            effect: EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                oopspecindex: OopSpecIndex::DictLookup,
                extradescrs: Some(vec![extra0, extra1]),
                ..Default::default()
            },
        })
    }

    #[test]
    fn array_caches_are_keyed_by_descr_identity_not_numeric_index() {
        // heap.py:332/399 keys cached_arrayitems and
        // corresponding_array_descrs by the descriptor object itself.
        // Two distinct descriptors may legitimately share `index()`.
        let array1: DescrRef = descr(81);
        let array2: DescrRef = descr(81);
        assert_ne!(
            majit_ir::descr::descr_identity(&array1),
            majit_ir::descr::descr_identity(&array2)
        );

        let mut heap = OptHeap::new();
        heap.arrayitem_submap(&array1);
        heap.arrayitem_submap(&array2);
        assert_eq!(heap.cached_arrayitems.len(), 2);

        heap.corresponding_array_descrs.insert(
            majit_ir::descr::descr_identity(&array1),
            (array1.clone(), 1),
        );
        heap.corresponding_array_descrs.insert(
            majit_ir::descr::descr_identity(&array2),
            (array2.clone(), 2),
        );
        assert_eq!(heap.corresponding_array_descrs.len(), 2);
    }

    /// heap.py:480-528 parity: FLAG_LOOKUP deduplicates consecutive dict lookups.
    #[test]
    fn test_dict_lookup_cache_flag_lookup() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        // Bind the dict/key InputArg slots so they resolve to canonical
        // boxes (production binds these via main-trace bind-at-alloc); the
        // box-identity dict cache key needs a bound box per arg.
        let mut ctx = OptContext::with_inputarg_types(256, &[Type::Ref, Type::Ref]);

        // Build args: [func_addr, dict, key, hash, flag=FLAG_LOOKUP(0)]
        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let key = OpRef::input_arg_typed(1, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag = ctx.make_constant_int(0); // FLAG_LOOKUP

        // First lookup — not cached yet, should return false (emit).
        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(
            OpCode::CallI,
            &[
                bound_arg(func_addr),
                bound_arg(dict),
                bound_arg(key),
                bound_arg(hash),
                bound_arg(flag),
            ],
        );
        op1.setdescr(descr.clone());
        op1.pos().set(pos1);
        assert!(!heap._optimize_call_dict_lookup(&op1, &OpRc::new(op1.clone()), &mut ctx));

        // Second lookup with same dict+key — should be cached.
        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op2 = Op::new(
            OpCode::CallI,
            &[
                bound_arg(func_addr),
                bound_arg(dict),
                bound_arg(key),
                bound_arg(hash),
                bound_arg(flag),
            ],
        );
        op2.setdescr(descr.clone());
        op2.pos().set(pos2);
        let op2_rc = OpRc::new(op2.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op2_rc));
        assert!(heap._optimize_call_dict_lookup(&op2, &op2_rc, &mut ctx));
        assert_eq!(ctx.get_replacement_opref(pos2), pos1);
        assert!(heap.last_emitted_removed);
    }

    /// heap.py:495-499 parity: FLAG_STORE reuses only if cached value >= 0.
    #[test]
    fn test_dict_lookup_cache_flag_store_nonneg() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        // Bind the dict/key InputArg slots so they resolve to canonical
        // boxes (production binds these via main-trace bind-at-alloc); the
        // box-identity dict cache key needs a bound box per arg.
        let mut ctx = OptContext::with_inputarg_types(256, &[Type::Ref, Type::Ref]);

        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let key = OpRef::input_arg_typed(1, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag_lookup = ctx.make_constant_int(0); // FLAG_LOOKUP
        let flag_store = ctx.make_constant_int(1); // FLAG_STORE

        // Seed cache with FLAG_LOOKUP, then try FLAG_STORE.
        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(
            OpCode::CallI,
            &[
                bound_arg(func_addr),
                bound_arg(dict),
                bound_arg(key),
                bound_arg(hash),
                bound_arg(flag_lookup),
            ],
        );
        op1.setdescr(descr.clone());
        op1.pos().set(pos1);
        // Pretend the result is known >= 0. The cache stores this op box
        // (`OptHeap._optimize_CALL_DICT_LOOKUP` `d[key] = op`), so
        // `getintbound(res_v)` reads the bound on that same box.
        let op1_rc = OpRc::new(op1.clone());
        ctx.setintbound(
            &Operand::from_bound_op(&op1_rc),
            &crate::optimizeopt::intutils::IntBound::from_constant(5),
        );
        assert!(!heap._optimize_call_dict_lookup(&op1, &op1_rc, &mut ctx));

        // FLAG_STORE with known non-negative cached value → reuse.
        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op2 = Op::new(
            OpCode::CallI,
            &[
                bound_arg(func_addr),
                bound_arg(dict),
                bound_arg(key),
                bound_arg(hash),
                bound_arg(flag_store),
            ],
        );
        op2.setdescr(descr.clone());
        op2.pos().set(pos2);
        let op2_rc = OpRc::new(op2.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op2_rc));
        assert!(heap._optimize_call_dict_lookup(&op2, &op2_rc, &mut ctx));
        assert_eq!(ctx.get_replacement_opref(pos2), pos1);
    }

    /// heap.py parity: clean_caches clears cached_dict_reads.
    #[test]
    fn test_dict_lookup_cache_cleared_by_clean_caches() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        // Bind the dict/key InputArg slots so they resolve to canonical
        // boxes (production binds these via main-trace bind-at-alloc); the
        // box-identity dict cache key needs a bound box per arg.
        let mut ctx = OptContext::with_inputarg_types(256, &[Type::Ref, Type::Ref]);

        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let key = OpRef::input_arg_typed(1, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag = ctx.make_constant_int(0);

        // Seed cache.
        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(
            OpCode::CallI,
            &[
                bound_arg(func_addr),
                bound_arg(dict),
                bound_arg(key),
                bound_arg(hash),
                bound_arg(flag),
            ],
        );
        op1.setdescr(descr.clone());
        op1.pos().set(pos1);
        heap._optimize_call_dict_lookup(&op1, &OpRc::new(op1.clone()), &mut ctx);
        assert!(!heap.cached_dict_reads.is_empty());

        // clean_caches should clear it.
        heap.clean_caches(&mut ctx);
        assert!(heap.cached_dict_reads.is_empty());

        // Second lookup after clean — should NOT be cached.
        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op2 = Op::new(
            OpCode::CallI,
            &[
                bound_arg(func_addr),
                bound_arg(dict),
                bound_arg(key),
                bound_arg(hash),
                bound_arg(flag),
            ],
        );
        op2.setdescr(descr.clone());
        op2.pos().set(pos2);
        assert!(!heap._optimize_call_dict_lookup(&op2, &OpRc::new(op2.clone()), &mut ctx));
    }

    /// util.py/127 args_dict() / args_eq parity: same_box treats two
    /// distinct ConstInt slots holding the same value as equal
    /// (history.py Const.same_box → same_constant). Two consecutive
    /// dict lookups whose constant key arg is encoded via different const
    /// slots must hit the cache.
    #[test]
    fn test_dict_lookup_cache_key_same_box_for_constants() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        // Bind the dict InputArg slot so it resolves to a canonical box
        // (production binds these via main-trace bind-at-alloc); the
        // box-identity dict cache key needs a bound box per arg.
        let mut ctx = OptContext::with_inputarg_types(256, &[Type::Ref, Type::Ref]);

        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag = ctx.make_constant_int(0);

        // history.py `ConstInt.same_constant` — Const equality is
        // value-based (not identity), so `ConstInt(7) == ConstInt(7)`
        // regardless of which call produced the box. Two
        // `make_constant_int(7)` calls return inline-Const OpRefs whose
        // variant payloads compare equal; `args_dict()` hashes them via
        // `_get_hash_()` (value-based for Const, history.py), so
        // they collide as the same key.
        let key_a = ctx.make_constant_int(7);
        let key_b = ctx.make_constant_int(7);
        assert_eq!(
            key_a, key_b,
            "ConstInt(7) compares equal to ConstInt(7) by value (history.py:251 same_constant)"
        );

        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(
            OpCode::CallI,
            &[
                bound_arg(func_addr),
                bound_arg(dict),
                bound_arg(key_a),
                bound_arg(hash),
                bound_arg(flag),
            ],
        );
        op1.setdescr(descr.clone());
        op1.pos().set(pos1);
        assert!(!heap._optimize_call_dict_lookup(&op1, &OpRc::new(op1.clone()), &mut ctx));

        // Same value via a different const slot — must hit the cache.
        let pos2 = ctx.reserve_pos_typed(Type::Int);
        let mut op2 = Op::new(
            OpCode::CallI,
            &[
                bound_arg(func_addr),
                bound_arg(dict),
                bound_arg(key_b),
                bound_arg(hash),
                bound_arg(flag),
            ],
        );
        op2.setdescr(descr.clone());
        op2.pos().set(pos2);
        let op2_rc = OpRc::new(op2.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op2_rc));
        assert!(heap._optimize_call_dict_lookup(&op2, &op2_rc, &mut ctx));
        assert_eq!(ctx.get_replacement_opref(pos2), pos1);
    }

    /// optimizer.py parity: every Optimization.emit overwrite path —
    /// including the PASS_OP_ON return at line 87 — sets
    /// `last_emitted_operation = op` first (line 86). Heap arms returning
    /// PassOn (NEW family, COND_CALL_N) hit that path in RPython, so a
    /// REMOVED sentinel left over from `_optimize_CALL_DICT_LOOKUP` must
    /// be cleared by the time a downstream GUARD_NO_EXCEPTION is dispatched
    /// — otherwise the guard is wrongly removed across an intervening op.
    #[test]
    fn test_pass_on_clears_last_emitted_removed_flag() {
        let mut heap = OptHeap::new();
        heap.setup();
        let mut ctx = OptContext::new(256);

        // Seed the REMOVED sentinel as if a DICT_LOOKUP cache hit just fired.
        heap.last_emitted_removed = true;

        // OpCode::New is a PassOn arm in heap.rs (allocation tracking only).
        let new_pos = ctx.reserve_pos_typed(Type::Ref);
        let mut new_op = Op::new(OpCode::New, &[]);
        new_op.pos().set(new_pos);
        let result = heap.propagate_forward(&new_op, &OpRc::new(new_op.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
        assert!(
            !heap.last_emitted_removed,
            "PassOn must clear last_emitted_removed (optimizer.py:86 fires before line 87 PASS_OP_ON)"
        );

        // Now a GUARD_NO_EXCEPTION must be emitted, not removed.
        let mut guard = Op::new(OpCode::GuardNoException, &[]);
        guard.pos().set(ctx.reserve_pos_typed(Type::Void));
        let guard_result = heap.propagate_forward(&guard, &OpRc::new(guard.clone()), &mut ctx);
        assert!(
            matches!(guard_result, OptimizationResult::PassOn),
            "GUARD_NO_EXCEPTION after a PassOn-emitted op must NOT be removed"
        );
    }

    /// FLAG_DELETE (2+) should never cache or reuse.
    #[test]
    fn test_dict_lookup_cache_flag_delete_no_cache() {
        let extra_field: DescrRef = descr(80);
        let extra_array: DescrRef = descr(81);
        let descr = dict_lookup_descr(90, extra_field, extra_array);

        let mut heap = OptHeap::new();
        let mut ctx = OptContext::new(256);

        let func_addr = ctx.make_constant_int(0xDEAD);
        let dict = OpRef::input_arg_typed(0, Type::Ref);
        let key = OpRef::input_arg_typed(1, Type::Ref);
        let hash = ctx.make_constant_int(42);
        let flag_delete = ctx.make_constant_int(2); // FLAG_DELETE

        let pos1 = ctx.reserve_pos_typed(Type::Int);
        let mut op1 = Op::new(
            OpCode::CallI,
            &[
                bound_arg(func_addr),
                bound_arg(dict),
                bound_arg(key),
                bound_arg(hash),
                bound_arg(flag_delete),
            ],
        );
        op1.setdescr(descr.clone());
        op1.pos().set(pos1);
        assert!(!heap._optimize_call_dict_lookup(&op1, &OpRc::new(op1.clone()), &mut ctx));
        assert!(heap.cached_dict_reads.is_empty());
    }
}
