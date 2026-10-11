/// Heap cache for the tracing phase.
///
/// During tracing, the heap cache tracks field reads/writes to eliminate
/// redundant loads. If we read a field from an object and it was already
/// read or written in the same trace, we can reuse the cached value.
///
/// Translated from rpython/jit/metainterp/heapcache.py.
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use indexmap::IndexSet;

use majit_ir::{EffectInfo, ExtraEffect, GcRef, OpCode, OpRef, Type, Value};

/// Value-equality predicate over constant OpRefs.  Mirrors
/// `Const.same_constant` (history.py): two ConstInt/ConstFloat/
/// ConstPtr instances are equal when they share the same subclass and
/// underlying value, independent of Box identity.
///
/// The trait is defined here (not in `majit-ir`) because `majit-trace`
/// is the lowest crate that needs the predicate (for the
/// `_unique_const_heuristic` ConstPtr canonicalisation,
/// heapcache.py) and the implementation lives in `majit-metainterp`
/// (`history::ConstOprefOracle`).  `&dyn SameConstantOracle`
/// keeps the heapcache layer agnostic of the oracle's representation.
/// The `field_index` that stands for a descr no numberer ever reached.
///
/// `heapcache.py get_field_updater(box, descr)` keys the per-box field
/// cache on the descr object, so two fields of one struct can never share
/// an entry.  Flattening that key to `Descr::index()` preserves the
/// property only while the number is assigned: the unassigned answer is
/// the `u32::MAX` sentinel (`descr.rs Descr::index`), a single key
/// standing for every field of every struct.
///
/// Distinct structs colliding here is harmless — an entry is read back
/// per box, and a box belongs to one struct.  What the flattening cannot
/// express is *this* case: one box's own fields, which reach the cache
/// with distinct offsets and so distinct numbers everywhere the number
/// exists, all aliasing onto one entry when it does not.  Decline the
/// sentinel rather than let a store under it answer a load of a
/// different field.
const UNNUMBERED_FIELD: u32 = u32::MAX;

/// Field-cache operations declined because the descr carried
/// [`UNNUMBERED_FIELD`].  A numberer covering every descr that reaches
/// the cache leaves this at zero; a non-zero count names lost caching,
/// not a wrong answer.
pub static UNNUMBERED_FIELD_DECLINES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Reads [`UNNUMBERED_FIELD_DECLINES`].
pub fn unnumbered_field_declines() -> u64 {
    UNNUMBERED_FIELD_DECLINES.load(std::sync::atomic::Ordering::Relaxed)
}

/// True when `field_index` names no field, counting the decline.
fn declines_unnumbered_field(field_index: u32) -> bool {
    let unnumbered = field_index == UNNUMBERED_FIELD;
    if unnumbered {
        UNNUMBERED_FIELD_DECLINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // A producer that reaches the cache is expected to have been
        // numbered; the release build declines and keeps going, and this
        // names the producer that was not rather than leaving the loss
        // to a counter nobody reads.  `pyjitpl/dispatch.rs` screens the
        // sentinel out ahead of the call for hand-assembled jitcodes,
        // which are the numberer's documented gap.
        debug_assert!(
            false,
            "field heapcache reached with an unnumbered descr: every field of \
             every struct aliases onto this one key, so the entry is declined"
        );
    }
    unnumbered
}

pub trait SameConstantOracle {
    fn same_constant(&self, a: OpRef, b: OpRef) -> bool;
}

// heapcache.py: HF_* flags stored per-box on RefFrontendOp.
// In majit these are tracked via separate HashSets (is_unescaped,
// seen_allocation, etc.), but we define the constants for reference.

bitflags::bitflags! {
    /// heapcache.py `HF_*` flags stored per-box on RefFrontendOp.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct HeapFlags: u8 {
        const LIKELY_VIRTUAL = 0x01;
        const KNOWN_CLASS = 0x02;
        const KNOWN_NULLITY = 0x04;
        const SEEN_ALLOCATION = 0x08;
        const IS_UNESCAPED = 0x10;
        const NONSTD_VABLE = 0x20;
    }
}

/// heapcache.py helper aliases.
const HF_VERSION_INC: u32 = 0x40;
pub const HF_VERSION_MAX: u32 = 0xffff_ffff - HF_VERSION_INC;
const _HF_VERSION_INC: u32 = HF_VERSION_INC;
const _HF_VERSION_MAX: u32 = HF_VERSION_MAX;

/// Per-box heapcache state stored on the FrontendOp record
/// (`history.py` `RefFrontendOp._heapc_flags` / `_heapc_deps`,
/// `FrontendOp.position_and_flags & FO_REPLACED_WITH_CONST`).
#[derive(Clone, Debug, Default)]
pub struct HeapcRecord {
    /// `RefFrontendOp._heapc_flags` — HF_* bits plus the version word
    /// `HeapCache.test_head_version` / `test_likely_virtual_version` compare.
    pub flags: u32,
    /// `RefFrontendOp._heapc_deps` — `deps[0]` is the cached array length,
    /// `deps[1:]` are escape dependencies from `_escape_from_write`.
    pub deps: Option<Vec<Option<OpRef>>>,
    /// `FrontendOp.position_and_flags & FO_REPLACED_WITH_CONST`.
    /// The replacement Const is `constant_from_op(box)` of this box's value.
    pub replaced_with_const: bool,
}

impl HeapcRecord {
    /// Forward ConstPtrs stored in `_heapc_deps`.
    pub fn walk_const_ptr_refs(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        if let Some(deps) = &mut self.deps {
            for slot in deps.iter_mut().flatten() {
                if let OpRef::ConstPtr(gcref) = slot {
                    visitor(gcref);
                }
            }
        }
    }
}

/// Store of FrontendOp records the heapcache reads and writes.
///
/// Inputargs occupy positions `0.._start`; value ops occupy `_index` after
/// that (`opencoder.py` `Trace._start`). A `Const` has no record.
pub trait HeapcBoxes {
    fn heapc(&self, opref: OpRef) -> Option<&HeapcRecord>;
    fn heapc_mut(&mut self, opref: OpRef) -> Option<&mut HeapcRecord>;
    /// `*FrontendOp.getint` / `getref_base` / `getfloatstorage` for
    /// `executor.constant_from_op` and `cls_of_box` / `box.nonnull()`.
    fn box_value(&self, opref: OpRef) -> Option<Value>;
}

/// heapcache.py `add_flags(ref_frontend_op, flags)`.
pub fn add_flags(boxes: &mut dyn HeapcBoxes, opref: OpRef, flags: HeapFlags) {
    if let Some(rec) = boxes.heapc_mut(opref) {
        rec.flags |= u32::from(flags.bits());
    }
}

/// heapcache.py `remove_flags(ref_frontend_op, flags)`.
pub fn remove_flags(boxes: &mut dyn HeapcBoxes, opref: OpRef, flags: HeapFlags) {
    if let Some(rec) = boxes.heapc_mut(opref) {
        rec.flags &= !u32::from(flags.bits());
    }
}

/// heapcache.py `test_flags(ref_frontend_op, flags)`.
pub fn test_flags(boxes: &dyn HeapcBoxes, opref: OpRef, flags: HeapFlags) -> bool {
    let f = boxes.heapc(opref).map(|rec| rec.flags).unwrap_or(0);
    (f & u32::from(flags.bits())) != 0
}

/// `executor.py constant_from_op(op)` — Const minted from the box's own value.
fn constant_from_op(opref: OpRef, boxes: &dyn HeapcBoxes) -> OpRef {
    match boxes.box_value(opref) {
        Some(Value::Int(n)) => OpRef::const_int(n),
        Some(Value::Ref(g)) => OpRef::const_ptr(g),
        Some(Value::Float(f)) => OpRef::const_float(f),
        Some(Value::Void) | None => opref,
    }
}

/// heapcache.py CacheEntry — per-descr cache of fieldbox values.
///
/// `cache_anything` / `cache_seen_allocation` store the cached
/// fieldbox as a bare [`OpRef`] — the Box identity itself.  RPython
/// `heapcache.py cache_anything[box] = valuebox` stores a Box
/// object (carrying both identity and value); pyre carries the same
/// fact through `OpRef` + the frontend object's `value`
/// (`Op` / `InputArg` `value: Cell<Option<Value>>`).  Cache-hit sanity
/// checks (`pyjitpl.py:937 assert resvalue == upd.currfieldbox.
/// getint()`) read the cached OpRef's value via
/// `TraceCtx::box_value` — composing the const pool, standard-
/// virtualizable shadow, and the frontend object's `value` field in one
/// call.  No separate side table.
#[derive(Clone, Debug, Default)]
pub struct CacheEntry {
    cache_anything: vecset::VecMap<OpRef, OpRef>,
    cache_seen_allocation: vecset::VecMap<OpRef, OpRef>,
    quasiimmut_seen: Option<IndexSet<OpRef>>,
    quasiimmut_seen_refs: Option<IndexSet<usize>>,
    last_const_box: Option<OpRef>,
}

impl CacheEntry {
    pub fn new() -> Self {
        Self::default()
    }

    /// heapcache.py _clear_cache_on_write
    pub fn _clear_cache_on_write(&mut self, seen_allocation_of_target: bool) {
        if !seen_allocation_of_target {
            self.cache_seen_allocation.clear();
        }
        self.cache_anything.clear();
        if let Some(seen) = &mut self.quasiimmut_seen {
            seen.clear();
        }
        if let Some(seen) = &mut self.quasiimmut_seen_refs {
            seen.clear();
        }
    }

    /// heapcache.py _seen_alloc
    ///
    /// Pyre adapt: needs an explicit `cache: &HeapCache` parameter
    /// because `CacheEntry` is a separate struct from `HeapCache`
    /// (RPython attaches the heapcache reference to CacheEntry at
    /// __init__ time; in Rust we pass it through to avoid a back-
    /// reference + interior mutability dance).
    pub fn _seen_alloc(&self, ref_box: OpRef, cache: &HeapCache, boxes: &dyn HeapcBoxes) -> bool {
        cache.saw_allocation(ref_box, boxes)
    }

    /// heapcache.py _getdict
    pub fn _getdict(&self, seen_alloc: bool) -> &vecset::VecMap<OpRef, OpRef> {
        if seen_alloc {
            &self.cache_seen_allocation
        } else {
            &self.cache_anything
        }
    }

    /// Pyre adapt: Python doesn't need a separate `_mut` accessor;
    /// Rust's borrow checker does.  Mirrors `_getdict`'s body.
    pub fn _getdict_mut(&mut self, seen_alloc: bool) -> &mut vecset::VecMap<OpRef, OpRef> {
        if seen_alloc {
            &mut self.cache_seen_allocation
        } else {
            &mut self.cache_anything
        }
    }

    /// heapcache.py do_write_with_aliasing
    pub fn do_write_with_aliasing(
        &mut self,
        ref_box: OpRef,
        fieldbox: OpRef,
        cache: &HeapCache,
        boxes: &dyn HeapcBoxes,
        oracle: &dyn SameConstantOracle,
    ) {
        let ref_box = self._unique_const_heuristic(ref_box, oracle);
        let seen_alloc = self._seen_alloc(ref_box, cache, boxes);
        self._clear_cache_on_write(seen_alloc);
        self._getdict_mut(seen_alloc).insert(ref_box, fieldbox);
    }

    /// heapcache.py _unique_const_heuristic.
    ///
    /// Only ConstPtr operands are canonicalised; non-constant OpRefs and
    /// non-Ref-typed constants pass through unchanged (matches the
    /// `isinstance(ref_box, ConstPtr)` guard on heapcache.py).
    /// `oracle.same_constant(last, ref_box)` is the value-aware
    /// comparison upstream uses (history.py `Const.same_constant`).
    pub fn _unique_const_heuristic(
        &mut self,
        ref_box: OpRef,
        oracle: &dyn SameConstantOracle,
    ) -> OpRef {
        if !(ref_box.is_constant() && ref_box.ty() == Some(Type::Ref)) {
            return ref_box;
        }
        if let Some(last) = self.last_const_box
            && oracle.same_constant(last, ref_box)
        {
            return last;
        }
        self.last_const_box = Some(ref_box);
        ref_box
    }

    /// heapcache.py read
    pub fn read(
        &mut self,
        ref_box: OpRef,
        cache: &HeapCache,
        boxes: &dyn HeapcBoxes,
        oracle: &dyn SameConstantOracle,
    ) -> Option<OpRef> {
        let ref_box = self._unique_const_heuristic(ref_box, oracle);
        let seen_alloc = self._seen_alloc(ref_box, cache, boxes);
        self._getdict(seen_alloc)
            .get(&ref_box)
            .map(|fieldbox| cache.maybe_replace_with_const(*fieldbox, boxes))
    }

    /// heapcache.py read_now_known
    pub fn read_now_known(
        &mut self,
        ref_box: OpRef,
        fieldbox: OpRef,
        cache: &HeapCache,
        boxes: &dyn HeapcBoxes,
        oracle: &dyn SameConstantOracle,
    ) {
        let ref_box = self._unique_const_heuristic(ref_box, oracle);
        let seen_alloc = self._seen_alloc(ref_box, cache, boxes);
        self._getdict_mut(seen_alloc).insert(ref_box, fieldbox);
    }

    /// heapcache.py invalidate_unescaped — RPython makes this a
    /// public method (no underscore prefix) and `_invalidate_unescaped`
    /// is the helper that walks both caches.  pyre keeps the same
    /// public/private pair.
    ///
    /// `cache: &HeapCache` matches upstream's stored-back-reference
    /// `self.heapcache` (heapcache.py `self.heapcache = heapcache`)
    /// so the per-entry filter calls the version-gated
    /// `HeapCache.is_unescaped(ref_box)` (heapcache.py / 457-460)
    /// instead of any pre-snapshotted bit table.
    pub fn invalidate_unescaped(&mut self, cache: &HeapCache, boxes: &dyn HeapcBoxes) {
        self._invalidate_unescaped(cache, boxes)
    }

    pub fn _invalidate_unescaped(&mut self, cache: &HeapCache, boxes: &dyn HeapcBoxes) {
        self.cache_anything
            .retain(|&ref_box, _| cache.is_unescaped(ref_box, boxes));
        self.cache_seen_allocation
            .retain(|&ref_box, _| cache.is_unescaped(ref_box, boxes));
        if let Some(seen) = &mut self.quasiimmut_seen {
            seen.clear();
        }
        if let Some(seen) = &mut self.quasiimmut_seen_refs {
            seen.clear();
        }
    }
}

/// RPython heapcache.py: FieldUpdater helper struct.
///
/// In Rust, safe ownership makes this harder to express directly, so it stores
/// a raw pointer back to the cache for writeback.
pub struct FieldUpdater<'a> {
    ref_box: OpRef,
    currfieldbox: Option<OpRef>,
    cache: *mut HeapCache,
    boxes: *mut (dyn HeapcBoxes + 'a),
    descr: Option<u32>,
    _marker: PhantomData<&'a mut HeapCache>,
}

impl<'a> FieldUpdater<'a> {
    pub fn with_cache(
        ref_box: OpRef,
        cache: &'a mut HeapCache,
        boxes: &'a mut dyn HeapcBoxes,
        descr: u32,
        fieldbox: Option<OpRef>,
    ) -> Self {
        Self {
            ref_box,
            currfieldbox: fieldbox,
            cache: cache as *mut HeapCache,
            boxes,
            descr: Some(descr),
            _marker: PhantomData,
        }
    }

    /// heapcache.py `self.currfieldbox` reader — exposes the
    /// in-flight Box the updater is wrapping.  Mirrors `pyjitpl.py:931
    /// upd.currfieldbox` direct attribute access.  Pyre carries the
    /// Box identity as an `OpRef`; downstream sanity readers look up
    /// the intrinsic value via `TraceCtx::box_value` (composing const
    /// pool, standard-virtualizable shadow, the frontend object's
    /// `value` field).
    pub fn currfieldbox(&self) -> Option<OpRef> {
        self.currfieldbox
    }

    /// heapcache.py getfield_now_known
    ///
    /// ```text
    ///  def getfield_now_known(self, fieldbox):
    ///      self.cache.read_now_known(self.ref_box, fieldbox)
    /// ```
    pub fn getfield_now_known(&mut self, fieldbox: OpRef, oracle: &dyn SameConstantOracle) {
        let ref_box = self.ref_box;
        let (cache, boxes, descr_index) = match self.cache_and_descr() {
            Some(pair) => pair,
            None => return,
        };
        let mut entry = cache.heap_cache.remove(&descr_index).unwrap_or_default();
        entry.read_now_known(ref_box, fieldbox, cache, boxes, oracle);
        cache.heap_cache.insert(descr_index, entry);
    }

    /// heapcache.py setfield
    ///
    /// ```text
    ///  def setfield(self, fieldbox):
    ///      self.cache.do_write_with_aliasing(self.ref_box, fieldbox)
    /// ```
    pub fn setfield(&mut self, fieldbox: OpRef, oracle: &dyn SameConstantOracle) {
        let ref_box = self.ref_box;
        let (cache, boxes, descr_index) = match self.cache_and_descr() {
            Some(pair) => pair,
            None => return,
        };
        let mut entry = cache.heap_cache.remove(&descr_index).unwrap_or_default();
        entry.do_write_with_aliasing(ref_box, fieldbox, cache, boxes, oracle);
        cache.heap_cache.insert(descr_index, entry);
    }

    fn cache_and_descr(&mut self) -> Option<(&mut HeapCache, &mut (dyn HeapcBoxes + 'a), u32)> {
        let descr_index = self.descr?;
        if self.cache.is_null() {
            return None;
        }
        // SAFETY: `cache` / `boxes` were supplied by `with_cache`; the
        // FieldUpdater's lifetime must not outlive those borrows
        // (callers hold it stack-locally during a single trace step,
        // matching `pyjitpl.py` `upd = heapcache.get_field_updater(...)`
        // consumed before any other heapcache operation).
        let cache = unsafe { &mut *self.cache };
        let boxes = unsafe { &mut *self.boxes };
        Some((cache, boxes, descr_index))
    }
}

/// Heap cache for the tracing interpreter.
///
/// Tracks field values, known classes, and allocation status during
/// a single trace recording session.
#[derive(Clone)]
pub struct HeapCache {
    /// heapcache.py:172 `self.heap_cache = {}` — maps descrs to
    /// `CacheEntry`.  Field reads/writes for a given descr land in the
    /// same `CacheEntry`, which owns the `cache_anything` /
    /// `cache_seen_allocation` dicts and the `last_const_box`
    /// `_unique_const_heuristic` LRU per heapcache.py.
    /// Backed by `vecset::VecMap` (sorted Vec + binary search) so the
    /// hot per-descr lookup is O(log n) instead of linear scan when the
    /// same descr is touched repeatedly across many frames.
    heap_cache: vecset::VecMap<u32, CacheEntry>,
    /// heapcache.py: `cached_arrayitems` — nested map descr → ConstInt-index → CacheEntry.
    /// heapcache.py `cache.get(index, None)` — array cache keyed by
    /// the `ConstInt.getint()` value, not the index Box's identity. Two
    /// distinct ConstInt boxes carrying the same `i64` index land in the
    /// same slot, matching the upstream lookup semantics. `i64` indices
    /// can be negative, so `vecset::VecMap` (sorted Vec + binary search)
    /// is the natural no-HashMap substitute.
    heap_array_cache: vecset::VecMap<u32, vecset::VecMap<i64, CacheEntry>>,

    /// heapcache.py: loop-invariant call result cache.
    /// RPython stores exactly ONE result: (descr, arg0_int) → result.
    /// Subsequent calls overwrite the single entry.
    ///
    /// TODO: upstream's `result` is a Box that
    /// carries both the symbolic identity and the concrete value
    /// together; pyre splits these into the symbolic `OpRef` plus a
    /// concrete `i64` so `do_residual_call` can return the same
    /// `(opref, value)` tuple shape on cache hits as it does on
    /// freshly-executed calls.
    loopinvariant_descr: Option<u32>,
    loopinvariant_arg0: Option<i64>,
    loopinvariant_result: Option<OpRef>,
    loopinvariant_resvalue: Option<i64>,

    /// heapcache.py: need_guard_not_invalidated — set True on reset,
    /// consumed by quasi-immut field recording to decide whether to emit
    /// GUARD_NOT_INVALIDATED.
    need_guard_not_invalidated: bool,

    head_version: u32,
    likely_virtual_version: u32,
}

impl HeapCache {
    /// Create a new, empty heap cache.
    pub fn new() -> Self {
        HeapCache {
            heap_cache: vecset::VecMap::new(),
            heap_array_cache: vecset::VecMap::new(),
            loopinvariant_descr: None,
            loopinvariant_arg0: None,
            loopinvariant_result: None,
            loopinvariant_resvalue: None,
            need_guard_not_invalidated: true,
            head_version: 0,
            likely_virtual_version: 0,
        }
    }

    /// heapcache.py `maybe_replace_with_const(box)`.
    pub(crate) fn maybe_replace_with_const(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> OpRef {
        if opref.is_constant() {
            return opref;
        }
        if boxes
            .heapc(opref)
            .is_some_and(|rec| rec.replaced_with_const)
        {
            let replaced = constant_from_op(opref, boxes);
            if replaced.ty() == opref.ty() {
                return replaced;
            }
        }
        opref
    }

    fn flags_for_ref(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> u32 {
        boxes.heapc(opref).map(|rec| rec.flags).unwrap_or(0)
    }

    fn set_flags_for_ref(&self, opref: OpRef, flags: u32, boxes: &mut dyn HeapcBoxes) {
        if let Some(rec) = boxes.heapc_mut(opref) {
            rec.flags = flags;
        }
    }

    fn versioned_or(self_flags: u32, op_version: u32) -> bool {
        self_flags >= op_version
    }

    /// RPython: test_head_version(ref_frontend_op)
    pub fn test_head_version(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> bool {
        Self::versioned_or(self.flags_for_ref(opref, boxes), self.head_version)
    }

    /// RPython: test_likely_virtual_version(ref_frontend_op)
    pub fn test_likely_virtual_version(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> bool {
        Self::versioned_or(
            self.flags_for_ref(opref, boxes),
            self.likely_virtual_version,
        )
    }

    /// RPython: update_version(ref_frontend_op)
    /// heapcache.py
    ///
    /// ```text
    ///  def update_version(self, ref_frontend_op):
    ///      """Ensure the version of 'ref_frontend_op' is current. If not,
    ///      it will update 'ref_frontend_op' (removing most flags currently set).
    ///      """
    ///      if not self.test_head_version(ref_frontend_op):
    ///          f = self.head_version
    ///          if (self.test_likely_virtual_version(ref_frontend_op) and
    ///              test_flags(ref_frontend_op, HF_LIKELY_VIRTUAL)):
    ///              f |= HF_LIKELY_VIRTUAL
    ///          ref_frontend_op._set_heapc_flags(f)
    ///          ref_frontend_op._heapc_deps = None
    /// ```
    pub fn update_version(&mut self, opref: OpRef, boxes: &mut dyn HeapcBoxes) {
        let old_flags = self.flags_for_ref(opref, boxes);
        if Self::versioned_or(old_flags, self.head_version) {
            return;
        }
        let mut flags = self.head_version;
        if Self::versioned_or(old_flags, self.likely_virtual_version)
            && (old_flags & u32::from(HeapFlags::LIKELY_VIRTUAL.bits())) != 0
        {
            flags |= u32::from(HeapFlags::LIKELY_VIRTUAL.bits());
        }
        self.set_flags_for_ref(opref, flags, boxes);
        // RPython: ref_frontend_op._heapc_deps = None
        self._remove_deps_for_box(opref, boxes);
    }

    /// RPython: _check_flag(box, flag)
    pub fn _check_flag(&self, opref: OpRef, flag: HeapFlags, boxes: &dyn HeapcBoxes) -> bool {
        if !self.test_head_version(opref, boxes) {
            return false;
        }
        (self.flags_for_ref(opref, boxes) & u32::from(flag.bits())) != 0
    }

    /// RPython: _set_flag(box, flag)
    pub fn _set_flag(&mut self, opref: OpRef, flag: HeapFlags, boxes: &mut dyn HeapcBoxes) {
        if opref.is_constant() {
            return;
        }
        self.update_version(opref, boxes);
        add_flags(boxes, opref, flag);
    }

    /// `heapcache.py HeapCache._get_deps`.
    ///
    /// ```text
    ///  def _get_deps(self, box):
    ///      if not isinstance(box, RefFrontendOp):
    ///          return None
    ///      self.update_version(box)
    ///      if box._heapc_deps is None:
    ///          box._heapc_deps = [None]
    ///      return box._heapc_deps
    /// ```
    ///
    /// The `isinstance` arm is why this returns an `Option`: a `Const` has no
    /// `_heapc_deps` slot.
    pub fn _get_deps<'b>(
        &mut self,
        opref: OpRef,
        boxes: &'b mut dyn HeapcBoxes,
    ) -> Option<&'b mut Vec<Option<OpRef>>> {
        if opref.is_constant() {
            return None;
        }
        self.update_version(opref, boxes);
        let rec = boxes.heapc_mut(opref)?;
        if rec.deps.is_none() {
            rec.deps = Some(vec![None]);
        }
        let deps = rec.deps.as_mut().unwrap();
        if deps.is_empty() {
            deps.push(None);
        }
        Some(deps)
    }

    /// heapcache.py _escape_from_write
    ///
    /// ```text
    ///  def _escape_from_write(self, box, fieldbox):
    ///      if self.is_unescaped(box) and self.is_unescaped(fieldbox):
    ///          deps = self._get_deps(box)
    ///          deps.append(fieldbox)
    ///      elif fieldbox is not None:
    ///          self._escape_box(fieldbox)
    /// ```
    pub fn _escape_from_write(
        &mut self,
        r#box: OpRef,
        fieldbox: OpRef,
        boxes: &mut dyn HeapcBoxes,
    ) {
        if self.is_unescaped(r#box, boxes) && self.is_unescaped(fieldbox, boxes) {
            let deps = self
                ._get_deps(r#box, boxes)
                .expect("is_unescaped answers false for a Const");
            deps.push(Some(fieldbox));
        } else {
            // RPython's `elif fieldbox is not None` — pyre's OpRef is always
            // present (no None equivalent), so the branch always fires.
            self._escape_box(fieldbox, boxes);
        }
    }

    /// heapcache.py `_escape_box(box)`.
    ///
    /// ```text
    ///  def _escape_box(self, box):
    ///      if isinstance(box, RefFrontendOp):
    ///          remove_flags(box, HF_LIKELY_VIRTUAL | HF_IS_UNESCAPED)
    ///          deps = box._heapc_deps
    ///          if deps is not None:
    ///              if not self.test_head_version(box):
    ///                  box._heapc_deps = None
    ///              else:
    ///                  # 'deps[0]' is abused to store the array length, keep it
    ///                  if deps[0] is None:
    ///                      box._heapc_deps = None
    ///                  else:
    ///                      box._heapc_deps = [deps[0]]
    ///                  for i in range(1, len(deps)):
    ///                      self._escape_box(deps[i])
    /// ```
    pub fn _escape_box(&mut self, opref: OpRef, boxes: &mut dyn HeapcBoxes) {
        if boxes.heapc(opref).is_none() {
            return;
        }
        remove_flags(
            boxes,
            opref,
            HeapFlags::LIKELY_VIRTUAL | HeapFlags::IS_UNESCAPED,
        );
        let deps = boxes.heapc_mut(opref).and_then(|rec| rec.deps.take());
        if let Some(deps) = deps
            && self.test_head_version(opref, boxes)
        {
            let kept_len = deps.first().cloned().flatten();
            if let Some(length) = kept_len
                && let Some(rec) = boxes.heapc_mut(opref)
            {
                rec.deps = Some(vec![Some(length)]);
            }
            for dep in deps.into_iter().skip(1).flatten() {
                self._escape_box(dep, boxes);
            }
        }
    }

    /// RPython: mark_escaped(opnum, descr, *argboxes) entrypoint.
    pub fn mark_escaped(
        &mut self,
        opnum: OpCode,
        _descr: Option<OpRef>,
        argboxes: &[OpRef],
        boxes: &mut dyn HeapcBoxes,
    ) {
        if opnum == OpCode::SetfieldGc {
            if argboxes.len() == 2 {
                self._escape_from_write(argboxes[0], argboxes[1], boxes);
            }
        } else if opnum == OpCode::SetarrayitemGc {
            if argboxes.len() == 3 {
                self._escape_from_write(argboxes[0], argboxes[2], boxes);
            }
        } else if !matches!(
            opnum,
            OpCode::GetfieldGcR
                | OpCode::GetfieldGcI
                | OpCode::GetfieldGcF
                | OpCode::PtrEq
                | OpCode::PtrNe
                | OpCode::InstancePtrEq
                | OpCode::InstancePtrNe
                | OpCode::AssertNotNone
        ) {
            self._escape_argboxes(argboxes, boxes);
        }
    }

    /// heapcache.py mark_escaped_varargs.
    ///
    /// Upstream splits the two flavors:
    ///   * `mark_escaped` (line 232) handles SETFIELD_GC / SETARRAYITEM_GC
    ///     and asserts `opnum != CALL_N`.
    ///   * `mark_escaped_varargs` (line 259) handles CALL_N and special-cases
    ///     ARRAYCOPY / ARRAYMOVE with constant starts+length+single-descr to
    ///     skip arg escape entirely.
    ///
    /// `effectinfo` + `const_value` carry the upstream
    /// `descr.get_extra_info()` lookups; the closure returns the
    /// `ConstInt.getint()` value (heapcache.py / :284-286).
    pub fn mark_escaped_varargs<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        const_value: F,
        boxes: &mut dyn HeapcBoxes,
    ) {
        if opnum == OpCode::CallN {
            if let Some(ei) = effectinfo
                && ei.single_write_descr_array.is_some()
            {
                // heapcache.py:272-281: CALL_N + OS_ARRAYCOPY with all
                // three index/length operands ConstInt → don't escape
                // argboxes.
                if ei.oopspecindex == majit_ir::OopSpecIndex::Arraycopy
                    && argboxes.len() >= 6
                    && const_value(argboxes[3]).is_some()
                    && const_value(argboxes[4]).is_some()
                    && const_value(argboxes[5]).is_some()
                {
                    return;
                }
                // heapcache.py:282-290: CALL_N + OS_ARRAYMOVE with all
                // three operands ConstInt → don't escape argboxes.
                if ei.oopspecindex == majit_ir::OopSpecIndex::Arraymove
                    && argboxes.len() >= 5
                    && const_value(argboxes[2]).is_some()
                    && const_value(argboxes[3]).is_some()
                    && const_value(argboxes[4]).is_some()
                {
                    return;
                }
            }
            // heapcache.py:291-293 fallback: escape all argboxes.
            self._escape_argboxes(argboxes, boxes);
            return;
        }
        self.mark_escaped(opnum, None, argboxes, boxes)
    }

    /// RPython: _escape_argboxes(*argboxes)
    pub fn _escape_argboxes(&mut self, args: &[OpRef], boxes: &mut dyn HeapcBoxes) {
        if args.is_empty() {
            return;
        }
        self._escape_box(args[0], boxes);
        self._escape_argboxes(&args[1..], boxes);
    }

    /// heapcache.py `getfield(self, box, descr)`.
    ///
    /// ```text
    ///  def getfield(self, box, descr):
    ///      cache = self.heap_cache.get(descr, None)
    ///      if cache:
    ///          return cache.read(box)
    ///      return None
    /// ```
    ///
    /// `CacheEntry.read` (heapcache.py) handles the
    /// `_unique_const_heuristic` ConstPtr canonicalisation and the
    /// `maybe_replace_with_const` forwarding internally.  We take the
    /// entry out of `heap_cache` for the duration of the call so the
    /// borrow checker accepts `&mut entry` and `&self.heap_cache`'s
    /// neighbour fields simultaneously, then put it back.
    pub fn getfield_cached(
        &mut self,
        obj: OpRef,
        field_index: u32,
        oracle: &dyn SameConstantOracle,
        boxes: &dyn HeapcBoxes,
    ) -> Option<OpRef> {
        if declines_unnumbered_field(field_index) {
            return None;
        }
        let mut entry = self.heap_cache.remove(&field_index)?;
        let result = entry.read(obj, self, boxes, oracle);
        self.heap_cache.insert(field_index, entry);
        result
    }

    /// heapcache.py `setfield(self, box, fieldbox, descr)`.
    ///
    /// ```text
    ///  def setfield(self, box, fieldbox, descr):
    ///      upd = self.get_field_updater(box, descr)
    ///      upd.setfield(fieldbox)
    /// ```
    ///
    /// The `upd.setfield` body is `cache.do_write_with_aliasing(ref_box,
    /// fieldbox)` (heapcache.py), which handles
    /// `_unique_const_heuristic`, `_clear_cache_on_write`, and the dict
    /// insertion in one step.  Aliasing semantics:
    /// `_clear_cache_on_write(seen_alloc)` clears `cache_anything` and,
    /// when `seen_alloc` is false (the target may alias anything else),
    /// also clears `cache_seen_allocation`, matching
    /// heapcache.py.
    pub fn setfield_cached(
        &mut self,
        obj: OpRef,
        field_index: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
        boxes: &dyn HeapcBoxes,
    ) {
        if declines_unnumbered_field(field_index) {
            return;
        }
        let mut entry = self.heap_cache.remove(&field_index).unwrap_or_default();
        entry.do_write_with_aliasing(obj, value, self, boxes, oracle);
        self.heap_cache.insert(field_index, entry);
    }

    /// heapcache.py `getfield_now_known(self, box, descr,
    /// fieldbox)`.
    ///
    /// ```text
    ///  def getfield_now_known(self, box, descr, fieldbox):
    ///      upd = self.get_field_updater(box, descr)
    ///      upd.getfield_now_known(fieldbox)
    /// ```
    ///
    /// `upd.getfield_now_known` delegates to
    /// `cache.read_now_known(ref_box, fieldbox)` (heapcache.py),
    /// which records the value without the aliasing-clear step.
    pub fn getfield_now_known(
        &mut self,
        obj: OpRef,
        field_index: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
        boxes: &dyn HeapcBoxes,
    ) {
        if declines_unnumbered_field(field_index) {
            return;
        }
        let mut entry = self.heap_cache.remove(&field_index).unwrap_or_default();
        entry.read_now_known(obj, value, self, boxes, oracle);
        self.heap_cache.insert(field_index, entry);
    }

    /// heapcache.py: invalidate_unescaped — clear cached values for
    /// escaped objects only. Unescaped (newly allocated) objects cannot
    /// be affected by external calls, so their caches are preserved.
    pub fn invalidate_caches_for_escaped(&mut self, boxes: &dyn HeapcBoxes) {
        // heapcache.py:362-365 — `for cache in self.heap_cache.itervalues():
        //                           cache.invalidate_unescaped()`.
        // Take/restore is the borrow-split equivalent of upstream's stored
        // back-reference (`CacheEntry.heapcache`): the entries are removed
        // so each `invalidate_unescaped` call receives a fresh `&HeapCache`
        // to run the version-gated `is_unescaped(ref_box)` check
        // (heapcache.py / 457-460) without the borrow checker
        // tripping over `entry` and `self.heap_cache` simultaneously.
        let mut heap_cache = std::mem::take(&mut self.heap_cache);
        for entry in heap_cache.values_mut() {
            entry.invalidate_unescaped(self, boxes);
        }
        self.heap_cache = heap_cache;
        // heapcache.py getarrayitem: iterate cached_arrayitems and invalidate
        // per-CacheEntry entries whose box is no longer unescaped.
        let mut heap_array_cache = std::mem::take(&mut self.heap_array_cache);
        for caches in heap_array_cache.values_mut() {
            for cache in caches.values_mut() {
                cache.invalidate_unescaped(self, boxes);
            }
        }
        self.heap_array_cache = heap_array_cache;
    }

    /// heapcache.py new
    ///
    /// ```text
    ///  def new(self, box):
    ///      assert isinstance(box, RefFrontendOp)
    ///      self.update_version(box)
    ///      add_flags(box, HF_LIKELY_VIRTUAL | HF_SEEN_ALLOCATION | HF_IS_UNESCAPED
    ///                     | HF_KNOWN_NULLITY)
    /// ```
    pub fn new_object(&mut self, opref: OpRef, boxes: &mut dyn HeapcBoxes) {
        if opref.is_constant() {
            return;
        }
        self.update_version(opref, boxes);
        add_flags(
            boxes,
            opref,
            HeapFlags::LIKELY_VIRTUAL
                | HeapFlags::SEEN_ALLOCATION
                | HeapFlags::IS_UNESCAPED
                | HeapFlags::KNOWN_NULLITY,
        );
    }

    /// heapcache.py new_array
    ///
    /// ```text
    ///  def new_array(self, box, lengthbox):
    ///      assert isinstance(box, RefFrontendOp)
    ///      self.update_version(box)
    ///      flags = HF_SEEN_ALLOCATION | HF_KNOWN_NULLITY
    ///      if isinstance(lengthbox, Const):
    ///          # only constant-length arrays are virtuals
    ///          flags |= HF_LIKELY_VIRTUAL | HF_IS_UNESCAPED
    ///      add_flags(box, flags)
    ///      self.arraylen_now_known(box, lengthbox)
    /// ```
    pub fn new_array(
        &mut self,
        opref: OpRef,
        lengthbox: OpRef,
        length_is_const: bool,
        boxes: &mut dyn HeapcBoxes,
    ) {
        if opref.is_constant() {
            return;
        }
        self.update_version(opref, boxes);
        let mut flags = HeapFlags::SEEN_ALLOCATION | HeapFlags::KNOWN_NULLITY;
        if length_is_const {
            flags |= HeapFlags::LIKELY_VIRTUAL | HeapFlags::IS_UNESCAPED;
        }
        add_flags(boxes, opref, flags);
        self.arraylen_now_known(opref, lengthbox, boxes);
    }

    /// heapcache.py is_known_nonstandard_virtualizable
    ///
    /// ```text
    ///  def is_known_nonstandard_virtualizable(self, box):
    ///      return self._check_flag(box, HF_NONSTD_VABLE) or self._check_flag(box, HF_SEEN_ALLOCATION)
    /// ```
    pub fn is_known_nonstandard_virtualizable(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> bool {
        self._check_flag(opref, HeapFlags::NONSTD_VABLE, boxes)
            || self._check_flag(opref, HeapFlags::SEEN_ALLOCATION, boxes)
    }

    /// heapcache.py nonstandard_virtualizables_now_known
    ///
    /// ```text
    ///  def nonstandard_virtualizables_now_known(self, box):
    ///      if isinstance(box, Const):
    ///          return
    ///      self._set_flag(box, HF_NONSTD_VABLE)
    /// ```
    pub fn nonstandard_virtualizables_now_known(
        &mut self,
        opref: OpRef,
        boxes: &mut dyn HeapcBoxes,
    ) {
        if opref.is_constant() {
            return;
        }
        self._set_flag(opref, HeapFlags::NONSTD_VABLE, boxes);
    }

    /// heapcache.py `replace_box(oldbox, newbox)`.
    ///
    /// ```text
    ///  def replace_box(self, oldbox, newbox):
    ///      # here, only for replacing a box with a const
    ///      if isinstance(oldbox, FrontendOp) and isinstance(newbox, Const):
    ///          assert newbox.same_constant(constant_from_op(oldbox))
    ///          oldbox.set_replaced_with_const()
    /// ```
    pub fn replace_box(&mut self, old: OpRef, new: OpRef, boxes: &mut dyn HeapcBoxes) {
        if !old.is_constant()
            && new.is_constant()
            && let Some(rec) = boxes.heapc_mut(old)
        {
            rec.replaced_with_const = true;
        }
    }

    /// heapcache.py class_now_known
    ///
    /// ```text
    ///  def class_now_known(self, box):
    ///      if isinstance(box, Const):
    ///          return
    ///      self._set_flag(box, HF_KNOWN_CLASS | HF_KNOWN_NULLITY)
    /// ```
    pub fn class_now_known(&mut self, opref: OpRef, boxes: &mut dyn HeapcBoxes) {
        if opref.is_constant() {
            return;
        }
        self._set_flag(
            opref,
            HeapFlags::KNOWN_CLASS | HeapFlags::KNOWN_NULLITY,
            boxes,
        );
    }

    /// heapcache.py is_class_known.
    ///   `return self._check_flag(box, HF_KNOWN_CLASS)`
    /// Version-gated through `_check_flag` so a `reset_keep_likely_virtuals`
    /// (which only bumps `head_version`) hides stale class info.
    pub fn is_class_known(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> bool {
        self._check_flag(opref, HeapFlags::KNOWN_CLASS, boxes)
    }

    /// Walk every cached *value* slot so a `ConstPtr` ref survives a moving
    /// minor collection. history.py `ConstPtr.value` is a gcref field
    /// the Python GC traces through the box object graph; pyre stores cached
    /// values as flat [`OpRef`] slots and forwards inline `ConstPtr` values in
    /// place. Every other `OpRef` kind is a no-op.
    ///
    /// Only value slots are walked — these are returned on cache hits and
    /// emitted into the op-graph (`loopinvariant_result` / `CacheEntry`
    /// field values), so a stale one is a use-after-move. The
    /// `cache_anything` / `cache_seen_allocation` / `quasiimmut_seen_refs`
    /// *keys* are deliberately left stale: a forwarded lookup key simply
    /// misses the stale-keyed entry and the cache repopulates (same contract
    /// as the `call_pure_results` cache), and an in-place key rewrite would
    /// break the sorted-`VecMap` ordering.
    ///
    /// `FO_REPLACED_WITH_CONST` and `_heapc_deps` live on the FrontendOp
    /// record; the recorder walks those ConstPtrs.
    pub fn for_each_cached_box(&self, mut visit: impl FnMut(OpRef)) {
        fn visit_entry(entry: &CacheEntry, visit: &mut impl FnMut(OpRef)) {
            for &value in entry.cache_anything.values() {
                visit(value);
            }
            for &value in entry.cache_seen_allocation.values() {
                visit(value);
            }
            if let Some(slot) = entry.last_const_box {
                visit(slot);
            }
        }
        for entry in self.heap_cache.values() {
            visit_entry(entry, &mut visit);
        }
        for index_map in self.heap_array_cache.values() {
            for entry in index_map.values() {
                visit_entry(entry, &mut visit);
            }
        }
        if let Some(slot) = self.loopinvariant_result {
            visit(slot);
        }
    }

    pub fn walk_const_ptr_refs(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        fn forward(slot: &mut OpRef, visitor: &mut dyn FnMut(&mut GcRef)) {
            if let OpRef::ConstPtr(gcref) = slot {
                visitor(gcref);
            }
        }
        fn forward_entry(entry: &mut CacheEntry, visitor: &mut dyn FnMut(&mut GcRef)) {
            for value in entry.cache_anything.values_mut() {
                forward(value, visitor);
            }
            for value in entry.cache_seen_allocation.values_mut() {
                forward(value, visitor);
            }
            if let Some(slot) = entry.last_const_box.as_mut() {
                forward(slot, visitor);
            }
        }
        for entry in self.heap_cache.values_mut() {
            forward_entry(entry, visitor);
        }
        for index_map in self.heap_array_cache.values_mut() {
            for entry in index_map.values_mut() {
                forward_entry(entry, visitor);
            }
        }
        if let Some(slot) = self.loopinvariant_result.as_mut() {
            forward(slot, visitor);
        }
    }

    /// heapcache.py is_unescaped.
    ///   `return self._check_flag(box, HF_IS_UNESCAPED)`
    pub fn is_unescaped(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> bool {
        self._check_flag(opref, HeapFlags::IS_UNESCAPED, boxes)
    }

    /// heapcache.py `CacheEntry._seen_alloc(box)`:
    ///
    /// ```text
    ///  if not isinstance(ref_box, RefFrontendOp):
    ///      return False
    ///  return self.heapcache._check_flag(ref_box, HF_SEEN_ALLOCATION)
    /// ```
    pub fn saw_allocation(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> bool {
        self._check_flag(opref, HeapFlags::SEEN_ALLOCATION, boxes)
    }

    /// Notify the cache about an operation, potentially invalidating entries.
    ///
    /// This should be called for every operation during tracing, so the cache
    /// can track which operations affect heap state.
    pub fn notify_op(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        result: OpRef,
        boxes: &mut dyn HeapcBoxes,
    ) {
        if opcode.is_malloc() {
            self.new_object(result, boxes);
            return;
        }
        // heapcache.py `mark_escaped` routes SETFIELD_GC /
        // SETARRAYITEM_GC through the single `_escape_from_write(box,
        // fieldbox)` body. SETFIELD_GC: box=args[0], fieldbox=args[1];
        // SETARRAYITEM_GC: box=args[0], fieldbox=args[2]. The dependency
        // is recorded only when both are unescaped; in every other case
        // — including container unescaped but value already escaped —
        // the value escapes (heapcache.py `elif fieldbox is not
        // None: self._escape_box(fieldbox)`).
        if opcode == OpCode::SetfieldGc && args.len() >= 2 {
            self._escape_from_write(args[0], args[1], boxes);
        }
        if opcode == OpCode::SetarrayitemGc && args.len() >= 3 {
            self._escape_from_write(args[0], args[2], boxes);
        }
        // heapcache.py: GUARD_VALUE → known constant + nonnull.
        if opcode == OpCode::GuardValue && args.len() >= 2 {
            self.nullity_now_known(args[0], boxes);
        }
        // heapcache.py `class_now_known(box)` sets HF_KNOWN_CLASS
        // on args[0]. The class pointer is the box's own typeptr
        // (`pyjitpl.py` `opimpl_guard_class` / `cls_of_box`).
        if opcode == OpCode::GuardClass || opcode == OpCode::GuardNonnullClass {
            self.class_now_known(args[0], boxes);
        }
        // heapcache.py: GUARD_NONNULL → known non-null.
        if opcode == OpCode::GuardNonnull && !args.is_empty() {
            self.nullity_now_known(args[0], boxes);
        }

        // heapcache.py: mark_escaped — escape arguments for
        // operations that are NOT in the whitelist.
        // GETFIELD_GC_*, PTR_EQ/NE, INSTANCE_PTR_EQ/NE, ASSERT_NOT_NONE
        // do NOT escape their arguments. SETFIELD_GC/SETARRAYITEM_GC are
        // handled above via _escape_from_write. Everything else escapes.
        let dont_escape = matches!(
            opcode,
            OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::PtrEq
                | OpCode::PtrNe
                | OpCode::InstancePtrEq
                | OpCode::InstancePtrNe
                | OpCode::AssertNotNone
                | OpCode::SetfieldGc
                | OpCode::SetarrayitemGc
        ) || opcode.is_guard()
            || opcode.is_malloc()
            || opcode.has_no_side_effect();

        if !dont_escape {
            for &arg in args {
                self._escape_box(arg, boxes);
            }
        }
    }

    /// heapcache.py invalidate_caches_varargs.
    ///
    /// `effectinfo` mirrors upstream `descr.get_extra_info()` consulted
    /// inside `clear_caches_varargs`; pyre threads the
    /// already-extracted EffectInfo through to avoid an extra
    /// `&dyn CallDescr` pass.
    pub fn invalidate_caches_varargs<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
        boxes: &mut dyn HeapcBoxes,
    ) {
        self.mark_escaped_varargs(opnum, effectinfo, argboxes, &const_value, boxes);
        if Self::_clear_caches_not_necessary(opnum) {
            return;
        }
        self.clear_caches_varargs(opnum, effectinfo, argboxes, oracle, const_value, boxes);
    }

    /// heapcache.py clear_caches_not_necessary
    ///
    /// ```text
    ///  def clear_caches_not_necessary(self, opnum, descr):
    ///      if (opnum == rop.SETFIELD_GC or
    ///          opnum == rop.SETARRAYITEM_GC or
    ///          opnum == rop.SETFIELD_RAW or
    ///          opnum == rop.SETARRAYITEM_RAW or
    ///          opnum == rop.SETINTERIORFIELD_GC or
    ///          opnum == rop.COPYSTRCONTENT or
    ///          opnum == rop.COPYUNICODECONTENT or
    ///          opnum == rop.STRSETITEM or
    ///          opnum == rop.UNICODESETITEM or
    ///          opnum == rop.SETFIELD_RAW or
    ///          opnum == rop.SETARRAYITEM_RAW or
    ///          opnum == rop.SETINTERIORFIELD_RAW or
    ///          opnum == rop.RECORD_EXACT_CLASS or
    ///          opnum == rop.RAW_STORE or
    ///          opnum == rop.ASSERT_NOT_NONE or
    ///          opnum == rop.RECORD_EXACT_CLASS or
    ///          opnum == rop.RECORD_EXACT_VALUE_I or
    ///          opnum == rop.RECORD_EXACT_VALUE_R):
    ///          return True
    ///      if (rop._OVF_FIRST <= opnum <= rop._OVF_LAST or
    ///          rop._NOSIDEEFFECT_FIRST <= opnum <= rop._NOSIDEEFFECT_LAST or
    ///          rop._GUARD_FIRST <= opnum <= rop._GUARD_LAST):
    ///          return True
    ///      return False
    /// ```
    ///
    /// CALL_* opcodes are deliberately NOT in this set — RPython invalidates
    /// caches whenever a residual call runs, since the callee could mutate
    /// fields the optimizer thinks are still cached.
    fn _clear_caches_not_necessary(opnum: OpCode) -> bool {
        matches!(
            opnum,
            OpCode::SetfieldGc
                | OpCode::SetarrayitemGc
                | OpCode::SetfieldRaw
                | OpCode::SetarrayitemRaw
                | OpCode::SetinteriorfieldGc
                | OpCode::SetinteriorfieldRaw
                | OpCode::Copystrcontent
                | OpCode::Copyunicodecontent
                | OpCode::Strsetitem
                | OpCode::Unicodesetitem
                | OpCode::RecordExactClass
                | OpCode::RecordExactValueR
                | OpCode::RecordExactValueI
                | OpCode::RawStore
                | OpCode::AssertNotNone
        ) || opnum.is_ovf()
            || opnum.has_no_side_effect()
            || opnum.is_guard()
    }

    /// RPython-compatible alias.
    pub fn clear_caches_not_necessary(&self, opnum: OpCode) -> bool {
        Self::_clear_caches_not_necessary(opnum)
    }

    /// RPython-compatible alias.
    pub fn clear_caches<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
        boxes: &mut dyn HeapcBoxes,
    ) {
        self.clear_caches_varargs(opnum, effectinfo, argboxes, oracle, const_value, boxes)
    }

    /// heapcache.py clear_caches_varargs.
    pub fn clear_caches_varargs<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
        boxes: &mut dyn HeapcBoxes,
    ) {
        self.need_guard_not_invalidated = true;
        // RPython `heapcache.py clear_caches_varargs`:
        //     if (OpHelpers.is_plain_call(opnum) or
        //         OpHelpers.is_call_loopinvariant(opnum) or
        //         OpHelpers.is_cond_call_value(opnum) or
        //         opnum == rop.COND_CALL):
        // The narrow arm below takes exactly that enumeration: `CALL_{I,R,F,N}`
        // (`is_plain_call`), `CALL_LOOPINVARIANT_*`, `COND_CALL_VALUE_*` and
        // `COND_CALL`.  `CALL_MAY_FORCE_*`, `CALL_ASSEMBLER_*` and
        // `CALL_RELEASE_GIL_*` are outside it and fall through to
        // `reset_keep_likely_virtuals` (the aggressive arm).  Pyre's
        // `is_call()` is the broader `_CALL_FIRST..=_CALL_LAST` range, so the
        // narrow `is_plain_call()` predicate mirrors upstream's enumeration.
        //
        // `CALL_PURE_*` is in the narrow arm here because pyre records the
        // pure opcode up front: `select_residual_call_opcode`
        // (pyre-jit-trace) picks `CallPure*` before
        // `heapcache_invalidate_caches_varargs` runs.  Upstream's
        // `MIFrame.execute_varargs` (pyjitpl.py) instead records the plain
        // `CALL_*` through `execute_and_record_varargs(rop.CALL_*)` -- which
        // is what runs `invalidate_caches` -- and patches the opcode to
        // `CALL_PURE_*` afterwards in `record_result_of_call_pure`.  The same
        // residual therefore reaches this gate spelled differently on the two
        // sides, and gating on `is_plain_call` alone would drop an
        // `EF_ELIDABLE_CANNOT_RAISE` residual onto the blanket
        // `reset_keep_likely_virtuals`, which bumps `head_version` and voids
        // every box's class and nullity knowledge mid-trace.
        if opnum.is_plain_call()
            || opnum.is_call_pure()
            || opnum.is_call_loopinvariant()
            || opnum.is_cond_call_value()
            || opnum == OpCode::CondCallN
        {
            if let Some(ei) = effectinfo {
                // heapcache.py:347-353 — elidable / loopinvariant calls
                // are pure (or already cached) and never invalidate the
                // heap.
                if matches!(
                    ei.extraeffect,
                    ExtraEffect::LoopInvariant
                        | ExtraEffect::ElidableCannotRaise
                        | ExtraEffect::ElidableOrMemoryError
                        | ExtraEffect::ElidableCanRaise,
                ) {
                    return;
                }
                // heapcache.py:355-361 — well-defined oopspec dispatch.
                let single_descr_idx = ei.single_write_descr_array.as_ref().map(|d| d.index());
                if ei.oopspecindex == majit_ir::OopSpecIndex::Arraycopy {
                    self._clear_caches_arraycopy(
                        opnum,
                        None,
                        argboxes,
                        single_descr_idx,
                        oracle,
                        const_value,
                        boxes,
                    );
                    return;
                }
                if ei.oopspecindex == majit_ir::OopSpecIndex::Arraymove {
                    self._clear_caches_arraymove(
                        opnum,
                        None,
                        argboxes,
                        single_descr_idx,
                        oracle,
                        const_value,
                        boxes,
                    );
                    return;
                }
            }
            // heapcache.py:362-369 — only invalidate things that escaped.
            // Take/restore mirrors `CacheEntry.heapcache` back-reference so
            // `invalidate_unescaped` calls the version-gated
            // `HeapCache.is_unescaped(ref_box)` per entry (heapcache.py /
            // 457-460) rather than reading any pre-snapshotted bit table.
            let mut heap_cache = std::mem::take(&mut self.heap_cache);
            for cache in heap_cache.values_mut() {
                cache.invalidate_unescaped(self, boxes);
            }
            self.heap_cache = heap_cache;
            let mut heap_array_cache = std::mem::take(&mut self.heap_array_cache);
            for caches in heap_array_cache.values_mut() {
                for cache in caches.values_mut() {
                    cache.invalidate_unescaped(self, boxes);
                }
            }
            self.heap_array_cache = heap_array_cache;
            return;
        }
        // heapcache.py — fallback: reset state for non-CALL ops
        // (release-GIL etc.) that we can't selectively invalidate.
        self.reset_keep_likely_virtuals();
    }

    /// Parity alias for RPython cache invalidation entrypoint.
    pub fn invalidate_caches<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
        boxes: &mut dyn HeapcBoxes,
    ) {
        self.mark_escaped(opnum, None, argboxes, boxes);
        if Self::_clear_caches_not_necessary(opnum) {
            return;
        }
        // `invalidate_caches_varargs` will re-issue `mark_escaped_varargs`
        // (matching upstream's double-call shape at heapcache.py);
        // do NOT also call `mark_escaped` for non-CALL_N argboxes here —
        // upstream `invalidate_caches` (heapcache.py) ONLY does
        // the `mark_escaped` for the SETFIELD/SETARRAYITEM special cases
        // that `mark_escaped_varargs` would skip.  The 1:1 split is kept
        // by `mark_escaped`'s opnum filter.
        self.invalidate_caches_varargs(opnum, effectinfo, argboxes, oracle, const_value, boxes);
    }

    /// heapcache.py _clear_caches_arraycopy
    ///
    /// ```text
    ///  def _clear_caches_arraycopy(self, opnum, descr, argboxes, effectinfo):
    ///      self._clear_caches_arrayop(argboxes[1], argboxes[2],
    ///                                 argboxes[3], argboxes[4], argboxes[5],
    ///                                 effectinfo)
    /// ```
    pub fn _clear_caches_arraycopy<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        _opnum: OpCode,
        _descr: Option<&EffectInfo>,
        argboxes: &[OpRef],
        single_write_descr_array: Option<u32>,
        oracle: &dyn SameConstantOracle,
        const_value: F,
        boxes: &dyn HeapcBoxes,
    ) {
        // argboxes layout from RPython oopspec ll_arraycopy:
        //   [func, src, dst, srcstart, dststart, length]
        if argboxes.len() < 6 {
            self.reset_keep_likely_virtuals();
            return;
        }
        self._clear_caches_arrayop(
            argboxes[1],
            argboxes[2],
            argboxes[3],
            argboxes[4],
            argboxes[5],
            single_write_descr_array,
            oracle,
            const_value,
            boxes,
        );
    }

    /// heapcache.py _clear_caches_arraymove
    ///
    /// ```text
    ///  def _clear_caches_arraymove(self, opnum, descr, argboxes, effectinfo):
    ///      self._clear_caches_arrayop(argboxes[1], argboxes[1],
    ///                                 argboxes[2], argboxes[3], argboxes[4],
    ///                                 effectinfo)
    /// ```
    pub fn _clear_caches_arraymove<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        _opnum: OpCode,
        _descr: Option<&EffectInfo>,
        argboxes: &[OpRef],
        single_write_descr_array: Option<u32>,
        oracle: &dyn SameConstantOracle,
        const_value: F,
        boxes: &dyn HeapcBoxes,
    ) {
        // argboxes layout from RPython oopspec ll_arraymove:
        //   [func, arr, srcstart, dststart, length]
        if argboxes.len() < 5 {
            self.reset_keep_likely_virtuals();
            return;
        }
        self._clear_caches_arrayop(
            argboxes[1],
            argboxes[1],
            argboxes[2],
            argboxes[3],
            argboxes[4],
            single_write_descr_array,
            oracle,
            const_value,
            boxes,
        );
    }

    /// heapcache.py _clear_caches_arrayop
    ///
    /// ```text
    ///  def _clear_caches_arrayop(self, source_box, dest_box,
    ///                            source_start_box, dest_start_box, length_box,
    ///                            effectinfo):
    ///      seen_allocation_of_target = self._check_flag(dest_box,
    ///                                                   HF_SEEN_ALLOCATION)
    ///      if (isinstance(source_start_box, ConstInt) and
    ///          isinstance(dest_start_box, ConstInt) and
    ///          isinstance(length_box, ConstInt) and
    ///          effectinfo.single_write_descr_array is not None):
    ///          ...per-index copy from source to dest...
    ///          return
    ///      elif effectinfo.single_write_descr_array is not None:
    ///          ...wholesale clear of dest descr submap...
    ///          return
    ///      self.reset_keep_likely_virtuals()
    /// ```
    ///
    /// `const_value` resolves a constant-namespace OpRef to its raw `i64`.
    /// RPython reads `box.getint()` directly from the ConstInt; majit needs
    /// a callback because HeapCache has no constant pool of its own.
    pub fn _clear_caches_arrayop_with_consts(
        &mut self,
        source_box: OpRef,
        dest_box: OpRef,
        source_start_box: OpRef,
        dest_start_box: OpRef,
        length_box: OpRef,
        single_write_descr_array: Option<u32>,
        const_value: impl Fn(OpRef) -> Option<i64>,
        oracle: &dyn SameConstantOracle,
        boxes: &dyn HeapcBoxes,
    ) {
        let seen_allocation_of_target = self.saw_allocation(dest_box, boxes);
        let seen_allocation_of_source = self.saw_allocation(source_box, boxes);
        let srcstart = const_value(source_start_box);
        let dststart = const_value(dest_start_box);
        let length = const_value(length_box);
        if let (Some(srcstart), Some(dststart), Some(length), Some(descr)) =
            (srcstart, dststart, length, single_write_descr_array)
        {
            // heapcache.py:405-411: pick iteration direction.
            // ARRAYMOVE with srcstart < dststart needs reverse-order to
            // avoid clobbering values it still needs to read.
            let (mut index_current, index_delta, index_stop): (i64, i64, i64) =
                if srcstart < dststart {
                    (length - 1, -1, -1)
                } else {
                    (0, 1, length)
                };
            while index_current != index_stop {
                let i = index_current;
                index_current += index_delta;
                debug_assert!(i >= 0);
                // heapcache.py — `indexcache.read(source_box)`.
                // The cache entry's `_unique_const_heuristic` canonicalises
                // the ConstPtr source so two distinct OpRefs for the same
                // gcref share the same dict slot.
                let raw_value = self
                    .heap_array_cache
                    .get_mut(&descr)
                    .and_then(|m| m.get_mut(&(srcstart + i)))
                    .and_then(|entry| {
                        let src = entry._unique_const_heuristic(source_box, oracle);
                        let dict = entry._getdict(seen_allocation_of_source);
                        dict.get(&src).cloned()
                    });
                // heapcache.py `return maybe_replace_with_const(res_box)`
                // — follow the FO_REPLACED_WITH_CONST forwarding so callers
                // see the canonical const replacement, not the stale Box.
                // The Box identity is the OpRef; its intrinsic `value`
                // travels with the frontend value slot, so the copy needs no
                // explicit payload handling.
                let value =
                    raw_value.map(|fieldbox| self.maybe_replace_with_const(fieldbox, boxes));
                // heapcache.py:423-429: ...and write it to the dest cell.
                if let Some(value) = value {
                    let dst_index = dststart + i;
                    let entry = self
                        .heap_array_cache
                        .entry(descr)
                        .or_default()
                        .entry(dst_index)
                        .or_default();
                    // heapcache.py `do_write_with_aliasing` —
                    // canonicalise dest, then `_clear_cache_on_write(seen_alloc)`
                    // BEFORE the insert so aliasing entries from prior
                    // writes get dropped (escaped target → wipe whole
                    // cache_anything; unescaped → only cache_anything).
                    let dst = entry._unique_const_heuristic(dest_box, oracle);
                    entry._clear_cache_on_write(seen_allocation_of_target);
                    entry
                        ._getdict_mut(seen_allocation_of_target)
                        .insert(dst, value);
                } else {
                    // heapcache.py:430-436: source had no cached value, so
                    // the dest's existing entry must be invalidated.
                    if let Some(idx_cache) = self
                        .heap_array_cache
                        .get_mut(&descr)
                        .and_then(|m| m.get_mut(&(dststart + i)))
                    {
                        idx_cache._clear_cache_on_write(seen_allocation_of_target);
                    }
                }
            }
            return;
        }
        // heapcache.py:438-446: known descr but non-constant indexes — clear
        // the entire dest descr submap.
        if let Some(descr) = single_write_descr_array {
            if let Some(submap) = self.heap_array_cache.get_mut(&descr) {
                for entry in submap.values_mut() {
                    entry._clear_cache_on_write(seen_allocation_of_target);
                }
            }
            return;
        }
        // heapcache.py:447: total fallback.
        self.reset_keep_likely_virtuals();
    }

    /// `_clear_caches_arrayop` accepts a const-resolution closure so
    /// production callers from `invalidate_caches_varargs` reach the
    /// per-index copy branch of `_clear_caches_arrayop_with_consts`
    /// (heapcache.py).  When the closure returns `None` for any
    /// index/length operand, the branch falls through to whole-descr
    /// clearing as upstream does (heapcache.py).
    pub fn _clear_caches_arrayop<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        source_box: OpRef,
        dest_box: OpRef,
        source_start_box: OpRef,
        dest_start_box: OpRef,
        length_box: OpRef,
        single_write_descr_array: Option<u32>,
        oracle: &dyn SameConstantOracle,
        const_value: F,
        boxes: &dyn HeapcBoxes,
    ) {
        self._clear_caches_arrayop_with_consts(
            source_box,
            dest_box,
            source_start_box,
            dest_start_box,
            length_box,
            single_write_descr_array,
            const_value,
            oracle,
            boxes,
        );
    }

    /// Alias kept for parity with older callsites.
    pub fn invalidate_caches_varargs_alias<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
        boxes: &mut dyn HeapcBoxes,
    ) {
        self.invalidate_caches_varargs(opnum, effectinfo, argboxes, oracle, const_value, boxes)
    }

    /// heapcache.py `get_field_updater(self, box, descr)`.
    ///
    /// ```text
    ///  def get_field_updater(self, box, descr):
    ///      cache = self.heap_cache.get(descr, None)
    ///      if cache is None:
    ///          cache = self.heap_cache[descr] = CacheEntry(self)
    ///          fieldbox = None
    ///      else:
    ///          fieldbox = cache.read(box)
    ///      return FieldUpdater(box, cache, fieldbox)
    /// ```
    ///
    /// `cache.read` (heapcache.py) handles the
    /// `_unique_const_heuristic` ConstPtr canonicalisation and the
    /// `maybe_replace_with_const` forwarding internally.  Need an
    /// `oracle` parameter because pyre's `same_constant` lives on the
    /// `ConstOprefOracle` (inline Const OpRef value compare) rather than
    /// on the OpRef itself.
    pub fn get_field_updater<'a>(
        &'a mut self,
        obj: OpRef,
        descr_index: u32,
        oracle: &dyn SameConstantOracle,
        boxes: &'a mut dyn HeapcBoxes,
    ) -> FieldUpdater<'a> {
        let fieldbox = if let Some(mut entry) = self.heap_cache.remove(&descr_index) {
            let result = entry.read(obj, self, boxes, oracle);
            self.heap_cache.insert(descr_index, entry);
            result
        } else {
            // heapcache.py `cache = self.heap_cache[descr] = CacheEntry(self)`.
            self.heap_cache.insert(descr_index, CacheEntry::new());
            None
        };
        FieldUpdater::with_cache(obj, self, boxes, descr_index, fieldbox)
    }

    // ── Array item caching (RPython heapcache.py cached_arrayitems) ──

    /// heapcache.py `getarrayitem(self, box, indexbox, descr)`.
    /// The caller supplies the index as the raw `i64` value extracted
    /// via `ConstInt.getint()`; non-ConstInt inputs short-circuit to
    /// `None` at the caller boundary so the cache key is always the
    /// upstream-equivalent value, not the index Box's identity.
    ///
    /// `array` is routed through the indexcache's `_unique_const_heuristic`
    /// (heapcache.py `indexcache.read(box)`) so two distinct
    /// ConstPtr OpRefs for the same gcref hit the same cache slot.
    pub fn getarrayitem_cache(
        &mut self,
        array: OpRef,
        index_value: i64,
        descr: u32,
        oracle: &dyn SameConstantOracle,
        boxes: &dyn HeapcBoxes,
    ) -> Option<OpRef> {
        let entry = self
            .heap_array_cache
            .get_mut(&descr)?
            .get_mut(&index_value)?;
        let array = entry._unique_const_heuristic(array, oracle);
        let seen_alloc = self.saw_allocation(array, boxes);
        let entry = self.heap_array_cache.get(&descr)?.get(&index_value)?;
        let cached = entry._getdict(seen_alloc).get(&array).cloned()?;
        Some(self.maybe_replace_with_const(cached, boxes))
    }

    /// heapcache.py `setarrayitem`. Non-ConstInt index (`None`
    /// here) clears the whole descr submap; otherwise the cache entry
    /// for `(descr, index_value)` writes through
    /// `do_write_with_aliasing` which canonicalises `array` via
    /// `_unique_const_heuristic` before keying.
    pub fn setarrayitem_cache(
        &mut self,
        array: OpRef,
        index_value: Option<i64>,
        descr: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
        boxes: &dyn HeapcBoxes,
    ) {
        let Some(index_value) = index_value else {
            if let Some(cache) = self.heap_array_cache.get_mut(&descr) {
                cache.clear();
            }
            return;
        };
        let seen_alloc = self.saw_allocation(array, boxes);
        let entry = self
            .heap_array_cache
            .entry(descr)
            .or_default()
            .entry(index_value)
            .or_default();
        // CacheEntry.do_write_with_aliasing internally canonicalises
        // ConstPtr operands via `_unique_const_heuristic`, replicating
        // heapcache.py `indexcache.do_write_with_aliasing(box, ...)`.
        let array = entry._unique_const_heuristic(array, oracle);
        entry._clear_cache_on_write(seen_alloc);
        entry._getdict_mut(seen_alloc).insert(array, value);
    }

    /// heapcache.py `getarrayitem_now_known`. Same canonical
    /// keying as `setarrayitem_cache` but without the alias clearing.
    pub fn getarrayitem_now_known(
        &mut self,
        array: OpRef,
        index_value: Option<i64>,
        descr: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
        boxes: &dyn HeapcBoxes,
    ) {
        let Some(index_value) = index_value else {
            return;
        };
        let seen_alloc = self.saw_allocation(array, boxes);
        let entry = self
            .heap_array_cache
            .entry(descr)
            .or_default()
            .entry(index_value)
            .or_default();
        let array = entry._unique_const_heuristic(array, oracle);
        entry._getdict_mut(seen_alloc).insert(array, value);
    }

    /// Invalidate array caches for a specific array across every descr/index.
    pub fn invalidate_array_cache(&mut self, array: OpRef) {
        for cache in self.heap_array_cache.values_mut() {
            for entry in cache.values_mut() {
                entry.cache_anything.remove(&array);
                entry.cache_seen_allocation.remove(&array);
            }
        }
    }

    // ── Quasi-immutable tracking (heapcache.py is_quasi_immut_known) ──

    /// The `quasiimmut_seen_refs` key: `box.getref_base()`
    /// (heapcache.py:609/622).  Upstream reads the raw GC pointer off the
    /// `ConstPtr` box; pyre's constant `OpRef` carries that pointer inline
    /// (`OpRef::const_ptr(GcRef)`), so `inline_const_bits` is the same read.
    fn quasiimmut_seen_ref_key(obj: OpRef) -> usize {
        obj.inline_const_bits().unwrap_or(0) as usize
    }

    /// heapcache.py is_quasi_immut_known
    ///
    /// ```text
    ///  def is_quasi_immut_known(self, fielddescr, box):
    ///      cache = self.heap_cache.get(fielddescr, None)
    ///      if cache is not None:
    ///          if isinstance(box, Const):
    ///              if cache.quasiimmut_seen_refs is not None:
    ///                  return box.getref_base() in cache.quasiimmut_seen_refs
    ///          else:
    ///              if cache.quasiimmut_seen is not None:
    ///                  return box in cache.quasiimmut_seen
    ///      return False
    /// ```
    ///
    /// The two sets live on the per-descr [`CacheEntry`], so
    /// `_clear_cache_on_write` and `_invalidate_unescaped` clear them
    /// alongside the value caches.  That lifetime is load-bearing: a
    /// residual call goes through `clear_caches_varargs`
    /// (heapcache.py), which both arms
    /// `need_guard_not_invalidated` and drops the "already marked" bit, so
    /// the next read of the field re-emits `QUASIIMMUT_FIELD` and that op
    /// in turn emits the second `GUARD_NOT_INVALIDATED`.
    pub fn is_quasi_immut_known(&self, field_index: u32, obj: OpRef) -> bool {
        let Some(cache) = self.heap_cache.get(&field_index) else {
            return false;
        };
        if obj.is_constant() {
            if let Some(seen) = &cache.quasiimmut_seen_refs {
                return seen.contains(&Self::quasiimmut_seen_ref_key(obj));
            }
        } else if let Some(seen) = &cache.quasiimmut_seen {
            return seen.contains(&obj);
        }
        false
    }

    /// heapcache.py quasi_immut_now_known
    ///
    /// ```text
    ///  def quasi_immut_now_known(self, fielddescr, box):
    ///      cache = self.heap_cache.get(fielddescr, None)
    ///      if cache is None:
    ///          cache = self.heap_cache[fielddescr] = CacheEntry(self)
    ///      if isinstance(box, Const):
    ///          if cache.quasiimmut_seen_refs is None:
    ///              cache.quasiimmut_seen_refs = new_ref_dict()
    ///          cache.quasiimmut_seen_refs[box.getref_base()] = None
    ///      else:
    ///          if cache.quasiimmut_seen is not None:
    ///              cache.quasiimmut_seen[box] = None
    ///          else:
    ///              cache.quasiimmut_seen = {box: None}
    /// ```
    pub fn quasi_immut_now_known(&mut self, field_index: u32, obj: OpRef) {
        let cache = self.heap_cache.entry(field_index).or_default();
        if obj.is_constant() {
            cache
                .quasiimmut_seen_refs
                .get_or_insert_with(IndexSet::new)
                .insert(Self::quasiimmut_seen_ref_key(obj));
        } else {
            cache
                .quasiimmut_seen
                .get_or_insert_with(IndexSet::new)
                .insert(obj);
        }
    }

    // ── Nullity tracking (heapcache.py nullity_now_known / is_nullity_known) ──

    /// heapcache.py nullity_now_known
    ///
    /// ```text
    ///  def nullity_now_known(self, box):
    ///      if isinstance(box, Const):
    ///          return
    ///      self._set_flag(box, HF_KNOWN_NULLITY)
    /// ```
    ///
    /// Which side of the nullity is known is the box's own `getref_base()`
    /// (`pyjitpl.py` `opimpl_goto_if_not_ptr_nonzero`).
    pub fn nullity_now_known(&mut self, opref: OpRef, boxes: &mut dyn HeapcBoxes) {
        if opref.is_constant() {
            return;
        }
        self._set_flag(opref, HeapFlags::KNOWN_NULLITY, boxes);
    }

    /// heapcache.py: is_nullity_known(box)
    ///   if isinstance(box, Const): return bool(box.getref_base())
    ///   return self._check_flag(box, HF_KNOWN_NULLITY)
    pub fn is_nullity_known(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> bool {
        if opref.is_constant() {
            // heapcache.py:477: return bool(box.getref_base())
            return match boxes
                .box_value(opref)
                .or_else(|| opref.inline_const_to_value())
            {
                Some(Value::Ref(g)) => g.0 != 0,
                Some(Value::Int(n)) => n != 0,
                _ => false,
            };
        }
        self._check_flag(opref, HeapFlags::KNOWN_NULLITY, boxes)
    }

    // ── Array length caching (heapcache.py arraylen_now_known / arraylen) ──

    /// heapcache.py arraylen
    ///
    /// ```text
    ///  def arraylen(self, box):
    ///      if (isinstance(box, RefFrontendOp) and
    ///          self.test_head_version(box) and
    ///          box._heapc_deps is not None):
    ///          res_box = box._heapc_deps[0]
    ///          if res_box is not None:
    ///              return maybe_replace_with_const(res_box)
    ///      return None
    /// ```
    ///
    pub fn arraylen(&self, array: OpRef, boxes: &dyn HeapcBoxes) -> Option<OpRef> {
        if array.is_constant() || !self.test_head_version(array, boxes) {
            return None;
        }
        boxes
            .heapc(array)
            .and_then(|rec| rec.deps.as_ref())
            .and_then(|deps| deps.first().cloned().flatten())
            .map(|length| self.maybe_replace_with_const(length, boxes))
    }

    /// heapcache.py arraylen_now_known
    ///
    /// ```text
    ///  def arraylen_now_known(self, box, lengthbox):
    ///      # we store in '_heapc_deps' a list of boxes: the *first* box
    ///      # is the known length or None, and the remaining boxes are
    ///      # the regular dependencies.
    ///      if isinstance(box, Const):
    ///          return
    ///      deps = self._get_deps(box)
    ///      assert deps is not None
    ///      deps[0] = lengthbox
    /// ```
    ///
    /// `_get_deps` runs `update_version` as a side effect and ensures the
    /// `_heapc_deps` list exists with slot 0 reserved for the array length.
    pub fn arraylen_now_known(&mut self, array: OpRef, length: OpRef, boxes: &mut dyn HeapcBoxes) {
        if array.is_constant() {
            return;
        }
        let deps = self
            ._get_deps(array, boxes)
            .expect("assert deps is not None — the Const arm above returned");
        deps[0] = Some(length);
    }

    // ── Likely virtual tracking (heapcache.py is_likely_virtual) ──

    /// Alias for `new_object` kept under the heapcache.py name `new`.
    /// Used by `opimpl_virtual_ref` (pyjitpl.py) which calls
    /// `self.metainterp.heapcache.new(resbox)` after recording VIRTUAL_REF.
    pub fn new_box(&mut self, opref: OpRef, boxes: &mut dyn HeapcBoxes) {
        self.new_object(opref, boxes);
    }

    /// heapcache.py is_likely_virtual.
    ///   `return (... self.test_likely_virtual_version(box) and
    ///            test_flags(box, HF_LIKELY_VIRTUAL))`
    ///
    /// Note: gates on `test_likely_virtual_version` (NOT
    /// `test_head_version`) so a `reset_keep_likely_virtuals` does not
    /// invalidate this flag — the older box is still trusted as likely
    /// virtual until the *next* version bump.
    pub fn is_likely_virtual(&self, opref: OpRef, boxes: &dyn HeapcBoxes) -> bool {
        if !self.test_likely_virtual_version(opref, boxes) {
            return false;
        }
        test_flags(boxes, opref, HeapFlags::LIKELY_VIRTUAL)
    }

    // ── Loop-invariant call result caching ──

    /// heapcache.py call_loopinvariant_known_result
    ///
    /// ```text
    ///  def call_loopinvariant_known_result(self, allboxes, descr):
    ///      if self.loop_invariant_descr is not descr:
    ///          return None
    ///      if self.loop_invariant_arg0int != allboxes[0].getint():
    ///          return None
    ///      return self.loop_invariant_result
    /// ```
    ///
    /// Only ONE result is stored at a time. RPython matches by descr
    /// **identity** and the arg0 **integer value**; majit keys both
    /// values directly because the trace HeapCache deals in `descr.index()`
    /// + `i64` rather than Python objects.
    pub fn call_loopinvariant_known_result(
        &self,
        descr_index: u32,
        arg0_int: i64,
    ) -> Option<(OpRef, i64)> {
        if self.loopinvariant_descr != Some(descr_index) {
            return None;
        }
        if self.loopinvariant_arg0 != Some(arg0_int) {
            return None;
        }
        // Pair the cached symbolic OpRef with its cached concrete value
        // so the caller can return the same `(opref, value)` shape it
        // would emit for a fresh call.  See `loopinvariant_resvalue` for
        // the rationale.
        Some((
            self.loopinvariant_result?,
            self.loopinvariant_resvalue.unwrap_or(0),
        ))
    }

    /// heapcache.py call_loopinvariant_now_known
    ///
    /// ```text
    ///  def call_loopinvariant_now_known(self, allboxes, descr, res):
    ///      self.loop_invariant_descr = descr
    ///      self.loop_invariant_arg0int = allboxes[0].getint()
    ///      self.loop_invariant_result = res
    /// ```
    pub fn call_loopinvariant_now_known(
        &mut self,
        descr_index: u32,
        arg0_int: i64,
        result: OpRef,
        resvalue: i64,
    ) {
        self.loopinvariant_descr = Some(descr_index);
        self.loopinvariant_arg0 = Some(arg0_int);
        self.loopinvariant_result = Some(result);
        self.loopinvariant_resvalue = Some(resvalue);
    }

    /// Void overload of `call_loopinvariant_now_known` — `pyjitpl.py`
    /// invokes `heapcache.call_loopinvariant_now_known(allboxes, descr, res)`
    /// for `tp == 'v'` with `res = None` (`_record_helper_varargs` returns
    /// None for void).  Upstream stores `res = None` in the slot, evicting
    /// any prior typed result that shared the (descr, arg0) key.  The Rust
    /// split between symbolic `OpRef` and concrete `i64` requires a separate
    /// entry point; semantics match the upstream `res = None` store.
    pub fn call_loopinvariant_now_known_void(&mut self, descr_index: u32, arg0_int: i64) {
        self.loopinvariant_descr = Some(descr_index);
        self.loopinvariant_arg0 = Some(arg0_int);
        self.loopinvariant_result = None;
        self.loopinvariant_resvalue = None;
    }

    /// Internal alias retained for older callsites.
    pub fn call_loopinvariant_cache(
        &mut self,
        descr_index: u32,
        arg0_int: i64,
        result: OpRef,
        resvalue: i64,
    ) {
        self.call_loopinvariant_now_known(descr_index, arg0_int, result, resvalue);
    }

    /// Internal alias retained for older callsites.
    pub fn call_loopinvariant_lookup(
        &self,
        descr_index: u32,
        arg0_int: i64,
    ) -> Option<(OpRef, i64)> {
        self.call_loopinvariant_known_result(descr_index, arg0_int)
    }

    // ── Reset variants ──

    /// heapcache.py reset
    ///
    /// ```text
    ///  def reset(self):
    ///      # Global reset of all flags. Update both version numbers so
    ///      # that any access to '_heapc_flags' will be marked as outdated.
    ///      assert self.head_version < _HF_VERSION_MAX
    ///      self.head_version += _HF_VERSION_INC
    ///      self.likely_virtual_version = self.head_version
    ///      #
    ///      # heap cache
    ///      self.heap_cache = {}
    ///      self.heap_array_cache = {}
    ///      self.need_guard_not_invalidated = True
    ///      #
    ///      # result of one loop invariant call
    ///      self.loop_invariant_result = None
    ///      self.loop_invariant_descr = None
    ///      self.loop_invariant_arg0int = -1
    /// ```
    ///
    pub fn reset(&mut self) {
        // heapcache.py:166-168: bump head_version, sync likely_virtual_version.
        assert!(self.head_version < HF_VERSION_MAX);
        self.head_version += HF_VERSION_INC;
        self.likely_virtual_version = self.head_version;
        // heapcache.py:172-175: clear heap_cache + heap_array_cache.
        // Replacing `heap_cache = {}` drops every per-descr
        // `CacheEntry`, which in turn drops the per-descr
        // `last_const_box` `_unique_const_heuristic` LRU.
        self.heap_cache.clear();
        self.heap_array_cache.clear();
        // heapcache.py:176: need_guard_not_invalidated = True
        self.need_guard_not_invalidated = true;
        // heapcache.py: loop_invariant_result/descr/arg0int reset.
        self.loopinvariant_descr = None;
        self.loopinvariant_arg0 = None;
        self.loopinvariant_result = None;
        // Per-box `_heapc_flags` / `_heapc_deps` / FO_REPLACED_WITH_CONST live
        // on the FrontendOp record. The version bump marks them outdated;
        // a cut drops records past the cut point together with their flags.
    }

    /// heapcache.py:176: check and consume need_guard_not_invalidated.
    /// Returns true the first time after reset (or after cache clearing).
    pub fn check_and_clear_guard_not_invalidated(&mut self) -> bool {
        let needed = self.need_guard_not_invalidated;
        self.need_guard_not_invalidated = false;
        needed
    }

    /// Whether GUARD_NOT_INVALIDATED is needed.
    pub fn need_guard_not_invalidated(&self) -> bool {
        self.need_guard_not_invalidated
    }

    /// heapcache.py reset_keep_likely_virtuals
    ///
    /// ```text
    ///  def reset_keep_likely_virtuals(self):
    ///      # Update only 'head_version', but 'likely_virtual_version'
    ///      # remains at its older value.
    ///      assert self.head_version < _HF_VERSION_MAX
    ///      self.head_version += _HF_VERSION_INC
    ///      self.heap_cache = {}
    ///      self.heap_array_cache = {}
    /// ```
    ///
    /// `likely_virtual`, `loopinvariant_*`, and `_heapc_deps`
    /// `need_guard_not_invalidated` are intentionally preserved (a residual
    /// call that releases the GIL invalidates heap caches but the JIT can
    /// still trust prior allocation/likely-virtual hints).
    pub fn reset_keep_likely_virtuals(&mut self) {
        assert!(self.head_version < HF_VERSION_MAX);
        self.head_version += HF_VERSION_INC;
        self.heap_cache.clear();
        self.heap_array_cache.clear();
    }

    /// `update_version`'s `ref_frontend_op._heapc_deps = None`, split out.
    ///
    /// A `Const` has no slot to clear, exactly as `flags_for_ref` and
    /// `set_flags_for_ref` already answer 0 and no-op for one. Without this
    /// arm `update_version` is a no-op for a constant right up to its last
    /// line, where `raw()` panics.
    pub fn _remove_deps_for_box(&mut self, opref: OpRef, boxes: &mut dyn HeapcBoxes) {
        if let Some(rec) = boxes.heapc_mut(opref) {
            rec.deps = None;
        }
    }
}

impl Default for HeapCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared view of [`HeapCache`] plus the FrontendOp records it reads.
pub struct HeapCacheView<'a> {
    cache: &'a HeapCache,
    boxes: &'a dyn HeapcBoxes,
}

impl<'a> HeapCacheView<'a> {
    pub fn new(cache: &'a HeapCache, boxes: &'a dyn HeapcBoxes) -> Self {
        Self { cache, boxes }
    }

    pub fn is_unescaped(&self, opref: OpRef) -> bool {
        self.cache.is_unescaped(opref, self.boxes)
    }
    pub fn saw_allocation(&self, opref: OpRef) -> bool {
        self.cache.saw_allocation(opref, self.boxes)
    }
    pub fn is_class_known(&self, opref: OpRef) -> bool {
        self.cache.is_class_known(opref, self.boxes)
    }
    pub fn is_nullity_known(&self, opref: OpRef) -> bool {
        self.cache.is_nullity_known(opref, self.boxes)
    }
    pub fn is_likely_virtual(&self, opref: OpRef) -> bool {
        self.cache.is_likely_virtual(opref, self.boxes)
    }
    pub fn is_known_nonstandard_virtualizable(&self, opref: OpRef) -> bool {
        self.cache
            .is_known_nonstandard_virtualizable(opref, self.boxes)
    }
    pub fn arraylen(&self, array: OpRef) -> Option<OpRef> {
        self.cache.arraylen(array, self.boxes)
    }
    pub fn test_head_version(&self, opref: OpRef) -> bool {
        self.cache.test_head_version(opref, self.boxes)
    }
    pub fn test_likely_virtual_version(&self, opref: OpRef) -> bool {
        self.cache.test_likely_virtual_version(opref, self.boxes)
    }
    pub fn _check_flag(&self, opref: OpRef, flag: HeapFlags) -> bool {
        self.cache._check_flag(opref, flag, self.boxes)
    }
}

impl Deref for HeapCacheView<'_> {
    type Target = HeapCache;
    fn deref(&self) -> &HeapCache {
        self.cache
    }
}

/// Mutable view of [`HeapCache`] plus the FrontendOp records it writes.
pub struct HeapCacheViewMut<'a> {
    cache: &'a mut HeapCache,
    boxes: &'a mut dyn HeapcBoxes,
}

impl<'a> HeapCacheViewMut<'a> {
    pub fn new(cache: &'a mut HeapCache, boxes: &'a mut dyn HeapcBoxes) -> Self {
        Self { cache, boxes }
    }

    pub fn as_view(&self) -> HeapCacheView<'_> {
        HeapCacheView {
            cache: self.cache,
            boxes: self.boxes,
        }
    }

    pub fn is_unescaped(&self, opref: OpRef) -> bool {
        self.cache.is_unescaped(opref, self.boxes)
    }
    pub fn saw_allocation(&self, opref: OpRef) -> bool {
        self.cache.saw_allocation(opref, self.boxes)
    }
    pub fn is_class_known(&self, opref: OpRef) -> bool {
        self.cache.is_class_known(opref, self.boxes)
    }
    pub fn is_nullity_known(&self, opref: OpRef) -> bool {
        self.cache.is_nullity_known(opref, self.boxes)
    }
    pub fn is_likely_virtual(&self, opref: OpRef) -> bool {
        self.cache.is_likely_virtual(opref, self.boxes)
    }
    pub fn is_known_nonstandard_virtualizable(&self, opref: OpRef) -> bool {
        self.cache
            .is_known_nonstandard_virtualizable(opref, self.boxes)
    }
    pub fn arraylen(&self, array: OpRef) -> Option<OpRef> {
        self.cache.arraylen(array, self.boxes)
    }
    pub fn new_object(&mut self, opref: OpRef) {
        self.cache.new_object(opref, self.boxes);
    }
    pub fn new_box(&mut self, opref: OpRef) {
        self.cache.new_box(opref, self.boxes);
    }
    pub fn new_array(&mut self, opref: OpRef, lengthbox: OpRef, length_is_const: bool) {
        self.cache
            .new_array(opref, lengthbox, length_is_const, self.boxes);
    }
    pub fn class_now_known(&mut self, opref: OpRef) {
        self.cache.class_now_known(opref, self.boxes);
    }
    pub fn nullity_now_known(&mut self, opref: OpRef) {
        self.cache.nullity_now_known(opref, self.boxes);
    }
    pub fn replace_box(&mut self, old: OpRef, new: OpRef) {
        self.cache.replace_box(old, new, self.boxes);
    }
    pub fn arraylen_now_known(&mut self, array: OpRef, length: OpRef) {
        self.cache.arraylen_now_known(array, length, self.boxes);
    }
    pub fn nonstandard_virtualizables_now_known(&mut self, opref: OpRef) {
        self.cache
            .nonstandard_virtualizables_now_known(opref, self.boxes);
    }
    pub fn notify_op(&mut self, opcode: OpCode, args: &[OpRef], result: OpRef) {
        self.cache.notify_op(opcode, args, result, self.boxes);
    }
    pub fn _escape_box(&mut self, opref: OpRef) {
        self.cache._escape_box(opref, self.boxes);
    }
    pub fn _get_deps(&mut self, opref: OpRef) -> Option<&mut Vec<Option<OpRef>>> {
        self.cache._get_deps(opref, self.boxes)
    }
    pub fn update_version(&mut self, opref: OpRef) {
        self.cache.update_version(opref, self.boxes);
    }
    pub fn _remove_deps_for_box(&mut self, opref: OpRef) {
        self.cache._remove_deps_for_box(opref, self.boxes);
    }
    pub fn getfield_cached(
        &mut self,
        obj: OpRef,
        field_index: u32,
        oracle: &dyn SameConstantOracle,
    ) -> Option<OpRef> {
        self.cache
            .getfield_cached(obj, field_index, oracle, self.boxes)
    }
    pub fn setfield_cached(
        &mut self,
        obj: OpRef,
        field_index: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
    ) {
        self.cache
            .setfield_cached(obj, field_index, value, oracle, self.boxes);
    }
    pub fn getfield_now_known(
        &mut self,
        obj: OpRef,
        field_index: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
    ) {
        self.cache
            .getfield_now_known(obj, field_index, value, oracle, self.boxes);
    }
    pub fn invalidate_caches_for_escaped(&mut self) {
        self.cache.invalidate_caches_for_escaped(self.boxes);
    }
    pub fn maybe_replace_with_const(&self, opref: OpRef) -> OpRef {
        self.cache.maybe_replace_with_const(opref, self.boxes)
    }
    pub fn getarrayitem_cache(
        &mut self,
        array: OpRef,
        index_value: i64,
        descr: u32,
        oracle: &dyn SameConstantOracle,
    ) -> Option<OpRef> {
        self.cache
            .getarrayitem_cache(array, index_value, descr, oracle, self.boxes)
    }
    pub fn setarrayitem_cache(
        &mut self,
        array: OpRef,
        index_value: Option<i64>,
        descr: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
    ) {
        self.cache
            .setarrayitem_cache(array, index_value, descr, value, oracle, self.boxes);
    }
    pub fn getarrayitem_now_known(
        &mut self,
        array: OpRef,
        index_value: Option<i64>,
        descr: u32,
        value: OpRef,
        oracle: &dyn SameConstantOracle,
    ) {
        self.cache
            .getarrayitem_now_known(array, index_value, descr, value, oracle, self.boxes);
    }
    pub fn invalidate_caches_varargs<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        self.cache.invalidate_caches_varargs(
            opnum,
            effectinfo,
            argboxes,
            oracle,
            const_value,
            self.boxes,
        );
    }
    pub fn mark_escaped(&mut self, opnum: OpCode, descr: Option<OpRef>, argboxes: &[OpRef]) {
        self.cache.mark_escaped(opnum, descr, argboxes, self.boxes);
    }
    pub fn invalidate_caches<F: Fn(OpRef) -> Option<i64>>(
        &mut self,
        opnum: OpCode,
        effectinfo: Option<&EffectInfo>,
        argboxes: &[OpRef],
        oracle: &dyn SameConstantOracle,
        const_value: F,
    ) {
        self.cache
            .invalidate_caches(opnum, effectinfo, argboxes, oracle, const_value, self.boxes);
    }
}

impl Deref for HeapCacheViewMut<'_> {
    type Target = HeapCache;
    fn deref(&self) -> &HeapCache {
        self.cache
    }
}

impl DerefMut for HeapCacheViewMut<'_> {
    fn deref_mut(&mut self) -> &mut HeapCache {
        self.cache
    }
}

/// Test FrontendOp table indexed by `OpRef.raw()`, the same coordinate
/// value ops and inputargs share after `_start`.
#[derive(Default)]
pub struct TestHeapcBoxes {
    records: Vec<HeapcRecord>,
    values: Vec<Option<Value>>,
}

impl TestHeapcBoxes {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure(&mut self, opref: OpRef) {
        if opref.is_constant() {
            return;
        }
        let i = opref.raw() as usize;
        if i >= self.records.len() {
            self.records.resize_with(i + 1, HeapcRecord::default);
            self.values.resize(i + 1, None);
        }
    }

    pub fn set_value(&mut self, opref: OpRef, value: Value) {
        self.ensure(opref);
        if !opref.is_constant() {
            self.values[opref.raw() as usize] = Some(value);
        }
    }
}

impl HeapcBoxes for TestHeapcBoxes {
    fn heapc(&self, opref: OpRef) -> Option<&HeapcRecord> {
        if opref.is_constant() {
            return None;
        }
        self.records.get(opref.raw() as usize)
    }

    fn heapc_mut(&mut self, opref: OpRef) -> Option<&mut HeapcRecord> {
        if opref.is_constant() {
            return None;
        }
        self.ensure(opref);
        self.records.get_mut(opref.raw() as usize)
    }

    fn box_value(&self, opref: OpRef) -> Option<Value> {
        if opref.is_constant() {
            return opref.inline_const_to_value();
        }
        self.values.get(opref.raw() as usize).and_then(|v| *v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        cache: HeapCache,
        boxes: TestHeapcBoxes,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                cache: HeapCache::new(),
                boxes: TestHeapcBoxes::new(),
            }
        }

        fn view(&mut self) -> HeapCacheViewMut<'_> {
            HeapCacheViewMut::new(&mut self.cache, &mut self.boxes)
        }
    }

    /// Test fixture for `_unique_const_heuristic`: two ConstPtr OpRefs
    /// `(typed-Ref, raw=10000)` and `(typed-Ref, raw=10001)` compare
    /// equal under the oracle iff their pre-registered indices are.
    struct FixedSameConstantOracle {
        same_pairs: Vec<(OpRef, OpRef)>,
    }

    impl SameConstantOracle for FixedSameConstantOracle {
        fn same_constant(&self, a: OpRef, b: OpRef) -> bool {
            if a == b {
                return true;
            }
            self.same_pairs
                .iter()
                .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
        }
    }

    /// Identity-only oracle for tests that exercise non-ConstPtr OpRefs.
    /// Same as `FixedSameConstantOracle { same_pairs: vec![] }` but
    /// shorter at the callsite.
    struct IdentitySameConstantOracle;

    impl SameConstantOracle for IdentitySameConstantOracle {
        fn same_constant(&self, a: OpRef, b: OpRef) -> bool {
            a == b
        }
    }

    const IDENTITY_ORACLE: &dyn SameConstantOracle = &IdentitySameConstantOracle;

    /// `_unique_const_heuristic` collapses consecutive equal ConstPtr
    /// arguments to the cached `last_const_box`, even when the two
    /// OpRefs are distinct (post-dedup-retirement shape).
    #[test]
    fn unique_const_heuristic_canonicalises_to_last_via_same_constant() {
        let mut entry = CacheEntry::new();
        let a = OpRef::const_ptr(majit_ir::GcRef(0xA000));
        let b = OpRef::const_ptr(majit_ir::GcRef(0xB000));
        let oracle = FixedSameConstantOracle {
            same_pairs: vec![(a, b)],
        };
        assert_eq!(entry._unique_const_heuristic(a, &oracle), a);
        assert_eq!(entry._unique_const_heuristic(b, &oracle), a);
    }

    /// Non-constant OpRefs bypass the heuristic unchanged
    /// (heapcache.py `isinstance(ref_box, ConstPtr)` guard).
    #[test]
    fn unique_const_heuristic_skips_non_constant() {
        let mut entry = CacheEntry::new();
        let oracle = FixedSameConstantOracle { same_pairs: vec![] };
        let op = OpRef::ref_op(7);
        assert_eq!(entry._unique_const_heuristic(op, &oracle), op);
        assert!(entry.last_const_box.is_none());
    }

    /// Non-Ref-typed constants (ConstInt / ConstFloat) bypass the
    /// heuristic — upstream only canonicalises ConstPtr.
    #[test]
    fn unique_const_heuristic_skips_non_ref_constants() {
        let mut entry = CacheEntry::new();
        let oracle = FixedSameConstantOracle { same_pairs: vec![] };
        let ci = OpRef::const_int(42);
        assert_eq!(entry._unique_const_heuristic(ci, &oracle), ci);
        assert!(entry.last_const_box.is_none());
    }

    /// `_get_deps`'s `if not isinstance(box, RefFrontendOp): return None`.
    /// A `Const` has no `raw()` index, so without the arm the accessor
    /// panics instead of declining.
    #[test]
    fn get_deps_declines_a_constant_instead_of_indexing_it() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        assert!(cache._get_deps(OpRef::const_ptr(GcRef(0x1000))).is_none());
        assert!(cache._get_deps(OpRef::const_int(42)).is_none());
        assert!(cache._get_deps(OpRef::ref_op(0)).is_some());
    }

    /// `update_version` is already a no-op for a `Const` through
    /// `flags_for_ref` / `set_flags_for_ref`; its last line must not then
    /// index one.
    #[test]
    fn update_version_is_a_no_op_for_a_constant() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        cache.update_version(OpRef::const_ptr(GcRef(0x1000)));
        cache._remove_deps_for_box(OpRef::const_int(42));
    }

    #[test]
    fn test_field_cache_basic() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(0);
        let field = 1;
        let val = OpRef::ref_op(2);

        assert_eq!(cache.getfield_cached(obj, field, IDENTITY_ORACLE), None);

        cache.getfield_now_known(obj, field, val, IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj, field, IDENTITY_ORACLE),
            Some(val)
        );
    }

    #[test]
    fn test_field_cache_overwrite() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(0);
        let field = 1;

        cache.getfield_now_known(obj, field, OpRef::ref_op(10), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(10))
        );

        cache.getfield_now_known(obj, field, OpRef::ref_op(20), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
    }

    #[test]
    fn test_setfield_aliasing() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj_a = OpRef::ref_op(0);
        let obj_b = OpRef::ref_op(1);
        let field = 5;

        // Both objects have a known field value
        cache.getfield_now_known(obj_a, field, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(obj_b, field, OpRef::ref_op(20), IDENTITY_ORACLE);

        // Writing to obj_a (which is NOT unescaped) should invalidate
        // obj_b's field cache for the same field (potential aliasing).
        cache.setfield_cached(obj_a, field, OpRef::ref_op(30), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj_a, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(30))
        );
        assert_eq!(cache.getfield_cached(obj_b, field, IDENTITY_ORACLE), None); // invalidated
    }

    /// heapcache.py `_clear_cache_on_write(seen_alloc)`.  When the
    /// write target is seen-allocated, only `cache_anything` is cleared
    /// — entries for other seen-allocated boxes in
    /// `cache_seen_allocation` survive because distinct
    /// seen-allocation identities cannot alias each other.
    #[test]
    fn test_setfield_seen_alloc_preserves_other_seen_alloc_entries() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj_a = OpRef::ref_op(0);
        let obj_b = OpRef::ref_op(1);
        let field = 5;

        // Both targets have been observed allocating, so each lives in
        // the seen-allocation bucket and they don't alias each other.
        cache.new_object(obj_a);
        cache.new_object(obj_b);
        cache.getfield_now_known(obj_a, field, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(obj_b, field, OpRef::ref_op(20), IDENTITY_ORACLE);

        // Writing to obj_a leaves obj_b's seen-alloc entry intact.
        cache.setfield_cached(obj_a, field, OpRef::ref_op(30), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj_a, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(30))
        );
        assert_eq!(
            cache.getfield_cached(obj_b, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
    }

    /// heapcache.py _clear_cache_on_write — when the write target is seen-allocated but
    /// some other cached box is not, the non-seen-alloc entry lives in
    /// `cache_anything` and is dropped by `_clear_cache_on_write` even
    /// though the target itself is in `cache_seen_allocation`.
    #[test]
    fn test_setfield_seen_alloc_clears_cache_anything() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj_a = OpRef::ref_op(0);
        let obj_b = OpRef::ref_op(1);
        let field = 5;

        cache.new_object(obj_a);
        // obj_b is NOT new_object'd → lives in cache_anything.
        cache.getfield_now_known(obj_a, field, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(obj_b, field, OpRef::ref_op(20), IDENTITY_ORACLE);

        cache.setfield_cached(obj_a, field, OpRef::ref_op(30), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj_a, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(30))
        );
        assert_eq!(cache.getfield_cached(obj_b, field, IDENTITY_ORACLE), None);
    }

    #[test]
    fn test_invalidate_caches() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        cache.getfield_now_known(OpRef::ref_op(0), 1, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(OpRef::ref_op(1), 2, OpRef::ref_op(20), IDENTITY_ORACLE);

        cache.reset_keep_likely_virtuals();
        assert_eq!(
            cache.getfield_cached(OpRef::ref_op(0), 1, IDENTITY_ORACLE),
            None
        );
        assert_eq!(
            cache.getfield_cached(OpRef::ref_op(1), 2, IDENTITY_ORACLE),
            None
        );
    }

    #[test]
    fn test_invalidate_caches_for_escaped() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let escaped_obj = OpRef::ref_op(0);
        let unescaped_obj = OpRef::ref_op(1);

        cache.new_object(unescaped_obj);
        cache.getfield_now_known(escaped_obj, 1, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(unescaped_obj, 1, OpRef::ref_op(20), IDENTITY_ORACLE);

        cache.invalidate_caches_for_escaped();
        assert_eq!(cache.getfield_cached(escaped_obj, 1, IDENTITY_ORACLE), None);
        assert_eq!(
            cache.getfield_cached(unescaped_obj, 1, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
    }

    /// `pyjitpl.py do_residual_call` step 5 uses `invalidate_caches_varargs`
    /// on CALL_MAY_FORCE, which takes `reset_keep_likely_virtuals` and
    /// drops an unescaped object's GETFIELD. `invalidate_caches_for_escaped`
    /// would keep that cache.
    #[test]
    fn call_may_force_varargs_drops_unescaped_getfield() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let unescaped_obj = OpRef::ref_op(1);
        cache.new_object(unescaped_obj);
        cache.getfield_now_known(unescaped_obj, 1, OpRef::ref_op(20), IDENTITY_ORACLE);
        cache.invalidate_caches_varargs(
            OpCode::CallMayForceR,
            None,
            &[unescaped_obj],
            IDENTITY_ORACLE,
            |_| None,
        );
        assert_eq!(
            cache.getfield_cached(unescaped_obj, 1, IDENTITY_ORACLE),
            None
        );
    }

    #[test]
    fn test_new_object() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(5);

        assert!(!cache.is_unescaped(obj));
        assert!(!cache.saw_allocation(obj));

        cache.new_object(obj);
        assert!(cache.is_unescaped(obj));
        assert!(cache.saw_allocation(obj));
    }

    #[test]
    fn test_mark_escaped() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(5);

        cache.new_object(obj);
        assert!(cache.is_unescaped(obj));

        cache._escape_box(obj);
        assert!(!cache.is_unescaped(obj));
        // saw_allocation is permanent
        assert!(cache.saw_allocation(obj));
    }

    #[test]
    fn test_known_class() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(0);

        assert!(!cache.is_class_known(obj));
        cache.class_now_known(obj);
        assert!(cache.is_class_known(obj));
    }

    #[test]
    fn test_notify_guard_class_sets_known_class_flag() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(3);
        let cls = OpRef::const_int(0xCAFE);

        cache.notify_op(OpCode::GuardClass, &[obj, cls], OpRef::NONE);

        assert!(cache.is_class_known(obj));
    }

    #[test]
    fn test_walk_const_ptr_refs_forwards_replaced_with_const() {
        // `FO_REPLACED_WITH_CONST` recovers the Const from the box's own
        // value (`constant_from_op`). Forwarding that value is the
        // recorder's walk; HeapCache no longer stores the Const.
        let mut fx = Fixture::new();
        let old = OpRef::ref_op(3);
        fx.boxes.set_value(old, Value::Ref(GcRef(0x1000)));
        {
            let mut cache = fx.view();
            cache.replace_box(old, OpRef::const_ptr(GcRef(0x1000)));
            assert_eq!(
                cache.maybe_replace_with_const(old),
                OpRef::const_ptr(GcRef(0x1000))
            );
        }
        if let Some(Value::Ref(mut gcref)) = fx.boxes.box_value(old) {
            gcref.0 = gcref.0.wrapping_add(0x1_0000);
            fx.boxes.set_value(old, Value::Ref(gcref));
        }
        assert_eq!(
            fx.view().maybe_replace_with_const(old),
            OpRef::const_ptr(GcRef(0x1_1000))
        );
    }

    #[test]
    fn test_walk_const_ptr_refs_leaves_cache_keys_stale() {
        // Cache *keys* are intentionally not forwarded: an in-place key
        // rewrite would break the sorted-VecMap ordering, and a stale key
        // simply misses + repopulates (the live lookup arrives already
        // forwarded). A `ConstPtr` object used as a `cache_anything`
        // key must therefore stay at its pre-collection address.
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let const_obj = OpRef::const_ptr(GcRef(0x2000));
        let field = 7;
        // A non-const cached value so the walk leaves the value slot alone
        // and the assertions isolate the key's address.
        cache.getfield_now_known(const_obj, field, OpRef::ref_op(20), IDENTITY_ORACLE);

        cache.walk_const_ptr_refs(&mut |gcref: &mut GcRef| {
            gcref.0 = gcref.0.wrapping_add(0x1_0000);
        });

        // The entry is still keyed by the original (pre-move) address...
        assert_eq!(
            cache.getfield_cached(const_obj, field, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
        // ...and was NOT re-keyed to the forwarded address.
        assert_eq!(
            cache.getfield_cached(OpRef::const_ptr(GcRef(0x1_2000)), field, IDENTITY_ORACLE),
            None
        );
    }

    #[test]
    fn test_notify_op_malloc() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let result = OpRef::ref_op(3);

        cache.notify_op(OpCode::New, &[], result);
        assert!(cache.is_unescaped(result));
        assert!(cache.saw_allocation(result));
    }

    #[test]
    fn test_reset() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        cache.new_object(OpRef::ref_op(0));
        cache.class_now_known(OpRef::ref_op(0));
        cache.getfield_now_known(OpRef::ref_op(0), 1, OpRef::ref_op(10), IDENTITY_ORACLE);

        cache.reset();
        assert!(!cache.is_unescaped(OpRef::ref_op(0)));
        assert!(!cache.is_class_known(OpRef::ref_op(0)));
        assert_eq!(
            cache.getfield_cached(OpRef::ref_op(0), 1, IDENTITY_ORACLE),
            None
        );
    }

    #[test]
    fn test_different_fields_independent() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(0);

        cache.getfield_now_known(obj, 1, OpRef::ref_op(10), IDENTITY_ORACLE);
        cache.getfield_now_known(obj, 2, OpRef::ref_op(20), IDENTITY_ORACLE);

        // Writing field 1 should not affect field 2
        cache.setfield_cached(obj, 1, OpRef::ref_op(30), IDENTITY_ORACLE);
        assert_eq!(
            cache.getfield_cached(obj, 1, IDENTITY_ORACLE),
            Some(OpRef::ref_op(30))
        );
        assert_eq!(
            cache.getfield_cached(obj, 2, IDENTITY_ORACLE),
            Some(OpRef::ref_op(20))
        );
    }

    #[test]
    fn test_recursive_escape() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let container = OpRef::ref_op(0);
        let value = OpRef::ref_op(1);
        let inner = OpRef::ref_op(2);

        cache.new_object(container);
        cache.new_object(value);
        cache.new_object(inner);

        // SETFIELD_GC(container, value): value stored in container
        cache.notify_op(OpCode::SetfieldGc, &[container, value], OpRef::NONE);
        // SETFIELD_GC(value, inner): inner stored in value
        cache.notify_op(OpCode::SetfieldGc, &[value, inner], OpRef::NONE);

        // Container is still unescaped
        assert!(cache.is_unescaped(container));
        // Value is still unescaped (container is unescaped)
        assert!(cache.is_unescaped(value));

        // Now mark container as escaped
        cache._escape_box(container);
        assert!(!cache.is_unescaped(container));
        // value should also be escaped (stored in container)
        assert!(!cache.is_unescaped(value));
    }

    #[test]
    fn test_nullity_tracking() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(10);

        assert!(!cache.is_nullity_known(obj));
        cache.nullity_now_known(obj);
        assert!(cache.is_nullity_known(obj));
    }

    /// heapcache.py is_quasi_immut_known — the mark is per (fielddescr, box), and a
    /// constant receiver keys on `getref_base()`, so two distinct `ConstPtr`
    /// `OpRef`s naming the same object share it while a different object or a
    /// different descr does not.
    #[test]
    fn quasi_immut_mark_is_keyed_per_descr_and_per_ref() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::const_ptr(GcRef(0x1000));
        let same_obj = OpRef::const_ptr(GcRef(0x1000));
        let other_obj = OpRef::const_ptr(GcRef(0x2000));

        assert!(!cache.is_quasi_immut_known(7, obj));
        cache.quasi_immut_now_known(7, obj);
        assert!(cache.is_quasi_immut_known(7, obj));
        assert!(cache.is_quasi_immut_known(7, same_obj));
        assert!(!cache.is_quasi_immut_known(7, other_obj));
        assert!(!cache.is_quasi_immut_known(8, obj));
    }

    /// heapcache.py `invalidate_unescaped` clears
    /// `quasiimmut_seen{,_refs}` — the lifetime that makes the second
    /// `GUARD_NOT_INVALIDATED` possible.  `clear_caches_varargs`
    /// (heapcache.py) runs this for every general call while also
    /// arming `need_guard_not_invalidated`, so the next read of the field
    /// re-emits `QUASIIMMUT_FIELD` and that op emits the guard.
    ///
    /// The receiver here is an escaped (non-allocated) box, which is what a
    /// pinned type constant is.
    #[test]
    fn quasi_immut_mark_is_dropped_by_the_call_invalidation() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::const_ptr(GcRef(0x1000));

        cache.quasi_immut_now_known(7, obj);
        assert!(cache.is_quasi_immut_known(7, obj));

        cache.invalidate_caches_for_escaped();
        assert!(!cache.is_quasi_immut_known(7, obj));
    }

    /// heapcache.py `_clear_cache_on_write` clears the same two sets, so
    /// a store to the field drops the mark as well.
    #[test]
    fn quasi_immut_mark_is_dropped_by_a_store_to_the_field() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::const_ptr(GcRef(0x1000));
        let value = OpRef::int_op(3);

        cache.quasi_immut_now_known(7, obj);
        cache.setfield_cached(obj, 7, value, IDENTITY_ORACLE);
        assert!(!cache.is_quasi_immut_known(7, obj));
    }

    #[test]
    fn test_arraylen_caching() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let arr = OpRef::ref_op(5);

        assert_eq!(cache.arraylen(arr), None);
        cache.arraylen_now_known(arr, OpRef::int_op(100));
        assert_eq!(cache.arraylen(arr), Some(OpRef::int_op(100)));
    }

    #[test]
    fn test_arraylen_reset_keep_likely_virtuals_invalidates_length() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let arr = OpRef::ref_op(5);

        cache.arraylen_now_known(arr, OpRef::int_op(100));
        assert_eq!(cache.arraylen(arr), Some(OpRef::int_op(100)));

        cache.reset_keep_likely_virtuals();
        assert_eq!(cache.arraylen(arr), None);
    }

    #[test]
    fn test_replace_box_marks_old_as_const() {
        let mut fx = Fixture::new();
        let old = OpRef::ref_op(5);
        let new = OpRef::const_ptr(majit_ir::GcRef(0xDEAD));
        fx.boxes.set_value(old, Value::Ref(GcRef(0xDEAD)));
        let mut cache = fx.view();

        cache.arraylen_now_known(old, old);
        cache.replace_box(old, new);

        assert_eq!(cache.arraylen(old), Some(new));
    }

    #[test]
    fn test_replace_box_keeps_typed_opref_identity() {
        let mut fx = Fixture::new();
        let old_ref = OpRef::ref_op(5);
        let same_raw_int = OpRef::int_op(5);
        let new_ref = OpRef::const_ptr(GcRef(0));
        fx.boxes.set_value(old_ref, Value::Ref(GcRef(0)));
        let mut cache = fx.view();

        cache.arraylen_now_known(old_ref, old_ref);
        cache.arraylen_now_known(OpRef::ref_op(6), same_raw_int);
        cache.replace_box(old_ref, new_ref);

        assert_eq!(cache.arraylen(old_ref), Some(new_ref));
        assert_eq!(cache.arraylen(OpRef::ref_op(6)), Some(same_raw_int));
    }

    #[test]
    fn test_likely_virtual() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(3);

        assert!(!cache.is_likely_virtual(obj));
        cache.new_object(obj);
        assert!(cache.is_likely_virtual(obj));

        // reset keeps likely_virtual
        cache.reset_keep_likely_virtuals();
        assert!(cache.is_likely_virtual(obj));

        // full reset clears it
        cache.reset();
        assert!(!cache.is_likely_virtual(obj));
    }

    #[test]
    fn test_guard_tracking_in_notify_op() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let obj = OpRef::ref_op(10);

        // GUARD_NONNULL makes nullity known
        cache.notify_op(OpCode::GuardNonnull, &[obj], OpRef::NONE);
        assert!(cache.is_nullity_known(obj));
    }

    #[test]
    fn test_loopinvariant_void_evicts_typed_slot() {
        let mut fx = Fixture::new();
        let mut cache = fx.view();
        let descr_index: u32 = 7;
        let arg0_int: i64 = 0xC0FFEE;

        // Prime with a typed entry — pyjitpl.py:2087-2110 tp == 'i' branch.
        let typed_result = OpRef::ref_op(42);
        cache.call_loopinvariant_now_known(descr_index, arg0_int, typed_result, 99);
        assert_eq!(
            cache.call_loopinvariant_known_result(descr_index, arg0_int),
            Some((typed_result, 99))
        );

        // pyjitpl.py:2103-2109 tp == 'v' branch: res = None,
        // call_loopinvariant_now_known(allboxes, descr, None).
        cache.call_loopinvariant_now_known_void(descr_index, arg0_int);

        // heapcache.py call_loopinvariant_known_result: subsequent lookup returns None — the
        // (descr, arg0) slot is still owned but its result is None,
        // so `if res is not None: return res` short-circuit misses.
        assert_eq!(
            cache.call_loopinvariant_known_result(descr_index, arg0_int),
            None
        );
    }
}
