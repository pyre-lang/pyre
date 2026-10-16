//! Side effect classification for calls.
//!
//! Translated from `rpython/jit/codewriter/effectinfo.py`.
//!
//! TODO: the upstream module path is
//! `rpython/jit/codewriter/effectinfo.py`, but in pyre this lives in
//! `majit-ir` because `LLType::for_call` (descr.rs) holds
//! `extraeffect`/`oopspecindex` indices and `EffectInfo` is constructed
//! from RPython call analysis. Putting it in `majit-translate` would
//! create a circular crate dependency (`majit-ir` ↔ `majit-translate`).
//! The name and contents otherwise mirror upstream line-by-line.

use crate::descr::DescrRef;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// effectinfo.py `class UnsupportedFieldExc(Exception)`.
#[derive(Debug, Clone)]
pub struct UnsupportedFieldExc(pub String);

impl std::fmt::Display for UnsupportedFieldExc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnsupportedFieldExc {}

/// Serializable projection of one member of the six raw EffectInfo descr
/// sets: the gccache key the analyzer minted the descr through.
///
/// `Arc<dyn Descr>` cannot cross the `descrs.bin` process boundary, but the
/// key can — both halves of the split agree on it by construction, since
/// `cpu.fielddescrof(STRUCT, fieldname)` / `cpu.arraydescrof(ARRAY)` /
/// `cpu.interiorfielddescrof(ARRAY, fieldname)` are cache lookups on exactly
/// these tuples.  The key alone identifies the slot; what it does *not* carry
/// is the layout needed to create the descr when the slot is empty, which is
/// what [`DescrMintSpec`] adds.
/// `Ord` is derived so build-time consumers can order a raw set by its
/// *members* rather than by `Arc::as_ptr`.  A heap address orders a set
/// reproducibly within one process and arbitrarily across two, which makes
/// any artifact carrying that order a non-function of its inputs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DescrSetMember {
    /// `descr.py get_field_descr(gccache, STRUCT, fieldname)` —
    /// `_cache_field[Struct(struct_id)][field_name]`.
    Field { struct_id: u64, field_name: String },
    /// `descr.py get_array_descr(gccache, ARRAY)` —
    /// `_cache_array[Array(array_id)]`.
    Array { array_id: u64 },
    /// `descr.py get_interiorfield_descr(gccache, ARRAY, fieldname)` —
    /// `_cache_interiorfield[(Array(array_id), name, "")]`.
    InteriorField { array_id: u64, name: String },
}

/// The layout arguments `descr.py`'s three getters pass to the descr
/// constructor on a cache miss.
///
/// `get_field_descr` (`descr.py`), `get_array_descr`
/// (`descr.py`) and `get_interiorfield_descr` (`descr.py`) are
/// all `try: return cache[key] / except KeyError: <construct>; cache[key] = d`
/// — they **mint** when the slot is empty.  Upstream never has to serialize
/// those arguments because there is one gccache in one process: the raw sets
/// hold the descr objects themselves, and `setup_descrs` (`descr.py`)
/// snapshots the same cache they were minted into, so
/// `compute_bitstrings` (`effectinfo.py`) only ever unions descrs that
/// are already present.
///
/// Pyre mints in the analyzer process and resolves in the runtime one, and only
/// the assembler's opcode table (`pyjitpl.py setup_descrs(asm.descrs)`)
/// crosses between them.  A descr the analyzer minted *solely* to fill a raw
/// set is referenced by no opcode, so it does not cross, and the runtime lookup
/// finds an empty slot.  Pairing each member with the arguments its cache miss
/// would need restores upstream's mint-or-hit at the far end.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DescrMintSpec {
    /// `descr.py` — the `FieldDescr(name, offset, size, flag,
    /// index_in_parent, is_pure)` arguments, plus `descr.py
    /// parent_descr = get_size_descr(gccache, STRUCT, vtable)`'s `STRUCT` size.
    ///
    /// No vtable: the analyzer has no vtable surface, and a runtime publish
    /// under the same key carries the real one — the authority rule in
    /// `descr::GcCache::register_keyed_size` keeps that one whichever side
    /// registers first.
    Field {
        struct_size: usize,
        offset: usize,
        field_size: usize,
        field_type: crate::value::Type,
        flag: crate::descr::ArrayFlag,
        is_immutable: bool,
        is_quasi_immutable: bool,
        index_in_parent: usize,
    },
    /// `descr.py` — the `ArrayDescr(basesize, itemsize, lendescr, flag,
    /// is_pure, concrete_type)` arguments. `length_offset` is what
    /// `get_field_arraylen_descr` (`descr.py`) needs, and is read only
    /// when `!nolength`, matching `descr.py`.
    Array {
        base_size: usize,
        item_size: usize,
        flag: crate::descr::ArrayFlag,
        item_type: crate::value::Type,
        nolength: bool,
        length_offset: usize,
        is_pure: bool,
        concrete_type: char,
    },
    /// `descr.py:429-436` — an interior field is minted out of the array descr
    /// and the element-struct field descr, so it carries both halves rather
    /// than a flat layout.
    ///
    /// `field_struct_id` is the `REALARRAY.OF` cache key
    /// (`descr.py get_field_descr(gc_ll_descr, REALARRAY.OF, name)`).
    /// Upstream recovers it from the ARRAY lltype; the serialized member names
    /// only the array, so the element struct's key travels explicitly.
    InteriorField {
        array: Box<DescrMintSpec>,
        field_struct_id: u64,
        field_name: String,
        field: Box<DescrMintSpec>,
    },
}

/// One `(slot key, how to fill it)` pair, as serialized alongside the jitcodes.
///
/// The union over every `EffectInfo`'s six raw sets — the same union
/// `compute_bitstrings` builds in `descrs = {'fields': set(), ...}`
/// (`effectinfo.py`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DescrMintEntry {
    pub member: DescrSetMember,
    pub spec: DescrMintSpec,
    /// `effectinfo.py:526 descr.ei_index = mapping.setdefault(...)`, frozen
    /// by source translation. `u32::MAX` until the build-time partition runs.
    pub ei_index: u32,
}

/// `effectinfo.py frozenset_or_none`: serializable projection of
/// the six raw EffectInfo descr sets. The vectors are kept in the same
/// canonical order as the raw `DescrRef` sets so deserialization can rebuild
/// the exact object graph before `compute_bitstrings`.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DescrSetKeys {
    pub readonly_fields: Vec<DescrSetMember>,
    pub write_fields: Vec<DescrSetMember>,
    pub readonly_arrays: Vec<DescrSetMember>,
    pub write_arrays: Vec<DescrSetMember>,
    pub readonly_interiorfields: Vec<DescrSetMember>,
    pub write_interiorfields: Vec<DescrSetMember>,
}

impl DescrSetKeys {
    /// The `frozenset_or_none(...)` image of six *empty* frozensets — an EI
    /// the analyzer proved touches no field, array or interiorfield.
    ///
    /// Distinct from `descr_set_keys = None`, which is the
    /// `EF_RANDOM_EFFECTS` wildcard: `effectinfo.py` makes the raw
    /// sets `None` **iff** the EI is random-effects, and
    /// `compute_bitstrings` (`effectinfo.py`) reads the two shapes
    /// oppositely — empty sets keep the bitstrings, `None` clears them.  A
    /// concrete-but-empty EI serialized with `None` would therefore come
    /// back from `descrs.bin` looking random-effects while its
    /// `extraeffect` still says otherwise, and `check_readonly_descr_field`
    /// would panic on the cleared bitstring.
    ///
    /// `const` so the `EffectInfo` associated constants can name it.
    pub const fn const_empty() -> Self {
        Self {
            readonly_fields: Vec::new(),
            write_fields: Vec::new(),
            readonly_arrays: Vec::new(),
            write_arrays: Vec::new(),
            readonly_interiorfields: Vec::new(),
            write_interiorfields: Vec::new(),
        }
    }

    fn canonicalize(&mut self) {
        fn sort_dedup(members: &mut Vec<DescrSetMember>) {
            members.sort();
            members.dedup();
        }
        sort_dedup(&mut self.readonly_fields);
        sort_dedup(&mut self.write_fields);
        sort_dedup(&mut self.readonly_arrays);
        sort_dedup(&mut self.write_arrays);
        sort_dedup(&mut self.readonly_interiorfields);
        sort_dedup(&mut self.write_interiorfields);
    }

    /// Every member of the six raw sets, in category order.
    pub fn iter_members(&self) -> impl Iterator<Item = &DescrSetMember> {
        self.readonly_fields
            .iter()
            .chain(&self.write_fields)
            .chain(&self.readonly_arrays)
            .chain(&self.write_arrays)
            .chain(&self.readonly_interiorfields)
            .chain(&self.write_interiorfields)
    }

    /// Map each raw-set member onto the dense translated descriptor table.
    ///
    /// RPython's `compute_bitstrings` mutates live `Descr` objects that
    /// `GcCache.setup_descrs` already numbered in the same process. Pyre's
    /// analyzer records those objects as stamp (opcode-published) then mint
    /// (`get_field_descr` / `get_array_descr` / `get_interiorfield_descr`
    /// cache-miss) slots; this rewrite is that numbering, so the runtime
    /// pass is an index lookup rather than a structural `DescrSetMember`
    /// walk.
    pub fn to_slots(
        &self,
        member_index: &std::collections::BTreeMap<DescrSetMember, u32>,
    ) -> DescrSetSlots {
        fn map_members(
            members: &[DescrSetMember],
            member_index: &std::collections::BTreeMap<DescrSetMember, u32>,
        ) -> Vec<u32> {
            members
                .iter()
                .map(|member| {
                    *member_index.get(member).unwrap_or_else(|| {
                        panic!(
                            "frozen EffectInfo raw-set member has no stamp/mint/lookup slot: {member:?}"
                        )
                    })
                })
                .collect()
        }
        DescrSetSlots {
            readonly_fields: map_members(&self.readonly_fields, member_index),
            write_fields: map_members(&self.write_fields, member_index),
            readonly_arrays: map_members(&self.readonly_arrays, member_index),
            write_arrays: map_members(&self.write_arrays, member_index),
            readonly_interiorfields: map_members(&self.readonly_interiorfields, member_index),
            write_interiorfields: map_members(&self.write_interiorfields, member_index),
        }
    }
}

/// Compact translated-image form of [`DescrSetKeys`]: each member is an
/// index into the stamp-then-mint-then-lookup descriptor table
/// (`compute_bitstrings` / `GcCache.setup_descrs` numbering across the
/// analyzer/runtime split).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescrSetSlots {
    pub readonly_fields: Vec<u32>,
    pub write_fields: Vec<u32>,
    pub readonly_arrays: Vec<u32>,
    pub write_arrays: Vec<u32>,
    pub readonly_interiorfields: Vec<u32>,
    pub write_interiorfields: Vec<u32>,
}

/// Compact owner for the translation-only raw-set projection.
///
/// PyPy's `EffectInfo.__new__` needs the six frozensets while translation runs,
/// but its own comment states that they are not part of the translated runtime
/// object. Pyre must carry their structural image across the build/runtime
/// boundary, so a populated image is boxed and dropped by
/// `prepare_frozen_effect_info_from_slots`; the overwhelmingly common
/// six-empty-sets case is a zero-allocation enum value usable by
/// `EffectInfo::const_new`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DescrSetKeysImage {
    Empty,
    Populated(Box<DescrSetKeys>),
}

impl From<DescrSetKeys> for DescrSetKeysImage {
    fn from(keys: DescrSetKeys) -> Self {
        if keys == DescrSetKeys::const_empty() {
            Self::Empty
        } else {
            Self::Populated(Box::new(keys))
        }
    }
}

impl std::ops::Deref for DescrSetKeysImage {
    type Target = DescrSetKeys;

    fn deref(&self) -> &Self::Target {
        static EMPTY: DescrSetKeys = DescrSetKeys::const_empty();
        match self {
            Self::Empty => &EMPTY,
            Self::Populated(keys) => keys,
        }
    }
}

impl std::ops::DerefMut for DescrSetKeysImage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if matches!(self, Self::Empty) {
            *self = Self::Populated(Box::new(DescrSetKeys::const_empty()));
        }
        let Self::Populated(keys) = self else {
            unreachable!()
        };
        keys
    }
}

/// `EffectInfo` with setup-time interior mutability for the bitstring
/// fields.
///
/// PyPy `effectinfo.py` documents that `bitstring_*_descrs_*`
/// fields are "initialized later, in compute_bitstrings()" — the EI
/// object is constructed once with raw `_*_descrs_*` frozensets, then
/// mutated by `effectinfo.compute_bitstrings(self.all_descrs)`
/// (`pyjitpl.py`) to install the compacted bitstrings.
/// Python's mutable object model makes the in-place setattr trivial;
/// Rust requires explicit interior mutability for the same shape.
///
/// `EffectInfoCell` wraps the EI in `UnsafeCell` and exposes:
/// - `get(&self) -> &EffectInfo` for the immutable-borrow read path
///   used by every `cd.get_extra_info()` consumer (heap.rs / virtualize.rs
///   / rewrite.rs / pure.rs / intbounds.rs / vstring.rs etc., 40+ sites).
/// - `set_bitstrings(&self, …)` for the single setup-time writer
///   (`MetaInterpStaticData::finish_setup_descrs`).
///
/// SAFETY contract (encoded via the `set_bitstrings` documentation,
/// not enforced statically): exactly one writer at JIT-setup time,
/// no concurrent reader-borrow in flight during the write. Pyre's
/// JIT pipeline is single-threaded today; the `OnceLock`-like
/// publish-once pattern at `pyre-jit-trace::state::finish_setup_if_needed`
/// (gated on `was_done`) honours this contract structurally. Replacing
/// the earlier raw `&EffectInfo as *const → *mut` cast with `UnsafeCell`
/// keeps the same operational behaviour while making the contract
/// explicit and respecting the Rust aliasing model — `UnsafeCell::get()`
/// returns a `*mut T` whose mutation is permitted given a unique-
/// access invariant on the caller, which is exactly what we have.
#[derive(Debug)]
pub struct EffectInfoCell {
    cell: std::cell::UnsafeCell<EffectInfo>,
}

// SAFETY: pyre's JIT pipeline is single-threaded today; `EffectInfoCell`
// is shared across threads only as part of `Arc<dyn Descr>` cache
// entries that read `get()` after `finish_setup_descrs` has installed
// the final bitstrings. The contract documented on `set_bitstrings`
// requires the writer to have unique access at the moment of the call.
unsafe impl Send for EffectInfoCell {}
unsafe impl Sync for EffectInfoCell {}

impl EffectInfoCell {
    /// Wrap an `EffectInfo` for setup-time mutation.
    pub fn new(ei: EffectInfo) -> Self {
        Self {
            cell: std::cell::UnsafeCell::new(ei),
        }
    }

    /// Read-only borrow.  Used by every `CallDescr::get_extra_info()`
    /// implementor.  SAFETY: shared `&EffectInfo` is sound as long as
    /// no concurrent `set_bitstrings` is in flight; see type doc.
    pub fn get(&self) -> &EffectInfo {
        // SAFETY: the only mutation goes through `set_bitstrings`,
        // which is documented as setup-time single-writer; readers
        // observe the final value after the JIT-setup happens-before
        // edge. Pyre is single-threaded today.
        unsafe { &*self.cell.get() }
    }

    /// `effectinfo.py:537-538 setattr(ei, 'bitstring_*', ...)` —
    /// install the compacted bitstrings produced by
    /// `compute_bitstrings`.  Sole caller is
    /// `MetaInterpStaticData::finish_setup_descrs`.
    ///
    /// SAFETY: see type doc. The argument signature mirrors
    /// `Descr::set_effect_bitstrings`, mutating the six bitstring
    /// fields and leaving every other EI field untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn set_bitstrings(
        &self,
        readonly_descrs_fields: Option<Vec<u8>>,
        write_descrs_fields: Option<Vec<u8>>,
        readonly_descrs_arrays: Option<Vec<u8>>,
        write_descrs_arrays: Option<Vec<u8>>,
        readonly_descrs_interiorfields: Option<Vec<u8>>,
        write_descrs_interiorfields: Option<Vec<u8>>,
    ) {
        // SAFETY: see type doc. Single-writer setup-time mutation.
        unsafe {
            let ei = &mut *self.cell.get();
            ei.readonly_descrs_fields = readonly_descrs_fields;
            ei.write_descrs_fields = write_descrs_fields;
            ei.readonly_descrs_arrays = readonly_descrs_arrays;
            ei.write_descrs_arrays = write_descrs_arrays;
            ei.readonly_descrs_interiorfields = readonly_descrs_interiorfields;
            ei.write_descrs_interiorfields = write_descrs_interiorfields;
        }
    }
}

impl Clone for EffectInfoCell {
    fn clone(&self) -> Self {
        // Cloning observes the current EI state; this is what
        // existing test fixtures need.
        Self::new(self.get().clone())
    }
}

/// `effectinfo.py, 147-148 EffectInfo._cache`.
///
/// PyPy constructs one canonical EffectInfo object for each structural raw-set
/// key and the call-descr cache stores that object by identity.  Keep the same
/// process-global ownership here — `LLType::func_key` keys on this cell's
/// address for exactly that reason, so anything that splits a cell mints a
/// distinct call descr.
///
/// Upstream's `_cache` is a dict; this is an insertion-ordered vector scanned
/// linearly, the shape `AGENTS.md`'s container rule allows for a small key
/// space.  Measured, rather than assumed: the population tops out between 32
/// and 63 distinct EffectInfos, and the heaviest workload tried (`test_pickle`,
/// 999 tests) took 6,519 lookups for 44,266 candidate comparisons — about seven
/// per lookup, most exiting on the first three integer fields.  A keyed map
/// would have to duplicate all six raw descriptor sets to find the cell it
/// replaces.
///
/// The key is `EffectInfo` equality, which carries `runtime_helper`.  Upstream has
/// no such field, and it must stay in: the walker matches a helper by that tag
/// on the descr's EffectInfo because it cannot match by fnaddr (`pyre-jit`'s
/// `flatten.rs`), so two helpers with identical effect sets but different tags
/// must not share a cell — one call site would be handed the other's tag and
/// re-emit the wrong specialization.
///
/// `effectinfo.py` deliberately bypasses the cache for release-gil
/// targets by appending a fresh `object()` to the key; those calls receive a
/// fresh cell here as well.
pub fn intern_effect_info(effect_info: EffectInfo) -> Arc<EffectInfoCell> {
    if effect_info.call_release_gil_target.0 != 0 {
        return Arc::new(EffectInfoCell::new(effect_info));
    }

    // `EffectInfo._cache` is a dict and `EffectInfo.__new__` probes it with
    // `key in cls._cache`, so one lookup costs one hash of the key whatever
    // the cache already holds. A sequence answered the same question
    // by running `EffectInfo`'s `==` — which compares six descr sets
    // element-wise — against every cell already interned, so the cost of
    // interning the n-th distinct EffectInfo grew with n.
    static CACHE: LazyLock<Mutex<HashMap<EffectInfo, Arc<EffectInfoCell>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    // `effectinfo.py:143-146` destructures `tgt_func, tgt_saveerr` but appends
    // the fresh-object breaker only `if tgt_func`, so `tgt_saveerr` never
    // enters the key, and upstream's only null-target value is
    // `_NO_CALL_RELEASE_GIL_TARGET = (NULL, 0)`.  The early return above took
    // every non-null target, so normalise the half no reader can reach from
    // here: `is_call_release_gil` tests `tgt_func` alone, and the two sites
    // that do consume the pair assert a resolved address first.  Without this,
    // a `(0, save_err)` with a null target
    // would key its own cell and mint a call descr upstream would have merged.
    let mut effect_info = effect_info;
    effect_info.call_release_gil_target = EffectInfo::_NO_CALL_RELEASE_GIL_TARGET;

    let mut cache = CACHE.lock();
    // The key is a copy rather than the cell's own EffectInfo because the cell
    // is interior-mutable. Only `set_bitstrings` writes it, and the fields it
    // writes are outside the `Eq`/`Hash` pair below, so the copy and the cell
    // answer every probe identically.
    if let Some(existing) = cache.get(&effect_info) {
        return existing.clone();
    }
    let canonical = Arc::new(EffectInfoCell::new(effect_info.clone()));
    cache.insert(effect_info, canonical.clone());
    canonical
}

/// Recover the EffectInfo object identity assigned by source translation.
///
/// RPython's translated image carries the canonical EffectInfo object itself.
/// Pyre serializes the image through a build-script process, so references to
/// that object cross the boundary as a dense id and are rejoined here. The
/// vector is the direct translated-object table; it is not keyed by descriptor
/// content and does not duplicate any raw descriptor set.
fn translated_effect_infos() -> &'static Mutex<Vec<Option<Arc<EffectInfoCell>>>> {
    static TRANSLATED: LazyLock<Mutex<Vec<Option<Arc<EffectInfoCell>>>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));
    &TRANSLATED
}

pub fn intern_translated_effect_info(
    translated_id: u32,
    effect_info: EffectInfo,
) -> Arc<EffectInfoCell> {
    let mut translated = translated_effect_infos().lock();
    let slot = translated_id as usize;
    if translated.len() <= slot {
        translated.resize_with(slot + 1, || None);
    }
    if let Some(existing) = &translated[slot] {
        return existing.clone();
    }
    let canonical = Arc::new(EffectInfoCell::new(effect_info));
    translated[slot] = Some(canonical.clone());
    canonical
}

/// Look up an EffectInfo already published from the translated object table.
/// Call descriptors carry only this dense identity at runtime; the canonical
/// object itself is loaded once from the frozen EffectInfo artifact.
pub fn translated_effect_info(translated_id: u32) -> Option<Arc<EffectInfoCell>> {
    translated_effect_infos()
        .lock()
        .get(translated_id as usize)
        .and_then(Clone::clone)
}

/// effectinfo.py: frozenset_or_none(x)
///
/// `frozenset(x)` if `x is not None`, else `None`. Pyre's lift of
/// `frozenset[Descr]` is `Vec<DescrRef>` sorted+deduped by
/// `Arc::as_ptr` (object identity, matching PyPy's
/// `id(descr)`-keyed frozenset).  Every consumer
/// (`PartialEq`, `Hash`, `EiCanonKey`, `EffectInfoKey`,
/// `compute_bitstrings`) walks the raw set expecting this
/// canonical Arc-pointer-sorted shape — without it, two
/// structurally-identical sets would compare unequal.
pub fn frozenset_or_none(x: Option<Vec<DescrRef>>) -> Option<Vec<DescrRef>> {
    x.map(canonicalize_descr_set)
}

/// Sort by `Arc::as_ptr` thin pointer (data address only — vtable
/// dropped via `*const () as usize`) and dedup adjacent duplicates.
/// Matches PyPy's `frozenset(x)` identity semantic where two clones
/// of the same `Arc` collapse to one entry, while two distinct Arcs
/// (even of the same descr type) remain separate.
pub fn canonicalize_descr_set(mut v: Vec<DescrRef>) -> Vec<DescrRef> {
    v.sort_by_key(descr_ptr_id);
    v.dedup_by(|a, b| std::sync::Arc::ptr_eq(a, b));
    v
}

/// Project a `DescrRef` to a thin-pointer identity key — `Arc::as_ptr`
/// returns a `*const dyn Descr` (wide pointer); we strip the vtable
/// component via `*const () as usize` so the key is plain `usize`,
/// suitable for `Hash`/`Ord`/`Eq`. Two clones of the same `Arc`
/// share a data pointer; two distinct `Arc`s never do.
#[inline]
pub fn descr_ptr_id(d: &DescrRef) -> usize {
    std::sync::Arc::as_ptr(d) as *const () as usize
}

/// Element-wise Arc-identity equality on a canonicalized
/// `Option<Vec<DescrRef>>`. Both sides assumed already passed
/// through `canonicalize_descr_set` — same Arc-ptr-sorted order,
/// dedup'd. Falls back to length compare + `Arc::ptr_eq` per slot.
fn descr_set_eq(a: &Option<Vec<DescrRef>>, b: &Option<Vec<DescrRef>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(va), Some(vb)) => {
            va.len() == vb.len()
                && va
                    .iter()
                    .zip(vb.iter())
                    .all(|(x, y)| std::sync::Arc::ptr_eq(x, y))
        }
        _ => false,
    }
}

/// Hash a canonicalized `Option<Vec<DescrRef>>` by its element ptr ids.
fn descr_set_hash<H: std::hash::Hasher>(s: &Option<Vec<DescrRef>>, state: &mut H) {
    use std::hash::Hash;
    match s {
        None => 0u8.hash(state),
        Some(v) => {
            1u8.hash(state);
            v.len().hash(state);
            for d in v {
                descr_ptr_id(d).hash(state);
            }
        }
    }
}

/// effectinfo.py: consider_struct(TYPE, fieldname)
///
/// In RPython this filters out `lltype.Void` fields and non-`GcStruct`
/// types, plus the `OBJECT.typeptr` special case. Pyre lacks `lltype`
/// metadata at this layer, so the predicate is intentionally permissive
/// (the codewriter has already filtered out non-GC fields earlier).
/// TODO.
pub fn consider_struct(_type_name: &str, _fieldname: &str) -> bool {
    true
}

/// effectinfo.py: consider_array(ARRAY)
///
/// Same caveat as `consider_struct`. TODO.
pub fn consider_array(_array_name: &str) -> bool {
    true
}

/// Side effect classification for calls.
///
/// effectinfo.py: `class EffectInfo`. The seven `EF_*` constants
/// and the `OS_*` family live as associated constants on this struct
/// plus the [`ExtraEffect`] / [`OopSpecIndex`] enums for type-safe match.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectInfo {
    pub extraeffect: ExtraEffect,
    pub oopspecindex: OopSpecIndex,
    /// pyre-only: identifies a pyre custom-bytecode helper `residual_call`
    /// (`binary_op` / `compare` / `load_const` / `box_int` / `store_subscr`
    /// / `load_global`) so the full-body walker can re-emit its speculative
    /// specialization.  Held OFF `oopspecindex` on purpose: these helpers have
    /// no upstream OS_* identity, so parking them on `oopspecindex` would make
    /// `has_oopspec()` true and divert the production reghint / vstring passes
    /// (both read `has_oopspec()` behaviorally) off the ordinary-call path, and
    /// would break the `_OS_CANRAISE` invariant for the CanRaise members.
    /// `None` for every ordinary call, so `has_oopspec()` and the OS_* universe
    /// stay identical to upstream.
    pub runtime_helper: RuntimeHelperKind,
    // ── effectinfo.py:128-145 raw descr sets ──
    //
    // PyPy stores `_readonly_descrs_fields: frozenset[Descr]` (and the
    // five sibling sets) at `EffectInfo.__init__` time
    // (`effectinfo.py frozenset_or_none`). `compute_bitstrings`
    // (`effectinfo.py`) later walks every EI, partitions the
    // descrs into (eisetr, eisetw) equivalence classes per category by
    // `id(descr)` (frozenset element identity), assigns
    // `descr.ei_index = class_index`, and only then encodes the
    // bitstring_* fields below.
    //
    // Pyre's lift carries `Vec<DescrRef>` (Arc identity) so the
    // partition step honours PyPy's object-identity semantic — two
    // distinct descrs with the same `descr.index()` (e.g. two structs'
    // `index_in_parent = 0` fields in `call.rs`) stay separate.
    // Each Vec is sorted-deduped by `Arc::as_ptr` at construction
    // (`canonicalize_descr_set`); `None` mirrors PyPy's
    // `_readonly_descrs_fields = None` wildcard (random-effects EI).
    /// effectinfo.py `_readonly_descrs_fields = frozenset_or_none(readonly_descrs_fields)`.
    #[serde(skip)]
    pub _readonly_descrs_fields: Option<Vec<DescrRef>>,
    /// effectinfo.py:131 `_write_descrs_fields`.
    #[serde(skip)]
    pub _write_descrs_fields: Option<Vec<DescrRef>>,
    /// effectinfo.py:129 `_readonly_descrs_arrays`.
    #[serde(skip)]
    pub _readonly_descrs_arrays: Option<Vec<DescrRef>>,
    /// effectinfo.py:132 `_write_descrs_arrays`.
    #[serde(skip)]
    pub _write_descrs_arrays: Option<Vec<DescrRef>>,
    /// effectinfo.py:130 `_readonly_descrs_interiorfields`.
    #[serde(skip)]
    pub _readonly_descrs_interiorfields: Option<Vec<DescrRef>>,
    /// effectinfo.py:133 `_write_descrs_interiorfields`.
    #[serde(skip)]
    pub _write_descrs_interiorfields: Option<Vec<DescrRef>>,
    /// `descr.py get_field_descr`, `descr.py get_array_descr`,
    /// `descr.py get_interiorfield_descr`: serialized channel for
    /// the six skipped raw descr sets above.
    ///
    /// Tracks the `Option`-ness of those sets exactly: `None` here **iff**
    /// they are the `EF_RANDOM_EFFECTS` wildcard, `Some` (possibly with
    /// empty vectors — [`DescrSetKeys::const_empty`]) whenever they are
    /// concrete.  `effectinfo.py:149-162` states that biconditional and
    /// `compute_bitstrings` relies on it, so the two must not drift apart.
    ///
    /// Pure derivation of the raw sets, so it is deliberately excluded from
    /// `PartialEq`, `Hash`, `LLType::func_key_with_release_gil_breaker`, and
    /// `EffectInfoKey::from_effect_info`; including it would split call-descr
    /// interning buckets on redundant data.
    #[serde(default)]
    pub descr_set_keys: Option<DescrSetKeysImage>,
    /// effectinfo.py:185 bitstring_readonly_descrs_fields. `None` = wildcard
    /// (effectinfo.py:488-489 sets the bitstring to `None` for `EF_RANDOM_EFFECTS`).
    pub readonly_descrs_fields: Option<Vec<u8>>,
    /// effectinfo.py:188 bitstring_write_descrs_fields.
    pub write_descrs_fields: Option<Vec<u8>>,
    /// effectinfo.py:186 bitstring_readonly_descrs_arrays.
    pub readonly_descrs_arrays: Option<Vec<u8>>,
    /// effectinfo.py:189 bitstring_write_descrs_arrays.
    pub write_descrs_arrays: Option<Vec<u8>>,
    /// effectinfo.py:187 bitstring_readonly_descrs_interiorfields.
    /// effectinfo.py:327-340: interiorfield reads also set array read bits.
    pub readonly_descrs_interiorfields: Option<Vec<u8>>,
    /// effectinfo.py:190 bitstring_write_descrs_interiorfields.
    pub write_descrs_interiorfields: Option<Vec<u8>>,
    /// effectinfo.py: can_invalidate
    pub can_invalidate: bool,
    /// effectinfo.py:194: can_collect — whether this call can trigger GC collection.
    /// RPython: set by collect_analyzer.analyze(op, self.seen_gc).
    pub can_collect: bool,
    /// effectinfo.py:201-206: single_write_descr_array
    #[serde(skip)]
    pub single_write_descr_array: Option<DescrRef>,
    /// effectinfo.py:196: extradescrs — extra descriptors carried by oopspec helpers
    /// (LIBFFI, ARRAYCOPY/ARRAYMOVE etc.).
    #[serde(skip)]
    pub extradescrs: Option<Vec<DescrRef>>,
    /// effectinfo.py:114, 197: call_release_gil_target = (target_fn_addr, save_err)
    /// `_NO_CALL_RELEASE_GIL_TARGET = (llmemory.NULL, 0)` by default.
    pub call_release_gil_target: (u64, i32),
}

/// Manual PartialEq: single_write_descr_array is excluded (like RPython's
/// cache key which also excludes it — effectinfo.py:155-164).
///
/// PyPy `effectinfo.py:152-164` keys the EI cache on the raw frozensets
/// (`(_readonly_descrs_fields, _write_descrs_fields, …)`); the bitstrings
/// don't exist yet at construction time, so they must NOT be part of
/// equality. We honour that here: identity is on `_*_descrs_*` (and
/// extraeffect / oopspec / can_invalidate / can_collect / release_gil),
/// not on the lazily-populated `bitstring_*` fields.
impl PartialEq for EffectInfo {
    fn eq(&self, other: &Self) -> bool {
        self.extraeffect == other.extraeffect
            && self.oopspecindex == other.oopspecindex
            && self.runtime_helper == other.runtime_helper
            && descr_set_eq(
                &self._readonly_descrs_fields,
                &other._readonly_descrs_fields,
            )
            && descr_set_eq(&self._write_descrs_fields, &other._write_descrs_fields)
            && descr_set_eq(
                &self._readonly_descrs_arrays,
                &other._readonly_descrs_arrays,
            )
            && descr_set_eq(&self._write_descrs_arrays, &other._write_descrs_arrays)
            && descr_set_eq(
                &self._readonly_descrs_interiorfields,
                &other._readonly_descrs_interiorfields,
            )
            && descr_set_eq(
                &self._write_descrs_interiorfields,
                &other._write_descrs_interiorfields,
            )
            && self.can_invalidate == other.can_invalidate
            && self.can_collect == other.can_collect
            && self.call_release_gil_target == other.call_release_gil_target
    }
}

impl Eq for EffectInfo {}

/// Manual Hash matching the PartialEq subset (skips
/// `single_write_descr_array` and `extradescrs` for the same reason
/// `effectinfo.py:155-164` excludes them from the cache key). Raw
/// descr sets are hashed by element `Arc::as_ptr` ids (Arc-identity
/// lift of PyPy's `id(descr)` frozenset hash).
impl std::hash::Hash for EffectInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.extraeffect.hash(state);
        self.oopspecindex.hash(state);
        self.runtime_helper.hash(state);
        descr_set_hash(&self._readonly_descrs_fields, state);
        descr_set_hash(&self._write_descrs_fields, state);
        descr_set_hash(&self._readonly_descrs_arrays, state);
        descr_set_hash(&self._write_descrs_arrays, state);
        descr_set_hash(&self._readonly_descrs_interiorfields, state);
        descr_set_hash(&self._write_descrs_interiorfields, state);
        self.can_invalidate.hash(state);
        self.can_collect.hash(state);
        self.call_release_gil_target.hash(state);
    }
}

impl Default for EffectInfo {
    fn default() -> Self {
        EffectInfo {
            extraeffect: ExtraEffect::CanRaise,
            oopspecindex: OopSpecIndex::None,
            runtime_helper: RuntimeHelperKind::None,
            // effectinfo.py frozenset_or_none: empty frozenset for
            // a non-random-effects EI with no field/array touches yet.
            _readonly_descrs_fields: Some(Vec::new()),
            _write_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            descr_set_keys: Some(DescrSetKeysImage::Empty),
            // effectinfo.py: empty frozenset for elidable, but `__new__`
            // requires a non-None value for non-RandomEffects EIs. Empty Vec
            // is the bitstring equivalent (no descrs touched).
            readonly_descrs_fields: Some(Vec::new()),
            write_descrs_fields: Some(Vec::new()),
            readonly_descrs_arrays: Some(Vec::new()),
            write_descrs_arrays: Some(Vec::new()),
            readonly_descrs_interiorfields: Some(Vec::new()),
            write_descrs_interiorfields: Some(Vec::new()),
            single_write_descr_array: None,
            extradescrs: None,
            // RPython effectinfo.py:125: can_collect=True default
            can_invalidate: false,
            can_collect: true,
            // effectinfo.py:114, 123: call_release_gil_target=_NO_CALL_RELEASE_GIL_TARGET
            call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
        }
    }
}

/// effectinfo.py:17-24: `EF_*` extraeffect constants.
///
/// TODO: the Rust enum is an adaptation over RPython's bare
/// integer constants for type safety. The `repr(u8)` discriminants are
/// the upstream values verbatim so ordering comparisons (`>`, `>=`)
/// behave identically to RPython.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ExtraEffect {
    /// effectinfo.py:17 `EF_ELIDABLE_CANNOT_RAISE = 0`
    ElidableCannotRaise = 0,
    /// effectinfo.py:18 `EF_LOOPINVARIANT = 1`
    LoopInvariant = 1,
    /// effectinfo.py:19 `EF_CANNOT_RAISE = 2`
    CannotRaise = 2,
    /// effectinfo.py:20 `EF_ELIDABLE_OR_MEMORYERROR = 3`
    ElidableOrMemoryError = 3,
    /// effectinfo.py:21 `EF_ELIDABLE_CAN_RAISE = 4`
    ElidableCanRaise = 4,
    /// effectinfo.py:22 `EF_CAN_RAISE = 5`
    CanRaise = 5,
    /// effectinfo.py `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE = 6`
    ForcesVirtualOrVirtualizable = 6,
    /// effectinfo.py:24 `EF_RANDOM_EFFECTS = 7`
    RandomEffects = 7,
}

/// effectinfo.py:27-105: `OS_*` oopspec index constants.
///
/// Rust enum + `repr(u16)` for type safety; values mirror upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
pub enum OopSpecIndex {
    None = 0,
    Arraycopy = 1,
    Str2Unicode = 2,
    ShrinkArray = 3,
    DictLookup = 4,
    ThreadlocalrefGet = 5,
    NotInTrace = 8,
    Arraymove = 9,
    IntPyDiv = 12,
    IntUdiv = 13,
    IntPyMod = 14,
    IntUmod = 15,
    StrConcat = 22,
    StrSlice = 23,
    StrEqual = 24,
    StreqSliceChecknull = 25,
    StreqSliceNonnull = 26,
    StreqSliceChar = 27,
    StreqNonnull = 28,
    StreqNonnullChar = 29,
    StreqChecknullChar = 30,
    StreqLengthok = 31,
    StrCmp = 32,
    UniConcat = 42,
    UniSlice = 43,
    UniEqual = 44,
    UnieqSliceChecknull = 45,
    UnieqSliceNonnull = 46,
    UnieqSliceChar = 47,
    UnieqNonnull = 48,
    UnieqNonnullChar = 49,
    UnieqChecknullChar = 50,
    UnieqLengthok = 51,
    UniCmp = 52,
    LibffiCall = 62,
    LlongInvert = 69,
    LlongAdd = 70,
    LlongSub = 71,
    LlongMul = 72,
    LlongLt = 73,
    LlongLe = 74,
    LlongEq = 75,
    LlongNe = 76,
    LlongGt = 77,
    LlongGe = 78,
    LlongAnd = 79,
    LlongOr = 80,
    LlongLshift = 81,
    LlongRshift = 82,
    LlongXor = 83,
    LlongFromInt = 84,
    LlongToInt = 85,
    LlongFromFloat = 86,
    LlongToFloat = 87,
    LlongUlt = 88,
    LlongUle = 89,
    LlongUgt = 90,
    LlongUge = 91,
    LlongUrshift = 92,
    LlongFromUint = 93,
    LlongUToFloat = 94,
    MathSqrt = 100,
    MathReadTimestamp = 101,
    RawMallocVarsizeChar = 110,
    RawFree = 111,
    StrCopyToRaw = 112,
    UniCopyToRaw = 113,
    JitForceVirtual = 120,
    JitForceVirtualizable = 121,
}

/// pyre-only recognition tag for custom-bytecode helper `residual_call`s.
///
/// The pyre codewriter lowers several Python bytecodes to a single runtime
/// helper `residual_call`: `binary_op` / `compare` carry the operator as a
/// compact int tag; `load_const` / `load_global` re-read an indexed code
/// slot; `box_int` wraps a raw Int; `store_subscr` does `obj[key] = value`.
/// The full-body walker re-emits each one's speculative specialization
/// (e.g. `guard_class` + `getfield_gc` + `int/float_OP` + `new_with_vtable`
/// for `binary_op`), but it cannot match the helper by fnaddr because
/// pyre-jit-trace does not depend on pyre-jit — so the calldescr's
/// [`EffectInfo`] carries this tag instead.
///
/// Deliberately separate from [`OopSpecIndex`]: these helpers have no
/// upstream OS_* identity, [`EffectInfo::has_oopspec`] must stay false for
/// them so the production reghint / vstring passes treat them as ordinary
/// calls, and the CanRaise members (`load_const` / `load_global` / `box_int`)
/// would otherwise violate the `_OS_CANRAISE` invariant (effectinfo.py).
/// Discriminants are pyre-internal (no upstream meaning) and variant order is
/// free: nothing decodes the tag numerically, and the one bincode artefact that
/// carries it (`descrs.bin`) is written and read back within a single build
/// whose codegen cache is content-keyed on this file.  Group a new helper with
/// its siblings rather than appending it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum RuntimeHelperKind {
    #[default]
    None,
    BinaryOp,
    CompareOp,
    LoadConst,
    BoxInt,
    StoreSubscr,
    LoadGlobal,
    /// `bh_load_name_fn(frame, w_name, namei)` — the LOAD_NAME frame-receiver
    /// helper (`pyopcode.py`, w_locals probe + LOAD_GLOBAL fallback).
    /// The full-body walker recognises this tag to fold a module-scope name
    /// read (frame's `w_locals_object` is null, so `w_locals` aliases
    /// `w_globals`) through the same module-dict cell fast path as
    /// [`RuntimeHelperKind::LoadGlobal`], eliding the per-iteration residual.
    LoadName,
    /// `bh_store_name_fn(frame, w_name, value)` — the STORE_NAME frame-receiver
    /// helper (`pyopcode.py`, delegates to `store_name_value` →
    /// `setitem_str` → `_setitem_str_cell_known` → `typeobject.py:53-71
    /// write_cell`).  The full-body walker recognises this tag to fold a
    /// module-scope int store whose slot holds an `IntMutableCell` (the
    /// stabilised in-place shape) to a `setfield_gc_i(cell, intvalue)`,
    /// eliding the boxing + residual dict setitem — the store dual of
    /// [`RuntimeHelperKind::LoadName`].
    StoreName,
    /// `bh_store_global_fn(frame, w_name, value)` — the STORE_GLOBAL
    /// frame-receiver helper (`pyopcode.py`, delegates to
    /// `store_global_value`).  Recognised for the same `IntMutableCell`
    /// in-place store fold as [`RuntimeHelperKind::StoreName`].
    StoreGlobal,
    /// `bh_delete_name_fn(frame, w_name)` — the DELETE_NAME frame-receiver
    /// helper (`pyopcode.py`, delegates to `delitem_str`).  Recognised
    /// by the walker only to answer upstream's
    /// `opimpl_jit_force_quasi_immutable` question (`pyjitpl.py`)
    /// ahead of the call: a successful delete runs `mutated()`
    /// (`celldict.py`), which assigns the `version?` quasi-immutable
    /// field.  Pyre's write lives inside this residual, so the walker never
    /// meets the rtyper operation and has to ask before executing.
    DeleteName,
    /// `bh_delete_global_fn(frame, w_name)` — the DELETE_GLOBAL twin of
    /// [`RuntimeHelperKind::DeleteName`], recognised for the same reason.
    DeleteGlobal,
    /// `bh_load_locals_fn(frame)` — the LOAD_LOCALS frame-receiver helper
    /// (`pyopcode.py`). Carries no fold; the tag exists so the class
    /// body's namespace read is a residual call like every other frame
    /// method, rather than an untranslatable op.
    LoadLocals,
    /// `bh_load_build_class_fn(frame)` — the LOAD_BUILD_CLASS frame-receiver
    /// helper (`pyopcode.py`). Same standing as
    /// [`RuntimeHelperKind::LoadLocals`].
    LoadBuildClass,
    /// `bh_load_import_fn(frame)` — the builtin lookup half of IMPORT_NAME.
    /// The following invocation is emitted through [`RuntimeHelperKind::CallFn`]
    /// so gateway builtins retain their ordinary meta-traceable call shape.
    LoadImport,
    /// `bh_load_import_locals_fn(frame)` — IMPORT_NAME's locals argument
    /// (`pyopcode.py`'s `IMPORT_NAME`).  Infallible, same standing as
    /// [`RuntimeHelperKind::LoadLocals`].
    LoadImportLocals,
    /// `bh_load_import_globals_fn(frame)` — IMPORT_NAME's globals argument,
    /// `pyopcode.py`'s `self.get_w_globals()`.  The frame reads its own
    /// namespace, so an inlined callee never inherits the caller's; infallible,
    /// same standing as [`RuntimeHelperKind::LoadImportLocals`].
    LoadImportGlobals,
    /// `bh_call_fn_N(callable, null_or_self, args...)` — the CALL-family
    /// Python-call helper.  `null_or_self` (arg index 1) is a sentinel
    /// the helper checks before use (a non-null receiver is prepended as
    /// arg0; `PY_NULL` means "no receiver" and is never dereferenced), so
    /// a concrete-NULL there is the NORMAL plain-call shape, not the
    /// broken baked-NULL-globals shape the walker's may-force NULL-ref
    /// gate rejects.
    CallFn,
    /// `bh_truth_fn(value)` — the TO_BOOL / `POP_JUMP_IF_*` truth helper
    /// (`opcode_ops::truth_value`).  Returns the operand's truthiness as a
    /// raw Int.  The full-body walker recognises this tag to specialise a
    /// provably-int operand to a pure `guard_class INT` + `getfield intval`
    /// + `int_is_true`, eliding the `CALL_MAY_FORCE` (and its
    ///   `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION`) the generic residual emits.
    Truth,
    /// `bh_unary_positive_fn(value)` — the CALL_INTRINSIC_1 UNARY_POSITIVE
    /// helper (`opcode_ops::unary_positive_value` → `space.pos`).  `int.__pos__`
    /// returns an exact int unchanged, so the full-body walker recognises this
    /// tag to fold a provably-int operand to the operand itself behind a
    /// `guard_class INT`, eliding the `CALL_MAY_FORCE` (and its
    /// `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION`) the generic residual emits.
    /// A `bool` (`+True` is int `1`, not identity) and every non-int operand
    /// fall through to the generic residual so their `__pos__` still runs.
    UnaryPositive,
    /// `bh_unary_negative_fn(value)` — the `UNARY_NEGATIVE` helper
    /// (`opcode_ops::unary_negative_value` → `space.neg`).  The full-body
    /// walker folds a provably-exact-int operand to `IntSubOvf(0, value)`
    /// behind a `guard_class INT`; a bool / subclass / non-int operand, and an
    /// `INT_MIN` value (whose negation is the `2**63` long), fall through to
    /// the generic residual.
    UnaryNegative,
    /// `bh_unary_invert_fn(value)` — the `UNARY_INVERT` helper
    /// (`opcode_ops::unary_invert_value` → `space.invert`).  The full-body
    /// walker folds a provably-exact-int operand to `IntInvert(value)` behind a
    /// `guard_class INT` (`~x` always fits an int, so no overflow guard); a
    /// bool / subclass / non-int operand falls through to the generic residual.
    UnaryInvert,
    /// `bh_unary_not_fn(value)` — the `UNARY_NOT` helper (`space.not_` →
    /// `w_bool_from(!is_true(value))`).  The full-body walker folds a
    /// provably-exact-int or exact-bool operand to `int_is_true` + negate +
    /// `newbool`, eliding the `CALL_MAY_FORCE`.  A subclass / non-int operand
    /// falls through so a user `__bool__` / `__len__` still runs.
    UnaryNot,
    /// `bh_format_simple_fn(value)` — the FORMAT_SIMPLE helper
    /// (`runtime_ops::format_value` with the empty spec).  A user
    /// `__format__` runs Python here, so the generic residual is opaque for
    /// the whole method body.
    ///
    /// The walker folds an exact `int` to `ll_int2dec` + `newutf8` and an
    /// exact `str` to identity — the empty-spec arms of `format_w`.  A bool,
    /// subclass, or Python `__format__` stays residual.  [`FormatWithSpec`]
    /// inlines a Python `__format__` when a spec operand is present.
    FormatSimple,
    /// `bh_format_with_spec_fn(value, spec)` — the FORMAT_WITH_SPEC helper,
    /// the two-operand sibling of [`RuntimeHelperKind::FormatSimple`] carrying
    /// the f-string's format spec.  Recognised for the same receiver-pinned
    /// `__format__` inline; `spec` is threaded to the callee as its second
    /// parameter.
    ///
    /// The walker also folds an exact `int` plus a constant decimal spec
    /// (`:d` / `:05d` / `:5d`) to `ll_int2dec` + optional `ll_strconcat` pad
    /// + residual `newutf8` — `newformat.py` `format_int_or_long` /
    /// `_int_to_base` / `_fill_number`.  A bool, subclass, sign-interior
    /// pad, or Python `__format__` stays residual or takes the inline.
    FormatWithSpec,
    /// `bh_unpack_sequence_fn(count, seq)` — the UNPACK_SEQUENCE validator
    /// emitted by the codewriter UNPACK_SEQUENCE arm.  Validates the exact
    /// length and returns the normalized tuple.  The full-body walker
    /// recognises this tag to fold an arity-2 specialised int tuple to a
    /// `guard_class` + pass-through, eliding the opaque residual so the
    /// partner [`RuntimeHelperKind::UnpackItem`] reads fold against it.
    UnpackSequence,
    /// `bh_unpack_item_fn(index, tuple)` — the per-index UNPACK_SEQUENCE item
    /// reader.  The full-body walker recognises this tag to read `value0` /
    /// `value1` from a specialised int tuple directly (`getfield_gc_pure_i`
    /// + `wrapint`), so the unpacked items stay unboxed ints through the
    ///   downstream BINARY_OP int fold.
    UnpackItem,
    /// `bh_newtuple_from_array(array)` — the BUILD_TUPLE array consumer that
    /// `lower_tuple_build_hlop_to_insn` emits after `new_array_clear` +
    /// `setarrayitem_gc`.  The full-body walker recognises this tag to
    /// virtualize an arity-2 plain-int tuple as a `spec_ii`
    /// `new_with_vtable` + `value0` / `value1` (mirroring the retired
    /// trace-side tuple builder), so the backing array build and the
    /// partner [`RuntimeHelperKind::UnpackItem`] reads DCE to a pure-int loop.
    NewtupleFromArray,
    /// `newlist_from_array(array)` — the BUILD_LIST array consumer that
    /// `lower_tuple_build_hlop_to_insn` emits after `new_array_clear` +
    /// `setarrayitem_gc`.  The full-body walker recognises this tag to
    /// decompose the list into the virtualizable `opimpl_newlist` shape
    /// (`pyjitpl.py`) — `new_with_vtable` + `new_array` +
    /// `setarrayitem_gc` + `setfield_gc` — choosing the storage strategy
    /// from the concrete element shadows the way `w_list_new` /
    /// `list_strategy_for` does at runtime, so the array build and the
    /// residual DCE when the list never escapes.  Unlike
    /// [`RuntimeHelperKind::NewtupleFromArray`], the element boxes are
    /// recovered from the backing array (const length + per-index element
    /// shadows) rather than from residual args.
    NewlistFromArray,
    /// `bh_build_map_from_array(array)` over the zero-length array BUILD_MAP 0
    /// emits — `space.newdict()` with no `space.setitem` after it
    /// (`pyopcode.py BUILD_MAP`).  No key is hashed, so the call allocates an
    /// empty dict and does nothing else.
    NewEmptyDict,
    /// `bh_build_string_from_array(array)` — the BUILD_STRING array consumer
    /// (`pyopcode.py BUILD_STRING`).  Fragments are already strings
    /// (FORMAT_* / CONVERT_VALUE ran first).  The walker recovers them
    /// from the backing-array heap-cache and left-folds `descr_add`
    /// (`jit_str_concat`), the same channel as `BINARY_OP ADD` of two
    /// exact `str`s.  Interpreter/residual construction is
    /// `Utf8StringBuilder`; the walker keeps this fold until
    /// `StringBuilder` virtualizes.
    BuildStringFromArray,
    /// `n_varargs_fn(frame, exc, cause)` — the RAISE-family residual the
    /// codewriter emits for `n argc>=1` (`build_n_varargs_fn_residual_call_r_r_insn`).
    /// `cause` (the trailing Ref arg) is a `PY_NULL` sentinel for `raise X`
    /// without `from` (checked, never dereferenced when null), so a concrete
    /// NULL there is the normal shape — not the broken baked-NULL shape the
    /// walker's may-force NULL-ref gate rejects.  The full-body walker
    /// recognises this tag to exempt the trailing
    /// NULL `cause` in both twin NULL-ref guards so the FBW path can own the
    /// raise instead of declining to the trait.
    RaiseVarargs,
    /// `get_current_exception()` — the read a catch-covered bare `raise` uses
    /// to obtain the exception it re-raises (`() → Ref`, TLS read via
    /// `cpu.get_current_exception_fn`).  The full-body walker recognises this
    /// tag to lower it to `GETFIELD_GC_R(ec, sys_exc_value)`.
    GetCurrentException,
    /// `current_exception_or_none()` — the PUSH_EXC_INFO `prev` save residual:
    /// `ec.sys_exc_value`, or the prebuilt `None` when it is empty (`() → Ref`,
    /// TLS read via `cpu.current_exception_or_none_fn`).  The save is followed
    /// by a store and a matching POP_EXCEPT restore.  The full-body walker
    /// recognises this tag to lower it to `GETFIELD_GC_R(ec, sys_exc_value)`
    /// plus a nullity guard, so the exc-info save participates in the balanced
    /// save/restore the heap optimizer dead-store-eliminates, letting a
    /// non-escaping exception stay virtual.
    CurrentExceptionOrNone,
    /// `set_current_exception(exc)` — the PUSH_EXC_INFO store and the
    /// POP_EXCEPT restore residual (`(exc: Ref) → Void`, TLS write via
    /// `cpu.set_current_exception_fn`, which clears the slot for `None`).  The
    /// full-body walker recognises this tag to lower it to `SETFIELD_GC(ec, exc,
    /// sys_exc_value)`; paired with the [`CurrentExceptionOrNone`] save on the
    /// same descr-identity field, a balanced never-read save/restore is
    /// dead-store-eliminated so the virtual exception de-escapes and DCEs.
    SetCurrentException,
    /// `for_iter_next(iter)` — the FOR_ITER advance residual the codewriter
    /// emits for the continue arm (`jit_next` via `cpu.for_iter_next_fn`).
    /// It advances the real shared heap iterator concretely during the
    /// authoritative walk and returns the consumed item (`Ref`, or null at
    /// exhaustion).  The full-body walker recognises this tag to stash the
    /// consumed item + the FOR_ITER body pc so an aborting walk can DELIVER
    /// the in-flight iteration to the live frame instead of dropping it (the
    /// iterator advance is an irreversible side effect with no journal undo).
    ForIterNext,
    /// `jit_exception_match(exc, match_class)` — the infallible Python-level
    /// exception MRO test used by FOR_ITER's materialized catch arm.
    ForIterExceptionMatch,
    /// `get_iter(obj)` — the GET_ITER residual (`iter(obj)`).  The full-body
    /// walker recognises exact machine-word `range` objects and emits the
    /// virtual `W_IntRangeIterator` allocation shape directly.
    GetIter,
    /// `store_deref_value(cell, value)` — the STORE_DEREF residual
    /// (`bh_store_deref_value_fn` via `cpu.store_deref_value_fn`).  It mutates
    /// the cell's contents in place and RETURNS the slot value (`Ref`), so it
    /// is a value-returning heap write the #57 Option C body-effect guard's
    /// `Void`-result write proxy cannot see; the tag lets that guard flag it
    /// (Finding #1) so an aborting FOR_ITER walk refuses delivery rather than
    /// re-running the body and doubling the cell write.
    StoreDeref,
    /// `load_deref_value(cell, code, deref_idx)` — the LOAD_DEREF read residual
    /// (`bh_load_deref_value_fn` via `cpu.load_deref_value_fn`).  The full-body
    /// walker recognises this tag to reproduce `nestedscope.py Cell.get`:
    /// a cell the trace holds as a constant whose family has never seen a
    /// bound-to-bound rebinding folds to its contents under a
    /// `QUASIIMMUT_FIELD(family, ever_mutated)`, and every other shape keeps the
    /// residual.  Recognition only — this tag confers no replay-safety
    /// standing, so an unfolded read still classifies the enclosing body dirty.
    LoadDeref,
    /// `jit_list_append(list, value)` — the LIST_APPEND residual the
    /// codewriter emits for a comprehension append (`[f(x) for x in xs]`
    /// inlines LIST_APPEND into the enclosing function).  The two Ref
    /// operands are the peeked target list and the popped value; the result
    /// is void.  The full-body walker recognises this tag to fold the append
    /// through the same orthodox `w_list_append` descent as the method-call
    /// form ([`RuntimeHelperKind::CallFn`] `lst.append(x)`), recording the
    /// native array store instead of the opaque `jit_list_append` residual.
    /// The recognition has no bound-method callable to pin: the list and
    /// value arrive directly as the residual's Ref operands.
    ListAppendValue,
    /// `bh_call_function_ex_fn(callable, self_or_null, starargs,
    /// kwargs_or_null)` — the CALL_FUNCTION_EX residual.  Both `self_or_null`
    /// (arg index 1) and `kwargs_or_null` (arg index 3) are `PY_NULL`
    /// sentinels the helper checks before use (`self_or_null` is prepended
    /// as arg0 only when non-null; a null `kwargs_or_null` skips the `**`
    /// merge), so a concrete-NULL there is the normal shape — not the broken
    /// baked-NULL-globals shape the walker's may-force NULL-ref gate rejects.
    CallFunctionEx,
    /// `bh_call_kw_<n>(callable, null_or_self, kwnames, arg0..arg{n-1})` —
    /// the CALL_KW residual.  `null_or_self` (arg index 1) is a `PY_NULL`
    /// sentinel the helper checks before use (prepended as arg0 only when
    /// non-null), so a concrete-NULL there is the normal shape — not the
    /// broken baked-NULL-globals shape the walker's may-force NULL-ref gate
    /// rejects.  `kwnames` (arg index 2) is the constant kwnames tuple and
    /// is always a live Ref.
    CallKw,
    /// `bh_with_except_start_fn(exit_func, exit_self, val)` — the
    /// WITH_EXCEPT_START residual.  `exit_self` (arg index 1) is a `PY_NULL`
    /// sentinel the helper checks before use: `with_except_start_values`
    /// prepends it as arg0 only when non-null, exactly as `LOAD_SPECIAL`
    /// pushes an already-bound `__exit__` beside a NULL receiver slot.  A
    /// concrete-NULL there is therefore the normal `with` shape — not the
    /// broken baked-NULL-globals shape the walker's may-force NULL-ref gate
    /// rejects.
    WithExceptStart,
    /// `load_attr_fn(obj, code, name_idx)` — the plain (non-method) LOAD_ATTR
    /// residual (`lower_getattr_hlop_to_insn` → `space.getattr`).  The Ref
    /// operands are the receiver and the jitcode's own PyCode; the Int operand
    /// is the co_names index.  The full-body walker recognises this tag to fold
    /// a monomorphic instance-attribute read (`mapdict.py
    /// LOAD_ATTR_caching`) to `guard_class(obj, w_type)` +
    /// `guard_value(getfield(obj, map), map)` + `getfield(obj, storage)` +
    /// `getarrayitem_gc_r(block, C_storageindex)` (the inline read
    /// `mapdict.py _mapdict_read_storage`), eliding the opaque
    /// `CALL_MAY_FORCE` MRO-walk residual.  An unboxed slot instead folds to a
    /// non-forcing raw read of its longlong list (`mapdict.py
    /// _prim_direct_read`) with the boxing left in the trace.  Falls through to
    /// the residual for every shape neither read covers (non-instance receiver,
    /// custom `__getattribute__`, data descriptor, attribute absent from this
    /// instance's map, and an unboxed slot whose class has frozen unboxing,
    /// whose read owes a migration the folded form cannot perform).
    LoadAttr,
    /// `bh_setattr_fn(obj, value, code, name_idx)` — the plain STORE_ATTR
    /// residual (`lower_setattr_hlop_to_insn` → `space.setattr`).  The Ref
    /// operands are the receiver, value, and jitcode's own PyCode; the Int
    /// operand is the co_names index.  The full-body walker recognises this
    /// tag to fold a same-type store to a non-forcing write: an unboxed int or
    /// float slot takes a raw longlong-list write, a boxed slot a direct
    /// storage write.  Every unsupported shape keeps the original generic
    /// setattr residual.
    StoreAttr,
    /// Headerless two-word node allocation, `(value, link)`, from the
    /// interpreter's own nursery.  The dynasm backend recognises this tag on
    /// the CallR descr to emit an inline nursery bump (RPython malloc_cond
    /// shape) instead of a full residual call; the node stays a real escaping
    /// alloc result (NOT a virtualized New), so the setfield that links it in
    /// remains a normal committed setfield. has_oopspec stays false.
    NurseryAlloc,
    /// `bh_load_method_self_fn(obj, attr, code, name_idx)` — the LOAD_METHOD
    /// binding half (`callmethod.py`).  The full-body walker recognises
    /// this tag to fold the pure `compute_load_method_bound` decision once the
    /// paired [`LoadAttr`] method-cache fold has made `attr` concrete.
    LoadMethodSelf,
    /// `load_super_attr(global_super, self, cls, frame, code, name_idx,
    /// is_two_arg)` — the LOAD_SUPER_ATTR residual (`bh_load_super_attr_fn`).
    /// It builds the super proxy and resolves the attribute through the proxy's
    /// `__getattribute__`, whose descriptor `__get__` may run Python, so it has
    /// the same standing as [`RuntimeHelperKind::LoadAttr`]: a body carrying one
    /// is DEFERRED rather than proven clean, and an unfolded residual still
    /// reaches the nested-residual backstop.
    LoadSuperAttr,
    /// `super_attr_unwrap(raw, which)` — the LOAD_SUPER_ATTR method-form split
    /// (`bh_super_attr_unwrap_fn`).  A total function of its argument: an
    /// `is_method` class test plus one of two field reads, no allocation, no
    /// raise, no Python.  Reading a field twice commits nothing, so unlike its
    /// producer this one is replay-SAFE, the standing
    /// [`RuntimeHelperKind::LoadConst`] and [`RuntimeHelperKind::LoadGlobal`]
    /// carry.
    SuperAttrUnwrap,
    /// `jit_bare_super_from_frame(frame)` — the target the walker substitutes
    /// for a traced zero-argument `super()` that arrives as a call rather than
    /// as LOAD_SUPER_ATTR (`try_walker_specialize_bare_super_call`).  It is the
    /// `descriptor.py _super_from_frame` half `bh_load_super_attr_fn` already
    /// calls, named directly so the frame travels as an operand; the generic
    /// `bh_call_fn` it replaces reaches `builtin_super`'s zero-argument tail,
    /// whose `gettopframe()` forces the traced virtualizable from inside an
    /// opaque residual and aborts the walk.
    ///
    /// It RETURNS the proxy (`Ref`) while `_super_check` may run a `__class__`
    /// property, i.e. arbitrary Python, so it is a value-returning call the
    /// `Void`-result write proxy cannot see — the same standing as
    /// [`RuntimeHelperKind::SetAddMethod`], and the reason the tag exists.
    /// Without it the substitution would replace a `CallFn` the
    /// `writes_live_heap` discriminator counts with a descr it does not, and a
    /// nested abort would rewind past a property that has already run.
    BareSuperFromFrame,
    /// `bh_delete_attr_fn(obj, code, name_idx)` — the plain DELETE_ATTR
    /// residual (`lower_delete_attr_hlop_to_insn` → `space.delattr`), the
    /// deletion counterpart of [`StoreAttr`](RuntimeHelperKind::StoreAttr).
    /// The full-body walker recognises this tag to fold the deterministic
    /// immutable-type raise (`del int.x` → TypeError before any dict
    /// access) to a traced inline exception construction the optimizer
    /// can virtualize; every other shape keeps the generic residual.
    DeleteAttr,
    /// `jit_make_function_from_globals(globals, code)` — the MAKE_FUNCTION
    /// residual (`lower_make_function_hlop_to_insn` →
    /// `function.py Function.__init__`).  Both Ref operands are baked
    /// constants: the defining frame's globals object and the popped code
    /// object.  The full-body walker recognises this tag to emit the
    /// construction as `NewWithVtable` + the `SetfieldGc` set, so a `def`
    /// inside a loop virtualizes away instead of allocating a `Function` per
    /// iteration through an opaque residual.
    MakeFunction,
    /// `jit_set_function_attribute(func, attr, flag)` — the
    /// SET_FUNCTION_ATTRIBUTE residual
    /// (`lower_set_function_attribute_hlop_to_insn` → `function.py`'s typed
    /// setters), which stamps one slot on the function MAKE_FUNCTION just
    /// built and hands the same object back.  The full-body walker recognises
    /// this tag to emit the single-slot arms as a `SetfieldGc` on that
    /// allocation and pass the operand through as the result, so the
    /// definition sequence stays one virtual instead of escaping through an
    /// opaque call whose result has to be re-read and re-guarded.
    SetFunctionAttribute,
    /// `jit_set_add_method(set, value)` — the target the walker substitutes for
    /// a traced `s.add(x)` on a `set` (`try_walker_specialize_set_add_method`).
    /// It inserts into the live set and RETURNS `None` (`Ref`), so it is a
    /// value-returning heap write the `Void`-result write proxy cannot see —
    /// the same standing as [`RuntimeHelperKind::StoreDeref`], and the reason the
    /// tag exists: without it the substitution would replace a `CallFn` the
    /// `writes_live_heap` discriminator counts with a descr it does not, and a
    /// nested abort would rewind past a completed insert.  The element's
    /// `__hash__` is not necessarily Python — a cpyext `tp_hash` runs native
    /// code and moves no frame-entry odometer — so the frame-entry signal
    /// cannot stand in for this one.
    SetAddMethod,
    /// `bh_clear_in_flight_exception()` — the `[] -> void` residual emitted by
    /// PUSH_EXC_INFO to complete the caught-exception ownership transfer.  The
    /// full-body walker applies the concrete clear during its authoritative
    /// walk and emits no runtime IR: compiled traceback construction never
    /// publishes this interpreter-only GC-liveness carrier.  Generic fallback
    /// also keeps the executed-effect odometer off it because the slot has no
    /// value reader and a non-committing walk has nothing to undo.
    ClearInFlightException,
    /// `load_special_fn(obj, method_kind)` — the LOAD_SPECIAL Enter/Exit
    /// residual (`lower_load_special_hlop_to_insn`).  The Ref operand is the
    /// context-manager object and the Int operand is the `SpecialMethod`
    /// discriminant naming `__enter__` / `__exit__`.  The full-body walker
    /// recognises this tag to fold the lookup the way PyPy's `BEFORE_WITH`
    /// gets it for free: `space.lookup` is ordinary RPython there, so under a
    /// promoted type it is constant and the bound method it builds virtualizes
    /// into the following CALL.  Falls through to the residual for every shape
    /// [`pyre_interpreter::baseobjspace::load_special_fast_path`] declines.
    LoadSpecial,
    /// `load_fast_check_fn(value, code, name_idx)` — the LOAD_FAST_CHECK
    /// residual (`lower_load_fast_check_hlop_to_insn`).  The Ref operands are
    /// the local's value and the jitcode's own PyCode; the Int operand is the
    /// `co_varnames` index the `UnboundLocalError` names.  The helper is the
    /// identity on a bound local, so the full-body walker recognises this tag
    /// to fold it to `guard_nonnull(value)` plus the operand itself — the
    /// nullity test upstream spells with a guard rather than a call
    /// (`pyjitpl.py _establish_nullity`).  A guard failure resumes at the
    /// opcode, where the interpreter re-runs it and raises.
    LoadFastCheck,
    /// `unbound_local_error_fn(code, name_idx)` — the unbound arm of
    /// DELETE_FAST (`pyopcode.py DELETE_FAST`, the `oefmt` under
    /// `if self.locals_cells_stack_w[varindex] is None`).  It builds the
    /// `UnboundLocalError` value from the code object's immutable
    /// `co_varnames` and returns it for the `raise` that follows: nothing is
    /// published and no user code runs.
    UnboundLocalError,
    /// `bh_convert_value_fn(value, conv)` — the CONVERT_VALUE helper
    /// (`runtime_ops::convert_value`: `!s`/`!r`/`!a`).  A user `__str__` /
    /// `__repr__` runs Python here, so the generic residual is opaque.
    ///
    /// The walker folds an exact `int` to `ll_int2dec` + `newutf8`
    /// (`intobject.py` `descr_str` / `descr_repr`, which share a body) and
    /// an exact `str` `!s` to identity (`unicodeobject.py` `descr_str`).
    /// A bool, subclass, or Python `__str__` / `__repr__` stays residual.
    ConvertValue,
    /// `bh_binary_slice_fn(obj, start, stop)` — BINARY_SLICE.  The walker
    /// folds an exact `str` plus exact-int / `None` bounds to
    /// `_unicode_sliced` (`ll_int2dec` sibling: elidable cut + residual
    /// wrap).  A custom `__index__`, subclass, or non-str stays residual.
    BinarySlice,
}

impl EffectInfo {
    pub fn check_is_elidable(&self) -> bool {
        self.extraeffect == ExtraEffect::ElidableCanRaise
            || self.extraeffect == ExtraEffect::ElidableOrMemoryError
            || self.extraeffect == ExtraEffect::ElidableCannotRaise
    }

    pub fn check_can_invalidate(&self) -> bool {
        self.can_invalidate
    }

    pub fn check_can_collect(&self) -> bool {
        self.can_collect
    }

    pub fn check_forces_virtual_or_virtualizable(&self) -> bool {
        self.extraeffect >= ExtraEffect::ForcesVirtualOrVirtualizable
    }

    /// `effectinfo.py` `has_random_effects(self)`:
    ///
    /// ```python
    /// def has_random_effects(self):
    ///     return self.extraeffect >= self.EF_RANDOM_EFFECTS
    /// ```
    ///
    /// Uses `>=` (not `==`).  `EF_RANDOM_EFFECTS = 7` is the highest
    /// ExtraEffect numeric value today, so the two operators pick the
    /// same set; the structural parity matters because future upstream
    /// additions that slot a value after `EF_RANDOM_EFFECTS` would need
    /// this predicate to remain inclusive.
    pub fn has_random_effects(&self) -> bool {
        self.extraeffect >= ExtraEffect::RandomEffects
    }

    /// Whether the oopspec identifies a special-cased operation.
    pub fn has_oopspec(&self) -> bool {
        self.oopspecindex != OopSpecIndex::None
    }

    /// effectinfo.py: check_can_raise(ignore_memoryerror)
    pub fn check_can_raise(&self, ignore_memoryerror: bool) -> bool {
        if ignore_memoryerror {
            self.extraeffect > ExtraEffect::ElidableOrMemoryError
        } else {
            self.extraeffect > ExtraEffect::CannotRaise
        }
    }

    /// effectinfo.py: is_call_release_gil()
    /// `tgt_func, tgt_saveerr = self.call_release_gil_target; return bool(tgt_func)`
    pub fn is_call_release_gil(&self) -> bool {
        let (tgt_func, _tgt_saveerr) = self.call_release_gil_target;
        tgt_func != 0
    }

    /// Const-compatible constructor for static initialization.
    ///
    /// Empty bitstrings (`Some(Vec::new())`) match `effectinfo.py:175-181`
    /// for elidable EIs whose `_write_descrs_*` collapse to `frozenset()`,
    /// later compiled to a zero-length bitstring by `compute_bitstrings`.
    pub const fn const_new(extraeffect: ExtraEffect, oopspecindex: OopSpecIndex) -> Self {
        EffectInfo {
            extraeffect,
            oopspecindex,
            runtime_helper: RuntimeHelperKind::None,
            _readonly_descrs_fields: Some(Vec::new()),
            _write_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            descr_set_keys: Some(DescrSetKeysImage::Empty),
            readonly_descrs_fields: Some(Vec::new()),
            write_descrs_fields: Some(Vec::new()),
            readonly_descrs_arrays: Some(Vec::new()),
            write_descrs_arrays: Some(Vec::new()),
            readonly_descrs_interiorfields: Some(Vec::new()),
            write_descrs_interiorfields: Some(Vec::new()),
            single_write_descr_array: None,
            extradescrs: None,
            can_invalidate: false,
            can_collect: true,
            call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
        }
    }

    /// Create a new EffectInfo with the given effect and oopspec.
    pub fn new(extraeffect: ExtraEffect, oopspecindex: OopSpecIndex) -> Self {
        EffectInfo {
            extraeffect,
            oopspecindex,
            ..Default::default()
        }
    }

    /// effectinfo.py:64: _OS_offset_uni = OS_UNI_CONCAT - OS_STR_CONCAT
    pub const _OS_OFFSET_UNI: u16 = OopSpecIndex::UniConcat as u16 - OopSpecIndex::StrConcat as u16;

    /// effectinfo.py:114: _NO_CALL_RELEASE_GIL_TARGET = (llmemory.NULL, 0)
    pub const _NO_CALL_RELEASE_GIL_TARGET: (u64, i32) = (0, 0);

    /// effectinfo.py:108-112: _OS_CANRAISE — oopspecindex values that can raise
    /// even when extraeffect <= EF_CANNOT_RAISE.
    pub fn _is_os_canraise(idx: OopSpecIndex) -> bool {
        matches!(
            idx,
            OopSpecIndex::None
                | OopSpecIndex::Str2Unicode
                | OopSpecIndex::LibffiCall
                | OopSpecIndex::RawMallocVarsizeChar
                | OopSpecIndex::JitForceVirtual
                | OopSpecIndex::ShrinkArray
                | OopSpecIndex::DictLookup
                | OopSpecIndex::NotInTrace
        )
    }

    /// effectinfo.py:271-273: MOST_GENERAL
    /// `EffectInfo(None, None, None, None, None, None, EF_RANDOM_EFFECTS, can_invalidate=True)`.
    ///
    /// `effectinfo.py` + `compute_bitstrings` line 488-489 keep
    /// the bitstrings as `None` for `EF_RANDOM_EFFECTS`; the optimizer's
    /// `has_random_effects()` guard (heap.py / heap.rs) prevents
    /// `check_*_descr_*` from being called on the wildcard.
    pub const MOST_GENERAL: EffectInfo = EffectInfo {
        extraeffect: ExtraEffect::RandomEffects,
        oopspecindex: OopSpecIndex::None,
        runtime_helper: RuntimeHelperKind::None,
        _readonly_descrs_fields: None,
        _write_descrs_fields: None,
        _readonly_descrs_arrays: None,
        _write_descrs_arrays: None,
        _readonly_descrs_interiorfields: None,
        _write_descrs_interiorfields: None,
        descr_set_keys: None,
        readonly_descrs_fields: None,
        write_descrs_fields: None,
        readonly_descrs_arrays: None,
        write_descrs_arrays: None,
        readonly_descrs_interiorfields: None,
        write_descrs_interiorfields: None,
        single_write_descr_array: None,
        extradescrs: None,
        can_invalidate: true,
        can_collect: true,
        call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
    };

    // ── Bitstring check methods (effectinfo.py check_readonly_descr_field parity) ──
    //
    // PyPy `effectinfo.py check_readonly_descr_field` does NOT short-circuit on
    // `EF_RANDOM_EFFECTS`: each `check_*` is a one-liner
    // `bitstring.bitcheck(self.bitstring_*, ei_index)`.  The
    // `EF_RANDOM_EFFECTS` case is handled at construction time —
    // `effectinfo.py:149-156` keeps the `_readonly_*`/`_write_*` sets as
    // `None` and `compute_bitstrings` (`effectinfo.py`) sets the
    // bitstring fields to `None` too.  The contract is that callers must
    // gate via `has_random_effects()` first (heap.py) so the
    // `None`-bitstring case is never queried.
    //
    // Pyre `MOST_GENERAL` (line 380) populates every bitset with `None`,
    // matching `effectinfo.py MOST_GENERAL`.  The optimizer
    // caller (`heap.rs`'s `call_has_random_effects`) gates the same
    // way as PyPy, so the `None`-bitstring case must never be queried;
    // the helpers below `expect()` the bitstring rather than silently
    // returning `false`, mirroring `bitstring.bitcheck(None, ...)`'s
    // RPython fail-fast (`TypeError: object of type 'NoneType' has no
    // len()`).

    /// effectinfo.py: check_readonly_descr_field(fielddescr)
    pub fn check_readonly_descr_field(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.readonly_descrs_fields
                .as_deref()
                .expect("check_readonly_descr_field on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// effectinfo.py: check_write_descr_field(fielddescr)
    pub fn check_write_descr_field(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.write_descrs_fields
                .as_deref()
                .expect("check_write_descr_field on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// Raw-set fallback for the macro / `JitDriver` path, where
    /// `compute_bitstrings` never runs: the bitstring `write_descrs_fields`
    /// stays empty and a field descr's `ei_index` stays at the `u32::MAX`
    /// sentinel, so every `check_write_descr_field` lookup misses.  Returns
    /// true iff this EI's concrete `_write_descrs_fields` names `descr` by
    /// pointer identity — the same identity `compute_bitstrings` keys on.
    /// `OptHeap::force_from_effectinfo` consults this only when
    /// `compute_bitstrings_has_run()` is false, leaving the translated
    /// (rustpython) bitstring path untouched.
    pub fn writes_field_descr_by_identity(&self, descr: &DescrRef) -> bool {
        match self._write_descrs_fields.as_deref() {
            Some(fields) => {
                let target = descr_ptr_id(descr);
                fields.iter().any(|d| descr_ptr_id(d) == target)
            }
            None => false,
        }
    }

    /// effectinfo.py: check_readonly_descr_array(arraydescr)
    pub fn check_readonly_descr_array(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.readonly_descrs_arrays
                .as_deref()
                .expect("check_readonly_descr_array on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// effectinfo.py: check_write_descr_array(arraydescr)
    pub fn check_write_descr_array(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.write_descrs_arrays
                .as_deref()
                .expect("check_write_descr_array on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// Array twin of [`Self::writes_field_descr_by_identity`], for the same
    /// macro / `JitDriver` path where `compute_bitstrings` never runs.
    ///
    /// Without it a `residual_writes` array declaration is inert: the raw set
    /// names the descr, but every array descr's `ei_index` is still the
    /// `u32::MAX` sentinel so `check_write_descr_array` always misses, and the
    /// trace keeps serving a `getarrayitem_gc_i` cached from before the call.
    /// The field side has been recovering exactly this case since the write-set
    /// table was introduced; the array side had no counterpart.
    ///
    /// This matches on the array's STRUCTURAL shape where the field twin uses
    /// pointer identity, and the asymmetry is deliberate.  A field descr has a
    /// single producer — `gc_cache()._cache_field`, which both the getfield
    /// lowering and the write-EI builder mint through — so two references to
    /// one field are always the same `Arc`.  An array descr on this path is
    /// minted per-`MetaInterpStaticData` in `dispatch_array_descr_cache` at
    /// trace time, while the write-EI is built during jitcode assembly, so the
    /// two are never the same `Arc`.  Structure is the right key regardless:
    /// `Assembler::add_raw_int_array_descr_signed` already dedups on exactly this
    /// tuple, so for these arrays the shape IS the identity.
    ///
    /// Conflating two arrays that share a shape can only over-invalidate,
    /// which costs a reload and can never miscompile — the same asymmetry the
    /// `residual_writes` table is built on.  And the caller gates this on
    /// `!compute_bitstrings_has_run()`, so the translated interpreter's
    /// registered array descrs never reach it.
    pub fn writes_array_descr_by_shape(&self, descr: &DescrRef) -> bool {
        let Some(arrays) = self._write_descrs_arrays.as_deref() else {
            return false;
        };
        let shape = |d: &DescrRef| {
            d.as_array_descr().map(|a| {
                (
                    a.base_size(),
                    a.item_size(),
                    a.item_type(),
                    a.is_item_signed(),
                )
            })
        };
        // A non-array descr in either position has no shape to compare, so it
        // matches nothing rather than matching everything.
        let Some(target) = shape(descr) else {
            return false;
        };
        arrays.iter().any(|d| shape(d) == Some(target))
    }

    /// effectinfo.py: check_readonly_descr_interiorfield (NOTE: not used so far)
    pub fn check_readonly_descr_interiorfield(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.readonly_descrs_interiorfields
                .as_deref()
                .expect("check_readonly_descr_interiorfield on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// effectinfo.py: check_write_descr_interiorfield (NOTE: not used so far)
    pub fn check_write_descr_interiorfield(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.write_descrs_interiorfields
                .as_deref()
                .expect("check_write_descr_interiorfield on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// effectinfo.py:201-206: set single_write_descr_array.
    ///
    /// Builder: attaches the actual array DescrRef for ARRAYCOPY/ARRAYMOVE
    /// unrolling. RPython sets this in `EffectInfo.__new__()` when
    /// `_write_descrs_arrays` has exactly one element.
    pub fn with_single_write_descr_array(mut self, descr: DescrRef) -> Self {
        self.single_write_descr_array = Some(descr);
        self
    }

    /// effectinfo.py:201-206: auto-set single_write_descr_array.
    ///
    /// RPython sets this when `_write_descrs_arrays` has exactly one
    /// element. Pyre's bitstring counterpart counts the set bits.
    pub fn set_single_write_descr_array(&mut self, descr: DescrRef) {
        let count: u32 = match &self.write_descrs_arrays {
            Some(bs) => bs.iter().map(|b| b.count_ones()).sum(),
            None => 0,
        };
        if count == 1 {
            self.single_write_descr_array = Some(descr);
        }
    }
}

// effectinfo.py: CallInfoCollection

/// One `CallInfoCollection` row carried across the build/runtime split.
///
/// `descr_index` is the slot of this row's calldescr in `Assembler.descrs`
/// (the same object the residual call encodes). `func` is the build-time
/// function address, including a symbolic path hash when the callee was
/// unbound, so the runtime rebinds it the way jitcode constants are rebound.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallInfoRow {
    pub oopspecindex: OopSpecIndex,
    pub descr_index: u32,
    pub func: i64,
}

/// effectinfo.py: `class CallInfoCollection(object)`.
///
/// Maps oopspec indices to `(calldescr, func_as_int)` pairs. Used to
/// look up the implementation of special-cased operations (arraycopy,
/// string ops, etc.).
#[derive(Debug, Clone, Default)]
pub struct CallInfoCollection {
    /// effectinfo.py: `_callinfo_for_oopspec` — `{oopspecindex: (calldescr, func_as_int)}`
    entries: std::collections::HashMap<OopSpecIndex, (DescrRef, u64)>,
    /// majit extension: func_as_int → function name.
    /// RPython derives names from `func.ptr._obj._name` at `see_raw_object` time.
    /// Since majit has no function pointers, we store the name at `add()` time.
    func_names: std::collections::HashMap<u64, String>,
}

impl CallInfoCollection {
    pub fn new() -> Self {
        Self::default()
    }

    /// effectinfo.py: add(oopspecindex, calldescr, func_as_int)
    pub fn add(&mut self, oopspec: OopSpecIndex, calldescr: DescrRef, func_addr: u64) {
        self.entries.insert(oopspec, (calldescr, func_addr));
    }

    /// Register the name for a function address.
    /// RPython: the name is `func.ptr._obj._name`, extracted by `see_raw_object`.
    /// In majit, we must store it explicitly since we have no pointer linkage.
    pub fn register_func_name(&mut self, func_addr: u64, name: String) {
        self.func_names.insert(func_addr, name);
    }

    /// effectinfo.py: has_oopspec(oopspecindex)
    pub fn has_oopspec(&self, oopspec: OopSpecIndex) -> bool {
        self.entries.contains_key(&oopspec)
    }

    /// effectinfo.py: callinfo_for_oopspec(oopspecindex)
    /// Returns (calldescr, func_as_int) for the oopspec, or `(None, 0)` on a
    /// miss (KeyError) — the calldescr is optional, the func defaults to 0.
    pub fn callinfo_for_oopspec(&self, oopspec: OopSpecIndex) -> (Option<&DescrRef>, u64) {
        match self.entries.get(&oopspec) {
            Some((calldescr, func)) => (Some(calldescr), *func),
            None => (None, 0),
        }
    }

    /// effectinfo.py: all_function_addresses_as_int()
    pub fn all_function_addresses_as_int(&self) -> Vec<u64> {
        self.entries.values().map(|(_, addr)| *addr).collect()
    }

    /// Rows in oopspec-discriminant order. `HashMap` iteration is not
    /// a function of the inserted keys, and the artifact must be.
    pub fn entries_in_oopspec_order(&self) -> Vec<(OopSpecIndex, DescrRef, u64)> {
        let mut rows: Vec<(OopSpecIndex, DescrRef, u64)> = self
            .entries
            .iter()
            .map(|(oopspec, (calldescr, func))| (*oopspec, calldescr.clone(), *func))
            .collect();
        rows.sort_by_key(|(oopspec, _, _)| *oopspec as u16);
        rows
    }

    /// Look up function name by address.
    /// RPython: `see_raw_object(func.ptr)` derives name from `func.ptr._obj._name`.
    pub fn func_name(&self, addr: u64) -> Option<&str> {
        self.func_names.get(&addr).map(String::as_str)
    }
}

// effectinfo.py `compute_bitstrings(all_descrs)`.
/// `effectinfo.py` "no new EffectInfo after compute_bitstrings"
/// invariant — flipped to `true` on the first
/// `MetaInterpStaticData::finish_setup_descrs` call.  Late call-descr
/// minters consult this through
/// [`compute_bitstrings_has_run()`] and refuse to introduce a new
/// `EffectInfo` whose raw `_*_descrs_*` sets are non-trivial (any of
/// the six `Some(non-empty Vec<u32>)`); a non-trivial raw set after
/// the bitstring compaction would shift the (eisetr, eisetw) class
/// partition without re-running `compute_bitstrings`, leaving every
/// existing EI's bitstring stale.
///
/// PyPy's analog is the `Ellipsis` sentinel at
/// `effectinfo.py` plus the implicit lifecycle ordering
/// (codewriter mints all EIs before `compute_bitstrings`).  Pyre's
/// architecture allows trace-time mints, so the invariant is enforced
/// at the construction site instead of via post-hoc bitcheck failure.
static COMPUTE_BITSTRINGS_RAN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Read accessor used by `make_call_descr_with_effect` to gate
/// post-setup mints.  Returns `true` iff
/// `MetaInterpStaticData::finish_setup_descrs` has executed.
pub fn compute_bitstrings_has_run() -> bool {
    COMPUTE_BITSTRINGS_RAN.load(std::sync::atomic::Ordering::Acquire)
}

/// Setter — invoked by `finish_setup_descrs` after `compute_bitstrings`
/// returns.  Idempotent; subsequent calls keep the flag set.
pub fn mark_compute_bitstrings_ran() {
    COMPUTE_BITSTRINGS_RAN.store(true, std::sync::atomic::Ordering::Release);
}

impl EffectInfo {
    /// `effectinfo.py:149-162` invariant probe.  An EI carries a
    /// non-trivial raw descr set iff at least one of the six
    /// `_*_descrs_*` slots is `Some(non-empty)`.  Trivial shapes
    /// (`None` ⇒ random-effects, `Some(empty)` ⇒ analyzer-confirmed
    /// no-heap) leave every existing EI's bitstring valid because
    /// `compute_bitstrings` would map an empty raw set to an empty
    /// bitstring without disturbing other EIs' (eisetr, eisetw)
    /// classes.
    pub fn has_non_trivial_raw_set(&self) -> bool {
        let any_non_empty = |s: &Option<Vec<DescrRef>>| s.as_ref().is_some_and(|v| !v.is_empty());
        any_non_empty(&self._readonly_descrs_fields)
            || any_non_empty(&self._write_descrs_fields)
            || any_non_empty(&self._readonly_descrs_arrays)
            || any_non_empty(&self._write_descrs_arrays)
            || any_non_empty(&self._readonly_descrs_interiorfields)
            || any_non_empty(&self._write_descrs_interiorfields)
    }
}

/// `effectinfo.py EffectInfo._cache` parity key.
///
/// PyPy keys `EffectInfo._cache` (the EI factory cache) on the tuple
/// `(readonly_descrs_fields, readonly_descrs_arrays,
///   readonly_descrs_interiorfields, write_descrs_fields,
///   write_descrs_arrays, write_descrs_interiorfields, extraeffect,
///   oopspecindex, can_invalidate, can_collect)` plus an `object()`
/// breaker for `call_release_gil_target != 0` (line 144-146).
/// `effectinfo.py frozenset(eisetr)` then collapses the
/// per-descr (eisetr, eisetw) lists by `id(ei)`, which is identical
/// to the cache key in PyPy's runtime because the cache hit returns
/// the same EI instance.
///
/// Production EffectInfos are interned by [`intern_effect_info`].  This key is
/// still reconstructed for `compute_bitstrings` because its input is a set of
/// owned snapshots, and test/macro callers can supply non-interned values.
/// Two structurally-equal values produce equal `EiCanonKey`s; release-gil
/// targets retain the fresh-object distinction below.
/// `effectinfo.py` release-gil cache-breaker.
///
/// PyPy: `if tgt_func: key += (object(),)` — every release-gil EI
/// gets a fresh `object()` so its `_cache` key is unique, making
/// each release-gil EI its own object identity even when
/// `(tgt_func, tgt_saveerr)` happens to match a previously-cached
/// instance. This shows up downstream in `effectinfo.py
/// frozenset(eisetr)` where `id(ei)` distinguishes release-gil EIs.
///
/// Pyre's lift: `NoTarget` for non-release-gil EIs (structural dedup
/// applies); `Unique(u64)` from a process-monotonic counter for
/// release-gil EIs (each `from_effect_info` call gets a fresh id,
/// matching PyPy's `object()` semantic — same release-gil EI cloned
/// twice still produces two distinct canonical keys, so
/// `compute_bitstrings`' canonical-id dedup never collapses
/// release-gil EIs across distinct mint sites).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ReleaseGilCacheKey {
    NoTarget,
    Unique(u64),
}

static NEXT_RELEASE_GIL_KEY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// `effectinfo.py` `EffectInfo._cache` post-frozenset key
/// canonicalization for compute_bitstrings's per-EI dedup pass.
///
/// Each raw set is projected to its sorted `Vec<usize>` of
/// `Arc::as_ptr` thin-pointer ids — the Rust lift of PyPy's
/// `frozenset[id(descr)]`.  Two structurally-identical EIs (same
/// extraeffect/oopspec/release_gil + same set of descr Arcs) hash
/// equal; two EIs whose raw sets differ in any descr Arc identity
/// hash unequal.  Conversion via `descr_set_to_ptr_set` deepens
/// `canonicalize_descr_set` (already sorted/dedup) by stripping the
/// vtable so the key is plain `Vec<usize>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EiCanonKey {
    extraeffect: ExtraEffect,
    oopspecindex: OopSpecIndex,
    readonly_descrs_fields: Option<Vec<usize>>,
    write_descrs_fields: Option<Vec<usize>>,
    readonly_descrs_arrays: Option<Vec<usize>>,
    write_descrs_arrays: Option<Vec<usize>>,
    readonly_descrs_interiorfields: Option<Vec<usize>>,
    write_descrs_interiorfields: Option<Vec<usize>>,
    can_invalidate: bool,
    can_collect: bool,
    /// `effectinfo.py:144-146` release-gil target breaker. PyPy
    /// inserts a fresh `object()` per release-gil EI; pyre's lift
    /// matches via [`ReleaseGilCacheKey::Unique`] from a process-
    /// monotonic counter.
    call_release_gil: ReleaseGilCacheKey,
}

/// Address-independent image of [`EiCanonKey`] for source translation.
///
/// RPython runs `compute_bitstrings` while translating, so descriptor and
/// EffectInfo object identity are stable inside the translated image. Pyre's
/// analyzer and runtime are separate processes; the serializable
/// [`DescrSetKeys`] are the translated-image names of those same descriptor
/// objects. This key therefore replaces each raw `DescrRef` frozenset with its
/// structural member set. It is build-time-only data, not a runtime side table.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FrozenEiCanonKey {
    extraeffect: ExtraEffect,
    oopspecindex: u16,
    runtime_helper: u8,
    descr_set_keys: Option<DescrSetKeys>,
    can_invalidate: bool,
    can_collect: bool,
    call_release_gil_site: Option<usize>,
}

impl FrozenEiCanonKey {
    fn from_effect_info(ei: &EffectInfo, position: usize) -> Self {
        Self {
            extraeffect: ei.extraeffect,
            oopspecindex: ei.oopspecindex as u16,
            runtime_helper: ei.runtime_helper as u8,
            descr_set_keys: ei.descr_set_keys.as_deref().cloned(),
            can_invalidate: ei.can_invalidate,
            can_collect: ei.can_collect,
            // effectinfo.py:144-146 inserts a fresh object into every
            // release-gil cache key. The stable build-time analogue is the
            // EI's position in the deterministic descriptor/JitCode walk.
            call_release_gil_site: (ei.call_release_gil_target.0 != 0).then_some(position),
        }
    }
}

pub struct FrozenBitstringLayout {
    pub descr_indices: [std::collections::BTreeMap<DescrSetMember, u32>; 3],
    /// Canonical translated-object identity parallel to the `all_eis` input.
    /// Ordinary structurally-identical EIs share an id; release-gil EIs do not.
    pub effect_info_ids: Vec<u32>,
}

/// Project an Arc-identity raw set to a thin-pointer ptr-id Vec.
/// Caller's `Vec<DescrRef>` is assumed already canonicalised
/// (sorted by `Arc::as_ptr`, dedup'd) — the resulting `Vec<usize>`
/// inherits that ordering.
fn descr_set_to_ptr_set(s: &Option<Vec<DescrRef>>) -> Option<Vec<usize>> {
    s.as_ref()
        .map(|v| v.iter().map(descr_ptr_id).collect::<Vec<_>>())
}

/// Public re-export of [`descr_set_to_ptr_set`] for cross-module
/// use — `LLType::func_key` builds its cache-key ptr-id Vecs by
/// projecting the EI raw sets through this helper.
pub fn descr_set_to_ptr_set_pub(s: &Option<Vec<DescrRef>>) -> Option<Vec<usize>> {
    descr_set_to_ptr_set(s)
}

impl EiCanonKey {
    fn from_effect_info(ei: &EffectInfo) -> Self {
        let call_release_gil = if ei.call_release_gil_target.0 != 0 {
            ReleaseGilCacheKey::Unique(
                NEXT_RELEASE_GIL_KEY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            )
        } else {
            ReleaseGilCacheKey::NoTarget
        };
        Self {
            extraeffect: ei.extraeffect,
            oopspecindex: ei.oopspecindex,
            readonly_descrs_fields: descr_set_to_ptr_set(&ei._readonly_descrs_fields),
            write_descrs_fields: descr_set_to_ptr_set(&ei._write_descrs_fields),
            readonly_descrs_arrays: descr_set_to_ptr_set(&ei._readonly_descrs_arrays),
            write_descrs_arrays: descr_set_to_ptr_set(&ei._write_descrs_arrays),
            readonly_descrs_interiorfields: descr_set_to_ptr_set(
                &ei._readonly_descrs_interiorfields,
            ),
            write_descrs_interiorfields: descr_set_to_ptr_set(&ei._write_descrs_interiorfields),
            can_invalidate: ei.can_invalidate,
            can_collect: ei.can_collect,
            call_release_gil,
        }
    }
}

/// Translation-time form of [`compute_bitstrings`] using stable descriptor
/// keys instead of live `Arc` addresses.
///
/// This is the same `effectinfo.py` partition: for each of fields,
/// arrays and interior fields, descriptors with identical `(readers, writers)`
/// membership share one compact `ei_index`, and the most popular classes are
/// assigned first. The returned maps are the translated-image equivalent of
/// `descr.ei_index`; a serializer can put each value onto the corresponding
/// descriptor construction record rather than retaining a runtime side table.
///
/// Numeric indices need not match [`compute_bitstrings`]' pointer-address
/// tie-break. Membership and bit checks do match, while the structural
/// [`DescrSetMember`] tie-break makes the result reproducible across analyzer
/// processes.
pub fn compute_frozen_bitstrings(all_eis: &mut [&mut EffectInfo]) -> FrozenBitstringLayout {
    use std::collections::{BTreeMap, BTreeSet};

    // `frozenset_or_none`: canonical member order is part of neither
    // semantics nor identity, so normalize it before forming cache keys.
    for ei in all_eis.iter_mut() {
        match ei.descr_set_keys.as_mut() {
            // Matched per variant rather than through `DerefMut`: that impl
            // materializes `Empty` into a boxed empty `DescrSetKeys`, so
            // canonicalizing every image would spend the compact
            // representation on every EI that has no keys — before
            // serialization, which is what the representation exists for.
            // `Empty` has no member order to normalize; it is already
            // canonical.
            Some(DescrSetKeysImage::Populated(keys)) => keys.canonicalize(),
            Some(DescrSetKeysImage::Empty) => {}
            None => {
                ei.readonly_descrs_fields = None;
                ei.write_descrs_fields = None;
                ei.readonly_descrs_arrays = None;
                ei.write_descrs_arrays = None;
                ei.readonly_descrs_interiorfields = None;
                ei.write_descrs_interiorfields = None;
            }
        }
    }

    // `EffectInfo._cache`: structurally identical ordinary EIs denote one
    // object. Release-gil EIs retain one identity per deterministic mint site.
    let mut canonical_id_for_position = Vec::with_capacity(all_eis.len());
    let mut first_position_for_shape: BTreeMap<FrozenEiCanonKey, usize> = BTreeMap::new();
    for (position, ei) in all_eis.iter().enumerate() {
        let key = FrozenEiCanonKey::from_effect_info(ei, position);
        let next = first_position_for_shape.len();
        let canonical_id = *first_position_for_shape.entry(key).or_insert(next);
        canonical_id_for_position.push(canonical_id);
    }

    let mut result: [BTreeMap<DescrSetMember, u32>; 3] = std::array::from_fn(|_| BTreeMap::new());
    for category in 0..3 {
        let mut category_members = BTreeSet::new();
        for ei in all_eis.iter() {
            let Some(keys) = ei.descr_set_keys.as_ref() else {
                continue;
            };
            let (read, write) = pick_key_category(keys, category);
            category_members.extend(read.iter().cloned());
            category_members.extend(write.iter().cloned());
        }

        let mut memberships = Vec::with_capacity(category_members.len());
        for member in category_members {
            assert_member_category(&member, category);
            let mut readers = Vec::new();
            let mut writers = Vec::new();
            for (position, ei) in all_eis.iter().enumerate() {
                let Some(keys) = ei.descr_set_keys.as_ref() else {
                    continue;
                };
                let canonical_id = canonical_id_for_position[position];
                let (read, write) = pick_key_category(keys, category);
                if read.binary_search(&member).is_ok() {
                    readers.push(canonical_id);
                }
                if write.binary_search(&member).is_ok() {
                    writers.push(canonical_id);
                }
            }
            readers.sort_unstable();
            readers.dedup();
            writers.sort_unstable();
            writers.dedup();
            memberships.push((member, readers, writers));
        }

        // effectinfo.py size_of_both_sets popularity ordering + mapping.setdefault.
        memberships.sort_by(|a, b| {
            (b.1.len() + b.2.len())
                .cmp(&(a.1.len() + a.2.len()))
                .then(a.0.cmp(&b.0))
        });
        let mut class_indices: BTreeMap<(Vec<usize>, Vec<usize>), u32> = BTreeMap::new();
        for (member, readers, writers) in memberships {
            let next = class_indices.len() as u32;
            let ei_index = *class_indices.entry((readers, writers)).or_insert(next);
            result[category].insert(member, ei_index);
        }

        // effectinfo.py:528-538: bitstrings are encoded from the descriptor's
        // compact class index, exactly as the live-object path does.
        for ei in all_eis.iter_mut() {
            let Some(keys) = ei.descr_set_keys.as_ref() else {
                continue;
            };
            let (read, write) = pick_key_category(keys, category);
            let read_indices: Vec<u32> = read
                .iter()
                .map(|member| {
                    *result[category]
                        .get(member)
                        .expect("frozen bitstrings: read member missing from partition")
                })
                .collect();
            let write_indices: Vec<u32> = write
                .iter()
                .map(|member| {
                    *result[category]
                        .get(member)
                        .expect("frozen bitstrings: write member missing from partition")
                })
                .collect();
            store_category_bitstrings(
                ei,
                category,
                Some(crate::bitstring::make_bitstring(&read_indices)),
                Some(crate::bitstring::make_bitstring(&write_indices)),
            );
        }
    }
    FrozenBitstringLayout {
        descr_indices: result,
        effect_info_ids: canonical_id_for_position
            .into_iter()
            .map(|id| u32::try_from(id).expect("too many translated EffectInfo identities"))
            .collect(),
    }
}

fn pick_key_category(
    keys: &DescrSetKeys,
    category: usize,
) -> (&Vec<DescrSetMember>, &Vec<DescrSetMember>) {
    match category {
        0 => (&keys.readonly_fields, &keys.write_fields),
        1 => (&keys.readonly_arrays, &keys.write_arrays),
        2 => (&keys.readonly_interiorfields, &keys.write_interiorfields),
        _ => unreachable!("frozen bitstrings: invalid category"),
    }
}

fn assert_member_category(member: &DescrSetMember, category: usize) {
    assert!(
        matches!(
            (category, member),
            (0, DescrSetMember::Field { .. })
                | (1, DescrSetMember::Array { .. })
                | (2, DescrSetMember::InteriorField { .. })
        ),
        "frozen bitstrings: descriptor member is in the wrong category: {member:?}"
    );
}

/// `effectinfo.py` `compute_bitstrings`.
///
/// Walks every `EffectInfo` once: for the three categories
/// (`fields`/`arrays`/`interiorfields`), partitions the descrs that
/// appear in some EI's `_readonly_descrs_*` / `_write_descrs_*` set
/// into (eisetr, eisetw) equivalence classes — descrs whose membership
/// pattern across all EIs is identical share an `ei_index`. The
/// bitstrings on each EI are then encoded with the descr-side
/// `descr.get_ei_index()` rather than the global descr index, which is
/// what makes upstream's "4000+ descrs → 373 ei classes" compaction
/// possible.
///
/// Inputs:
/// - `all_descrs`: every descriptor that participates in any EI's
///   readonly/write set (typically: every field/array/interiorfield
///   descr the codewriter has minted by this point). Descrs not in any
///   EI keep their `ei_index = u32::MAX` sentinel; PyPy
///   `effectinfo.py` writes the same `descr.ei_index = sys.maxint`
///   default for the no-EI branch.
/// - `all_eis`: the full set of EI handles to compact. EIs whose
///   `_readonly_descrs_fields` is `None` (random-effects /
///   `MOST_GENERAL`) follow PyPy's `effectinfo.py` rule:
///   their bitstring fields are also forced to `None`, the call site
///   is expected to gate via `has_random_effects()`.
///
/// Effect: writes to `descr.set_ei_index(...)` (interior atomic) for
/// descrs reachable through `all_descrs` AND rewrites every
/// `EffectInfo.bitstring_*` field. Idempotent — calling it twice on
/// the same input is a no-op.
///
/// PyPy `effectinfo.py descr.ei_index = …` stamps the per-descr
/// `ei_index` directly on the descriptor object; pyre matches via the
/// `Descr::set_ei_index` atomic. Readers (`heap.rs::field_effect_index`
/// etc.) then resolve through `descr.get_ei_index()` alone — no
/// process-global side table.
pub fn compute_bitstrings(all_descrs: &[DescrRef], all_eis: &mut [&mut EffectInfo]) {
    use std::collections::HashMap;

    // `effectinfo.py` `for descr in all_descrs:` pre-loop —
    // every non-call descr's `ei_index` is initialised to
    // `sys.maxint` BEFORE any bitstring classification.  Pyre lifts
    // this so re-running `compute_bitstrings` over an `all_descrs`
    // that already received per-descr stamps from a prior invocation
    // does not leak stale `ei_index` values into the post-classification
    // raw-set probes (`heap.rs::field_effect_index` reads
    // `descr.get_ei_index()` directly with no side-table fallback).
    for descr in all_descrs {
        if descr.as_call_descr().is_none() {
            descr.set_ei_index(u32::MAX);
        }
    }

    // effectinfo.py:484-489: random-effects EIs zero out bitstrings.
    // The matching invariant — when `_readonly_descrs_fields` is None,
    // every sibling raw set is None too — is asserted line-for-line.
    //
    // Same loop also enforces the *frozenset-or-none* canonical form
    // (sorted-by-`Arc::as_ptr` + dedup'd by `Arc::ptr_eq`) on every
    // non-None raw set.  Caller sites usually pre-canonicalise via
    // `canonicalize_descr_set`, but routing every set through here
    // is the self-enforcing equivalent of `effectinfo.py:128-145`'s
    // `frozenset_or_none(...)` invariant — once `compute_bitstrings`
    // returns, the `_*_descrs_*` Vecs match PyPy's frozenset semantic.
    fn canonicalize(set: &mut Option<Vec<DescrRef>>) {
        if let Some(v) = set.take() {
            *set = Some(canonicalize_descr_set(v));
        }
    }
    for ei in all_eis.iter_mut() {
        if ei._readonly_descrs_fields.is_none() {
            assert!(ei._write_descrs_fields.is_none());
            assert!(ei._readonly_descrs_arrays.is_none());
            assert!(ei._write_descrs_arrays.is_none());
            assert!(ei._readonly_descrs_interiorfields.is_none());
            assert!(ei._write_descrs_interiorfields.is_none());
            ei.readonly_descrs_fields = None;
            ei.write_descrs_fields = None;
            ei.readonly_descrs_arrays = None;
            ei.write_descrs_arrays = None;
            ei.readonly_descrs_interiorfields = None;
            ei.write_descrs_interiorfields = None;
        } else {
            canonicalize(&mut ei._readonly_descrs_fields);
            canonicalize(&mut ei._write_descrs_fields);
            canonicalize(&mut ei._readonly_descrs_arrays);
            canonicalize(&mut ei._write_descrs_arrays);
            canonicalize(&mut ei._readonly_descrs_interiorfields);
            canonicalize(&mut ei._write_descrs_interiorfields);
        }
    }

    // Per-category descr-Arc enumeration.
    //
    // `effectinfo.py:493-495`:
    // ```python
    // descrs = {'fields': set(), 'arrays': set(),
    //           'interiorfields': set()}
    // for ei in all_eis:
    //     descrs['fields'].update(ei._readonly_descrs_fields)
    //     ...
    // ```
    // Python's `set.update` keys on `id(descr)`, so the partition is
    // intrinsically collision-free by object identity. Pyre's lift
    // builds the same set per category inside the loop below
    // (`category_descrs`), keyed on `Arc::as_ptr` ptr-id — direct lift
    // of Python's `id()`.
    //
    // Descrs in `all_descrs` that never enter any EI raw set keep
    // their `ei_index = u32::MAX` sentinel via the entry-loop pre-init
    // (`effectinfo.py` `descr.ei_index = sys.maxint`), which runs
    // unconditionally at the top of `compute_bitstrings` over every
    // non-call descr — no per-category lookup is needed here.

    // `effectinfo.py` `EffectInfo._cache` parity — PyPy keys
    // the EffectInfo factory cache on the structural tuple (raw sets,
    // extraeffect, oopspecindex, can_invalidate, can_collect,
    // call_release_gil_target) and returns the same EI instance for
    // structurally-identical requests. `effectinfo.py
    // frozenset(eisetr)` then collapses by identity (id(ei)). Pyre
    // lacks the EI._cache today (each `EffectInfoCell` is per-call-
    // descr), so structurally-identical EIs land at distinct
    // positions in `all_eis`. Canonicalise here: assign each position
    // a `canonical_id` based on first-occurrence of its structural
    // shape; identical shapes share the canonical id, matching PyPy's
    // post-frozenset (eisetr, eisetw) class identity.
    let mut canonical_id_for_position: Vec<usize> = Vec::with_capacity(all_eis.len());
    let mut first_position_for_shape: HashMap<EiCanonKey, usize> = HashMap::new();
    for ei in all_eis.iter() {
        let key = EiCanonKey::from_effect_info(ei);
        let canon = *first_position_for_shape
            .entry(key)
            .or_insert(canonical_id_for_position.len());
        canonical_id_for_position.push(canon);
    }

    // Three category iterations match `for key in descrs:` at
    // `effectinfo.py:498-540`. Each category is independent: a descr
    // is partitioned within its category, so the same `descr.ei_index`
    // bank is shared across categories — `descr.ei_index` is unique
    // per category (a fielddescr never appears in `arrays` etc.).
    for category in 0..3usize {
        // Gather every descr Arc that appears in any active EI for
        // this category (`effectinfo.py:493-494`
        // `descrs[key].update(...)`). `category_descrs` is keyed by
        // ptr-id (Arc identity) and stores `DescrRef` clones so the
        // downstream `set_ei_index` fan-out has the actual Arc.
        let mut category_descrs: HashMap<usize, DescrRef> = HashMap::new();
        for ei in all_eis.iter() {
            let (r, w) = pick_category(ei, category);
            if let Some(rs) = r {
                for d in rs {
                    category_descrs
                        .entry(descr_ptr_id(d))
                        .or_insert_with(|| d.clone());
                }
            }
            if let Some(ws) = w {
                for d in ws {
                    category_descrs
                        .entry(descr_ptr_id(d))
                        .or_insert_with(|| d.clone());
                }
            }
        }

        // For each descr in the category, compute (eisetr, eisetw) —
        // PyPy `effectinfo.py:505-512` builds these as Python lists
        // then collapses via `frozenset()` on `id(ei)`. Pyre's lift
        // pushes `canonical_id_for_position[ei_i]` (instead of the raw
        // position) and `dedup`s per descr to mirror the frozenset
        // collapse. The popularity-sort count
        // (`size_of_both_sets = len(r) + len(w)`,
        // `effectinfo.py`) thus weights each logical EI once
        // per descr, regardless of how many call descrs share its
        // structural shape.
        //
        // Membership probe is `Arc::as_ptr` ptr-id binary_search.
        // The raw sets are pre-canonicalised by ptr-id at the
        // `canonicalize_descr_set` step above so binary_search by
        // ptr-id is exact.
        let mut all_sets: Vec<(usize, DescrRef, Vec<usize>, Vec<usize>)> =
            Vec::with_capacity(category_descrs.len());
        // Fix the iteration order so the popularity-sort tie-break below
        // does not inherit `category_descrs`' arbitrary one.
        //
        // A ptr-id is a heap address: it orders the walk reproducibly
        // within one process and arbitrarily across two. That is the whole
        // requirement here — `compute_bitstrings` runs at runtime, and no
        // output of it crosses a process boundary. Anything moved from here
        // to build time loses the property, because the two sides are two
        // processes.
        let mut sorted_descrs: Vec<(usize, DescrRef)> = category_descrs.into_iter().collect();
        sorted_descrs.sort_by_key(|(pid, _)| *pid);
        for (pid, descr) in sorted_descrs {
            let mut eisetr: Vec<usize> = Vec::new();
            let mut eisetw: Vec<usize> = Vec::new();
            for (ei_i, ei) in all_eis.iter().enumerate() {
                let canon = canonical_id_for_position[ei_i];
                let (r, w) = pick_category(ei, category);
                if let Some(rs) = r
                    && rs.binary_search_by_key(&pid, descr_ptr_id).is_ok()
                {
                    eisetr.push(canon);
                }
                if let Some(ws) = w
                    && ws.binary_search_by_key(&pid, descr_ptr_id).is_ok()
                {
                    eisetw.push(canon);
                }
            }
            // `frozenset(eisetr)` semantic: dedup canonical IDs (sort
            // stays trivial for `Vec<usize>` Hash/Eq downstream).
            eisetr.sort_unstable();
            eisetr.dedup();
            eisetw.sort_unstable();
            eisetw.dedup();
            all_sets.push((pid, descr, eisetr, eisetw));
        }

        // `effectinfo.py size_of_both_sets`: heuristic — sort by len(eisetr) +
        // len(eisetw) descending so the most popular descrs claim the
        // low ei_index slots, reducing total bitstring length. Tie-
        // break on ptr-id ascending, which fixes the order within this
        // process (see the sort above for what a ptr-id does and does not
        // promise).
        all_sets.sort_by(|a, b| {
            (b.2.len() + b.3.len())
                .cmp(&(a.2.len() + a.3.len()))
                .then(a.0.cmp(&b.0))
        });

        // `effectinfo.py:523-526`: assign ei_index per (eisetr, eisetw)
        // class. `mapping.setdefault((eisetr, eisetw), len(mapping))`
        // gives each new class the next sequential index.
        let mut mapping: HashMap<(Vec<usize>, Vec<usize>), u32> = HashMap::new();
        let mut descr_to_eiindex_by_ptr: HashMap<usize, (DescrRef, u32)> = HashMap::new();
        for (pid, descr, eisetr, eisetw) in all_sets {
            let next = mapping.len() as u32;
            let ei_index = *mapping.entry((eisetr, eisetw)).or_insert(next);
            descr_to_eiindex_by_ptr.insert(pid, (descr, ei_index));
        }

        // Write `descr.ei_index = class_idx` (`effectinfo.py:526`).
        // The Arc is guaranteed to be the actual descr from the raw
        // set, so two distinct Arcs that share `descr.index()` each
        // get their own `set_ei_index` call.
        for (descr, ei_index) in descr_to_eiindex_by_ptr.values() {
            descr.set_ei_index(*ei_index);
        }

        // `effectinfo.py` `descr.ei_index = sys.maxint`:
        // descrs from `all_descrs` that did NOT appear in any EI raw
        // set keep their `ei_index = u32::MAX` sentinel set by the
        // entry-loop pre-init at the top of this function. Readers
        // (`heap.rs::field_effect_index`, etc.) treat MAX as
        // no-bitstring per `bitstring.py:18 if byte_number >=
        // len(bitstring)`.

        // `effectinfo.py:528-538`: encode the bitstring on each active
        // EI using `descr.get_ei_index()` per descr in the raw set.
        // Inactive (random-effects) EIs were already zeroed above.
        for ei in all_eis.iter_mut() {
            let (r_raw, w_raw) = pick_category(ei, category);
            // Skip inactive EIs.
            if r_raw.is_none() {
                continue;
            }
            let r = r_raw.unwrap();
            let w = w_raw.unwrap();
            let r_indices: Vec<u32> = r
                .iter()
                .map(|d| {
                    descr_to_eiindex_by_ptr
                        .get(&descr_ptr_id(d))
                        .map(|(_, ei_index)| *ei_index)
                        .expect("compute_bitstrings: descr in EI raw set must be in the partition")
                })
                .collect();
            let w_indices: Vec<u32> = w
                .iter()
                .map(|d| {
                    descr_to_eiindex_by_ptr
                        .get(&descr_ptr_id(d))
                        .map(|(_, ei_index)| *ei_index)
                        .expect("compute_bitstrings: descr in EI raw set must be in the partition")
                })
                .collect();
            // `effectinfo.py assert sys.maxint not in bitstrr` —
            // every descr in an active EI's set must have been assigned
            // a finite `ei_index` above (we panic with `expect` if not).
            let r_bs = crate::bitstring::make_bitstring(&r_indices);
            let w_bs = crate::bitstring::make_bitstring(&w_indices);
            store_category_bitstrings(ei, category, Some(r_bs), Some(w_bs));
        }
    }
}

/// Helper: project `(_readonly_descrs_*, _write_descrs_*)` for a category.
fn pick_category(
    ei: &EffectInfo,
    category: usize,
) -> (Option<&Vec<DescrRef>>, Option<&Vec<DescrRef>>) {
    match category {
        0 => (
            ei._readonly_descrs_fields.as_ref(),
            ei._write_descrs_fields.as_ref(),
        ),
        1 => (
            ei._readonly_descrs_arrays.as_ref(),
            ei._write_descrs_arrays.as_ref(),
        ),
        2 => (
            ei._readonly_descrs_interiorfields.as_ref(),
            ei._write_descrs_interiorfields.as_ref(),
        ),
        _ => unreachable!("compute_bitstrings: invalid category"),
    }
}

/// Helper: install `(bitstring_readonly_*, bitstring_write_*)` for a
/// category. Mirrors `setattr(ei, 'bitstring_readonly_descrs_' + key, …)`
/// at `effectinfo.py:537-538`.
fn store_category_bitstrings(
    ei: &mut EffectInfo,
    category: usize,
    r: Option<Vec<u8>>,
    w: Option<Vec<u8>>,
) {
    match category {
        0 => {
            ei.readonly_descrs_fields = r;
            ei.write_descrs_fields = w;
        }
        1 => {
            ei.readonly_descrs_arrays = r;
            ei.write_descrs_arrays = w;
        }
        2 => {
            ei.readonly_descrs_interiorfields = r;
            ei.write_descrs_interiorfields = w;
        }
        _ => unreachable!("compute_bitstrings: invalid category"),
    }
}

#[cfg(test)]
mod compute_bitstrings_tests {
    use super::*;
    use crate::descr::{DescrRef, SimpleArrayDescr, SimpleFieldDescr};
    use crate::value::Type;
    use std::sync::Arc;

    fn mk_field(idx: u32) -> DescrRef {
        Arc::new(SimpleFieldDescr::new(idx, 0, 8, Type::Int, false)) as DescrRef
    }
    fn mk_array(idx: u32) -> DescrRef {
        Arc::new(SimpleArrayDescr::new(idx, 0, 8, 0, Type::Int)) as DescrRef
    }

    #[test]
    fn effect_info_cache_reuses_ordinary_identity() {
        let first = intern_effect_info(EffectInfo::default());
        let second = intern_effect_info(EffectInfo::default());
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn effect_info_cache_keeps_release_gil_identity_fresh() {
        let mut effect_info = EffectInfo::default();
        effect_info.call_release_gil_target = (1, 0);
        let first = intern_effect_info(effect_info.clone());
        let second = intern_effect_info(effect_info);
        assert!(!Arc::ptr_eq(&first, &second));
    }

    /// `effectinfo.py:143-146` reads `tgt_saveerr` out of the pair but never
    /// puts it in the key, so a null target carrying a stray `save_err` must
    /// not buy its own canonical cell — and therefore its own call descr.
    #[test]
    fn effect_info_cache_ignores_save_err_on_a_null_release_gil_target() {
        let mut with_save_err = EffectInfo::default();
        with_save_err.call_release_gil_target = (0, 3);
        // The premise the normalisation exists for: equality does split on the
        // pair, so without it the two below would land in separate cells.
        assert_ne!(with_save_err, EffectInfo::default());

        let plain = intern_effect_info(EffectInfo::default());
        let stray = intern_effect_info(with_save_err);
        assert!(Arc::ptr_eq(&plain, &stray));
        assert_eq!(
            stray.get().call_release_gil_target,
            EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
        );
    }

    /// Two EIs share a fielddescr in their write set; the descr's
    /// `ei_index` should be the same after compute_bitstrings, and
    /// each EI's `bitstring_write_descrs_fields` should set bit
    /// `descr.get_ei_index()`.
    #[test]
    fn shared_descr_collapses_to_one_class() {
        let f0 = mk_field(0);
        let f1 = mk_field(1);
        let mut ei_a = EffectInfo {
            _write_descrs_fields: Some(vec![f0.clone(), f1.clone()]),
            _readonly_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let mut ei_b = EffectInfo {
            _write_descrs_fields: Some(vec![f0.clone(), f1.clone()]),
            _readonly_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let descrs = vec![f0.clone(), f1.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei_a, &mut ei_b];
        compute_bitstrings(&descrs, &mut eis);

        // Both descrs appear in identical EIs (eisetr=∅, eisetw={A,B}),
        // so both go in the same class → same ei_index.
        assert_eq!(f0.get_ei_index(), f1.get_ei_index());
        assert_ne!(f0.get_ei_index(), u32::MAX);

        // Both EIs encode the same descr → bitstring with the bit at
        // `descr.get_ei_index()` set.
        let bs = ei_a.write_descrs_fields.as_ref().unwrap();
        assert!(crate::bitstring::bitcheck(bs, f0.get_ei_index()));
        assert!(crate::bitstring::bitcheck(bs, f1.get_ei_index()));
    }

    /// Two descrs with different (eisetr, eisetw) get different
    /// `ei_index`. Only one EI reads `f0`; both write `f1`.
    #[test]
    fn divergent_membership_yields_distinct_classes() {
        let f0 = mk_field(0);
        let f1 = mk_field(1);
        let mut ei_a = EffectInfo {
            _readonly_descrs_fields: Some(vec![f0.clone()]),
            _write_descrs_fields: Some(vec![f1.clone()]),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let mut ei_b = EffectInfo {
            _readonly_descrs_fields: Some(Vec::new()),
            _write_descrs_fields: Some(vec![f1.clone()]),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let descrs = vec![f0.clone(), f1.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei_a, &mut ei_b];
        compute_bitstrings(&descrs, &mut eis);
        assert_ne!(f0.get_ei_index(), f1.get_ei_index());
    }

    /// `MOST_GENERAL`-style EI with `_readonly_descrs_* = None` keeps
    /// bitstrings None and asserts `effectinfo.py:485-487`.
    #[test]
    fn random_effects_zeros_bitstrings() {
        let f0 = mk_field(0);
        let mut ei_random = EffectInfo {
            _readonly_descrs_fields: None,
            _write_descrs_fields: None,
            _readonly_descrs_arrays: None,
            _write_descrs_arrays: None,
            _readonly_descrs_interiorfields: None,
            _write_descrs_interiorfields: None,
            extraeffect: ExtraEffect::RandomEffects,
            // Pre-existing bitstring should be wiped.
            readonly_descrs_fields: Some(vec![0xff]),
            write_descrs_fields: Some(vec![0xff]),
            readonly_descrs_arrays: Some(vec![0xff]),
            write_descrs_arrays: Some(vec![0xff]),
            readonly_descrs_interiorfields: Some(vec![0xff]),
            write_descrs_interiorfields: Some(vec![0xff]),
            ..Default::default()
        };
        let descrs = vec![f0.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei_random];
        compute_bitstrings(&descrs, &mut eis);
        assert!(ei_random.readonly_descrs_fields.is_none());
        assert!(ei_random.write_descrs_fields.is_none());
        assert!(ei_random.readonly_descrs_arrays.is_none());
        assert!(ei_random.write_descrs_arrays.is_none());
        assert!(ei_random.readonly_descrs_interiorfields.is_none());
        assert!(ei_random.write_descrs_interiorfields.is_none());
        // Untouched descr keeps the sentinel.
        assert_eq!(f0.get_ei_index(), u32::MAX);
    }

    /// Field and array descrs with the same global `index()` get
    /// independent `ei_index`es because the partitioning is per-category.
    #[test]
    fn category_partitions_are_independent() {
        let f0 = mk_field(0);
        let a0 = mk_array(0);
        let mut ei = EffectInfo {
            _readonly_descrs_fields: Some(vec![f0.clone()]),
            _write_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(vec![a0.clone()]),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        // Both descrs have global index 0 but live in different
        // partitions; both should get ei_index = 0 in their own
        // partition (the class assignment starts fresh per category).
        let descrs = vec![f0.clone(), a0.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei];
        compute_bitstrings(&descrs, &mut eis);
        // Now that compute_bitstrings keys partitions by Arc identity,
        // the field and the array both get assigned ei_index in their
        // own categories — each should be != u32::MAX.
        assert_ne!(f0.get_ei_index(), u32::MAX);
        assert_ne!(a0.get_ei_index(), u32::MAX);
    }

    /// Two distinct field descrs that happen to share the same
    /// `descr.index()` (e.g. two structs' `index_in_parent = 0`
    /// fields per `call.rs`) MUST get distinct `ei_index`
    /// values when their (eisetr, eisetw) shape diverges.  PyPy
    /// `effectinfo.py:493-526` guarantees this by `id(descr)`
    /// (object identity) keyed `frozenset`/`set` — the Arc-identity
    /// lift in pyre matches the same semantic.  Earlier versions
    /// keyed the partition on `descr.index()`, silently collapsing
    /// these two distinct descrs into one ei_index.  This test
    /// pins the parity invariant.
    #[test]
    fn distinct_arcs_share_index_get_distinct_ei_indices() {
        let f_a = mk_field(99);
        let f_b = mk_field(99);
        // Two different EIs reference different Arcs even though
        // both Arcs report `descr.index() == 99`.
        let mut ei_reads_a = EffectInfo {
            _readonly_descrs_fields: Some(vec![f_a.clone()]),
            _write_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let mut ei_writes_b = EffectInfo {
            _readonly_descrs_fields: Some(Vec::new()),
            _write_descrs_fields: Some(vec![f_b.clone()]),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let descrs = vec![f_a.clone(), f_b.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei_reads_a, &mut ei_writes_b];
        compute_bitstrings(&descrs, &mut eis);
        assert_ne!(f_a.get_ei_index(), u32::MAX);
        assert_ne!(f_b.get_ei_index(), u32::MAX);
        // f_a appears only in eisetr, f_b only in eisetw → distinct
        // (eisetr, eisetw) class → distinct ei_index.
        assert_ne!(f_a.get_ei_index(), f_b.get_ei_index());
    }

    fn field_member(struct_id: u64, field_name: &str) -> DescrSetMember {
        DescrSetMember::Field {
            struct_id,
            field_name: field_name.to_owned(),
        }
    }

    #[test]
    fn frozen_partition_matches_live_membership_semantics() {
        let f0 = mk_field(0);
        let f1 = mk_field(1);
        let f2 = mk_field(2);
        let k0 = field_member(10, "a");
        let k1 = field_member(10, "b");
        let k2 = field_member(20, "c");

        let make_eis = || {
            vec![
                EffectInfo {
                    _readonly_descrs_fields: Some(vec![f0.clone(), f1.clone()]),
                    _write_descrs_fields: Some(vec![f2.clone()]),
                    descr_set_keys: Some(DescrSetKeysImage::from(DescrSetKeys {
                        readonly_fields: vec![k0.clone(), k1.clone()],
                        write_fields: vec![k2.clone()],
                        ..DescrSetKeys::default()
                    })),
                    ..EffectInfo::default()
                },
                EffectInfo {
                    _readonly_descrs_fields: Some(vec![f0.clone()]),
                    _write_descrs_fields: Some(vec![f1.clone(), f2.clone()]),
                    descr_set_keys: Some(DescrSetKeysImage::from(DescrSetKeys {
                        readonly_fields: vec![k0.clone()],
                        write_fields: vec![k1.clone(), k2.clone()],
                        ..DescrSetKeys::default()
                    })),
                    ..EffectInfo::default()
                },
            ]
        };

        let mut live_eis = make_eis();
        let mut live_refs: Vec<&mut EffectInfo> = live_eis.iter_mut().collect();
        compute_bitstrings(&[f0.clone(), f1.clone(), f2.clone()], &mut live_refs);

        let mut frozen_eis = make_eis();
        let mut frozen_refs: Vec<&mut EffectInfo> = frozen_eis.iter_mut().collect();
        let frozen_indices = compute_frozen_bitstrings(&mut frozen_refs);

        for (live, frozen) in live_eis.iter().zip(&frozen_eis) {
            for (descr, member) in [(&f0, &k0), (&f1, &k1), (&f2, &k2)] {
                let live_index = descr.get_ei_index();
                let frozen_index = frozen_indices.descr_indices[0][member];
                assert_eq!(
                    crate::bitstring::bitcheck(
                        live.readonly_descrs_fields.as_ref().unwrap(),
                        live_index,
                    ),
                    crate::bitstring::bitcheck(
                        frozen.readonly_descrs_fields.as_ref().unwrap(),
                        frozen_index,
                    )
                );
                assert_eq!(
                    crate::bitstring::bitcheck(
                        live.write_descrs_fields.as_ref().unwrap(),
                        live_index,
                    ),
                    crate::bitstring::bitcheck(
                        frozen.write_descrs_fields.as_ref().unwrap(),
                        frozen_index,
                    )
                );
            }
        }
    }

    #[test]
    fn frozen_partition_is_independent_of_member_input_order() {
        let ka = field_member(1, "a");
        let kb = field_member(1, "b");
        let mut ordered = EffectInfo {
            descr_set_keys: Some(DescrSetKeysImage::from(DescrSetKeys {
                readonly_fields: vec![ka.clone(), kb.clone()],
                write_fields: vec![kb.clone()],
                ..DescrSetKeys::default()
            })),
            ..EffectInfo::default()
        };
        let mut shuffled = EffectInfo {
            descr_set_keys: Some(DescrSetKeysImage::from(DescrSetKeys {
                readonly_fields: vec![kb.clone(), ka.clone(), kb.clone()],
                write_fields: vec![kb],
                ..DescrSetKeys::default()
            })),
            ..EffectInfo::default()
        };

        let ordered_indices = compute_frozen_bitstrings(&mut [&mut ordered]);
        let shuffled_indices = compute_frozen_bitstrings(&mut [&mut shuffled]);
        assert_eq!(
            ordered_indices.descr_indices,
            shuffled_indices.descr_indices
        );
        assert_eq!(
            ordered.readonly_descrs_fields,
            shuffled.readonly_descrs_fields
        );
        assert_eq!(ordered.write_descrs_fields, shuffled.write_descrs_fields);
    }

    #[test]
    fn frozen_partition_keeps_random_effects_wildcard() {
        let mut random = EffectInfo {
            extraeffect: ExtraEffect::RandomEffects,
            descr_set_keys: None,
            readonly_descrs_fields: Some(vec![0xff]),
            write_descrs_fields: Some(vec![0xff]),
            readonly_descrs_arrays: Some(vec![0xff]),
            write_descrs_arrays: Some(vec![0xff]),
            readonly_descrs_interiorfields: Some(vec![0xff]),
            write_descrs_interiorfields: Some(vec![0xff]),
            ..EffectInfo::default()
        };
        let indices = compute_frozen_bitstrings(&mut [&mut random]);
        assert!(
            indices
                .descr_indices
                .iter()
                .all(|category| category.is_empty())
        );
        assert!(random.readonly_descrs_fields.is_none());
        assert!(random.write_descrs_fields.is_none());
        assert!(random.readonly_descrs_arrays.is_none());
        assert!(random.write_descrs_arrays.is_none());
        assert!(random.readonly_descrs_interiorfields.is_none());
        assert!(random.write_descrs_interiorfields.is_none());
    }

    #[test]
    fn to_slots_indexes_the_stamp_then_mint_table() {
        let ka = field_member(1, "a");
        let kb = field_member(1, "b");
        let arr = DescrSetMember::Array { array_id: 7 };
        let mut member_index = std::collections::BTreeMap::new();
        member_index.insert(ka.clone(), 0);
        member_index.insert(kb.clone(), 1);
        member_index.insert(arr.clone(), 2);
        let keys = DescrSetKeys {
            readonly_fields: vec![kb.clone(), ka.clone()],
            write_arrays: vec![arr],
            ..DescrSetKeys::default()
        };
        let slots = keys.to_slots(&member_index);
        assert_eq!(slots.readonly_fields, vec![1, 0]);
        assert_eq!(slots.write_arrays, vec![2]);
        assert!(slots.write_fields.is_empty());
    }
}
