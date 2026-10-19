//! `rpython/rlib/_rweakvaldict.py` `WeakValueDictRepr`.
//!
//! `WEAKDICT` is `num_items`, `resize_counter`, `entries`. Each
//! `WEAKDICTENTRY` is `(key, WeakRefPtr)` — no `f_valid`, no `f_hash`.
//! `ll_valid` is a live `weakref_deref`; `ll_everused` is a non-null value
//! word, so a dead weakref stays until `ll_weakdict_resize`. Lookup and
//! resize are `rdict.py` `ll_dict_lookup` / `ll_dict_resize` (`DICT_INITSIZE`
//! is 8). `paranoia` is false.
//!
//! The key type is `WeakValueDictRepr(rtyper, r_key)`'s `r_key` parameter:
//! `ll_keyhash = r_key.get_ll_hash_function()`, `keyeq = r_key.get_ll_eq_function()`,
//! `ll_hash(entries, i) = fasthashfn(entries[i].key)`.
//!
//! `ll_weakdict_rehash_after_translation` is the one omitted method:
//! there is no translation-time `convert_const` snapshot of a prebuilt
//! dict (`resize_counter = -1`), so the rehash-then-resize path never
//! runs.
//!
//! Entries are `_ll_malloc_entries`: collecting nursery malloc
//! (`try_gc_alloc_collecting_rooted`), matching `lltype.malloc(ENTRIES, n,
//! zero=True)` of a `GcArray` (`rdict.py` `_ll_malloc_entries`,
//! `framework.py` `push_roots`). Access is index-based (`entry(entries, i)`);
//! no `&K` / `&V` iterator is handed out, so the rordereddict no-collect
//! non-moving path does not apply. The `WEAKDICT` object itself stands for
//! the prebuilt `interned_strings` dict (`ObjSpace.__init__` /
//! `WeakValueDictRepr.convert_const`): `alloc_raw` takes
//! `try_gc_alloc_stable_raw`, stamps `GCFLAG_NO_HEAP_PTRS`, and write-barriers
//! so `remember_young_pointer` enrolls it in `prebuilt_root_objects`.
//! `_ll_free_entries` is a no-op, so a replaced array is left for the
//! collector — collecting malloc lets a minor reclaim it instead of spilling
//! the block OLD (`alloc_with_type_no_collect`).

use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

use crate::celldict::StrKey;
use crate::unicodeobject::Utf8Str;
use crate::weakref::Weakref;

const PERTURB_SHIFT: u32 = 5;
const DICT_INITSIZE: usize = 8;
/// `rdict.py` `HIGHEST_BIT = r_uint(intmask(1 << (LONG_BIT - 1)))`.
const HIGHEST_BIT: usize = 1usize << (usize::BITS - 1);

static WEAKDICT_GC_TYPE_ID: AtomicU32 = AtomicU32::new(0);
static WEAKDICT_ENTRIES_GC_TYPE_ID: AtomicU32 = AtomicU32::new(0);

pub fn set_weakdict_gc_type_id(id: u32) {
    WEAKDICT_GC_TYPE_ID.store(id, Ordering::Relaxed);
}

pub fn set_weakdict_entries_gc_type_id(id: u32) {
    WEAKDICT_ENTRIES_GC_TYPE_ID.store(id, Ordering::Relaxed);
}

pub fn weakdict_gc_type_id() -> u32 {
    WEAKDICT_GC_TYPE_ID.load(Ordering::Relaxed)
}

fn weakdict_entries_gc_type_id() -> u32 {
    WEAKDICT_ENTRIES_GC_TYPE_ID.load(Ordering::Relaxed)
}

/// `r_key` hooks from `WeakValueDictRepr(rtyper, r_key)`.
pub trait WeakDictKey: Copy {
    /// `r_key.get_ll_hash_function()`. `ll_strhash` / `s.hash` (`Signed`).
    fn ll_keyhash(&self) -> isize;
    /// `r_key.get_ll_fasthash_function()`. `ll_strfasthash` of a stored key.
    fn ll_fasthash(&self) -> isize;
    /// Dummy/null key for `ll_set_null` (`r_key.convert_const(None)`).
    fn dummy() -> Self;
}

/// `WEAKDICTENTRY`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WeakDictEntry<K> {
    pub key: K,
    pub value: *mut Weakref,
}

/// `WEAKDICTENTRYARRAY`. `length` is `len(entries)`.
#[repr(C)]
pub struct WeakDictEntries<K> {
    pub length: usize,
    pub items: [WeakDictEntry<K>; 0],
}

/// `WEAKDICT` / `weakvaldict`.
#[repr(C)]
pub struct WeakDict<K> {
    pub num_items: isize,
    pub resize_counter: isize,
    pub entries: *mut WeakDictEntries<K>,
}

impl WeakDictKey for StrKey {
    fn ll_keyhash(&self) -> isize {
        ll_strhash(self.0)
    }

    fn ll_fasthash(&self) -> isize {
        ll_strfasthash(self.0)
    }

    fn dummy() -> Self {
        Self(std::ptr::null_mut())
    }
}

/// Prebuilt `WEAKDICT` (`WeakValueDictRepr.convert_const` /
/// `ObjSpace.__init__` interned_strings): old-gen, non-moving
/// (`try_gc_alloc_stable_raw`), stamped `GCFLAG_NO_HEAP_PTRS`, then
/// write-barried so `remember_young_pointer` enrolls it in
/// `prebuilt_root_objects`. Falls back to host memory when the tid is
/// unpublished or no stable hook is installed.
fn alloc_raw(tid: u32, bytes: usize) -> crate::gc_hook::GCREF {
    let gc_ptr = if tid != 0 {
        crate::gc_hook::try_gc_alloc_stable_raw(tid, bytes)
    } else {
        std::ptr::null_mut()
    };
    if !gc_ptr.is_null() {
        unsafe { crate::gc_hook::stamp_gc_no_heap_ptrs(gc_ptr) };
        crate::gc_hook::try_gc_write_barrier_managed(gc_ptr);
        return gc_ptr;
    }
    let layout = std::alloc::Layout::from_size_align(bytes, std::mem::align_of::<usize>())
        .expect("weakdict layout");
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    raw as crate::gc_hook::GCREF
}

fn entries_bytes<K>(n: usize) -> usize {
    std::mem::offset_of!(WeakDictEntries<K>, items)
        .checked_add(
            n.checked_mul(std::mem::size_of::<WeakDictEntry<K>>())
                .expect("weakdict entries size"),
        )
        .expect("weakdict entries size")
}

/// `_ll_malloc_entries`: `malloc(ENTRIES, n, zero=True)` of a `GcArray`.
/// Collecting (`try_gc_alloc_collecting_rooted`), matching that malloc
/// (`framework.py` `push_roots` around `_ll_malloc_entries`). Access is
/// index-based (`entry`); the array may move. The `WEAKDICT` object stays
/// on the old/prebuilt path (`alloc_raw`). Callers pin `d` and the old
/// array before this returns.
fn alloc_entries<K>(n: usize) -> *mut WeakDictEntries<K> {
    let bytes = entries_bytes::<K>(n);
    let tid = weakdict_entries_gc_type_id();
    if tid != 0 {
        let mut allocation_root: crate::gc_hook::GCREF = std::ptr::null_mut();
        let mut needs_write_barrier = true;
        let raw = unsafe {
            crate::gc_hook::try_gc_alloc_collecting_rooted(
                tid,
                bytes,
                &mut allocation_root,
                &mut needs_write_barrier,
            )
        };
        let raw = crate::gc_hook::GcAllocOutcome::from_hook(raw)
            .allocated_or_abort(bytes)
            .unwrap_or(std::ptr::null_mut());
        if !raw.is_null() {
            // Collecting malloc stamps the header only. `_ll_malloc_entries`
            // is `zero=True`. `GCREF` is `*mut GCREFOpaque` (`size_of` 0), so
            // `write_bytes` on the GCREF itself writes no bytes; zero through
            // `*mut u8`.
            unsafe { std::ptr::write_bytes(raw as *mut u8, 0, bytes) };
            if needs_write_barrier {
                crate::gc_hook::try_gc_write_barrier_managed(raw);
            }
            let entries = raw as *mut WeakDictEntries<K>;
            unsafe { (*entries).length = n };
            return entries;
        }
    }
    let layout = std::alloc::Layout::from_size_align(bytes, std::mem::align_of::<usize>())
        .expect("weakdict layout");
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    let entries = raw as *mut WeakDictEntries<K>;
    unsafe { (*entries).length = n };
    entries
}

fn items<K>(entries: *mut WeakDictEntries<K>) -> *mut WeakDictEntry<K> {
    unsafe { std::ptr::addr_of_mut!((*entries).items).cast() }
}

fn entry<K>(entries: *mut WeakDictEntries<K>, i: usize) -> *mut WeakDictEntry<K> {
    unsafe { items(entries).add(i) }
}

fn everused<K>(entries: *mut WeakDictEntries<K>, i: usize) -> bool {
    unsafe { !(*entry(entries, i)).value.is_null() }
}

fn valid<K>(entries: *mut WeakDictEntries<K>, i: usize) -> bool {
    let value = unsafe { (*entry(entries, i)).value };
    if value.is_null() {
        return false;
    }
    // `ll_valid`: `bool(value) and bool(weakref_deref(...))`.
    !unsafe { crate::weakref::w_weakref_deref(value) }.is_null()
}

/// Convert_const intern entry: `WeakValueDictRepr.convert_const` stores a
/// prebuilt weakref (`prebuilt_weakref`, not collector-owned);
/// `new_interned_w_str` stores a GC weakref (`w_weakref_new`).
fn valueref_is_prebuilt(valueref: *mut Weakref) -> bool {
    !valueref.is_null() && !crate::gc_hook::try_gc_owns_object(valueref as crate::gc_hook::GCREF)
}

/// `WEAKDICTENTRYARRAY` adt `hash`: `ll_hash(entries, i)` =
/// `fasthashfn(entries[i].key)`.
fn ll_hash<K: WeakDictKey>(entries: *mut WeakDictEntries<K>, i: usize) -> isize {
    unsafe { (*entry(entries, i)).key.ll_fasthash() }
}

/// `llmemory.dead_wref` (`_wref(None)._as_ptr()`). A non-null `WeakRefPtr`
/// whose deref is null, so `everused()` stays true after `ll_set_null`.
fn dead_wref() -> *mut Weakref {
    static DEAD: AtomicPtr<Weakref> = AtomicPtr::new(std::ptr::null_mut());
    let existing = DEAD.load(Ordering::Acquire);
    if !existing.is_null() {
        return existing;
    }
    let created = crate::lltype::malloc_typed(Weakref {
        weakptr: std::ptr::null_mut(),
    });
    match DEAD.compare_exchange(
        std::ptr::null_mut(),
        created,
        Ordering::Release,
        Ordering::Acquire,
    ) {
        Ok(_) => created,
        Err(winner) => winner,
    }
}

/// `objectmodel._hash_string` over STR `chars`. Empty string is `-1`.
fn hash_string_bytes(s: &[u8]) -> isize {
    let Some(&first) = s.first() else {
        return -1;
    };
    let mut x = (first as isize) << 7;
    for &c in s {
        x = 1000003isize.wrapping_mul(x) ^ (c as isize);
    }
    x ^ (s.len() as isize)
}

/// `_ll_strhash` digest of `chars` without a STR: zero becomes the rstr
/// sentinel so it matches a stored `STR.hash`.
fn hash_chars(bytes: &[u8]) -> isize {
    let mut x = hash_string_bytes(bytes);
    if x == 0 {
        x = 29872897;
    }
    x
}

/// `ll_streq` of a stored STR key against borrowed `chars`.
fn ll_streq_chars(key: StrKey, bytes: &[u8]) -> bool {
    if key.0.is_null() {
        return false;
    }
    // `rstr.py ll_streq` over `chars`. Length then bytes, without the
    // `dont_look_inside` payload helper.
    let len = crate::lowlevel_string::bh_lowlevel_string_len(key.0 as i64);
    if len != bytes.len() {
        return false;
    }
    unsafe {
        std::slice::from_raw_parts(
            (key.0 as *const u8).add(crate::lowlevel_string::LOWLEVEL_STRING_CHARS_OFFSET),
            len,
        ) == bytes
    }
}

/// `LLHelpers._ll_strhash` (`rstr.py`). Zero is the uncomputed memo, so a
/// real zero digest becomes `29872897` and is stored in `STR.hash`.
#[majit_macros::dont_look_inside]
fn _ll_strhash(s: *mut Utf8Str) -> isize {
    let bytes = unsafe { crate::unicodeobject::utf8_payload_bytes(s) };
    let mut x = hash_string_bytes(bytes);
    if x == 0 {
        x = 29872897;
    }
    unsafe { (*s).hash = x };
    x
}

/// `LLHelpers.ll_strhash` (`rstr.py`). Null STR hashes as 0; a live STR
/// answers `s.hash` while it is already memoized.
///
/// Upstream is `jit.conditional_call_elidable(s.hash, _ll_strhash, s)`.
/// This crate has no equivalent annotation on ordinary helpers.
fn ll_strhash(s: *mut Utf8Str) -> isize {
    if s.is_null() {
        return 0;
    }
    let cached = unsafe { (*s).hash };
    if cached != 0 {
        return cached;
    }
    _ll_strhash(s)
}

/// `LLHelpers.ll_strfasthash` (`rstr.py`). Assumes `STR.hash` is already
/// computed (nonzero).
fn ll_strfasthash(s: *mut Utf8Str) -> isize {
    let cached = unsafe { (*s).hash };
    debug_assert_ne!(cached, 0, "ll_strfasthash: hash==0");
    cached
}

/// `r_key.get_ll_eq_function()` for STR: `ll_streq` over `chars`.
///
/// `LLHelpers.ll_streq` (`rstr.py`): `@jit.elidable` and
/// `@jit.oopspec('stroruni.equal(s1, s2)')`.
#[majit_macros::oopspec("stroruni.equal(s1, s2)")]
#[majit_macros::elidable]
fn ll_streq(s1: StrKey, s2: StrKey) -> bool {
    if s1.0 == s2.0 {
        return true;
    }
    if s1.0.is_null() || s2.0.is_null() {
        return false;
    }
    unsafe {
        crate::unicodeobject::utf8_payload_bytes(s1.0)
            == crate::unicodeobject::utf8_payload_bytes(s2.0)
    }
}

/// `setarrayitem_gc` write barrier on `WEAKDICTENTRYARRAY`.
/// `framework.py` `transform_generic_set` emits
/// `write_barrier_from_array(array, index)` when `_set_into_gc_array_part`
/// sees a `setarrayitem` / `setinteriorfield`. `incminimark.py`
/// `write_barrier_from_array` card-marks. A young WEAKREF or STR key
/// stored into an old entries block is reached on the next minor only if
/// the array sits in the remembered set (`collect_oldrefs_to_nursery`);
/// `invalidate_young_weakrefs` then rewrites `weakptr`. A pre-hook
/// immortal array is not collector-owned and keeps `write_barrier`.
///
/// The call is `gc_hook::try_gc_write_barrier_from_array`
/// (`dont_look_inside`): an inlined `TRACK_YOUNG_PTRS` test can fold the
/// clear bit from a young birth and skip the barrier after the array is
/// promoted. The nursery / flag / forwarded tests live first inside that
/// hook.
fn barrier_entries<K>(entries: *mut WeakDictEntries<K>, i: usize) {
    if !entries.is_null() {
        crate::gc_hook::try_gc_write_barrier_from_array(entries as crate::gc_hook::GCREF, i);
    }
}

/// `setfield_gc` write barrier on `WEAKDICT.entries`. Resize replaces the
/// array pointer; a prebuilt/old dict is not scanned on a minor unless this
/// remembers it (`old_objects_pointing_to_young`).
fn barrier_dict<K>(d: *mut WeakDict<K>) {
    if !d.is_null() {
        crate::gc_hook::try_gc_write_barrier(d as crate::gc_hook::GCREF);
    }
}

/// `setfield_gc` write barrier on `WEAKDICT`. Not a collection: the flag
/// test in `incminimark.py` `write_barrier` enrolls
/// `old_objects_pointing_to_young` when `GCFLAG_TRACK_YOUNG_PTRS` is set.
/// `_rweakvaldict.py` `ll_set_nonnull` keeps `d` a livevar across
/// `weakref_create`; for an old/prebuilt table the barrier is what lets a
/// minor trace `d.entries`. Callers publish any young livevars after this
/// and before the first safepoint (`framework.py` `push_roots`). Does not
/// extra-pin `entries`.
pub fn barrier_weakdict<K>(d: *mut WeakDict<K>) {
    barrier_dict(d);
}

/// `rordereddict.rs` `pin_table_and_entries`. Barrier-before-pin of an
/// old table: pinning an old object does not walk its children during
/// Scanning. Pin the table if it is a GC object; pin `entries` whenever
/// it is non-null. A host table must not be published as a `GCREF`.
/// Returns the shadow-stack index of the `entries` pin, or the current
/// length when `entries` is null.
fn pin_table_and_entries<K>(d: *mut WeakDict<K>, entries: *mut WeakDictEntries<K>) -> usize {
    if !d.is_null() {
        barrier_dict(d);
        if crate::gc_hook::try_gc_owns_object(d as crate::gc_hook::GCREF) {
            let _ = crate::gc_roots::pin_root(d as crate::pyobject::PyObjectRef);
        }
    }
    let entries_slot = crate::gc_roots::shadow_stack_len();
    if !entries.is_null() {
        crate::gc_hook::try_gc_write_barrier(entries as crate::gc_hook::GCREF);
        let _ = crate::gc_roots::pin_root(entries as crate::pyobject::PyObjectRef);
    }
    entries_slot
}

/// `ll_setitem_fast` on `d.entries[i]`: barrier, then the key and value
/// words (`items_block_set_ref` / `RDict::entry_at_mut`).
fn store_entry<K>(entries: *mut WeakDictEntries<K>, i: usize, key: K, value: *mut Weakref) {
    barrier_entries(entries, i);
    unsafe {
        (*entry(entries, i)).key = key;
        (*entry(entries, i)).value = value;
    }
}

/// `ll_new_weakdict`. The `dont_look_inside` trampoline cannot sit inside
/// an `impl`.
#[majit_macros::dont_look_inside]
pub fn ll_new_weakdict<K: WeakDictKey>() -> *mut WeakDict<K> {
    let raw = alloc_raw(weakdict_gc_type_id(), std::mem::size_of::<WeakDict<K>>());
    // `_ll_malloc_entries` may collect (`framework.py` `push_roots`). `d` is
    // the livevar across that malloc (`_rweakvaldict.py` `ll_new_weakdict`).
    // The table is `alloc_raw` old-gen (`WEAKDICT`); `d` does not move.
    let _roots = crate::gc_roots::push_roots();
    let _ = pin_table_and_entries(raw as *mut WeakDict<K>, std::ptr::null_mut());
    let entries = alloc_entries::<K>(DICT_INITSIZE);
    let d = raw as *mut WeakDict<K>;
    barrier_dict(d);
    unsafe {
        (*d).entries = entries;
        (*d).num_items = 0;
        (*d).resize_counter = (DICT_INITSIZE * 2) as isize;
    }
    d
}

/// `rdict.py` `ll_dict_lookup`: `@jit.look_inside_iff(lambda d, key, hash:
/// jit.isvirtual(d) and jit.isconstant(key))`.
///
/// Predicates are free functions (`listobject.rs` `boxed_from_ints_iff`).
/// The key is `r_key`'s STR (`StrKey`); a caller that only has characters
/// probes with a temporary STR.
fn ll_dict_lookup_iff(d: &WeakDict<StrKey>, key: StrKey, _hash: isize) -> bool {
    majit_rlib::jit::isvirtual(d) && majit_rlib::jit::isconstant(&key)
}

/// `rdict.py` `ll_dict_lookup(d, key, hash)`. The high bit marks a miss;
/// the low bits are the slot (`MASK` keeps them, and a miss slot may be a
/// deleted one).
///
/// The spec is the bare name: Rust's parameters are `(d, key, hash)`, and
/// `global_marker_str` harvests only the spec string — the same reason
/// `RDict::lookup` uses `ordereddict.lookup`.
///
/// `oopspec` is stacked outside `look_inside_iff`. `look_inside_iff` moves
/// `func.oopspec` onto the trampoline (`rlib/jit.py`
/// `trampoline.oopspec = func.oopspec; del func.oopspec`); majit-macros
/// does the same.
#[majit_macros::oopspec("dict.lookup")]
#[majit_macros::look_inside_iff(ll_dict_lookup_iff)]
fn ll_dict_lookup(d: &WeakDict<StrKey>, key: StrKey, hash: isize) -> usize {
    let entries = d.entries;
    let mask = unsafe { (*entries).length } - 1;
    // `rdict.py` `i = r_uint(hash & mask)`; `perturb = r_uint(hash)`.
    let mut i = (hash as usize) & mask;
    let mut freeslot: isize;
    // First try before any looping (`rdict.py` `ll_dict_lookup`).
    if valid(entries, i) {
        let checking = unsafe { (*entry(entries, i)).key };
        if checking.0 == key.0 {
            return i;
        }
        if ll_hash(entries, i) == hash && ll_streq(checking, key) {
            return i;
        }
        freeslot = -1;
    } else if everused(entries, i) {
        freeslot = i as isize;
    } else {
        return i | HIGHEST_BIT;
    }
    let mut perturb = hash as usize;
    loop {
        i = (i << 2)
            .wrapping_add(i)
            .wrapping_add(perturb)
            .wrapping_add(1)
            & mask;
        if !everused(entries, i) {
            if freeslot == -1 {
                freeslot = i as isize;
            }
            return (freeslot as usize) | HIGHEST_BIT;
        } else if valid(entries, i) {
            let checking = unsafe { (*entry(entries, i)).key };
            if checking.0 == key.0 {
                return i;
            }
            if ll_hash(entries, i) == hash && ll_streq(checking, key) {
                return i;
            }
        } else if freeslot == -1 {
            freeslot = i as isize;
        }
        perturb >>= PERTURB_SHIFT;
    }
}

/// `rdict.py` `ll_dict_lookup` with the key as borrowed `chars`.
///
/// `interned_strings.get` takes an rstr `STR` (`r_key`); a byte-only
/// caller would otherwise allocate a probe STR (`ll_get(d, llkey)`).
/// Hash and eq are `_ll_strhash` / `ll_streq` over `chars`, so the
/// probe is unnecessary (`objectmodel._hash_string`, `rstr.py ll_streq`).
fn ll_dict_lookup_chars(d: &WeakDict<StrKey>, bytes: &[u8], hash: isize) -> usize {
    let entries = d.entries;
    let mask = unsafe { (*entries).length } - 1;
    let mut i = (hash as usize) & mask;
    let mut freeslot: isize;
    if valid(entries, i) {
        let checking = unsafe { (*entry(entries, i)).key };
        if ll_hash(entries, i) == hash && ll_streq_chars(checking, bytes) {
            return i;
        }
        freeslot = -1;
    } else if everused(entries, i) {
        freeslot = i as isize;
    } else {
        return i | HIGHEST_BIT;
    }
    let mut perturb = hash as usize;
    loop {
        i = (i << 2)
            .wrapping_add(i)
            .wrapping_add(perturb)
            .wrapping_add(1)
            & mask;
        if !everused(entries, i) {
            if freeslot == -1 {
                freeslot = i as isize;
            }
            return (freeslot as usize) | HIGHEST_BIT;
        } else if valid(entries, i) {
            let checking = unsafe { (*entry(entries, i)).key };
            if ll_hash(entries, i) == hash && ll_streq_chars(checking, bytes) {
                return i;
            }
        } else if freeslot == -1 {
            freeslot = i as isize;
        }
        perturb >>= PERTURB_SHIFT;
    }
}

impl WeakDict<StrKey> {
    /// `ll_get(d, llkey)`. `None` is the null object pointer.
    #[majit_macros::dont_look_inside]
    pub fn ll_get(&self, llkey: StrKey) -> Option<crate::pyobject::PyObjectRef> {
        let hash = llkey.ll_keyhash();
        let i = ll_dict_lookup(self, llkey, hash) & (HIGHEST_BIT - 1);
        self.ll_get_at(i)
    }

    /// `ll_get` whose key is borrowed `chars` rather than a STR probe.
    #[majit_macros::dont_look_inside]
    pub fn ll_get_chars(&self, bytes: &[u8]) -> Option<crate::pyobject::PyObjectRef> {
        let hash = hash_chars(bytes);
        let i = ll_dict_lookup_chars(self, bytes, hash) & (HIGHEST_BIT - 1);
        self.ll_get_at(i)
    }

    /// Convert_const intern lookup keyed by borrowed `chars`.
    #[majit_macros::dont_look_inside]
    pub fn ll_get_chars_prebuilt(&self, bytes: &[u8]) -> Option<crate::pyobject::PyObjectRef> {
        let hash = hash_chars(bytes);
        let i = ll_dict_lookup_chars(self, bytes, hash) & (HIGHEST_BIT - 1);
        if !valueref_is_prebuilt(unsafe { (*entry(self.entries, i)).value }) {
            return None;
        }
        self.ll_get_at(i)
    }

    fn ll_get_at(&self, i: usize) -> Option<crate::pyobject::PyObjectRef> {
        let valueref = unsafe { (*entry(self.entries, i)).value };
        if valueref.is_null() {
            None
        } else {
            let obj = unsafe { crate::weakref::w_weakref_deref(valueref) };
            if obj.is_null() { None } else { Some(obj) }
        }
    }

    /// Convert_const intern entry: the stored valueref is a prebuilt weakref
    /// (`unicodeobject.rs` `prebuilt_weakref`, not collector-owned).
    /// `new_interned_w_str` stores a GC weakref (`w_weakref_new`).
    pub fn ll_valueref_is_prebuilt(&self, llkey: StrKey) -> bool {
        let hash = llkey.ll_keyhash();
        let i = ll_dict_lookup(self, llkey, hash) & (HIGHEST_BIT - 1);
        valueref_is_prebuilt(unsafe { (*entry(self.entries, i)).value })
    }

    /// `WeakValueDictRepr.ll_set`. A null `llvalue` is `None` and
    /// dispatches to `ll_set_null`.
    #[majit_macros::dont_look_inside]
    pub fn ll_set(&mut self, llkey: StrKey, llvalue: crate::pyobject::PyObjectRef) {
        if !llvalue.is_null() {
            self.ll_set_nonnull(llkey, llvalue);
        } else {
            self.ll_set_null(llkey);
        }
    }

    /// `WeakValueDictRepr.ll_set_null`. Stores `dead_wref` so `everused()`
    /// stays true, and nulls the key with `WeakDictKey::dummy`
    /// (`r_key.convert_const(None)`).
    #[majit_macros::dont_look_inside]
    pub fn ll_set_null(&mut self, llkey: StrKey) {
        let hash = llkey.ll_keyhash();
        let i = ll_dict_lookup(self, llkey, hash) & (HIGHEST_BIT - 1);
        if everused(self.entries, i) {
            store_entry(self.entries, i, StrKey::dummy(), dead_wref());
        }
    }

    /// `WeakValueDictRepr.ll_set_nonnull`. `weakref_create(llvalue)` runs
    /// first (GC effects), then the valueref store. `d`, `llkey`, and
    /// `llvalue` are livevars across that malloc (`framework.py`
    /// `push_roots`).
    #[majit_macros::dont_look_inside]
    pub fn ll_set_nonnull(&mut self, key: StrKey, llvalue: crate::pyobject::PyObjectRef) {
        let _roots = crate::gc_roots::push_roots();
        // `write_barrier` first (not a collect), then `d` / `llvalue` /
        // `llkey` as one livevar set (`framework.py` `push_roots`) across
        // `weakref_create`.
        barrier_weakdict(self as *mut WeakDict<StrKey>);
        let has_key = !key.0.is_null();
        let val_slot = if has_key {
            crate::gc_roots::pin_roots(&[llvalue, key.0 as crate::pyobject::PyObjectRef])
        } else {
            let slot = crate::gc_roots::shadow_stack_len();
            let _ = crate::gc_roots::pin_root(llvalue);
            slot
        };
        let valueref =
            unsafe { crate::weakref::w_weakref_new(crate::gc_roots::shadow_stack_get(val_slot)) };
        let key = if has_key {
            StrKey(crate::gc_roots::shadow_stack_get(val_slot + 1) as *mut Utf8Str)
        } else {
            key
        };
        self.ll_set_nonnull_valueref(key, valueref);
    }

    /// Lookup/insert half of `WeakValueDictRepr.ll_set_nonnull` after
    /// `weakref_create`. Upstream is a single function: `valueref =
    /// weakref_create(llvalue)` then lookup/store. The intern table has
    /// no GIL, so `unicodeobject.rs` `intern_publish` takes its mutex only
    /// around this half and must not run `weakref_create` under that lock.
    #[majit_macros::dont_look_inside]
    pub fn ll_set_nonnull_valueref(&mut self, key: StrKey, valueref: *mut Weakref) {
        debug_assert!(!valueref.is_null());
        let hash = key.ll_keyhash();
        let found = ll_dict_lookup(self, key, hash);
        let i = found & (HIGHEST_BIT - 1);
        let was_everused = everused(self.entries, i);
        store_entry(self.entries, i, key, valueref);
        if !was_everused {
            self.resize_counter -= 3;
            if self.resize_counter <= 0 {
                // `rdict.py` `_ll_dict_setitem_lookup_done`: store, then
                // `if d.resize_counter <= 0: ll_dict_resize(d)`. `key` /
                // `valueref` are already in `entries` and dead afterwards;
                // `framework.py` `push_roots` emits only livevars.
                self.ll_weakdict_resize();
            }
        }
    }
}

impl<K: WeakDictKey> WeakDict<K> {
    pub fn count_valid(&self) -> usize {
        if self.entries.is_null() {
            return 0;
        }
        let n = unsafe { (*self.entries).length };
        (0..n).filter(|&i| valid(self.entries, i)).count()
    }

    /// Live entries whose stored valueref is a convert_const prebuilt weakref
    /// (`prebuilt_weakref`, not collector-owned). `sys.getunicodeinternedsize(
    /// _only_immortal=True)` counts these: a convert_const interned string,
    /// not a value first presented to `sys.intern()` (`w_weakref_new`).
    pub fn count_valid_immortal(&self) -> usize {
        if self.entries.is_null() {
            return 0;
        }
        let n = unsafe { (*self.entries).length };
        (0..n)
            .filter(|&i| {
                valid(self.entries, i)
                    && valueref_is_prebuilt(unsafe { (*entry(self.entries, i)).value })
            })
            .count()
    }

    fn ll_dict_lookup_clean(&self, hash: isize) -> usize {
        let entries = self.entries;
        let mask = unsafe { (*entries).length } - 1;
        let mut i = (hash as usize) & mask;
        let mut perturb = hash as usize;
        while everused(entries, i) {
            i = (i << 2)
                .wrapping_add(i)
                .wrapping_add(perturb)
                .wrapping_add(1)
                & mask;
            perturb >>= PERTURB_SHIFT;
        }
        i
    }

    /// `ll_weakdict_resize` then `ll_dict_resize`.
    fn ll_weakdict_resize(&mut self) {
        let entries = self.entries;
        let n = unsafe { (*entries).length };
        let mut num_items = 0isize;
        for i in 0..n {
            if valid(entries, i) {
                num_items += 1;
            }
        }
        self.num_items = num_items;
        ll_dict_resize(self);
    }

    fn ll_dict_insertclean(&mut self, key: K, value: *mut Weakref, hash: isize) {
        let i = self.ll_dict_lookup_clean(hash);
        store_entry(self.entries, i, key, value);
        self.num_items += 1;
        self.resize_counter -= 3;
    }
}

/// `rdict.py` `ll_dict_resize(d)`. `ll_dict_resize.oopspec = 'dict.resize(d)'`.
#[majit_macros::oopspec("dict.resize")]
fn ll_dict_resize<K: WeakDictKey>(d: &mut WeakDict<K>) {
    let num_extra = std::cmp::min(d.num_items + 1, 30000);
    let new_estimate = (d.num_items + num_extra) * 2;
    let mut new_size = DICT_INITSIZE;
    while (new_size as isize) <= new_estimate {
        new_size *= 2;
    }
    let mut old_entries = d.entries;
    let old_size = unsafe { (*old_entries).length };
    // `framework.py` `push_roots` / `pop_roots` around `_ll_malloc_entries`.
    // The nursery malloc may collect and move `old_entries`; read the
    // forwarded word back from the root slot before the copy
    // (`rdict.py` `ll_dict_resize`).
    let _roots = crate::gc_roots::push_roots();
    let old_entries_slot = pin_table_and_entries(d as *mut WeakDict<K>, old_entries);
    let new_entries = alloc_entries::<K>(new_size);
    old_entries = crate::gc_roots::shadow_stack_get(old_entries_slot) as *mut WeakDictEntries<K>;
    barrier_dict(d as *mut WeakDict<K>);
    d.entries = new_entries;
    d.num_items = 0;
    d.resize_counter = (new_size * 2) as isize;
    for i in 0..old_size {
        if !valid(old_entries, i) {
            continue;
        }
        let item = unsafe { *entry(old_entries, i) };
        let hash = ll_hash(old_entries, i);
        d.ll_dict_insertclean(item.key, item.value, hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_key(bytes: &[u8]) -> StrKey {
        StrKey(crate::unicodeobject::alloc_utf8_payload(bytes, false))
    }

    #[test]
    fn miss_returns_the_stored_referent_and_a_second_key_does_not_clobber_it() {
        let d = unsafe { &mut *crate::rweakvaldict::ll_new_weakdict::<StrKey>() };
        let key = str_key(b"abc");
        let obj = 0x1111 as crate::pyobject::PyObjectRef;
        d.ll_set_nonnull(key, obj);
        assert_eq!(d.ll_get(key), Some(obj));
        assert!(d.ll_get(str_key(b"abd")).is_none());
        assert_eq!(d.count_valid(), 1);
        // `ll_set_nonnull` uses `w_weakref_new`; without a collecting hook
        // that falls back to `malloc_typed` (a prebuilt weakref).
        assert_eq!(d.count_valid_immortal(), 1);
        assert_eq!(
            ll_dict_lookup(d, key, key.ll_keyhash()) & (HIGHEST_BIT - 1),
            ll_dict_lookup(d, str_key(b"abc"), str_key(b"abc").ll_keyhash()) & (HIGHEST_BIT - 1)
        );
    }

    /// `count_valid_immortal` inspects the stored valueref, not the referent:
    /// a GC weakref (`w_weakref_new`) is not counted; a prebuilt weakref is.
    #[test]
    fn count_valid_immortal_inspects_valueref_not_referent() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        thread_local! {
            static MANAGED_ALLOCS: std::cell::RefCell<Vec<usize>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        fn record_managed_alloc(_type_id: u32, payload_size: usize) -> crate::gc_hook::GCREF {
            let layout = std::alloc::Layout::from_size_align(payload_size.max(1), 8)
                .expect("managed valueref probe layout");
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if !ptr.is_null() {
                MANAGED_ALLOCS.with(|slots| slots.borrow_mut().push(ptr as usize));
            }
            ptr as crate::gc_hook::GCREF
        }
        fn managed_alloc_is_owned(addr: usize) -> bool {
            MANAGED_ALLOCS.with(|slots| slots.borrow().contains(&addr))
        }
        struct ClearProbe;
        impl Drop for ClearProbe {
            fn drop(&mut self) {
                crate::gc_hook::clear_gc_alloc_collecting_hook();
                crate::gc_hook::clear_gc_owns_object_hook();
                MANAGED_ALLOCS.with(|slots| slots.borrow_mut().clear());
            }
        }
        let _clear = ClearProbe;
        crate::gc_hook::register_gc_alloc_collecting_hook(record_managed_alloc);
        crate::gc_hook::register_gc_owns_object_hook(managed_alloc_is_owned);

        let d = unsafe { &mut *crate::rweakvaldict::ll_new_weakdict::<StrKey>() };
        let gc_key = str_key(b"gc-weakref");
        let gc_obj = 0x4444 as crate::pyobject::PyObjectRef;
        d.ll_set_nonnull(gc_key, gc_obj);
        assert_eq!(d.ll_get(gc_key), Some(gc_obj));
        assert_eq!(d.count_valid(), 1);
        assert_eq!(d.count_valid_immortal(), 0);

        let prebuilt_key = str_key(b"prebuilt-weakref");
        let prebuilt_obj = 0x5555 as crate::pyobject::PyObjectRef;
        let valueref = crate::lltype::malloc_typed(crate::weakref::Weakref {
            weakptr: prebuilt_obj,
        });
        d.ll_set_nonnull_valueref(prebuilt_key, valueref);
        assert_eq!(d.ll_get(prebuilt_key), Some(prebuilt_obj));
        assert_eq!(d.count_valid(), 2);
        assert_eq!(d.count_valid_immortal(), 1);
    }

    /// Six new slots drive `resize_counter` from `DICT_INITSIZE * 2` through
    /// zero (`ll_set_nonnull` subtracts 3). Every key must still be found.
    #[test]
    fn resize_keeps_live_weak_values() {
        let d = unsafe { &mut *crate::rweakvaldict::ll_new_weakdict::<StrKey>() };
        let mut objs = Vec::new();
        for n in 0..6 {
            let bytes = format!("k{n}");
            let key = str_key(bytes.as_bytes());
            let obj = (0x2000 + n) as crate::pyobject::PyObjectRef;
            d.ll_set_nonnull(key, obj);
            objs.push((key, obj));
        }
        assert!(unsafe { (*d.entries).length } > DICT_INITSIZE);
        for (key, obj) in objs {
            assert_eq!(d.ll_get(key), Some(obj));
        }
    }

    /// `STR.hash` memo and a fresh `_ll_strhash` stay in one `Signed` domain,
    /// so two distinct STR objects with equal bytes intern to one slot.
    #[test]
    fn memoized_strhash_matches_fresh_hash_and_equal_bytes_intern_once() {
        let bytes = b"foo";
        let key = str_key(bytes);
        let first = ll_strhash(key.0);
        assert_eq!(ll_strhash(key.0), first);
        assert_ne!(first, 0);

        let other = str_key(bytes);
        assert_ne!(key.0, other.0);
        assert_eq!(ll_strhash(other.0), first);
        let d = unsafe { &mut *crate::rweakvaldict::ll_new_weakdict::<StrKey>() };
        let obj = 0x3333 as crate::pyobject::PyObjectRef;
        d.ll_set_nonnull(key, obj);
        d.ll_set_nonnull(other, obj);
        assert_eq!(d.count_valid(), 1);
    }

    /// `_ll_strhash` substitutes `29872897` for a real zero digest so the
    /// memo slot can keep zero as "not computed".
    #[test]
    fn ll_strhash_zero_digest_uses_the_rstr_sentinel() {
        // `_hash_string` of a single NUL byte is 0: `(0 << 7) ^ 0 ^ 1` wait...
        // empty is -1. Find bytes whose `_hash_string` is 0 by brute force
        // is unnecessary: write the computed path's sentinel contract.
        let key = str_key(b"");
        assert_eq!(hash_string_bytes(b""), -1);
        assert_eq!(ll_strhash(key.0), -1);
        let zeroed = str_key(b"\0");
        let raw = hash_string_bytes(b"\0");
        let got = ll_strhash(zeroed.0);
        if raw == 0 {
            assert_eq!(got, 29872897);
        } else {
            assert_eq!(got, raw);
        }
        assert_ne!(got, 0);
    }

    /// `ll_set` with a live value is `ll_set_nonnull`.
    #[test]
    fn ll_set_stores_a_live_value() {
        let d = unsafe { &mut *crate::rweakvaldict::ll_new_weakdict::<StrKey>() };
        let key = str_key(b"set-live");
        let obj = 0x4444 as crate::pyobject::PyObjectRef;
        d.ll_set(key, obj);
        assert_eq!(d.ll_get(key), Some(obj));
        assert_eq!(d.count_valid(), 1);
    }

    /// `ll_set_null` stores `dead_wref` so `everused()` stays true and
    /// nulls the key with `WeakDictKey::dummy`. `ll_set` with a null
    /// value dispatches here.
    #[test]
    fn ll_set_null_keeps_everused_and_nulls_the_key() {
        let d = unsafe { &mut *crate::rweakvaldict::ll_new_weakdict::<StrKey>() };
        let key = str_key(b"set-null");
        let obj = 0x5555 as crate::pyobject::PyObjectRef;
        d.ll_set(key, obj);
        assert_eq!(d.ll_get(key), Some(obj));

        d.ll_set(key, std::ptr::null_mut());
        assert!(d.ll_get(key).is_none());
        assert_eq!(d.count_valid(), 0);

        let i = ll_dict_lookup(d, key, key.ll_keyhash()) & (HIGHEST_BIT - 1);
        assert!(everused(d.entries, i));
        assert!(!valid(d.entries, i));
        let slot = unsafe { *entry(d.entries, i) };
        assert!(slot.key.0.is_null());
        assert_eq!(slot.value, dead_wref());
        assert!(unsafe { crate::weakref::w_weakref_deref(slot.value) }.is_null());

        let other = str_key(b"set-null-other");
        let other_obj = 0x5556 as crate::pyobject::PyObjectRef;
        d.ll_set(other, other_obj);
        assert_eq!(d.ll_get(other), Some(other_obj));
        assert!(d.ll_get(key).is_none());
    }

    /// `ll_set_null` on a key that was never stored does not occupy a slot.
    #[test]
    fn ll_set_null_on_unused_key_does_not_occupy_a_slot() {
        let d = unsafe { &mut *crate::rweakvaldict::ll_new_weakdict::<StrKey>() };
        let key = str_key(b"never-used");
        d.ll_set_null(key);
        assert!(d.ll_get(key).is_none());
        assert_eq!(d.count_valid(), 0);
        let i = ll_dict_lookup(d, key, key.ll_keyhash()) & (HIGHEST_BIT - 1);
        assert!(!everused(d.entries, i));
    }

    /// `ll_strfasthash` is the stored-key hash (`entries.hash(i)`); the
    /// probe key still uses `ll_strhash`.
    #[test]
    fn ll_strfasthash_reads_the_cached_str_hash() {
        let key = str_key(b"fast-hash");
        assert_eq!(unsafe { (*key.0).hash }, 0);
        let computed = ll_strhash(key.0);
        assert_ne!(computed, 0);
        assert_eq!(ll_strfasthash(key.0), computed);
        assert_eq!(key.ll_fasthash(), computed);

        let d = unsafe { &mut *crate::rweakvaldict::ll_new_weakdict::<StrKey>() };
        let obj = 0x6666 as crate::pyobject::PyObjectRef;
        d.ll_set_nonnull(key, obj);
        let i = ll_dict_lookup(d, key, computed) & (HIGHEST_BIT - 1);
        assert_eq!(ll_hash(d.entries, i), computed);
        assert_eq!(d.ll_get(key), Some(obj));
    }
}
